from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import os
import tempfile
from typing import Any

import h5py
import numpy as np
import SimpleITK as sitk
from scipy import ndimage

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture


DEFAULT_PCA_VARIANCE_PERCENT = 95.0
VALID_CLUSTERING_SPACES = {"pca", "zscore"}
VALID_CLUSTERERS = {"kmeans", "gmm"}
GMM_REG_COVAR_RETRY_VALUES = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2)
PES_SCORE_SCHEMA_VERSION = "radiomic_deviation_v1"
VALID_SCORE_MODES = {
    "weighted_deviation",
    "unweighted_deviation",
    "legacy_signed_mean",
}
DEFAULT_FEATURE_FAMILY_WEIGHTS = {
    "firstorder": 1.0,
    "glcm": 1.0,
    "glrlm": 1.0,
    "glszm": 1.0,
    "gldm": 1.0,
    "ngtdm": 1.0,
}


@dataclass(frozen=True)
class RadiomicDeviationConfig:
    """Outcome-free score used to order PES parents and select children.

    The primary score is the square root of a weighted sum of squared,
    support-standardized feature values. Feature families receive equal
    declared weight by default. Within each family, absolute-correlation
    connected components receive equal weight and channels split their
    component weight. The frozen threshold and weighting rule prevent dense
    or duplicated feature groups from receiving weight merely because they
    contain more channels.
    """

    mode: str = "weighted_deviation"
    correlation_threshold: float = 0.90
    family_weights: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_FEATURE_FAMILY_WEIGHTS)
    )
    schema_version: str = PES_SCORE_SCHEMA_VERSION


# Modality-agnostic phenotypic tier clustering

@dataclass
class TierClusteringResult:
    """Result of multi-feature phenotypic tier clustering.

    Valid voxel features are represented in either retained PCA-score space or
    feature-wise z-score space and then clustered. Generic tier names are
    assigned from the composite standardized enrichment score:
      - tier_h: highest mean image-derived enrichment score
      - tier_m: middle enrichment
      - tier_l: lowest mean image-derived enrichment score

    The labels are image-derived ranks and do not imply biological activity,
    predictive importance, malignancy, or clinical risk.
    """
    tier_h_mask: np.ndarray       # [1, Z, Y, X] uint8 — high enrichment tier
    tier_m_mask: np.ndarray       # [1, Z, Y, X] uint8 — medium enrichment tier
    tier_l_mask: np.ndarray       # [1, Z, Y, X] uint8 — low enrichment tier
    labels_3d: np.ndarray         # [1, Z, Y, X] int16, cluster labels (0,1,2)
    enrichment_scores: np.ndarray # [1, Z, Y, X] float32, per-voxel enrichment
    cluster_sizes: dict[str, int]
    cluster_enrichment_means: dict[str, float]
    method: str
    n_features_used: int
    n_pca_components: int
    pca_variance_percent: float
    pca_explained_variance_ratio: float
    clustering_space: str
    clusterer: str
    normalization: str
    score_definition: str
    score_metadata: dict[str, Any] = field(default_factory=dict)


def _get_feature_class(feature_name: str) -> str:
    """Extract the feature class from a feature name.

    Handles both formats:
      - 'original_firstorder_Mean' -> 'firstorder'
      - 'firstorder_Mean'          -> 'firstorder'
    """
    parts = feature_name.split("_")
    if parts[0] == "original" and len(parts) >= 3:
        return parts[1].lower()
    return parts[0].lower()


def _filter_by_feature_classes(
    feature_maps: np.ndarray,
    feature_names: list[str],
    feature_subset: list[str] | str | None,
) -> tuple[np.ndarray, list[str]]:
    """Filter feature maps to only include specified feature classes.

    Parameters
    ----------
    feature_maps : (C, Z, Y, X) array
    feature_names : list of C feature names
    feature_subset : which feature classes to keep.
        - None or "all" -> keep all features (no filtering)
        - list of class names, e.g. ["firstorder"], ["firstorder", "glcm"]
          Valid classes: firstorder, glcm, glrlm, glszm, gldm, ngtdm

    Returns
    -------
    (filtered_maps, filtered_names) with only the selected classes
    """
    if feature_subset is None or feature_subset == "all":
        return feature_maps, feature_names

    if isinstance(feature_subset, str):
        feature_subset = [feature_subset]

    allowed = {cls.lower() for cls in feature_subset}
    keep_idx = [i for i, name in enumerate(feature_names)
                if _get_feature_class(name) in allowed]

    if not keep_idx:
        raise ValueError(
            f"No features match the requested classes {feature_subset}. "
            f"Available classes: {sorted(set(_get_feature_class(n) for n in feature_names))}"
        )

    return feature_maps[keep_idx], [feature_names[i] for i in keep_idx]


def _validate_pca_variance_percent(value: float) -> float:
    percent = float(value)
    if not 0.0 < percent <= 100.0:
        raise ValueError("pca_variance_percent must be in (0, 100]")
    return percent


def _validate_clustering_space(value: str) -> str:
    space = str(value).strip().lower()
    if space not in VALID_CLUSTERING_SPACES:
        raise ValueError(
            f"clustering_space must be one of {sorted(VALID_CLUSTERING_SPACES)}"
        )
    return space


def _validate_clusterer(value: str) -> str:
    clusterer = str(value).strip().lower()
    if clusterer not in VALID_CLUSTERERS:
        raise ValueError(f"clusterer must be one of {sorted(VALID_CLUSTERERS)}")
    return clusterer


def _zscore_columns(raw: np.ndarray) -> np.ndarray:
    means = np.mean(raw, axis=0)
    stds = np.std(raw, axis=0)
    nonconstant = stds > 1e-8
    if not np.any(nonconstant):
        raise ValueError("No non-constant dimensions remain after normalization")
    z = (raw[:, nonconstant] - means[nonconstant]) / stds[nonconstant]
    return np.nan_to_num(
        z,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(np.float32)


def _validate_score_config(
    config: RadiomicDeviationConfig | None,
) -> RadiomicDeviationConfig:
    selected = config or RadiomicDeviationConfig()
    if selected.mode not in VALID_SCORE_MODES:
        raise ValueError(f"score mode must be one of {sorted(VALID_SCORE_MODES)}")
    threshold = float(selected.correlation_threshold)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("correlation_threshold must be in [0, 1]")
    if any(float(value) < 0.0 for value in selected.family_weights.values()):
        raise ValueError("feature-family weights must be non-negative")
    return selected


def _absolute_correlation_blocks(
    z: np.ndarray,
    indices: list[int],
    threshold: float,
) -> list[list[int]]:
    """Return deterministic within-family absolute-correlation components."""
    if not indices:
        return []
    if len(indices) == 1:
        return [list(indices)]

    local = np.asarray(z[:, indices], dtype=np.float64)
    correlation = np.corrcoef(local, rowvar=False)
    correlation = np.nan_to_num(correlation, nan=0.0, posinf=0.0, neginf=0.0)
    parents = list(range(len(indices)))

    def find(value: int) -> int:
        while parents[value] != value:
            parents[value] = parents[parents[value]]
            value = parents[value]
        return value

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parents[max(left_root, right_root)] = min(left_root, right_root)

    for left in range(len(indices)):
        for right in range(left + 1, len(indices)):
            if abs(float(correlation[left, right])) >= float(threshold):
                union(left, right)

    grouped: dict[int, list[int]] = {}
    for local_index, global_index in enumerate(indices):
        grouped.setdefault(find(local_index), []).append(global_index)
    return sorted(
        (sorted(block) for block in grouped.values()),
        key=lambda block: (block[0], len(block)),
    )


def radiomic_deviation_scores(
    raw: np.ndarray,
    feature_names: list[str],
    config: RadiomicDeviationConfig | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Calculate sign-invariant, family- and correlation-balanced scores.

    ``raw`` must contain finite, nonconstant columns. The returned metadata
    freezes the effective family, block, and channel weights used for this
    support so every patient-level ordering can be audited.
    """
    selected = _validate_score_config(config)
    raw = np.asarray(raw, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] != len(feature_names):
        raise ValueError("raw columns must match feature_names")
    means = np.mean(raw, axis=0, dtype=np.float64)
    stds = np.std(raw, axis=0, dtype=np.float64)
    valid = np.isfinite(means) & np.isfinite(stds) & (stds > 1e-8)
    if not np.any(valid):
        raise ValueError("radiomic deviation has no non-constant feature columns")
    dropped_names = [
        name for name, keep in zip(feature_names, valid) if not bool(keep)
    ]
    effective_names = [
        name for name, keep in zip(feature_names, valid) if bool(keep)
    ]
    z = (raw[:, valid] - means[valid]) / stds[valid]
    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)

    if selected.mode == "legacy_signed_mean":
        weights = np.full(z.shape[1], 1.0 / z.shape[1], dtype=np.float64)
        scores = np.sum(z * weights[np.newaxis], axis=1)
        blocks = {
            "legacy": [[name] for name in effective_names],
        }
        family_totals = {"legacy": 1.0}
    elif selected.mode == "unweighted_deviation":
        weights = np.full(z.shape[1], 1.0 / z.shape[1], dtype=np.float64)
        scores = np.sqrt(np.sum(np.square(z) * weights[np.newaxis], axis=1))
        blocks = {
            "unweighted": [[name] for name in effective_names],
        }
        family_totals = {"unweighted": 1.0}
    else:
        family_indices: dict[str, list[int]] = {}
        for index, name in enumerate(effective_names):
            family_indices.setdefault(_get_feature_class(name), []).append(index)

        declared = {
            family: float(selected.family_weights.get(family, 0.0))
            for family in family_indices
        }
        positive_total = float(sum(value for value in declared.values() if value > 0.0))
        if positive_total <= 0.0:
            raise ValueError("no positive feature-family weight is available")

        weights = np.zeros(z.shape[1], dtype=np.float64)
        blocks: dict[str, list[list[str]]] = {}
        family_totals: dict[str, float] = {}
        for family in sorted(family_indices):
            declared_weight = declared[family]
            if declared_weight <= 0.0:
                continue
            family_weight = declared_weight / positive_total
            index_blocks = _absolute_correlation_blocks(
                z,
                family_indices[family],
                selected.correlation_threshold,
            )
            block_weight = family_weight / len(index_blocks)
            blocks[family] = []
            for block in index_blocks:
                channel_weight = block_weight / len(block)
                weights[block] = channel_weight
                blocks[family].append([effective_names[index] for index in block])
            family_totals[family] = float(np.sum(weights[family_indices[family]]))
        scores = np.sqrt(np.sum(np.square(z) * weights[np.newaxis], axis=1))

    metadata = {
        "schema_version": selected.schema_version,
        "score_mode": selected.mode,
        "score_definition": (
            "sqrt_weighted_mean_squared_support_zscore"
            if selected.mode == "weighted_deviation"
            else (
                "sqrt_unweighted_mean_squared_support_zscore"
                if selected.mode == "unweighted_deviation"
                else "legacy_mean_support_zscore"
            )
        ),
        "sign_invariant": selected.mode != "legacy_signed_mean",
        "correlation_metric": "absolute_pearson_within_feature_family",
        "correlation_threshold": float(selected.correlation_threshold),
        "feature_names": list(effective_names),
        "dropped_constant_feature_names": dropped_names,
        "input_feature_count": len(feature_names),
        "feature_weights": {
            name: float(weights[index])
            for index, name in enumerate(effective_names)
        },
        "family_weight_totals": family_totals,
        "correlation_blocks": blocks,
        "valid_feature_count": len(effective_names),
        "weight_sum": float(np.sum(weights)),
    }
    return np.asarray(scores, dtype=np.float32), metadata


def _orient_pca_by_dominant_loading(pca: PCA) -> None:
    """Lock each PCA sign so its largest absolute loading is positive.

    PCA component signs are otherwise mathematically interchangeable. This
    outcome-free rule makes the downstream composite score reproducible and
    matches the deterministic sign convention used by current scikit-learn
    full-SVD PCA implementations.
    """
    components = np.asarray(pca.components_, dtype=np.float64)
    anchors = np.argmax(np.abs(components), axis=1)
    anchor_values = components[np.arange(components.shape[0]), anchors]
    signs = np.where(anchor_values < 0.0, -1.0, 1.0)
    pca.components_ = components * signs[:, np.newaxis]


def _pca_scores_and_zscore(
    raw: np.ndarray,
    *,
    pca_variance_percent: float,
    seed: int,
    max_fit_voxels: int | None,
) -> tuple[np.ndarray, np.ndarray, int, float]:
    """Return raw retained PCA scores and their standardized counterpart."""
    percent = _validate_pca_variance_percent(pca_variance_percent)
    fit_raw = raw
    if max_fit_voxels and raw.shape[0] > int(max_fit_voxels):
        rng = np.random.default_rng(int(seed))
        selected = rng.choice(raw.shape[0], size=int(max_fit_voxels), replace=False)
        fit_raw = raw[np.sort(selected)]

    n_components: float | None = percent / 100.0 if percent < 100.0 else None
    pca = PCA(n_components=n_components, svd_solver="full")
    pca.fit(fit_raw)
    _orient_pca_by_dominant_loading(pca)
    reduced = pca.transform(raw)

    stds = np.std(reduced, axis=0)
    nonconstant = stds > 1e-8
    if not np.any(nonconstant):
        raise ValueError("PCA produced no non-constant components")
    reduced = np.nan_to_num(
        reduced[:, nonconstant],
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(np.float32)
    z = _zscore_columns(reduced)

    explained = float(np.sum(pca.explained_variance_ratio_[nonconstant]))
    return reduced, z, int(z.shape[1]), explained


def _prepare_clustering_spaces(
    raw: np.ndarray,
    *,
    clustering_space: str,
    pca_variance_percent: float,
    seed: int,
    max_fit_voxels: int | None,
) -> tuple[np.ndarray, np.ndarray, int, float, str, str]:
    """Prepare the clustering matrix and standardized enrichment matrix."""
    space = _validate_clustering_space(clustering_space)
    if space == "pca":
        cluster_x, enrichment_x, n_components, explained = _pca_scores_and_zscore(
            raw,
            pca_variance_percent=pca_variance_percent,
            seed=seed,
            max_fit_voxels=max_fit_voxels,
        )
        return (
            cluster_x,
            enrichment_x,
            n_components,
            explained,
            "pca_scores_then_component_zscore_for_enrichment",
            "mean_zscored_pca_components",
        )

    enrichment_x = _zscore_columns(raw)
    return (
        enrichment_x,
        enrichment_x,
        0,
        0.0,
        "featurewise_zscore",
        "mean_zscored_filtered_features",
    )


def _fit_cluster_labels(
    x: np.ndarray,
    *,
    n_clusters: int,
    clusterer: str,
    seed: int,
    n_init: int,
    max_iter: int,
    max_fit_voxels: int | None,
) -> np.ndarray:
    """Fit the requested unsupervised model and predict labels for all voxels."""
    method = _validate_clusterer(clusterer)
    fit_x = x
    if max_fit_voxels and x.shape[0] > int(max_fit_voxels):
        rng = np.random.default_rng(int(seed))
        selected = rng.choice(x.shape[0], size=int(max_fit_voxels), replace=False)
        fit_x = x[np.sort(selected)]

    if method == "gmm":
        last_error = None
        for reg_covar in GMM_REG_COVAR_RETRY_VALUES:
            model = GaussianMixture(
                n_components=int(n_clusters),
                covariance_type="diag",
                reg_covar=float(reg_covar),
                n_init=int(n_init),
                max_iter=int(max_iter),
                random_state=int(seed),
            )
            try:
                model.fit(fit_x)
                return model.predict(x).astype(np.int16)
            except ValueError as exc:
                last_error = exc
        if last_error is not None:
            raise last_error
        raise RuntimeError("GMM fitting failed without raising a ValueError.")
    else:
        model = KMeans(
            n_clusters=int(n_clusters),
            n_init=int(n_init),
            max_iter=int(max_iter),
            random_state=int(seed),
        )
        model.fit(fit_x)
        return model.predict(x).astype(np.int16)


def phenotypic_tier_clustering(
    feature_maps: np.ndarray,
    roi_mask: np.ndarray,
    feature_names: list[str],
    seed: int = 42,
    method: str | None = None,
    max_fit_voxels: int = 50000,
    feature_subset: list[str] | str | None = None,
    pca_variance_percent: float = DEFAULT_PCA_VARIANCE_PERCENT,
    clustering_space: str = "pca",
    clusterer: str | None = None,
    score_config: RadiomicDeviationConfig | None = None,
) -> TierClusteringResult:
    """Cluster ROI voxels into 3 phenotypic tiers using voxel features.

    Pipeline:
      1. Filter to selected feature classes (default: all)
      2. Remove non-finite and constant feature channels
      3. Build either retained raw PCA scores or z-scored filtered features
      4. Cluster in the selected space with K-means or a Gaussian mixture
      5. Z-score the selected representation for enrichment scoring
      6. Rank clusters by mean composite standardized score
      7. Assign tiers: tier_h (highest), tier_m (middle), tier_l (lowest)

    Parameters
    ----------
    feature_maps : (C, Z, Y, X) float array — voxel feature maps
    roi_mask : (1, Z, Y, X) uint8 array — ROI mask
    feature_names : list of C feature names
    seed : random seed
    method : deprecated alias for clusterer
    max_fit_voxels : subsample for clustering if ROI has more voxels
    feature_subset : which feature classes to use for clustering.
        None or "all" uses all features (default, best for CT/MRI).
        A list of class names, e.g. ["firstorder"] for intensity-only,
        ["firstorder", "glcm"] for intensity+texture.
        Valid classes: firstorder, glcm, glrlm, glszm, gldm, ngtdm.
    pca_variance_percent : percentage of filtered feature variance retained by
        PCA. Default 95. Ignored when clustering_space="zscore".
    clustering_space : "pca" or "zscore"
        PCA clusters on unstandardized retained PCA scores. Z-score clusters
        directly on standardized filtered feature channels.
    clusterer : "kmeans" or "gmm"

    Returns
    -------
    TierClusteringResult with tier_h/tier_m/tier_l masks and enrichment maps
    """
    if feature_maps.ndim != 4:
        raise ValueError("feature_maps must be (C, Z, Y, X)")
    if roi_mask.ndim != 4 or roi_mask.shape[0] != 1:
        raise ValueError("roi_mask must be (1, Z, Y, X)")

    fm_sel, fn_sel = _filter_by_feature_classes(
        feature_maps, feature_names, feature_subset)

    mask_3d = roi_mask[0].astype(bool)
    roi_indices = np.flatnonzero(mask_3d.ravel())
    if roi_indices.size < 3:
        raise ValueError("ROI has fewer than 3 voxels, cannot cluster")

    filtered_maps, valid_names, _ = filter_constant_features(
        fm_sel,
        roi_mask,
        fn_sel,
    )
    if not valid_names:
        raise ValueError("No valid feature channels found in ROI")
    raw = filtered_maps.reshape(len(valid_names), -1)[:, roi_indices].T
    n_features_used = len(valid_names)

    selected_clusterer = _validate_clusterer(clusterer or method or "kmeans")
    selected_space = _validate_clustering_space(clustering_space)

    (
        cluster_x,
        enrichment_x,
        n_pca_components,
        pca_explained_variance_ratio,
        normalization,
        score_definition,
    ) = _prepare_clustering_spaces(
        raw,
        clustering_space=selected_space,
        pca_variance_percent=pca_variance_percent,
        seed=seed,
        max_fit_voxels=max_fit_voxels,
    )

    labels = _fit_cluster_labels(
        cluster_x,
        n_clusters=3,
        clusterer=selected_clusterer,
        seed=seed,
        n_init=10,
        max_iter=300,
        max_fit_voxels=max_fit_voxels,
    )

    # Score standardized channels, including when clustering uses PCA.
    selected_score = _validate_score_config(score_config)
    if selected_score.mode == "legacy_signed_mean":
        enrichment_per_voxel = np.mean(enrichment_x, axis=1).astype(np.float32)
        score_metadata = {
            "schema_version": selected_score.schema_version,
            "score_mode": selected_score.mode,
            "score_definition": "legacy_mean_selected_representation_zscore",
            "sign_invariant": False,
            "valid_feature_count": len(valid_names),
        }
    else:
        enrichment_per_voxel, score_metadata = radiomic_deviation_scores(
            raw,
            valid_names,
            selected_score,
        )
    score_definition = str(score_metadata["score_definition"])

    cluster_enrichment = {}
    for c in range(3):
        mask_c = labels == c
        if np.any(mask_c):
            cluster_enrichment[c] = float(np.mean(enrichment_per_voxel[mask_c]))
        else:
            cluster_enrichment[c] = -np.inf

    sorted_by_enrichment = sorted(cluster_enrichment.keys(),
                                   key=lambda c: cluster_enrichment[c],
                                   reverse=True)
    tier_h_id = sorted_by_enrichment[0]
    tier_m_id = sorted_by_enrichment[1]
    tier_l_id = sorted_by_enrichment[2]

    labels_3d = np.full(mask_3d.shape, -1, dtype=np.int16)
    labels_3d.ravel()[roi_indices] = labels

    tier_h_mask = np.zeros_like(mask_3d, dtype=np.uint8)
    tier_m_mask = np.zeros_like(mask_3d, dtype=np.uint8)
    tier_l_mask = np.zeros_like(mask_3d, dtype=np.uint8)
    tier_h_mask.ravel()[roi_indices[labels == tier_h_id]] = 1
    tier_m_mask.ravel()[roi_indices[labels == tier_m_id]] = 1
    tier_l_mask.ravel()[roi_indices[labels == tier_l_id]] = 1

    score_3d = np.zeros(mask_3d.shape, dtype=np.float32)
    score_3d.ravel()[roi_indices] = enrichment_per_voxel

    return TierClusteringResult(
        tier_h_mask=tier_h_mask[np.newaxis],
        tier_m_mask=tier_m_mask[np.newaxis],
        tier_l_mask=tier_l_mask[np.newaxis],
        labels_3d=labels_3d[np.newaxis],
        enrichment_scores=score_3d[np.newaxis],
        cluster_sizes={
            "tier_h": int(np.sum(tier_h_mask)),
            "tier_m": int(np.sum(tier_m_mask)),
            "tier_l": int(np.sum(tier_l_mask)),
        },
        cluster_enrichment_means={
            "tier_h": cluster_enrichment[tier_h_id],
            "tier_m": cluster_enrichment[tier_m_id],
            "tier_l": cluster_enrichment[tier_l_id],
        },
        method=selected_clusterer,
        n_features_used=n_features_used,
        n_pca_components=n_pca_components,
        pca_variance_percent=float(pca_variance_percent),
        pca_explained_variance_ratio=pca_explained_variance_ratio,
        clustering_space=selected_space,
        clusterer=selected_clusterer,
        normalization=normalization,
        score_definition=score_definition,
        score_metadata=score_metadata,
    )


def spatial_interaction_features(
    tier_result: TierClusteringResult,
    pes_mask: np.ndarray | None = None,
) -> "SpatialFeatures":
    """Compute spatial interaction features for tier-based clustering.

    Parameters
    ----------
    tier_result : TierClusteringResult from phenotypic_tier_clustering()
    pes_mask : (1, Z, Y, X) uint8 array (optional) — PES mask

    Returns
    -------
    SpatialFeatures with dict of feature_name → float
    """
    tier_h = tier_result.tier_h_mask[0].astype(bool)
    tier_m = tier_result.tier_m_mask[0].astype(bool)
    tier_l = tier_result.tier_l_mask[0].astype(bool)
    roi = tier_h | tier_m | tier_l

    roi_vol = max(int(np.sum(roi)), 1)
    tier_h_vol = int(np.sum(tier_h))
    tier_m_vol = int(np.sum(tier_m))
    tier_l_vol = int(np.sum(tier_l))

    feats: dict[str, float] = {}

    feats["tier_h_volume_fraction"] = float(tier_h_vol / roi_vol)
    feats["tier_m_volume_fraction"] = float(tier_m_vol / roi_vol)
    feats["tier_l_volume_fraction"] = float(tier_l_vol / roi_vol)

    struct = ndimage.generate_binary_structure(3, 1)

    tier_h_dilated = ndimage.binary_dilation(tier_h, structure=struct, iterations=1)
    tier_h_surface = tier_h_dilated & ~tier_h

    tier_h_surface_size = max(int(np.sum(tier_h_surface)), 1)
    feats["tier_h_tier_l_interface"] = float(np.sum(tier_h_surface & tier_l) / tier_h_surface_size)
    feats["tier_h_tier_m_interface"] = float(np.sum(tier_h_surface & tier_m) / tier_h_surface_size)

    if tier_h_vol > 0 and tier_m_vol > 0:
        tier_h_dist = ndimage.distance_transform_edt(~tier_h)
        tier_m_distances = tier_h_dist[tier_m]
        feats["tier_m_distance_to_h_mean"] = float(np.mean(tier_m_distances))
        feats["tier_m_distance_to_h_std"] = float(np.std(tier_m_distances))
    else:
        feats["tier_m_distance_to_h_mean"] = 0.0
        feats["tier_m_distance_to_h_std"] = 0.0

    if tier_h_vol > 0:
        try:
            from scipy.spatial import ConvexHull
            tier_h_coords = np.argwhere(tier_h)
            if tier_h_coords.shape[0] >= 4:
                hull = ConvexHull(tier_h_coords)
                feats["tier_h_solidity"] = float(tier_h_vol / max(hull.volume, 1.0))
            else:
                feats["tier_h_solidity"] = 1.0
        except Exception:
            feats["tier_h_solidity"] = 1.0
    else:
        feats["tier_h_solidity"] = 0.0

    labels_3d = tier_result.labels_3d[0]
    n_slices_with_roi = 0
    entropy_sum = 0.0
    for z_idx in range(labels_3d.shape[0]):
        slice_roi = roi[z_idx]
        if not np.any(slice_roi):
            continue
        n_slices_with_roi += 1
        slice_labels = labels_3d[z_idx][slice_roi]
        unique, counts = np.unique(slice_labels[slice_labels >= 0], return_counts=True)
        if len(unique) < 2:
            continue
        probs = counts / counts.sum()
        entropy_sum += float(-np.sum(probs * np.log2(probs + 1e-12)))
    feats["spatial_entropy"] = float(entropy_sum / max(n_slices_with_roi, 1))

    has_pes = pes_mask is not None
    if has_pes:
        pes = pes_mask[0].astype(bool) if pes_mask.ndim == 4 else pes_mask.astype(bool)
        pes_vol = int(np.sum(pes))
    else:
        pes_vol = 0

    if has_pes and pes_vol > 0:
        feats["pes_volume_fraction"] = float(pes_vol / roi_vol)

        labeled_pes, n_components = ndimage.label(pes, structure=struct)
        feats["pes_fragmentation"] = float(n_components)

        if tier_h_vol > 0:
            tier_h_com = np.array(ndimage.center_of_mass(tier_h))
            pes_com = np.array(ndimage.center_of_mass(pes))
            dist = float(np.linalg.norm(pes_com - tier_h_com))
            tier_h_radius = float((3.0 * tier_h_vol / (4.0 * np.pi)) ** (1.0 / 3.0))
            feats["pes_eccentricity"] = float(dist / max(tier_h_radius, 1.0))
        else:
            feats["pes_eccentricity"] = 0.0

        pes_dilated = ndimage.binary_dilation(pes, structure=struct, iterations=1)
        pes_surface = pes_dilated & ~pes
        pes_surface_size = max(int(np.sum(pes_surface)), 1)
        feats["pes_tier_m_adjacency"] = float(np.sum(pes_surface & tier_m) / pes_surface_size)

        if pes_vol >= 4:
            pes_eroded = ndimage.binary_erosion(pes, structure=struct, iterations=1)
            pes_surface_voxels = int(np.sum(pes & ~pes_eroded))
            sphere_sa = float((36.0 * np.pi * pes_vol ** 2) ** (1.0 / 3.0))
            feats["pes_sphericity"] = float(sphere_sa / max(pes_surface_voxels, 1.0))
        else:
            feats["pes_sphericity"] = 0.0
    else:
        feats["pes_volume_fraction"] = 0.0
        feats["pes_fragmentation"] = 0.0
        feats["pes_eccentricity"] = 0.0
        feats["pes_tier_m_adjacency"] = 0.0
        feats["pes_sphericity"] = 0.0

    return SpatialFeatures(features=feats)


def filter_constant_features(
    feature_maps: np.ndarray,
    roi_mask: np.ndarray,
    feature_names: list[str],
    std_threshold: float = 1e-6,
) -> tuple[np.ndarray, list[str], list[str]]:
    """Remove non-finite and constant feature channels within a mask.

    Parameters
    ----------
    feature_maps : (C, Z, Y, X) float array
    roi_mask : (1, Z, Y, X) uint8 array
    feature_names : list of C names
    std_threshold : features with std below this are dropped

    Returns
    -------
    filtered_maps : (C', Z, Y, X) — only non-constant channels
    kept_names : list of C' kept feature names
    dropped_names : list of non-finite or constant feature names
    """
    mask_flat = roi_mask[0].astype(bool).ravel()
    roi_indices = np.flatnonzero(mask_flat)
    if roi_indices.size == 0:
        return feature_maps, list(feature_names), []

    keep_idx: list[int] = []
    dropped: list[str] = []
    for c in range(feature_maps.shape[0]):
        # Float64 avoids cancellation that can disguise constant float32 maps.
        vals = feature_maps[c].ravel()[roi_indices].astype(np.float64, copy=False)
        if vals.size == 0 or not np.all(np.isfinite(vals)):
            dropped.append(feature_names[c])
        elif float(np.std(vals)) < std_threshold:
            dropped.append(feature_names[c])
        else:
            keep_idx.append(c)

    if not keep_idx:
        return feature_maps[:0], [], list(feature_names)

    return (
        feature_maps[keep_idx],
        [feature_names[i] for i in keep_idx],
        dropped,
    )


def construct_pes(
    feature_maps: np.ndarray,
    parent_mask: np.ndarray,
    feature_names: list[str],
    config: "PESConfig | None" = None,
) -> "PESSubregionResult":
    """Construct one distilled PES subregion within a phenotypic parent tier.

    Pipeline:
      1. Remove non-finite and constant feature channels
      2. Build retained PCA scores or z-scored filtered features
      3. Cluster the selected space with K-means or a Gaussian mixture
      4. Standardize the selected representation for enrichment scoring
      5. Select the cluster with the higher mean composite enrichment score
      6. Post-process the selected mask

    Parameters
    ----------
    feature_maps : (C, Z, Y, X) float array
    parent_mask : (1, Z, Y, X) uint8 parent tier mask
    feature_names : list of C names
    config : PESConfig (optional)

    Returns
    -------
    PESSubregionResult
    """
    config = config or PESConfig()
    rules = config.stopping_rules
    _validate_pca_variance_percent(config.pca_variance_percent)

    if feature_maps.ndim != 4:
        raise ValueError("feature_maps must be (C, Z, Y, X)")
    if parent_mask.ndim != 4 or parent_mask.shape[0] != 1:
        raise ValueError("parent_mask must be (1, Z, Y, X)")

    filtered_maps, kept_names, dropped = filter_constant_features(
        feature_maps, parent_mask, feature_names,
    )
    if len(kept_names) < int(rules.min_valid_features):
        return PESSubregionResult(
            completed=False,
            stopping_reason="no_valid_feature_channels",
            selected_feature_names=kept_names,
        )

    parent = parent_mask[0].astype(bool)
    parent_voxels = int(np.sum(parent))
    if parent_voxels < int(rules.min_parent_voxels):
        return PESSubregionResult(
            completed=False,
            stopping_reason="parent_too_small",
            selected_feature_names=kept_names,
        )

    parent_indices = np.flatnonzero(parent.ravel())
    raw = filtered_maps.reshape(len(kept_names), -1)[:, parent_indices].T
    if raw.shape[0] < int(rules.min_valid_voxels):
        return PESSubregionResult(
            completed=False,
            stopping_reason="parent_too_small",
            selected_feature_names=kept_names,
        )

    try:
        (
            cluster_x,
            enrichment_x,
            n_pca_components,
            pca_explained_variance_ratio,
            normalization,
            score_definition,
        ) = _prepare_clustering_spaces(
            raw,
            clustering_space=config.clustering.clustering_space,
            pca_variance_percent=config.pca_variance_percent,
            seed=int(config.clustering.random_seed),
            max_fit_voxels=config.clustering.max_fit_voxels,
        )
    except ValueError:
        return PESSubregionResult(
            completed=False,
            stopping_reason="pca_failed",
            selected_feature_names=kept_names,
        )

    labels = _fit_cluster_labels(
        cluster_x,
        n_clusters=2,
        clusterer=config.clustering.clusterer,
        seed=int(config.clustering.random_seed),
        n_init=int(config.clustering.n_init),
        max_iter=int(config.clustering.max_iter),
        max_fit_voxels=config.clustering.max_fit_voxels,
    )

    selected_score = _validate_score_config(config.score)
    if selected_score.mode == "legacy_signed_mean":
        scores = np.mean(enrichment_x, axis=1).astype(np.float32)
        score_metadata = {
            "schema_version": selected_score.schema_version,
            "score_mode": selected_score.mode,
            "score_definition": "legacy_mean_selected_representation_zscore",
            "sign_invariant": False,
            "valid_feature_count": len(kept_names),
        }
    else:
        scores, score_metadata = radiomic_deviation_scores(
            raw,
            kept_names,
            selected_score,
        )
    score_definition = str(score_metadata["score_definition"])
    cluster_scores = [
        float(np.mean(scores[labels == cluster])) if np.any(labels == cluster) else -np.inf
        for cluster in range(2)
    ]
    selected_cluster = int(np.argmax(cluster_scores))
    selected_members = labels == selected_cluster

    raw_flat = np.zeros(parent.size, dtype=bool)
    raw_flat[parent_indices[selected_members]] = True
    raw_pes = raw_flat.reshape(parent.shape) & parent
    pes_mask, post_stats = _postprocess_pes_mask(
        raw_pes,
        parent,
        config.postprocessing,
        rules,
    )

    pes_voxels = int(np.sum(pes_mask))
    fraction = float(pes_voxels / max(parent_voxels, 1))
    counts = [int(np.sum(labels == cluster)) for cluster in range(2)]
    stopping_reason = "completed"
    if sum(count > 0 for count in counts) < 2:
        stopping_reason = "single_nonempty_cluster"
    elif pes_voxels < int(rules.min_pes_voxels):
        stopping_reason = "pes_too_small"
    elif fraction < float(rules.min_pes_fraction_of_parent):
        stopping_reason = "pes_too_small"
    elif fraction > float(rules.max_pes_fraction_of_parent):
        stopping_reason = "pes_too_large"

    if stopping_reason != "completed":
        return PESSubregionResult(
            completed=False,
            stopping_reason=stopping_reason,
            selected_feature_names=kept_names,
        )

    score_map = np.zeros(parent.shape, dtype=np.float32)
    score_map.ravel()[parent_indices] = scores.astype(np.float32)
    labels_3d = np.full(parent.shape, -1, dtype=np.int16)
    labels_3d.ravel()[parent_indices] = labels
    summary = {
        "parent_voxel_count": parent_voxels,
        "pes_voxel_count": pes_voxels,
        "pes_fraction_of_parent": fraction,
        "selected_cluster": selected_cluster,
        "cluster_scores": cluster_scores,
        "cluster_voxel_counts": counts,
        "valid_feature_count": len(kept_names),
        "dropped_feature_count": len(dropped),
        "n_pca_components": n_pca_components,
        "pca_variance_percent": float(config.pca_variance_percent),
        "pca_explained_variance_ratio": pca_explained_variance_ratio,
        "clustering_space": config.clustering.clustering_space,
        "clusterer": config.clustering.clusterer,
        "normalization": normalization,
        "score_definition": score_definition,
        "score_metadata": score_metadata,
        **post_stats,
    }
    return PESSubregionResult(
        completed=True,
        stopping_reason="completed",
        selected_feature_names=kept_names,
        mask=pes_mask[np.newaxis].astype(np.uint8),
        labels=labels_3d[np.newaxis],
        score_map=score_map[np.newaxis],
        summary=summary,
    )


# Spatial Interaction Features

@dataclass
class SpatialFeatures:
    """Spatial interaction features between sub-regions."""
    features: dict[str, float]


# Standard Habitat Analysis (intensity-based baseline)

@dataclass
class HabitatResult:
    """Result of standard intensity-based habitat analysis."""
    masks: list[np.ndarray]   # list of K masks, each [1, Z, Y, X] uint8
    labels_3d: np.ndarray     # [1, Z, Y, X] int16
    cluster_sizes: list[int]
    n_clusters: int


def standard_habitat_analysis(
    image_array: np.ndarray,
    roi_mask: np.ndarray,
    n_clusters: int = 3,
    seed: int = 42,
) -> HabitatResult:
    """Standard intensity-based habitat analysis baseline.

    Applies k-means clustering on RAW VOXEL INTENSITIES only (not radiomics
    features). This is the most common "habitat analysis" approach in the
    literature and serves as a fair training-free baseline.

    Parameters
    ----------
    image_array : (Z, Y, X) or (1, Z, Y, X) float array — raw image
    roi_mask : (1, Z, Y, X) uint8 — ROI mask
    n_clusters : number of clusters (default 3)
    seed : random seed

    Returns
    -------
    HabitatResult
    """
    if image_array.ndim == 4:
        image_array = image_array[0]
    mask_3d = roi_mask[0].astype(bool)
    roi_indices = np.flatnonzero(mask_3d.ravel())

    if roi_indices.size < n_clusters * 4:
        masks = [roi_mask.copy()]
        for _ in range(n_clusters - 1):
            masks.append(np.zeros_like(roi_mask))
        return HabitatResult(
            masks=masks,
            labels_3d=np.zeros_like(roi_mask, dtype=np.int16),
            cluster_sizes=[int(roi_indices.size)] + [0] * (n_clusters - 1),
            n_clusters=n_clusters,
        )

    intensities = image_array.ravel()[roi_indices].astype(np.float64)
    intensities = np.nan_to_num(intensities, nan=0.0, posinf=0.0, neginf=0.0)

    X = intensities.reshape(-1, 1)

    fit_X = X
    if X.shape[0] > 50000:
        rng = np.random.default_rng(seed)
        sel = rng.choice(X.shape[0], size=50000, replace=False)
        fit_X = X[np.sort(sel)]

    km = KMeans(n_clusters=n_clusters, n_init=10, max_iter=300, random_state=seed)
    km.fit(fit_X)
    labels = km.predict(X).astype(np.int16)

    means = [float(np.mean(intensities[labels == c])) for c in range(n_clusters)]
    order = np.argsort(means)

    labels_3d = np.full(mask_3d.shape, -1, dtype=np.int16)
    labels_3d.ravel()[roi_indices] = labels

    masks = []
    cluster_sizes = []
    for new_idx, old_idx in enumerate(order):
        m = np.zeros(mask_3d.shape, dtype=np.uint8)
        m.ravel()[roi_indices[labels == old_idx]] = 1
        masks.append(m[np.newaxis])
        cluster_sizes.append(int(np.sum(labels == old_idx)))

    remap = np.full(n_clusters, -1, dtype=np.int16)
    for new_idx, old_idx in enumerate(order):
        remap[old_idx] = new_idx
    valid = labels_3d >= 0
    labels_3d[valid] = remap[labels_3d[valid]]

    return HabitatResult(
        masks=masks,
        labels_3d=labels_3d[np.newaxis],
        cluster_sizes=cluster_sizes,
        n_clusters=n_clusters,
    )


@dataclass
class ClusteringConfig:
    n_clusters: int = 2
    random_seed: int = 0
    n_init: int = 10
    max_iter: int = 300
    max_fit_voxels: int | None = 50000
    clustering_space: str = "pca"
    clusterer: str = "kmeans"


@dataclass
class StoppingRules:
    min_parent_voxels: int = 32
    min_pes_voxels: int = 16
    min_pes_fraction_of_parent: float = 0.01
    max_pes_fraction_of_parent: float = 0.95
    min_valid_features: int = 1
    min_valid_voxels: int = 32


@dataclass
class PESPostProcessingConfig:
    enabled: bool = True
    min_component_voxels: int = 8
    min_component_fraction_of_parent: float = 0.002
    keep_largest_components: int = 0
    closing_radius: int = 2
    opening_radius: int = 1
    dilation_radius: int = 0
    fill_holes: bool = True
    smooth_in_plane: bool = False
    smooth_before_filter: bool = True


@dataclass
class PESConfig:
    pca_variance_percent: float = DEFAULT_PCA_VARIANCE_PERCENT
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    postprocessing: PESPostProcessingConfig = field(default_factory=PESPostProcessingConfig)
    stopping_rules: StoppingRules = field(default_factory=StoppingRules)
    score: RadiomicDeviationConfig = field(default_factory=RadiomicDeviationConfig)


@dataclass
class PESSubregionResult:
    completed: bool
    stopping_reason: str
    selected_feature_names: list[str]
    mask: np.ndarray | None = None
    labels: np.ndarray | None = None
    score_map: np.ndarray | None = None
    summary: dict[str, Any] = field(default_factory=dict)


@dataclass
class PESResult:
    stopping_reason: str
    selected_tiers: tuple[str, ...]
    completed_tiers: tuple[str, ...]
    failed_tiers: dict[str, str]
    per_tier: dict[str, PESSubregionResult]
    fused: PESSubregionResult | None
    summary: dict[str, Any]


def tier_mask_by_name(tier_result: TierClusteringResult, tier: str) -> np.ndarray:
    """Return one phenotypic tier mask using H/M/L naming."""
    key = str(tier).strip().upper()
    if key == "H":
        return tier_result.tier_h_mask
    if key == "M":
        return tier_result.tier_m_mask
    if key == "L":
        return tier_result.tier_l_mask
    raise ValueError(f"Unsupported tier: {tier!r}; expected H, M, or L")


def construct_fused_tier_pes(
    feature_maps: np.ndarray,
    tier_result: TierClusteringResult,
    feature_names: list[str],
    config: PESConfig | None = None,
    selected_tiers: list[str] | tuple[str, ...] = ("H", "M", "L"),
) -> PESResult:
    """Construct PES inside selected H/M/L tiers and fuse them by union.

    The final PES ROI is the union of successfully distilled tier masks.
    """
    config = config or PESConfig()
    selected = [str(t).strip().upper() for t in selected_tiers]
    selected = [t for t in selected if t in {"H", "M", "L"}]
    if not selected:
        selected = ["H", "M", "L"]

    roi_mask = (
        tier_result.tier_h_mask.astype(bool)
        | tier_result.tier_m_mask.astype(bool)
        | tier_result.tier_l_mask.astype(bool)
    )
    fused = np.zeros_like(roi_mask, dtype=bool)
    fused_labels = np.full(roi_mask.shape, -1, dtype=np.int16)
    fused_score = np.zeros(roi_mask.shape, dtype=np.float32)
    per_tier: dict[str, PESSubregionResult] = {}
    completed_tiers: list[str] = []
    failed_tiers: dict[str, str] = {}

    tier_label_values = {"H": 2, "M": 1, "L": 0}
    for tier in selected:
        parent = tier_mask_by_name(tier_result, tier)
        parent_voxels = int(np.sum(parent > 0))
        if parent_voxels < int(config.stopping_rules.min_parent_voxels):
            failed_tiers[tier] = "parent_too_small"
            continue
        result = construct_pes(feature_maps, parent, feature_names, config)
        if not result.completed or result.mask is None or result.labels is None or result.score_map is None:
            failed_tiers[tier] = result.stopping_reason
            continue

        tier_mask = result.mask.astype(bool)
        if int(np.sum(tier_mask)) == 0:
            failed_tiers[tier] = "empty_pes_mask"
            continue

        fused |= tier_mask
        fused_labels[tier_mask] = tier_label_values[tier]
        fused_score[tier_mask] = result.score_map.astype(np.float32, copy=False)[tier_mask]
        completed_tiers.append(tier)

        tier_summary = dict(result.summary)
        tier_summary["tier"] = tier
        tier_summary["tier_parent_voxel_count"] = parent_voxels
        tier_summary["tier_pes_voxel_count"] = int(np.sum(tier_mask))
        tier_summary["tier_pes_fraction_of_roi"] = float(int(np.sum(tier_mask)) / max(int(np.sum(roi_mask)), 1))
        per_tier[tier] = PESSubregionResult(
            completed=True,
            stopping_reason="completed",
            selected_feature_names=list(result.selected_feature_names),
            mask=result.mask.astype(np.uint8, copy=False),
            labels=result.labels.astype(np.int16, copy=False),
            score_map=result.score_map.astype(np.float32, copy=False),
            summary=tier_summary,
        )

    roi_voxels = int(np.sum(roi_mask))
    fused_voxels = int(np.sum(fused))
    fused_summary = {
        "mode": f"tier_fused_{config.clustering.clustering_space}_{config.clustering.clusterer}",
        "selected_tiers": selected,
        "completed_tiers": completed_tiers,
        "failed_tiers": failed_tiers,
        "fusion_rule": "union",
        "original_roi_voxel_count": roi_voxels,
        "pes_voxel_count": fused_voxels,
        "pes_fraction_of_original_roi": float(fused_voxels / max(roi_voxels, 1)),
        "tier_cluster_sizes": dict(tier_result.cluster_sizes),
        "tier_cluster_enrichment_means": dict(tier_result.cluster_enrichment_means),
        "pca_variance_percent": float(config.pca_variance_percent),
        "clustering_space": config.clustering.clustering_space,
        "clusterer": config.clustering.clusterer,
        "normalization": tier_result.normalization,
        "score_definition": tier_result.score_definition,
        "score_schema_version": str(
            tier_result.score_metadata.get("schema_version", "")
        ),
        "score_mode": str(tier_result.score_metadata.get("score_mode", "")),
        "score_sign_invariant": bool(
            tier_result.score_metadata.get("sign_invariant", False)
        ),
    }
    fused_result = None
    stopping_reason = "no_successful_tiers"
    if fused_voxels > 0:
        stopping_reason = "completed"
        fused_result = PESSubregionResult(
            completed=True,
            stopping_reason="completed",
            selected_feature_names=[],
            mask=fused.astype(np.uint8, copy=False),
            labels=fused_labels.astype(np.int16, copy=False),
            score_map=fused_score.astype(np.float32, copy=False),
            summary=fused_summary,
        )

    return PESResult(
        stopping_reason=stopping_reason,
        selected_tiers=tuple(selected),
        completed_tiers=tuple(completed_tiers),
        failed_tiers=failed_tiers,
        per_tier=per_tier,
        fused=fused_result,
        summary=fused_summary,
    )


def sitk_image_to_array_1czyx(image: sitk.Image, dtype=np.float32) -> np.ndarray:
    array = sitk.GetArrayFromImage(image).astype(dtype, copy=False)
    return array[np.newaxis, ...]


def resample_mask_to_reference(mask: sitk.Image, label: int, reference: sitk.Image) -> np.ndarray:
    roi = sitk.Equal(mask, int(label))
    roi = sitk.Resample(
        roi,
        reference,
        sitk.Transform(),
        sitk.sitkNearestNeighbor,
        0,
        sitk.sitkUInt8,
    )
    return sitk_image_to_array_1czyx(roi, dtype=np.uint8)


def resample_image_to_reference(image: sitk.Image, reference: sitk.Image) -> np.ndarray:
    resampled = sitk.Resample(
        image,
        reference,
        sitk.Transform(),
        sitk.sitkLinear,
        0.0,
        sitk.sitkFloat32,
    )
    return sitk_image_to_array_1czyx(resampled, dtype=np.float32)


def feature_tensor_from_result(result: dict[str, Any]) -> tuple[np.ndarray, list[str], sitk.Image]:
    feature_names: list[str] = []
    feature_arrays: list[np.ndarray] = []
    reference: sitk.Image | None = None
    for key, value in result.items():
        if not str(key).startswith("original_") or not isinstance(value, sitk.Image):
            continue
        if reference is None:
            reference = value
        array = sitk.GetArrayFromImage(value).astype(np.float32, copy=False)
        if array.shape != sitk.GetArrayFromImage(reference).shape:
            continue
        feature_names.append(str(key))
        feature_arrays.append(array)
    if reference is None or not feature_arrays:
        raise ValueError("No voxel feature maps were found in the extraction result")
    return np.stack(feature_arrays, axis=0).astype(np.float32, copy=False), feature_names, reference


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _binary_structure(ndim: int, radius: int) -> np.ndarray:
    structure = ndimage.generate_binary_structure(ndim, 1)
    radius = int(radius)
    if radius <= 1:
        return structure
    return ndimage.iterate_structure(structure, radius)


def _filter_components(
    mask: np.ndarray,
    *,
    min_component_voxels: int,
    keep_largest_components: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    labels, n_components = ndimage.label(mask, structure=np.ones((3, 3, 3), dtype=bool))
    if n_components == 0:
        return mask, {
            "component_count_before_filter": 0,
            "component_count_after_filter": 0,
            "removed_component_count": 0,
            "removed_voxel_count": 0,
            "kept_component_sizes": [],
        }

    sizes = np.bincount(labels.ravel())[1:]
    component_ids = np.arange(1, n_components + 1)
    keep = component_ids[sizes >= int(min_component_voxels)]
    if keep.size == 0:
        keep = np.asarray([int(np.argmax(sizes)) + 1])
    if int(keep_largest_components) > 0 and keep.size > int(keep_largest_components):
        keep_sizes = sizes[keep - 1]
        order = np.argsort(keep_sizes)[::-1][: int(keep_largest_components)]
        keep = keep[order]

    filtered = np.isin(labels, keep)
    kept_sizes = [int(sizes[int(component_id) - 1]) for component_id in keep]
    return filtered, {
        "component_count_before_filter": int(n_components),
        "component_count_after_filter": int(keep.size),
        "removed_component_count": int(n_components - keep.size),
        "removed_voxel_count": int(np.sum(mask) - np.sum(filtered)),
        "kept_component_sizes": sorted(kept_sizes, reverse=True),
    }


def _smooth_mask(mask: np.ndarray, config: PESPostProcessingConfig) -> np.ndarray:
    smoothed = mask
    if int(config.closing_radius) > 0:
        if config.smooth_in_plane:
            structure_2d = _binary_structure(2, int(config.closing_radius))
            smoothed = np.stack(
                [ndimage.binary_closing(section, structure=structure_2d) for section in smoothed],
                axis=0,
            )
        else:
            smoothed = ndimage.binary_closing(smoothed, structure=_binary_structure(3, int(config.closing_radius)))
    if int(config.dilation_radius) > 0:
        if config.smooth_in_plane:
            structure_2d = _binary_structure(2, int(config.dilation_radius))
            smoothed = np.stack(
                [ndimage.binary_dilation(section, structure=structure_2d) for section in smoothed],
                axis=0,
            )
        else:
            smoothed = ndimage.binary_dilation(smoothed, structure=_binary_structure(3, int(config.dilation_radius)))
    if bool(config.fill_holes):
        if config.smooth_in_plane:
            smoothed = np.stack([ndimage.binary_fill_holes(section) for section in smoothed], axis=0)
        else:
            smoothed = ndimage.binary_fill_holes(smoothed)
    if int(config.opening_radius) > 0:
        if config.smooth_in_plane:
            structure_2d = _binary_structure(2, int(config.opening_radius))
            smoothed = np.stack(
                [ndimage.binary_opening(section, structure=structure_2d) for section in smoothed],
                axis=0,
            )
        else:
            smoothed = ndimage.binary_opening(smoothed, structure=_binary_structure(3, int(config.opening_radius)))
    return smoothed.astype(bool, copy=False)


def _postprocess_pes_mask(
    pes_mask: np.ndarray,
    parent_mask: np.ndarray,
    config: PESPostProcessingConfig,
    rules: StoppingRules,
) -> tuple[np.ndarray, dict[str, Any]]:
    raw = pes_mask.astype(bool, copy=False) & parent_mask.astype(bool, copy=False)
    raw_voxels = int(np.sum(raw))
    if not bool(config.enabled) or raw_voxels == 0:
        return raw, {
            "postprocessing_applied": False,
            "raw_pes_voxel_count": raw_voxels,
            "postprocessed_pes_voxel_count": raw_voxels,
        }

    parent_voxels = int(np.sum(parent_mask))
    min_component_voxels = max(
        int(config.min_component_voxels),
        int(np.ceil(parent_voxels * float(config.min_component_fraction_of_parent))),
    )

    if bool(config.smooth_before_filter):
        # Smoothing first bridges scattered clusters before component removal.
        smoothed = _smooth_mask(raw, config) & parent_mask
        smoothed, component_stats = _filter_components(
            smoothed,
            min_component_voxels=min_component_voxels,
            keep_largest_components=int(config.keep_largest_components),
        )
        smoothed &= parent_mask
        final_component_stats = component_stats
    else:
        # Preserve the legacy filter-smooth-filter order when requested.
        filtered, component_stats = _filter_components(
            raw,
            min_component_voxels=min_component_voxels,
            keep_largest_components=int(config.keep_largest_components),
        )
        smoothed = _smooth_mask(filtered, config) & parent_mask
        smoothed, final_component_stats = _filter_components(
            smoothed,
            min_component_voxels=min_component_voxels,
            keep_largest_components=int(config.keep_largest_components),
        )
        smoothed &= parent_mask

    fallback_used = False
    if int(np.sum(smoothed)) < int(rules.min_pes_voxels):
        smoothed = raw & parent_mask
        fallback_used = True

    stats = {
        "postprocessing_applied": True,
        "postprocessing_fallback_used": fallback_used,
        "postprocessing_smooth_before_filter": bool(config.smooth_before_filter),
        "raw_pes_voxel_count": raw_voxels,
        "postprocessed_pes_voxel_count": int(np.sum(smoothed)),
        "postprocessing_min_component_voxels": int(min_component_voxels),
        "postprocessing_keep_largest_components": int(config.keep_largest_components),
        "postprocessing_closing_radius": int(config.closing_radius),
        "postprocessing_dilation_radius": int(config.dilation_radius),
        "postprocessing_opening_radius": int(config.opening_radius),
        "postprocessing_fill_holes": bool(config.fill_holes),
        "postprocessing_smooth_in_plane": bool(config.smooth_in_plane),
        **component_stats,
        "final_component_count": int(final_component_stats["component_count_after_filter"]),
        "final_kept_component_sizes": final_component_stats["kept_component_sizes"],
    }
    return smoothed.astype(bool, copy=False), stats


def summarize_voxel_features(
    feature_maps: np.ndarray,
    mask: np.ndarray,
    feature_names: list[str],
    prefix: str,
) -> dict[str, float]:
    support = mask[0].astype(bool, copy=False)
    values: dict[str, float] = {}
    for idx, name in enumerate(feature_names):
        arr = feature_maps[idx]
        valid = support & np.isfinite(arr)
        if not np.any(valid):
            continue
        sample = arr[valid].astype(np.float64, copy=False)
        key = name.replace("original_", "")
        values[f"{prefix}__{key}__mean"] = float(np.mean(sample))
        values[f"{prefix}__{key}__std"] = float(np.std(sample))
        values[f"{prefix}__{key}__p10"] = float(np.percentile(sample, 10))
        values[f"{prefix}__{key}__p90"] = float(np.percentile(sample, 90))
    return values


def _create_dataset(group: h5py.Group, name: str, data: np.ndarray, compression: str = "lzf") -> h5py.Dataset:
    return group.create_dataset(name, data=data, compression=compression, shuffle=True)


def write_flashrad_hdf5(
    output_path: Path,
    *,
    case_id: str,
    roi_id: str,
    roi_name: str,
    image: np.ndarray,
    mask: np.ndarray,
    feature_maps: np.ndarray,
    feature_names: list[str],
    tier_result: TierClusteringResult,
    pes_result: PESResult,
    geometry: dict[str, Any] | None = None,
    processing: dict[str, Any] | None = None,
    compression: str = "lzf",
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        dir=str(output_path.parent),
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    string_dtype = h5py.string_dtype(encoding="utf-8")
    try:
        with h5py.File(tmp_path, "w") as h5:
            # Preserve the established cache contract for later scalar merges.
            h5.attrs["schema_name"] = "flashrad_hdf5"
            h5.attrs["schema_version"] = "0.1"
            h5.attrs["case_id"] = str(case_id)
            h5.attrs["created_by"] = "flash_radiomics"
            h5.attrs["created_utc"] = datetime.now(timezone.utc).isoformat()
            if geometry:
                h5.attrs["geometry_json"] = json.dumps(_jsonable(geometry), sort_keys=True)
            if processing:
                h5.attrs["processing_json"] = json.dumps(_jsonable(processing), sort_keys=True)

            roi_group = h5.require_group(f"rois/{roi_id}")
            roi_group.attrs["roi_id"] = str(roi_id)
            roi_group.attrs["roi_name"] = str(roi_name)
            roi_group.attrs["crop_shape"] = json.dumps(list(image.shape[1:]))
            roi_group.attrs["feature_channel_count"] = int(feature_maps.shape[0])
            roi_group.attrs["has_tier_pes"] = True
            roi_group.attrs["has_fused_pes"] = pes_result.fused is not None
            if pes_result.fused is not None:
                roi_group.attrs["default_pes_group"] = "phenotypes/tier_pes_fusion/fused"

            image_ds = _create_dataset(roi_group, "image", image.astype(np.float32, copy=False), compression)
            mask_ds = _create_dataset(roi_group, "mask", mask.astype(np.uint8, copy=False), compression)
            maps_ds = _create_dataset(roi_group, "feature_maps", feature_maps.astype(np.float32, copy=False), compression)
            names_ds = roi_group.create_dataset(
                "feature_names",
                data=np.asarray(feature_names, dtype=object),
                dtype=string_dtype,
            )
            for dataset in (image_ds, mask_ds, maps_ds):
                dataset.attrs["shape_convention"] = "channel_first_3d_crop"
                dataset.attrs["spatial_grid"] = "roi_crop_grid"
            maps_ds.attrs["feature_axis"] = 0
            maps_ds.attrs["feature_count"] = int(feature_maps.shape[0])
            maps_ds.attrs["feature_name_dataset"] = names_ds.name

            pes_group = roi_group.require_group("phenotypes/tier_pes_fusion")
            pes_group.attrs["method_name"] = "tier_pes_fusion"
            pes_group.attrs["fusion_rule"] = "union"
            pes_group.attrs["stopping_reason"] = pes_result.stopping_reason
            pes_group.attrs["selected_tiers_json"] = json.dumps(list(pes_result.selected_tiers))
            pes_group.attrs["completed_tiers_json"] = json.dumps(list(pes_result.completed_tiers))
            pes_group.attrs["failed_tiers_json"] = json.dumps(pes_result.failed_tiers, sort_keys=True)
            pes_group.attrs["pca_variance_percent"] = float(tier_result.pca_variance_percent)
            pes_group.attrs["tier_valid_feature_count"] = int(tier_result.n_features_used)
            pes_group.attrs["tier_pca_component_count"] = int(tier_result.n_pca_components)
            pes_group.attrs["tier_pca_explained_variance_ratio"] = float(
                tier_result.pca_explained_variance_ratio
            )
            pes_group.attrs["clustering_space"] = tier_result.clustering_space
            pes_group.attrs["clusterer"] = tier_result.clusterer
            pes_group.attrs["normalization"] = tier_result.normalization
            pes_group.attrs["score_definition"] = tier_result.score_definition
            pes_group.attrs["score_schema_version"] = str(
                tier_result.score_metadata.get("schema_version", "")
            )
            pes_group.attrs["score_mode"] = str(
                tier_result.score_metadata.get("score_mode", "")
            )
            pes_group.attrs["score_sign_invariant"] = bool(
                tier_result.score_metadata.get("sign_invariant", False)
            )
            pes_group.attrs["score_metadata_json"] = json.dumps(
                _jsonable(tier_result.score_metadata), sort_keys=True
            )
            tiers_group = pes_group.require_group("tiers")
            for tier in ("H", "M", "L"):
                tier_group = tiers_group.require_group(tier)
                tier_mask = tier_mask_by_name(tier_result, tier).astype(np.uint8, copy=False)
                _create_dataset(tier_group, "mask", tier_mask, compression)
                tier_group.attrs["tier_name"] = tier
                tier_group.attrs["voxel_count"] = int(np.sum(tier_mask))
                tier_group.attrs["enrichment_mean"] = float(
                    tier_result.cluster_enrichment_means[f"tier_{tier.lower()}"]
                )

            pes_by_tier_group = pes_group.require_group("pes_by_tier")
            for tier, subregion in pes_result.per_tier.items():
                if subregion.mask is None or subregion.labels is None or subregion.score_map is None:
                    continue
                tier_group = pes_by_tier_group.require_group(tier)
                for key, value in subregion.summary.items():
                    if isinstance(value, (dict, list, tuple)):
                        tier_group.attrs[f"{key}_json"] = json.dumps(_jsonable(value), sort_keys=True)
                    elif value is None:
                        tier_group.attrs[key] = ""
                    else:
                        tier_group.attrs[key] = value
                _create_dataset(tier_group, "mask", subregion.mask, compression)
                _create_dataset(tier_group, "labels", subregion.labels, compression)
                score_ds = _create_dataset(tier_group, "score_map", subregion.score_map, compression)
                score_ds.attrs["score_type"] = str(
                    subregion.summary.get("score_definition", tier_result.score_definition)
                )

            if pes_result.fused is not None:
                fused = pes_result.fused
                if fused.mask is not None and fused.labels is not None and fused.score_map is not None:
                    fused_group = pes_group.require_group("fused")
                    for key, value in fused.summary.items():
                        if isinstance(value, (dict, list, tuple)):
                            fused_group.attrs[f"{key}_json"] = json.dumps(_jsonable(value), sort_keys=True)
                        elif value is None:
                            fused_group.attrs[key] = ""
                        else:
                            fused_group.attrs[key] = value
                    _create_dataset(fused_group, "mask", fused.mask, compression)
                    _create_dataset(fused_group, "labels", fused.labels, compression)
                    _create_dataset(fused_group, "score_map", fused.score_map, compression)

            summaries_group = pes_group.require_group("summaries")
            summaries_group.create_dataset(
                "summary",
                data=json.dumps(_jsonable(pes_result.summary), sort_keys=True),
                dtype=string_dtype,
            )
            h5.flush()
        tmp_path.replace(output_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def write_metadata_json(
    output_path: Path,
    *,
    cases: dict[str, Any],
    processing: dict[str, Any],
    hdf5_files: dict[str, str],
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "cases": _jsonable(cases),
        "processing": _jsonable(processing),
        "hdf5_files": dict(hdf5_files),
    }
    output_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
