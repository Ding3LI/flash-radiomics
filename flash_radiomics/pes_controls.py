"""Outcome-free comparator masks for the PES reviewer experiments.

The controls in this module operate only on radiomic maps and masks.  They do
not inspect clinical labels.  Every returned representation is an ordered list
of named masks so scalar feature extraction can preserve region identity.
"""

from __future__ import annotations

from dataclasses import replace
import hashlib
from typing import Any
import warnings

import numpy as np
from scipy import ndimage
from sklearn.exceptions import ConvergenceWarning

from .spatial_phenotyping import (
    PESConfig,
    _fit_cluster_labels,
    _prepare_clustering_spaces,
    construct_pes,
    filter_constant_features,
    radiomic_deviation_scores,
)


RANDOM_CHILD_PREFIX = "control_random_child_structured_r"
RANDOM_CHILD_ENSEMBLE_TARGET = "control_random_child_structured_50x"
RANDOM_CHILD_REPRESENTATIONS = tuple(
    f"{RANDOM_CHILD_PREFIX}{repeat:02d}" for repeat in range(50)
)

CONTROL_REPRESENTATIONS = (
    "control_global_six_top3_structured",
    "control_parent_distance_core_structured",
    "ablation_pes_no_morphology_structured",
    "ablation_both_children_six_structured",
    "ablation_unweighted_deviation_structured",
    "ablation_legacy_signed_mean_structured",
) + RANDOM_CHILD_REPRESENTATIONS


def _as_bool_3d(mask: np.ndarray) -> np.ndarray:
    array = np.asarray(mask)
    if array.ndim == 4:
        array = array[0]
    if array.ndim != 3:
        raise ValueError("control masks must be three- or four-dimensional")
    return array.astype(bool, copy=False)


def _as_mask_4d(mask: np.ndarray) -> np.ndarray:
    return _as_bool_3d(mask)[np.newaxis].astype(np.uint8, copy=False)


def _exact_largest_values(
    values: np.ndarray,
    support: np.ndarray,
    voxel_count: int,
) -> np.ndarray:
    """Select exactly ``voxel_count`` support voxels with deterministic ties."""
    support_3d = _as_bool_3d(support)
    support_indices = np.flatnonzero(support_3d.ravel())
    requested = int(voxel_count)
    if requested <= 0 or requested > support_indices.size:
        raise ValueError("requested control volume is outside the support")
    local_values = np.asarray(values, dtype=np.float64).ravel()[support_indices]
    # Sort by descending value, then ascending flat index.
    order = np.lexsort((support_indices, -local_values))
    chosen = support_indices[order[:requested]]
    output = np.zeros(support_3d.size, dtype=np.uint8)
    output[chosen] = 1
    return output.reshape(support_3d.shape)[np.newaxis]


def _global_six_top3(
    feature_maps: np.ndarray,
    roi_mask: np.ndarray,
    feature_names: list[str],
    config: PESConfig,
) -> tuple[list[tuple[str, np.ndarray]], dict[str, Any]]:
    filtered, kept_names, dropped_names = filter_constant_features(
        feature_maps,
        roi_mask,
        feature_names,
    )
    if not kept_names:
        raise ValueError("global six-cluster control has no valid channels")
    roi = _as_bool_3d(roi_mask)
    indices = np.flatnonzero(roi.ravel())
    if indices.size < 96:
        raise ValueError("global six-cluster control requires at least 96 voxels")
    raw = filtered.reshape(len(kept_names), -1)[:, indices].T
    cluster_x, _, _, _, _, _ = _prepare_clustering_spaces(
        raw,
        clustering_space=config.clustering.clustering_space,
        pca_variance_percent=config.pca_variance_percent,
        seed=int(config.clustering.random_seed),
        max_fit_voxels=config.clustering.max_fit_voxels,
    )
    with warnings.catch_warnings():
        # Quantized maps may produce fewer populated clusters; record this in QC.
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        labels = _fit_cluster_labels(
            cluster_x,
            n_clusters=6,
            clusterer=config.clustering.clusterer,
            seed=int(config.clustering.random_seed),
            n_init=int(config.clustering.n_init),
            max_iter=int(config.clustering.max_iter),
            max_fit_voxels=config.clustering.max_fit_voxels,
        )
    scores, score_metadata = radiomic_deviation_scores(
        raw,
        kept_names,
        config.score,
    )
    cluster_scores = {
        cluster: float(np.mean(scores[labels == cluster]))
        for cluster in range(6)
        if np.any(labels == cluster)
    }
    ranked = sorted(cluster_scores, key=lambda item: (-cluster_scores[item], item))
    selected = ranked[:3]
    blocks: list[tuple[str, np.ndarray]] = []
    counts: dict[str, int] = {}
    for rank, cluster in enumerate(selected, start=1):
        flat = np.zeros(roi.size, dtype=np.uint8)
        flat[indices[labels == cluster]] = 1
        mask = flat.reshape(roi.shape)[np.newaxis]
        count = int(np.sum(mask))
        if count < 16:
            raise ValueError("global six-cluster control produced an undersized block")
        name = f"global_rank_{rank}"
        blocks.append((name, mask))
        counts[name] = count
    if len(blocks) != 3:
        raise ValueError("global six-cluster control did not produce three blocks")
    return blocks, {
        "definition": "top three of one global six-cluster solution ranked by sign-invariant RD score",
        "requested_cluster_count": 6,
        "observed_cluster_count": len(cluster_scores),
        "degenerate_solution": len(cluster_scores) != 6,
        "selected_cluster_ids": selected,
        "selected_voxel_counts": counts,
        "dropped_feature_names": dropped_names,
        "score_metadata": score_metadata,
    }


def construct_pes_control_masks(
    feature_maps: np.ndarray,
    roi_mask: np.ndarray,
    feature_names: list[str],
    tier_masks: dict[str, np.ndarray | None],
    final_pes_masks: dict[str, np.ndarray | None],
    config: PESConfig,
    requested: set[str] | tuple[str, ...] | list[str],
    case_key: str = "",
) -> tuple[dict[str, list[tuple[str, np.ndarray]]], dict[str, dict[str, Any]]]:
    """Construct the prespecified minimum hierarchy/morphology controls."""
    requested_set = set(requested) & set(CONTROL_REPRESENTATIONS)
    if not requested_set:
        return {}, {}

    controls: dict[str, list[tuple[str, np.ndarray]]] = {}
    metadata: dict[str, dict[str, Any]] = {}
    tiers = ("H", "M", "L")
    for tier in tiers:
        if tier_masks.get(tier) is None or final_pes_masks.get(tier) is None:
            raise ValueError(f"missing {tier} tier/PES mask required by controls")

    if "control_global_six_top3_structured" in requested_set:
        blocks, details = _global_six_top3(
            feature_maps,
            roi_mask,
            feature_names,
            config,
        )
        controls["control_global_six_top3_structured"] = blocks
        metadata["control_global_six_top3_structured"] = details

    if "control_parent_distance_core_structured" in requested_set:
        blocks = []
        matched_counts = {}
        for tier in tiers:
            parent = _as_bool_3d(tier_masks[tier])
            target_count = int(np.sum(_as_bool_3d(final_pes_masks[tier])))
            distance = ndimage.distance_transform_edt(parent)
            mask = _exact_largest_values(distance, parent, target_count)
            if int(np.sum(mask)) != target_count:
                raise AssertionError("distance-core volume matching failed")
            blocks.append((f"distance_core_{tier.lower()}", mask))
            matched_counts[tier] = target_count
        controls["control_parent_distance_core_structured"] = blocks
        metadata["control_parent_distance_core_structured"] = {
            "definition": "per-parent Euclidean distance core exactly volume-matched to final PES child",
            "matched_voxel_counts": matched_counts,
            "volume_tolerance_voxels": 0,
        }

    score_variants = {
        "ablation_unweighted_deviation_structured": "unweighted_deviation",
        "ablation_legacy_signed_mean_structured": "legacy_signed_mean",
    }
    for representation in sorted(requested_set & set(score_variants)):
        mode = score_variants[representation]
        variant_config = replace(
            config,
            score=replace(config.score, mode=mode),
        )
        blocks = []
        counts = {}
        for tier in tiers:
            result = construct_pes(
                feature_maps,
                _as_mask_4d(tier_masks[tier]),
                feature_names,
                variant_config,
            )
            if not result.completed or result.mask is None:
                raise ValueError(
                    f"score ablation {mode}/{tier} failed: {result.stopping_reason}"
                )
            blocks.append((f"{mode}_{tier.lower()}", _as_mask_4d(result.mask)))
            counts[tier] = int(np.sum(result.mask))
        controls[representation] = blocks
        metadata[representation] = {
            "definition": f"ordered H/M/L children selected with score mode {mode}",
            "score_mode": mode,
            "sign_invariant": mode != "legacy_signed_mean",
            "voxel_counts": counts,
        }

    need_raw_children = bool(
        requested_set
        & (
            {
                "ablation_pes_no_morphology_structured",
                "ablation_both_children_six_structured",
            }
            | set(RANDOM_CHILD_REPRESENTATIONS)
        )
    )
    raw_results = {}
    if need_raw_children:
        raw_config = replace(
            config,
            postprocessing=replace(config.postprocessing, enabled=False),
        )
        for tier in tiers:
            result = construct_pes(
                feature_maps,
                _as_mask_4d(tier_masks[tier]),
                feature_names,
                raw_config,
            )
            if not result.completed or result.mask is None or result.labels is None:
                raise ValueError(
                    f"no-morphology {tier} child failed: {result.stopping_reason}"
                )
            raw_results[tier] = result

    if "ablation_pes_no_morphology_structured" in requested_set:
        blocks = [
            (f"raw_child_{tier.lower()}", _as_mask_4d(raw_results[tier].mask))
            for tier in tiers
        ]
        controls["ablation_pes_no_morphology_structured"] = blocks
        metadata["ablation_pes_no_morphology_structured"] = {
            "definition": "ordered RD-H/M/L selected children before morphology",
            "voxel_counts": {
                tier: int(np.sum(raw_results[tier].mask)) for tier in tiers
            },
        }

    if "ablation_both_children_six_structured" in requested_set:
        blocks = []
        counts = {}
        for tier in tiers:
            result = raw_results[tier]
            parent = _as_bool_3d(tier_masks[tier])
            labels = _as_bool_3d(result.labels == int(result.summary["selected_cluster"]))
            selected = labels & parent
            other = parent & ~selected
            for child_name, child in (("selected", selected), ("other", other)):
                count = int(np.sum(child))
                if count < 16:
                    raise ValueError(
                        f"both-children {tier}/{child_name} has fewer than 16 voxels"
                    )
                block_name = f"{tier.lower()}_{child_name}"
                blocks.append((block_name, child[np.newaxis].astype(np.uint8)))
                counts[block_name] = count
        controls["ablation_both_children_six_structured"] = blocks
        metadata["ablation_both_children_six_structured"] = {
            "definition": "both raw binary children retained within each ordered parent tier",
            "voxel_counts": counts,
        }

    requested_random = sorted(
        requested_set & set(RANDOM_CHILD_REPRESENTATIONS)
    )
    random_child_masks = {}
    if requested_random:
        for tier in tiers:
            result = raw_results[tier]
            parent = _as_bool_3d(tier_masks[tier])
            selected = (
                _as_bool_3d(
                    result.labels == int(result.summary["selected_cluster"])
                )
                & parent
            )
            random_child_masks[tier] = {
                "selected": selected[np.newaxis].astype(np.uint8),
                "other": (parent & ~selected)[np.newaxis].astype(np.uint8),
            }

    for representation in requested_random:
        repeat = int(representation.removeprefix(RANDOM_CHILD_PREFIX))
        blocks = []
        choices = {}
        for tier in tiers:
            digest = hashlib.sha256(
                f"{case_key}|{config.clustering.random_seed}|{repeat}|{tier}".encode(
                    "utf-8"
                )
            ).digest()
            child_name = "selected" if digest[0] % 2 == 0 else "other"
            child = random_child_masks[tier][child_name]
            if int(np.sum(child)) < 16:
                raise ValueError(
                    f"random-child {repeat:02d}/{tier}/{child_name} has fewer than 16 voxels"
                )
            blocks.append((f"random_{tier.lower()}", child))
            choices[tier] = child_name
        controls[representation] = blocks
        metadata[representation] = {
            "definition": "outcome-free deterministic random child within each ordered parent",
            "repeat": repeat,
            "choices": choices,
            "case_key_sha256": hashlib.sha256(
                str(case_key).encode("utf-8")
            ).hexdigest(),
        }

    return controls, metadata
