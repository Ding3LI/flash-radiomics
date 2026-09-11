"""
Gray Level Distance Zone Matrix (GLDZM) features.

This implementation targets segment-based extraction and computes IBSI-style
distance-zone statistics from discretized voxels inside the ROI.
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np

from .base import RadiomicsFeaturesBase

try:
    import scipy.ndimage as ndi
except Exception:  # pragma: no cover - optional dependency guard
    ndi = None


class RadiomicsGLDZM(RadiomicsFeaturesBase):
    """GLDZM texture features extractor for segment-based workflows."""

    _FEATURE_ORDER = (
        "SmallDistanceEmphasis",
        "LargeDistanceEmphasis",
        "LowGrayLevelZoneEmphasis",
        "HighGrayLevelZoneEmphasis",
        "SmallDistanceLowGrayLevelEmphasis",
        "SmallDistanceHighGrayLevelEmphasis",
        "LargeDistanceLowGrayLevelEmphasis",
        "LargeDistanceHighGrayLevelEmphasis",
        "GrayLevelNonUniformity",
        "GrayLevelNonUniformityNormalized",
        "ZoneDistanceNonUniformity",
        "ZoneDistanceNonUniformityNormalized",
        "ZonePercentage",
        "GrayLevelVariance",
        "ZoneDistanceVariance",
        "ZoneDistanceEntropy",
    )

    @staticmethod
    def _compute_border_distance(mask: np.ndarray) -> np.ndarray:
        if ndi is None:
            raise ImportError(
                "scipy is required for GLDZM feature extraction (scipy.ndimage.distance_transform_cdt)."
            )
        if not np.any(mask):
            return np.zeros(mask.shape, dtype=np.int32)

        # Background padding assigns boundary voxels a distance of 1.
        padded_mask = np.pad(mask.astype(np.uint8), 1, mode="constant", constant_values=0)
        dist_padded = ndi.distance_transform_cdt(padded_mask, metric="taxicab")
        slices = tuple(slice(1, dim + 1) for dim in mask.shape)
        return np.asarray(dist_padded[slices], dtype=np.int32)

    @staticmethod
    def _label_gray_zones(discretized: np.ndarray, roi_mask: np.ndarray) -> np.ndarray:
        if ndi is None:
            raise ImportError("scipy is required for GLDZM feature extraction.")

        zone_labels = np.zeros(discretized.shape, dtype=np.int64)
        if not np.any(roi_mask):
            return zone_labels

        structure = ndi.generate_binary_structure(discretized.ndim, discretized.ndim)
        gray_levels = np.unique(discretized[roi_mask])
        gray_levels = gray_levels[gray_levels > 0]

        next_label = 1
        for gray_value in gray_levels:
            component_mask = roi_mask & (discretized == gray_value)
            component_labels, n_components = ndi.label(component_mask, structure=structure)
            if n_components <= 0:
                continue
            nonzero = component_labels > 0
            zone_labels[nonzero] = component_labels[nonzero] + (next_label - 1)
            next_label += int(n_components)

        return zone_labels

    @classmethod
    def _build_dzm_counts_3d(
        cls,
        discretized: np.ndarray,
        roi_mask: np.ndarray,
        ng: int,
        distance_mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, int, int]:
        n_voxels = int(np.count_nonzero(roi_mask))
        if n_voxels <= 0 or ng <= 0:
            return np.zeros((max(int(ng), 0), 0), dtype=np.float64), 0, n_voxels

        if distance_mask is None:
            distance_mask = roi_mask
        border_distance = cls._compute_border_distance(np.asarray(distance_mask) != 0)
        zone_labels = cls._label_gray_zones(discretized, roi_mask)
        valid = roi_mask & (zone_labels > 0)
        if not np.any(valid):
            return np.zeros((int(ng), 0), dtype=np.float64), 0, n_voxels

        zone_ids = zone_labels[valid].astype(np.int64, copy=False)
        gray_values = discretized[valid].astype(np.int64, copy=False)
        distances = border_distance[valid].astype(np.int64, copy=False)

        order = np.argsort(zone_ids, kind="mergesort")
        zone_ids_sorted = zone_ids[order]
        gray_values_sorted = gray_values[order]
        distances_sorted = distances[order]

        starts = np.empty(zone_ids_sorted.shape[0], dtype=bool)
        starts[0] = True
        starts[1:] = zone_ids_sorted[1:] != zone_ids_sorted[:-1]
        group_starts = np.flatnonzero(starts)

        zone_gray = gray_values_sorted[group_starts]
        zone_min_distance = np.minimum.reduceat(distances_sorted, group_starts)
        n_zones = int(group_starts.shape[0])
        if n_zones <= 0:
            return np.zeros((int(ng), 0), dtype=np.float64), 0, n_voxels

        max_distance = int(np.max(zone_min_distance))
        linear = (zone_gray - 1) * max_distance + (zone_min_distance - 1)
        counts = np.bincount(
            linear,
            minlength=int(ng) * max_distance,
        ).reshape(int(ng), max_distance)

        return counts.astype(np.float64, copy=False), n_zones, n_voxels

    @classmethod
    def _build_dzm_counts(
        cls,
        discretized: np.ndarray,
        roi_mask: np.ndarray,
        ng: int,
        distance_mask: np.ndarray | None = None,
        force2d: bool = False,
    ) -> tuple[np.ndarray, int, int]:
        if not force2d or discretized.ndim != 3:
            return cls._build_dzm_counts_3d(
                discretized=discretized,
                roi_mask=roi_mask,
                ng=ng,
                distance_mask=distance_mask,
            )

        counts_total = np.zeros((int(ng), 0), dtype=np.float64)
        n_zones_total = 0
        n_voxels_total = int(np.count_nonzero(roi_mask))
        if n_voxels_total <= 0 or ng <= 0:
            return counts_total, 0, n_voxels_total

        if distance_mask is None:
            distance_mask = roi_mask
        for slice_idx in range(discretized.shape[0]):
            slice_roi = roi_mask[slice_idx]
            if not np.any(slice_roi):
                continue
            slice_counts, slice_zones, _ = cls._build_dzm_counts_3d(
                discretized=discretized[slice_idx],
                roi_mask=slice_roi,
                ng=ng,
                distance_mask=np.asarray(distance_mask)[slice_idx],
            )
            if slice_counts.shape[1] > counts_total.shape[1]:
                pad = slice_counts.shape[1] - counts_total.shape[1]
                counts_total = np.pad(counts_total, ((0, 0), (0, pad)))
            if slice_counts.shape[1] < counts_total.shape[1]:
                pad = counts_total.shape[1] - slice_counts.shape[1]
                slice_counts = np.pad(slice_counts, ((0, 0), (0, pad)))
            counts_total += slice_counts
            n_zones_total += int(slice_zones)

        return counts_total, n_zones_total, n_voxels_total

    @classmethod
    def _compute_features_from_counts(
        cls,
        counts: np.ndarray,
        n_zones: int,
        n_voxels: int,
    ) -> OrderedDict:
        features = OrderedDict((name, np.nan) for name in cls._FEATURE_ORDER)
        if n_zones <= 0 or counts.size == 0:
            return features

        counts = np.asarray(counts, dtype=np.float64)
        n_s = float(n_zones)
        i = np.arange(1, counts.shape[0] + 1, dtype=np.float64)
        j = np.arange(1, counts.shape[1] + 1, dtype=np.float64)

        di = np.sum(counts, axis=1)
        dj = np.sum(counts, axis=0)

        i_col = i[:, None]
        j_row = j[None, :]

        features["SmallDistanceEmphasis"] = float(np.sum(dj / (j**2.0)) / n_s)
        features["LargeDistanceEmphasis"] = float(np.sum(dj * (j**2.0)) / n_s)
        features["LowGrayLevelZoneEmphasis"] = float(np.sum(di / (i**2.0)) / n_s)
        features["HighGrayLevelZoneEmphasis"] = float(np.sum(di * (i**2.0)) / n_s)
        features["SmallDistanceLowGrayLevelEmphasis"] = float(
            np.sum(counts / ((i_col * j_row) ** 2.0)) / n_s
        )
        features["SmallDistanceHighGrayLevelEmphasis"] = float(
            np.sum(counts * (i_col**2.0) / (j_row**2.0)) / n_s
        )
        features["LargeDistanceLowGrayLevelEmphasis"] = float(
            np.sum(counts * (j_row**2.0) / (i_col**2.0)) / n_s
        )
        features["LargeDistanceHighGrayLevelEmphasis"] = float(
            np.sum(counts * (i_col**2.0) * (j_row**2.0)) / n_s
        )
        features["GrayLevelNonUniformity"] = float(np.sum(di**2.0) / n_s)
        features["GrayLevelNonUniformityNormalized"] = float(np.sum(di**2.0) / (n_s**2.0))
        features["ZoneDistanceNonUniformity"] = float(np.sum(dj**2.0) / n_s)
        features["ZoneDistanceNonUniformityNormalized"] = float(np.sum(dj**2.0) / (n_s**2.0))
        features["ZonePercentage"] = float(n_s / float(max(n_voxels, 1)))

        mu_i = float(np.sum(counts * i_col) / n_s)
        features["GrayLevelVariance"] = float(
            np.sum(((i_col - mu_i) ** 2.0) * counts) / n_s
        )

        mu_j = float(np.sum(counts * j_row) / n_s)
        features["ZoneDistanceVariance"] = float(
            np.sum(((j_row - mu_j) ** 2.0) * counts) / n_s
        )

        p = counts / n_s
        nonzero = p > 0.0
        features["ZoneDistanceEntropy"] = float(-np.sum(p[nonzero] * np.log2(p[nonzero])))

        return features

    def execute(self):
        roi_mask = np.asarray(self.inputMask) != 0
        discretized, ng = self._get_binimage_discretized()
        if discretized is None:
            self.featureValues = self._filter_features(
                OrderedDict((name, np.nan) for name in self._FEATURE_ORDER)
            )
            return self.featureValues

        distance_mask = self.kwargs.get("_gldzm_distance_mask_array")
        if distance_mask is not None:
            distance_mask = np.asarray(distance_mask) != 0

        counts, n_zones, n_voxels = self._build_dzm_counts(
            np.asarray(discretized, dtype=np.int32),
            roi_mask,
            int(ng),
            distance_mask=distance_mask,
            force2d=bool(self.kwargs.get("force2D", False)),
        )
        features = self._compute_features_from_counts(
            counts=counts,
            n_zones=n_zones,
            n_voxels=n_voxels,
        )
        self.featureValues = self._filter_features(features)
        return self.featureValues
