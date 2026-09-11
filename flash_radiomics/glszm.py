"""
Gray Level Size Zone Matrix (GLSZM) features

Supports both 2D (8-connectivity) and 3D (26-connectivity) images with automatic adaptation.
"""

# PYRADIOMICS: Feature names and default availability are aligned to PyRadiomics.

import ctypes
from collections import OrderedDict
import numpy as np
from .base import RadiomicsFeaturesBase
from . import _bindings

try:
    import scipy.ndimage as ndi
except Exception:  # pragma: no cover - optional dependency guard
    ndi = None


class RadiomicsGLSZM(RadiomicsFeaturesBase):
    """
    GLSZM texture features extractor
    
    Computes size zone texture features including:
    - Small/Large Area Emphasis
    - Gray Level Non-Uniformity
    - Size Zone Non-Uniformity
    - Zone Percentage
    - Low/High Gray Level Zone Emphasis
    - Small/Large Area Low/High Gray Level Emphasis
    - Gray Level Variance, Size Zone Variance
    
    Automatically uses:
    - 8-connectivity for 2D images
    - 26-connectivity for 3D images
    """
    
    def __init__(self, inputImage, inputMask, **kwargs):
        """
        Initialize GLSZM feature extractor
        
        Args:
            inputImage: numpy array of image data (2D or 3D)
            inputMask: numpy array of mask data (2D or 3D)
            **kwargs: Additional parameters (backend, num_threads, spacing, origin)
        """
        super().__init__(inputImage, inputMask, **kwargs)

    _FEATURE_ORDER = (
        "SmallAreaEmphasis",
        "LargeAreaEmphasis",
        "LowGrayLevelZoneEmphasis",
        "HighGrayLevelZoneEmphasis",
        "SmallAreaLowGrayLevelEmphasis",
        "SmallAreaHighGrayLevelEmphasis",
        "LargeAreaLowGrayLevelEmphasis",
        "LargeAreaHighGrayLevelEmphasis",
        "GrayLevelNonUniformity",
        "GrayLevelNonUniformityNormalized",
        "SizeZoneNonUniformity",
        "SizeZoneNonUniformityNormalized",
        "ZonePercentage",
        "GrayLevelVariance",
        "SizeZoneVariance",
        "ZoneEntropy",
    )

    @staticmethod
    def _label_gray_zones(discretized: np.ndarray, roi_mask: np.ndarray) -> np.ndarray:
        if ndi is None:
            raise ImportError("scipy is required for 2.5D GLSZM feature extraction.")

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
    def _build_szm_counts_2d_merged(
        cls,
        discretized: np.ndarray,
        roi_mask: np.ndarray,
        ng: int,
    ) -> tuple[np.ndarray, int, int]:
        n_voxels = int(np.count_nonzero(roi_mask))
        counts_total = np.zeros((int(ng), 0), dtype=np.float64)
        n_zones_total = 0
        if n_voxels <= 0 or ng <= 0:
            return counts_total, 0, n_voxels

        for slice_idx in range(discretized.shape[0]):
            slice_roi = roi_mask[slice_idx]
            if not np.any(slice_roi):
                continue
            zone_labels = cls._label_gray_zones(discretized[slice_idx], slice_roi)
            valid = slice_roi & (zone_labels > 0)
            if not np.any(valid):
                continue

            zone_ids = zone_labels[valid].astype(np.int64, copy=False)
            gray_values = discretized[slice_idx][valid].astype(np.int64, copy=False)
            order = np.argsort(zone_ids, kind="mergesort")
            zone_ids_sorted = zone_ids[order]
            gray_values_sorted = gray_values[order]

            starts = np.empty(zone_ids_sorted.shape[0], dtype=bool)
            starts[0] = True
            starts[1:] = zone_ids_sorted[1:] != zone_ids_sorted[:-1]
            group_starts = np.flatnonzero(starts)
            zone_gray = gray_values_sorted[group_starts]
            zone_size = np.diff(np.r_[group_starts, zone_ids_sorted.shape[0]])

            max_size = int(np.max(zone_size))
            linear = (zone_gray - 1) * max_size + (zone_size - 1)
            slice_counts = np.bincount(
                linear,
                minlength=int(ng) * max_size,
            ).reshape(int(ng), max_size).astype(np.float64, copy=False)

            if slice_counts.shape[1] > counts_total.shape[1]:
                counts_total = np.pad(
                    counts_total,
                    ((0, 0), (0, slice_counts.shape[1] - counts_total.shape[1])),
                )
            if slice_counts.shape[1] < counts_total.shape[1]:
                slice_counts = np.pad(
                    slice_counts,
                    ((0, 0), (0, counts_total.shape[1] - slice_counts.shape[1])),
                )
            counts_total += slice_counts
            n_zones_total += int(group_starts.shape[0])

        return counts_total, n_zones_total, n_voxels

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

        n_s = float(n_zones)
        i = np.arange(1, counts.shape[0] + 1, dtype=np.float64)
        j = np.arange(1, counts.shape[1] + 1, dtype=np.float64)
        di = np.sum(counts, axis=1)
        dj = np.sum(counts, axis=0)
        i_col = i[:, None]
        j_row = j[None, :]

        features["SmallAreaEmphasis"] = float(np.sum(dj / (j**2.0)) / n_s)
        features["LargeAreaEmphasis"] = float(np.sum(dj * (j**2.0)) / n_s)
        features["LowGrayLevelZoneEmphasis"] = float(np.sum(di / (i**2.0)) / n_s)
        features["HighGrayLevelZoneEmphasis"] = float(np.sum(di * (i**2.0)) / n_s)
        features["SmallAreaLowGrayLevelEmphasis"] = float(
            np.sum(counts / ((i_col * j_row) ** 2.0)) / n_s
        )
        features["SmallAreaHighGrayLevelEmphasis"] = float(
            np.sum(counts * (i_col**2.0) / (j_row**2.0)) / n_s
        )
        features["LargeAreaLowGrayLevelEmphasis"] = float(
            np.sum(counts * (j_row**2.0) / (i_col**2.0)) / n_s
        )
        features["LargeAreaHighGrayLevelEmphasis"] = float(
            np.sum(counts * (i_col**2.0) * (j_row**2.0)) / n_s
        )
        features["GrayLevelNonUniformity"] = float(np.sum(di**2.0) / n_s)
        features["GrayLevelNonUniformityNormalized"] = float(np.sum(di**2.0) / (n_s**2.0))
        features["SizeZoneNonUniformity"] = float(np.sum(dj**2.0) / n_s)
        features["SizeZoneNonUniformityNormalized"] = float(np.sum(dj**2.0) / (n_s**2.0))
        features["ZonePercentage"] = float(n_s / float(max(n_voxels, 1)))

        mu_i = float(np.sum(counts * i_col) / n_s)
        features["GrayLevelVariance"] = float(
            np.sum(((i_col - mu_i) ** 2.0) * counts) / n_s
        )
        mu_j = float(np.sum(counts * j_row) / n_s)
        features["SizeZoneVariance"] = float(
            np.sum(((j_row - mu_j) ** 2.0) * counts) / n_s
        )
        p = counts / n_s
        nonzero = p > 0.0
        features["ZoneEntropy"] = float(-np.sum(p[nonzero] * np.log2(p[nonzero])))
        return features
    
    def execute(self):
        """
        Execute GLSZM feature extraction
        
        Returns:
            OrderedDict of feature values
        """
        img_c = self._numpy_to_image(self.inputImage)
        mask_c = self._numpy_to_mask(self.inputMask)
        
        resolved_backend = self._resolved_backend_name()

        discretized, ng = (None, 0)
        # Use the CPU-discretized path for segment GLSZM.
        use_cpu_discretized = self._should_use_cpu_discretized_path() or resolved_backend == "cuda"
        if use_cpu_discretized:
            discretized, ng = self._get_binimage_discretized()
        added_size_zone_variance_alias = False
        if (
            self._enabled_features is not None
            and "ZoneVariance" in self._enabled_features
            and "SizeZoneVariance" not in self._enabled_features
        ):
            self._enabled_features.add("SizeZoneVariance")
            added_size_zone_variance_alias = True

        if (
            bool(self.kwargs.get("force2D", False))
            and discretized is not None
            and ng > 0
            and np.asarray(discretized).ndim == 3
        ):
            counts, n_zones, n_voxels = self._build_szm_counts_2d_merged(
                np.asarray(discretized, dtype=np.int32),
                np.asarray(self.inputMask) != 0,
                int(ng),
            )
            features = self._compute_features_from_counts(counts, n_zones, n_voxels)
            if added_size_zone_variance_alias:
                self._enabled_features.discard("SizeZoneVariance")
            if "SizeZoneVariance" in features and "ZoneVariance" not in features:
                remapped = OrderedDict()
                for key, value in features.items():
                    remapped["ZoneVariance" if key == "SizeZoneVariance" else key] = value
                features = remapped
            self.featureValues = self._filter_features(features)
            return self.featureValues

        if discretized is not None and ng > 0 and use_cpu_discretized:
            feature_lib = self._get_cpu_feature_lib()
            discretized_array = discretized.reshape(-1)
            discretized_c = discretized_array.ctypes.data_as(
                ctypes.POINTER(ctypes.c_int)
            )
            result_ptr = feature_lib.flash_radiomics_glszm_cpu_discretized(
                img_c,
                mask_c,
                self.num_threads,
                discretized_c,
                ng,
            )
        else:
            feature_lib = self._feature_lib
            bin_width = self.kwargs.get("binWidth", 25)
            bin_count = self.kwargs.get("binCount", 0) or 0
            result_ptr = feature_lib.flash_radiomics_glszm(
                img_c,
                mask_c,
                self.backend_enum,
                self.num_threads,
                bin_width,
                bin_count,
            )

        features = self._result_to_dict(
            result_ptr,
            "GLSZM",
            strip_prefix="glszm_",
            feature_lib=feature_lib,
        )

        if added_size_zone_variance_alias:
            self._enabled_features.discard("SizeZoneVariance")

        normalize_zone_variance_name = (
            self._enabled_features is None or "ZoneVariance" in self._enabled_features
        )
        if (
            normalize_zone_variance_name
            and "SizeZoneVariance" in features
            and "ZoneVariance" not in features
        ):
            remapped = OrderedDict()
            for key, value in features.items():
                if key == "SizeZoneVariance":
                    remapped["ZoneVariance"] = value
                else:
                    remapped[key] = value
            features = remapped

        self.featureValues = features
        
        
        return self.featureValues
