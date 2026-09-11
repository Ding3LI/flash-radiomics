"""
Gray Level Run Length Matrix (GLRLM) features

Supports both 2D (4 directions) and 3D (13 directions) images with automatic adaptation.
"""

# PYRADIOMICS: Feature names and default availability are aligned to PyRadiomics.

import ctypes
from collections import OrderedDict
import numpy as np
from .base import RadiomicsFeaturesBase
from . import _bindings


class RadiomicsGLRLM(RadiomicsFeaturesBase):
    _DIRECTIONS_2D = (
        (0, 1),
        (1, 0),
        (1, 1),
        (1, -1),
    )

    """
    GLRLM texture features extractor
    
    Computes run-length texture features including:
    - Short Run Emphasis, Long Run Emphasis
    - Gray Level Non-Uniformity
    - Run Length Non-Uniformity
    - Run Percentage
    - Low/High Gray Level Run Emphasis
    - Short/Long Run Low/High Gray Level Emphasis
    - Gray Level Variance, Run Length Variance
    
    Automatically uses:
    - 4 directions for 2D images
    - 13 directions for 3D images
    """
    
    def __init__(self, inputImage, inputMask, **kwargs):
        """
        Initialize GLRLM feature extractor
        
        Args:
            inputImage: numpy array of image data (2D or 3D)
            inputMask: numpy array of mask data (2D or 3D)
            **kwargs: Additional parameters (backend, num_threads, spacing, origin)
        """
        super().__init__(inputImage, inputMask, **kwargs)

    @staticmethod
    def _direction_matrix_2d_merged(discretized, roi_mask, ng, direction):
        dz = 0
        dy, dx = direction
        max_run = max(discretized.shape[1], discretized.shape[2])
        matrix = np.zeros((int(ng), int(max_run)), dtype=np.float64)
        n_voxels = int(np.count_nonzero(roi_mask))
        z_size, y_size, x_size = discretized.shape

        for z in range(z_size):
            for y in range(y_size):
                for x in range(x_size):
                    if not roi_mask[z, y, x]:
                        continue
                    gray = int(discretized[z, y, x])
                    if gray <= 0 or gray > ng:
                        continue
                    py = y - dy
                    px = x - dx
                    if 0 <= py < y_size and 0 <= px < x_size and roi_mask[z, py, px] and int(discretized[z, py, px]) == gray:
                        continue
                    run_len = 1
                    ny = y + dy
                    nx = x + dx
                    while 0 <= ny < y_size and 0 <= nx < x_size and roi_mask[z, ny, nx] and int(discretized[z, ny, nx]) == gray:
                        run_len += 1
                        ny += dy
                        nx += dx
                    matrix[gray - 1, run_len - 1] += 1.0
        return matrix, n_voxels

    @staticmethod
    def _features_from_matrix(matrix, n_voxels):
        features = OrderedDict()
        n_s = float(np.sum(matrix))
        if n_s <= 0.0:
            return features
        i = np.arange(1, matrix.shape[0] + 1, dtype=np.float64)
        j = np.arange(1, matrix.shape[1] + 1, dtype=np.float64)
        ri = np.sum(matrix, axis=1)
        rj = np.sum(matrix, axis=0)
        i_col = i[:, None]
        j_row = j[None, :]
        features["ShortRunEmphasis"] = float(np.sum(rj / (j**2.0)) / n_s)
        features["LongRunEmphasis"] = float(np.sum(rj * (j**2.0)) / n_s)
        features["LowGrayLevelRunEmphasis"] = float(np.sum(ri / (i**2.0)) / n_s)
        features["HighGrayLevelRunEmphasis"] = float(np.sum(ri * (i**2.0)) / n_s)
        features["ShortRunLowGrayLevelEmphasis"] = float(np.sum(matrix / ((i_col * j_row) ** 2.0)) / n_s)
        features["ShortRunHighGrayLevelEmphasis"] = float(np.sum(matrix * (i_col**2.0) / (j_row**2.0)) / n_s)
        features["LongRunLowGrayLevelEmphasis"] = float(np.sum(matrix * (j_row**2.0) / (i_col**2.0)) / n_s)
        features["LongRunHighGrayLevelEmphasis"] = float(np.sum(matrix * (i_col**2.0) * (j_row**2.0)) / n_s)
        features["GrayLevelNonUniformity"] = float(np.sum(ri**2.0) / n_s)
        features["GrayLevelNonUniformityNormalized"] = float(np.sum(ri**2.0) / (n_s**2.0))
        features["RunLengthNonUniformity"] = float(np.sum(rj**2.0) / n_s)
        features["RunLengthNonUniformityNormalized"] = float(np.sum(rj**2.0) / (n_s**2.0))
        features["RunPercentage"] = float(n_s / float(max(n_voxels, 1)))
        mu_i = float(np.sum(matrix * i_col) / n_s)
        features["GrayLevelVariance"] = float(np.sum(((i_col - mu_i) ** 2.0) * matrix) / n_s)
        mu_j = float(np.sum(matrix * j_row) / n_s)
        features["RunLengthVariance"] = float(np.sum(((j_row - mu_j) ** 2.0) * matrix) / n_s)
        p = matrix / n_s
        features["RunEntropy"] = float(-np.sum(p[p > 0.0] * np.log2(p[p > 0.0])))
        return features

    @classmethod
    def _compute_2d_direction_merged_features(cls, discretized, roi_mask, ng):
        values = []
        for direction in cls._DIRECTIONS_2D:
            matrix, n_voxels = cls._direction_matrix_2d_merged(discretized, roi_mask, int(ng), direction)
            if np.sum(matrix) > 0.0:
                values.append(cls._features_from_matrix(matrix, n_voxels))
        if not values:
            return OrderedDict()
        out = OrderedDict()
        for key in values[0]:
            out[key] = float(np.nanmean([feature_values[key] for feature_values in values]))
        if "RunLengthVariance" in out:
            remapped = OrderedDict()
            for key, value in out.items():
                remapped["RunVariance" if key == "RunLengthVariance" else key] = value
            out = remapped
        return out
    
    def execute(self):
        """
        Execute GLRLM feature extraction
        
        Returns:
            OrderedDict of feature values
        """
        img_c = self._numpy_to_image(self.inputImage)
        mask_c = self._numpy_to_mask(self.inputMask)
        
        resolved_backend = self._resolved_backend_name()

        discretized, ng = (None, 0)
        # Use the CPU-discretized path for segment GLRLM.
        use_cpu_discretized = self._should_use_cpu_discretized_path() or resolved_backend == "cuda"
        if use_cpu_discretized:
            discretized, ng = self._get_binimage_discretized()

        added_run_variance_alias = False
        if (
            self._enabled_features is not None
            and "RunVariance" in self._enabled_features
            and "RunLengthVariance" not in self._enabled_features
        ):
            self._enabled_features.add("RunLengthVariance")
            added_run_variance_alias = True

        if (
            bool(self.kwargs.get("force2D", False))
            and discretized is not None
            and ng > 0
            and np.asarray(discretized).ndim == 3
        ):
            features = self._compute_2d_direction_merged_features(
                np.asarray(discretized, dtype=np.int32),
                np.asarray(self.inputMask) != 0,
                int(ng),
            )
            self.featureValues = self._filter_features(features)
            return self.featureValues

        if discretized is not None and ng > 0 and use_cpu_discretized:
            feature_lib = self._get_cpu_feature_lib()
            discretized_array = discretized.reshape(-1)
            discretized_c = discretized_array.ctypes.data_as(
                ctypes.POINTER(ctypes.c_int)
            )
            result_ptr = feature_lib.flash_radiomics_glrlm_cpu_discretized(
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
            result_ptr = feature_lib.flash_radiomics_glrlm(
                img_c,
                mask_c,
                self.backend_enum,
                self.num_threads,
                bin_width,
                bin_count,
            )
        
        features = self._result_to_dict(
            result_ptr,
            "GLRLM",
            strip_prefix="glrlm_",
            feature_lib=feature_lib,
        )

        if added_run_variance_alias:
            self._enabled_features.discard("RunLengthVariance")

        if "RunLengthVariance" in features and "RunVariance" not in features:
            remapped = OrderedDict()
            for key, value in features.items():
                if key == "RunLengthVariance":
                    remapped["RunVariance"] = value
                else:
                    remapped[key] = value
            features = remapped

        self.featureValues = features
        
        
        return self.featureValues
