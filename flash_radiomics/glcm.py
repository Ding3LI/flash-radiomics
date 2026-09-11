"""
Gray Level Co-occurrence Matrix (GLCM) features

Supports both 2D (4 directions) and 3D (13 directions) images with automatic adaptation.
"""

# PYRADIOMICS: Feature names and default availability are aligned to PyRadiomics.

import ctypes
from collections import OrderedDict
import numpy as np
from .base import RadiomicsFeaturesBase
from . import _bindings


class RadiomicsGLCM(RadiomicsFeaturesBase):
    """
    GLCM texture features extractor
    
    Computes texture features from gray level co-occurrence matrices including:
    - Autocorrelation
    - Contrast, Dissimilarity
    - Energy, Entropy
    - Homogeneity
    - Correlation
    - Sum Average, Sum Entropy
    - Difference Average, Difference Entropy
    - Information Measure of Correlation
    - Cluster Tendency, Cluster Shade, Cluster Prominence
    - Maximum Probability
    
    Automatically uses:
    - 4 directions for 2D images (0°, 45°, 90°, 135°)
    - 13 directions for 3D images
    """

    _DIRECTIONS_2D = (
        (1, 0),   # 0°
        (1, 1),   # 45°
        (0, 1),   # 90°
        (-1, 1),  # 135°
    )

    _DIRECTIONS_3D = (
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
        (1, 1, 0),
        (1, -1, 0),
        (1, 0, 1),
        (1, 0, -1),
        (0, 1, 1),
        (0, 1, -1),
        (1, 1, 1),
        (1, 1, -1),
        (1, -1, 1),
        (1, -1, -1),
    )
    
    def __init__(self, inputImage, inputMask, **kwargs):
        """
        Initialize GLCM feature extractor
        
        Args:
            inputImage: numpy array of image data (2D or 3D)
            inputMask: numpy array of mask data (2D or 3D)
            **kwargs: Additional parameters including:
                - distances: list of distances (default: [1])
                - backend: 'cpu', 'mps', 'cuda', or 'auto'
                - num_threads: number of threads for CPU
        """
        super().__init__(inputImage, inputMask, **kwargs)
        self.distances = kwargs.get("distances", [1])

    @staticmethod
    def _axis_slices(length, shift):
        if abs(shift) >= length:
            return None, None
        if shift >= 0:
            return slice(0, length - shift), slice(shift, length)
        return slice(-shift, length), slice(0, length + shift)

    @staticmethod
    def _compute_mcc_from_matrix(glcm_matrix):
        if glcm_matrix.shape[0] < 2:
            return 1.0

        px = np.sum(glcm_matrix, axis=1)
        py = np.sum(glcm_matrix, axis=0)
        denom = px[:, None] * py[None, :] + 2.2e-16
        q_matrix = (glcm_matrix / denom) @ glcm_matrix.T
        eigvals = np.linalg.eigvals(q_matrix)
        eigvals.sort()
        if eigvals.shape[0] < 2:
            return 1.0
        return float(np.sqrt(eigvals[-2]).real)

    @classmethod
    def _compute_mcc(cls, discretized, mask_array, ng, distances):
        discretized = np.asarray(discretized, dtype=np.int32)
        mask = np.asarray(mask_array) != 0

        if discretized.ndim == 2:
            directions = cls._DIRECTIONS_2D
        elif discretized.ndim == 3:
            directions = cls._DIRECTIONS_3D
        else:
            raise ValueError(f"Unsupported ndim for GLCM MCC: {discretized.ndim}")

        mcc_values = []
        for distance in distances:
            if distance <= 0:
                continue
            for direction in directions:
                if discretized.ndim == 2:
                    dx = direction[0] * distance
                    dy = direction[1] * distance

                    sx, nx = cls._axis_slices(discretized.shape[1], dx)
                    sy, ny = cls._axis_slices(discretized.shape[0], dy)
                    if sx is None or sy is None:
                        continue

                    src_vals = discretized[sy, sx]
                    nbr_vals = discretized[ny, nx]
                    src_mask = mask[sy, sx]
                    nbr_mask = mask[ny, nx]
                else:
                    dx = direction[0] * distance
                    dy = direction[1] * distance
                    dz = direction[2] * distance

                    sx, nx = cls._axis_slices(discretized.shape[2], dx)
                    sy, ny = cls._axis_slices(discretized.shape[1], dy)
                    sz, nz = cls._axis_slices(discretized.shape[0], dz)
                    if sx is None or sy is None or sz is None:
                        continue

                    src_vals = discretized[sz, sy, sx]
                    nbr_vals = discretized[nz, ny, nx]
                    src_mask = mask[sz, sy, sx]
                    nbr_mask = mask[nz, ny, nx]

                valid = (
                    src_mask
                    & nbr_mask
                    & (src_vals > 0)
                    & (nbr_vals > 0)
                    & (src_vals <= ng)
                    & (nbr_vals <= ng)
                )
                if not np.any(valid):
                    continue

                i_idx = (src_vals[valid] - 1).astype(np.int32)
                j_idx = (nbr_vals[valid] - 1).astype(np.int32)
                matrix = np.zeros((ng, ng), dtype=np.float64)
                np.add.at(matrix, (i_idx, j_idx), 1.0)
                np.add.at(matrix, (j_idx, i_idx), 1.0)

                matrix_sum = float(np.sum(matrix))
                if matrix_sum <= 0.0:
                    continue
                matrix /= matrix_sum
                mcc_values.append(cls._compute_mcc_from_matrix(matrix))

        if not mcc_values:
            return 1.0
        return float(np.nanmean(np.asarray(mcc_values, dtype=np.float64)).real)

    @classmethod
    def _iter_direction_matrices(cls, discretized, mask_array, ng, distances, force2d=False):
        discretized = np.asarray(discretized, dtype=np.int32)
        mask = np.asarray(mask_array) != 0

        if discretized.ndim == 2:
            directions = cls._DIRECTIONS_2D
        elif discretized.ndim == 3 and force2d:
            directions = (
                (1, 0, 0),
                (0, 1, 0),
                (1, 1, 0),
                (1, -1, 0),
            )
        elif discretized.ndim == 3:
            directions = cls._DIRECTIONS_3D
        else:
            raise ValueError(f"Unsupported ndim for GLCM: {discretized.ndim}")

        for distance in distances:
            if distance <= 0:
                continue
            for direction in directions:
                if discretized.ndim == 2:
                    dx = direction[0] * distance
                    dy = direction[1] * distance

                    sx, nx = cls._axis_slices(discretized.shape[1], dx)
                    sy, ny = cls._axis_slices(discretized.shape[0], dy)
                    if sx is None or sy is None:
                        continue

                    src_vals = discretized[sy, sx]
                    nbr_vals = discretized[ny, nx]
                    src_mask = mask[sy, sx]
                    nbr_mask = mask[ny, nx]
                else:
                    dx = direction[0] * distance
                    dy = direction[1] * distance
                    dz = direction[2] * distance

                    sx, nx = cls._axis_slices(discretized.shape[2], dx)
                    sy, ny = cls._axis_slices(discretized.shape[1], dy)
                    sz, nz = cls._axis_slices(discretized.shape[0], dz)
                    if sx is None or sy is None or sz is None:
                        continue

                    src_vals = discretized[sz, sy, sx]
                    nbr_vals = discretized[nz, ny, nx]
                    src_mask = mask[sz, sy, sx]
                    nbr_mask = mask[nz, ny, nx]

                valid = (
                    src_mask
                    & nbr_mask
                    & (src_vals > 0)
                    & (nbr_vals > 0)
                    & (src_vals <= ng)
                    & (nbr_vals <= ng)
                )
                if not np.any(valid):
                    continue

                i_idx = (src_vals[valid] - 1).astype(np.int32)
                j_idx = (nbr_vals[valid] - 1).astype(np.int32)
                matrix = np.zeros((ng, ng), dtype=np.float64)
                np.add.at(matrix, (i_idx, j_idx), 1.0)
                np.add.at(matrix, (j_idx, i_idx), 1.0)

                matrix_sum = float(np.sum(matrix))
                if matrix_sum <= 0.0:
                    continue
                yield matrix / matrix_sum

    @classmethod
    def _compute_ibsi_redundant_features(cls, discretized, mask_array, ng, distances, force2d=False):
        values = {
            "SumVariance": [],
            "Dissimilarity": [],
        }
        for matrix in cls._iter_direction_matrices(
            discretized,
            mask_array,
            int(ng),
            distances,
            force2d=force2d,
        ):
            i = np.arange(1, matrix.shape[0] + 1, dtype=np.float64)
            j = np.arange(1, matrix.shape[1] + 1, dtype=np.float64)
            i_col = i[:, None]
            j_row = j[None, :]

            values["Dissimilarity"].append(float(np.sum(np.abs(i_col - j_row) * matrix)))

            sum_indices = np.add.outer(i, j).reshape(-1)
            probabilities = matrix.reshape(-1)
            mu_sum = float(np.sum(sum_indices * probabilities))
            values["SumVariance"].append(
                float(np.sum(((sum_indices - mu_sum) ** 2.0) * probabilities))
            )

        return OrderedDict(
            (name, float(np.nanmean(feature_values)) if feature_values else np.nan)
            for name, feature_values in values.items()
        )

    @classmethod
    def _compute_features_from_matrix(cls, matrix):
        eps = np.finfo(np.float64).eps
        p = np.asarray(matrix, dtype=np.float64)
        ng = p.shape[0]
        i = np.arange(1, ng + 1, dtype=np.float64)
        j = np.arange(1, ng + 1, dtype=np.float64)
        i_col = i[:, None]
        j_row = j[None, :]
        px = np.sum(p, axis=1)
        py = np.sum(p, axis=0)
        ux = float(np.sum(i * px))
        uy = float(np.sum(j * py))
        sx = float(np.sqrt(np.sum(((i - ux) ** 2.0) * px)))
        sy = float(np.sqrt(np.sum(((j - uy) ** 2.0) * py)))
        diff = np.abs(i_col - j_row)
        summ = i_col + j_row

        diff_values = np.arange(0, ng, dtype=np.float64)
        p_diff = np.array([np.sum(p[diff == value]) for value in diff_values], dtype=np.float64)
        sum_values = np.arange(2, 2 * ng + 1, dtype=np.float64)
        p_sum = np.array([np.sum(p[summ == value]) for value in sum_values], dtype=np.float64)
        diff_avg = float(np.sum(diff_values * p_diff))
        sum_avg = float(np.sum(sum_values * p_sum))

        hxy = float(-np.sum(p[p > 0.0] * np.log2(p[p > 0.0])))
        hx = float(-np.sum(px[px > 0.0] * np.log2(px[px > 0.0])))
        hy = float(-np.sum(py[py > 0.0] * np.log2(py[py > 0.0])))
        pxpy = px[:, None] * py[None, :]
        hxy1 = float(-np.sum(p[p > 0.0] * np.log2(pxpy[p > 0.0] + eps)))
        hxy2 = float(-np.sum(pxpy[pxpy > 0.0] * np.log2(pxpy[pxpy > 0.0])))

        features = OrderedDict()
        features["MaximumProbability"] = float(np.max(p))
        features["JointAverage"] = ux
        features["SumSquares"] = float(np.sum(((i_col - ux) ** 2.0) * p))
        features["JointEntropy"] = hxy
        features["DifferenceAverage"] = diff_avg
        features["DifferenceVariance"] = float(np.sum(((diff_values - diff_avg) ** 2.0) * p_diff))
        features["DifferenceEntropy"] = float(-np.sum(p_diff[p_diff > 0.0] * np.log2(p_diff[p_diff > 0.0])))
        features["SumAverage"] = sum_avg
        features["SumVariance"] = float(np.sum(((sum_values - sum_avg) ** 2.0) * p_sum))
        features["SumEntropy"] = float(-np.sum(p_sum[p_sum > 0.0] * np.log2(p_sum[p_sum > 0.0])))
        features["JointEnergy"] = float(np.sum(p**2.0))
        features["Contrast"] = float(np.sum((diff**2.0) * p))
        features["Dissimilarity"] = float(np.sum(diff * p))
        features["Id"] = float(np.sum(p / (1.0 + diff)))
        features["Idn"] = float(np.sum(p / (1.0 + diff / float(ng))))
        features["Idm"] = float(np.sum(p / (1.0 + diff**2.0)))
        features["Idmn"] = float(np.sum(p / (1.0 + (diff**2.0) / float(ng * ng))))
        inv_var = np.zeros_like(p)
        nonzero_diff = diff > 0.0
        inv_var[nonzero_diff] = p[nonzero_diff] / (diff[nonzero_diff] ** 2.0)
        features["InverseVariance"] = float(np.sum(inv_var))
        features["Correlation"] = float(0.0 if sx == 0.0 or sy == 0.0 else np.sum((i_col - ux) * (j_row - uy) * p) / (sx * sy))
        features["Autocorrelation"] = float(np.sum(i_col * j_row * p))
        cluster = i_col + j_row - ux - uy
        features["ClusterTendency"] = float(np.sum((cluster**2.0) * p))
        features["ClusterShade"] = float(np.sum((cluster**3.0) * p))
        features["ClusterProminence"] = float(np.sum((cluster**4.0) * p))
        denom = max(hx, hy)
        features["Imc1"] = float(0.0 if denom == 0.0 else (hxy - hxy1) / denom)
        features["Imc2"] = float(np.sqrt(max(0.0, 1.0 - np.exp(-2.0 * (hxy2 - hxy)))))
        features["MCC"] = cls._compute_mcc_from_matrix(p)
        return features

    @classmethod
    def _compute_direction_merged_features(cls, discretized, mask_array, ng, distances):
        values = []
        for matrix in cls._iter_direction_matrices(
            discretized,
            mask_array,
            int(ng),
            distances,
            force2d=True,
        ):
            values.append(cls._compute_features_from_matrix(matrix))
        if not values:
            return OrderedDict()
        out = OrderedDict()
        for key in values[0]:
            out[key] = float(np.nanmean([feature_values[key] for feature_values in values]))
        return out
    
    def execute(self):
        """
        Execute GLCM feature extraction
        
        Returns:
            OrderedDict of feature values
        """
        img_c = self._numpy_to_image(self.inputImage)
        mask_c = self._numpy_to_mask(self.inputMask)
        
        distances_c = (ctypes.c_int * len(self.distances))(*self.distances)
        
        discretized_volume, ng = (None, 0)
        use_cpu_discretized = self._should_use_cpu_discretized_path()
        include_mcc = self._enabled_features is None or "MCC" in self._enabled_features
        include_sum_variance = (
            self._enabled_features is None or "SumVariance" in self._enabled_features
        )
        include_dissimilarity = (
            self._enabled_features is None or "Dissimilarity" in self._enabled_features
        )
        python_mcc = bool(self.kwargs.get("glcmPythonMCC", False))
        include_native_mcc = include_mcc and not python_mcc
        if use_cpu_discretized or python_mcc or include_sum_variance or include_dissimilarity:
            discretized_volume, ng = self._get_binimage_discretized()

        if (
            bool(self.kwargs.get("force2D", False))
            and discretized_volume is not None
            and ng > 0
            and np.asarray(discretized_volume).ndim == 3
        ):
            self.featureValues = self._filter_features(
                self._compute_direction_merged_features(
                    discretized_volume,
                    self.inputMask,
                    int(ng),
                    self.distances,
                )
            )
            return self.featureValues

        if (
            discretized_volume is not None
            and ng > 0
            and use_cpu_discretized
        ):
            feature_lib = self._get_cpu_feature_lib()
            discretized_array = discretized_volume.reshape(-1)
            discretized_c = discretized_array.ctypes.data_as(
                ctypes.POINTER(ctypes.c_int)
            )
            selective_fn = getattr(
                feature_lib,
                "flash_radiomics_glcm_cpu_discretized_selective",
                None,
            )
            if selective_fn is not None:
                result_ptr = selective_fn(
                    img_c,
                    mask_c,
                    self.num_threads,
                    distances_c,
                    len(self.distances),
                    discretized_c,
                    ng,
                    1 if include_native_mcc else 0,
                )
            else:
                result_ptr = feature_lib.flash_radiomics_glcm_cpu_discretized(
                    img_c,
                    mask_c,
                    self.num_threads,
                    distances_c,
                    len(self.distances),
                    discretized_c,
                    ng,
                )
        else:
            feature_lib = self._feature_lib
            bin_width = self.kwargs.get("binWidth", 25)
            bin_count = self.kwargs.get("binCount", 0) or 0
            result_ptr = feature_lib.flash_radiomics_glcm(
                img_c,
                mask_c,
                self.backend_enum,
                self.num_threads,
                distances_c,
                len(self.distances),
                bin_width,
                bin_count,
            )
        
        features = self._result_to_dict(
            result_ptr,
            "GLCM",
            strip_prefix="glcm_",
            feature_lib=feature_lib,
        )

        if include_mcc and python_mcc:
            try:
                if discretized_volume is None or ng <= 0:
                    discretized_volume, ng = self._get_binimage_discretized()
                    if discretized_volume is None or ng <= 0:
                        raise ValueError("Discretization produced no gray levels")
                features["MCC"] = self._compute_mcc(
                    discretized_volume,
                    self.inputMask,
                    int(ng),
                    self.distances,
                )
            except Exception:
                features["MCC"] = float("nan")
        elif include_mcc and "MCC" not in features:
            try:
                if discretized_volume is None or ng <= 0:
                    discretized_volume, ng = self._get_binimage_discretized()
                    if discretized_volume is None or ng <= 0:
                        raise ValueError("Discretization produced no gray levels")
                features["MCC"] = self._compute_mcc(
                    discretized_volume,
                    self.inputMask,
                    int(ng),
                    self.distances,
                )
            except Exception:
                features["MCC"] = float("nan")

        if include_sum_variance or include_dissimilarity:
            try:
                if discretized_volume is None or ng <= 0:
                    discretized_volume, ng = self._get_binimage_discretized()
                    if discretized_volume is None or ng <= 0:
                        raise ValueError("Discretization produced no gray levels")
                redundant = self._compute_ibsi_redundant_features(
                    discretized_volume,
                    self.inputMask,
                    int(ng),
                    self.distances,
                    force2d=bool(self.kwargs.get("force2D", False)),
                )
                if include_sum_variance:
                    features["SumVariance"] = redundant["SumVariance"]
                if include_dissimilarity:
                    features["Dissimilarity"] = redundant["Dissimilarity"]
            except Exception:
                if include_sum_variance:
                    features["SumVariance"] = float("nan")
                if include_dissimilarity:
                    features["Dissimilarity"] = float("nan")

        self.featureValues = features
        
        
        return self.featureValues
