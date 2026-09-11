"""
Gray Level Dependence Matrix (GLDM) features
"""

# PYRADIOMICS: Feature names and default availability are aligned to PyRadiomics.

import ctypes
import numpy as np
from collections import OrderedDict
from .base import RadiomicsFeaturesBase
from . import _bindings


class RadiomicsGLDM(RadiomicsFeaturesBase):
    """
    GLDM texture features extractor.
    """

    _DIRECTIONS_2D = (
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
        (1, 1),
        (-1, -1),
        (1, -1),
        (-1, 1),
    )
    _DIRECTIONS_3D = (
        (1, 0, 0),
        (-1, 0, 0),
        (0, 1, 0),
        (0, -1, 0),
        (0, 0, 1),
        (0, 0, -1),
        (1, 1, 0),
        (-1, -1, 0),
        (1, -1, 0),
        (-1, 1, 0),
        (1, 0, 1),
        (-1, 0, -1),
        (1, 0, -1),
        (-1, 0, 1),
        (0, 1, 1),
        (0, -1, -1),
        (0, 1, -1),
        (0, -1, 1),
        (1, 1, 1),
        (-1, -1, -1),
        (1, 1, -1),
        (-1, -1, 1),
        (1, -1, 1),
        (-1, 1, -1),
        (1, -1, -1),
        (-1, 1, 1),
    )

    def __init__(self, inputImage, inputMask, **kwargs):
        """
        Initialize GLDM feature extractor.

        Args:
            inputImage: numpy array of image data (2D or 3D)
            inputMask: numpy array of mask data (2D or 3D)
            **kwargs: Additional parameters including:
                - distances: list of distances (default: [1])
                - gldm_a: dependence threshold (default: 0)
                - backend: 'cpu', 'mps', 'cuda', or 'auto'
                - num_threads: number of threads for CPU
        """
        super().__init__(inputImage, inputMask, **kwargs)
        self.distances = kwargs.get("distances", [1])
        self.gldm_a = kwargs.get("gldm_a", 0)

    @staticmethod
    def _shift_slices(offset: int):
        if offset > 0:
            return slice(0, -offset), slice(offset, None)
        if offset < 0:
            return slice(-offset, None), slice(0, offset)
        return slice(None), slice(None)

    def _compute_dependence_count_energy(
        self,
        discretized: np.ndarray,
        roi_mask: np.ndarray,
        ng: int,
    ) -> tuple[float, int]:
        roi_mask = np.asarray(roi_mask, dtype=bool)
        discretized = np.asarray(discretized, dtype=np.int32)
        if discretized.ndim not in (2, 3):
            return np.nan, 0

        dep = np.zeros(discretized.shape, dtype=np.int32)
        dirs = self._DIRECTIONS_2D if discretized.ndim == 2 else self._DIRECTIONS_3D
        dep_len = len(dirs) * len(self.distances) + 1

        for dist in self.distances:
            step = int(dist)
            if step <= 0:
                continue
            for direction in dirs:
                src_slices = []
                nbr_slices = []
                for axis, sign in enumerate(direction):
                    src_slice, nbr_slice = self._shift_slices(sign * step)
                    src_slices.append(src_slice)
                    nbr_slices.append(nbr_slice)
                src_slices = tuple(src_slices)
                nbr_slices = tuple(nbr_slices)

                src_mask = roi_mask[src_slices]
                nbr_mask = roi_mask[nbr_slices]
                valid = src_mask & nbr_mask
                if not np.any(valid):
                    continue

                src_gray = discretized[src_slices]
                nbr_gray = discretized[nbr_slices]
                similar = valid & (src_gray > 0) & (nbr_gray > 0)
                similar &= (np.abs(src_gray - nbr_gray) <= float(self.gldm_a))
                if not np.any(similar):
                    continue

                dep_view = dep[src_slices]
                dep_view[similar] += 1

        centers = roi_mask & (discretized > 0)
        if not np.any(centers):
            return np.nan, 0

        g_idx = discretized[centers] - 1
        d_idx = dep[centers]
        n_s = int(g_idx.size)
        linear_idx = g_idx * dep_len + d_idx
        counts = np.bincount(
            linear_idx,
            minlength=int(max(int(ng), int(np.max(discretized[centers])))) * dep_len,
        ).astype(np.float64, copy=False)
        return float(np.sum(counts * counts) / (float(n_s) * float(n_s))), n_s

    @classmethod
    def _compute_ngldm_counts_2d_merged(cls, discretized, roi_mask, ng, distances, alpha):
        dirs = cls._DIRECTIONS_2D
        max_dep = len(dirs) * len(distances) + 1
        counts = np.zeros((int(ng), int(max_dep)), dtype=np.float64)
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
                    dep = 1
                    for dist in distances:
                        step = int(dist)
                        if step <= 0:
                            continue
                        for dy, dx in dirs:
                            ny = y + dy * step
                            nx = x + dx * step
                            if 0 <= ny < y_size and 0 <= nx < x_size and roi_mask[z, ny, nx]:
                                nbr_gray = int(discretized[z, ny, nx])
                                if nbr_gray > 0 and abs(nbr_gray - gray) <= alpha:
                                    dep += 1
                    counts[gray - 1, dep - 1] += 1.0
        return counts, n_voxels

    @staticmethod
    def _features_from_counts(counts, n_voxels):
        features = OrderedDict()
        n_s = float(np.sum(counts))
        if n_s <= 0.0:
            return features
        i = np.arange(1, counts.shape[0] + 1, dtype=np.float64)
        j = np.arange(1, counts.shape[1] + 1, dtype=np.float64)
        si = np.sum(counts, axis=1)
        sj = np.sum(counts, axis=0)
        i_col = i[:, None]
        j_row = j[None, :]
        features["SmallDependenceEmphasis"] = float(np.sum(sj / (j**2.0)) / n_s)
        features["LargeDependenceEmphasis"] = float(np.sum(sj * (j**2.0)) / n_s)
        features["LowGrayLevelEmphasis"] = float(np.sum(si / (i**2.0)) / n_s)
        features["HighGrayLevelEmphasis"] = float(np.sum(si * (i**2.0)) / n_s)
        features["SmallDependenceLowGrayLevelEmphasis"] = float(np.sum(counts / ((i_col * j_row) ** 2.0)) / n_s)
        features["SmallDependenceHighGrayLevelEmphasis"] = float(np.sum(counts * (i_col**2.0) / (j_row**2.0)) / n_s)
        features["LargeDependenceLowGrayLevelEmphasis"] = float(np.sum(counts * (j_row**2.0) / (i_col**2.0)) / n_s)
        features["LargeDependenceHighGrayLevelEmphasis"] = float(np.sum(counts * (i_col**2.0) * (j_row**2.0)) / n_s)
        features["GrayLevelNonUniformity"] = float(np.sum(si**2.0) / n_s)
        features["GrayLevelNonUniformityNormalized"] = float(np.sum(si**2.0) / (n_s**2.0))
        features["DependenceNonUniformity"] = float(np.sum(sj**2.0) / n_s)
        features["DependenceNonUniformityNormalized"] = float(np.sum(sj**2.0) / (n_s**2.0))
        features["DependenceCountPercentage"] = float(n_s / float(max(n_voxels, 1)))
        mu_i = float(np.sum(counts * i_col) / n_s)
        features["GrayLevelVariance"] = float(np.sum(((i_col - mu_i) ** 2.0) * counts) / n_s)
        mu_j = float(np.sum(counts * j_row) / n_s)
        features["DependenceVariance"] = float(np.sum(((j_row - mu_j) ** 2.0) * counts) / n_s)
        p = counts / n_s
        features["DependenceEntropy"] = float(-np.sum(p[p > 0.0] * np.log2(p[p > 0.0])))
        features["DependenceCountEnergy"] = float(np.sum(counts**2.0) / (n_s**2.0))
        return features

    def execute(self):
        """
        Execute GLDM feature extraction.

        Returns:
            OrderedDict of feature values
        """
        img_c = self._numpy_to_image(self.inputImage)
        mask_c = self._numpy_to_mask(self.inputMask)

        distances_c = (ctypes.c_int * len(self.distances))(*self.distances)

        discretized, ng = (None, 0)
        resolved_backend = self._resolved_backend_name()
        use_cpu_discretized = self._should_use_cpu_discretized_path()
        use_2d_merged = (
            bool(self.kwargs.get("force2D", False))
            and np.asarray(self.inputImage).ndim == 3
        )
        if use_cpu_discretized or (use_2d_merged and resolved_backend != "cuda"):
            discretized, ng = self._get_binimage_discretized()

        if (
            use_2d_merged
            and resolved_backend != "cuda"
            and discretized is not None
            and ng > 0
        ):
            counts, n_voxels = self._compute_ngldm_counts_2d_merged(
                np.asarray(discretized, dtype=np.int32),
                np.asarray(self.inputMask) != 0,
                int(ng),
                self.distances,
                float(self.gldm_a),
            )
            self.featureValues = self._filter_features(
                self._features_from_counts(counts, n_voxels)
            )
            return self.featureValues

        if discretized is not None and ng > 0 and use_cpu_discretized:
            feature_lib = self._get_cpu_feature_lib()
            discretized_array = discretized.reshape(-1)
            discretized_c = discretized_array.ctypes.data_as(
                ctypes.POINTER(ctypes.c_int)
            )
            result_ptr = feature_lib.flash_radiomics_gldm_cpu_discretized(
                img_c,
                mask_c,
                self.num_threads,
                distances_c,
                len(self.distances),
                discretized_c,
                ng,
                self.gldm_a,
            )
        else:
            feature_lib = self._feature_lib
            bin_width = self.kwargs.get("binWidth", 25)
            bin_count = self.kwargs.get("binCount", 0) or 0
            result_ptr = feature_lib.flash_radiomics_gldm(
                img_c,
                mask_c,
                self.backend_enum,
                self.num_threads,
                distances_c,
                len(self.distances),
                bin_width,
                bin_count,
                self.gldm_a,
                int(use_2d_merged),
            )

        enabled_features = self._enabled_features
        self._enabled_features = None
        try:
            features = self._result_to_dict(
                result_ptr,
                "GLDM",
                strip_prefix="gldm_",
                feature_lib=feature_lib,
            )
        finally:
            self._enabled_features = enabled_features

        include_dep_count_percentage = (
            enabled_features is None
            or "DependenceCountPercentage" in enabled_features
        )
        include_dep_count_energy = (
            enabled_features is None
            or "DependenceCountEnergy" in enabled_features
        )
        include_glnu_norm = (
            enabled_features is None
            or "GrayLevelNonUniformityNormalized" in enabled_features
        )
        need_dep_count_percentage = (
            include_dep_count_percentage
            and "DependenceCountPercentage" not in features
        )
        need_dep_count_energy = (
            include_dep_count_energy
            and "DependenceCountEnergy" not in features
        )
        need_glnu_norm = (
            include_glnu_norm
            and "GrayLevelNonUniformityNormalized" not in features
        )
        if need_dep_count_percentage or need_dep_count_energy or need_glnu_norm:
            roi_mask = np.asarray(self.inputMask) != 0
            n_voxels = int(np.count_nonzero(roi_mask))
            if n_voxels <= 0:
                if need_dep_count_percentage:
                    features["DependenceCountPercentage"] = np.nan
                if need_dep_count_energy:
                    features["DependenceCountEnergy"] = np.nan
                if need_glnu_norm:
                    features["GrayLevelNonUniformityNormalized"] = np.nan
            else:
                if discretized is None:
                    discretized, _ = self._get_binimage_discretized()
                if discretized is None:
                    if need_dep_count_percentage:
                        features["DependenceCountPercentage"] = np.nan
                    if need_dep_count_energy:
                        features["DependenceCountEnergy"] = np.nan
                    if need_glnu_norm:
                        features["GrayLevelNonUniformityNormalized"] = np.nan
                else:
                    n_s = int(
                        np.count_nonzero(
                            np.asarray(discretized, dtype=np.int32)[roi_mask] > 0
                        )
                    )
                    if need_dep_count_percentage:
                        features["DependenceCountPercentage"] = float(n_s) / float(n_voxels)
                    if need_dep_count_energy:
                        energy, _ = self._compute_dependence_count_energy(
                            discretized=np.asarray(discretized, dtype=np.int32),
                            roi_mask=roi_mask,
                            ng=int(ng),
                        )
                        features["DependenceCountEnergy"] = energy
                    if need_glnu_norm:
                        glnu = float(features.get("GrayLevelNonUniformity", np.nan))
                        features["GrayLevelNonUniformityNormalized"] = (
                            float(glnu / float(n_s)) if n_s > 0 else np.nan
                        )

        self.featureValues = self._filter_features(features)

        return self.featureValues
