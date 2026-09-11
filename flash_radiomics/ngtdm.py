"""
Neighboring Gray Tone Difference Matrix (NGTDM) features
"""

# PYRADIOMICS: Feature names and default availability are aligned to PyRadiomics.

import ctypes
import numpy as np
from collections import OrderedDict
from .base import RadiomicsFeaturesBase
from . import _bindings


class RadiomicsNGTDM(RadiomicsFeaturesBase):
    """
    NGTDM texture features extractor.
    """

    def __init__(self, inputImage, inputMask, **kwargs):
        """
        Initialize NGTDM feature extractor.

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

    _DIRECTIONS_2D = (
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    )

    @classmethod
    def _compute_2d_merged_features(cls, discretized, roi_mask, ng):
        n_by_gray = np.zeros(int(ng), dtype=np.float64)
        s_by_gray = np.zeros(int(ng), dtype=np.float64)
        z_size, y_size, x_size = discretized.shape
        for z in range(z_size):
            for y in range(y_size):
                for x in range(x_size):
                    if not roi_mask[z, y, x]:
                        continue
                    gray = int(discretized[z, y, x])
                    if gray <= 0 or gray > ng:
                        continue
                    total = 0.0
                    count = 0
                    for dy, dx in cls._DIRECTIONS_2D:
                        ny = y + dy
                        nx = x + dx
                        if 0 <= ny < y_size and 0 <= nx < x_size and roi_mask[z, ny, nx]:
                            nbr_gray = int(discretized[z, ny, nx])
                            if nbr_gray > 0:
                                total += float(nbr_gray)
                                count += 1
                    if count <= 0:
                        continue
                    n_by_gray[gray - 1] += 1.0
                    s_by_gray[gray - 1] += abs(float(gray) - total / float(count))

        n_voxels = float(np.sum(n_by_gray))
        features = OrderedDict(
            (
                (name, np.nan)
                for name in ("Coarseness", "Contrast", "Busyness", "Complexity", "Strength")
            )
        )
        if n_voxels <= 0.0:
            return features

        present = n_by_gray > 0.0
        levels = np.arange(1, int(ng) + 1, dtype=np.float64)[present]
        n = n_by_gray[present]
        s = s_by_gray[present]
        p = n / n_voxels
        n_p = float(levels.size)
        weighted_s = float(np.sum(p * s))
        features["Coarseness"] = float(1.0 / max(weighted_s, 1.0e-6))
        if n_p > 1.0:
            i = levels[:, None]
            j = levels[None, :]
            pi = p[:, None]
            pj = p[None, :]
            si = s[:, None]
            sj = s[None, :]
            features["Contrast"] = float(
                np.sum(pi * pj * ((i - j) ** 2.0)) / (n_p * (n_p - 1.0)) * np.sum(s) / n_voxels
            )
            denom = float(np.sum(np.abs(i * pi - j * pj)))
            features["Busyness"] = float(weighted_s / denom) if denom > 0.0 else 0.0
            features["Complexity"] = float(
                np.sum(np.abs(i - j) * (pi * si + pj * sj) / (pi + pj)) / n_voxels
            )
            strength_denom = float(np.sum(s))
            features["Strength"] = float(
                np.sum((pi + pj) * ((i - j) ** 2.0)) / strength_denom
            ) if strength_denom > 0.0 else 0.0
        else:
            features["Contrast"] = 0.0
            features["Busyness"] = 0.0
            features["Complexity"] = 0.0
            features["Strength"] = 0.0
        return features

    def execute(self):
        """
        Execute NGTDM feature extraction.

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
            self.featureValues = self._filter_features(
                self._compute_2d_merged_features(
                    np.asarray(discretized, dtype=np.int32),
                    np.asarray(self.inputMask) != 0,
                    int(ng),
                )
            )
            return self.featureValues

        if discretized is not None and ng > 0 and use_cpu_discretized:
            feature_lib = self._get_cpu_feature_lib()
            discretized_array = discretized.reshape(-1)
            discretized_c = discretized_array.ctypes.data_as(
                ctypes.POINTER(ctypes.c_int)
            )
            result_ptr = feature_lib.flash_radiomics_ngtdm_cpu_discretized(
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
            result_ptr = feature_lib.flash_radiomics_ngtdm(
                img_c,
                mask_c,
                self.backend_enum,
                self.num_threads,
                distances_c,
                len(self.distances),
                bin_width,
                bin_count,
                int(use_2d_merged),
            )

        self.featureValues = self._result_to_dict(
            result_ptr,
            "NGTDM",
            strip_prefix="ngtdm_",
            feature_lib=feature_lib,
        )

        return self.featureValues
