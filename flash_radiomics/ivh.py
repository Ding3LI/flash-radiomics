"""
Intensity Volume Histogram (IVH) features.

This implementation targets segment-based extraction and computes the
10/90-percentile IVH features used by the current IBSI missing-feature plan.
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np

from .base import RadiomicsFeaturesBase


class RadiomicsIVH(RadiomicsFeaturesBase):
    """Intensity volume histogram features for segment-based workflows."""

    _FEATURE_ORDER = (
        "VolumeFractionAt10Intensity",
        "VolumeFractionAt90Intensity",
        "IntensityAt10Volume",
        "IntensityAt90Volume",
        "VolumeFractionDifferenceBetween10And90Intensity",
        "IntensityDifferenceBetween10And90Volume",
    )

    @classmethod
    def _empty_feature_dict(cls) -> OrderedDict:
        return OrderedDict((name, np.nan) for name in cls._FEATURE_ORDER)

    @staticmethod
    def _volume_fraction_at_intensity(
        gamma: np.ndarray,
        nu: np.ndarray,
        percentile: float,
    ) -> float:
        selected = nu[gamma >= (percentile / 100.0)]
        if selected.size == 0:
            return np.nan
        return float(np.max(selected))

    @staticmethod
    def _intensity_at_volume(
        levels: np.ndarray,
        nu: np.ndarray,
        percentile: float,
        next_max_intensity: float,
    ) -> float:
        selected = levels[nu <= (percentile / 100.0)]
        if selected.size == 0:
            return float(next_max_intensity)
        return float(np.min(selected))

    @classmethod
    def _compute_histogram_data(
        cls,
        roi_values: np.ndarray,
        discretization_method: str = "none",
        bin_count: int | None = None,
        bin_width: float | None = None,
        intensity_range: tuple[float, float] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        n_voxels = int(roi_values.size)
        method = str(discretization_method or "none").lower()
        if intensity_range is None:
            lo = float(np.min(roi_values))
            hi = float(np.max(roi_values))
        else:
            lo, hi = sorted(float(v) for v in intensity_range)

        if method == "none":
            lo_i = int(lo)
            hi_i = int(hi)
            levels = np.arange(lo_i, hi_i + 1, dtype=np.float64)
            if levels.size == 0:
                return levels, levels, levels, float("nan")
            shifted = np.rint(roi_values).astype(np.int64, copy=False) - lo_i
            valid = (shifted >= 0) & (shifted < levels.size)
            counts = np.bincount(shifted[valid], minlength=levels.size).astype(np.float64, copy=False)
            next_max_intensity = float(hi + 1.0)
        elif method == "fixed_bin_size":
            width = float(bin_width if bin_width is not None else 1.0)
            n_bins = int(np.ceil((hi - lo) / width) + 1.0)
            bins = np.floor((roi_values - lo) / width).astype(np.int64, copy=False) + 1
            bins[bins <= 0] = 1
            bins[bins > n_bins] = n_bins
            counts = np.bincount(bins, minlength=n_bins + 1)[1 : n_bins + 1].astype(np.float64, copy=False)
            levels = lo + (np.arange(1, n_bins + 1, dtype=np.float64) - 0.5) * width
            lo = float(np.min(levels))
            hi = float(np.max(levels))
            next_max_intensity = float(hi + width)
        elif method == "fixed_bin_number":
            n_bins = int(bin_count if bin_count is not None else 1000)
            lo = float(np.min(roi_values))
            hi = float(np.max(roi_values))
            if hi <= lo:
                bins = np.ones(roi_values.shape, dtype=np.int64)
            else:
                bins = np.floor(n_bins * (roi_values - lo) / (hi - lo)).astype(np.int64, copy=False) + 1
            bins[bins <= 0] = 1
            bins[bins >= n_bins] = n_bins
            counts = np.bincount(bins, minlength=n_bins + 1)[1 : n_bins + 1].astype(np.float64, copy=False)
            levels = np.arange(1, n_bins + 1, dtype=np.float64)
            lo = 1.0
            hi = float(n_bins)
            next_max_intensity = float(n_bins + 1.0)
        else:
            raise ValueError(f"Unsupported IVH discretization method: {discretization_method}")

        if levels.size <= 1 or hi == lo:
            gamma = np.zeros(levels.shape, dtype=np.float64)
        else:
            gamma = (levels - lo) / (hi - lo)

        cumulative_before = np.cumsum(np.concatenate(([0.0], counts[:-1])))
        nu = 1.0 - cumulative_before / float(max(n_voxels, 1))
        return levels, gamma, nu, next_max_intensity

    @classmethod
    def _compute_features(
        cls,
        levels: np.ndarray,
        gamma: np.ndarray,
        nu: np.ndarray,
        next_max_intensity: float,
    ) -> OrderedDict:
        features = cls._empty_feature_dict()
        if levels.size == 0:
            return features

        v10 = cls._volume_fraction_at_intensity(gamma, nu, 10.0)
        v90 = cls._volume_fraction_at_intensity(gamma, nu, 90.0)
        i10 = cls._intensity_at_volume(levels, nu, 10.0, next_max_intensity)
        i90 = cls._intensity_at_volume(levels, nu, 90.0, next_max_intensity)

        features["VolumeFractionAt10Intensity"] = v10
        features["VolumeFractionAt90Intensity"] = v90
        features["IntensityAt10Volume"] = i10
        features["IntensityAt90Volume"] = i90
        features["VolumeFractionDifferenceBetween10And90Intensity"] = float(v10 - v90)
        features["IntensityDifferenceBetween10And90Volume"] = float(i10 - i90)
        return features

    def execute(self):
        roi_mask = np.asarray(self.inputMask) != 0
        discretized = self.kwargs.get("discretized")
        use_shared_discretization = (
            discretized is not None
            and "ivhDiscretizationMethod" not in self.kwargs
        )
        if use_shared_discretization:
            discretized_array = np.asarray(discretized)
            if discretized_array.shape != roi_mask.shape:
                raise ValueError(
                    "IVH discretized input must match the image and mask shape"
                )
            roi_values = np.asarray(discretized_array, dtype=np.float64)[roi_mask]
        else:
            roi_values = np.asarray(self.inputImage, dtype=np.float64)[roi_mask]
        if roi_values.size == 0:
            self.featureValues = self._filter_features(self._empty_feature_dict())
            return self.featureValues

        intensity_range = self.kwargs.get("ivhIntensityRange")
        if intensity_range is not None:
            intensity_range = tuple(float(v) for v in intensity_range)
        elif use_shared_discretization:
            ng = int(self.kwargs.get("ng", 0) or 0)
            if ng > 0:
                intensity_range = (1.0, float(ng))
        levels, gamma, nu, next_max_intensity = self._compute_histogram_data(
            roi_values=roi_values,
            discretization_method=(
                "none"
                if use_shared_discretization
                else self.kwargs.get("ivhDiscretizationMethod", "none")
            ),
            bin_count=self.kwargs.get("ivhBinCount"),
            bin_width=self.kwargs.get("ivhBinWidth"),
            intensity_range=intensity_range,
        )
        features = self._compute_features(
            levels=levels,
            gamma=gamma,
            nu=nu,
            next_max_intensity=next_max_intensity,
        )
        self.featureValues = self._filter_features(features)
        return self.featureValues
