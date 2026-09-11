"""
Intensity Histogram (IH) features.

This implementation targets segment-based extraction and computes IBSI-style
discretized intensity histogram statistics from ROI voxels.
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np

from .base import RadiomicsFeaturesBase


class RadiomicsIH(RadiomicsFeaturesBase):
    """Intensity histogram features for segment-based workflows."""

    _FEATURE_ORDER = (
        "Mean",
        "Variance",
        "Skewness",
        "Kurtosis",
        "Median",
        "Minimum",
        "10Percentile",
        "90Percentile",
        "Maximum",
        "Mode",
        "InterquartileRange",
        "Range",
        "MeanAbsoluteDeviation",
        "RobustMeanAbsoluteDeviation",
        "MedianAbsoluteDeviation",
        "CoefficientOfVariation",
        "QuartileCoefficientOfDispersion",
        "Entropy",
        "Uniformity",
        "MaximumHistogramGradient",
        "MaximumHistogramGradientIntensity",
        "MinimumHistogramGradient",
        "MinimumHistogramGradientIntensity",
    )

    @classmethod
    def _empty_feature_dict(cls) -> OrderedDict:
        return OrderedDict((name, np.nan) for name in cls._FEATURE_ORDER)

    @classmethod
    def _compute_features(
        cls,
        roi_values: np.ndarray,
        counts: np.ndarray,
        levels: np.ndarray,
        probabilities: np.ndarray,
    ) -> OrderedDict:
        features = cls._empty_feature_dict()
        if roi_values.size == 0:
            return features

        mu = float(np.sum(levels * probabilities))
        var = float(np.sum(((levels - mu) ** 2.0) * probabilities))
        sigma = float(np.sqrt(max(var, 0.0)))

        features["Mean"] = mu
        features["Variance"] = var
        if sigma == 0.0:
            features["Skewness"] = 0.0
            features["Kurtosis"] = 0.0
        else:
            features["Skewness"] = float(
                np.sum(((levels - mu) ** 3.0) * probabilities) / (sigma**3.0)
            )
            features["Kurtosis"] = float(
                np.sum(((levels - mu) ** 4.0) * probabilities) / (sigma**4.0) - 3.0
            )

        features["Median"] = float(np.median(roi_values))
        features["Minimum"] = float(np.min(roi_values))
        features["10Percentile"] = float(np.percentile(roi_values, 10.0))
        features["90Percentile"] = float(np.percentile(roi_values, 90.0))
        features["Maximum"] = float(np.max(roi_values))

        max_count = int(np.max(counts))
        mode_candidates = levels[counts == max_count]
        features["Mode"] = float(mode_candidates[np.argmin(np.abs(mode_candidates - mu))])

        p25 = float(np.percentile(roi_values, 25.0))
        p75 = float(np.percentile(roi_values, 75.0))
        iqr = p75 - p25
        features["InterquartileRange"] = float(iqr)
        features["Range"] = float(features["Maximum"] - features["Minimum"])
        features["MeanAbsoluteDeviation"] = float(np.mean(np.abs(roi_values - mu)))

        p10 = float(features["10Percentile"])
        p90 = float(features["90Percentile"])
        robust_values = roi_values[(roi_values >= p10) & (roi_values <= p90)]
        if robust_values.size == 0:
            features["RobustMeanAbsoluteDeviation"] = 0.0
        else:
            features["RobustMeanAbsoluteDeviation"] = float(
                np.mean(np.abs(robust_values - np.mean(robust_values)))
            )

        median_value = float(features["Median"])
        features["MedianAbsoluteDeviation"] = float(np.mean(np.abs(roi_values - median_value)))

        features["CoefficientOfVariation"] = 0.0 if sigma == 0.0 else float(sigma / mu)
        denom = p75 + p25
        if denom == 0.0:
            features["QuartileCoefficientOfDispersion"] = 1.0
        else:
            features["QuartileCoefficientOfDispersion"] = float(iqr / denom)

        nonzero_p = probabilities[probabilities > 0.0]
        features["Entropy"] = float(-np.sum(nonzero_p * np.log2(nonzero_p)))
        features["Uniformity"] = float(np.sum(probabilities**2.0))

        if counts.size > 1:
            gradients = np.gradient(counts.astype(np.float64))
        else:
            gradients = np.zeros(1, dtype=np.float64)
        max_grad_index = int(np.argmax(gradients))
        min_grad_index = int(np.argmin(gradients))
        features["MaximumHistogramGradient"] = float(gradients[max_grad_index])
        features["MaximumHistogramGradientIntensity"] = float(levels[max_grad_index])
        features["MinimumHistogramGradient"] = float(gradients[min_grad_index])
        features["MinimumHistogramGradientIntensity"] = float(levels[min_grad_index])

        return features

    def execute(self):
        roi_mask = np.asarray(self.inputMask) != 0
        discretized, ng = self._get_binimage_discretized()
        if discretized is None:
            self.featureValues = self._filter_features(self._empty_feature_dict())
            return self.featureValues

        roi_values = np.asarray(discretized, dtype=np.int64)[roi_mask]
        roi_values = roi_values[roi_values > 0]
        if roi_values.size == 0:
            self.featureValues = self._filter_features(self._empty_feature_dict())
            return self.featureValues

        ng = int(max(int(ng), int(np.max(roi_values))))
        counts = np.bincount(roi_values, minlength=ng + 1)[1 : ng + 1].astype(np.float64, copy=False)
        probabilities = counts / float(roi_values.size)
        levels = np.arange(1, ng + 1, dtype=np.float64)

        features = self._compute_features(
            roi_values=roi_values.astype(np.float64, copy=False),
            counts=counts,
            levels=levels,
            probabilities=probabilities,
        )
        self.featureValues = self._filter_features(features)
        return self.featureValues
