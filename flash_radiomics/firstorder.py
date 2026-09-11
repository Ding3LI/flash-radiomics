"""
First-order statistical features

Supports both 2D and 3D images with automatic dimensionality detection.
"""

# PYRADIOMICS: Feature names and default availability are aligned to PyRadiomics.

import ctypes
import numpy as np
from .base import RadiomicsFeaturesBase
from . import _bindings


class RadiomicsFirstOrder(RadiomicsFeaturesBase):
    """
    First-order statistical features extractor
    
    Computes statistical features from the intensity histogram including:
    - Energy, Total Energy
    - Entropy
    - Mean, Median, Mode
    - Standard Deviation, Variance
    - Skewness, Kurtosis
    - Minimum, Maximum, Range
    - Percentiles (10th, 90th)
    - Interquartile Range
    - Mean Absolute Deviation
    - Robust Mean Absolute Deviation
    - Root Mean Squared
    - Uniformity
    
    Works with both 2D and 3D images automatically.
    """
    
    def __init__(self, inputImage, inputMask, **kwargs):
        """
        Initialize first-order feature extractor
        
        Args:
            inputImage: numpy array of image data (2D or 3D)
            inputMask: numpy array of mask data (2D or 3D)
            **kwargs: Additional parameters (backend, num_threads, spacing, origin)
        """
        super().__init__(inputImage, inputMask, **kwargs)

    @staticmethod
    def _moment(values, order):
        if order == 1:
            return 0.0
        centered = values - np.nanmean(values)
        return float(np.nanmean(np.power(centered, order)))

    @classmethod
    def _compute_pyradiomics_scalar_overrides(
        cls,
        image_array,
        mask_array,
        spacing,
        voxel_array_shift,
        requested_features=None,
    ):
        requested = None
        if requested_features is not None:
            requested = set(requested_features)
            if not requested:
                return {}

        roi_values = np.asarray(image_array, dtype=np.float64)[
            np.asarray(mask_array) != 0
        ]
        if roi_values.size == 0:
            return {}

        overrides = {}
        need_mean = requested is None or (
            {
                "Mean",
                "Variance",
                "StandardDeviation",
                "Skewness",
                "Kurtosis",
                "MeanAbsoluteDeviation",
                "CoefficientOfVariation",
            }
            & requested
        )
        mean = float(np.nanmean(roi_values)) if need_mean else None

        if requested is None or "Variance" in requested:
            overrides["Variance"] = float(np.nanvar(roi_values))
        if requested is None or "StandardDeviation" in requested:
            overrides["StandardDeviation"] = float(np.nanstd(roi_values))

        need_skewness = requested is None or "Skewness" in requested
        need_kurtosis = requested is None or "Kurtosis" in requested
        if need_skewness or need_kurtosis:
            m2 = cls._moment(roi_values, 2)
            if need_skewness:
                m3 = cls._moment(roi_values, 3)
                overrides["Skewness"] = float(m3 / m2**1.5) if m2 > 0 else 0.0
            if need_kurtosis:
                m4 = cls._moment(roi_values, 4)
                overrides["Kurtosis"] = (
                    float(m4 / m2**2.0 - 3.0) if m2 > 0 else 0.0
                )

        if requested is None or "Mean" in requested:
            overrides["Mean"] = float(mean if mean is not None else np.nanmean(roi_values))
        if requested is None or "Minimum" in requested:
            overrides["Minimum"] = float(np.nanmin(roi_values))
        if requested is None or "Maximum" in requested:
            overrides["Maximum"] = float(np.nanmax(roi_values))
        if requested is None or "Range" in requested:
            overrides["Range"] = float(np.nanmax(roi_values) - np.nanmin(roi_values))
        if requested is None or "Median" in requested:
            overrides["Median"] = float(np.nanmedian(roi_values))

        need_percentiles = requested is None or (
            {
                "10Percentile",
                "90Percentile",
                "InterquartileRange",
                "RobustMeanAbsoluteDeviation",
            }
            & requested
        )
        p10 = p90 = None
        if need_percentiles:
            p10 = float(np.nanpercentile(roi_values, 10.0))
            p25 = float(np.nanpercentile(roi_values, 25.0))
            p75 = float(np.nanpercentile(roi_values, 75.0))
            p90 = float(np.nanpercentile(roi_values, 90.0))
            if requested is None or "10Percentile" in requested:
                overrides["10Percentile"] = p10
            if requested is None or "90Percentile" in requested:
                overrides["90Percentile"] = p90
            if requested is None or "InterquartileRange" in requested:
                overrides["InterquartileRange"] = float(p75 - p25)

        need_shifted = requested is None or (
            {"Energy", "TotalEnergy", "RootMeanSquared"} & requested
        )
        if need_shifted:
            shifted_values = roi_values + float(voxel_array_shift)
            energy = float(np.nansum(shifted_values**2))
            if requested is None or "Energy" in requested:
                overrides["Energy"] = energy
            if requested is None or "TotalEnergy" in requested:
                voxel_volume = float(np.multiply.reduce(np.asarray(spacing, dtype=np.float64)))
                overrides["TotalEnergy"] = float(energy * voxel_volume)
            if requested is None or "RootMeanSquared" in requested:
                overrides["RootMeanSquared"] = float(np.sqrt(np.nanmean(shifted_values**2)))

        if requested is None or "MeanAbsoluteDeviation" in requested:
            if mean is None:
                mean = float(np.nanmean(roi_values))
            overrides["MeanAbsoluteDeviation"] = float(np.nanmean(np.abs(roi_values - mean)))

        need_median = requested is None or ("Median" in requested or "MedianAbsoluteDeviation" in requested)
        median = None
        if need_median:
            median = float(np.nanmedian(roi_values))
            if requested is None or "Median" in requested:
                overrides["Median"] = median
        if requested is None or "MedianAbsoluteDeviation" in requested:
            if median is None:
                median = float(np.nanmedian(roi_values))
            overrides["MedianAbsoluteDeviation"] = float(
                np.nanmean(np.abs(roi_values - median))
            )

        if requested is None or "CoefficientOfVariation" in requested:
            sigma = float(np.nanstd(roi_values))
            if sigma == 0.0:
                overrides["CoefficientOfVariation"] = 0.0
            else:
                if mean is None:
                    mean = float(np.nanmean(roi_values))
                with np.errstate(divide="ignore", invalid="ignore"):
                    overrides["CoefficientOfVariation"] = float(
                        np.divide(np.float64(sigma), np.float64(mean))
                    )

        if requested is None or "QuartileCoefficientOfDispersion" in requested:
            p25 = float(np.nanpercentile(roi_values, 25.0))
            p75 = float(np.nanpercentile(roi_values, 75.0))
            denom = p75 + p25
            if denom == 0.0:
                overrides["QuartileCoefficientOfDispersion"] = 1.0
            else:
                overrides["QuartileCoefficientOfDispersion"] = float((p75 - p25) / denom)

        if requested is None or "RobustMeanAbsoluteDeviation" in requested:
            if p10 is None or p90 is None:
                p10 = float(np.nanpercentile(roi_values, 10.0))
                p90 = float(np.nanpercentile(roi_values, 90.0))
            percentile_values = roi_values[(roi_values >= p10) & (roi_values <= p90)]
            if percentile_values.size == 0:
                overrides["RobustMeanAbsoluteDeviation"] = 0.0
            else:
                overrides["RobustMeanAbsoluteDeviation"] = float(
                    np.nanmean(
                        np.abs(percentile_values - np.nanmean(percentile_values))
                    )
                )

        return overrides

    @staticmethod
    def _compute_robust_mean_absolute_deviation(image_array, mask_array):
        roi_values = np.asarray(image_array, dtype=np.float64)[
            np.asarray(mask_array) != 0
        ]
        if roi_values.size == 0:
            return 0.0

        p10 = np.percentile(roi_values, 10.0)
        p90 = np.percentile(roi_values, 90.0)
        percentile_values = roi_values[(roi_values >= p10) & (roi_values <= p90)]
        if percentile_values.size == 0:
            return 0.0

        mean_subset = np.mean(percentile_values)
        return float(np.mean(np.abs(percentile_values - mean_subset)))
    
    def execute(self):
        """
        Execute first-order feature extraction
        
        Returns:
            OrderedDict of feature values
        """
        img_c = self._numpy_to_image(self.inputImage)
        mask_c = self._numpy_to_mask(self.inputMask)
        
        voxel_array_shift = self.kwargs.get("voxelArrayShift", 0)
        discretized, ng = self._get_binimage_discretized()

        if discretized is not None and ng > 0 and self._should_use_cpu_discretized_path():
            feature_lib = self._get_cpu_feature_lib()
            discretized_array = discretized.reshape(-1)
            discretized_c = discretized_array.ctypes.data_as(
                ctypes.POINTER(ctypes.c_int)
            )
            result_ptr = feature_lib.flash_radiomics_firstorder_cpu_discretized(
                img_c,
                mask_c,
                self.num_threads,
                discretized_c,
                ng,
                voxel_array_shift,
            )
        else:
            feature_lib = self._feature_lib
            bin_width = self.kwargs.get("binWidth", 25)
            bin_count = self.kwargs.get("binCount", 0) or 0
            result_ptr = feature_lib.flash_radiomics_firstorder(
                img_c,
                mask_c,
                self.backend_enum,
                self.num_threads,
                bin_width,
                bin_count,
                voxel_array_shift,
            )

        enabled_features = self._enabled_features
        self._enabled_features = None
        try:
            features = self._result_to_dict(
                result_ptr,
                "First-order",
                strip_prefix="firstorder_",
                feature_lib=feature_lib,
            )
        finally:
            self._enabled_features = enabled_features

        requested_overrides = set(features.keys())
        requested_overrides.add("RobustMeanAbsoluteDeviation")
        requested_overrides.add("MedianAbsoluteDeviation")
        requested_overrides.add("CoefficientOfVariation")
        requested_overrides.add("QuartileCoefficientOfDispersion")
        scalar_overrides = self._compute_pyradiomics_scalar_overrides(
            self.inputImage,
            self.inputMask,
            self.kwargs.get("spacing", tuple([1.0] * self.inputImage.ndim)),
            voxel_array_shift,
            requested_features=requested_overrides,
        )
        use_c_scalar_values = self._resolved_backend_name() in {"cpu", "mps"}
        for feature_name, feature_value in scalar_overrides.items():
            # Convert Pearson kurtosis to IBSI excess kurtosis.
            if feature_name == "Kurtosis" and feature_name in features:
                features[feature_name] = feature_value
            elif use_c_scalar_values:
                features.setdefault(feature_name, feature_value)
            elif feature_name in features:
                features[feature_name] = feature_value

        include_robust_mad = (
            enabled_features is None
            or "RobustMeanAbsoluteDeviation" in enabled_features
        )
        if include_robust_mad:
            if "RobustMeanAbsoluteDeviation" not in features:
                features["RobustMeanAbsoluteDeviation"] = scalar_overrides.get(
                    "RobustMeanAbsoluteDeviation",
                    self._compute_robust_mean_absolute_deviation(
                        self.inputImage,
                        self.inputMask,
                    ),
                )

        include_median_abs_dev = (
            enabled_features is None or "MedianAbsoluteDeviation" in enabled_features
        )
        if include_median_abs_dev:
            if "MedianAbsoluteDeviation" not in features:
                features["MedianAbsoluteDeviation"] = scalar_overrides.get(
                    "MedianAbsoluteDeviation",
                    float(
                        np.nanmean(
                            np.abs(
                                np.asarray(self.inputImage, dtype=np.float64)[
                                    np.asarray(self.inputMask) != 0
                                ]
                                - np.nanmedian(
                                    np.asarray(self.inputImage, dtype=np.float64)[
                                        np.asarray(self.inputMask) != 0
                                    ]
                                )
                            )
                        )
                    ),
                )

        include_cov = enabled_features is None or "CoefficientOfVariation" in enabled_features
        if include_cov:
            if "CoefficientOfVariation" not in features:
                features["CoefficientOfVariation"] = scalar_overrides.get(
                    "CoefficientOfVariation",
                    0.0,
                )

        include_qcod = (
            enabled_features is None
            or "QuartileCoefficientOfDispersion" in enabled_features
        )
        if include_qcod:
            if "QuartileCoefficientOfDispersion" not in features:
                features["QuartileCoefficientOfDispersion"] = scalar_overrides.get(
                    "QuartileCoefficientOfDispersion",
                    1.0,
                )

        self.featureValues = self._filter_features(features)
        
        return self.featureValues
