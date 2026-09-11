from __future__ import annotations

import datetime as _datetime
import math
import os
from pathlib import Path
from typing import Literal

import numpy as np
import SimpleITK as sitk


SUVDecayCorrection = Literal["NONE", "START", "ADMIN"]
SUVUnits = Literal["CNTS", "BQML", "GML", "CM2ML"]


def _as_float_image(image: sitk.Image) -> sitk.Image:
    return sitk.Cast(image, sitk.sitkFloat64)


def _datetime_delta_seconds(
    later: _datetime.datetime, earlier: _datetime.datetime
) -> float:
    return float((later - earlier).total_seconds())


def compute_pet_decayed_dose(
    injected_dose_bq: float,
    decay_correction: SUVDecayCorrection = "ADMIN",
    half_life_s: float | None = None,
    injection_datetime: _datetime.datetime | None = None,
    acquisition_datetime: _datetime.datetime | None = None,
    frame_duration_s: float | None = None,
    frame_reference_time_s: float | None = None,
) -> float:
    """
    Compute the effective injected dose for SUV scaling.

    This follows the same core convention used by MIRP: ADMIN means PET values
    are already decay-corrected to administration time, START means values are
    corrected to scan start, and NONE means no PET decay correction was applied.
    """
    injected_dose_bq = float(injected_dose_bq)
    if injected_dose_bq <= 0.0:
        raise ValueError("injected_dose_bq must be positive")

    decay_correction = decay_correction.upper()
    if decay_correction == "ADMIN":
        return injected_dose_bq

    if half_life_s is None or half_life_s <= 0.0:
        raise ValueError("half_life_s must be positive for NONE or START decay correction")
    if injection_datetime is None or acquisition_datetime is None:
        raise ValueError(
            "injection_datetime and acquisition_datetime are required for NONE or START decay correction"
        )

    if decay_correction == "NONE":
        if frame_duration_s is None:
            raise ValueError("frame_duration_s is required for NONE decay correction")
        frame_center = acquisition_datetime + _datetime.timedelta(
            seconds=float(frame_duration_s) / 2.0
        )
        decay_time_s = _datetime_delta_seconds(frame_center, injection_datetime)
        return injected_dose_bq / math.pow(2.0, decay_time_s / float(half_life_s))

    if decay_correction == "START":
        if frame_duration_s is None or frame_reference_time_s is None:
            raise ValueError(
                "frame_duration_s and frame_reference_time_s are required for START decay correction"
            )
        decay_constant = math.log(2.0) / float(half_life_s)
        decay_during_frame = decay_constant * float(frame_duration_s)
        time_count_average_s = (
            math.log(decay_during_frame / (1.0 - math.exp(-decay_during_frame)))
            / decay_constant
        )
        reference_start = acquisition_datetime + _datetime.timedelta(
            seconds=time_count_average_s - float(frame_reference_time_s)
        )
        decay_time_s = _datetime_delta_seconds(reference_start, injection_datetime)
        return injected_dose_bq / math.pow(2.0, decay_time_s / float(half_life_s))

    raise ValueError("decay_correction must be one of NONE, START, or ADMIN")


def convert_pet_counts_to_suv(
    image: sitk.Image,
    patient_weight_kg: float,
    injected_dose_bq: float | None = None,
    units: SUVUnits = "CNTS",
    counts_to_bqml_scale: float | None = None,
    philips_suv_scale: float | None = None,
    decay_correction: SUVDecayCorrection = "ADMIN",
    half_life_s: float | None = None,
    injection_datetime: _datetime.datetime | None = None,
    acquisition_datetime: _datetime.datetime | None = None,
    frame_duration_s: float | None = None,
    frame_reference_time_s: float | None = None,
) -> sitk.Image:
    """
    Convert PET voxel values to body-weight SUV.

    For CNTS images, pass either a direct Philips SUV scale or a counts-to-BQML
    scale plus injected dose metadata. For BQML images, pass injected dose
    metadata. GML and CM2ML inputs are already SUV-like and are returned as a
    floating-point copy.
    """
    patient_weight_kg = float(patient_weight_kg)
    if patient_weight_kg <= 0.0:
        raise ValueError("patient_weight_kg must be positive")

    units = units.upper()
    if units in {"GML", "CM2ML"}:
        return _as_float_image(image)

    if units == "CNTS" and philips_suv_scale is not None:
        return _as_float_image(image) * float(philips_suv_scale)

    if units == "CNTS":
        if counts_to_bqml_scale is None:
            raise ValueError(
                "counts_to_bqml_scale or philips_suv_scale is required for CNTS input"
            )
        activity_scale = float(counts_to_bqml_scale)
    elif units == "BQML":
        activity_scale = 1.0
    else:
        raise ValueError("units must be one of CNTS, BQML, GML, or CM2ML")

    if injected_dose_bq is None:
        raise ValueError("injected_dose_bq is required when converting activity to SUV")

    decayed_dose_bq = compute_pet_decayed_dose(
        injected_dose_bq=injected_dose_bq,
        decay_correction=decay_correction,
        half_life_s=half_life_s,
        injection_datetime=injection_datetime,
        acquisition_datetime=acquisition_datetime,
        frame_duration_s=frame_duration_s,
        frame_reference_time_s=frame_reference_time_s,
    )
    suv_scale = activity_scale * patient_weight_kg * 1000.0 / decayed_dose_bq
    return _as_float_image(image) * suv_scale


def n4_bias_field_correction(
    image: sitk.Image,
    mask: sitk.Image | None = None,
    n_fitting_levels: int = 3,
    max_iterations: int | list[int] | tuple[int, ...] | None = None,
    convergence_threshold: float = 0.001,
    shrink_factor: int = 1,
) -> sitk.Image:
    """
    Apply N4 bias field correction to an MR image.

    When no mask is provided, an Otsu foreground mask is created, matching the
    common SimpleITK N4 workflow. MIRP uses a tissue mask when available.
    """
    if n_fitting_levels <= 0:
        raise ValueError("n_fitting_levels must be positive")
    if shrink_factor <= 0:
        raise ValueError("shrink_factor must be positive")

    input_image = _as_float_image(image)
    if mask is None:
        input_mask = sitk.OtsuThreshold(input_image, 0, 1, 200)
    else:
        input_mask = sitk.Cast(mask > 0, sitk.sitkUInt8)
        input_mask.CopyInformation(mask)

    if max_iterations is None:
        iterations = [50] * int(n_fitting_levels)
    elif isinstance(max_iterations, int):
        iterations = [int(max_iterations)] * int(n_fitting_levels)
    else:
        iterations = [int(value) for value in max_iterations]
        if len(iterations) != int(n_fitting_levels):
            raise ValueError("max_iterations length must match n_fitting_levels")

    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations(iterations)
    corrector.SetConvergenceThreshold(float(convergence_threshold))

    if shrink_factor == 1:
        return corrector.Execute(input_image, input_mask)

    shrunk_image = sitk.Shrink(input_image, [int(shrink_factor)] * input_image.GetDimension())
    shrunk_mask = sitk.Shrink(input_mask, [int(shrink_factor)] * input_mask.GetDimension())
    corrector.Execute(shrunk_image, shrunk_mask)
    log_bias = corrector.GetLogBiasFieldAsImage(input_image)
    return input_image / sitk.Exp(log_bias)


def normalize_mr_by_reference_mask(
    image: sitk.Image,
    reference_mask: sitk.Image,
    target_value: float = 1.0,
    statistic: Literal["median", "mean"] = "median",
) -> sitk.Image:
    """
    Scale MR intensities by a reference tissue mask, e.g. subcutaneous fat.
    """
    image_array = sitk.GetArrayFromImage(image).astype(np.float64, copy=False)
    mask_array = sitk.GetArrayFromImage(reference_mask) > 0
    values = image_array[mask_array]
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("reference_mask does not contain finite image values")

    if statistic == "median":
        reference_value = float(np.median(values))
    elif statistic == "mean":
        reference_value = float(np.mean(values))
    else:
        raise ValueError("statistic must be median or mean")

    if reference_value == 0.0:
        raise ValueError("reference tissue statistic is zero")

    return _as_float_image(image) * (float(target_value) / reference_value)


def isolate_mask_label(
    mask: sitk.Image,
    label: int | float,
    output_label: int = 1,
    pixel_type: int = sitk.sitkUInt8,
) -> sitk.Image:
    """
    Isolate one label from a label map, such as a converted GTV_Mass contour.
    """
    isolated = sitk.Cast(mask == label, pixel_type) * int(output_label)
    isolated.CopyInformation(mask)
    return isolated


def crop_image_and_mask_around_mask(
    image: sitk.Image,
    mask: sitk.Image,
    label: int | float = 1,
    margin_mm: float = 50.0,
) -> tuple[sitk.Image, sitk.Image]:
    """
    Crop image and mask to a physical margin around the selected mask label.
    """
    if margin_mm < 0.0:
        raise ValueError("margin_mm must be non-negative")

    label_stats = sitk.LabelShapeStatisticsImageFilter()
    label_stats.Execute(mask)
    if label not in label_stats.GetLabels():
        raise ValueError(f"label {label} is not present in mask")

    dimension = mask.GetDimension()
    bounding_box = np.array(label_stats.GetBoundingBox(label), dtype=np.int64)
    lower = bounding_box[:dimension]
    size = bounding_box[dimension:]
    upper = lower + size - 1
    spacing = np.array(mask.GetSpacing(), dtype=np.float64)
    margin_voxels = np.ceil(float(margin_mm) / spacing).astype(np.int64)

    crop_lower = np.maximum(lower - margin_voxels, 0)
    crop_upper_index = np.minimum(upper + margin_voxels, np.array(mask.GetSize()) - 1)
    crop_size = crop_upper_index - crop_lower + 1

    roi = sitk.RegionOfInterestImageFilter()
    roi.SetIndex([int(value) for value in crop_lower])
    roi.SetSize([int(value) for value in crop_size])
    return roi.Execute(image), roi.Execute(mask)


def load_dicom_series(directory: str | os.PathLike[str]) -> sitk.Image:
    """
    Load a DICOM image series from a directory using SimpleITK.
    """
    directory = str(directory)
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(directory)
    if not series_ids:
        raise ValueError(f"no DICOM series found in {directory}")
    file_names = reader.GetGDCMSeriesFileNames(directory, series_ids[0])
    reader.SetFileNames(file_names)
    return reader.Execute()


def write_nifti_pair(
    image: sitk.Image,
    mask: sitk.Image,
    image_path: str | os.PathLike[str],
    mask_path: str | os.PathLike[str],
) -> tuple[Path, Path]:
    """
    Write image and mask volumes to NIfTI files.
    """
    image_path = Path(image_path)
    mask_path = Path(mask_path)
    image_path.parent.mkdir(parents=True, exist_ok=True)
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(image, str(image_path))
    sitk.WriteImage(mask, str(mask_path))
    return image_path, mask_path


def clip_image_intensity(
    image: sitk.Image,
    lower_bound: float,
    upper_bound: float,
) -> sitk.Image:
    """
    Clip image intensities to the specified lower and upper bounds.
    Retains the original physical voxel values (e.g. HU) but caps outliers.
    """
    return sitk.Clamp(image, sitk.sitkFloat64, float(lower_bound), float(upper_bound))


def normalize_image_zscore_roi(
    image: sitk.Image,
    mask: sitk.Image,
    label: int = 1,
    clip_sigma: float | None = 3.0,
    scale: float = 100.0,
    shift: float = 300.0,
) -> sitk.Image:
    """
    Normalize image intensity by calculating the mean and std within the ROI mask,
    z-score normalizing the entire image, clipping outliers to +/- clip_sigma (optional),
    and rescaling/shifting to a target positive range.
    """
    image_arr = sitk.GetArrayFromImage(image).astype(np.float64)
    mask_arr = sitk.GetArrayFromImage(mask) == label
    if not np.any(mask_arr):
        mask_arr = sitk.GetArrayFromImage(mask) > 0

    roi_pixels = image_arr[mask_arr]
    roi_pixels = roi_pixels[np.isfinite(roi_pixels)]
    if roi_pixels.size == 0:
        raise ValueError(
            f"ROI mask (label={label} or any positive value) does not contain "
            "any valid finite pixels in the image."
        )

    mean = np.mean(roi_pixels)
    std = np.std(roi_pixels)
    if std == 0.0:
        std = 1.0

    normalized = (image_arr - mean) / std
    if clip_sigma is not None and clip_sigma > 0.0:
        normalized = np.clip(normalized, -float(clip_sigma), float(clip_sigma))

    scaled = normalized * float(scale) + float(shift)

    out_image = sitk.GetImageFromArray(scaled)
    out_image.CopyInformation(image)
    return out_image


__all__ = [
    "clip_image_intensity",
    "compute_pet_decayed_dose",
    "convert_pet_counts_to_suv",
    "crop_image_and_mask_around_mask",
    "isolate_mask_label",
    "load_dicom_series",
    "n4_bias_field_correction",
    "normalize_image_zscore_roi",
    "normalize_mr_by_reference_mask",
    "write_nifti_pair",
]

