from __future__ import annotations

import collections
import concurrent.futures
import ctypes
from dataclasses import dataclass
import logging
import os
import threading
import time

import numpy as np
import SimpleITK as sitk

from . import _bindings

logger = logging.getLogger(__name__)

_NATIVE_VOXEL_FEATURE_CLASSES = frozenset(
    {"firstorder", "glcm", "glrlm", "glszm", "gldm", "ngtdm"}
)

_FIRSTORDER_ALL_FEATURES = (
    "Mean",
    "Variance",
    "StandardDeviation",
    "Skewness",
    "Kurtosis",
    "Minimum",
    "Maximum",
    "Range",
    "Median",
    "10Percentile",
    "90Percentile",
    "InterquartileRange",
    "Energy",
    "TotalEnergy",
    "Entropy",
    "RootMeanSquared",
    "Uniformity",
    "MeanAbsoluteDeviation",
    "RobustMeanAbsoluteDeviation",
    "MedianAbsoluteDeviation",
    "CoefficientOfVariation",
    "QuartileCoefficientOfDispersion",
)

_FIRSTORDER_DEFAULT_FEATURES = tuple(
    feature for feature in _FIRSTORDER_ALL_FEATURES if feature != "StandardDeviation"
)

# Features natively emitted by the C/CUDA batch kernel.
_GLCM_NATIVE_BATCH_FEATURE_ORDER = (
    "Autocorrelation",
    "JointAverage",
    "ClusterProminence",
    "ClusterShade",
    "ClusterTendency",
    "Contrast",
    "Correlation",
    "DifferenceAverage",
    "DifferenceEntropy",
    "DifferenceVariance",
    "JointEnergy",
    "JointEntropy",
    "Imc1",
    "Imc2",
    "Idm",
    "MCC",
    "Idmn",
    "Id",
    "Idn",
    "InverseVariance",
    "MaximumProbability",
    "SumAverage",
    "SumEntropy",
    "SumSquares",
)
# Full order including Python-derived supplements (Dissimilarity, SumVariance).
_GLCM_BATCH_FEATURE_ORDER = _GLCM_NATIVE_BATCH_FEATURE_ORDER + (
    "Dissimilarity",
    "SumVariance",
)
_GLCM_BATCH_FEATURE_INDEX = {
    feature_name: index
    for index, feature_name in enumerate(_GLCM_BATCH_FEATURE_ORDER)
}
_GLCM_PYTHON_SUPPLEMENTAL_FEATURES = frozenset({"Dissimilarity", "SumVariance"})

_GLRLM_BATCH_FEATURE_ORDER = (
    "ShortRunEmphasis",
    "LongRunEmphasis",
    "GrayLevelNonUniformity",
    "GrayLevelNonUniformityNormalized",
    "RunLengthNonUniformity",
    "RunLengthNonUniformityNormalized",
    "RunPercentage",
    "GrayLevelVariance",
    "RunVariance",
    "RunEntropy",
    "LowGrayLevelRunEmphasis",
    "HighGrayLevelRunEmphasis",
    "ShortRunLowGrayLevelEmphasis",
    "ShortRunHighGrayLevelEmphasis",
    "LongRunLowGrayLevelEmphasis",
    "LongRunHighGrayLevelEmphasis",
)
_GLRLM_BATCH_FEATURE_INDEX = {
    feature_name: index
    for index, feature_name in enumerate(_GLRLM_BATCH_FEATURE_ORDER)
}

_GLSZM_BATCH_FEATURE_ORDER = (
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
    "ZoneVariance",
    "ZoneEntropy",
)
_GLSZM_BATCH_FEATURE_INDEX = {
    feature_name: index
    for index, feature_name in enumerate(_GLSZM_BATCH_FEATURE_ORDER)
}

# Features natively emitted by the C/CUDA batch kernel.
_GLDM_NATIVE_BATCH_FEATURE_ORDER = (
    "SmallDependenceEmphasis",
    "LargeDependenceEmphasis",
    "GrayLevelNonUniformity",
    "DependenceNonUniformity",
    "DependenceNonUniformityNormalized",
    "GrayLevelVariance",
    "DependenceVariance",
    "DependenceEntropy",
    "LowGrayLevelEmphasis",
    "HighGrayLevelEmphasis",
    "SmallDependenceLowGrayLevelEmphasis",
    "SmallDependenceHighGrayLevelEmphasis",
    "LargeDependenceLowGrayLevelEmphasis",
    "LargeDependenceHighGrayLevelEmphasis",
)
# Full order including Python-derived supplements.
_GLDM_BATCH_FEATURE_ORDER = _GLDM_NATIVE_BATCH_FEATURE_ORDER + (
    "DependenceCountPercentage",
    "DependenceCountEnergy",
    "GrayLevelNonUniformityNormalized",
)
_GLDM_BATCH_FEATURE_INDEX = {
    feature_name: index
    for index, feature_name in enumerate(_GLDM_BATCH_FEATURE_ORDER)
}
_GLDM_PYTHON_SUPPLEMENTAL_FEATURES = frozenset(
    {"DependenceCountPercentage", "DependenceCountEnergy", "GrayLevelNonUniformityNormalized"}
)

_NGTDM_BATCH_FEATURE_ORDER = (
    "Coarseness",
    "Contrast",
    "Busyness",
    "Complexity",
    "Strength",
)
_NGTDM_BATCH_FEATURE_INDEX = {
    feature_name: index
    for index, feature_name in enumerate(_NGTDM_BATCH_FEATURE_ORDER)
}

_TEXTURE_BATCH_FEATURE_ORDERS = {
    "glcm": _GLCM_BATCH_FEATURE_ORDER,
    "glrlm": _GLRLM_BATCH_FEATURE_ORDER,
    "glszm": _GLSZM_BATCH_FEATURE_ORDER,
    "gldm": _GLDM_BATCH_FEATURE_ORDER,
    "ngtdm": _NGTDM_BATCH_FEATURE_ORDER,
}


@dataclass(frozen=True)
class VoxelContext:
    image_array: np.ndarray
    roi_mask: np.ndarray
    discretized: np.ndarray | None
    spacing: tuple[float, ...]
    origin: tuple[float, ...]
    label: int
    bin_width: float
    bin_count: int
    kernel_radius: int
    kernel_shape: tuple[int, ...]
    masked_kernel: bool
    force2d: bool
    force2d_dimension: int
    init_value: float
    ng: int
    roi_coordinates: np.ndarray
    window_starts: np.ndarray
    image_windows: np.ndarray | None
    discretized_windows: np.ndarray | None
    mask_windows: np.ndarray | None
    batch_size: int
    backend: str
    num_threads: int
    shared_state: dict[str, object]


_VOXEL_TIMING_STAGE_NAMES = (
    "gather_prep_s",
    "native_call_s",
    "output_assign_s",
    "masked_value_prep_s",
    "order_stats_s",
    "moment_s",
    "percentile_s",
    "minmax_s",
    "median_s",
    "energy_s",
    "histogram_s",
    "robust_mad_s",
)


def _resolve_voxel_batch_fn(
    context: VoxelContext,
    symbol_names: str | tuple[str, ...],
):
    if isinstance(symbol_names, str):
        candidate_symbols = (symbol_names,)
    else:
        candidate_symbols = tuple(symbol_names)

    requested_backend = (context.backend or "cpu").lower()
    preferred_backend = requested_backend
    strict_backend = requested_backend in {"cuda", "gpu", "mps"}
    if preferred_backend == "auto":
        try:
            available = _bindings.detect_backends()
            if available.get("cuda", False):
                preferred_backend = "cuda"
            elif available.get("mps", False):
                preferred_backend = "mps"
            else:
                preferred_backend = "cpu"
        except Exception:  # pragma: no cover - defensive backend fallback
            preferred_backend = "cpu"

    if preferred_backend in {"cuda", "gpu"}:
        try:
            cuda_lib = _bindings.get_feature_library("cuda")
            for symbol_name in candidate_symbols:
                cuda_fn = getattr(cuda_lib, symbol_name, None)
                if cuda_fn is not None:
                    return cuda_fn, "CUDA", getattr(_bindings, "_lib_cuda_path", "<unknown>")
            logger.warning(
                "None of voxel batch symbols %s are available in CUDA library; falling back.",
                candidate_symbols,
            )
            if strict_backend:
                return None, "CUDA", getattr(_bindings, "_lib_cuda_path", "<unknown>")
        except Exception as exc:  # pragma: no cover - defensive backend fallback
            logger.warning(
                "Failed to resolve CUDA voxel batch symbols %s: %s. Falling back.",
                candidate_symbols,
                exc,
            )
            if strict_backend:
                return None, "CUDA", getattr(_bindings, "_lib_cuda_path", "<unknown>")

    if preferred_backend == "mps":
        try:
            mps_lib = _bindings.get_feature_library("mps")
            for symbol_name in candidate_symbols:
                mps_fn = getattr(mps_lib, symbol_name, None)
                if mps_fn is not None:
                    return mps_fn, "MPS", getattr(_bindings, "_lib_mps_path", "<unknown>")
            logger.warning(
                "None of voxel batch symbols %s are available in MPS library; falling back to CPU.",
                candidate_symbols,
            )
            if strict_backend:
                return None, "MPS", getattr(_bindings, "_lib_mps_path", "<unknown>")
        except Exception as exc:  # pragma: no cover - defensive backend fallback
            logger.warning(
                "Failed to resolve MPS voxel batch symbols %s: %s. Falling back to CPU.",
                candidate_symbols,
                exc,
            )
            if strict_backend:
                return None, "MPS", getattr(_bindings, "_lib_mps_path", "<unknown>")

    cpu_lib = _bindings.get_feature_library("cpu")
    cpu_fn = None
    for symbol_name in candidate_symbols:
        cpu_fn = getattr(cpu_lib, symbol_name, None)
        if cpu_fn is not None:
            break
    return (
        cpu_fn,
        "CPU",
        getattr(_bindings, "_lib_cpu_path", "<unknown>"),
    )


def compute_native_voxel_features(extractor, image, mask, image_type_name, **kwargs):
    feature_vector = collections.OrderedDict()
    enabled_items = [
        (feature_class_name, feature_names)
        for feature_class_name, feature_names in extractor.enabledFeatures.items()
        if not feature_class_name.startswith("shape")
        and feature_class_name in _NATIVE_VOXEL_FEATURE_CLASSES
    ]
    unsupported = [
        feature_class_name
        for feature_class_name in extractor.enabledFeatures
        if not feature_class_name.startswith("shape")
        and feature_class_name not in _NATIVE_VOXEL_FEATURE_CLASSES
    ]
    if unsupported:
        raise RuntimeError(
            "Unsupported voxel feature classes without native Flash batch "
            f"implementations: {', '.join(sorted(set(unsupported)))}"
        )
    if not enabled_items:
        return feature_vector

    resolved_feature_names = {}
    need_firstorder = False
    need_firstorder_histogram = False
    need_texture = False
    for feature_class_name, feature_names in enabled_items:
        if feature_class_name == "firstorder":
            need_firstorder = True
            resolved = _resolve_firstorder_features(feature_names)
            resolved_feature_names[feature_class_name] = resolved
            need_firstorder_histogram = bool(
                set(resolved) & {"Entropy", "Uniformity"}
            )
        else:
            resolved_feature_names[feature_class_name] = _resolve_texture_features(
                feature_class_name,
                feature_names,
            )
            need_texture = True

    context = _build_voxel_context(
        extractor,
        image,
        mask,
        need_discretized=(need_texture or need_firstorder_histogram),
        need_image_windows=need_firstorder,
        need_discretized_windows=(need_texture or need_firstorder_histogram),
        need_mask_windows=(need_firstorder or need_texture),
        **kwargs,
    )
    telemetry = _get_voxel_telemetry(context)
    if context.roi_coordinates.size == 0:
        telemetry["class_workers"] = 0
        telemetry["class_order"] = tuple(feature_class_name for feature_class_name, _ in enabled_items)
        extractor._last_voxel_telemetry = telemetry
        if _voxel_timing_enabled():
            logger.info("Voxel telemetry: %s", telemetry)
        return feature_vector

    per_class_results = {}
    remaining = []
    for feature_class_name, _feature_names in enabled_items:
        if feature_class_name == "firstorder":
            try:
                per_class_results[feature_class_name] = _compute_firstorder_feature_maps(
                    context=context,
                    image=image,
                    image_type_name=image_type_name,
                    feature_names=resolved_feature_names[feature_class_name],
                    kwargs=kwargs,
                )
            except Exception as exc:
                logger.error(
                    "FAILED voxel feature class '%s': %s",
                    feature_class_name,
                    exc,
                    exc_info=True,
                )
                per_class_results[feature_class_name] = _build_failed_feature_maps(
                    context=context,
                    image=image,
                    image_type_name=image_type_name,
                    feature_class_name=feature_class_name,
                    feature_names=resolved_feature_names[feature_class_name],
                )
        else:
            remaining.append(feature_class_name)

    class_workers = _get_class_workers(context, len(remaining))
    telemetry["class_workers"] = class_workers
    telemetry["class_order"] = tuple(feature_class_name for feature_class_name, _ in enabled_items)

    cuda_unified_handled = False
    if remaining and _is_cuda_unified_dispatch_available(context):
        try:
            per_class_results.update(
                _compute_texture_feature_maps_multi_class_cuda(
                    context=context,
                    image=image,
                    image_type_name=image_type_name,
                    texture_classes=remaining,
                    resolved_feature_names=resolved_feature_names,
                    kwargs=kwargs,
                )
            )
            cuda_unified_handled = True
            telemetry["cuda_unified_dispatch"] = True
        except Exception:
            logger.debug(
                "Unified CUDA dispatch failed, falling back to per-class",
                exc_info=True,
            )

    if not cuda_unified_handled:
        if class_workers > 1 and len(remaining) > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=class_workers) as executor:
                futures = {
                    executor.submit(
                        _compute_texture_feature_maps,
                        context,
                        image,
                        image_type_name,
                        feature_class_name,
                        resolved_feature_names[feature_class_name],
                        kwargs,
                    ): feature_class_name
                    for feature_class_name in remaining
                }
                for future in concurrent.futures.as_completed(futures):
                    feature_class_name = futures[future]
                    try:
                        per_class_results[feature_class_name] = future.result()
                    except Exception as exc:
                        logger.error(
                            "FAILED voxel feature class '%s': %s",
                            feature_class_name,
                            exc,
                            exc_info=True,
                        )
                        per_class_results[feature_class_name] = _build_failed_feature_maps(
                            context=context,
                            image=image,
                            image_type_name=image_type_name,
                            feature_class_name=feature_class_name,
                            feature_names=resolved_feature_names[feature_class_name],
                        )
        else:
            for feature_class_name in remaining:
                try:
                    per_class_results[feature_class_name] = _compute_texture_feature_maps(
                        context=context,
                        image=image,
                        image_type_name=image_type_name,
                        feature_class_name=feature_class_name,
                        feature_names=resolved_feature_names[feature_class_name],
                        kwargs=kwargs,
                    )
                except Exception as exc:
                    logger.error(
                        "FAILED voxel feature class '%s': %s",
                        feature_class_name,
                        exc,
                        exc_info=True,
                    )
                    per_class_results[feature_class_name] = _build_failed_feature_maps(
                        context=context,
                        image=image,
                        image_type_name=image_type_name,
                        feature_class_name=feature_class_name,
                        feature_names=resolved_feature_names[feature_class_name],
                    )

    for feature_class_name, _feature_names in enabled_items:
        feature_vector.update(per_class_results.get(feature_class_name, {}))
    extractor._last_voxel_telemetry = telemetry
    if _voxel_timing_enabled():
        logger.info("Voxel telemetry: %s", telemetry)
    return feature_vector


def _build_voxel_context(
    extractor,
    image,
    mask,
    *,
    need_discretized=True,
    need_image_windows=True,
    need_discretized_windows=True,
    need_mask_windows=True,
    **kwargs,
) -> VoxelContext:
    context_t0 = time.perf_counter()
    label = int(kwargs.get("label", 1))
    image_array, mask_array, spacing, origin = extractor._to_numpy(image, mask, label)
    roi_mask = np.asarray(mask_array) != 0

    kernel_radius = int(kwargs.get("kernelRadius", 1))
    masked_kernel = bool(kwargs.get("maskedKernel", True))
    force2d = bool(kwargs.get("force2D", False))
    force2d_dimension = int(kwargs.get("force2Ddimension", 0))
    init_value = float(kwargs.get("initValue", 0.0))
    batch_size = int(kwargs.get("voxelBatch", -1))
    if batch_size <= 0:
        batch_size = extractor._get_default_voxel_batch(getattr(extractor, "_backend", "cpu"))

    if need_discretized_windows and not need_discretized:
        raise RuntimeError("discretized windows requested without discretized input")

    discretized_array = None
    ng = 0
    discretize_elapsed_s = 0.0
    if need_discretized:
        discretize_t0 = time.perf_counter()
        if masked_kernel:
            discretization_mask = roi_mask.astype(np.uint8, copy=False)
        else:
            discretization_mask = np.ones(image_array.shape, dtype=np.uint8)
        discretized, ng = extractor._get_discretized_buffer(
            image_array,
            discretization_mask,
            spacing,
            origin,
            label,
            float(kwargs.get("binWidth", 25)),
            int(kwargs.get("binCount", 0) or 0),
        )
        discretized_array = np.asarray(discretized, dtype=np.int32)
        discretize_elapsed_s = float(time.perf_counter() - discretize_t0)

    kernel_shape = _get_kernel_shape(
        ndim=image_array.ndim,
        kernel_radius=kernel_radius,
        force2d=force2d,
        force2d_dimension=force2d_dimension,
    )
    pad_widths = [
        (kernel_radius, kernel_radius)
        if kernel_dim > 1
        else (0, 0)
        for kernel_dim in kernel_shape
    ]

    roi_coordinates = np.argwhere(roi_mask).astype(np.intp, copy=False)
    image_windows = None
    discretized_windows = None
    mask_windows = None
    window_prep_t0 = time.perf_counter()
    if roi_coordinates.size == 0:
        empty = np.empty((0, image_array.ndim), dtype=np.intp)
        window_starts = empty
    else:
        if need_image_windows:
            padded_image = np.pad(
                np.asarray(image_array, dtype=np.float64),
                pad_widths,
                mode="constant",
                constant_values=0.0,
            )
            image_windows = np.lib.stride_tricks.sliding_window_view(
                padded_image,
                kernel_shape,
            )

        if need_discretized_windows:
            if discretized_array is None:
                raise RuntimeError("discretized windows requested with empty discretized")
            padded_discretized = np.pad(
                discretized_array,
                pad_widths,
                mode="constant",
                constant_values=0,
            )
            discretized_windows = np.lib.stride_tricks.sliding_window_view(
                padded_discretized,
                kernel_shape,
            )

        if need_mask_windows:
            kernel_mask_source = (
                roi_mask
                if masked_kernel
                else np.ones(image_array.shape, dtype=bool)
            )
            padded_mask = np.pad(
                np.asarray(kernel_mask_source, dtype=bool),
                pad_widths,
                mode="constant",
                constant_values=False,
            )
            mask_windows = np.lib.stride_tricks.sliding_window_view(
                padded_mask,
                kernel_shape,
            )
        window_starts = roi_coordinates

    telemetry = collections.OrderedDict(
        context_build_s=0.0,
        discretize_s=discretize_elapsed_s,
        window_prep_s=float(time.perf_counter() - window_prep_t0),
        roi_voxels=int(roi_coordinates.shape[0]),
        roi_size_bin=_classify_roi_size(int(roi_coordinates.shape[0])),
        batch_size=int(batch_size),
        kernel_shape=tuple(int(dim) for dim in kernel_shape),
        window_voxels=int(np.multiply.reduce(np.asarray(kernel_shape, dtype=np.int64))),
        ng=int(ng),
        backend=str(getattr(extractor, "_backend", "cpu")),
        num_threads=int(getattr(extractor, "_num_threads", 1)),
        context_static_bytes=int(
            np.asarray(image_array, dtype=np.float64).nbytes
            + roi_mask.nbytes
            + (0 if discretized_array is None else discretized_array.nbytes)
        ),
        max_batch_bytes=0,
        cache_hits=0,
        cache_misses=0,
        class_workers=0,
        class_order=tuple(),
        classes=collections.OrderedDict(),
    )
    telemetry["context_build_s"] = float(time.perf_counter() - context_t0)

    return VoxelContext(
        image_array=np.asarray(image_array, dtype=np.float64),
        roi_mask=roi_mask,
        discretized=discretized_array,
        spacing=tuple(spacing),
        origin=tuple(origin),
        label=label,
        bin_width=float(kwargs.get("binWidth", 25)),
        bin_count=int(kwargs.get("binCount", 0) or 0),
        kernel_radius=kernel_radius,
        kernel_shape=kernel_shape,
        masked_kernel=masked_kernel,
        force2d=force2d,
        force2d_dimension=force2d_dimension,
        init_value=init_value,
        ng=int(ng),
        roi_coordinates=roi_coordinates,
        window_starts=window_starts,
        image_windows=image_windows,
        discretized_windows=discretized_windows,
        mask_windows=mask_windows,
        batch_size=batch_size,
        backend=getattr(extractor, "_backend", "cpu"),
        num_threads=int(getattr(extractor, "_num_threads", 1)),
        shared_state={
            "batch_cache": {},
            "batch_cache_lock": threading.Lock(),
            "include_flags_cache": {},
            "telemetry": telemetry,
            "telemetry_lock": threading.Lock(),
        },
    )


def _compute_firstorder_feature_maps(
    context: VoxelContext,
    image: sitk.Image,
    image_type_name: str,
    feature_names,
    kwargs,
):
    selected_features = _resolve_firstorder_features(feature_names)
    if not selected_features:
        return collections.OrderedDict()

    selected_feature_set = set(selected_features)
    output_arrays = collections.OrderedDict(
        (
            f"{image_type_name}_firstorder_{feature_name}",
            np.full(context.image_array.shape, context.init_value, dtype=np.float64),
        )
        for feature_name in selected_features
    )

    voxel_array_shift = float(kwargs.get("voxelArrayShift", 0))
    voxel_volume = float(np.multiply.reduce(np.asarray(context.spacing, dtype=np.float64)))
    need_mean = bool(
        selected_feature_set
        & {
            "Mean",
            "Variance",
            "StandardDeviation",
            "Skewness",
            "Kurtosis",
            "MeanAbsoluteDeviation",
            "CoefficientOfVariation",
        }
    )
    need_centered = bool(
        selected_feature_set
        & {
            "Variance",
            "StandardDeviation",
            "Skewness",
            "Kurtosis",
            "MeanAbsoluteDeviation",
            "CoefficientOfVariation",
        }
    )
    need_m3 = "Skewness" in selected_feature_set
    need_m4 = "Kurtosis" in selected_feature_set
    need_order_stats = bool(
        selected_feature_set
        & {
            "Minimum",
            "Maximum",
            "Range",
            "Median",
            "10Percentile",
            "90Percentile",
            "InterquartileRange",
            "RobustMeanAbsoluteDeviation",
            "QuartileCoefficientOfDispersion",
            "MedianAbsoluteDeviation",
        }
    )
    need_percentiles = bool(
        selected_feature_set
        & {
            "10Percentile",
            "90Percentile",
            "InterquartileRange",
            "RobustMeanAbsoluteDeviation",
            "QuartileCoefficientOfDispersion",
        }
    )
    need_minmax = bool(selected_feature_set & {"Minimum", "Maximum", "Range"})
    need_median = bool(selected_feature_set & {"Median", "MedianAbsoluteDeviation"})
    need_shifted_energy = bool(
        selected_feature_set & {"Energy", "TotalEnergy", "RootMeanSquared"}
    )
    need_histogram = bool(selected_feature_set & {"Entropy", "Uniformity"})
    if context.image_windows is None or context.mask_windows is None:
        raise RuntimeError("Firstorder voxel path requires native image and mask windows")
    if need_histogram and context.discretized_windows is None:
        raise RuntimeError("Histogram firstorder voxel features require discretized windows")

    for batch_slice in _iter_batch_slices(
        total=context.roi_coordinates.shape[0],
        batch_size=context.batch_size,
    ):
        gather_t0 = time.perf_counter()
        coords = context.roi_coordinates[batch_slice]
        coord_key = tuple(coords[:, axis] for axis in range(coords.shape[1]))
        batch_arrays, cache_hit = _get_batch_patch_cache(
            context,
            batch_slice,
            need_image=True,
            need_discretized=need_histogram,
            need_mask=True,
        )
        image_patches = batch_arrays["image_patches"]
        mask_patches = batch_arrays["mask_patches_bool"]
        discretized_patches = batch_arrays.get("discretized_patches")
        gather_elapsed_s = float(time.perf_counter() - gather_t0)
        compute_t0 = time.perf_counter()

        counts = mask_patches.sum(axis=1).astype(np.float64)
        counts[counts == 0] = 1.0

        feature_values = {}
        extra_metrics = {}

        if need_mean or need_centered:
            moment_t0 = time.perf_counter()
            masked_numeric = np.where(mask_patches, image_patches, 0.0)
            mean = masked_numeric.sum(axis=1) / counts
            if "Mean" in selected_feature_set:
                feature_values["Mean"] = mean
        else:
            mean = None

        if need_centered:
            centered = np.where(mask_patches, image_patches - mean[:, None], 0.0)
            centered_sq = centered * centered
            m2 = centered_sq.sum(axis=1) / counts

            if "Variance" in selected_feature_set:
                feature_values["Variance"] = m2
            if "StandardDeviation" in selected_feature_set:
                feature_values["StandardDeviation"] = np.sqrt(m2)
            if need_m3:
                m3 = (centered_sq * centered).sum(axis=1) / counts
                skewness = np.zeros_like(m2, dtype=np.float64)
                np.divide(
                    m3,
                    np.power(m2, 1.5),
                    out=skewness,
                    where=m2 > 0.0,
                )
                feature_values["Skewness"] = skewness
            if need_m4:
                m4 = (centered_sq * centered_sq).sum(axis=1) / counts
                kurtosis = np.zeros_like(m2, dtype=np.float64)
                np.divide(
                    m4,
                    np.square(m2),
                    out=kurtosis,
                    where=m2 > 0.0,
                )
                kurtosis[m2 > 0.0] -= 3.0
                feature_values["Kurtosis"] = kurtosis
            if "MeanAbsoluteDeviation" in selected_feature_set:
                feature_values["MeanAbsoluteDeviation"] = (
                    np.abs(centered).sum(axis=1) / counts
                )
            if "CoefficientOfVariation" in selected_feature_set:
                std_values = np.sqrt(m2)
                cov = np.zeros_like(m2, dtype=np.float64)
                with np.errstate(divide="ignore", invalid="ignore"):
                    np.divide(std_values, mean, out=cov)
                feature_values["CoefficientOfVariation"] = cov
        if need_mean or need_centered:
            extra_metrics["moment_s"] = float(time.perf_counter() - moment_t0)

        order_stats = None
        if need_order_stats:
            order_stats_t0 = time.perf_counter()
            order_stats = _compute_batch_order_stats(
                image_patches=image_patches,
                valid_mask=mask_patches,
                need_percentiles=need_percentiles,
                need_minmax=need_minmax,
                need_median=need_median,
                need_robust_mad="RobustMeanAbsoluteDeviation" in selected_feature_set,
            )
            extra_metrics["order_stats_s"] = float(time.perf_counter() - order_stats_t0)

        if need_percentiles:
            percentile_t0 = time.perf_counter()
            if "10Percentile" in selected_feature_set:
                feature_values["10Percentile"] = order_stats["percentile10"]
            if "90Percentile" in selected_feature_set:
                feature_values["90Percentile"] = order_stats["percentile90"]
            if "InterquartileRange" in selected_feature_set:
                iqr = order_stats["percentile75"] - order_stats["percentile25"]
                feature_values["InterquartileRange"] = iqr
            if "QuartileCoefficientOfDispersion" in selected_feature_set:
                iqr = order_stats["percentile75"] - order_stats["percentile25"]
                denom = order_stats["percentile75"] + order_stats["percentile25"]
                qcod = np.ones_like(iqr, dtype=np.float64)
                np.divide(iqr, denom, out=qcod, where=denom != 0.0)
                feature_values["QuartileCoefficientOfDispersion"] = qcod
            extra_metrics["percentile_s"] = float(time.perf_counter() - percentile_t0)

        if need_minmax:
            minmax_t0 = time.perf_counter()
            min_values = order_stats["minimum"]
            max_values = order_stats["maximum"]
            if "Minimum" in selected_feature_set:
                feature_values["Minimum"] = min_values
            if "Maximum" in selected_feature_set:
                feature_values["Maximum"] = max_values
            if "Range" in selected_feature_set:
                feature_values["Range"] = max_values - min_values
            extra_metrics["minmax_s"] = float(time.perf_counter() - minmax_t0)

        if need_median:
            median_t0 = time.perf_counter()
            if "Median" in selected_feature_set:
                feature_values["Median"] = order_stats["median"]
            if "MedianAbsoluteDeviation" in selected_feature_set:
                median_centered = np.where(
                    mask_patches,
                    np.abs(image_patches - order_stats["median"][:, None]),
                    0.0,
                )
                feature_values["MedianAbsoluteDeviation"] = median_centered.sum(axis=1) / counts
            extra_metrics["median_s"] = float(time.perf_counter() - median_t0)

        if need_shifted_energy:
            energy_t0 = time.perf_counter()
            shifted_values = np.where(
                mask_patches,
                image_patches + voxel_array_shift,
                0.0,
            )
            energy = np.sum(shifted_values * shifted_values, axis=1)
            if "Energy" in selected_feature_set:
                feature_values["Energy"] = energy
            if "TotalEnergy" in selected_feature_set:
                feature_values["TotalEnergy"] = energy * voxel_volume
            if "RootMeanSquared" in selected_feature_set:
                feature_values["RootMeanSquared"] = np.sqrt(energy / counts)
            extra_metrics["energy_s"] = float(time.perf_counter() - energy_t0)

        if need_histogram:
            histogram_t0 = time.perf_counter()
            hist_counts = _batch_histogram_counts(
                discretized_patches=discretized_patches,
                valid_mask=mask_patches,
                ng=context.ng,
            )
            hist_sums = hist_counts.sum(axis=1, keepdims=True).astype(np.float64)
            hist_sums[hist_sums == 0] = 1.0
            p_i = hist_counts.astype(np.float64) / hist_sums
            if "Entropy" in selected_feature_set:
                eps = np.spacing(1.0)
                feature_values["Entropy"] = -np.sum(
                    p_i * np.log2(p_i + eps),
                    axis=1,
                )
            if "Uniformity" in selected_feature_set:
                feature_values["Uniformity"] = np.sum(np.square(p_i), axis=1)
            extra_metrics["histogram_s"] = float(time.perf_counter() - histogram_t0)

        if "RobustMeanAbsoluteDeviation" in selected_feature_set:
            robust_mad_t0 = time.perf_counter()
            feature_values["RobustMeanAbsoluteDeviation"] = order_stats["robust_mad"]
            extra_metrics["robust_mad_s"] = float(time.perf_counter() - robust_mad_t0)

        compute_elapsed_s = float(time.perf_counter() - compute_t0)
        assign_t0 = time.perf_counter()
        for feature_name in selected_features:
            output_name = f"{image_type_name}_firstorder_{feature_name}"
            output_arrays[output_name][coord_key] = np.asarray(
                feature_values[feature_name],
                dtype=np.float64,
            )
        _record_voxel_batch_metrics(
            context,
            "firstorder",
            gather_prep_s=gather_elapsed_s,
            native_call_s=compute_elapsed_s,
            output_assign_s=float(time.perf_counter() - assign_t0),
            batch_bytes=_estimate_batch_bytes(batch_arrays),
            cache_hit=cache_hit,
            extra_metrics=extra_metrics,
        )

    return collections.OrderedDict(
        (
            output_name,
            _array_to_image(output_array, image),
        )
        for output_name, output_array in output_arrays.items()
    )


def _compute_texture_feature_maps(
    context: VoxelContext,
    image: sitk.Image,
    image_type_name: str,
    feature_class_name: str,
    feature_names,
    kwargs,
):
    requested_feature_names = tuple(feature_names) if feature_names is not None else None

    if feature_class_name == "glcm":
        return _compute_glcm_feature_maps_batch(
            context=context,
            image=image,
            image_type_name=image_type_name,
            feature_names=requested_feature_names,
            kwargs=kwargs,
        )

    if feature_class_name == "glrlm":
        return _compute_glrlm_feature_maps_batch(
            context=context,
            image=image,
            image_type_name=image_type_name,
            feature_names=requested_feature_names,
            kwargs=kwargs,
        )

    if feature_class_name == "glszm":
        return _compute_glszm_feature_maps_batch(
            context=context,
            image=image,
            image_type_name=image_type_name,
            feature_names=requested_feature_names,
            kwargs=kwargs,
        )

    if feature_class_name == "gldm":
        return _compute_gldm_feature_maps_batch(
            context=context,
            image=image,
            image_type_name=image_type_name,
            feature_names=requested_feature_names,
            kwargs=kwargs,
        )

    if feature_class_name == "ngtdm":
        return _compute_ngtdm_feature_maps_batch(
            context=context,
            image=image,
            image_type_name=image_type_name,
            feature_names=requested_feature_names,
            kwargs=kwargs,
        )

    raise RuntimeError(
        "Unsupported voxel texture class for native Flash batch execution: "
        f"{feature_class_name}"
    )


def _compute_glcm_feature_maps_batch(
    context: VoxelContext,
    image: sitk.Image,
    image_type_name: str,
    feature_names,
    kwargs,
):
    batch_fn, backend_label, lib_path = _resolve_voxel_batch_fn(
        context,
        (
            "flash_radiomics_glcm_cuda_discretized_voxel_batch",
            "flash_radiomics_glcm_cpu_discretized_voxel_batch",
        ),
    )
    if batch_fn is None:
        raise RuntimeError(
            f"GLCM voxel batch API is not available in the loaded {backend_label} "
            f"library ({lib_path}). Rebuild native libs from project root with: "
            "'cd flash_radiomics && mkdir -p build && cd build && "
            "cmake -DCMAKE_BUILD_TYPE=Release .. && cmake --build . -j4'"
        )

    requested_set = (
        set(feature_names)
        if feature_names is not None and len(feature_names) > 0
        else None
    )
    selected_feature_names = [
        name
        for name in _GLCM_BATCH_FEATURE_ORDER
        if requested_set is None or name in requested_set
    ]
    if not selected_feature_names:
        return collections.OrderedDict()

    if context.discretized_windows is None or context.mask_windows is None:
        raise RuntimeError("GLCM voxel batch path requires discretized and mask windows")

    # Split into native (C kernel) and supplemental (Python-derived) features.
    native_selected = [
        name for name in selected_feature_names
        if name not in _GLCM_PYTHON_SUPPLEMENTAL_FEATURES
    ]
    supplemental_selected = [
        name for name in selected_feature_names
        if name in _GLCM_PYTHON_SUPPLEMENTAL_FEATURES
    ]
    need_supplemental = len(supplemental_selected) > 0

    # Include flags follow the native GLCM feature order.
    native_feature_index = {
        name: idx for idx, name in enumerate(_GLCM_NATIVE_BATCH_FEATURE_ORDER)
    }
    native_include_flags = np.zeros(len(_GLCM_NATIVE_BATCH_FEATURE_ORDER), dtype=np.uint8)
    for name in native_selected:
        native_include_flags[native_feature_index[name]] = 1
    native_include_flags_ptr = native_include_flags.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))

    distances = np.asarray(kwargs.get("distances", [1]), dtype=np.int32).reshape(-1)
    if distances.size == 0:
        distances = np.asarray([1], dtype=np.int32)
    distances = np.ascontiguousarray(distances, dtype=np.int32)
    distance_ptr = distances.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

    if context.image_array.ndim == 2:
        window_dims = np.asarray(
            [context.kernel_shape[1], context.kernel_shape[0], 1],
            dtype=np.int32,
        )
    else:
        window_dims = np.asarray(
            [
                context.kernel_shape[2],
                context.kernel_shape[1],
                context.kernel_shape[0],
            ],
            dtype=np.int32,
        )
    window_dims_ptr = window_dims.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

    output_arrays = collections.OrderedDict(
        (
            f"{image_type_name}_glcm_{feature_name}",
            np.full(context.image_array.shape, context.init_value, dtype=np.float64),
        )
        for feature_name in selected_feature_names
    )

    native_feature_count = len(native_selected)
    native_batch_outputs = np.empty(
        (context.batch_size, native_feature_count),
        dtype=np.float64,
    ) if native_feature_count > 0 else None

    for batch_slice in _iter_batch_slices(
        total=context.roi_coordinates.shape[0],
        batch_size=context.batch_size,
    ):
        gather_t0 = time.perf_counter()
        coords = context.roi_coordinates[batch_slice]
        current_batch = int(coords.shape[0])
        coord_key = tuple(coords[:, axis] for axis in range(coords.shape[1]))
        batch_arrays, cache_hit = _get_batch_patch_cache(
            context,
            batch_slice,
            need_discretized=True,
            need_mask=True,
        )
        discretized_patches = batch_arrays["discretized_patches"]
        mask_patches = batch_arrays["mask_patches_u8"]
        gather_elapsed_s = float(time.perf_counter() - gather_t0)

        native_t0 = time.perf_counter()
        if native_batch_outputs is not None and native_feature_count > 0:
            native_view = native_batch_outputs[:current_batch, :]
            status = batch_fn(
                discretized_patches.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                mask_patches.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                current_batch,
                int(context.image_array.ndim),
                window_dims_ptr,
                distance_ptr,
                int(distances.size),
                native_include_flags_ptr,
                int(context.ng),
                native_view.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                int(native_view.shape[1]),
            )
            if status != int(_bindings.FlashRadiomicsErrorCode.SUCCESS):
                raise _bindings.FlashRadiomicsComputationError(
                    f"GLCM voxel batch call failed with error code {status}"
                )
        native_elapsed_s = float(time.perf_counter() - native_t0)

        supplemental_values = {}
        if need_supplemental:
            supplemental_values = _compute_glcm_supplemental_batch(
                discretized_patches=discretized_patches,
                mask_patches=mask_patches,
                ng=context.ng,
                distances=distances,
                ndim=context.image_array.ndim,
                window_dims=window_dims,
                supplemental_names=supplemental_selected,
            )

        assign_t0 = time.perf_counter()
        native_col = 0
        for feature_name in selected_feature_names:
            output_name = f"{image_type_name}_glcm_{feature_name}"
            if feature_name in _GLCM_PYTHON_SUPPLEMENTAL_FEATURES:
                output_arrays[output_name][coord_key] = supplemental_values[feature_name]
            else:
                output_arrays[output_name][coord_key] = native_batch_outputs[:current_batch, native_col]
                native_col += 1
        _record_voxel_batch_metrics(
            context,
            "glcm",
            gather_prep_s=gather_elapsed_s,
            native_call_s=native_elapsed_s,
            output_assign_s=float(time.perf_counter() - assign_t0),
            batch_bytes=_estimate_batch_bytes(batch_arrays) + (
                native_batch_outputs[:current_batch].nbytes if native_batch_outputs is not None else 0
            ),
            cache_hit=cache_hit,
            backend_label=backend_label,
        )

    return collections.OrderedDict(
        (
            output_name,
            _array_to_image(output_array, image),
        )
        for output_name, output_array in output_arrays.items()
    )


def _compute_glcm_supplemental_batch(
    *,
    discretized_patches: np.ndarray,
    mask_patches: np.ndarray,
    ng: int,
    distances: np.ndarray,
    ndim: int,
    window_dims: np.ndarray,
    supplemental_names: list[str],
) -> dict[str, np.ndarray]:
    """Compute Dissimilarity and/or SumVariance per voxel window.

    These IBSI features are not emitted by the native C/CUDA batch kernel
    and are computed in Python from the discretized patches.  The approach
    mirrors ``RadiomicsGLCM._compute_ibsi_redundant_features`` but is
    vectorised over the batch dimension.
    """
    batch_size = int(discretized_patches.shape[0])
    need_dissimilarity = "Dissimilarity" in supplemental_names
    need_sum_variance = "SumVariance" in supplemental_names

    dissimilarity_out = np.full(batch_size, np.nan, dtype=np.float64)
    sum_variance_out = np.full(batch_size, np.nan, dtype=np.float64)

    if ndim == 3:
        wz, wy, wx = int(window_dims[2]), int(window_dims[1]), int(window_dims[0])
    else:
        wy, wx = int(window_dims[1]), int(window_dims[0])
        wz = 1
    patch_voxels = int(discretized_patches.shape[1])
    spatial_shape = (wz, wy, wx) if ndim == 3 else (wy, wx)

    if ndim == 3:
        _DIRECTIONS = (
            (0, 0, 1), (0, 1, 0), (1, 0, 0),
            (0, 1, 1), (0, 1, -1),
            (1, 0, 1), (1, 0, -1),
            (1, 1, 0), (1, -1, 0),
            (1, 1, 1), (1, -1, 1), (1, 1, -1), (1, -1, -1),
        )
    else:
        _DIRECTIONS = (
            (0, 1), (1, 0), (1, 1), (1, -1),
        )

    for row_idx in range(batch_size):
        disc = discretized_patches[row_idx].reshape(spatial_shape)
        mask = mask_patches[row_idx].reshape(spatial_shape).astype(bool)

        diss_accum = []
        sv_accum = []

        for dist_val in distances:
            dist_int = int(dist_val)
            if dist_int <= 0:
                continue
            for direction in _DIRECTIONS:
                src_slices = []
                nbr_slices = []
                valid_dir = True
                for axis_idx, sign in enumerate(direction):
                    offset = sign * dist_int
                    dim_size = spatial_shape[axis_idx]
                    if abs(offset) >= dim_size:
                        valid_dir = False
                        break
                    if offset > 0:
                        src_slices.append(slice(0, dim_size - offset))
                        nbr_slices.append(slice(offset, dim_size))
                    elif offset < 0:
                        src_slices.append(slice(-offset, dim_size))
                        nbr_slices.append(slice(0, dim_size + offset))
                    else:
                        src_slices.append(slice(None))
                        nbr_slices.append(slice(None))
                if not valid_dir:
                    continue
                src_slices = tuple(src_slices)
                nbr_slices = tuple(nbr_slices)

                src_vals = disc[src_slices]
                nbr_vals = disc[nbr_slices]
                src_mask = mask[src_slices]
                nbr_mask = mask[nbr_slices]
                valid = (
                    src_mask & nbr_mask
                    & (src_vals > 0) & (nbr_vals > 0)
                    & (src_vals <= ng) & (nbr_vals <= ng)
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
                p = matrix / matrix_sum

                i_vals = np.arange(1, ng + 1, dtype=np.float64)
                j_vals = np.arange(1, ng + 1, dtype=np.float64)
                i_col = i_vals[:, None]
                j_row = j_vals[None, :]

                if need_dissimilarity:
                    diss_accum.append(float(np.sum(np.abs(i_col - j_row) * p)))
                if need_sum_variance:
                    sum_indices = np.add.outer(i_vals, j_vals).reshape(-1)
                    probabilities = p.reshape(-1)
                    mu_sum = float(np.sum(sum_indices * probabilities))
                    sv_accum.append(
                        float(np.sum(((sum_indices - mu_sum) ** 2.0) * probabilities))
                    )

        if need_dissimilarity and diss_accum:
            dissimilarity_out[row_idx] = float(np.mean(diss_accum))
        if need_sum_variance and sv_accum:
            sum_variance_out[row_idx] = float(np.mean(sv_accum))

    result = {}
    if need_dissimilarity:
        result["Dissimilarity"] = dissimilarity_out
    if need_sum_variance:
        result["SumVariance"] = sum_variance_out
    return result


def _compute_glrlm_feature_maps_batch(
    context: VoxelContext,
    image: sitk.Image,
    image_type_name: str,
    feature_names,
    kwargs,
):
    del kwargs
    batch_fn, backend_label, lib_path = _resolve_voxel_batch_fn(
        context,
        (
            "flash_radiomics_glrlm_cuda_discretized_voxel_batch",
            "flash_radiomics_glrlm_cpu_discretized_voxel_batch",
        ),
    )
    if batch_fn is None:
        raise RuntimeError(
            f"GLRLM voxel batch API is not available in the loaded {backend_label} "
            f"library ({lib_path}). Rebuild native libs from project root with: "
            "'cd flash_radiomics && mkdir -p build && cd build && "
            "cmake -DCMAKE_BUILD_TYPE=Release .. && cmake --build . -j4'"
        )

    requested_set = (
        set(feature_names)
        if feature_names is not None and len(feature_names) > 0
        else None
    )
    selected_feature_names = [
        name
        for name in _GLRLM_BATCH_FEATURE_ORDER
        if requested_set is None or name in requested_set
    ]
    if not selected_feature_names:
        return collections.OrderedDict()

    if context.discretized_windows is None or context.mask_windows is None:
        raise RuntimeError("GLRLM voxel batch path requires discretized and mask windows")

    include_flags = _get_include_flags(
        context,
        class_name="glrlm",
        feature_order=_GLRLM_BATCH_FEATURE_ORDER,
        feature_index_map=_GLRLM_BATCH_FEATURE_INDEX,
        selected_feature_names=selected_feature_names,
    )
    include_flags_ptr = include_flags.ctypes.data_as(
        ctypes.POINTER(ctypes.c_uint8)
    )

    if context.image_array.ndim == 2:
        window_dims = np.asarray(
            [context.kernel_shape[1], context.kernel_shape[0], 1],
            dtype=np.int32,
        )
    else:
        window_dims = np.asarray(
            [
                context.kernel_shape[2],
                context.kernel_shape[1],
                context.kernel_shape[0],
            ],
            dtype=np.int32,
        )
    window_dims_ptr = window_dims.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

    output_arrays = collections.OrderedDict(
        (
            f"{image_type_name}_glrlm_{feature_name}",
            np.full(context.image_array.shape, context.init_value, dtype=np.float64),
        )
        for feature_name in selected_feature_names
    )
    output_assignments = [
        (
            output_arrays[f"{image_type_name}_glrlm_{feature_name}"],
            _GLRLM_BATCH_FEATURE_INDEX[feature_name],
        )
        for feature_name in selected_feature_names
    ]

    full_feature_count = len(_GLRLM_BATCH_FEATURE_ORDER)
    batch_outputs = np.empty(
        (context.batch_size, full_feature_count),
        dtype=np.float64,
    )
    for batch_slice in _iter_batch_slices(
        total=context.roi_coordinates.shape[0],
        batch_size=context.batch_size,
    ):
        gather_t0 = time.perf_counter()
        coords = context.roi_coordinates[batch_slice]
        current_batch = int(coords.shape[0])
        coord_key = tuple(coords[:, axis] for axis in range(coords.shape[1]))
        batch_arrays, cache_hit = _get_batch_patch_cache(
            context,
            batch_slice,
            need_discretized=True,
            need_mask=True,
        )
        discretized_patches = batch_arrays["discretized_patches"]
        mask_patches = batch_arrays["mask_patches_u8"]
        batch_outputs_view = batch_outputs[:current_batch, :]
        gather_elapsed_s = float(time.perf_counter() - gather_t0)

        native_t0 = time.perf_counter()
        status = batch_fn(
            discretized_patches.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            mask_patches.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            current_batch,
            int(context.image_array.ndim),
            window_dims_ptr,
            include_flags_ptr,
            int(context.ng),
            batch_outputs_view.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            int(batch_outputs_view.shape[1]),
        )
        if status != int(_bindings.FlashRadiomicsErrorCode.SUCCESS):
            raise _bindings.FlashRadiomicsComputationError(
                f"GLRLM voxel batch call failed with error code {status}"
            )
        native_elapsed_s = float(time.perf_counter() - native_t0)

        assign_t0 = time.perf_counter()
        for output_array, feature_index in output_assignments:
            output_array[coord_key] = batch_outputs_view[:, feature_index]
        _record_voxel_batch_metrics(
            context,
            "glrlm",
            gather_prep_s=gather_elapsed_s,
            native_call_s=native_elapsed_s,
            output_assign_s=float(time.perf_counter() - assign_t0),
            batch_bytes=_estimate_batch_bytes(batch_arrays) + batch_outputs_view.nbytes,
            cache_hit=cache_hit,
            backend_label=backend_label,
        )

    return collections.OrderedDict(
        (
            output_name,
            _array_to_image(output_array, image),
        )
        for output_name, output_array in output_arrays.items()
    )


def _compute_glszm_feature_maps_batch(
    context: VoxelContext,
    image: sitk.Image,
    image_type_name: str,
    feature_names,
    kwargs,
):
    del kwargs
    batch_fn, backend_label, lib_path = _resolve_voxel_batch_fn(
        context,
        (
            "flash_radiomics_glszm_cuda_discretized_voxel_batch",
            "flash_radiomics_glszm_cpu_discretized_voxel_batch",
        ),
    )
    if batch_fn is None:
        raise RuntimeError(
            f"GLSZM voxel batch API is not available in the loaded {backend_label} "
            f"library ({lib_path}). Rebuild native libs from project root with: "
            "'cd flash_radiomics && mkdir -p build && cd build && "
            "cmake -DCMAKE_BUILD_TYPE=Release .. && cmake --build . -j4'"
        )

    requested_set = (
        set(feature_names)
        if feature_names is not None and len(feature_names) > 0
        else None
    )
    selected_feature_names = [
        name
        for name in _GLSZM_BATCH_FEATURE_ORDER
        if requested_set is None or name in requested_set
    ]
    if not selected_feature_names:
        return collections.OrderedDict()

    if context.discretized_windows is None or context.mask_windows is None:
        raise RuntimeError("GLSZM voxel batch path requires discretized and mask windows")

    include_flags = _get_include_flags(
        context,
        class_name="glszm",
        feature_order=_GLSZM_BATCH_FEATURE_ORDER,
        feature_index_map=_GLSZM_BATCH_FEATURE_INDEX,
        selected_feature_names=selected_feature_names,
    )
    include_flags_ptr = include_flags.ctypes.data_as(
        ctypes.POINTER(ctypes.c_uint8)
    )

    if context.image_array.ndim == 2:
        window_dims = np.asarray(
            [context.kernel_shape[1], context.kernel_shape[0], 1],
            dtype=np.int32,
        )
    else:
        window_dims = np.asarray(
            [
                context.kernel_shape[2],
                context.kernel_shape[1],
                context.kernel_shape[0],
            ],
            dtype=np.int32,
        )
    window_dims_ptr = window_dims.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

    output_arrays = collections.OrderedDict(
        (
            f"{image_type_name}_glszm_{feature_name}",
            np.full(context.image_array.shape, context.init_value, dtype=np.float64),
        )
        for feature_name in selected_feature_names
    )

    selected_feature_count = len(selected_feature_names)
    batch_outputs = np.empty(
        (context.batch_size, selected_feature_count),
        dtype=np.float64,
    )
    for batch_slice in _iter_batch_slices(
        total=context.roi_coordinates.shape[0],
        batch_size=context.batch_size,
    ):
        gather_t0 = time.perf_counter()
        coords = context.roi_coordinates[batch_slice]
        current_batch = int(coords.shape[0])
        coord_key = tuple(coords[:, axis] for axis in range(coords.shape[1]))
        batch_arrays, cache_hit = _get_batch_patch_cache(
            context,
            batch_slice,
            need_discretized=True,
            need_mask=True,
        )
        discretized_patches = batch_arrays["discretized_patches"]
        mask_patches = batch_arrays["mask_patches_u8"]
        batch_outputs_view = batch_outputs[:current_batch, :]
        gather_elapsed_s = float(time.perf_counter() - gather_t0)

        native_t0 = time.perf_counter()
        status = batch_fn(
            discretized_patches.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            mask_patches.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            current_batch,
            int(context.image_array.ndim),
            window_dims_ptr,
            include_flags_ptr,
            int(context.ng),
            batch_outputs_view.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            int(batch_outputs_view.shape[1]),
        )
        if status != int(_bindings.FlashRadiomicsErrorCode.SUCCESS):
            raise _bindings.FlashRadiomicsComputationError(
                f"GLSZM voxel batch call failed with error code {status}"
            )
        native_elapsed_s = float(time.perf_counter() - native_t0)

        assign_t0 = time.perf_counter()
        for output_idx, feature_name in enumerate(selected_feature_names):
            output_name = f"{image_type_name}_glszm_{feature_name}"
            output_arrays[output_name][coord_key] = batch_outputs_view[:, output_idx]
        _record_voxel_batch_metrics(
            context,
            "glszm",
            gather_prep_s=gather_elapsed_s,
            native_call_s=native_elapsed_s,
            output_assign_s=float(time.perf_counter() - assign_t0),
            batch_bytes=_estimate_batch_bytes(batch_arrays) + batch_outputs_view.nbytes,
            cache_hit=cache_hit,
            backend_label=backend_label,
        )

    return collections.OrderedDict(
        (
            output_name,
            _array_to_image(output_array, image),
        )
        for output_name, output_array in output_arrays.items()
    )


def _compute_gldm_feature_maps_batch(
    context: VoxelContext,
    image: sitk.Image,
    image_type_name: str,
    feature_names,
    kwargs,
):
    batch_fn, backend_label, lib_path = _resolve_voxel_batch_fn(
        context,
        (
            "flash_radiomics_gldm_cuda_discretized_voxel_batch",
            "flash_radiomics_gldm_cpu_discretized_voxel_batch",
        ),
    )
    if batch_fn is None:
        raise RuntimeError(
            f"GLDM voxel batch API is not available in the loaded {backend_label} "
            f"library ({lib_path}). Rebuild native libs from project root with: "
            "'cd flash_radiomics && mkdir -p build && cd build && "
            "cmake -DCMAKE_BUILD_TYPE=Release .. && cmake --build . -j4'"
        )

    requested_set = (
        set(feature_names)
        if feature_names is not None and len(feature_names) > 0
        else None
    )
    selected_feature_names = [
        name
        for name in _GLDM_BATCH_FEATURE_ORDER
        if requested_set is None or name in requested_set
    ]
    if not selected_feature_names:
        return collections.OrderedDict()

    if context.discretized_windows is None or context.mask_windows is None:
        raise RuntimeError("GLDM voxel batch path requires discretized and mask windows")

    # Split into native (C kernel) and supplemental (Python-derived) features.
    native_selected = [
        name for name in selected_feature_names
        if name not in _GLDM_PYTHON_SUPPLEMENTAL_FEATURES
    ]
    supplemental_selected = [
        name for name in selected_feature_names
        if name in _GLDM_PYTHON_SUPPLEMENTAL_FEATURES
    ]
    need_supplemental = len(supplemental_selected) > 0

    # Normalized GLNU requires native GrayLevelNonUniformity.
    need_glnu_for_norm = (
        "GrayLevelNonUniformityNormalized" in supplemental_selected
        and "GrayLevelNonUniformity" not in native_selected
    )
    if need_glnu_for_norm:
        native_selected.append("GrayLevelNonUniformity")

    # Include flags follow the native GLDM feature order.
    native_feature_index = {
        name: idx for idx, name in enumerate(_GLDM_NATIVE_BATCH_FEATURE_ORDER)
    }
    native_include_flags = np.zeros(len(_GLDM_NATIVE_BATCH_FEATURE_ORDER), dtype=np.uint8)
    for name in native_selected:
        native_include_flags[native_feature_index[name]] = 1
    native_include_flags_ptr = native_include_flags.ctypes.data_as(
        ctypes.POINTER(ctypes.c_uint8)
    )

    distances = np.asarray(kwargs.get("distances", [1]), dtype=np.int32).reshape(-1)
    if distances.size == 0:
        distances = np.asarray([1], dtype=np.int32)
    distances = np.ascontiguousarray(distances, dtype=np.int32)
    distance_ptr = distances.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    gldm_a = float(kwargs.get("gldm_a", 0.0))

    if context.image_array.ndim == 2:
        window_dims = np.asarray(
            [context.kernel_shape[1], context.kernel_shape[0], 1],
            dtype=np.int32,
        )
    else:
        window_dims = np.asarray(
            [
                context.kernel_shape[2],
                context.kernel_shape[1],
                context.kernel_shape[0],
            ],
            dtype=np.int32,
        )
    window_dims_ptr = window_dims.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

    output_arrays = collections.OrderedDict(
        (
            f"{image_type_name}_gldm_{feature_name}",
            np.full(context.image_array.shape, context.init_value, dtype=np.float64),
        )
        for feature_name in selected_feature_names
    )

    # Map native feature names to their column index in the native output.
    native_col_map = {name: idx for idx, name in enumerate(native_selected)}
    native_feature_count = len(native_selected)

    full_native_count = len(_GLDM_NATIVE_BATCH_FEATURE_ORDER)
    batch_outputs = np.empty(
        (context.batch_size, full_native_count),
        dtype=np.float64,
    )
    for batch_slice in _iter_batch_slices(
        total=context.roi_coordinates.shape[0],
        batch_size=context.batch_size,
    ):
        gather_t0 = time.perf_counter()
        coords = context.roi_coordinates[batch_slice]
        current_batch = int(coords.shape[0])
        coord_key = tuple(coords[:, axis] for axis in range(coords.shape[1]))
        batch_arrays, cache_hit = _get_batch_patch_cache(
            context,
            batch_slice,
            need_discretized=True,
            need_mask=True,
        )
        discretized_patches = batch_arrays["discretized_patches"]
        mask_patches = batch_arrays["mask_patches_u8"]
        batch_outputs_view = batch_outputs[:current_batch, :]
        gather_elapsed_s = float(time.perf_counter() - gather_t0)

        native_t0 = time.perf_counter()
        status = batch_fn(
            discretized_patches.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            mask_patches.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            current_batch,
            int(context.image_array.ndim),
            window_dims_ptr,
            distance_ptr,
            int(distances.size),
            native_include_flags_ptr,
            int(context.ng),
            gldm_a,
            batch_outputs_view.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            int(batch_outputs_view.shape[1]),
        )
        if status != int(_bindings.FlashRadiomicsErrorCode.SUCCESS):
            raise _bindings.FlashRadiomicsComputationError(
                f"GLDM voxel batch call failed with error code {status}"
            )
        native_elapsed_s = float(time.perf_counter() - native_t0)

        supplemental_values = {}
        if need_supplemental:
            mask_bool = mask_patches.astype(bool)
            n_voxels = mask_bool.sum(axis=1).astype(np.float64)
            n_voxels[n_voxels == 0] = 1.0

            if "DependenceCountPercentage" in supplemental_selected:
                # Match segment GLDM semantics: positive-gray voxels are centres.
                disc_vals = discretized_patches
                disc_mask = mask_bool & (disc_vals > 0)
                n_s = disc_mask.sum(axis=1).astype(np.float64)
                supplemental_values["DependenceCountPercentage"] = n_s / n_voxels

            if "DependenceCountEnergy" in supplemental_selected:
                # Reconstruct the matrix for segment-compatible DependenceCountEnergy.
                supplemental_values["DependenceCountEnergy"] = _compute_gldm_energy_batch(
                    discretized_patches=discretized_patches,
                    mask_patches=mask_patches,
                    ng=context.ng,
                    distances=distances,
                    gldm_a=gldm_a,
                    ndim=context.image_array.ndim,
                    window_dims=window_dims,
                )

            if "GrayLevelNonUniformityNormalized" in supplemental_selected:
                glnu_idx = native_feature_index.get("GrayLevelNonUniformity")
                if glnu_idx is not None:
                    glnu = batch_outputs_view[:, glnu_idx].copy()
                    disc_vals = discretized_patches
                    disc_mask = mask_patches.astype(bool) & (disc_vals > 0)
                    n_s = disc_mask.sum(axis=1).astype(np.float64)
                    n_s[n_s == 0] = 1.0
                    supplemental_values["GrayLevelNonUniformityNormalized"] = glnu / n_s
                else:
                    supplemental_values["GrayLevelNonUniformityNormalized"] = np.full(
                        current_batch, np.nan, dtype=np.float64
                    )

        assign_t0 = time.perf_counter()
        for feature_name in selected_feature_names:
            output_name = f"{image_type_name}_gldm_{feature_name}"
            if feature_name in _GLDM_PYTHON_SUPPLEMENTAL_FEATURES:
                output_arrays[output_name][coord_key] = supplemental_values[feature_name]
            elif feature_name in native_feature_index:
                output_arrays[output_name][coord_key] = batch_outputs_view[
                    :, native_feature_index[feature_name]
                ]
        _record_voxel_batch_metrics(
            context,
            "gldm",
            gather_prep_s=gather_elapsed_s,
            native_call_s=native_elapsed_s,
            output_assign_s=float(time.perf_counter() - assign_t0),
            batch_bytes=_estimate_batch_bytes(batch_arrays) + batch_outputs_view.nbytes,
            cache_hit=cache_hit,
            backend_label=backend_label,
        )

    return collections.OrderedDict(
        (
            output_name,
            _array_to_image(output_array, image),
        )
        for output_name, output_array in output_arrays.items()
    )


def _compute_gldm_energy_batch(
    *,
    discretized_patches: np.ndarray,
    mask_patches: np.ndarray,
    ng: int,
    distances: np.ndarray,
    gldm_a: float,
    ndim: int,
    window_dims: np.ndarray,
) -> np.ndarray:
    """Compute DependenceCountEnergy per voxel window.

    Builds the full GLDM dependence matrix per window and computes
    Σ(count²) / N_s².  This mirrors ``RadiomicsGLDM._compute_dependence_count_energy``.
    """
    batch_size = int(discretized_patches.shape[0])
    energy_out = np.full(batch_size, np.nan, dtype=np.float64)

    if ndim == 3:
        wz, wy, wx = int(window_dims[2]), int(window_dims[1]), int(window_dims[0])
        spatial_shape = (wz, wy, wx)
        _DIRECTIONS = (
            (1, 0, 0), (-1, 0, 0),
            (0, 1, 0), (0, -1, 0),
            (0, 0, 1), (0, 0, -1),
            (1, 1, 0), (-1, -1, 0), (1, -1, 0), (-1, 1, 0),
            (1, 0, 1), (-1, 0, -1), (1, 0, -1), (-1, 0, 1),
            (0, 1, 1), (0, -1, -1), (0, 1, -1), (0, -1, 1),
            (1, 1, 1), (-1, -1, -1), (1, 1, -1), (-1, -1, 1),
            (1, -1, 1), (-1, 1, -1), (1, -1, -1), (-1, 1, 1),
        )
    else:
        wy, wx = int(window_dims[1]), int(window_dims[0])
        spatial_shape = (wy, wx)
        _DIRECTIONS = (
            (1, 0), (-1, 0), (0, 1), (0, -1),
            (1, 1), (-1, -1), (1, -1), (-1, 1),
        )

    max_dep = len(_DIRECTIONS) * len(distances) + 1

    for row_idx in range(batch_size):
        disc = discretized_patches[row_idx].reshape(spatial_shape)
        mask = mask_patches[row_idx].reshape(spatial_shape).astype(bool)

        dep = np.zeros(spatial_shape, dtype=np.int32)
        for dist_val in distances:
            step = int(dist_val)
            if step <= 0:
                continue
            for direction in _DIRECTIONS:
                src_slices = []
                nbr_slices = []
                valid_dir = True
                for axis_idx, sign in enumerate(direction):
                    offset = sign * step
                    dim_size = spatial_shape[axis_idx]
                    if abs(offset) >= dim_size:
                        valid_dir = False
                        break
                    if offset > 0:
                        src_slices.append(slice(0, dim_size - offset))
                        nbr_slices.append(slice(offset, dim_size))
                    elif offset < 0:
                        src_slices.append(slice(-offset, dim_size))
                        nbr_slices.append(slice(0, dim_size + offset))
                    else:
                        src_slices.append(slice(None))
                        nbr_slices.append(slice(None))
                if not valid_dir:
                    continue
                src_slices = tuple(src_slices)
                nbr_slices = tuple(nbr_slices)

                src_mask = mask[src_slices]
                nbr_mask = mask[nbr_slices]
                valid = src_mask & nbr_mask
                if not np.any(valid):
                    continue
                src_gray = disc[src_slices]
                nbr_gray = disc[nbr_slices]
                similar = valid & (src_gray > 0) & (nbr_gray > 0)
                similar &= (np.abs(src_gray.astype(np.float64) - nbr_gray.astype(np.float64)) <= gldm_a)
                if not np.any(similar):
                    continue
                dep_view = dep[src_slices]
                dep_view[similar] += 1

        centres = mask & (disc > 0)
        if not np.any(centres):
            continue
        g_idx = disc[centres] - 1
        d_idx = dep[centres]
        n_s = int(g_idx.size)
        linear_idx = g_idx * max_dep + d_idx
        counts = np.bincount(
            linear_idx,
            minlength=max(int(ng), int(np.max(disc[centres]))) * max_dep,
        ).astype(np.float64, copy=False)
        energy_out[row_idx] = float(np.sum(counts * counts) / (float(n_s) * float(n_s)))

    return energy_out


def _compute_ngtdm_feature_maps_batch(
    context: VoxelContext,
    image: sitk.Image,
    image_type_name: str,
    feature_names,
    kwargs,
):
    batch_fn, backend_label, lib_path = _resolve_voxel_batch_fn(
        context,
        (
            "flash_radiomics_ngtdm_cuda_discretized_voxel_batch",
            "flash_radiomics_ngtdm_cpu_discretized_voxel_batch",
        ),
    )
    if batch_fn is None:
        raise RuntimeError(
            f"NGTDM voxel batch API is not available in the loaded {backend_label} "
            f"library ({lib_path}). Rebuild native libs from project root with: "
            "'cd flash_radiomics && mkdir -p build && cd build && "
            "cmake -DCMAKE_BUILD_TYPE=Release .. && cmake --build . -j4'"
        )

    requested_set = (
        set(feature_names)
        if feature_names is not None and len(feature_names) > 0
        else None
    )
    selected_feature_names = [
        name
        for name in _NGTDM_BATCH_FEATURE_ORDER
        if requested_set is None or name in requested_set
    ]
    if not selected_feature_names:
        return collections.OrderedDict()

    if context.discretized_windows is None or context.mask_windows is None:
        raise RuntimeError("NGTDM voxel batch path requires discretized and mask windows")

    include_flags = _get_include_flags(
        context,
        class_name="ngtdm",
        feature_order=_NGTDM_BATCH_FEATURE_ORDER,
        feature_index_map=_NGTDM_BATCH_FEATURE_INDEX,
        selected_feature_names=selected_feature_names,
    )
    include_flags_ptr = include_flags.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))

    distances = np.asarray(kwargs.get("distances", [1]), dtype=np.int32).reshape(-1)
    if distances.size == 0:
        distances = np.asarray([1], dtype=np.int32)
    distances = np.ascontiguousarray(distances, dtype=np.int32)
    distance_ptr = distances.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

    if context.image_array.ndim == 2:
        window_dims = np.asarray(
            [context.kernel_shape[1], context.kernel_shape[0], 1],
            dtype=np.int32,
        )
    else:
        window_dims = np.asarray(
            [
                context.kernel_shape[2],
                context.kernel_shape[1],
                context.kernel_shape[0],
            ],
            dtype=np.int32,
        )
    window_dims_ptr = window_dims.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

    output_arrays = collections.OrderedDict(
        (
            f"{image_type_name}_ngtdm_{feature_name}",
            np.full(context.image_array.shape, context.init_value, dtype=np.float64),
        )
        for feature_name in selected_feature_names
    )
    output_assignments = [
        (
            output_arrays[f"{image_type_name}_ngtdm_{feature_name}"],
            _NGTDM_BATCH_FEATURE_INDEX[feature_name],
        )
        for feature_name in selected_feature_names
    ]

    full_feature_count = len(_NGTDM_BATCH_FEATURE_ORDER)
    batch_outputs = np.empty(
        (context.batch_size, full_feature_count),
        dtype=np.float64,
    )
    for batch_slice in _iter_batch_slices(
        total=context.roi_coordinates.shape[0],
        batch_size=context.batch_size,
    ):
        gather_t0 = time.perf_counter()
        coords = context.roi_coordinates[batch_slice]
        current_batch = int(coords.shape[0])
        coord_key = tuple(coords[:, axis] for axis in range(coords.shape[1]))
        batch_arrays, cache_hit = _get_batch_patch_cache(
            context,
            batch_slice,
            need_discretized=True,
            need_mask=True,
        )
        discretized_patches = batch_arrays["discretized_patches"]
        mask_patches = batch_arrays["mask_patches_u8"]

        batch_outputs_view = batch_outputs[:current_batch, :]
        gather_elapsed_s = float(time.perf_counter() - gather_t0)

        native_t0 = time.perf_counter()
        status = batch_fn(
            discretized_patches.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            mask_patches.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            current_batch,
            int(context.image_array.ndim),
            window_dims_ptr,
            distance_ptr,
            int(distances.size),
            include_flags_ptr,
            int(context.ng),
            batch_outputs_view.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            int(batch_outputs.shape[1]),
        )
        if status != int(_bindings.FlashRadiomicsErrorCode.SUCCESS):
            raise _bindings.FlashRadiomicsComputationError(
                f"NGTDM voxel batch call failed with error code {status}"
            )
        native_elapsed_s = float(time.perf_counter() - native_t0)

        assign_t0 = time.perf_counter()
        for output_array, feature_index in output_assignments:
            output_array[coord_key] = batch_outputs_view[:, feature_index]
        _record_voxel_batch_metrics(
            context,
            "ngtdm",
            gather_prep_s=gather_elapsed_s,
            native_call_s=native_elapsed_s,
            output_assign_s=float(time.perf_counter() - assign_t0),
            batch_bytes=_estimate_batch_bytes(batch_arrays) + batch_outputs_view.nbytes,
            cache_hit=cache_hit,
            backend_label=backend_label,
        )

    return collections.OrderedDict(
        (
            output_name,
            _array_to_image(output_array, image),
        )
        for output_name, output_array in output_arrays.items()
    )


def _resolve_firstorder_features(feature_names):
    if not feature_names:
        return _FIRSTORDER_DEFAULT_FEATURES
    requested = _deduplicate_feature_names(feature_names)
    unknown = [
        feature_name
        for feature_name in requested
        if feature_name not in _FIRSTORDER_ALL_FEATURES
    ]
    if unknown:
        raise RuntimeError(
            "Unsupported voxel firstorder features without native Flash batch "
            f"implementations: {', '.join(sorted(unknown))}"
        )
    return requested


def _resolve_texture_features(feature_class_name, feature_names):
    feature_order = _TEXTURE_BATCH_FEATURE_ORDERS.get(feature_class_name)
    if feature_order is None:
        raise RuntimeError(
            "Unsupported voxel texture class for native Flash batch execution: "
            f"{feature_class_name}"
        )

    if not feature_names:
        return feature_order

    requested = _deduplicate_feature_names(feature_names)
    supported = set(feature_order)
    unknown = [
        feature_name
        for feature_name in requested
        if feature_name not in supported
    ]
    if unknown:
        raise RuntimeError(
            f"Unsupported voxel {feature_class_name} features without native Flash "
            f"batch implementations: {', '.join(sorted(unknown))}"
        )

    requested_set = set(requested)
    return tuple(name for name in feature_order if name in requested_set)


def _deduplicate_feature_names(feature_names):
    return tuple(dict.fromkeys(feature_names))


def _build_failed_feature_maps(
    *,
    context: VoxelContext,
    image: sitk.Image,
    image_type_name: str,
    feature_class_name: str,
    feature_names,
):
    return collections.OrderedDict(
        (
            f"{image_type_name}_{feature_class_name}_{feature_name}",
            _array_to_image(
                np.full(
                    context.image_array.shape,
                    context.init_value,
                    dtype=np.float64,
                ),
                image,
            ),
        )
        for feature_name in feature_names
    )


def _linear_percentile_from_sorted(sorted_values: np.ndarray, percentile: float) -> float:
    value_count = int(sorted_values.shape[0])
    if value_count <= 0:
        return float("nan")
    if value_count == 1:
        return float(sorted_values[0])

    bounded_percentile = float(min(100.0, max(0.0, percentile)))
    rank = (value_count - 1) * (bounded_percentile / 100.0)
    lower_index = int(np.floor(rank))
    upper_index = int(np.ceil(rank))
    if lower_index == upper_index:
        return float(sorted_values[lower_index])
    interpolation = rank - lower_index
    lower = float(sorted_values[lower_index])
    upper = float(sorted_values[upper_index])
    return lower + (upper - lower) * interpolation


def _compute_robust_mad_from_sorted(
    sorted_values: np.ndarray,
    percentile10: float,
    percentile90: float,
) -> float:
    low_index = int(np.searchsorted(sorted_values, percentile10, side="left"))
    high_index = int(np.searchsorted(sorted_values, percentile90, side="right"))
    if high_index <= low_index:
        return 0.0
    subset = np.asarray(sorted_values[low_index:high_index], dtype=np.float64)
    subset_mean = float(np.mean(subset, dtype=np.float64))
    return float(np.mean(np.abs(subset - subset_mean), dtype=np.float64))


def _compute_batch_order_stats(
    *,
    image_patches: np.ndarray,
    valid_mask: np.ndarray,
    need_percentiles: bool,
    need_minmax: bool,
    need_median: bool,
    need_robust_mad: bool,
) -> dict[str, np.ndarray]:
    batch_size = int(image_patches.shape[0])
    outputs: dict[str, np.ndarray] = {}

    if need_percentiles or need_robust_mad:
        outputs["percentile10"] = np.full(batch_size, np.nan, dtype=np.float64)
        outputs["percentile90"] = np.full(batch_size, np.nan, dtype=np.float64)
    if need_percentiles:
        outputs["percentile25"] = np.full(batch_size, np.nan, dtype=np.float64)
        outputs["percentile75"] = np.full(batch_size, np.nan, dtype=np.float64)
    if need_minmax:
        outputs["minimum"] = np.full(batch_size, np.nan, dtype=np.float64)
        outputs["maximum"] = np.full(batch_size, np.nan, dtype=np.float64)
    if need_median:
        outputs["median"] = np.full(batch_size, np.nan, dtype=np.float64)
    if need_robust_mad:
        outputs["robust_mad"] = np.full(batch_size, np.nan, dtype=np.float64)

    for row_idx in range(batch_size):
        row_mask = valid_mask[row_idx]
        if not np.any(row_mask):
            continue
        sorted_values = np.sort(
            np.asarray(image_patches[row_idx, row_mask], dtype=np.float64),
            kind="quicksort",
        )

        percentile10 = None
        percentile90 = None
        if need_percentiles or need_robust_mad:
            percentile10 = _linear_percentile_from_sorted(sorted_values, 10.0)
            percentile90 = _linear_percentile_from_sorted(sorted_values, 90.0)
            outputs["percentile10"][row_idx] = percentile10
            outputs["percentile90"][row_idx] = percentile90
        if need_percentiles:
            outputs["percentile25"][row_idx] = _linear_percentile_from_sorted(
                sorted_values,
                25.0,
            )
            outputs["percentile75"][row_idx] = _linear_percentile_from_sorted(
                sorted_values,
                75.0,
            )
        if need_minmax:
            outputs["minimum"][row_idx] = float(sorted_values[0])
            outputs["maximum"][row_idx] = float(sorted_values[-1])
        if need_median:
            outputs["median"][row_idx] = _linear_percentile_from_sorted(
                sorted_values,
                50.0,
            )
        if need_robust_mad:
            outputs["robust_mad"][row_idx] = _compute_robust_mad_from_sorted(
                sorted_values,
                float(percentile10),
                float(percentile90),
            )
    return outputs


def _batch_histogram_counts(discretized_patches, valid_mask, ng):
    if ng <= 0:
        return np.zeros((discretized_patches.shape[0], 0), dtype=np.int64)

    flat_values = discretized_patches.reshape(-1)
    flat_valid = valid_mask.reshape(-1) & (flat_values > 0) & (flat_values <= ng)
    if not np.any(flat_valid):
        return np.zeros((discretized_patches.shape[0], ng), dtype=np.int64)

    batch_indices = np.repeat(
        np.arange(discretized_patches.shape[0], dtype=np.int64),
        discretized_patches.shape[1],
    )
    bincount_index = batch_indices[flat_valid] * ng + (flat_values[flat_valid] - 1)
    counts = np.bincount(
        bincount_index,
        minlength=discretized_patches.shape[0] * ng,
    )
    return counts.reshape(discretized_patches.shape[0], ng)


def _gather_windows(windows, starts):
    if windows is None:
        raise RuntimeError("Requested voxel windows are not available in current context")
    return windows[tuple(starts[:, axis] for axis in range(starts.shape[1]))]


def _iter_batch_slices(total: int, batch_size: int):
    for start in range(0, total, batch_size):
        yield slice(start, min(total, start + batch_size))


def _get_kernel_shape(
    ndim: int,
    kernel_radius: int,
    force2d: bool,
    force2d_dimension: int,
) -> tuple[int, ...]:
    kernel_shape = []
    for axis in range(ndim):
        if force2d and ndim == 3 and axis == force2d_dimension:
            kernel_shape.append(1)
        else:
            kernel_shape.append(kernel_radius * 2 + 1)
    return tuple(kernel_shape)


def _is_cuda_unified_dispatch_available(context: VoxelContext) -> bool:
    """Check if the unified multi-class CUDA dispatch is available."""
    backend_name = str(context.backend or "cpu").strip().lower()
    if backend_name == "gpu":
        backend_name = "cuda"
    if backend_name != "cuda":
        return False
    try:
        cuda_lib = _bindings.get_feature_library("cuda")
        return hasattr(cuda_lib, "flash_radiomics_voxel_multi_class_cuda_batch")
    except Exception:
        return False


def _compute_texture_feature_maps_multi_class_cuda(
    context: VoxelContext,
    image: sitk.Image,
    image_type_name: str,
    texture_classes: list[str],
    resolved_feature_names: dict[str, tuple[str, ...]],
    kwargs: dict,
) -> dict[str, collections.OrderedDict]:
    """Compute multiple texture feature classes via unified CUDA dispatch.

    Calls flash_radiomics_voxel_multi_class_cuda_batch which uploads
    discretized_windows and mask_windows to the GPU once, then launches
    all enabled feature class kernels concurrently on separate CUDA streams.
    """
    cuda_lib = _bindings.get_feature_library("cuda")
    multi_class_fn = cuda_lib.flash_radiomics_voxel_multi_class_cuda_batch

    class_mask = 0
    _class_name_to_bit = {
        "glcm": _bindings.FLASH_VOXEL_CLASS_GLCM,
        "glrlm": _bindings.FLASH_VOXEL_CLASS_GLRLM,
        "glszm": _bindings.FLASH_VOXEL_CLASS_GLSZM,
        "gldm": _bindings.FLASH_VOXEL_CLASS_GLDM,
        "ngtdm": _bindings.FLASH_VOXEL_CLASS_NGTDM,
    }
    _class_native_order = {
        "glcm": _GLCM_NATIVE_BATCH_FEATURE_ORDER,
        "glrlm": _GLRLM_BATCH_FEATURE_ORDER,
        "glszm": _GLSZM_BATCH_FEATURE_ORDER,
        "gldm": _GLDM_NATIVE_BATCH_FEATURE_ORDER,
        "ngtdm": _NGTDM_BATCH_FEATURE_ORDER,
    }
    _class_full_order = {
        "glcm": _GLCM_BATCH_FEATURE_ORDER,
        "glrlm": _GLRLM_BATCH_FEATURE_ORDER,
        "glszm": _GLSZM_BATCH_FEATURE_ORDER,
        "gldm": _GLDM_BATCH_FEATURE_ORDER,
        "ngtdm": _NGTDM_BATCH_FEATURE_ORDER,
    }
    _class_supplemental = {
        "glcm": _GLCM_PYTHON_SUPPLEMENTAL_FEATURES,
        "gldm": _GLDM_PYTHON_SUPPLEMENTAL_FEATURES,
    }

    per_class_info = {}
    for cls_name in texture_classes:
        if cls_name not in _class_name_to_bit:
            raise RuntimeError(f"Unsupported texture class: {cls_name}")
        class_mask |= _class_name_to_bit[cls_name]

        native_order = _class_native_order[cls_name]
        full_order = _class_full_order[cls_name]
        requested = resolved_feature_names.get(cls_name)
        requested_set = set(requested) if requested else None
        supplemental_features = _class_supplemental.get(cls_name, frozenset())

        selected = [
            name for name in full_order
            if requested_set is None or name in requested_set
        ]
        native_selected = [
            name for name in selected if name not in supplemental_features
        ]
        supplemental_selected = [
            name for name in selected if name in supplemental_features
        ]

        native_index = {name: idx for idx, name in enumerate(native_order)}
        include_flags = np.zeros(len(native_order), dtype=np.uint8)
        for name in native_selected:
            include_flags[native_index[name]] = 1

        # GLDM normalization requires GLNU even when GLNU was not requested.
        if (cls_name == "gldm"
                and "GrayLevelNonUniformityNormalized" in supplemental_selected
                and "GrayLevelNonUniformity" not in native_selected):
            glnu_idx = native_index.get("GrayLevelNonUniformity")
            if glnu_idx is not None:
                include_flags[glnu_idx] = 1

        per_class_info[cls_name] = {
            "selected": selected,
            "native_selected": native_selected,
            "supplemental_selected": supplemental_selected,
            "include_flags": include_flags,
            "native_count": len(native_selected),
        }

    distances = np.asarray(kwargs.get("distances", [1]), dtype=np.int32).reshape(-1)
    if distances.size == 0:
        distances = np.asarray([1], dtype=np.int32)
    distances = np.ascontiguousarray(distances, dtype=np.int32)
    distance_ptr = distances.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    gldm_a = float(kwargs.get("gldm_a", 0.0))

    if context.image_array.ndim == 2:
        window_dims = np.asarray(
            [context.kernel_shape[1], context.kernel_shape[0], 1],
            dtype=np.int32,
        )
    else:
        window_dims = np.asarray(
            [
                context.kernel_shape[2],
                context.kernel_shape[1],
                context.kernel_shape[0],
            ],
            dtype=np.int32,
        )
    window_dims_ptr = window_dims.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

    all_output_arrays = {}
    for cls_name in texture_classes:
        info = per_class_info[cls_name]
        all_output_arrays[cls_name] = collections.OrderedDict(
            (
                f"{image_type_name}_{cls_name}_{feature_name}",
                np.full(context.image_array.shape, context.init_value, dtype=np.float64),
            )
            for feature_name in info["selected"]
        )

    # Native kernels require full-width buffers to preserve ABI strides.
    batch_outputs = {}
    for cls_name in texture_classes:
        native_order = _class_native_order[cls_name]
        full_count = len(native_order)
        if per_class_info[cls_name]["native_count"] > 0:
            batch_outputs[cls_name] = np.empty(
                (context.batch_size, full_count), dtype=np.float64,
            )
        else:
            batch_outputs[cls_name] = None

    for batch_slice in _iter_batch_slices(
        total=context.roi_coordinates.shape[0],
        batch_size=context.batch_size,
    ):
        coords = context.roi_coordinates[batch_slice]
        current_batch = int(coords.shape[0])
        coord_key = tuple(coords[:, axis] for axis in range(coords.shape[1]))
        batch_arrays, _cache_hit = _get_batch_patch_cache(
            context,
            batch_slice,
            need_discretized=True,
            need_mask=True,
        )
        discretized_patches = batch_arrays["discretized_patches"]
        mask_patches = batch_arrays["mask_patches_u8"]

        def _make_ptr(cls_name):
            info = per_class_info[cls_name]
            buf = batch_outputs.get(cls_name)
            if buf is not None and info["native_count"] > 0:
                view = buf[:current_batch, :]
                return (
                    info["include_flags"].ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                    view.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                    int(view.shape[1]),
                )
            return (None, None, 0)

        glcm_inc, glcm_out, glcm_stride = _make_ptr("glcm") if "glcm" in per_class_info else (None, None, 0)
        glrlm_inc, glrlm_out, glrlm_stride = _make_ptr("glrlm") if "glrlm" in per_class_info else (None, None, 0)
        glszm_inc, glszm_out, glszm_stride = _make_ptr("glszm") if "glszm" in per_class_info else (None, None, 0)
        gldm_inc, gldm_out, gldm_stride = _make_ptr("gldm") if "gldm" in per_class_info else (None, None, 0)
        ngtdm_inc, ngtdm_out, ngtdm_stride = _make_ptr("ngtdm") if "ngtdm" in per_class_info else (None, None, 0)

        status = multi_class_fn(
            discretized_patches.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            mask_patches.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            current_batch,
            int(context.image_array.ndim),
            window_dims_ptr,
            int(context.ng),
            ctypes.c_uint(class_mask),
            distance_ptr,
            int(distances.size),
            ctypes.c_double(gldm_a),
            glcm_inc, glcm_out, glcm_stride,
            glrlm_inc, glrlm_out, glrlm_stride,
            glszm_inc, glszm_out, glszm_stride,
            gldm_inc, gldm_out, gldm_stride,
            ngtdm_inc, ngtdm_out, ngtdm_stride,
        )
        if status != int(_bindings.FlashRadiomicsErrorCode.SUCCESS):
            raise _bindings.FlashRadiomicsComputationError(
                f"Unified multi-class CUDA voxel batch call failed "
                f"with error code {status}"
            )

        for cls_name in texture_classes:
            info = per_class_info[cls_name]
            output_arrays = all_output_arrays[cls_name]
            native_buf = batch_outputs.get(cls_name)
            supplemental_features = _class_supplemental.get(cls_name, frozenset())

            supplemental_values = {}
            if info["supplemental_selected"]:
                if cls_name == "glcm":
                    supplemental_values = _compute_glcm_supplemental_batch(
                        discretized_patches=discretized_patches,
                        mask_patches=mask_patches,
                        ng=context.ng,
                        distances=distances,
                        ndim=context.image_array.ndim,
                        window_dims=window_dims,
                        supplemental_names=info["supplemental_selected"],
                    )
                elif cls_name == "gldm":
                    mask_bool = mask_patches.astype(bool)
                    n_voxels = mask_bool.sum(axis=1).astype(np.float64)
                    n_voxels[n_voxels == 0] = 1.0
                    if "DependenceCountPercentage" in info["supplemental_selected"]:
                        disc_mask = mask_bool & (discretized_patches > 0)
                        n_s = disc_mask.sum(axis=1).astype(np.float64)
                        supplemental_values["DependenceCountPercentage"] = n_s / n_voxels
                    if "DependenceCountEnergy" in info["supplemental_selected"]:
                        supplemental_values["DependenceCountEnergy"] = _compute_gldm_energy_batch(
                            discretized_patches=discretized_patches,
                            mask_patches=mask_patches,
                            ng=context.ng,
                            distances=distances,
                            gldm_a=gldm_a,
                            ndim=context.image_array.ndim,
                            window_dims=window_dims,
                        )
                    if "GrayLevelNonUniformityNormalized" in info["supplemental_selected"]:
                        native_order_gldm = _GLDM_NATIVE_BATCH_FEATURE_ORDER
                        native_idx_map = {n: i for i, n in enumerate(native_order_gldm)}
                        glnu_full_idx = native_idx_map.get("GrayLevelNonUniformity")
                        if glnu_full_idx is not None and native_buf is not None:
                            glnu_in_native = info["include_flags"][glnu_full_idx]
                            if glnu_in_native:
                                glnu = native_buf[:current_batch, glnu_full_idx].copy()
                                disc_mask = mask_bool & (discretized_patches > 0)
                                n_s = disc_mask.sum(axis=1).astype(np.float64)
                                n_s[n_s == 0] = 1.0
                                supplemental_values["GrayLevelNonUniformityNormalized"] = glnu / n_s
                            else:
                                supplemental_values["GrayLevelNonUniformityNormalized"] = np.full(
                                    current_batch, np.nan, dtype=np.float64
                                )
                        else:
                            supplemental_values["GrayLevelNonUniformityNormalized"] = np.full(
                                current_batch, np.nan, dtype=np.float64
                            )

            # GLCM/GLSZM outputs are compact; GLRLM/GLDM/NGTDM use native indices.
            compact_output = cls_name in ("glcm", "glszm")
            native_order = _class_native_order[cls_name]
            native_index = {name: idx for idx, name in enumerate(native_order)}

            if compact_output:
                native_col = 0
                for feature_name in info["selected"]:
                    output_name = f"{image_type_name}_{cls_name}_{feature_name}"
                    if feature_name in supplemental_features:
                        if feature_name in supplemental_values:
                            output_arrays[output_name][coord_key] = supplemental_values[feature_name]
                    else:
                        if native_buf is not None:
                            output_arrays[output_name][coord_key] = native_buf[:current_batch, native_col]
                        native_col += 1
            else:
                for feature_name in info["selected"]:
                    output_name = f"{image_type_name}_{cls_name}_{feature_name}"
                    if feature_name in supplemental_features:
                        if feature_name in supplemental_values:
                            output_arrays[output_name][coord_key] = supplemental_values[feature_name]
                    elif feature_name in native_index:
                        if native_buf is not None:
                            output_arrays[output_name][coord_key] = native_buf[
                                :current_batch, native_index[feature_name]
                            ]

    results = {}
    for cls_name in texture_classes:
        results[cls_name] = collections.OrderedDict(
            (
                output_name,
                _array_to_image(output_array, image),
            )
            for output_name, output_array in all_output_arrays[cls_name].items()
        )
    return results


def _get_class_workers(context: VoxelContext, total_classes: int) -> int:
    if total_classes <= 1:
        return 1
    raw_value = os.environ.get("FLASH_RADIOMICS_VOXEL_CLASS_WORKERS", "").strip()
    if raw_value:
        try:
            parsed = int(raw_value)
            if parsed > 0:
                return max(1, min(total_classes, parsed))
        except ValueError:
            logger.warning(
                "Ignoring invalid FLASH_RADIOMICS_VOXEL_CLASS_WORKERS=%s",
                raw_value,
            )
    backend_name = str(context.backend or "cpu").strip().lower()
    if backend_name == "gpu":
        backend_name = "cuda"
    if backend_name in {"cuda", "mps"}:
        return 1
    cpu_count = context.num_threads if context.num_threads > 0 else (os.cpu_count() or 1)
    return max(1, min(total_classes, 2, cpu_count))


def _voxel_timing_enabled() -> bool:
    raw_value = os.environ.get("FLASH_RADIOMICS_VOXEL_TIMING", "").strip().lower()
    return raw_value in {"1", "true", "yes", "on"}


def _classify_roi_size(roi_voxels: int) -> str:
    if roi_voxels < 10_000:
        return "small"
    if roi_voxels < 100_000:
        return "medium"
    return "large"


def _get_voxel_telemetry(context: VoxelContext) -> collections.OrderedDict:
    return context.shared_state["telemetry"]


def _get_include_flags(
    context: VoxelContext,
    *,
    class_name: str,
    feature_order,
    feature_index_map,
    selected_feature_names,
):
    cache = context.shared_state["include_flags_cache"]
    cache_key = (class_name, tuple(selected_feature_names))
    include_flags = cache.get(cache_key)
    if include_flags is None:
        include_flags = np.zeros(len(feature_order), dtype=np.uint8)
        for feature_name in selected_feature_names:
            include_flags[feature_index_map[feature_name]] = 1
        cache[cache_key] = include_flags
    return include_flags


def _get_batch_patch_cache(
    context: VoxelContext,
    batch_slice: slice,
    *,
    need_image: bool = False,
    need_discretized: bool = False,
    need_mask: bool = False,
):
    cache_key = (int(batch_slice.start or 0), int(batch_slice.stop or 0))
    cache_lock = context.shared_state["batch_cache_lock"]
    batch_cache = context.shared_state["batch_cache"]
    with cache_lock:
        entry = batch_cache.get(cache_key)
        if entry is None:
            entry = {}
            batch_cache[cache_key] = entry
        cache_hit = (
            (not need_image or "image_patches" in entry)
            and (not need_discretized or "discretized_patches" in entry)
            and (not need_mask or "mask_patches_bool" in entry)
        )

    if cache_hit:
        return entry, True

    starts = context.window_starts[batch_slice]
    current_batch = int(starts.shape[0])
    new_arrays = {}
    if need_image and "image_patches" not in entry:
        image_patches = _gather_windows(context.image_windows, starts).reshape(current_batch, -1)
        if (
            not image_patches.flags.c_contiguous
            or image_patches.dtype != np.float64
        ):
            image_patches = np.ascontiguousarray(image_patches, dtype=np.float64)
        new_arrays["image_patches"] = image_patches
    if need_discretized and "discretized_patches" not in entry:
        discretized_patches = _gather_windows(
            context.discretized_windows,
            starts,
        ).reshape(current_batch, -1)
        if (
            not discretized_patches.flags.c_contiguous
            or discretized_patches.dtype != np.int32
        ):
            discretized_patches = np.ascontiguousarray(discretized_patches, dtype=np.int32)
        new_arrays["discretized_patches"] = discretized_patches
    if need_mask and "mask_patches_bool" not in entry:
        mask_patches_bool = _gather_windows(context.mask_windows, starts).reshape(
            current_batch,
            -1,
        )
        if not mask_patches_bool.flags.c_contiguous:
            mask_patches_bool = np.ascontiguousarray(mask_patches_bool)
        new_arrays["mask_patches_bool"] = mask_patches_bool
        new_arrays["mask_patches_u8"] = mask_patches_bool.view(np.uint8)

    with cache_lock:
        entry = batch_cache.setdefault(cache_key, {})
        entry.update(new_arrays)
        return entry, False


def _estimate_batch_bytes(batch_arrays) -> int:
    total_bytes = 0
    seen = set()
    for array in batch_arrays.values():
        if array is None or id(array) in seen:
            continue
        seen.add(id(array))
        total_bytes += int(array.nbytes)
    return total_bytes


def _record_voxel_batch_metrics(
    context: VoxelContext,
    class_name: str,
    *,
    gather_prep_s: float,
    native_call_s: float,
    output_assign_s: float,
    batch_bytes: int,
    cache_hit: bool,
    backend_label: str | None = None,
    extra_metrics: dict[str, float] | None = None,
):
    telemetry = _get_voxel_telemetry(context)
    telemetry_lock = context.shared_state["telemetry_lock"]
    with telemetry_lock:
        class_metrics = telemetry["classes"].setdefault(
            class_name,
            collections.OrderedDict(
                batch_count=0,
                max_batch_bytes=0,
                cache_hits=0,
                cache_misses=0,
                backend=backend_label or class_name.upper(),
                gather_prep_s=0.0,
                native_call_s=0.0,
                output_assign_s=0.0,
            ),
        )
        class_metrics["batch_count"] += 1
        class_metrics["max_batch_bytes"] = max(class_metrics["max_batch_bytes"], int(batch_bytes))
        class_metrics["cache_hits"] += int(cache_hit)
        class_metrics["cache_misses"] += int(not cache_hit)
        if backend_label is not None:
            class_metrics["backend"] = backend_label
        for stage_name, stage_value in (
            ("gather_prep_s", gather_prep_s),
            ("native_call_s", native_call_s),
            ("output_assign_s", output_assign_s),
        ):
            class_metrics[stage_name] += float(stage_value)
        if extra_metrics:
            for metric_name, metric_value in extra_metrics.items():
                class_metrics[metric_name] = float(class_metrics.get(metric_name, 0.0)) + float(metric_value)
        telemetry["max_batch_bytes"] = max(telemetry["max_batch_bytes"], int(batch_bytes))
        telemetry["cache_hits"] += int(cache_hit)
        telemetry["cache_misses"] += int(not cache_hit)


def _array_to_image(array: np.ndarray, reference_image: sitk.Image) -> sitk.Image:
    output = sitk.GetImageFromArray(np.asarray(array, dtype=np.float64))
    output.CopyInformation(reference_image)
    return output
