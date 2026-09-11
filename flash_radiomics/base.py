"""
Base classes for radiomics feature extraction
"""

import ctypes
import numpy as np
from collections import OrderedDict
from typing import Optional

from . import _bindings, imageoperations


class RadiomicsBase:
    """
    Base class for all radiomics feature classes

    Provides common functionality for feature extraction including
    C library interaction and result formatting.
    """

    def __init__(self, inputImage, inputMask, **kwargs):
        """
        Initialize base feature extractor

        Args:
            inputImage: numpy array of image data (2D or 3D)
            inputMask: numpy array of mask data (2D or 3D)
            **kwargs: Additional parameters including:
                - backend: cpu, mps, cuda, or auto
                - num_threads: number of threads for CPU
                - spacing: voxel/pixel spacing
                - origin: image origin
        """
        self.inputImage = inputImage
        self.inputMask = inputMask
        self.kwargs = kwargs

        self.ndim = inputImage.ndim
        self.dims = inputImage.shape[::-1]  # Reverse for C order
        self.spacing = kwargs.get("spacing", tuple([1.0] * self.ndim))
        self.origin = kwargs.get("origin", tuple([0.0] * self.ndim))

        self.backend = kwargs.get("backend", "auto")
        self.num_threads = kwargs.get("num_threads", 4)
        self._feature_lib = _bindings.get_feature_library(self.backend)

        backend_map = {
            "cpu": _bindings.FlashRadiomicsBackend.CPU,
            "mps": _bindings.FlashRadiomicsBackend.MPS,
            "cuda": _bindings.FlashRadiomicsBackend.CUDA,
            "auto": _bindings.FlashRadiomicsBackend.AUTO,
        }
        self.backend_enum = backend_map.get(
            self.backend.lower(), _bindings.FlashRadiomicsBackend.AUTO
        )
        if self._resolved_backend_name() == "cuda":
            set_bin_minimum = getattr(_bindings, "set_cuda_bin_minimum", None)
            if callable(set_bin_minimum):
                bin_minimum = kwargs.get("binMinimum")
                set_bin_minimum(None if bin_minimum is None else float(bin_minimum))

        self.featureValues = OrderedDict()
        self._enabled_features = None

    def _numpy_to_image(self, data: np.ndarray) -> ctypes.POINTER(
        _bindings.FlashRadiomicsImage
    ):
        """Convert numpy array to C FlashRadiomicsImage structure"""
        img = _bindings.FlashRadiomicsImage()
        img.ndim = self.ndim

        for i in range(self.ndim):
            img.dims[i] = self.dims[i]
        for i in range(self.ndim, 3):
            img.dims[i] = 1

        for i in range(self.ndim):
            img.spacing[i] = self.spacing[i]
            img.origin[i] = self.origin[i]
        for i in range(self.ndim, 3):
            img.spacing[i] = 1.0
            img.origin[i] = 0.0

        # Reuse pre-flattened shared buffers when provided by caller.
        shared_flat = self.kwargs.get("_image_flat_f32")
        expected_size = int(np.prod(data.shape, dtype=np.int64))
        if (
            isinstance(shared_flat, np.ndarray)
            and shared_flat.dtype == np.float32
            and shared_flat.flags.c_contiguous
            and shared_flat.size == expected_size
        ):
            data_flat = shared_flat
        else:
            data_flat = np.ascontiguousarray(data, dtype=np.float32).reshape(-1)
        img.data = data_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        # The C pointer remains valid only while this NumPy reference is alive.
        img._data_ref = data_flat

        img.error_code = 0

        return ctypes.pointer(img)

    def _numpy_to_mask(
        self, data: np.ndarray, label: int = 1
    ) -> ctypes.POINTER(_bindings.FlashRadiomicsMask):
        """Convert numpy array to C FlashRadiomicsMask structure"""
        mask = _bindings.FlashRadiomicsMask()
        mask.ndim = self.ndim
        mask.label = label

        for i in range(self.ndim):
            mask.dims[i] = self.dims[i]
        for i in range(self.ndim, 3):
            mask.dims[i] = 1

        # Reuse pre-flattened shared buffers when provided by caller.
        shared_flat = self.kwargs.get("_mask_flat_u8")
        expected_size = int(np.prod(data.shape, dtype=np.int64))
        if (
            isinstance(shared_flat, np.ndarray)
            and shared_flat.dtype == np.uint8
            and shared_flat.flags.c_contiguous
            and shared_flat.size == expected_size
        ):
            data_flat = shared_flat
        else:
            data_flat = np.ascontiguousarray(data, dtype=np.uint8).reshape(-1)
        mask.data = data_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))

        # The C pointer remains valid only while this NumPy reference is alive.
        mask._data_ref = data_flat

        mask.error_code = 0

        return ctypes.pointer(mask)

    def _result_to_dict(
        self,
        result_ptr,
        prefix: str,
        strip_prefix: str | None = None,
        feature_lib=None,
    ) -> OrderedDict:
        """Convert C result to OrderedDict"""
        if not result_ptr:
            raise _bindings.FlashRadiomicsError(
                f"{prefix} feature extraction returned NULL"
            )

        result = result_ptr.contents
        _bindings._check_error(result.error_code, result.error_msg)

        features = OrderedDict()
        for i in range(result.count):
            feature = result.features[i]
            name = feature.name.decode("utf-8")
            if strip_prefix and name.startswith(strip_prefix):
                name = name[len(strip_prefix) :]
            features[name] = feature.value

        (feature_lib or self._feature_lib).flash_radiomics_free_result(result_ptr)

        return self._filter_features(features)

    def _get_binimage_discretized(self) -> tuple[np.ndarray | None, int]:
        discretized = self.kwargs.get("discretized")
        ng = int(self.kwargs.get("ng", 0) or 0)
        if discretized is not None and ng > 0:
            return np.ascontiguousarray(np.asarray(discretized, dtype=np.int32)), ng

        roi_mask = np.asarray(self.inputMask) != 0
        if not np.any(roi_mask):
            return None, 0

        binning_kwargs = {"binWidth": float(self.kwargs.get("binWidth", 25))}
        if self.kwargs.get("binMinimum") is not None:
            binning_kwargs["binMinimum"] = float(self.kwargs["binMinimum"])
        bin_count = int(self.kwargs.get("binCount", 0) or 0)
        if bin_count > 0:
            binning_kwargs["binCount"] = bin_count

        discretized, _ = imageoperations.binImage(
            np.asarray(self.inputImage, dtype=np.float64),
            roi_mask,
            **binning_kwargs,
        )
        discretized = np.ascontiguousarray(np.asarray(discretized, dtype=np.int32))
        return discretized, int(np.max(discretized[roi_mask]))

    @staticmethod
    def _get_cpu_feature_lib():
        return _bindings.get_feature_library("cpu")

    @staticmethod
    def _normalize_backend_name(backend: str) -> str:
        name = (backend or "auto").strip().lower()
        return "cuda" if name == "gpu" else name

    def _resolved_backend_name(self) -> str:
        backend_name = self._normalize_backend_name(self.backend)
        if backend_name in {"cpu", "mps", "cuda"}:
            return backend_name

        try:
            available = _bindings.detect_backends()
        except Exception:
            return "cpu"

        if available.get("cuda", False):
            return "cuda"
        if available.get("mps", False):
            return "mps"
        return "cpu"

    def _should_use_cpu_discretized_path(self) -> bool:
        return self._resolved_backend_name() == "cpu"

    def execute(self):
        """
        Execute feature extraction

        Must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement execute()")

    def enableAllFeatures(self):
        """Enable all features in this class"""
        self._enabled_features = None

    def disableAllFeatures(self):
        """Disable all features in this class"""
        self.featureValues = OrderedDict()
        self._enabled_features = set()

    def enableFeatureByName(self, featureName: str):
        if self._enabled_features is None:
            self._enabled_features = set()
        self._enabled_features.add(featureName)

    def _filter_features(self, features: OrderedDict) -> OrderedDict:
        if not self._enabled_features:
            return features
        return OrderedDict((k, v) for k, v in features.items() if k in self._enabled_features)


class RadiomicsFeaturesBase(RadiomicsBase):
    """
    Alias base class used for feature class discovery.
    """

    pass
