"""
C library bindings using ctypes

This module provides Python bindings to the flash_radiomics C library,
wrapping all C API functions and data structures.
"""

import ctypes
import os
import sys
import platform
from typing import Optional, Tuple
import numpy as np


def _normalize_arch(machine: str) -> str:
    machine = (machine or "").lower()
    if machine in ("arm64", "arm64e", "aarch64"):
        return "arm64"
    if machine in ("x86_64", "amd64", "i386", "i686"):
        return "x86_64"
    return machine or "unknown"


def _dedupe_paths(paths):
    seen = set()
    out = []
    for path in paths:
        if not path:
            continue
        norm = os.path.normpath(path)
        if norm in seen:
            continue
        seen.add(norm)
        out.append(norm)
    return out


def _find_library():
    """Find architecture-compatible flash_radiomics shared libraries."""
    package_dir = os.path.dirname(os.path.abspath(__file__))
    flash_radiomics_pkg = os.path.dirname(package_dir)
    arch = _normalize_arch(platform.machine())

    if platform.system() == "Darwin":
        lib_cpu = "libflash_radiomics_cpu.dylib"
        lib_mps = "libflash_radiomics_mps.dylib"
        lib_cuda = "libflash_radiomics_cuda.dylib"
    elif platform.system() == "Linux":
        lib_cpu = "libflash_radiomics_cpu.so"
        lib_mps = "libflash_radiomics_mps.so"
        lib_cuda = "libflash_radiomics_cuda.so"
    elif platform.system() == "Windows":
        lib_cpu = "flash_radiomics_cpu.dll"
        lib_mps = "flash_radiomics_mps.dll"
        lib_cuda = "flash_radiomics_cuda.dll"
    else:
        raise RuntimeError(f"Unsupported platform: {platform.system()}")

    env_lib_dir = os.environ.get("FLASH_RADIOMICS_LIB_DIR")
    env_build_dir = os.environ.get("FLASH_RADIOMICS_BUILD_DIR")
    candidate_dirs = _dedupe_paths(
        [
            env_lib_dir,
            env_build_dir,
            package_dir,
            os.path.join(flash_radiomics_pkg, "build", arch),
            os.path.join(flash_radiomics_pkg, f"build_{arch}"),
            os.path.join(flash_radiomics_pkg, "build"),
        ]
    )

    for build_dir in candidate_dirs:
        cpu_path = os.path.join(build_dir, lib_cpu)
        if not os.path.exists(cpu_path):
            continue
        mps_path = os.path.join(build_dir, lib_mps)
        cuda_path = os.path.join(build_dir, lib_cuda)
        return (
            cpu_path,
            mps_path if os.path.exists(mps_path) else None,
            cuda_path if os.path.exists(cuda_path) else None,
        )

    raise RuntimeError(
        "CPU library not found for this Python architecture. "
        f"Tried: {candidate_dirs}. Set FLASH_RADIOMICS_BUILD_DIR if needed."
    )


_lib_cpu_path, _lib_mps_path, _lib_cuda_path = _find_library()
_lib_cpu = ctypes.CDLL(_lib_cpu_path)


def _load_optional_library(path):
    """Load an optional accelerator library without disabling CPU use."""
    if not path:
        return None
    try:
        return ctypes.CDLL(path)
    except OSError:
        # Keep CPU available if the accelerator library cannot load.
        return None


_lib_mps = _load_optional_library(_lib_mps_path)
_lib_cuda = _load_optional_library(_lib_cuda_path)


# Enums
class FlashRadiomicsBackend(ctypes.c_int):
    """Backend enumeration"""
    CPU = 0
    MPS = 1
    CUDA = 2
    AUTO = 3


class FlashRadiomicsImageFormat(ctypes.c_int):
    """Image format enumeration"""
    NIFTI = 0
    DICOM_SERIES = 1
    AUTO = 2


class FlashRadiomicsErrorCode(ctypes.c_int):
    """Error code enumeration"""
    SUCCESS = 0
    FILE_NOT_FOUND = 1
    INVALID_FORMAT = 2
    DIMENSION_MISMATCH = 3
    MEMORY_ALLOCATION = 4
    INVALID_PARAMETER = 5
    COMPUTATION_FAILED = 6
    BACKEND_UNAVAILABLE = 7
    IO_FAILED = 8
    CORRUPTED_DATA = 9
    UNSUPPORTED_OPERATION = 10


# Structures
class FlashRadiomicsImage(ctypes.Structure):
    """Image data structure"""
    _fields_ = [
        ('data', ctypes.POINTER(ctypes.c_float)),
        ('ndim', ctypes.c_int),
        ('dims', ctypes.c_int * 3),
        ('spacing', ctypes.c_float * 3),
        ('origin', ctypes.c_float * 3),
        ('error_code', ctypes.c_int),
        ('error_msg', ctypes.c_char * 256),
    ]


class FlashRadiomicsMask(ctypes.Structure):
    """Mask data structure"""
    _fields_ = [
        ('data', ctypes.POINTER(ctypes.c_uint8)),
        ('ndim', ctypes.c_int),
        ('dims', ctypes.c_int * 3),
        ('label', ctypes.c_int),
        ('error_code', ctypes.c_int),
        ('error_msg', ctypes.c_char * 256),
    ]


class FlashRadiomicsFeature(ctypes.Structure):
    """Feature result structure"""
    _fields_ = [
        ('name', ctypes.c_char * 64),
        ('value', ctypes.c_double),
    ]


class FlashRadiomicsResult(ctypes.Structure):
    """Result structure"""
    _fields_ = [
        ('features', ctypes.POINTER(FlashRadiomicsFeature)),
        ('count', ctypes.c_int),
        ('compute_time_ms', ctypes.c_double),
        ('error_code', ctypes.c_int),
        ('error_msg', ctypes.c_char * 256),
    ]


class CudaDeviceInfo(ctypes.Structure):
    """CUDA device information structure"""
    _fields_ = [
        ('device_id', ctypes.c_int),
        ('device_name', ctypes.c_char * 256),
        ('compute_capability_major', ctypes.c_int),
        ('compute_capability_minor', ctypes.c_int),
        ('total_memory', ctypes.c_size_t),
        ('free_memory', ctypes.c_size_t),
        ('multiprocessor_count', ctypes.c_int),
        ('max_threads_per_block', ctypes.c_int),
        ('is_available', ctypes.c_int),
    ]


class CudaContextStruct(ctypes.Structure):
    """CUDA context structure (opaque pointer in Python)"""
    pass


class FlashRadiomicsContext(ctypes.Structure):
    """Execution context structure"""
    _fields_ = [
        ('backend', ctypes.c_int),
        ('num_threads', ctypes.c_int),
        ('device_handle', ctypes.c_void_p),
        ('cuda_ctx', ctypes.POINTER(CudaContextStruct)),
        ('available_backends', ctypes.c_int),
        ('initialized', ctypes.c_int),
    ]


# Exception classes
class FlashRadiomicsError(Exception):
    """Base exception for flash_radiomics errors"""
    pass


class FlashRadiomicsFileNotFoundError(FlashRadiomicsError):
    """File not found error"""
    pass


class FlashRadiomicsInvalidFormatError(FlashRadiomicsError):
    """Invalid format error"""
    pass


class FlashRadiomicsDimensionMismatchError(FlashRadiomicsError):
    """Dimension mismatch error"""
    pass


class FlashRadiomicsMemoryError(FlashRadiomicsError):
    """Memory allocation error"""
    pass


class FlashRadiomicsInvalidParameterError(FlashRadiomicsError):
    """Invalid parameter error"""
    pass


class FlashRadiomicsComputationError(FlashRadiomicsError):
    """Computation failed error"""
    pass


class FlashRadiomicsBackendUnavailableError(FlashRadiomicsError):
    """Backend unavailable error"""
    pass


class FlashRadiomicsIOError(FlashRadiomicsError):
    """I/O error"""
    pass


class FlashRadiomicsCorruptedDataError(FlashRadiomicsError):
    """Corrupted data error"""
    pass


class FlashRadiomicsUnsupportedOperationError(FlashRadiomicsError):
    """Unsupported operation error"""
    pass


# Error code to exception mapping
_ERROR_MAP = {
    FlashRadiomicsErrorCode.FILE_NOT_FOUND: FlashRadiomicsFileNotFoundError,
    FlashRadiomicsErrorCode.INVALID_FORMAT: FlashRadiomicsInvalidFormatError,
    FlashRadiomicsErrorCode.DIMENSION_MISMATCH: FlashRadiomicsDimensionMismatchError,
    FlashRadiomicsErrorCode.MEMORY_ALLOCATION: FlashRadiomicsMemoryError,
    FlashRadiomicsErrorCode.INVALID_PARAMETER: FlashRadiomicsInvalidParameterError,
    FlashRadiomicsErrorCode.COMPUTATION_FAILED: FlashRadiomicsComputationError,
    FlashRadiomicsErrorCode.BACKEND_UNAVAILABLE: FlashRadiomicsBackendUnavailableError,
    FlashRadiomicsErrorCode.IO_FAILED: FlashRadiomicsIOError,
    FlashRadiomicsErrorCode.CORRUPTED_DATA: FlashRadiomicsCorruptedDataError,
    FlashRadiomicsErrorCode.UNSUPPORTED_OPERATION: FlashRadiomicsUnsupportedOperationError,
}


def _check_error(error_code: int, error_msg: bytes):
    """Convert C error code to Python exception"""
    if error_code == FlashRadiomicsErrorCode.SUCCESS:
        return
    
    msg = error_msg.decode('utf-8') if error_msg else "Unknown error"
    exception_class = _ERROR_MAP.get(error_code, FlashRadiomicsError)
    raise exception_class(f"Error {error_code}: {msg}")


# Image I/O functions
_lib_cpu.flash_radiomics_load_image.argtypes = [ctypes.c_char_p, ctypes.c_int]
_lib_cpu.flash_radiomics_load_image.restype = ctypes.POINTER(FlashRadiomicsImage)

_lib_cpu.flash_radiomics_load_mask.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
_lib_cpu.flash_radiomics_load_mask.restype = ctypes.POINTER(FlashRadiomicsMask)

_lib_cpu.flash_radiomics_extract_slice.argtypes = [ctypes.POINTER(FlashRadiomicsImage), ctypes.c_int, ctypes.c_int]
_lib_cpu.flash_radiomics_extract_slice.restype = ctypes.POINTER(FlashRadiomicsImage)

_lib_cpu.flash_radiomics_extract_mask_slice.argtypes = [ctypes.POINTER(FlashRadiomicsMask), ctypes.c_int, ctypes.c_int]
_lib_cpu.flash_radiomics_extract_mask_slice.restype = ctypes.POINTER(FlashRadiomicsMask)

_lib_cpu.flash_radiomics_validate_inputs.argtypes = [ctypes.POINTER(FlashRadiomicsImage), ctypes.POINTER(FlashRadiomicsMask)]
_lib_cpu.flash_radiomics_validate_inputs.restype = ctypes.c_int

_lib_cpu.flash_radiomics_free_image.argtypes = [ctypes.POINTER(FlashRadiomicsImage)]
_lib_cpu.flash_radiomics_free_image.restype = None

_lib_cpu.flash_radiomics_free_mask.argtypes = [ctypes.POINTER(FlashRadiomicsMask)]
_lib_cpu.flash_radiomics_free_mask.restype = None

_lib_cpu.flash_radiomics_free.argtypes = [ctypes.c_void_p]
_lib_cpu.flash_radiomics_free.restype = None

# Context functions
_lib_cpu.flash_radiomics_init_context.argtypes = [ctypes.c_int]
_lib_cpu.flash_radiomics_init_context.restype = ctypes.POINTER(FlashRadiomicsContext)

_lib_cpu.flash_radiomics_detect_backends.argtypes = []
_lib_cpu.flash_radiomics_detect_backends.restype = ctypes.c_int

_lib_cpu.flash_radiomics_free_context.argtypes = [ctypes.POINTER(FlashRadiomicsContext)]
_lib_cpu.flash_radiomics_free_context.restype = None

if hasattr(_lib_cpu, "flash_radiomics_cuda_reset_segment_workspace"):
    _lib_cpu.flash_radiomics_cuda_reset_segment_workspace.argtypes = []
    _lib_cpu.flash_radiomics_cuda_reset_segment_workspace.restype = None

if hasattr(_lib_cpu, "flash_radiomics_cuda_segment_workspace_required_bytes"):
    _lib_cpu.flash_radiomics_cuda_segment_workspace_required_bytes.argtypes = [
        ctypes.POINTER(FlashRadiomicsImage),
        ctypes.POINTER(FlashRadiomicsMask),
    ]
    _lib_cpu.flash_radiomics_cuda_segment_workspace_required_bytes.restype = ctypes.c_size_t

if hasattr(_lib_cpu, "flash_radiomics_cuda_set_deterministic_mode"):
    _lib_cpu.flash_radiomics_cuda_set_deterministic_mode.argtypes = [ctypes.c_int]
    _lib_cpu.flash_radiomics_cuda_set_deterministic_mode.restype = None

if hasattr(_lib_cpu, "flash_radiomics_cuda_get_deterministic_mode"):
    _lib_cpu.flash_radiomics_cuda_get_deterministic_mode.argtypes = []
    _lib_cpu.flash_radiomics_cuda_get_deterministic_mode.restype = ctypes.c_int

if hasattr(_lib_cpu, "flash_radiomics_cuda_configure_glszm_determinism"):
    _lib_cpu.flash_radiomics_cuda_configure_glszm_determinism.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    _lib_cpu.flash_radiomics_cuda_configure_glszm_determinism.restype = None

if hasattr(_lib_cpu, "flash_radiomics_cuda_set_bin_minimum"):
    _lib_cpu.flash_radiomics_cuda_set_bin_minimum.argtypes = [
        ctypes.c_int,
        ctypes.c_double,
    ]
    _lib_cpu.flash_radiomics_cuda_set_bin_minimum.restype = None

# Feature extraction functions
_lib_cpu.flash_radiomics_firstorder.argtypes = [
    ctypes.POINTER(FlashRadiomicsImage),
    ctypes.POINTER(FlashRadiomicsMask),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_double,
    ctypes.c_int,
    ctypes.c_double
]
_lib_cpu.flash_radiomics_firstorder.restype = ctypes.POINTER(FlashRadiomicsResult)

_lib_cpu.flash_radiomics_glcm.argtypes = [
    ctypes.POINTER(FlashRadiomicsImage),
    ctypes.POINTER(FlashRadiomicsMask),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_int,
    ctypes.c_double,
    ctypes.c_int
]
_lib_cpu.flash_radiomics_glcm.restype = ctypes.POINTER(FlashRadiomicsResult)

_lib_cpu.flash_radiomics_glrlm.argtypes = [
    ctypes.POINTER(FlashRadiomicsImage),
    ctypes.POINTER(FlashRadiomicsMask),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_double,
    ctypes.c_int
]
_lib_cpu.flash_radiomics_glrlm.restype = ctypes.POINTER(FlashRadiomicsResult)

_lib_cpu.flash_radiomics_glszm.argtypes = [
    ctypes.POINTER(FlashRadiomicsImage),
    ctypes.POINTER(FlashRadiomicsMask),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_double,
    ctypes.c_int
]
_lib_cpu.flash_radiomics_glszm.restype = ctypes.POINTER(FlashRadiomicsResult)

_lib_cpu.flash_radiomics_gldm.argtypes = [
    ctypes.POINTER(FlashRadiomicsImage),
    ctypes.POINTER(FlashRadiomicsMask),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_int,
    ctypes.c_double,
    ctypes.c_int,
    ctypes.c_double,
    ctypes.c_int
]
_lib_cpu.flash_radiomics_gldm.restype = ctypes.POINTER(FlashRadiomicsResult)

_lib_cpu.flash_radiomics_ngtdm.argtypes = [
    ctypes.POINTER(FlashRadiomicsImage),
    ctypes.POINTER(FlashRadiomicsMask),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_int,
    ctypes.c_double,
    ctypes.c_int,
    ctypes.c_int
]
_lib_cpu.flash_radiomics_ngtdm.restype = ctypes.POINTER(FlashRadiomicsResult)

_lib_cpu.flash_radiomics_discretize_image.argtypes = [
    ctypes.POINTER(FlashRadiomicsImage),
    ctypes.POINTER(FlashRadiomicsMask),
    ctypes.c_double,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int),
]
_lib_cpu.flash_radiomics_discretize_image.restype = ctypes.POINTER(ctypes.c_int)

_lib_cpu.flash_radiomics_firstorder_cpu_discretized.argtypes = [
    ctypes.POINTER(FlashRadiomicsImage),
    ctypes.POINTER(FlashRadiomicsMask),
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_int,
    ctypes.c_double,
]
_lib_cpu.flash_radiomics_firstorder_cpu_discretized.restype = ctypes.POINTER(
    FlashRadiomicsResult
)

_lib_cpu.flash_radiomics_glcm_cpu_discretized.argtypes = [
    ctypes.POINTER(FlashRadiomicsImage),
    ctypes.POINTER(FlashRadiomicsMask),
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_int,
]
_lib_cpu.flash_radiomics_glcm_cpu_discretized.restype = ctypes.POINTER(
    FlashRadiomicsResult
)

if hasattr(_lib_cpu, "flash_radiomics_glcm_cpu_discretized_selective"):
    _lib_cpu.flash_radiomics_glcm_cpu_discretized_selective.argtypes = [
        ctypes.POINTER(FlashRadiomicsImage),
        ctypes.POINTER(FlashRadiomicsMask),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
        ctypes.c_int,
    ]
    _lib_cpu.flash_radiomics_glcm_cpu_discretized_selective.restype = ctypes.POINTER(
        FlashRadiomicsResult
    )

if hasattr(_lib_cpu, "flash_radiomics_glcm_cpu_discretized_voxel_batch"):
    _lib_cpu.flash_radiomics_glcm_cpu_discretized_voxel_batch.argtypes = [
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
    ]
    _lib_cpu.flash_radiomics_glcm_cpu_discretized_voxel_batch.restype = ctypes.c_int

_lib_cpu.flash_radiomics_glrlm_cpu_discretized.argtypes = [
    ctypes.POINTER(FlashRadiomicsImage),
    ctypes.POINTER(FlashRadiomicsMask),
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_int,
]
_lib_cpu.flash_radiomics_glrlm_cpu_discretized.restype = ctypes.POINTER(
    FlashRadiomicsResult
)

if hasattr(_lib_cpu, "flash_radiomics_glrlm_cpu_discretized_voxel_batch"):
    _lib_cpu.flash_radiomics_glrlm_cpu_discretized_voxel_batch.argtypes = [
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
    ]
    _lib_cpu.flash_radiomics_glrlm_cpu_discretized_voxel_batch.restype = ctypes.c_int

_lib_cpu.flash_radiomics_glszm_cpu_discretized.argtypes = [
    ctypes.POINTER(FlashRadiomicsImage),
    ctypes.POINTER(FlashRadiomicsMask),
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_int,
]
_lib_cpu.flash_radiomics_glszm_cpu_discretized.restype = ctypes.POINTER(
    FlashRadiomicsResult
)

if hasattr(_lib_cpu, "flash_radiomics_glszm_cpu_discretized_voxel_batch"):
    _lib_cpu.flash_radiomics_glszm_cpu_discretized_voxel_batch.argtypes = [
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
    ]
    _lib_cpu.flash_radiomics_glszm_cpu_discretized_voxel_batch.restype = ctypes.c_int

_lib_cpu.flash_radiomics_gldm_cpu_discretized.argtypes = [
    ctypes.POINTER(FlashRadiomicsImage),
    ctypes.POINTER(FlashRadiomicsMask),
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_int,
    ctypes.c_double,
]
_lib_cpu.flash_radiomics_gldm_cpu_discretized.restype = ctypes.POINTER(
    FlashRadiomicsResult
)

if hasattr(_lib_cpu, "flash_radiomics_gldm_cpu_discretized_voxel_batch"):
    _lib_cpu.flash_radiomics_gldm_cpu_discretized_voxel_batch.argtypes = [
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_int,
        ctypes.c_double,
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
    ]
    _lib_cpu.flash_radiomics_gldm_cpu_discretized_voxel_batch.restype = ctypes.c_int

if hasattr(_lib_cpu, "flash_radiomics_ngtdm_cpu_discretized_voxel_batch"):
    _lib_cpu.flash_radiomics_ngtdm_cpu_discretized_voxel_batch.argtypes = [
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
    ]
    _lib_cpu.flash_radiomics_ngtdm_cpu_discretized_voxel_batch.restype = ctypes.c_int

_lib_cpu.flash_radiomics_ngtdm_cpu_discretized.argtypes = [
    ctypes.POINTER(FlashRadiomicsImage),
    ctypes.POINTER(FlashRadiomicsMask),
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_int,
]
_lib_cpu.flash_radiomics_ngtdm_cpu_discretized.restype = ctypes.POINTER(
    FlashRadiomicsResult
)

_lib_cpu.flash_radiomics_shape.argtypes = [
    ctypes.POINTER(FlashRadiomicsMask),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int
]
_lib_cpu.flash_radiomics_shape.restype = ctypes.POINTER(FlashRadiomicsResult)

_lib_cpu.flash_radiomics_shape2d.argtypes = [
    ctypes.POINTER(FlashRadiomicsMask),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int
]
_lib_cpu.flash_radiomics_shape2d.restype = ctypes.POINTER(FlashRadiomicsResult)

_lib_cpu.flash_radiomics_free_result.argtypes = [ctypes.POINTER(FlashRadiomicsResult)]
_lib_cpu.flash_radiomics_free_result.restype = None

# Mirror C signatures onto MPS library when available.
if _lib_mps:
    _mps_signature_names = [
        "flash_radiomics_load_image",
        "flash_radiomics_load_mask",
        "flash_radiomics_extract_slice",
        "flash_radiomics_extract_mask_slice",
        "flash_radiomics_validate_inputs",
        "flash_radiomics_free_image",
        "flash_radiomics_free_mask",
        "flash_radiomics_free",
        "flash_radiomics_init_context",
        "flash_radiomics_detect_backends",
        "flash_radiomics_free_context",
        "flash_radiomics_cuda_reset_segment_workspace",
        "flash_radiomics_cuda_segment_workspace_required_bytes",
        "flash_radiomics_cuda_set_deterministic_mode",
        "flash_radiomics_cuda_get_deterministic_mode",
        "flash_radiomics_cuda_configure_glszm_determinism",
        "flash_radiomics_cuda_set_bin_minimum",
        "flash_radiomics_firstorder",
        "flash_radiomics_glcm",
        "flash_radiomics_glrlm",
        "flash_radiomics_glszm",
        "flash_radiomics_gldm",
        "flash_radiomics_ngtdm",
        "flash_radiomics_discretize_image",
        "flash_radiomics_firstorder_cpu_discretized",
        "flash_radiomics_glcm_cpu_discretized",
        "flash_radiomics_glrlm_cpu_discretized",
        "flash_radiomics_glszm_cpu_discretized",
        "flash_radiomics_gldm_cpu_discretized",
        "flash_radiomics_ngtdm_cpu_discretized",
        "flash_radiomics_glcm_cpu_discretized_voxel_batch",
        "flash_radiomics_glrlm_cpu_discretized_voxel_batch",
        "flash_radiomics_glszm_cpu_discretized_voxel_batch",
        "flash_radiomics_gldm_cpu_discretized_voxel_batch",
        "flash_radiomics_ngtdm_cpu_discretized_voxel_batch",
        "flash_radiomics_shape",
        "flash_radiomics_shape2d",
        "flash_radiomics_free_result",
    ]
    for _fname in _mps_signature_names:
        if not hasattr(_lib_mps, _fname):
            continue
        _src_fn = getattr(_lib_cpu, _fname, None)
        if _src_fn is None:
            continue
        _dst_fn = getattr(_lib_mps, _fname)
        _dst_fn.argtypes = getattr(_src_fn, "argtypes", None)
        _dst_fn.restype = getattr(_src_fn, "restype", None)

# CUDA functions (only if CUDA library is available)
if _lib_cuda:
    _cuda_signature_names = [
        "flash_radiomics_load_image",
        "flash_radiomics_load_mask",
        "flash_radiomics_extract_slice",
        "flash_radiomics_extract_mask_slice",
        "flash_radiomics_validate_inputs",
        "flash_radiomics_free_image",
        "flash_radiomics_free_mask",
        "flash_radiomics_free",
        "flash_radiomics_init_context",
        "flash_radiomics_detect_backends",
        "flash_radiomics_free_context",
        "flash_radiomics_cuda_reset_segment_workspace",
        "flash_radiomics_cuda_segment_workspace_required_bytes",
        "flash_radiomics_cuda_set_deterministic_mode",
        "flash_radiomics_cuda_get_deterministic_mode",
        "flash_radiomics_cuda_configure_glszm_determinism",
        "flash_radiomics_cuda_set_bin_minimum",
        "flash_radiomics_firstorder",
        "flash_radiomics_glcm",
        "flash_radiomics_glrlm",
        "flash_radiomics_glszm",
        "flash_radiomics_gldm",
        "flash_radiomics_ngtdm",
        "flash_radiomics_discretize_image",
        "flash_radiomics_firstorder_cpu_discretized",
        "flash_radiomics_glcm_cpu_discretized",
        "flash_radiomics_glrlm_cpu_discretized",
        "flash_radiomics_glszm_cpu_discretized",
        "flash_radiomics_gldm_cpu_discretized",
        "flash_radiomics_ngtdm_cpu_discretized",
        "flash_radiomics_glcm_cpu_discretized_voxel_batch",
        "flash_radiomics_glrlm_cpu_discretized_voxel_batch",
        "flash_radiomics_glszm_cpu_discretized_voxel_batch",
        "flash_radiomics_gldm_cpu_discretized_voxel_batch",
        "flash_radiomics_ngtdm_cpu_discretized_voxel_batch",
        "flash_radiomics_shape",
        "flash_radiomics_shape2d",
        "flash_radiomics_free_result",
    ]
    for _fname in _cuda_signature_names:
        if not hasattr(_lib_cuda, _fname):
            continue
        _src_fn = getattr(_lib_cpu, _fname, None)
        if _src_fn is None:
            continue
        _dst_fn = getattr(_lib_cuda, _fname)
        _dst_fn.argtypes = getattr(_src_fn, "argtypes", None)
        _dst_fn.restype = getattr(_src_fn, "restype", None)

    _cuda_voxel_aliases = {
        "flash_radiomics_glcm_cuda_discretized_voxel_batch":
            "flash_radiomics_glcm_cpu_discretized_voxel_batch",
        "flash_radiomics_glrlm_cuda_discretized_voxel_batch":
            "flash_radiomics_glrlm_cpu_discretized_voxel_batch",
        "flash_radiomics_glszm_cuda_discretized_voxel_batch":
            "flash_radiomics_glszm_cpu_discretized_voxel_batch",
        "flash_radiomics_gldm_cuda_discretized_voxel_batch":
            "flash_radiomics_gldm_cpu_discretized_voxel_batch",
        "flash_radiomics_ngtdm_cuda_discretized_voxel_batch":
            "flash_radiomics_ngtdm_cpu_discretized_voxel_batch",
    }
    for _cuda_name, _cpu_name in _cuda_voxel_aliases.items():
        if not hasattr(_lib_cuda, _cuda_name):
            continue
        _src_fn = getattr(_lib_cpu, _cpu_name, None)
        if _src_fn is None:
            continue
        _dst_fn = getattr(_lib_cuda, _cuda_name)
        _dst_fn.argtypes = getattr(_src_fn, "argtypes", None)
        _dst_fn.restype = getattr(_src_fn, "restype", None)

    _lib_cuda.cuda_detect_device.argtypes = [ctypes.POINTER(CudaDeviceInfo)]
    _lib_cuda.cuda_detect_device.restype = ctypes.c_int
    
    _lib_cuda.cuda_init_context.argtypes = [ctypes.c_int]
    _lib_cuda.cuda_init_context.restype = ctypes.POINTER(CudaContextStruct)
    
    _lib_cuda.cuda_free_context.argtypes = [ctypes.POINTER(CudaContextStruct)]
    _lib_cuda.cuda_free_context.restype = None
    
    _lib_cuda.cuda_get_error_string.argtypes = [ctypes.c_int]
    _lib_cuda.cuda_get_error_string.restype = ctypes.c_char_p

    # Unified multi-class CUDA voxel batch dispatch.
    if hasattr(_lib_cuda, "flash_radiomics_voxel_multi_class_cuda_batch"):
        _lib_cuda.flash_radiomics_voxel_multi_class_cuda_batch.argtypes = [
            # Shared input
            ctypes.POINTER(ctypes.c_int),     # discretized_windows
            ctypes.POINTER(ctypes.c_uint8),   # mask_windows
            ctypes.c_int,                     # batch_size
            ctypes.c_int,                     # ndim
            ctypes.POINTER(ctypes.c_int),     # window_dims[3]
            ctypes.c_int,                     # ng
            ctypes.c_uint,                    # class_mask
            # Shared distances
            ctypes.POINTER(ctypes.c_int),     # distances
            ctypes.c_int,                     # num_distances
            # GLDM-specific
            ctypes.c_double,                  # gldm_a
            # GLCM output
            ctypes.POINTER(ctypes.c_uint8),   # glcm_include_features
            ctypes.POINTER(ctypes.c_double),  # glcm_out_features
            ctypes.c_int,                     # glcm_out_feature_stride
            # GLRLM output
            ctypes.POINTER(ctypes.c_uint8),   # glrlm_include_features
            ctypes.POINTER(ctypes.c_double),  # glrlm_out_features
            ctypes.c_int,                     # glrlm_out_feature_stride
            # GLSZM output
            ctypes.POINTER(ctypes.c_uint8),   # glszm_include_features
            ctypes.POINTER(ctypes.c_double),  # glszm_out_features
            ctypes.c_int,                     # glszm_out_feature_stride
            # GLDM output
            ctypes.POINTER(ctypes.c_uint8),   # gldm_include_features
            ctypes.POINTER(ctypes.c_double),  # gldm_out_features
            ctypes.c_int,                     # gldm_out_feature_stride
            # NGTDM output
            ctypes.POINTER(ctypes.c_uint8),   # ngtdm_include_features
            ctypes.POINTER(ctypes.c_double),  # ngtdm_out_features
            ctypes.c_int,                     # ngtdm_out_feature_stride
        ]
        _lib_cuda.flash_radiomics_voxel_multi_class_cuda_batch.restype = ctypes.c_int

# Class bitmask constants matching voxel_multi_class_cuda.h
FLASH_VOXEL_CLASS_GLCM = 1 << 0
FLASH_VOXEL_CLASS_GLRLM = 1 << 1
FLASH_VOXEL_CLASS_GLSZM = 1 << 2
FLASH_VOXEL_CLASS_GLDM = 1 << 3
FLASH_VOXEL_CLASS_NGTDM = 1 << 4



def get_feature_library(backend: Optional[str] = None):
    """
    Return the ctypes library handle used for feature extraction dispatch.

    Explicit accelerator requests select their library. Auto selects an
    available CUDA, MPS, or CPU library in that order; native AUTO dispatch
    can still choose CPU for smaller workloads.
    """
    backend_name = (backend or "auto").strip().lower()
    if backend_name == "auto":
        available = detect_backends()
        if available.get("cuda") and _lib_cuda is not None:
            return _lib_cuda
        if available.get("mps") and _lib_mps is not None:
            return _lib_mps
        return _lib_cpu
    if backend_name == "gpu":
        backend_name = "cuda"
    if backend_name == "cuda":
        if _lib_cuda is None:
            raise FlashRadiomicsBackendUnavailableError(
                "CUDA backend requested, but libflash_radiomics_cuda was not found for this architecture."
            )
        return _lib_cuda
    if backend_name == "mps":
        if _lib_mps is None:
            raise FlashRadiomicsBackendUnavailableError(
                "MPS backend requested, but libflash_radiomics_mps was not found for this architecture."
            )
        return _lib_mps
    return _lib_cpu


def get_loaded_library_paths() -> dict:
    """Return resolved shared-library paths currently loaded by ctypes."""
    return {
        "cpu": _lib_cpu_path,
        "mps": _lib_mps_path,
        "cuda": _lib_cuda_path,
    }


def discretize_image(
    image_array: np.ndarray,
    mask_array: np.ndarray,
    spacing: Optional[Tuple[float, ...]] = None,
    origin: Optional[Tuple[float, ...]] = None,
    bin_width: float = 25.0,
    bin_count: int = 0,
) -> Tuple[np.ndarray, int]:
    """
    Discretize an image using the CPU C implementation.

    Returns:
        (discretized_int32_array, ng)
    """
    if image_array.ndim not in (2, 3):
        raise FlashRadiomicsInvalidParameterError(
            f"Unsupported ndim={image_array.ndim}, expected 2D or 3D"
        )
    if image_array.shape != mask_array.shape:
        raise FlashRadiomicsDimensionMismatchError(
            f"Image/mask shape mismatch: {image_array.shape} vs {mask_array.shape}"
        )

    image_array = np.ascontiguousarray(image_array.astype(np.float32, copy=False))
    mask_array = np.ascontiguousarray(mask_array.astype(np.uint8, copy=False))
    ndim = int(image_array.ndim)
    dims = image_array.shape[::-1]

    if spacing is None:
        spacing = tuple([1.0] * ndim)
    if origin is None:
        origin = tuple([0.0] * ndim)

    img = FlashRadiomicsImage()
    img.ndim = ndim
    img.error_code = 0
    for i in range(ndim):
        img.dims[i] = int(dims[i])
        img.spacing[i] = float(spacing[i])
        img.origin[i] = float(origin[i])
    for i in range(ndim, 3):
        img.dims[i] = 1
        img.spacing[i] = 1.0
        img.origin[i] = 0.0
    img.data = image_array.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    mask = FlashRadiomicsMask()
    mask.ndim = ndim
    mask.label = 1
    mask.error_code = 0
    for i in range(ndim):
        mask.dims[i] = int(dims[i])
    for i in range(ndim, 3):
        mask.dims[i] = 1
    mask.data = mask_array.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))

    ng = ctypes.c_int(0)
    ptr = _lib_cpu.flash_radiomics_discretize_image(
        ctypes.byref(img),
        ctypes.byref(mask),
        float(bin_width),
        int(bin_count),
        ctypes.byref(ng),
    )
    if not ptr:
        raise FlashRadiomicsComputationError(
            "C discretization failed and returned NULL"
        )

    total_size = int(np.prod(image_array.shape))
    discretized_flat = np.ctypeslib.as_array(ptr, shape=(total_size,)).copy()
    _lib_cpu.flash_radiomics_free(ptr)
    discretized = discretized_flat.reshape(image_array.shape).astype(np.int32, copy=False)

    if ng.value <= 0:
        raise FlashRadiomicsComputationError("Discretization produced no gray levels")
    return discretized, int(ng.value)


# High-level Python wrapper classes
class CudaContext:
    """
    Wrapper for CUDA context and device management
    
    This class provides a Python interface for CUDA device detection,
    context initialization, and error handling.
    """
    
    def __init__(self, device_id: int = 0):
        """
        Initialize CUDA context
        
        Args:
            device_id: CUDA device ID (default: 0)
        
        Raises:
            FlashRadiomicsBackendUnavailableError: If CUDA is not available
        """
        self._ctx = None
        self._device_info = None
        self._available = self._check_cuda_available()
        
        if self._available:
            try:
                self._device_info = self._get_device_info(device_id)
                if self._device_info['is_available']:
                    self._ctx = _lib_cuda.cuda_init_context(device_id)
                    if not self._ctx:
                        raise FlashRadiomicsBackendUnavailableError("Failed to initialize CUDA context")
            except Exception as e:
                self._available = False
                self._ctx = None
                raise FlashRadiomicsBackendUnavailableError(f"CUDA initialization failed: {e}")
    
    def _check_cuda_available(self) -> bool:
        """
        Check if CUDA is available
        
        Returns:
            True if CUDA library is loaded and device is available
        """
        if not _lib_cuda:
            return False
        
        try:
            device_info = CudaDeviceInfo()
            result = _lib_cuda.cuda_detect_device(ctypes.byref(device_info))
            return result == 0 and device_info.is_available == 1
        except Exception:
            return False
    
    def _get_device_info(self, device_id: int = 0) -> dict:
        """
        Get CUDA device information
        
        Args:
            device_id: CUDA device ID (default: 0)
        
        Returns:
            Dictionary with device properties
        
        Raises:
            FlashRadiomicsBackendUnavailableError: If device detection fails
        """
        if not _lib_cuda:
            raise FlashRadiomicsBackendUnavailableError("CUDA library not loaded")
        
        device_info = CudaDeviceInfo()
        result = _lib_cuda.cuda_detect_device(ctypes.byref(device_info))
        
        if result != 0:
            error_msg = _lib_cuda.cuda_get_error_string(result)
            if error_msg:
                error_msg = error_msg.decode('utf-8')
            else:
                error_msg = f"Unknown CUDA error: {result}"
            raise FlashRadiomicsBackendUnavailableError(f"CUDA device detection failed: {error_msg}")
        
        return {
            'device_id': device_info.device_id,
            'device_name': device_info.device_name.decode('utf-8'),
            'compute_capability': f"{device_info.compute_capability_major}.{device_info.compute_capability_minor}",
            'compute_capability_major': device_info.compute_capability_major,
            'compute_capability_minor': device_info.compute_capability_minor,
            'total_memory': device_info.total_memory,
            'free_memory': device_info.free_memory,
            'multiprocessor_count': device_info.multiprocessor_count,
            'max_threads_per_block': device_info.max_threads_per_block,
            'is_available': device_info.is_available == 1,
        }
    
    def get_device_info(self) -> Optional[dict]:
        """
        Get GPU device properties
        
        Returns:
            Dictionary with device properties or None if CUDA unavailable
        """
        return self._device_info
    
    def is_available(self) -> bool:
        """
        Check if CUDA context is available and initialized
        
        Returns:
            True if CUDA is available and context is initialized
        """
        return self._available and self._ctx is not None
    
    def get_context_ptr(self) -> Optional[ctypes.POINTER(CudaContextStruct)]:
        """
        Get the C CUDA context pointer
        
        Returns:
            Pointer to C CUDA context or None if unavailable
        """
        return self._ctx
    
    def __del__(self):
        """Free CUDA context on deletion"""
        if self._ctx and _lib_cuda:
            try:
                _lib_cuda.cuda_free_context(self._ctx)
            except Exception:
                pass  # Ignore errors during cleanup
            self._ctx = None
    
    def __repr__(self):
        if self._available and self._device_info:
            return (f"CudaContext(device='{self._device_info['device_name']}', "
                   f"compute_capability={self._device_info['compute_capability']}, "
                   f"memory={self._device_info['total_memory'] / (1024**3):.1f}GB)")
        else:
            return "CudaContext(unavailable)"


class ImageLoader:
    """
    High-level wrapper for image loading functions
    
    This class provides a convenient Python interface for loading medical images
    and masks from NIfTI or DICOM files.
    """
    
    @staticmethod
    def load_image(path: str, format: str = 'auto') -> Tuple[np.ndarray, dict]:
        """
        Load medical image from NIfTI file or DICOM series
        
        Args:
            path: File path (NIfTI) or directory path (DICOM series)
            format: 'nifti', 'dicom', or 'auto' (default)
        
        Returns:
            Tuple of (image_data, metadata) where:
                - image_data: numpy array of image data (2D or 3D)
                - metadata: dict with 'ndim', 'dims', 'spacing', 'origin'
        
        Raises:
            FlashRadiomicsError: If loading fails
        """
        format_map = {
            'nifti': FlashRadiomicsImageFormat.NIFTI,
            'dicom': FlashRadiomicsImageFormat.DICOM_SERIES,
            'auto': FlashRadiomicsImageFormat.AUTO,
        }
        fmt = format_map.get(format.lower(), FlashRadiomicsImageFormat.AUTO)
        
        path_bytes = path.encode('utf-8')
        img_ptr = _lib_cpu.flash_radiomics_load_image(path_bytes, fmt)
        
        if not img_ptr:
            raise FlashRadiomicsError("Failed to load image: returned NULL pointer")
        
        img = img_ptr.contents
        _check_error(img.error_code, img.error_msg)
        
        ndim = img.ndim
        dims = tuple(img.dims[:ndim])
        total_size = 1
        for d in dims:
            total_size *= d
        
        data = np.ctypeslib.as_array(img.data, shape=(total_size,)).copy()
        data = data.reshape(dims[::-1])  # Reverse dims for numpy (C order)
        
        metadata = {
            'ndim': ndim,
            'dims': dims,
            'spacing': tuple(img.spacing[:ndim]),
            'origin': tuple(img.origin[:ndim]),
        }
        
        _lib_cpu.flash_radiomics_free_image(img_ptr)
        
        return data, metadata
    
    @staticmethod
    def load_mask(path: str, label: int = 1, format: str = 'auto') -> Tuple[np.ndarray, dict]:
        """
        Load segmentation mask from NIfTI file or DICOM series
        
        Args:
            path: File path (NIfTI) or directory path (DICOM series)
            label: ROI label value (default: 1)
            format: 'nifti', 'dicom', or 'auto' (default)
        
        Returns:
            Tuple of (mask_data, metadata) where:
                - mask_data: boolean numpy array of mask data (2D or 3D)
                - metadata: dict with 'ndim', 'dims', 'label'
        
        Raises:
            FlashRadiomicsError: If loading fails
        """
        format_map = {
            'nifti': FlashRadiomicsImageFormat.NIFTI,
            'dicom': FlashRadiomicsImageFormat.DICOM_SERIES,
            'auto': FlashRadiomicsImageFormat.AUTO,
        }
        fmt = format_map.get(format.lower(), FlashRadiomicsImageFormat.AUTO)
        
        path_bytes = path.encode('utf-8')
        mask_ptr = _lib_cpu.flash_radiomics_load_mask(path_bytes, label, fmt)
        
        if not mask_ptr:
            raise FlashRadiomicsError("Failed to load mask: returned NULL pointer")
        
        mask = mask_ptr.contents
        _check_error(mask.error_code, mask.error_msg)
        
        ndim = mask.ndim
        dims = tuple(mask.dims[:ndim])
        total_size = 1
        for d in dims:
            total_size *= d
        
        data = np.ctypeslib.as_array(mask.data, shape=(total_size,)).copy()
        data = data.reshape(dims[::-1]).astype(bool)  # Reverse dims for numpy (C order)
        
        metadata = {
            'ndim': ndim,
            'dims': dims,
            'label': mask.label,
        }
        
        _lib_cpu.flash_radiomics_free_mask(mask_ptr)
        
        return data, metadata


def detect_backends() -> dict:
    """
    Detect available hardware backends
    
    Returns:
        dict with 'cpu', 'mps', and 'cuda' keys indicating availability
    """
    backends = 0

    probe_lib = _lib_mps if _lib_mps is not None else _lib_cpu
    if probe_lib is not None and hasattr(probe_lib, "flash_radiomics_detect_backends"):
        try:
            backends |= int(probe_lib.flash_radiomics_detect_backends())
        except Exception:
            pass

    # Probe the separate CUDA library.
    if _lib_cuda is not None and hasattr(_lib_cuda, "flash_radiomics_detect_backends"):
        try:
            backends |= int(_lib_cuda.flash_radiomics_detect_backends())
        except Exception:
            pass

    if backends == 0:
        backends = 0x01

    return {
        'cpu': bool(backends & 0x01),  # FLASH_RADIOMICS_BACKEND_CPU_AVAILABLE
        'mps': bool(backends & 0x02),  # FLASH_RADIOMICS_BACKEND_MPS_AVAILABLE
        'cuda': bool(backends & 0x04),  # FLASH_RADIOMICS_BACKEND_CUDA_AVAILABLE
    }


def reset_cuda_segment_workspace() -> None:
    """Reset the thread-local CUDA segment workspace cache."""
    for lib in (_lib_cuda, _lib_cpu):
        if lib is None:
            continue
        reset_fn = getattr(lib, "flash_radiomics_cuda_reset_segment_workspace", None)
        if reset_fn is not None:
            reset_fn()
            return


def set_cuda_deterministic_mode(enabled: bool) -> None:
    """Set thread-local CUDA deterministic mode for segment extraction."""
    enabled_i = 1 if enabled else 0
    for lib in (_lib_cuda, _lib_cpu):
        if lib is None:
            continue
        setter = getattr(lib, "flash_radiomics_cuda_set_deterministic_mode", None)
        if setter is not None:
            setter(enabled_i)
            return


def get_cuda_deterministic_mode() -> bool:
    """Get thread-local CUDA deterministic mode."""
    for lib in (_lib_cuda, _lib_cpu):
        if lib is None:
            continue
        getter = getattr(lib, "flash_radiomics_cuda_get_deterministic_mode", None)
        if getter is not None:
            try:
                return bool(getter())
            except Exception:
                return False
    return False


def configure_cuda_glszm_determinism(
    tie_break_lowest_label: Optional[bool] = None,
    deterministic_reduction: Optional[bool] = None,
    allow_early_exit: Optional[bool] = None,
    ccl_sync_interval: Optional[int] = None,
    ccl_flatten_interval: Optional[int] = None,
) -> None:
    """Configure thread-local GLSZM CUDA determinism knobs."""
    tie_break_i = -1 if tie_break_lowest_label is None else (1 if tie_break_lowest_label else 0)
    reduction_i = -1 if deterministic_reduction is None else (1 if deterministic_reduction else 0)
    early_exit_i = -1 if allow_early_exit is None else (1 if allow_early_exit else 0)
    sync_i = -1 if ccl_sync_interval is None else int(ccl_sync_interval)
    flatten_i = -1 if ccl_flatten_interval is None else int(ccl_flatten_interval)

    for lib in (_lib_cuda, _lib_cpu):
        if lib is None:
            continue
        setter = getattr(lib, "flash_radiomics_cuda_configure_glszm_determinism", None)
        if setter is not None:
            setter(tie_break_i, reduction_i, early_exit_i, sync_i, flatten_i)
            return


def set_cuda_bin_minimum(bin_minimum: Optional[float]) -> None:
    """Set thread-local CUDA fixed-bin-width origin for segment extraction."""
    enabled_i = 0 if bin_minimum is None else 1
    value = 0.0 if bin_minimum is None else float(bin_minimum)
    for lib in (_lib_cuda, _lib_cpu):
        if lib is None:
            continue
        setter = getattr(lib, "flash_radiomics_cuda_set_bin_minimum", None)
        if setter is not None:
            setter(enabled_i, value)
            return


__all__ = [
    'ImageLoader',
    'CudaContext',
    'detect_backends',
    'reset_cuda_segment_workspace',
    'set_cuda_deterministic_mode',
    'get_cuda_deterministic_mode',
    'configure_cuda_glszm_determinism',
    'set_cuda_bin_minimum',
    'get_feature_library',
    'get_loaded_library_paths',
    'FlashRadiomicsBackend',
    'FlashRadiomicsImageFormat',
    'FlashRadiomicsError',
    'FlashRadiomicsFileNotFoundError',
    'FlashRadiomicsInvalidFormatError',
    'FlashRadiomicsDimensionMismatchError',
    'FlashRadiomicsMemoryError',
    'FlashRadiomicsInvalidParameterError',
    'FlashRadiomicsComputationError',
    'FlashRadiomicsBackendUnavailableError',
    'FlashRadiomicsIOError',
    'FlashRadiomicsCorruptedDataError',
    'FlashRadiomicsUnsupportedOperationError',
]
