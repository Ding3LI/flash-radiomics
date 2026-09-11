from __future__ import annotations

import inspect
import logging
import os
import pkgutil
import sys
import threading
from typing import Any, Callable

from . import imageoperations

__version__ = "1.0.0"


def deprecated(func):
    """
    Decorator to mark functions as deprecated so they are not enabled by default.
    """
    func._is_deprecated = True
    return func


class _SingletonFeatureClasses:
    _instance = None
    _initialized = False
    _lock = threading.Lock()

    def __new__(cls, *_args, **_kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            with self._lock:
                if not self._initialized:
                    self._featureClasses = {}
                    package_dir = os.path.dirname(__file__)
                    for _, mod, _ in pkgutil.iter_modules([package_dir]):
                        # Skip native ctypes libraries during Python feature discovery.
                        if str(mod).startswith(
                            ("_", "libflash_radiomics_", "flash_radiomics_")
                        ):
                            continue
                        __import__("flash_radiomics." + mod)
                        module = sys.modules["flash_radiomics." + mod]
                        attributes = inspect.getmembers(module, inspect.isclass)
                        for name, klass in attributes:
                            if name.startswith("Radiomics"):
                                for parent in inspect.getmro(klass)[1:]:
                                    if parent.__name__ == "RadiomicsFeaturesBase":
                                        self._featureClasses[mod] = klass
                                        break
                    _SingletonFeatureClasses._initialized = True


def getFeatureClasses():
    """
    Return a dict of feature classes keyed by module name.
    """
    return _SingletonFeatureClasses()._featureClasses


class _SingletonImageTypes:
    _instance = None
    _initialized = False
    _lock = threading.Lock()

    def __new__(cls, *_args, **_kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            with self._lock:
                if not self._initialized:
                    self._imageTypes = [
                        member[3:-5]
                        for member in dir(imageoperations)
                        if member.startswith("get") and member.endswith("Image")
                    ]
                    _SingletonImageTypes._initialized = True


def getImageTypes():
    """
    Return a list of available image types based on imageoperations.get*Image functions.
    """
    return _SingletonImageTypes()._imageTypes


def getParameterValidationFiles():
    """
    Return file locations for the parameter schema and custom validation functions.
    """
    dataDir = os.path.abspath(os.path.join(os.path.dirname(__file__), "schemas"))
    schemaFile = os.path.join(dataDir, "paramSchema.yaml")
    schemaFuncs = os.path.join(dataDir, "schemaFuncs.py")
    return schemaFile, schemaFuncs


class _DummyProgressReporter:
    def __init__(self, iterable=None, desc="", _total=None):
        self.desc = desc
        self.iterable = iterable

    def __iter__(self):
        return self.iterable.__iter__()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, tb):
        pass

    def update(self, n=1):
        pass


def getProgressReporter(*args, **kwargs):
    if progressReporter is not None and logging.NOTSET < handler.level <= logging.INFO:
        return progressReporter(*args, **kwargs)
    return _DummyProgressReporter(*args, **kwargs)


progressReporter: Callable[[Any, Any], None] | None = None

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
formatter = logging.Formatter("%(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


def setVerbosity(level):
    level = max(level, 10)
    level = min(level, 60)
    handler.setLevel(level)
    if handler.level < logger.level:
        logger.setLevel(level)


__all__ = [
    "PreprocessedCase",
    "RadiomicsFeatureExtractor",
    "RadiomicsFirstOrder",
    "RadiomicsIH",
    "RadiomicsIVH",
    "RadiomicsGLCM",
    "RadiomicsGLRLM",
    "RadiomicsGLSZM",
    "RadiomicsGLDZM",
    "RadiomicsGLDM",
    "RadiomicsNGTDM",
    "RadiomicsShape",
    "RadiomicsShape2D",
    "RadiomicsBase",
    "compute_pet_decayed_dose",
    "convert_pet_counts_to_suv",
    "crop_image_and_mask_around_mask",
    "isolate_mask_label",
    "load_dicom_series",
    "n4_bias_field_correction",
    "normalize_mr_by_reference_mask",
    "write_nifti_pair",
    "__version__",
    "deprecated",
    "detect_backends",
    "extract_scalar_and_spatial",
    "getFeatureClasses",
    "getImageTypes",
    "getParameterValidationFiles",
    "getProgressReporter",
    "setVerbosity",
]

from ._bindings import detect_backends
from .featureextractor import (
    PreprocessedCase,
    RadiomicsFeatureExtractor,
    extract_scalar_and_spatial,
)
from .firstorder import RadiomicsFirstOrder
from .ih import RadiomicsIH
from .ivh import RadiomicsIVH
from .glcm import RadiomicsGLCM
from .glrlm import RadiomicsGLRLM
from .glszm import RadiomicsGLSZM
from .gldzm import RadiomicsGLDZM
from .gldm import RadiomicsGLDM
from .ngtdm import RadiomicsNGTDM
from .shape import RadiomicsShape
from .shape2D import RadiomicsShape2D
from .base import RadiomicsBase
from .preprocessing import (
    compute_pet_decayed_dose,
    convert_pet_counts_to_suv,
    crop_image_and_mask_around_mask,
    isolate_mask_label,
    load_dicom_series,
    n4_bias_field_correction,
    normalize_mr_by_reference_mask,
    write_nifti_pair,
)
