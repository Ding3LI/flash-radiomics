from __future__ import annotations

import collections
import concurrent.futures
import copy
from dataclasses import dataclass
import json
import logging
import os
import pathlib
import threading
import time
from itertools import chain

import numpy as np
import pykwalify.core
import SimpleITK as sitk

from . import _bindings
from .hdf5_io import write_extraction_hdf5
from .voxel import compute_native_voxel_features
from flash_radiomics import (
    generalinfo,
    getFeatureClasses,
    getImageTypes,
    getParameterValidationFiles,
    imageoperations,
)

logger = logging.getLogger(__name__)

_DEFAULT_DISABLED_FEATURE_CLASSES = frozenset({"shape2D", "gldzm", "ih", "ivh"})

_FULL_SCALAR_FEATURE_CLASSES = (
    "firstorder",
    "ih",
    "ivh",
    "glcm",
    "glrlm",
    "glszm",
    "gldzm",
    "gldm",
    "ngtdm",
    "shape",
)

_FULL_SPATIAL_FEATURE_CLASSES = (
    "firstorder",
    "glcm",
    "glrlm",
    "glszm",
    "gldm",
    "ngtdm",
)


_PREPROCESS_TIMING_KEYS = (
    "load_image_s",
    "normalize_s",
    "resample_s",
    "pre_crop_s",
    "post_load_check_mask_s",
    "resegment_s",
    "post_resegment_check_mask_s",
    "preprocess_total_s",
)

_EXTRACT_TIMING_KEYS = (
    "shape_s",
    "compute_features_s",
    "post_preprocess_feature_total_s",
)


def _empty_timings(keys):
    return {key: 0.0 for key in keys}


@dataclass
class PreprocessedCase:
    image: sitk.Image
    shape_mask: sitk.Image
    feature_mask: sitk.Image
    bounding_box: np.ndarray
    general_info: collections.OrderedDict
    label: int
    label_channel: int | None
    spatial_mode: bool
    timing_breakdown: dict[str, float]


class _SingletonGeometryTolerance:
    _instance = None
    _initialized = False
    _lock = threading.Lock()

    def __new__(cls, *_args, **_kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, tolerance=None):
        if not self._initialized:
            with self._lock:
                if not self._initialized:
                    self.geometryTolerance = tolerance
                    _SingletonGeometryTolerance._initialized = True


class RadiomicsFeatureExtractor:
    r"""
    Wrapper class for calculation of a radiomics signature.
    At and after initialisation various settings can be used to customize the resultant signature.
    This includes which classes and features to use, as well as what should be done in terms of preprocessing the image
    and what images (original and/or filtered) should be used as input.

    Then a call to :py:func:`execute` generates the radiomics
    signature specified by these settings for the passed image and labelmap combination. This function can be called
    repeatedly in a batch process to calculate the radiomics signature for all image and labelmap combinations.

    At initialization, a parameters file (string pointing to yaml or json structured file) or dictionary can be provided
    containing all necessary settings (top-level keys include ``setting``,
    ``imageType``, ``featureClass``, ``scalarFeatureClass``, and
    ``spatialFeatureClass``). This is
    done by passing it as the first positional argument. If no positional argument is supplied, or the argument is not
    either a dictionary or a string pointing to a valid file, defaults will be applied.
    Moreover, at initialisation, custom settings (*NOT enabled image types and/or feature classes*) can be provided
    as keyword arguments, with the setting name as key and its value as the argument value (e.g. ``binWidth=25``).
    Settings specified here will override those in the parameter file/dict/default settings.
    For more information on possible settings and customization, see
    :ref:`Customizing the Extraction <radiomics-customization-label>`.

    By default, all features in all feature classes are enabled.
    By default, only `Original` input image is enabled (No filter applied).
    """

    def __init__(self, *args, **kwargs):
        self.settings = {}
        self.enabledImagetypes = {}
        self.enabledFeatures = {}
        self._mode_feature_sets = None
        self._discretization_cache = {}
        self._preprocess_cache = {}
        self._preprocess_cache_order = []
        self._preprocess_cache_max_entries = 8
        self._last_hdf5_path = None
        self._backend = kwargs.pop("backend", "auto")
        self._num_threads = kwargs.pop("num_threads", os.cpu_count() or 4)

        self.featureClassNames = list(getFeatureClasses().keys())

        if len(args) == 1 and isinstance(args[0], dict):
            logger.info("Loading parameter dictionary")
            self._applyParams(paramsDict=args[0])
        elif len(args) == 1 and (isinstance(args[0], (str, pathlib.PurePath))):
            if not os.path.isfile(args[0]):
                msg = f"Parameter file {args[0]} does not exist."
                raise OSError(msg)
            logger.info("Loading parameter file %s", str(args[0]))
            self._applyParams(paramsFile=str(args[0]))
        else:
            self.settings = self._getDefaultSettings()
            logger.info("No valid config parameter, using defaults: %s", self.settings)

            self.enabledImagetypes = {"Original": {}}
            logger.info("Enabled image types: %s", self.enabledImagetypes)

            for featureClassName in self.featureClassNames:
                if featureClassName in _DEFAULT_DISABLED_FEATURE_CLASSES:
                    continue
                self.enabledFeatures[featureClassName] = []
            logger.info("Enabled features: %s", self.enabledFeatures)

        if len(kwargs) > 0:
            logger.info("Applying custom setting overrides: %s", kwargs)
            self.settings.update(kwargs)
            logger.debug("Settings: %s", self.settings)

        if self.settings.get("binCount", None) is not None:
            logger.warning(
                "Fixed bin Count enabled! However, we recommend using a fixed bin Width. See "
                "http://pyradiomics.readthedocs.io/en/latest/faq.html#radiomics-fixed-bin-width for more "
                "details"
            )

        self._setTolerance()

    def _setTolerance(self):
        _SingletonGeometryTolerance(self.settings.get("geometryTolerance"))
        if _SingletonGeometryTolerance().geometryTolerance is not None:
            logger.debug(
                "Setting SimpleITK tolerance to %s",
                _SingletonGeometryTolerance().geometryTolerance,
            )
            sitk.ProcessObject.SetGlobalDefaultCoordinateTolerance(
                _SingletonGeometryTolerance().geometryTolerance
            )
            sitk.ProcessObject.SetGlobalDefaultDirectionTolerance(
                _SingletonGeometryTolerance().geometryTolerance
            )

    @staticmethod
    def _normalize_cache_value(value):
        if isinstance(value, dict):
            return tuple(
                (str(k), RadiomicsFeatureExtractor._normalize_cache_value(v))
                for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))
            )
        if isinstance(value, (list, tuple, set)):
            return tuple(RadiomicsFeatureExtractor._normalize_cache_value(v) for v in value)
        if isinstance(value, np.generic):
            return value.item()
        try:
            hash(value)
        except TypeError:
            return (type(value).__name__, id(value))
        return value

    def _settings_cache_key(self):
        return (
            self._normalize_cache_value(self.settings),
            self._normalize_cache_value(self.enabledImagetypes),
            self._normalize_cache_value(self.enabledFeatures),
            self._backend,
            int(self._num_threads),
        )

    def _make_preprocess_cache_key(
        self,
        imageFilepath,
        maskFilepath,
        label,
        label_channel,
        spatial_mode,
    ):
        if not isinstance(imageFilepath, sitk.SimpleITK.Image):
            return None
        if not isinstance(maskFilepath, sitk.SimpleITK.Image):
            return None
        return (
            id(imageFilepath),
            id(maskFilepath),
            int(label),
            None if label_channel is None else int(label_channel),
            bool(spatial_mode),
            self._settings_cache_key(),
        )

    def _get_cached_preprocessed(self, cache_key):
        if cache_key is None:
            return None
        return self._preprocess_cache.get(cache_key)

    def _store_preprocessed(self, cache_key, preprocessed):
        if cache_key is None:
            return
        self._preprocess_cache[cache_key] = preprocessed
        self._preprocess_cache_order.append(cache_key)
        while len(self._preprocess_cache_order) > self._preprocess_cache_max_entries:
            oldest = self._preprocess_cache_order.pop(0)
            if oldest in self._preprocess_cache:
                del self._preprocess_cache[oldest]

    def addProvenance(self, provenance_on=True):
        """
        Enable or disable reporting of additional information on the extraction. This information includes toolbox version,
        enabled input images and applied settings. Furthermore, additional information on the image and region of interest
        (ROI) is also provided, including original image spacing, total number of voxels in the ROI and total number of
        fully connected volumes in the ROI.

        To disable this, call ``addProvenance(False)``.
        """
        self.settings["additionalInfo"] = provenance_on

    @staticmethod
    def _getDefaultSettings():
        """
        Returns a dictionary containing the default settings specified in this class. These settings cover global settings,
        such as ``additionalInfo``, as well as the image pre-processing settings (e.g. resampling). Feature class specific
        are defined in the respective feature classes and and not included here. Similarly, filter specific settings are
        defined in ``imageoperations.py`` and also not included here.
        """
        return {
            "minimumROIDimensions": 2,
            "minimumROISize": None,  # Skip testing the ROI size by default
            "geometryTolerance": None,
            "correctMask": False,
            "normalize": False,
            "normalizeScale": 1,
            "removeOutliers": None,
            "resampledPixelSpacing": None,  # No resampling by default
            "interpolator": "sitkBSpline",  # Alternative: sitk.sitkBSpline
            "preCrop": False,
            "padDistance": 5,
            "distances": [1],
            "binWidth": 25.0,
            "binCount": None,
            "binMinimum": None,
            "force2D": False,
            "force2Ddimension": 0,
            "resegmentRange": None,  # No resegmentation by default
            "resegmentMode": "absolute",
            "resegmentShape": False,
            "sigma": [],
            "start_level": 0,
            "level": 1,
            "wavelet": "coif1",
            "gradientUseSpacing": True,
            "lbp2DRadius": 1.0,
            "lbp2DSamples": 8,
            "lbp2DMethod": "uniform",
            "lbp3DLevels": 2,
            "lbp3DIcosphereRadius": 1.0,
            "lbp3DIcosphereSubdivision": 1,
            "voxelArrayShift": 0,
            "symmetricalGLCM": True,
            "weightingNorm": None,
            "gldm_a": 0,
            "segmentClassWorkers": 0,  # 0=auto
            "cudaGlszmTieBreakLowestLabel": True,
            "cudaGlszmDeterministicReduction": False,
            "cudaGlszmAllowEarlyExit": True,
            "cudaGlszmCclSyncInterval": 4,
            "cudaGlszmCclFlattenInterval": 4,
            "glcmPythonMCC": False,
            "label": 1,
            "label_channel": 0,
            "additionalInfo": True,
            "hdf5OutputPath": None,
            "hdf5OutputDirectory": None,
            "hdf5Compression": "lzf",
            "kernelRadius": 1,
            "maskedKernel": True,
            "initValue": 0.0,
            "voxelBatch": -1,
        }

    def _default_enabled_features(self, mode=None):
        if mode == "scalar":
            names = _FULL_SCALAR_FEATURE_CLASSES
        elif mode == "spatial":
            names = _FULL_SPATIAL_FEATURE_CLASSES
        else:
            names = (
                name
                for name in self.featureClassNames
                if name not in _DEFAULT_DISABLED_FEATURE_CLASSES
            )
        return {name: [] for name in names if name in self.featureClassNames}

    def _activate_feature_mode(self, spatial_mode):
        if self._mode_feature_sets is None:
            return
        mode = "spatial" if spatial_mode else "scalar"
        if mode not in self._mode_feature_sets:
            raise ValueError(f"{mode.capitalize()} extraction is disabled by the configuration")
        self.enabledFeatures = copy.deepcopy(self._mode_feature_sets[mode])

    def loadParams(self, paramsFile):
        """
        Parse specified parameters file and use it to update settings, enabled feature(Classes) and image types. For more
        information on the structure of the parameter file, see
        :ref:`Customizing the extraction <radiomics-customization-label>`.

        If supplied file does not match the requirements (i.e. unrecognized names or invalid values for a setting), a
        pykwalify error is raised.
        """
        self._applyParams(paramsFile=paramsFile)

    def loadJSONParams(self, JSON_configuration):
        """
        Pars JSON structured configuration string and use it to update settings, enabled feature(Classes) and image types.
        For more information on the structure of the parameter file, see
        :ref:`Customizing the extraction <radiomics-customization-label>`.

        If supplied string does not match the requirements (i.e. unrecognized names or invalid values for a setting), a
        pykwalify error is raised.
        """
        parameter_data = json.loads(JSON_configuration)
        self._applyParams(paramsDict=parameter_data)

    def _applyParams(self, paramsFile=None, paramsDict=None):
        """
        Validates and applies a parameter dictionary. See :py:func:`loadParams` and :py:func:`loadJSONParams` for more info.
        """

        if (
            len(pykwalify.core.log.handlers) == 0
            and len(logging.getLogger().handlers) == 0
        ):
            pykwalify.core.log.addHandler(
                logging.getLogger("flash_radiomics").handlers[0]
            )

        schemaFile, schemaFuncs = getParameterValidationFiles()
        c = pykwalify.core.Core(
            source_file=paramsFile,
            source_data=paramsDict,
            schema_files=[schemaFile],
            extensions=[schemaFuncs],
        )
        params = c.validate()
        logger.debug("Parameters parsed, input is valid.")

        enabledImageTypes = params.get("imageType", {})
        legacyFeatures = params.get("featureClass")
        hasScalarFeatures = "scalarFeatureClass" in params
        hasSpatialFeatures = "spatialFeatureClass" in params
        scalarFeatures = params.get("scalarFeatureClass")
        spatialFeatures = params.get("spatialFeatureClass")
        settings = params.get("setting", {})
        voxelSettings = params.get("voxelSetting", {})

        if legacyFeatures is not None and (hasScalarFeatures or hasSpatialFeatures):
            raise ValueError(
                "Use featureClass or scalarFeatureClass/spatialFeatureClass, not both"
            )

        logger.debug("Applying settings")

        if len(enabledImageTypes) == 0:
            self.enabledImagetypes = {"Original": {}}
        else:
            self.enabledImagetypes = enabledImageTypes

        logger.debug("Enabled image types: %s", self.enabledImagetypes)

        if hasScalarFeatures or hasSpatialFeatures:
            self._mode_feature_sets = {}
            if scalarFeatures is not None:
                self._mode_feature_sets["scalar"] = copy.deepcopy(
                    scalarFeatures or self._default_enabled_features("scalar")
                )
            if spatialFeatures is not None:
                self._mode_feature_sets["spatial"] = copy.deepcopy(
                    spatialFeatures or self._default_enabled_features("spatial")
                )
            if not self._mode_feature_sets:
                raise ValueError(
                    "Configuration disables both scalar and spatial extraction"
                )
            initial_mode = "scalar" if "scalar" in self._mode_feature_sets else "spatial"
            self.enabledFeatures = copy.deepcopy(self._mode_feature_sets[initial_mode])
        elif not legacyFeatures:
            self._mode_feature_sets = None
            self.enabledFeatures = self._default_enabled_features()
        else:
            self._mode_feature_sets = None
            self.enabledFeatures = legacyFeatures

        logger.debug("Enabled features: %s", self.enabledFeatures)

        self.settings = self._getDefaultSettings()
        self.settings.update(settings)
        self.settings.update(voxelSettings)

        logger.debug("Settings: %s", settings)

    def execute(
        self,
        imageFilepath,
        maskFilepath,
        label=None,
        label_channel=None,
        spatialMode=None,
        output_hdf5_path=None,
        case_id=None,
        roi_id=None,
        roi_name=None,
    ):
        """
        Compute radiomics signature for provide image and mask combination. It comprises of the following steps:

        1. Image and mask are loaded and normalized/resampled if necessary.
        2. Validity of ROI is checked using :py:func:`~imageoperations.checkMask`, which also computes and returns the
           bounding box.
        3. If enabled, provenance information is calculated and stored as part of the result. (Not available in spatial
           extraction)
        4. Shape features are calculated on a cropped (no padding) version of the original image. (Not available in
           spatial extraction)
        5. If enabled, resegment the mask based upon the range specified in ``resegmentRange`` (default None: resegmentation
           disabled).
        6. Other enabled feature classes are calculated using all specified image types in ``_enabledImageTypes``. Images
           are cropped to tumor mask (no padding) after application of any filter and before being passed to the feature
           class.
        7. The calculated features is returned as ``collections.OrderedDict``.

        :param imageFilepath: SimpleITK Image, or string pointing to image file location
        :param maskFilepath: SimpleITK Image, or string pointing to labelmap file location
        :param label: Integer, value of the label for which to extract features. If not specified, last specified label
            is used. Default label is 1.
        :param label_channel: Integer, index of the channel to use when maskFilepath yields a SimpleITK.Image with a vector
            pixel type. Default index is 0.
        :param spatialMode: Select spatial (True) or scalar (False) extraction.
            When omitted, mode-specific YAML sections are applied automatically;
            if both are enabled, both modes are extracted into one HDF5 file.
        :param output_hdf5_path: Optional explicit HDF5 output path. When omitted,
            a deterministic path is created in ``hdf5OutputDirectory``, the
            ``FLASH_RADIOMICS_HDF5_DIR`` environment variable, or
            ``flash_radiomics_hdf5`` in the working directory.
        :param case_id: Optional case identifier for the HDF5 hierarchy.
        :param roi_id: Optional ROI identifier for the HDF5 hierarchy.
        :param roi_name: Optional human-readable ROI name for the HDF5 hierarchy.
        :returns: dictionary containing calculated signature ("<imageType>_<featureClass>_<featureName>":value).
            Scalar values are floats and spatial values are ``SimpleITK.Image``.
            Automatic combined extraction returns ``scalar``, ``spatial``, and
            ``hdf5_path`` entries.
            Type of diagnostic features differs, but can always be represented as a string.
            Every completed extraction is additionally persisted as HDF5; the
            returned dictionary remains unchanged.
        """
        if spatialMode is None and self._mode_feature_sets is not None:
            configured_modes = set(self._mode_feature_sets)
            if configured_modes == {"scalar", "spatial"}:
                common_options = {
                    "label": label,
                    "label_channel": label_channel,
                    "output_hdf5_path": output_hdf5_path,
                    "case_id": case_id,
                    "roi_id": roi_id,
                    "roi_name": roi_name,
                }
                scalar = self._execute_single(
                    imageFilepath, maskFilepath, spatialMode=False, **common_options
                )
                spatial = self._execute_single(
                    imageFilepath, maskFilepath, spatialMode=True, **common_options
                )
                return {
                    "scalar": scalar,
                    "spatial": spatial,
                    "hdf5_path": self.last_hdf5_path,
                }
            spatialMode = "spatial" in configured_modes
        elif spatialMode is None:
            spatialMode = False

        return self._execute_single(
            imageFilepath,
            maskFilepath,
            label=label,
            label_channel=label_channel,
            spatialMode=bool(spatialMode),
            output_hdf5_path=output_hdf5_path,
            case_id=case_id,
            roi_id=roi_id,
            roi_name=roi_name,
        )

    def _execute_single(
        self,
        imageFilepath,
        maskFilepath,
        *,
        label=None,
        label_channel=None,
        spatialMode=False,
        output_hdf5_path=None,
        case_id=None,
        roi_id=None,
        roi_name=None,
    ):
        self._activate_feature_mode(spatialMode)
        effective_label = (
            int(label) if label is not None else int(self.settings.get("label", 1))
        )
        if label_channel is not None:
            effective_label_channel = int(label_channel)
        else:
            _default_channel = self.settings.get("label_channel")
            effective_label_channel = (
                None if _default_channel is None else int(_default_channel)
            )

        cache_key = self._make_preprocess_cache_key(
            imageFilepath,
            maskFilepath,
            effective_label,
            effective_label_channel,
            spatialMode,
        )
        preprocessed = self._get_cached_preprocessed(cache_key)
        if preprocessed is None:
            preprocessed = self.preprocess_case(
                imageFilepath,
                maskFilepath,
                label=effective_label,
                label_channel=effective_label_channel,
                spatialMode=spatialMode,
            )
            self._store_preprocessed(cache_key, preprocessed)
        feature_vector = self.extract_from_preprocessed(preprocessed)
        self._last_hdf5_path = self._save_extraction_hdf5(
            image_source=imageFilepath,
            preprocessed=preprocessed,
            feature_vector=feature_vector,
            output_hdf5_path=output_hdf5_path,
            case_id=case_id,
            roi_id=roi_id,
            roi_name=roi_name,
        )
        return feature_vector

    @property
    def last_hdf5_path(self):
        """Path to the HDF5 file created by the most recent ``execute`` call."""
        return self._last_hdf5_path

    @staticmethod
    def _case_id_from_input(image_source) -> str:
        if isinstance(image_source, (str, pathlib.PurePath)):
            name = pathlib.Path(image_source).name
            if name.endswith(".nii.gz"):
                return name[:-7]
            suffix = pathlib.Path(name).suffix
            return name[: -len(suffix)] if suffix else name
        return "in_memory_case"

    @staticmethod
    def _safe_hdf5_filename_component(value) -> str:
        text = str(value).strip() or "unnamed"
        return "".join(
            character if character.isalnum() or character in {"-", "_", "."} else "_"
            for character in text
        )

    def _resolve_hdf5_output_path(
        self,
        *,
        output_hdf5_path,
        case_id: str,
        roi_id: str,
    ) -> pathlib.Path:
        requested_path = (
            output_hdf5_path
            if output_hdf5_path is not None
            else self.settings.get("hdf5OutputPath")
        )
        if requested_path:
            path = pathlib.Path(requested_path)
            if path.suffix.lower() in {".h5", ".hdf5"}:
                return path
            output_directory = path
        else:
            output_directory = self.settings.get("hdf5OutputDirectory")
            if not output_directory:
                output_directory = os.environ.get("FLASH_RADIOMICS_HDF5_DIR")
            if not output_directory:
                output_directory = pathlib.Path.cwd() / "flash_radiomics_hdf5"

        filename = "{}_{}.flashrad.h5".format(
            self._safe_hdf5_filename_component(case_id),
            self._safe_hdf5_filename_component(roi_id),
        )
        return pathlib.Path(output_directory) / filename

    def _save_extraction_hdf5(
        self,
        *,
        image_source,
        preprocessed,
        feature_vector,
        output_hdf5_path,
        case_id,
        roi_id,
        roi_name,
    ) -> pathlib.Path | None:
        if not isinstance(preprocessed, PreprocessedCase):
            return None

        mode = "spatial" if preprocessed.spatial_mode else "scalar"
        resolved_case_id = str(case_id or self._case_id_from_input(image_source))
        resolved_roi_id = str(roi_id or f"label_{preprocessed.label}")
        resolved_roi_name = str(roi_name or resolved_roi_id)
        output_path = self._resolve_hdf5_output_path(
            output_hdf5_path=output_hdf5_path,
            case_id=resolved_case_id,
            roi_id=resolved_roi_id,
        )
        return write_extraction_hdf5(
            output_path,
            mode=mode,
            case_id=resolved_case_id,
            roi_id=resolved_roi_id,
            roi_name=resolved_roi_name,
            image=preprocessed.image,
            feature_mask=preprocessed.feature_mask,
            feature_vector=feature_vector,
            settings=self.settings,
            backend=self._backend,
            enabled_image_types=self.enabledImagetypes,
            enabled_features=self.enabledFeatures,
            compression=self.settings.get("hdf5Compression", "lzf"),
        )

    def preprocess_case(
        self,
        imageFilepath,
        maskFilepath,
        label=None,
        label_channel=None,
        spatialMode=False,
    ) -> PreprocessedCase:
        _settings = self.settings.copy()
        self._discretization_cache = {}
        timings = _empty_timings(_PREPROCESS_TIMING_KEYS)

        tolerance = _settings.get("geometryTolerance")
        additionalInfo = _settings.get("additionalInfo", False)
        resegmentShape = _settings.get("resegmentShape", False)

        if label is not None:
            _settings["label"] = label
        else:
            label = _settings.get("label", 1)

        if label_channel is not None:
            _settings["label_channel"] = label_channel
        else:
            label_channel = _settings.get("label_channel")

        if _SingletonGeometryTolerance().geometryTolerance != tolerance:
            self._setTolerance()

        if additionalInfo:
            generalInfo = generalinfo.GeneralInfo()
            generalInfo.addGeneralSettings(_settings)
            generalInfo.addEnabledImageTypes(self.enabledImagetypes)
        else:
            generalInfo = None

        if spatialMode:
            _settings["spatialMode"] = True
            logger.info("Starting spatial extraction")

        logger.info("Calculating features with label: %d", label)
        logger.debug("Enabled images types: %s", self.enabledImagetypes)
        logger.debug("Enabled features: %s", self.enabledFeatures)
        logger.debug("Current settings: %s", _settings)

        image, mask, load_timings = self._loadImageWithTiming(
            imageFilepath, maskFilepath, generalInfo, **_settings
        )
        timings.update(load_timings)

        check_t0 = time.perf_counter()
        boundingBox, correctedMask = imageoperations.checkMask(image, mask, **_settings)
        timings["post_load_check_mask_s"] = float(time.perf_counter() - check_t0)

        if correctedMask is not None:
            if generalInfo is not None:
                generalInfo.addMaskElements(image, correctedMask, label, "corrected")
            mask = correctedMask

        logger.debug("Image and Mask loaded and valid, starting extraction")

        resegmentedMask = None
        if _settings.get("resegmentRange", None) is not None:
            resegment_t0 = time.perf_counter()
            resegmentedMask = imageoperations.resegmentMask(image, mask, **_settings)
            timings["resegment_s"] = float(time.perf_counter() - resegment_t0)

            recheck_t0 = time.perf_counter()
            boundingBox, correctedMask = imageoperations.checkMask(
                image, resegmentedMask, **_settings
            )
            timings["post_resegment_check_mask_s"] = float(
                time.perf_counter() - recheck_t0
            )

            if generalInfo is not None:
                generalInfo.addMaskElements(
                    image, resegmentedMask, label, "resegmented"
                )

        general_info = collections.OrderedDict()
        if generalInfo is not None:
            general_info.update(generalInfo.getGeneralInfo())

        timings["preprocess_total_s"] = float(
            timings["load_image_s"]
            + timings["post_load_check_mask_s"]
            + timings["resegment_s"]
            + timings["post_resegment_check_mask_s"]
        )

        shape_mask = mask
        if resegmentShape and resegmentedMask is not None:
            shape_mask = resegmentedMask

        feature_mask = mask
        if resegmentedMask is not None:
            feature_mask = resegmentedMask

        return PreprocessedCase(
            image=image,
            shape_mask=shape_mask,
            feature_mask=feature_mask,
            bounding_box=boundingBox,
            general_info=general_info,
            label=int(label),
            label_channel=label_channel,
            spatial_mode=bool(spatialMode),
            timing_breakdown=timings,
        )

    def extract_from_preprocessed(
        self, preprocessed: PreprocessedCase, return_timing=False
    ):
        self._activate_feature_mode(preprocessed.spatial_mode)
        _settings = self.settings.copy()
        _settings["label"] = preprocessed.label
        if preprocessed.label_channel is not None:
            _settings["label_channel"] = preprocessed.label_channel

        if preprocessed.spatial_mode:
            _settings["spatialMode"] = True
            kernelRadius = _settings.get("kernelRadius", 1)
        else:
            kernelRadius = 0

        featureVector = collections.OrderedDict()
        featureVector.update(preprocessed.general_info)

        timings = _empty_timings(_EXTRACT_TIMING_KEYS)
        extract_t0 = time.perf_counter()

        if not preprocessed.spatial_mode:
            shape_t0 = time.perf_counter()
            featureVector.update(
                self.computeShape(
                    preprocessed.image,
                    preprocessed.shape_mask,
                    preprocessed.bounding_box,
                    **_settings,
                )
            )
            timings["shape_s"] = float(time.perf_counter() - shape_t0)

        logger.debug("Creating image type iterator")
        imageGenerators = []
        for imageType, customKwargs in self.enabledImagetypes.items():
            args = _settings.copy()
            args.update(customKwargs)
            msg = f'Adding image type "{imageType}" with custom settings: {customKwargs!s}'
            logger.info(msg)
            imageGenerators = chain(
                imageGenerators,
                getattr(imageoperations, f"get{imageType}Image")(
                    preprocessed.image,
                    preprocessed.feature_mask,
                    **args,
                ),
            )

        logger.debug("Extracting features")
        compute_t0 = time.perf_counter()
        for originputImage, imageTypeName, inputKwargs in imageGenerators:
            logger.info("Calculating features for %s image", imageTypeName)
            inputImage, inputMask = imageoperations.cropToTumorMask(
                originputImage,
                preprocessed.feature_mask,
                preprocessed.bounding_box,
                padDistance=kernelRadius,
            )
            gldzm_distance_mask = inputKwargs.get("gldzmDistanceMask")
            if isinstance(gldzm_distance_mask, sitk.Image):
                _, cropped_distance_mask = imageoperations.cropToTumorMask(
                    originputImage,
                    gldzm_distance_mask,
                    preprocessed.bounding_box,
                    padDistance=kernelRadius,
                )
                inputKwargs = dict(inputKwargs)
                inputKwargs["_gldzm_distance_mask_array"] = (
                    sitk.GetArrayFromImage(cropped_distance_mask) != 0
                )
            featureVector.update(
                self.computeFeatures(
                    inputImage, inputMask, imageTypeName, **inputKwargs
                )
            )
        timings["compute_features_s"] = float(time.perf_counter() - compute_t0)
        timings["post_preprocess_feature_total_s"] = float(
            time.perf_counter() - extract_t0
        )

        logger.debug("Features extracted")

        if return_timing:
            return featureVector, timings
        return featureVector

    @staticmethod
    def loadImage(ImageFilePath, MaskFilePath, generalInfo=None, **kwargs):
        local_kwargs = dict(kwargs)
        local_kwargs.setdefault("validateMaskLabel", True)
        image, mask, _ = RadiomicsFeatureExtractor._loadImageWithTiming(
            ImageFilePath, MaskFilePath, generalInfo, **local_kwargs
        )
        return image, mask

    @staticmethod
    def _loadImageWithTiming(ImageFilePath, MaskFilePath, generalInfo=None, **kwargs):
        """
        Load and pre-process the image and labelmap.
        If ImageFilePath is a string, it is loaded as SimpleITK Image and assigned to ``image``,
        if it already is a SimpleITK Image, it is just assigned to ``image``.
        All other cases are ignored (nothing calculated).
        Equal approach is used for assignment of ``mask`` using MaskFilePath. If necessary, a segmentation object (i.e. mask
        volume with vector-image type) is then converted to a labelmap (=scalar image type). Data type is forced to UInt32.
        See also :py:func:`~imageoperations.getMask()`.

        If normalizing is enabled image is first normalized before any resampling is applied.

        If resampling is enabled, both image and mask are resampled and cropped to the tumor mask (with additional
        padding as specified in padDistance) after assignment of image and mask.

        :param ImageFilePath: SimpleITK.Image object or string pointing to SimpleITK readable file representing the image
                              to use.
        :param MaskFilePath: SimpleITK.Image object or string pointing to SimpleITK readable file representing the mask
                             to use.
        :param generalInfo: GeneralInfo Object. If provided, it is used to store diagnostic information of the
                            pre-processing.
        :param kwargs: Dictionary containing the settings to use for this particular image type.
        :return: 2 SimpleITK.Image objects representing the loaded image and mask, respectively.
        """
        timings = _empty_timings(
            ("load_image_s", "normalize_s", "resample_s", "pre_crop_s")
        )

        normalize = kwargs.get("normalize", False)
        interpolator = kwargs.get("interpolator")
        resampledPixelSpacing = kwargs.get("resampledPixelSpacing")
        preCrop = kwargs.get("preCrop", False)
        label = kwargs.get("label", 1)

        logger.info("Loading image and mask")
        load_t0 = time.perf_counter()
        if isinstance(ImageFilePath, str) and os.path.isfile(ImageFilePath):
            image = sitk.ReadImage(ImageFilePath)
        elif isinstance(ImageFilePath, str) and os.path.isdir(ImageFilePath):
            reader = sitk.ImageSeriesReader()
            dicom_names = reader.GetGDCMSeriesFileNames(ImageFilePath)
            reader.SetFileNames(dicom_names)
            image = reader.Execute()
        elif isinstance(ImageFilePath, sitk.SimpleITK.Image):
            image = ImageFilePath
        else:
            msg = "Error reading image Filepath or SimpleITK object"
            raise ValueError(msg)

        if isinstance(MaskFilePath, str) and os.path.isfile(MaskFilePath):
            mask = sitk.ReadImage(MaskFilePath)
        elif isinstance(MaskFilePath, sitk.SimpleITK.Image):
            mask = MaskFilePath
        else:
            msg = "Error reading mask Filepath or SimpleITK object"
            raise ValueError(msg)

        if "validateMaskLabel" in kwargs:
            mask_kwargs = dict(kwargs)
        else:
            mask_kwargs = dict(kwargs)
            mask_kwargs["validateMaskLabel"] = False
        mask = imageoperations.getMask(mask, **mask_kwargs)

        if generalInfo is not None:
            generalInfo.addImageElements(image)
            generalInfo.addMaskElements(None, mask, label)

        if normalize:
            normalize_t0 = time.perf_counter()
            image = imageoperations.normalizeImage(image, **kwargs)
            timings["normalize_s"] = float(time.perf_counter() - normalize_t0)

        if interpolator is not None and resampledPixelSpacing is not None:
            resample_t0 = time.perf_counter()
            image, mask = imageoperations.resampleImage(image, mask, **kwargs)
            timings["resample_s"] = float(time.perf_counter() - resample_t0)
            if generalInfo is not None:
                generalInfo.addImageElements(image, "interpolated")
                generalInfo.addMaskElements(image, mask, label, "interpolated")

        elif preCrop:
            pre_crop_t0 = time.perf_counter()
            bb, correctedMask = imageoperations.checkMask(image, mask, **kwargs)
            if correctedMask is not None:
                mask = correctedMask
            if bb is None:
                msg = "Mask checks failed during pre-crop"
                raise ValueError(msg)

            image, mask = imageoperations.cropToTumorMask(image, mask, bb, **kwargs)
            timings["pre_crop_s"] = float(time.perf_counter() - pre_crop_t0)

        timings["load_image_s"] = float(time.perf_counter() - load_t0)
        return image, mask, timings

    @staticmethod
    def _to_numpy(image, mask, label):
        image_array = sitk.GetArrayFromImage(image).astype(np.float64, copy=False)
        mask_array = sitk.GetArrayFromImage(mask)
        mask_array = (mask_array == label).astype(np.uint8)
        spacing = tuple(image.GetSpacing())
        origin = tuple(image.GetOrigin())
        return image_array, mask_array, spacing, origin

    def _get_discretized_buffer(
        self,
        image_array: np.ndarray,
        mask_array: np.ndarray,
        spacing,
        origin,
        label: int,
        bin_width: float,
        bin_count: int,
        bin_minimum: float | None = None,
    ):
        image_array = np.asarray(image_array)
        mask_array = np.asarray(mask_array)
        key = (
            int(image_array.__array_interface__["data"][0]),
            tuple(image_array.shape),
            tuple(image_array.strides),
            str(image_array.dtype),
            int(mask_array.__array_interface__["data"][0]),
            tuple(mask_array.shape),
            tuple(mask_array.strides),
            str(mask_array.dtype),
            int(label),
            float(bin_width),
            int(bin_count),
            None if bin_minimum is None else float(bin_minimum),
        )
        if key not in self._discretization_cache:
            binning_kwargs = {"binWidth": float(bin_width)}
            if bin_minimum is not None:
                binning_kwargs["binMinimum"] = float(bin_minimum)
            if int(bin_count) > 0:
                binning_kwargs["binCount"] = int(bin_count)

            discretized, _ = imageoperations.binImage(
                image_array,
                np.asarray(mask_array) != 0,
                **binning_kwargs,
            )
            discretized = np.asarray(discretized, dtype=np.int32)
            ng = int(np.max(discretized[np.asarray(mask_array) != 0]))
            self._discretization_cache[key] = (discretized, ng)
        return self._discretization_cache[key]

    @staticmethod
    def _get_default_voxel_batch(backend: str | None = None) -> int:
        raw_value = os.environ.get("FLASH_RADIOMICS_VOXEL_BATCH", "").strip()
        if raw_value:
            try:
                parsed = int(raw_value)
                if parsed > 0:
                    return parsed
            except ValueError:
                logger.warning(
                    "Ignoring invalid FLASH_RADIOMICS_VOXEL_BATCH=%s",
                    raw_value,
                )
        backend_name = str(backend or "").strip().lower()
        if backend_name == "gpu":
            backend_name = "cuda"
        if backend_name == "cuda":
            return 32768
        if backend_name == "mps":
            return 8192
        return 2048

    def _resolve_segment_class_workers(self, total_classes: int, kwargs) -> int:
        if total_classes <= 1:
            return 1

        requested = kwargs.get("segmentClassWorkers", None)
        if requested is None:
            requested = self.settings.get("segmentClassWorkers", 0)
        try:
            requested = int(requested)
        except (TypeError, ValueError):
            requested = 0

        if requested <= 0:
            env_value = os.environ.get("FLASH_RADIOMICS_SEGMENT_CLASS_WORKERS", "").strip()
            if env_value:
                try:
                    requested = int(env_value)
                except ValueError:
                    requested = 0

        if requested <= 0:
            backend_name = str(self._backend or "auto").strip().lower()
            if backend_name == "gpu":
                backend_name = "cuda"
            if backend_name == "cuda":
                # Default CUDA segment extraction to one worker.
                return 1
            if self._num_threads <= 2:
                return 1
            requested = 2

        return max(1, min(total_classes, int(requested)))

    def computeVoxelFeatures(self, image, mask, imageTypeName, **kwargs):
        return compute_native_voxel_features(
            self,
            image,
            mask,
            imageTypeName,
            **kwargs,
        )

    def computeShape(self, image, mask, boundingBox, **kwargs):
        """
        Calculate the shape (2D and/or 3D) features for the passed image and mask.

        :param image: SimpleITK.Image object representing the image used
        :param mask: SimpleITK.Image object representing the mask used
        :param boundingBox: The boundingBox calculated by :py:func:`~imageoperations.checkMask()`, i.e. a tuple with lower
          (even indices) and upper (odd indices) bound of the bounding box for each dimension.
        :param kwargs: Dictionary containing the settings to use.
        :return: collections.OrderedDict containing the calculated shape features. If no features are calculated, an empty
          OrderedDict will be returned.
        """
        featureVector = collections.OrderedDict()

        enabledFeatures = self.enabledFeatures

        croppedImage, croppedMask = imageoperations.cropToTumorMask(
            image, mask, boundingBox
        )
        image_array, mask_array, spacing, origin = self._to_numpy(
            croppedImage, croppedMask, kwargs.get("label", 1)
        )
        shared_image_flat_f32 = np.ascontiguousarray(
            image_array, dtype=np.float32
        ).reshape(-1)
        shared_mask_flat_u8 = np.ascontiguousarray(
            mask_array, dtype=np.uint8
        ).reshape(-1)

        def compute(shape_type):
            logger.info("Computing %s", shape_type)
            featureNames = enabledFeatures[shape_type]
            shapeClass = getFeatureClasses()[shape_type](
                image_array,
                mask_array,
                spacing=spacing,
                origin=origin,
                backend=self._backend,
                num_threads=self._num_threads,
                _image_flat_f32=shared_image_flat_f32,
                _mask_flat_u8=shared_mask_flat_u8,
                **kwargs,
            )

            if featureNames is not None:
                for feature in featureNames:
                    shapeClass.enableFeatureByName(feature)

            for featureName, featureValue in shapeClass.execute().items():
                newFeatureName = f"original_{shape_type}_{featureName}"
                featureVector[newFeatureName] = featureValue

        Nd = mask.GetDimension()
        if "shape" in enabledFeatures:
            if Nd == 3:
                compute("shape")
            else:
                logger.warning(
                    "Shape features are only available 3D input (for 2D input, use shape2D). Found %iD input",
                    Nd,
                )

        if "shape2D" in enabledFeatures:
            if Nd == 3:
                force2D = kwargs.get("force2D", False)
                force2Ddimension = kwargs.get("force2Ddimension", 0)
                if not force2D:
                    logger.warning(
                        "parameter force2D must be set to True to enable shape2D extraction"
                    )
                elif (
                    not (boundingBox[1::2] - boundingBox[0::2] + 1)[force2Ddimension]
                    > 1
                ):
                    logger.warning(
                        "Size in specified 2D dimension (%i) is greater than 1, cannot calculate 2D shape",
                        force2Ddimension,
                    )
                else:
                    compute("shape2D")
            elif Nd == 2:
                compute("shape2D")
            else:
                logger.warning(
                    "Shape2D features are only available for 2D and 3D (with force2D=True) input. "
                    "Found %iD input",
                    Nd,
                )

        return featureVector

    def computeFeatures(self, image, mask, imageTypeName, **kwargs):
        r"""
        Compute signature using image, mask and \*\*kwargs settings.

        This function computes the signature for just the passed image (original or derived), it does not pre-process or
        apply a filter to the passed image. Features / Classes to use for calculation of signature are defined in
        ``self.enabledFeatures``. See also :py:func:`enableFeaturesByName`.

        :param image: The cropped (and optionally filtered) SimpleITK.Image object representing the image used
        :param mask: The cropped SimpleITK.Image object representing the mask used
        :param imageTypeName: String specifying the filter applied to the image, or "original" if no filter was applied.
        :param kwargs: Dictionary containing the settings to use for this particular image type.
        :return: collections.OrderedDict containing the calculated features for all enabled classes.
          If no features are calculated, an empty OrderedDict will be returned.

        .. note::

          shape descriptors are independent of gray level and therefore calculated separately (handled in `execute`). In
          this function, no shape features are calculated.
        """
        if kwargs.get("spatialMode", False):
            return self.computeVoxelFeatures(image, mask, imageTypeName, **kwargs)

        featureVector = collections.OrderedDict()
        featureClasses = getFeatureClasses()

        enabledFeatures = self.enabledFeatures
        image_array, mask_array, spacing, origin = self._to_numpy(
            image, mask, kwargs.get("label", 1)
        )

        backend_name = str(self._backend or "auto").strip().lower()
        if backend_name == "gpu":
            backend_name = "cuda"
        if backend_name == "auto":
            try:
                available = _bindings.detect_backends()
            except Exception:
                available = {}
            if available.get("cuda", False):
                backend_name = "cuda"
            elif available.get("mps", False):
                backend_name = "mps"
            else:
                backend_name = "cpu"

        bin_width = float(kwargs.get("binWidth", 25))
        bin_count = int(kwargs.get("binCount", 0) or 0)
        bin_minimum = kwargs.get("binMinimum")
        bin_minimum = None if bin_minimum is None else float(bin_minimum)
        discretized = None
        ng = 0
        if backend_name == "cpu":
            discretized, ng = self._get_discretized_buffer(
                image_array,
                mask_array,
                spacing,
                origin,
                kwargs.get("label", 1),
                bin_width,
                bin_count,
                bin_minimum,
            )
        elif backend_name == "cuda":
            set_bin_minimum = getattr(_bindings, "set_cuda_bin_minimum", None)
            if callable(set_bin_minimum):
                set_bin_minimum(bin_minimum)

            set_det = getattr(_bindings, "set_cuda_deterministic_mode", None)
            if callable(set_det):
                cuda_deterministic = kwargs.get(
                    "cudaDeterministic",
                    self.settings.get("cudaDeterministic", False),
                )
                set_det(bool(cuda_deterministic))

            configure_glszm = getattr(_bindings, "configure_cuda_glszm_determinism", None)
            if callable(configure_glszm):
                def _cuda_setting(name, legacy_name, default):
                    if name in kwargs:
                        return kwargs.get(name)
                    if legacy_name in kwargs:
                        return kwargs.get(legacy_name)
                    if name in self.settings:
                        return self.settings.get(name)
                    if legacy_name in self.settings:
                        return self.settings.get(legacy_name)
                    return default

                def _cuda_bool(value, default):
                    if value is None:
                        return bool(default)
                    if isinstance(value, str):
                        return value.strip().lower() in {"1", "true", "yes", "on"}
                    return bool(value)

                def _cuda_int(value, default):
                    try:
                        return int(value)
                    except (TypeError, ValueError):
                        return int(default)

                configure_glszm(
                    tie_break_lowest_label=_cuda_bool(
                        _cuda_setting(
                            "cudaGlszmTieBreakLowestLabel",
                            "cudaGLSZMTieBreakLowestLabel",
                            True,
                        ),
                        True,
                    ),
                    deterministic_reduction=_cuda_bool(
                        _cuda_setting(
                            "cudaGlszmDeterministicReduction",
                            "cudaGLSZMDeterministicReduction",
                            False,
                        ),
                        False,
                    ),
                    allow_early_exit=_cuda_bool(
                        _cuda_setting(
                            "cudaGlszmAllowEarlyExit",
                            "cudaGLSZMAllowEarlyExit",
                            True,
                        ),
                        True,
                    ),
                    ccl_sync_interval=_cuda_int(
                        _cuda_setting(
                            "cudaGlszmCclSyncInterval",
                            "cudaGLSZMCCLSyncInterval",
                            4,
                        ),
                        4,
                    ),
                    ccl_flatten_interval=_cuda_int(
                        _cuda_setting(
                            "cudaGlszmCclFlattenInterval",
                            "cudaGLSZMCCLFlattenInterval",
                            4,
                        ),
                        4,
                    ),
                )

            reset_ws = getattr(_bindings, "reset_cuda_segment_workspace", None)
            if callable(reset_ws):
                reset_ws()

        shared_image_flat_f32 = np.ascontiguousarray(
            image_array, dtype=np.float32
        ).reshape(-1)
        shared_mask_flat_u8 = np.ascontiguousarray(
            mask_array, dtype=np.uint8
        ).reshape(-1)

        class_requests = [
            (featureClassName, featureNames)
            for featureClassName, featureNames in enabledFeatures.items()
            if not featureClassName.startswith("shape")
            and featureClassName in featureClasses
        ]
        if not class_requests:
            return featureVector

        workers = self._resolve_segment_class_workers(len(class_requests), kwargs)
        class_num_threads = self._num_threads
        if workers > 1:
            class_num_threads = max(1, int(self._num_threads // workers))

        def _compute_one(featureClassName, featureNames):
            logger.info("Computing %s", featureClassName)
            class_kwargs = dict(kwargs)
            if discretized is not None and ng > 0:
                class_kwargs["discretized"] = discretized
                class_kwargs["ng"] = ng
            class_kwargs["_image_flat_f32"] = shared_image_flat_f32
            class_kwargs["_mask_flat_u8"] = shared_mask_flat_u8
            featureClass = featureClasses[featureClassName](
                image_array,
                mask_array,
                spacing=spacing,
                origin=origin,
                backend=self._backend,
                num_threads=class_num_threads,
                **class_kwargs,
            )

            if featureNames is not None:
                for feature in featureNames:
                    featureClass.enableFeatureByName(feature)

            class_result = collections.OrderedDict()
            for featureName, featureValue in featureClass.execute().items():
                newFeatureName = f"{imageTypeName}_{featureClassName}_{featureName}"
                class_result[newFeatureName] = featureValue
            return class_result

        class_results = {}
        if workers > 1 and len(class_requests) > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(_compute_one, featureClassName, featureNames): featureClassName
                    for featureClassName, featureNames in class_requests
                }
                for future in concurrent.futures.as_completed(futures):
                    class_results[futures[future]] = future.result()
        else:
            for featureClassName, featureNames in class_requests:
                class_results[featureClassName] = _compute_one(
                    featureClassName, featureNames
                )

        for featureClassName, _featureNames in class_requests:
            featureVector.update(class_results.get(featureClassName, {}))

        return featureVector

    def enableAllImageTypes(self):
        """
        Enable all possible image types without any custom settings.
        """

        logger.debug("Enabling all image types")
        for imageType in getImageTypes():
            self.enabledImagetypes[imageType] = {}
        logger.debug("Enabled images types: %s", self.enabledImagetypes)

    def disableAllImageTypes(self):
        """
        Disable all image types.
        """
        logger.debug("Disabling all image types")
        self.enabledImagetypes = {}

    def enableImageTypeByName(self, imageType, enabled=True, customArgs=None):
        r"""
        Enable or disable specified image type. If enabling image type, optional custom settings can be specified in
        customArgs.

        Current possible image types are:

        - Original: No filter applied
        - Wavelet: Wavelet filtering, yields 8 decompositions per level (all possible combinations of applying either
          a High or a Low pass filter in each of the three dimensions.
          See also :py:func:`~radiomics.imageoperations.getWaveletImage`
        - LoG: Laplacian of Gaussian filter, edge enhancement filter. Emphasizes areas of gray level change, where sigma
          defines how coarse the emphasised texture should be. A low sigma emphasis on fine textures (change over a
          short distance), where a high sigma value emphasises coarse textures (gray level change over a large distance).
          See also :py:func:`~radiomics.imageoperations.getLoGImage`
        - Square: Takes the square of the image intensities and linearly scales them back to the original range.
          Negative values in the original image will be made negative again after application of filter.
        - SquareRoot: Takes the square root of the absolute image intensities and scales them back to original range.
          Negative values in the original image will be made negative again after application of filter.
        - Logarithm: Takes the logarithm of the absolute intensity + 1. Values are scaled to original range and
          negative original values are made negative again after application of filter.
        - Exponential: Takes the the exponential, where filtered intensity is e^(absolute intensity). Values are
          scaled to original range and negative original values are made negative again after application of filter.
        - Gradient: Returns the gradient magnitude.
        - LBP2D: Calculates and returns a local binary pattern applied in 2D.
        - LBP3D: Calculates and returns local binary pattern maps applied in 3D using spherical harmonics. Last returned
          image is the corresponding kurtosis map.

        For the mathmetical formulas of square, squareroot, logarithm and exponential, see their respective functions in
        :ref:`imageoperations<radiomics-imageoperations-label>`
        (:py:func:`~radiomics.imageoperations.getSquareImage`,
        :py:func:`~radiomics.imageoperations.getSquareRootImage`,
        :py:func:`~radiomics.imageoperations.getLogarithmImage`,
        :py:func:`~radiomics.imageoperations.getExponentialImage`,
        :py:func:`~radiomics.imageoperations.getGradientImage`,
        :py:func:`~radiomics.imageoperations.getLBP2DImage` and
        :py:func:`~radiomics.imageoperations.getLBP3DImage`,
        respectively).
        """
        if imageType not in getImageTypes():
            logger.warning("Image type %s is not recognized", imageType)
            return

        if enabled:
            if customArgs is None:
                customArgs = {}
                logger.debug(
                    "Enabling image type %s (no additional custom settings)", imageType
                )
            else:
                logger.debug(
                    "Enabling image type %s (additional custom settings: %s)",
                    imageType,
                    customArgs,
                )
            self.enabledImagetypes[imageType] = customArgs
        elif imageType in self.enabledImagetypes:
            logger.debug("Disabling image type %s", imageType)
            del self.enabledImagetypes[imageType]
        logger.debug("Enabled images types: %s", self.enabledImagetypes)

    def enableImageTypes(self, **enabledImagetypes):
        """
        Enable input images, with optionally custom settings, which are applied to the respective input image.
        Settings specified here override those in kwargs.
        The following settings are not customizable:

        - interpolator
        - resampledPixelSpacing
        - padDistance

        Updates current settings: If necessary, enables input image. Always overrides custom settings specified
        for input images passed in inputImages.
        To disable input images, use :py:func:`enableInputImageByName` or :py:func:`disableAllInputImages`
        instead.

        :param enabledImagetypes: dictionary, key is imagetype (original, wavelet or log) and value is custom settings
          (dictionary)
        """
        logger.debug("Updating enabled images types with %s", enabledImagetypes)
        self.enabledImagetypes.update(enabledImagetypes)
        logger.debug("Enabled images types: %s", self.enabledImagetypes)

    def enableAllFeatures(self):
        """
        Enable all classes and all features.

        .. note::
          Individual features that have been marked "deprecated" are not enabled by this function. They can still be enabled
          manually by a call to :py:func:`~radiomics.base.RadiomicsBase.enableFeatureByName()`,
          :py:func:`~radiomics.featureextractor.RadiomicsFeaturesExtractor.enableFeaturesByName()`
          or in the parameter file (by specifying the feature by name, not when enabling all features).
          However, in most cases this will still result only in a deprecation warning.
        """
        self._mode_feature_sets = None
        logger.debug("Enabling all features in all feature classes")
        for featureClassName in self.featureClassNames:
            self.enabledFeatures[featureClassName] = []
        logger.debug("Enabled features: %s", self.enabledFeatures)

    def enableAllScalarFeatures(self):
        """Enable every supported scalar feature family for a 3D extraction."""
        self._mode_feature_sets = None
        self.enabledFeatures = {
            name: []
            for name in _FULL_SCALAR_FEATURE_CLASSES
            if name in self.featureClassNames
        }
        logger.debug("Enabled full scalar feature set: %s", self.enabledFeatures)

    def enableAllSpatialFeatures(self):
        """Enable every feature family supported by spatial extraction."""
        self._mode_feature_sets = None
        self.enabledFeatures = {
            name: []
            for name in _FULL_SPATIAL_FEATURE_CLASSES
            if name in self.featureClassNames
        }
        logger.debug("Enabled full spatial feature set: %s", self.enabledFeatures)

    def disableAllFeatures(self):
        """
        Disable all classes.
        """
        self._mode_feature_sets = None
        logger.debug("Disabling all feature classes")
        self.enabledFeatures = {}

    def enableFeatureClassByName(self, featureClass, enabled=True):
        """
        Enable or disable all features in given class.

        .. note::
          Individual features that have been marked "deprecated" are not enabled by this function. They can still be enabled
          manually by a call to :py:func:`~radiomics.base.RadiomicsBase.enableFeatureByName()`,
          :py:func:`~radiomics.featureextractor.RadiomicsFeaturesExtractor.enableFeaturesByName()`
          or in the parameter file (by specifying the feature by name, not when enabling all features).
          However, in most cases this will still result only in a deprecation warning.
        """
        self._mode_feature_sets = None
        if featureClass not in self.featureClassNames:
            logger.warning("Feature class %s is not recognized", featureClass)
            return

        if enabled:
            logger.debug("Enabling all features in class %s", featureClass)
            self.enabledFeatures[featureClass] = []
        elif featureClass in self.enabledFeatures:
            logger.debug("Disabling feature class %s", featureClass)
            del self.enabledFeatures[featureClass]
        logger.debug("Enabled features: %s", self.enabledFeatures)

    def enableFeaturesByName(self, **enabledFeatures):
        """
        Specify which features to enable. Key is feature class name, value is a list of enabled feature names.

        To enable all features for a class, provide the class name with an empty list or None as value.
        Settings for feature classes specified in enabledFeatures.keys are updated, settings for feature classes
        not yet present in enabledFeatures.keys are added.
        To disable the entire class, use :py:func:`disableAllFeatures` or :py:func:`enableFeatureClassByName` instead.
        """
        self._mode_feature_sets = None
        logger.debug("Updating enabled features with %s", enabledFeatures)
        self.enabledFeatures.update(enabledFeatures)
        logger.debug("Enabled features: %s", self.enabledFeatures)


def extract_scalar_and_spatial(
    image,
    mask,
    *,
    output_hdf5_path,
    config=None,
    scalar_config=None,
    spatial_config=None,
    backend="auto",
    num_threads=None,
    label=1,
    label_channel=None,
    case_id=None,
    roi_id=None,
    roi_name=None,
    **settings,
):
    """Extract scalar values and spatial maps into one HDF5 container.

    ``config`` may enable either or both mode-specific feature sections. Use
    ``scalar_config`` and ``spatial_config`` when the modes come from separate
    files. With no configuration, every supported family is enabled.
    """
    if config is not None and (scalar_config is not None or spatial_config is not None):
        raise ValueError(
            "Use either config for both modes or scalar_config/spatial_config, not both"
        )

    scalar_source = config if config is not None else scalar_config
    spatial_source = config if config is not None else spatial_config
    constructor_options = dict(settings)
    constructor_options["backend"] = backend
    if num_threads is not None:
        constructor_options["num_threads"] = num_threads

    if scalar_source is None:
        scalar_extractor = RadiomicsFeatureExtractor(**constructor_options)
        scalar_extractor.enableAllScalarFeatures()
    else:
        scalar_extractor = RadiomicsFeatureExtractor(
            scalar_source,
            **constructor_options,
        )

    if spatial_source is None:
        spatial_extractor = RadiomicsFeatureExtractor(**constructor_options)
        spatial_extractor.enableAllSpatialFeatures()
    else:
        spatial_extractor = RadiomicsFeatureExtractor(
            spatial_source,
            **constructor_options,
        )

    common_execute_options = {
        "label": label,
        "label_channel": label_channel,
        "output_hdf5_path": output_hdf5_path,
        "case_id": case_id,
        "roi_id": roi_id,
        "roi_name": roi_name,
    }
    image_input = str(image) if isinstance(image, pathlib.PurePath) else image
    mask_input = str(mask) if isinstance(mask, pathlib.PurePath) else mask
    scalar_enabled = (
        scalar_extractor._mode_feature_sets is None
        or "scalar" in scalar_extractor._mode_feature_sets
    )
    spatial_enabled = (
        spatial_extractor._mode_feature_sets is None
        or "spatial" in spatial_extractor._mode_feature_sets
    )
    scalar = None
    spatial = None
    output_paths = []
    if scalar_enabled:
        scalar = scalar_extractor.execute(
            image_input,
            mask_input,
            spatialMode=False,
            **common_execute_options,
        )
        output_paths.append(scalar_extractor.last_hdf5_path)
    if spatial_enabled:
        spatial = spatial_extractor.execute(
            image_input,
            mask_input,
            spatialMode=True,
            **common_execute_options,
        )
        output_paths.append(spatial_extractor.last_hdf5_path)
    if len(set(output_paths)) != 1:
        raise RuntimeError(
            "Scalar and spatial extraction resolved to different HDF5 containers: "
            f"{output_paths}"
        )
    return {
        "scalar": scalar,
        "spatial": spatial,
        "hdf5_path": output_paths[0],
    }
