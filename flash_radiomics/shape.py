"""
Shape-based features

Supports both 2D and 3D images with automatic feature selection based on dimensionality.
"""

# PYRADIOMICS: Feature names and default availability are aligned to PyRadiomics.

import ctypes
from collections import OrderedDict

import numpy as np
from .base import RadiomicsFeaturesBase
from . import _bindings


class RadiomicsShape(RadiomicsFeaturesBase):
    """
    Shape features extractor
    
    Computes shape-based features from the segmentation mask.
    
    2D Shape Features:
    - Area, Perimeter
    - Perimeter to Area Ratio
    - Circularity, Eccentricity
    - Maximum 2D Diameter
    - Major/Minor Axis Length
    - Elongation
    
    3D Shape Features:
    - Volume, Surface Area
    - Surface to Volume Ratio
    - Sphericity, Compactness
    - Maximum 3D Diameter
    - Major/Minor/Least Axis Length
    - Elongation, Flatness
    
    Automatically selects appropriate features based on image dimensionality.
    """

    _DERIVED_FEATURE_ORDER = (
        "Asphericity",
        "CentreOfMassShift",
        "IntegratedIntensity",
        "VolumeDensityAABB",
        "AreaDensityAABB",
        "VolumeDensityAEE",
        "AreaDensityAEE",
        "VolumeDensityConvexHull",
        "AreaDensityConvexHull",
    )
    _HULL_DENSITY_FEATURES = frozenset(
        {
            "VolumeDensityAABB",
            "AreaDensityAABB",
            "VolumeDensityConvexHull",
            "AreaDensityConvexHull",
        }
    )
    
    def __init__(self, inputImage, inputMask, **kwargs):
        """
        Initialize shape feature extractor
        
        Args:
            inputImage: numpy array of image data (2D or 3D) - not used but kept for API compatibility
            inputMask: numpy array of mask data (2D or 3D)
            **kwargs: Additional parameters including:
                - spacing: voxel/pixel spacing (required)
        """
        super().__init__(inputImage, inputMask, **kwargs)

        if self.ndim != 3:
            raise ValueError("Shape features are only available in 3D. Use shape2D for 2D inputs.")

        if 'spacing' not in kwargs:
            raise ValueError("Shape features require 'spacing' parameter")

    @staticmethod
    def _safe_ratio(numerator: float, denominator: float) -> float:
        if not np.isfinite(denominator) or denominator == 0.0:
            return np.nan
        return float(numerator / denominator)

    @staticmethod
    def _ellipsoid_surface_area_from_semi_axes(
        semi_axes: np.ndarray,
        n_degree: int = 20,
    ) -> float:
        # MIRP-compatible approximation with exact spheroid shortcuts.
        semi_axes = np.sort(np.asarray(semi_axes, dtype=np.float64))
        if semi_axes.shape != (3,) or np.any(~np.isfinite(semi_axes)) or np.any(semi_axes <= 0.0):
            return np.nan

        a, b, c = semi_axes
        if np.isclose(a, b) and np.isclose(a, c):
            return float(4.0 * np.pi * a * a)

        if np.isclose(a, b):
            ecc_sq = 1.0 - (a * a) / (c * c)
            ecc = np.sqrt(max(0.0, ecc_sq))
            if np.isclose(ecc, 0.0):
                return float(4.0 * np.pi * a * a)
            return float(
                2.0 * np.pi * a * a
                * (1.0 + c * np.arcsin(ecc) / (a * ecc))
            )

        if np.isclose(b, c):
            ecc_sq = 1.0 - (a * a) / (c * c)
            ecc = np.sqrt(max(0.0, ecc_sq))
            if np.isclose(ecc, 0.0):
                return float(4.0 * np.pi * c * c)
            return float(
                2.0 * np.pi * c * c
                * (1.0 + ((1.0 - ecc * ecc) / ecc) * np.arctanh(ecc))
            )

        from numpy.polynomial.legendre import legval

        ecc_alpha = np.sqrt(max(0.0, 1.0 - (b * b) / (c * c)))
        ecc_beta = np.sqrt(max(0.0, 1.0 - (a * a) / (c * c)))
        if np.isclose(ecc_alpha, 0.0) or np.isclose(ecc_beta, 0.0):
            return float(4.0 * np.pi * b * c)

        nu = np.arange(0.0, float(n_degree) + 1.0, 1.0, dtype=np.float64)
        leg_coeff = (ecc_alpha * ecc_beta) ** nu / (1.0 - 4.0 * nu * nu)
        leg_x = (ecc_alpha * ecc_alpha + ecc_beta * ecc_beta) / (2.0 * ecc_alpha * ecc_beta)
        return float(4.0 * np.pi * c * b * legval(x=leg_x, c=leg_coeff))

    def _compute_hull_metrics(self, roi_mask: np.ndarray) -> tuple[float, float, float, float]:
        try:
            from scipy.spatial import ConvexHull
            from skimage.measure import marching_cubes
        except Exception:
            return (np.nan, np.nan, np.nan, np.nan)

        mask_u8 = np.asarray(roi_mask, dtype=np.uint8)
        if not np.any(mask_u8):
            return (np.nan, np.nan, np.nan, np.nan)

        try:
            spacing_zyx = tuple(float(v) for v in self.spacing[::-1])
            vertices, _, _, _ = marching_cubes(
                np.pad(mask_u8, pad_width=1, mode="constant", constant_values=0),
                level=0.5,
                spacing=spacing_zyx,
            )
            if vertices.shape[0] < 4:
                return (np.nan, np.nan, np.nan, np.nan)

            hull = ConvexHull(vertices, qhull_options="QJ")
            hull_volume = float(hull.volume)
            hull_area = float(hull.area)

            hull_vertices = vertices[hull.vertices, :]
            hull_vertices = hull_vertices - np.mean(vertices, axis=0, keepdims=True)
            dims = np.max(hull_vertices, axis=0) - np.min(hull_vertices, axis=0)
            box_volume = float(np.prod(dims))
            box_area = float(
                2.0 * (dims[0] * dims[1] + dims[0] * dims[2] + dims[1] * dims[2])
            )
            return (hull_volume, hull_area, box_volume, box_area)
        except Exception:
            return (np.nan, np.nan, np.nan, np.nan)

    def _compute_derived_features(
        self,
        base_features: OrderedDict,
        required_features: set[str],
    ) -> OrderedDict:
        derived = OrderedDict()
        if not required_features:
            return derived

        roi_mask = np.asarray(self.inputMask) != 0
        if not np.any(roi_mask):
            for feature_name in self._DERIVED_FEATURE_ORDER:
                if feature_name in required_features:
                    derived[feature_name] = np.nan
            return derived

        image_arr = np.asarray(self.inputImage, dtype=np.float64)
        roi_values = image_arr[roi_mask]
        mesh_volume = float(base_features.get("MeshVolume", np.nan))
        surface_area = float(base_features.get("SurfaceArea", np.nan))

        if "Asphericity" in required_features:
            if mesh_volume > 0.0 and surface_area > 0.0:
                x = 36.0 * np.pi * mesh_volume * mesh_volume / (surface_area ** 3.0)
                derived["Asphericity"] = float(x ** (-1.0 / 3.0) - 1.0) if x > 0.0 else np.nan
            else:
                derived["Asphericity"] = np.nan

        if "IntegratedIntensity" in required_features:
            if roi_values.size > 0 and np.isfinite(mesh_volume):
                derived["IntegratedIntensity"] = float(mesh_volume * np.mean(roi_values))
            else:
                derived["IntegratedIntensity"] = np.nan

        if "CentreOfMassShift" in required_features:
            coords = np.argwhere(roi_mask).astype(np.float64, copy=False)
            int_sum = float(np.sum(roi_values))
            if coords.shape[0] == 0 or np.isclose(int_sum, 0.0):
                derived["CentreOfMassShift"] = np.nan
            else:
                com_morph = np.mean(coords, axis=0)
                com_int = np.sum(coords * roi_values[:, None], axis=0) / int_sum
                spacing_zyx = np.asarray(self.spacing[::-1], dtype=np.float64)
                derived["CentreOfMassShift"] = float(
                    np.sqrt(np.sum(((com_morph - com_int) * spacing_zyx) ** 2.0))
                )

        pca_required = {"VolumeDensityAEE", "AreaDensityAEE"} & required_features
        if pca_required:
            semi_axes = 0.5 * np.asarray(
                [
                    base_features.get("LeastAxisLength", np.nan),
                    base_features.get("MinorAxisLength", np.nan),
                    base_features.get("MajorAxisLength", np.nan),
                ],
                dtype=np.float64,
            )

            if np.any(~np.isfinite(semi_axes)) or np.any(semi_axes <= 0.0):
                if "VolumeDensityAEE" in pca_required:
                    derived["VolumeDensityAEE"] = np.nan
                if "AreaDensityAEE" in pca_required:
                    derived["AreaDensityAEE"] = np.nan
            else:
                if "VolumeDensityAEE" in pca_required:
                    derived["VolumeDensityAEE"] = self._safe_ratio(
                        3.0 * mesh_volume,
                        4.0 * np.pi * float(np.prod(semi_axes)),
                    )
                if "AreaDensityAEE" in pca_required:
                    ellipsoid_area = self._ellipsoid_surface_area_from_semi_axes(
                        semi_axes,
                        n_degree=20,
                    )
                    derived["AreaDensityAEE"] = self._safe_ratio(surface_area, ellipsoid_area)

        hull_required = self._HULL_DENSITY_FEATURES & required_features
        if hull_required:
            hull_volume, hull_area, box_volume, box_area = self._compute_hull_metrics(roi_mask)
            if "VolumeDensityConvexHull" in hull_required:
                derived["VolumeDensityConvexHull"] = self._safe_ratio(mesh_volume, hull_volume)
            if "AreaDensityConvexHull" in hull_required:
                derived["AreaDensityConvexHull"] = self._safe_ratio(surface_area, hull_area)
            if "VolumeDensityAABB" in hull_required:
                derived["VolumeDensityAABB"] = self._safe_ratio(mesh_volume, box_volume)
            if "AreaDensityAABB" in hull_required:
                derived["AreaDensityAABB"] = self._safe_ratio(surface_area, box_area)

        for feature_name in self._DERIVED_FEATURE_ORDER:
            if feature_name in required_features and feature_name not in derived:
                derived[feature_name] = np.nan
        return derived
    
    def execute(self):
        """
        Execute shape feature extraction
        
        Returns:
            OrderedDict of feature values
        """
        mask_c = self._numpy_to_mask(self.inputMask)
        
        spacing_c = (ctypes.c_float * self.ndim)(*self.spacing)
        
        feature_lib = self._feature_lib
        result_ptr = feature_lib.flash_radiomics_shape(
            mask_c,
            spacing_c,
            self.backend_enum,
        )

        enabled_features = self._enabled_features
        self._enabled_features = None
        try:
            features = self._result_to_dict(
                result_ptr,
                "Shape",
                strip_prefix="shape_",
            )
        finally:
            self._enabled_features = enabled_features

        required_derived = (
            set(self._DERIVED_FEATURE_ORDER)
            if enabled_features is None
            else (set(enabled_features) & set(self._DERIVED_FEATURE_ORDER))
        )
        if required_derived:
            features.update(self._compute_derived_features(features, required_derived))

        self.featureValues = self._filter_features(features)
        
        return self.featureValues
