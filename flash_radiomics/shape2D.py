"""
2D Shape-based features

Supports 2D images or 3D images with force2D enabled and a single-slice mask.
"""

# PYRADIOMICS: Feature names and default availability are aligned to PyRadiomics.

import ctypes
import numpy as np
from .base import RadiomicsFeaturesBase
from . import _bindings


class RadiomicsShape2D(RadiomicsFeaturesBase):
    """
    Shape2D features extractor.
    """

    def __init__(self, inputImage, inputMask, **kwargs):
        super().__init__(inputImage, inputMask, **kwargs)

        if "spacing" not in kwargs:
            raise ValueError("Shape2D features require 'spacing' parameter")

        if self.ndim == 3:
            if not kwargs.get("force2D", False):
                raise ValueError("Shape2D requires force2D=True for 3D inputs")

            force2d_dim = kwargs.get("force2Ddimension", 0)
            if self.inputMask.shape[force2d_dim] != 1:
                raise ValueError("Shape2D requires the out-of-plane dimension to have size 1")

            self.inputImage = np.squeeze(self.inputImage, axis=force2d_dim)
            self.inputMask = np.squeeze(self.inputMask, axis=force2d_dim)

            spacing_array = list(self.spacing[::-1])
            axes = [0, 1, 2]
            axes.remove(force2d_dim)
            spacing_selected = [spacing_array[a] for a in axes]
            self.spacing = (spacing_selected[1], spacing_selected[0])

            self.ndim = 2
            self.dims = self.inputImage.shape[::-1]
        elif self.ndim == 2:
            self.spacing = tuple(self.spacing[:2])
        else:
            raise ValueError("Shape2D features are only available in 2D or forced 2D inputs")

    def execute(self):
        """
        Execute shape2D feature extraction.

        Returns:
            OrderedDict of feature values
        """
        mask_c = self._numpy_to_mask(self.inputMask)
        spacing_c = (ctypes.c_float * self.ndim)(*self.spacing)

        feature_lib = self._feature_lib
        result_ptr = feature_lib.flash_radiomics_shape2d(
            mask_c,
            spacing_c,
            self.backend_enum,
        )

        self.featureValues = self._result_to_dict(
            result_ptr,
            "Shape2D",
            strip_prefix="shape2D_",
        )

        # Enable deprecated spherical disproportion only on explicit request.
        if self._enabled_features is None:
            self.featureValues.pop("SphericalDisproportion", None)

        return self.featureValues
