from __future__ import annotations

import logging

import numpy as np
import pywt
from scipy import fft as sp_fft
from scipy import ndimage as ndi
import SimpleITK as sitk

logger = logging.getLogger(__name__)


def getMask(mask, **kwargs):
    """
    Function to get the correct mask. Includes enforcing a correct pixel data type (UInt32).

    Also supports extracting the mask for a segmentation (stored as SimpleITK Vector image) if necessary.
    In this case, the mask at index ``label_channel`` is extracted. The resulting 3D volume is then treated as it were a
    scalar input volume (i.e. with the region of interest defined by voxels with value matching ``label``).

    .. note::
      If only one or non-overlapping Segments are defined when using 3D Slicer, it may be the case that it is stored as a
      labelmap (i.e. only 1 ``label_channel``, with different segmentations identified by different values for ``label``).
      This is easy to check by loading the mask as a SimpleITK image and checking the `GetNumberOfComponentsPerPixel()`,
      if the return value is ``1``, it is a label map (i.e. use ``label``), otherwise it is a VectorImage (i.e. use
      ``label_channel``).

    Finally, checks if the mask volume contains an ROI identified by ``label``. Raises a value error if the label is not
    present (including a list of valid labels found).

    :param mask: SimpleITK Image object representing the mask. Can be a vector image to allow for overlapping masks.
    :param kwargs: keyword arguments. If argument ``label_channel`` is present, this is used to select the channel.
      Otherwise label_channel ``0`` is assumed.
    :return: SimpleITK.Image with pixel type UInt32 representing the mask volume
    """
    label = kwargs.get("label", 1)
    label_channel = kwargs.get("label_channel", 0)
    validate_mask_label = bool(kwargs.get("validateMaskLabel", True))
    if "vector" in mask.GetPixelIDTypeAsString().lower():
        logger.debug(
            "Mask appears to be a segmentation object (=stored as vector image)."
        )
        n_components = mask.GetNumberOfComponentsPerPixel()
        assert (
            label_channel < n_components
        ), f"Mask {label_channel} requested, but segmentation object only contains {n_components} objects"
        msg = f"Extracting mask at index {label_channel}"
        logger.info(msg)
        selector = sitk.VectorIndexSelectionCastImageFilter()
        selector.SetIndex(int(label_channel))
        mask = selector.Execute(mask)

    logger.debug("Force casting mask to UInt32 to ensure correct datatype.")
    mask = sitk.Cast(mask, sitk.sitkUInt32)

    if validate_mask_label:
        labels = np.unique(sitk.GetArrayFromImage(mask))
        if len(labels) == 1 and labels[0] == 0:
            msg = "No labels found in this mask (i.e. nothing is segmented)!"
            raise ValueError(msg)
        if label not in labels:
            msg = (
                f"Label ({label:g}) not present in mask. Choose from {labels[labels != 0]}"
            )
            raise ValueError(msg)

    return mask


def getBinEdges(parameterValues, **kwargs):
    r"""
  Calculate and return the histogram using parameterValues (1D array of all segmented voxels in the image).

  **Fixed bin width:**

  Returns the bin edges, a list of the edges of the calculated bins, length is N(bins) + 1. Bins are defined such, that
  the bin edges are equally spaced from zero, and that the leftmost edge :math:`\leq \min(X_{gl})`. These bin edges
  represent the half-open ranges of each bin :math:`[\text{lower_edge}, \text{upper_edge})` and result in gray value
  discretization as follows:

  .. math::
    X_{b, i} = \lfloor \frac{X_{gl, i}}{W} \rfloor - \lfloor \frac {\min(X_{gl})}{W} \rfloor + 1

  Here, :math:`X_{gl, i}` and :math:`X_{b, i}` are gray level intensities before and after discretization, respectively.
  :math:`{W}` is the bin width value (specified in ``binWidth`` parameter). The first part of the formula ensures that
  the bins are equally spaced from 0, whereas the second part ensures that the minimum gray level intensity inside the
  ROI after binning is always 1.

  In the case where the maximum gray level intensity is equally dividable by the binWidth, i.e.
  :math:`\max(X_{gl}) \mod W = 0`, this will result in that maximum gray level being assigned to bin
  :math:`[\max(X_{gl}), \max(X_{gl}) + W)`, which is consistent with numpy.digitize, but different from the behaviour
  of numpy.histogram, where the final bin has a closed range, including the maximum gray level, i.e.
  :math:`[\max(X_{gl}) - W, \max(X_{gl})]`.

  .. note::
    This method is slightly different from the fixed bin size discretization method described by IBSI. The two most
    notable differences are 1) that PyRadiomics uses a floor division (and adds 1), as opposed to a ceiling division and
    2) that in PyRadiomics, bins are always equally spaced from 0, as opposed to equally spaced from the minimum
    gray level intensity.

  *Example: for a ROI with values ranging from 54 to 166, and a bin width of 25, the bin edges will be [50, 75, 100,
  125, 150, 175].*

  This value can be directly passed to ``numpy.histogram`` to generate a histogram or ``numpy.digitize`` to discretize
  the ROI gray values. See also :py:func:`binImage()`.

  **Fixed bin Count:**

  .. math::
    X_{b, i} = \left\{ {\begin{array}{lcl}
    \lfloor N_b\frac{(X_{gl, i} - \min(X_{gl})}{\max(X_{gl}) - \min(X_{gl})} \rfloor + 1 &
    \mbox{for} & X_{gl, i} < \max(X_{gl}) \\
    N_b & \mbox{for} & X_{gl, i} = \max(X_{gl}) \end{array}} \right.

  Here, :math:`N_b` is the number of bins to use, as defined in ``binCount``.

  References

  - Leijenaar RTH, Nalbantov G, Carvalho S, et al. The effect of SUV discretization in quantitative FDG-PET Radiomics:
    the need for standardized methodology in tumor texture analysis. Sci Rep. 2015;5(August):11075.
  """
    binWidth = kwargs.get("binWidth", 25)
    binCount = kwargs.get("binCount")

    if binCount is not None:
        binEdges = np.histogram(parameterValues, binCount)[1]
        binEdges[
            -1
        ] += 1  # Ensures that the maximum value is included in the topmost bin when using numpy.digitize
    else:
        minimum = min(parameterValues)
        maximum = max(parameterValues)

        binMinimum = kwargs.get("binMinimum")
        if binMinimum is None:
            lowBound = minimum - (minimum % binWidth)
        else:
            lowBound = float(binMinimum)
        # The extra edge keeps the upper boundary half-open for np.digitize.
        highBound = maximum + 2 * binWidth

        binEdges = np.arange(lowBound, highBound, binWidth)

        if len(binEdges) == 1:
            binEdges = [
                binEdges[0] - 0.5,
                binEdges[0] + 0.5,
            ]
        msg = f"Calculated {len(binEdges) - 1} bins for bin width {binWidth} with edges: {binEdges})"
        logger.debug(msg)

    return binEdges  # numpy.histogram(parameterValues, bins=binedges)


def binImage(parameterMatrix, parameterMatrixCoordinates=None, **kwargs):
    r"""
    Discretizes the parameterMatrix (matrix representation of the gray levels in the ROI) using the binEdges calculated
    using :py:func:`getBinEdges`. Only voxels defined by parameterMatrixCoordinates (defining the segmentation) are used
    for calculation of histogram and subsequently discretized. Voxels outside segmentation are left unchanged.
    """
    logger.debug("Discretizing gray levels inside ROI")

    discretizedParameterMatrix = np.zeros(parameterMatrix.shape, dtype="int")
    if parameterMatrixCoordinates is None:
        binEdges = getBinEdges(parameterMatrix.flatten(), **kwargs)
        discretizedParameterMatrix = np.digitize(parameterMatrix, binEdges)
    else:
        binEdges = getBinEdges(parameterMatrix[parameterMatrixCoordinates], **kwargs)
        discretizedParameterMatrix[parameterMatrixCoordinates] = np.digitize(
            parameterMatrix[parameterMatrixCoordinates], binEdges
        )

    return discretizedParameterMatrix, binEdges


def checkMask(imageNode, maskNode, **kwargs):
    """
    Checks whether the Region of Interest (ROI) defined in the mask size and dimensions match constraints, specified in
    settings. The following checks are performed.

    1. Check whether the mask corresponds to the image (i.e. has a similar size, spacing, direction and origin). **N.B.
       This check is performed by SimpleITK, if it fails, an error is logged, with additional error information from
       SimpleITK logged with level DEBUG (i.e. logging-level has to be set to debug to store this information in the log
       file).** The tolerance can be increased using the ``geometryTolerance`` parameter. Alternatively, if the
       ``correctMask`` parameter is ``True``, PyRadiomics will check if the mask contains a valid ROI (inside image
       physical area) and if so, resample the mask to image geometry. See :ref:`radiomics-settings-label` for more info.

    2. Check if the label is present in the mask
    3. Count the number of dimensions in which the size of the ROI > 1 (i.e. does the ROI represent a single voxel (0), a
       line (1), a surface (2) or a volume (3)) and compare this to the minimum number of dimension required (specified in
       ``minimumROIDimensions``).
    4. Optional. Check if there are at least N voxels in the ROI. N is defined in ``minimumROISize``, this test is skipped
       if ``minimumROISize = None``.

    This function returns a tuple of two items. The first item is the bounding box of the mask. The second item is the
    mask that has been corrected by resampling to the input image geometry (if that resampling was successful).

    If a check fails, a ValueError is raised. No features will be extracted for this mask.
    If the mask passes all tests, this function returns the bounding box, which is used in the :py:func:`cropToTumorMask`
    function.

    The bounding box is calculated during (1.) and used for the subsequent checks. The bounding box is
    calculated by SimpleITK.LabelStatisticsImageFilter() and returned as a tuple of indices: (L_x, U_x, L_y, U_y, L_z,
    U_z), where 'L' and 'U' are lower and upper bound, respectively, and 'x', 'y' and 'z' the three image dimensions.

    By reusing the bounding box calculated here, calls to SimpleITK.LabelStatisticsImageFilter() are reduced, improving
    performance.

    Uses the following settings:

    - minimumROIDimensions [1]: Integer, range 1-3, specifies the minimum dimensions (1D, 2D or 3D, respectively).
      Single-voxel segmentations are always excluded.
    - minimumROISize [None]: Integer, > 0,  specifies the minimum number of voxels required. Test is skipped if
      this parameter is set to None.

    .. note::

      If the first check fails there are generally 2 possible causes:

       1. The image and mask are matched, but there is a slight difference in origin, direction or spacing. The exact
          cause, difference and used tolerance are stored with level DEBUG in a log (if enabled). For more information on
          setting up logging, see ":ref:`setting up logging <radiomics-logging-label>`" and the helloRadiomics examples
          (located in the ``pyradiomics/examples`` folder). This problem can be fixed by changing the global tolerance
          (``geometryTolerance`` parameter) or enabling mask correction (``correctMask`` parameter).
       2. The image and mask do not match, but the ROI contained within the mask does represent a physical volume
          contained within the image. If this is the case, resampling is needed to ensure matching geometry between image
          and mask before features can be extracted. This can be achieved by enabling mask correction using the
          ``correctMask`` parameter.
    """
    correctedMask = None

    label = int(kwargs.get("label", 1))
    minDims = kwargs.get("minimumROIDimensions", 2)
    minSize = kwargs.get("minimumROISize")

    msg = f"Checking mask with label {label}"
    logger.debug(msg)
    logger.debug("Calculating bounding box")
    lsif = sitk.LabelStatisticsImageFilter()
    try:
        lsif.Execute(imageNode, maskNode)

        if label not in lsif.GetLabels():
            msg = f"Label ({label:g}) not present in mask"
            raise ValueError(msg)
    except RuntimeError as e:
        if not kwargs.get("correctMask", False):
            if (
                "Both images for LabelStatisticsImageFilter don't match type or dimension!"
                in e.args[0]
            ):
                logger.debug("Additional information on error.", exc_info=True)
                msg = (
                    "Image/Mask datatype or size mismatch. Potential fix: enable correctMask, see "
                    "Documentation:Usage:Customizing the Extraction:Settings:correctMask for more information"
                )
                raise ValueError(msg) from e
            if "Inputs do not occupy the same physical space!" in e.args[0]:
                logger.debug("Additional information on error.", exc_info=True)
                msg = (
                    "Image/Mask geometry mismatch. Potential fix: increase tolerance using geometryTolerance, "
                    "see Documentation:Usage:Customizing the Extraction:Settings:geometryTolerance for more "
                    "information"
                )
                raise ValueError(msg) from e
            raise e  # unhandled error

        logger.warning("Image/Mask geometry mismatch, attempting to correct Mask")

        correctedMask = _correctMask(
            imageNode, maskNode, **kwargs
        )  # Raises Value error if ROI outside image physical space

        try:
            lsif.Execute(imageNode, correctedMask)
        except RuntimeError as e:
            logger.debug(
                "Bounding box calculation with resampled mask failed", exc_info=True
            )
            msg = "Calculation of bounding box failed, for more information run with DEBUG logging and check log"
            raise ValueError(msg) from e

    boundingBox = np.array(lsif.GetBoundingBox(label))

    msg = f"Checking minimum number of dimensions requirements ({minDims})"
    logger.debug(msg)
    ndims = np.sum(
        (boundingBox[1::2] - boundingBox[0::2] + 1) > 1
    )  # UBound - LBound + 1 = Size
    if ndims == 0:
        msg = "mask only contains 1 segmented voxel! Cannot extract features for a single voxel."
        raise ValueError(msg)
    if ndims < minDims:
        msg = f"mask has too few dimensions (number of dimensions {ndims}, minimum required {minDims})"
        raise ValueError(msg)

    if minSize is not None:
        msg = f"Checking minimum size requirements (minimum size: {minSize})"
        logger.debug(msg)
        roiSize = lsif.GetCount(label)
        if roiSize <= minSize:
            msg = f"Size of the ROI is too small (minimum size: {minSize:g}, ROI size: {roiSize:g}"
            raise ValueError(msg)

    return boundingBox, correctedMask


def _correctMask(imageNode, maskNode, **kwargs):
    """
    If the mask geometry does not match the image geometry, this function can be used to resample the mask to the image
    physical space.

    First, the mask is checked for a valid ROI (i.e. maskNode contains an ROI with the given label value, which does not
    include areas outside of the physical image bounds).

    If the ROI is valid, the maskNode is resampled using the imageNode as a reference image and a nearest neighbor
    interpolation.

    If the ROI is valid, the resampled mask is returned, otherwise ``None`` is returned.
    """
    logger.debug("Resampling mask to image geometry")

    _checkROI(imageNode, maskNode, **kwargs)  # Raises a value error if ROI is invalid

    rif = sitk.ResampleImageFilter()
    rif.SetReferenceImage(imageNode)
    rif.SetInterpolator(sitk.sitkNearestNeighbor)

    logger.debug("Resampling...")

    return rif.Execute(maskNode)


def _checkROI(imageNode, maskNode, **kwargs):
    """
    Check whether maskNode contains a valid ROI defined by label:

    1. Check whether the label value is present in the maskNode.
    2. Check whether the ROI defined by the label does not include an area outside the physical area of the image.

    For the second check, a tolerance of 1e-3 is allowed.

    If the ROI is valid, the bounding box (lower bounds, followed by size in all dimensions (X, Y, Z ordered)) is
    returned. Otherwise, a ValueError is raised.
    """
    label = int(kwargs.get("label", 1))

    logger.debug("Checking ROI validity")

    lssif = sitk.LabelShapeStatisticsImageFilter()
    lssif.Execute(maskNode)

    msg = f"Checking if label {label} is persistent in the mask"
    logger.debug(msg)
    if label not in lssif.GetLabels():
        msg = f"Label ({label}) not present in mask"
        raise ValueError(msg)

    bb = np.array(lssif.GetBoundingBox(label))
    Nd = maskNode.GetDimension()

    logger.debug("Comparing physical space of bounding box to physical space of image")
    # Half-voxel offsets convert center indices to physical ROI corners.
    ROIBounds = (
        maskNode.TransformContinuousIndexToPhysicalPoint(bb[:Nd] - 0.5),  # Origin
        maskNode.TransformContinuousIndexToPhysicalPoint(bb[:Nd] + bb[Nd:] - 0.5),
    )  # UBound
    ROIBounds = (
        imageNode.TransformPhysicalPointToContinuousIndex(ROIBounds[0]),  # Origin
        imageNode.TransformPhysicalPointToContinuousIndex(ROIBounds[1]),
    )

    msg = f"ROI bounds (image coordinate space): {ROIBounds}"
    logger.debug(msg)

    tolerance = 1e-3
    if np.any(np.min(ROIBounds, axis=0) < (-0.5 - tolerance)) or np.any(
        np.max(ROIBounds, axis=0) > (np.array(imageNode.GetSize()) - 0.5 + tolerance)
    ):
        msg = (
            "Bounding box of ROI is larger than image space:\n\t"
            f"ROI bounds (x, y, z image coordinate space) {ROIBounds}\n\tImage Size {imageNode.GetSize()}"
        )
        raise ValueError(msg)

    logger.debug("ROI valid, calculating resampling grid")

    return bb


def cropToTumorMask(imageNode, maskNode, boundingBox, **kwargs):
    """
    Create a sitkImage of the segmented region of the image based on the input label.

    Create a sitkImage of the labelled region of the image, cropped to have a
    cuboid shape equal to the ijk boundaries of the label.

    :param boundingBox: The bounding box used to crop the image. This is the bounding box as returned by
      :py:func:`checkMask`.
    :param label: [1], value of the label, onto which the image and mask must be cropped.
    :return: Cropped image and mask (SimpleITK image instances).

    """
    padDistance = kwargs.get("padDistance", 0)

    size = np.array(maskNode.GetSize())

    ijkMinBounds = boundingBox[0::2] - padDistance
    ijkMaxBounds = size - boundingBox[1::2] - padDistance - 1

    ijkMinBounds = np.maximum(ijkMinBounds, 0)
    ijkMaxBounds = np.maximum(ijkMaxBounds, 0)

    msg = f"Cropping to size {(boundingBox[1::2] - boundingBox[0::2]) + 1}"
    logger.debug(msg)
    cif = sitk.CropImageFilter()
    try:
        cif.SetLowerBoundaryCropSize(ijkMinBounds)
        cif.SetUpperBoundaryCropSize(ijkMaxBounds)
    except TypeError:
        cif.SetLowerBoundaryCropSize(ijkMinBounds.tolist())
        cif.SetUpperBoundaryCropSize(ijkMaxBounds.tolist())
    croppedImageNode = cif.Execute(imageNode)
    croppedMaskNode = cif.Execute(maskNode)

    return croppedImageNode, croppedMaskNode


def resampleImage(imageNode, maskNode, **kwargs):
    """
    Resamples image and mask to the specified pixel spacing (The default interpolator is Bspline).

    Resampling can be enabled using the settings 'interpolator' and 'resampledPixelSpacing' in the parameter file or as
    part of the settings passed to the feature extractor. See also
    :ref:`feature extractor <radiomics-featureextractor-label>`.

    'imageNode' and 'maskNode' are SimpleITK Objects, and 'resampledPixelSpacing' is the output pixel spacing (sequence of
    3 elements).

    If only in-plane resampling is required, set the output pixel spacing for the out-of-plane dimension (usually the last
    dimension) to 0. Spacings with a value of 0 are replaced by the spacing as it is in the original mask.

    Only part of the image and labelmap are resampled. The resampling grid is aligned to the input origin, but only voxels
    covering the area of the image ROI (defined by the bounding box) and the padDistance are resampled. This results in a
    resampled and partially cropped image and mask. Additional padding is required as some filters also sample voxels
    outside of segmentation boundaries. For feature calculation, image and mask are cropped to the bounding box without
    any additional padding, as the feature classes do not need the gray level values outside the segmentation.

    The resampling grid is calculated using only the input mask. Even when image and mask have different directions, both
    the cropped image and mask will have the same direction (equal to direction of the mask). Spacing and size are
    determined by settings and bounding box of the ROI.

    .. note::
      Before resampling the bounds of the non-padded ROI are compared to the bounds. If the ROI bounding box includes
      areas outside of the physical space of the image, an error is logged and (None, None) is returned. No features will
      be extracted. This enables the input image and mask to have different geometry, so long as the ROI defines an area
      within the image.

    .. note::
      The additional padding is adjusted, so that only the physical space within the mask is resampled. This is done to
      prevent resampling outside of the image. Please note that this assumes the image and mask to image the same physical
      space. If this is not the case, it is possible that voxels outside the image are included in the resampling grid,
      these will be assigned a value of 0. It is therefore recommended, but not enforced, to use an input mask which has
      the same or a smaller physical space than the image.
    """
    resampledPixelSpacing = kwargs["resampledPixelSpacing"]
    interpolator = kwargs.get("interpolator", sitk.sitkBSpline)
    padDistance = kwargs.get("padDistance", 5)
    label = int(kwargs.get("label", 1))

    logger.debug("Resampling image and mask")

    if imageNode is None or maskNode is None:
        msg = "Requires both image and mask to resample"
        raise ValueError(msg)

    maskSpacing = np.array(maskNode.GetSpacing())
    imageSpacing = np.array(imageNode.GetSpacing())

    Nd_resampled = len(resampledPixelSpacing)
    Nd_mask = len(maskSpacing)
    assert (
        Nd_resampled == Nd_mask
    ), f"Wrong dimensionality ({Nd_resampled}-D) of resampledPixelSpacing!, {Nd_mask}-D required"

    logger.debug(
        "Where resampled spacing is set to 0, set it to the original spacing (mask)"
    )
    resampledPixelSpacing = np.array(resampledPixelSpacing)
    resampledPixelSpacing = np.where(
        resampledPixelSpacing == 0, maskSpacing, resampledPixelSpacing
    )

    bb = _checkROI(imageNode, maskNode, **kwargs)

    # Do not resample in those directions where labelmap spans only one slice.
    maskSize = np.array(maskNode.GetSize())
    resampledPixelSpacing = np.where(
        bb[Nd_mask:] != 1, resampledPixelSpacing, maskSpacing
    )

    logger.debug("Comparing resampled spacing to original spacing (image")
    if np.allclose(imageSpacing, resampledPixelSpacing):
        logger.info(
            "New spacing equal to original image spacing, just resampling the mask"
        )

        rif = sitk.ResampleImageFilter()
        rif.SetReferenceImage(imageNode)
        rif.SetInterpolator(sitk.sitkNearestNeighbor)
        maskNode = rif.Execute(maskNode)

        lssif = sitk.LabelShapeStatisticsImageFilter()
        lssif.Execute(maskNode)
        bb = np.array(lssif.GetBoundingBox(label))

        low_up_bb = np.empty(Nd_mask * 2, dtype=int)
        low_up_bb[::2] = bb[:Nd_mask]
        low_up_bb[1::2] = bb[:Nd_mask] + bb[Nd_mask:] - 1
        return cropToTumorMask(imageNode, maskNode, low_up_bb, **kwargs)

    spacingRatio = maskSpacing / resampledPixelSpacing

    # Outward rounding plus half-voxel padding retains the full segmentation.
    bbNewLBound = np.floor((bb[:Nd_mask] - 0.5) * spacingRatio - padDistance)
    bbNewUBound = np.ceil(
        (bb[:Nd_mask] + bb[Nd_mask:] - 0.5) * spacingRatio + padDistance
    )

    maxUbound = np.ceil(maskSize * spacingRatio) - 1
    bbNewLBound = np.where(bbNewLBound < 0, 0, bbNewLBound)
    bbNewUBound = np.where(bbNewUBound > maxUbound, maxUbound, bbNewUBound)

    newSize = np.array(bbNewUBound - bbNewLBound + 1, dtype="int").tolist()

    bbOriginalLBound = bbNewLBound / spacingRatio

    # Shift from voxel corners to centers before mapping the crop origin.
    newOriginIndex = np.array(0.5 * (resampledPixelSpacing - maskSpacing) / maskSpacing)
    newCroppedOriginIndex = newOriginIndex + bbOriginalLBound
    newOrigin = maskNode.TransformContinuousIndexToPhysicalPoint(newCroppedOriginIndex)

    imagePixelType = imageNode.GetPixelID()
    maskPixelType = maskNode.GetPixelID()

    direction = np.array(maskNode.GetDirection())
    msg = f"Applying resampling from spacing {maskSpacing} and size {maskSize} to spacing {resampledPixelSpacing} and size {newSize}"
    logger.info(msg)

    try:
        if isinstance(interpolator, str):
            interpolator = getattr(sitk, interpolator)
    except Exception:
        msg = f'interpolator "{interpolator}" not recognized, using sitkBSpline'
        logger.warning(msg)
        interpolator = sitk.sitkBSpline

    rif = sitk.ResampleImageFilter()

    rif.SetOutputSpacing(resampledPixelSpacing)
    rif.SetOutputDirection(direction)
    rif.SetSize(newSize)
    rif.SetOutputOrigin(newOrigin)

    logger.debug("Resampling image")
    rif.SetOutputPixelType(imagePixelType)
    rif.SetInterpolator(interpolator)
    resampledImageNode = rif.Execute(imageNode)

    logger.debug("Resampling mask")
    rif.SetOutputPixelType(maskPixelType)
    rif.SetInterpolator(sitk.sitkNearestNeighbor)
    resampledMaskNode = rif.Execute(maskNode)

    return resampledImageNode, resampledMaskNode


def normalizeImage(image, **kwargs):
    r"""
    Normalizes the image by centering it at the mean with standard deviation. Normalization is based on all gray values in
    the image, not just those inside the segmentation.

    :math:`f(x) = \frac{s(x - \mu_x)}{\sigma_x}`

    Where:

    - :math:`x` and :math:`f(x)` are the original and normalized intensity, respectively.
    - :math:`\mu_x` and :math:`\sigma_x` are the mean and standard deviation of the image instensity values.
    - :math:`s` is an optional scaling defined by ``scale``. By default, it is set to 1.

    Optionally, outliers can be removed, in which case values for which :math:`x > \mu_x + n\sigma_x` or
    :math:`x < \mu_x - n\sigma_x` are set to :math:`\mu_x + n\sigma_x` and :math:`\mu_x - n\sigma_x`, respectively.
    Here, :math:`n>0` and defined by ``outliers``. This, in turn, is controlled by the ``removeOutliers`` parameter.
    Removal of outliers is done after the values of the image are normalized, but before ``scale`` is applied.
    """
    scale = kwargs.get("normalizeScale", 1)
    outliers = kwargs.get("removeOutliers")

    msg = f"Normalizing image with scale {scale}"
    logger.debug(msg)
    image = sitk.Normalize(image)

    if outliers is not None:
        msg = f"Removing outliers > {outliers} standard deviations"
        logger.debug(msg)
        imageArr = sitk.GetArrayFromImage(image)

        imageArr[imageArr > outliers] = outliers
        imageArr[imageArr < -outliers] = -outliers

        newImage = sitk.GetImageFromArray(imageArr)
        newImage.CopyInformation(image)
        image = newImage

    image *= scale

    return image


def resegmentMask(imageNode, maskNode, **kwargs):
    r"""
    Resegment the Mask based on the range specified by the threshold(s) in ``resegmentRange``. Either 1 or 2 thresholds
    can be defined. In case of 1 threshold, all values equal to or higher than that threshold are included. If there are
    2 thresholds, all voxels with a value inside the closed-range defined by these thresholds is included
    (i.e. a voxels is included if :math:`T_{lower} \leq X_gl \leq T_{upper}`).
    The resegmented mask is therefore always equal or smaller in size than the original mask.
    In the case where either resegmentRange or resegmentMode contains illegal values, a ValueError is raised.

    There are 3 modes for defining the threshold:

    1. absolute (default): The values in resegmentRange define  as absolute values (i.e. corresponding to the gray values
       in the image
    2. relative: The values in resegmentRange define the threshold as relative to the maximum value found in the ROI.
       (e.g. 0.5 indicates a threshold at 50% of maximum gray value)
    3. sigma: The threshold is defined as the number of sigma from the mean. (e.g. resegmentRange [-3, 3] will include
       all voxels that have a value that differs 3 or less standard deviations from the mean).

    """
    resegmentRange = kwargs["resegmentRange"]
    resegmentMode = kwargs.get("resegmentMode", "absolute")
    label = kwargs.get("label", 1)

    if resegmentRange is None:
        msg = "resegmentRange is None."
        raise ValueError(msg)
    if len(resegmentRange) == 0 or len(resegmentRange) > 2:
        msg = f"Length {len(resegmentRange)} is not allowed for resegmentRange"
        raise ValueError(msg)

    msg = f"Resegmenting mask (range {resegmentRange}, mode {resegmentMode})"
    logger.debug(msg)

    im_arr = sitk.GetArrayFromImage(imageNode)
    ma_arr = sitk.GetArrayFromImage(maskNode) == label  # boolean array

    oldSize = np.sum(ma_arr)

    if resegmentMode == "absolute":
        logger.debug("Resegmenting in absolute mode")
        thresholds = sorted(resegmentRange)
    elif resegmentMode == "relative":
        max_gl = np.max(im_arr[ma_arr])
        msg = f"Resegmenting in relative mode, max {max_gl}"
        logger.debug(msg)
        thresholds = [max_gl * th for th in sorted(resegmentRange)]
    elif resegmentMode == "sigma":
        mean_gl = np.mean(im_arr[ma_arr])
        sd_gl = np.std(im_arr[ma_arr])
        msg = f"Resegmenting in sigma mode, mean {mean_gl}, std {sd_gl}"
        logger.debug(msg)
        thresholds = [mean_gl + sd_gl * th for th in sorted(resegmentRange)]
    else:
        msg = f"Resegment mode {resegmentMode} not recognized."
        raise ValueError(msg)

    msg = f"Applying lower threshold ({thresholds[0]})"
    logger.debug(msg)
    ma_arr[ma_arr] = im_arr[ma_arr] >= thresholds[0]

    if len(thresholds) == 2:
        msg = f"Applying upper threshold ({thresholds[1]})"
        logger.debug(msg)
        ma_arr[ma_arr] = im_arr[ma_arr] <= thresholds[1]

    roiSize = np.sum(ma_arr)

    if roiSize <= 1:
        msg = (
            f"Resegmentation excluded too many voxels with label {label} "
            f"(retained {roiSize} voxel(s))! Cannot extract features"
        )
        raise ValueError(msg)

    newMask_arr = np.zeros(ma_arr.shape, dtype="int")
    newMask_arr[ma_arr] = label

    newMask = sitk.GetImageFromArray(newMask_arr)
    newMask.CopyInformation(maskNode)
    msg = f"Resegmentation complete, new size: {roiSize} voxels (excluded {oldSize - roiSize} voxels)"
    logger.debug(msg)

    return newMask


def getOriginalImage(inputImage, _inputMask, **kwargs):
    """
    This function does not apply any filter, but returns the original image. This function is needed to
    dynamically expose the original image as a valid image type.

    :return: Yields original image, 'original' and ``kwargs``
    """
    logger.debug("Yielding original image")
    yield inputImage, "original", kwargs


def getMeanImage(inputImage, _inputMask, **kwargs):
    """Apply a local arithmetic mean filter and yield the response map."""

    kernel_size = int(kwargs.get("meanKernelSize", kwargs.get("mean_filter_kernel_size", 3)))
    if kernel_size <= 0:
        logger.warning("Mean filter kernel size must be positive: %s", kernel_size)
        return

    boundary_condition = str(
        kwargs.get("meanBoundaryCondition", kwargs.get("mean_filter_boundary_condition", "reflect"))
    )
    boundary_map = {
        "constant": "constant",
        "nearest": "nearest",
        "mirror": "mirror",
        "reflect": "reflect",
        "wrap": "wrap",
    }
    mode = boundary_map.get(boundary_condition.lower(), "reflect")

    arr = sitk.GetArrayFromImage(inputImage).astype(np.float64, copy=False)
    size = [kernel_size] * arr.ndim
    if kwargs.get("force2D", False) and arr.ndim == 3:
        force_axis = int(kwargs.get("force2Ddimension", 0))
        if 0 <= force_axis < arr.ndim:
            size[force_axis] = 1

    response = ndi.uniform_filter(arr, size=tuple(size), mode=mode)
    out = sitk.GetImageFromArray(response.astype(np.float32, copy=False))
    out.CopyInformation(inputImage)

    dimensionality = "2D" if kwargs.get("force2D", False) else f"{arr.ndim}D"
    inputImageName = f"mean-d-{kernel_size}-{dimensionality}"
    logger.debug("Yielding %s image", inputImageName)
    yield out, inputImageName, kwargs


def getLoGImage(inputImage, _inputMask, **kwargs):
    r"""
    Applies a Laplacian of Gaussian filter to the input image and yields a derived image for each sigma value specified.

    A Laplacian of Gaussian image is obtained by convolving the image with the second derivative (Laplacian) of a Gaussian
    kernel.

    The Gaussian kernel is used to smooth the image and is defined as

    .. math::

      G(x, y, z, \sigma) = \frac{1}{(\sigma \sqrt{2 \pi})^3}e^{-\frac{x^2 + y^2 + z^2}{2\sigma^2}}

    The Gaussian kernel is convolved by the laplacian kernel :math:`\nabla^2G(x, y, z)`, which is sensitive to areas with
    rapidly changing intensities, enhancing edges. The width of the filter in the Gaussian kernel is determined by
    :math:`\sigma` and can be used to emphasize more fine (low :math:`\sigma` values) or coarse (high :math:`\sigma`
    values) textures.

    .. warning::

      The LoG filter implemented in PyRadiomics is a 3D LoG filter, and therefore requires 3D input. Features using a
      single slice (2D) segmentation can still be extracted, but the input image *must* be a 3D image, with a minimum size
      in all dimensions :math:`\geq \sigma`. If input image is too small, a warning is logged and :math:`\sigma` value is
      skipped. Moreover, the image size *must* be at least 4 voxels in each dimensions, if this constraint is not met, no
      LoG derived images can be generated.

    Following settings are possible:

    - sigma: List of floats or integers, must be greater than 0. Filter width (mm) to use for the Gaussian kernel
      (determines coarseness).

    .. warning::
      Setting for sigma must be provided. If omitted, no LoG image features are calculated and the function
      will return an empty dictionary.

    Returned filter name reflects LoG settings:
    log-sigma-<sigmaValue>-3D.

    References:

    - `SimpleITK Doxygen documentation
      <https://itk.org/SimpleITKDoxygen/html/classitk_1_1simple_1_1LaplacianRecursiveGaussianImageFilter.html>`_
    - `ITK Doxygen documentation <https://itk.org/Doxygen/html/classitk_1_1LaplacianRecursiveGaussianImageFilter.html>`_
    - `<https://en.wikipedia.org/wiki/Blob_detection#The_Laplacian_of_Gaussian>`_

    :return: Yields log filtered image for each specified sigma, corresponding image type name and ``kwargs`` (customized
      settings).
    """

    logger.debug("Generating LoG images")

    size = np.array(inputImage.GetSize())
    spacing = np.array(inputImage.GetSpacing())
    force2D = bool(kwargs.get("force2D", False))
    force2Ddimension = int(kwargs.get("force2Ddimension", 0))

    if not force2D and np.min(size) < 4:
        msg = f"Image too small to apply LoG filter, size: {size}"
        logger.warning(msg)
        return

    sigmaValues = kwargs.get("sigma", [])
    boundary_condition = str(kwargs.get("logBoundaryCondition", kwargs.get("log_boundary_condition", "mirror")))
    boundary_map = {
        "constant": "constant",
        "nearest": "nearest",
        "mirror": "mirror",
        "reflect": "reflect",
        "wrap": "wrap",
    }
    boundary_mode = boundary_map.get(boundary_condition.lower(), "mirror")

    for sigma in sigmaValues:
        msg = f"Computing LoG with sigma {sigma}"
        logger.info(msg)

        if sigma > 0.0:
            if force2D or kwargs.get("logUseScipy", False):
                image_arr = sitk.GetArrayFromImage(inputImage).astype(np.float64, copy=False)
                array_spacing = np.array([spacing[2], spacing[1], spacing[0]], dtype=float)
                sigma_voxels = sigma / array_spacing
                if force2D and 0 <= force2Ddimension < image_arr.ndim:
                    sigma_voxels[force2Ddimension] = 0.0

                if np.any(np.array(image_arr.shape)[sigma_voxels > 0] < np.ceil(sigma_voxels[sigma_voxels > 0]) + 1):
                    msg = (
                        "applyLoG: sigma/spacing + 1 must be greater than the size "
                        f"of filtered dimensions for 2D LoG, sigma: {sigma}, spacing: {spacing}, size: {size}"
                    )
                    logger.warning(msg)
                    continue

                response = np.zeros_like(image_arr, dtype=np.float64)
                filter_axes = [axis for axis, axis_sigma in enumerate(sigma_voxels) if axis_sigma > 0.0]
                for axis in filter_axes:
                    order = [0] * image_arr.ndim
                    order[axis] = 2
                    response += ndi.gaussian_filter(
                        image_arr,
                        sigma=tuple(float(value) for value in sigma_voxels),
                        order=tuple(order),
                        mode=boundary_mode,
                        truncate=float(kwargs.get("logTruncate", 4.0)),
                    )
                dimensionality = "2D" if force2D else "3D"
                inputImageName = f"log-sigma-{str(sigma).replace('.', '-')}-mm-{dimensionality}"
                out = sitk.GetImageFromArray(response.astype(np.float32, copy=False))
                out.CopyInformation(inputImage)
                msg = f"Yielding {inputImageName} image"
                logger.debug(msg)
                yield out, inputImageName, kwargs
                continue

            if np.all(size >= np.ceil(sigma / spacing) + 1):
                lrgif = sitk.LaplacianRecursiveGaussianImageFilter()
                lrgif.SetNormalizeAcrossScale(bool(kwargs.get("logNormalizeAcrossScale", True)))
                lrgif.SetSigma(sigma)
                inputImageName = f"log-sigma-{str(sigma).replace('.', '-')}-mm-3D"
                msg = f"Yielding {inputImageName} image"
                logger.debug(msg)
                yield lrgif.Execute(inputImage), inputImageName, kwargs
            else:
                msg = f"applyLoG: sigma({sigma})/spacing({spacing}) + 1 must be greater than the size({size}) of the inputImage"
                logger.warning(msg)
        else:
            msg = f"applyLoG: sigma must be greater than 0.0: {sigma}"
            logger.warning(msg)


def _ibsi_boundary_mode(value, default="mirror"):
    mode = str(value or default).lower()
    if mode == "periodic":
        return "wrap"
    if mode in {"constant", "nearest", "mirror", "reflect", "wrap"}:
        return mode
    return default


def _pool_arrays(current, new, method):
    if current is None:
        return np.array(new, dtype=np.float64, copy=True)
    if method == "max":
        return np.maximum(current, new)
    if method == "min":
        return np.minimum(current, new)
    if method in {"mean", "average", "sum"}:
        return current + new
    raise ValueError(f"Unknown pooling method: {method}")


def _as_sitk_image(array, reference_image):
    out = sitk.GetImageFromArray(np.asarray(array, dtype=np.float32))
    out.CopyInformation(reference_image)
    return out


def _convolve_separable(array, kernels, mode, cval=0.0):
    result = np.asarray(array, dtype=np.float64)
    for axis, kernel in enumerate(kernels):
        if kernel is not None:
            result = ndi.convolve1d(result, np.asarray(kernel, dtype=np.float64), axis=axis, mode=mode, cval=cval)
    return result


def _kernel_symbols_for_permutations(kernels):
    symbols = []
    for axis, kernel in enumerate(kernels):
        base = f"g{axis}"
        flipped = f"jg{axis}"
        for previous_axis, previous_kernel in enumerate(kernels[:axis]):
            if np.array_equiv(kernel, previous_kernel):
                base = f"g{previous_axis}"
                flipped = f"jg{previous_axis}"
                break
        if np.array_equiv(kernel, np.flip(kernel)):
            flipped = base
        symbols.append((base, flipped))
    return symbols


def _permuted_separable_kernels(kernels, rotational_invariance):
    axis_offset = 0
    prefix = ()
    if kernels and kernels[0] is None:
        axis_offset = 1
        prefix = (None,)
        kernels = tuple(kernels[1:])
    else:
        kernels = tuple(kernels)
    if not rotational_invariance:
        return [prefix + kernels]

    symbols = _kernel_symbols_for_permutations(kernels)
    lookup = {}
    for axis, kernel in enumerate(kernels):
        lookup[f"g{axis}"] = kernel
        lookup[f"jg{axis}"] = np.flip(kernel)

    if len(kernels) == 2:
        x, y = symbols
        patterns = [
            (x[0], y[0]),
            (y[1], x[0]),
            (x[1], y[1]),
            (y[0], x[1]),
        ]
    elif len(kernels) == 3:
        x, y, z = symbols
        patterns = [
            (x[0], y[0], z[0]),
            (z[1], y[0], x[0]),
            (x[1], y[0], z[1]),
            (z[0], y[0], x[1]),
            (y[0], z[0], x[0]),
            (y[0], z[1], x[1]),
            (y[0], x[1], z[0]),
            (x[1], y[1], z[0]),
            (y[1], x[0], z[0]),
            (z[1], x[1], y[0]),
            (z[1], y[1], x[1]),
            (z[1], x[0], y[1]),
            (y[1], x[1], z[1]),
            (x[0], y[1], z[1]),
            (y[0], x[0], z[1]),
            (z[0], x[1], y[1]),
            (z[0], y[1], x[0]),
            (z[0], x[0], y[0]),
            (x[1], z[0], y[0]),
            (y[1], z[0], x[1]),
            (x[0], z[0], y[1]),
            (x[1], z[1], y[1]),
            (y[1], z[1], x[0]),
            (x[0], z[1], y[0]),
        ]
    else:
        return [kernels]

    unique = []
    seen = set()
    for pattern in patterns:
        key = tuple(tuple(np.asarray(lookup[symbol], dtype=np.float64)) for symbol in pattern)
        if key not in seen:
            seen.add(key)
            unique.append(prefix + tuple(np.asarray(lookup[symbol], dtype=np.float64) for symbol in pattern))
    return unique


def _separable_permutation_patterns(kernels):
    symbols = _kernel_symbols_for_permutations(kernels)
    if len(kernels) == 2:
        x, y = symbols
        return [
            (x[0], y[0]),
            (y[1], x[0]),
            (x[1], y[1]),
            (y[0], x[1]),
        ]
    if len(kernels) == 3:
        x, y, z = symbols
        return [
            (x[0], y[0], z[0]),
            (z[1], y[0], x[0]),
            (x[1], y[0], z[1]),
            (z[0], y[0], x[1]),
            (y[0], z[0], x[0]),
            (y[0], z[1], x[1]),
            (y[0], x[1], z[0]),
            (x[1], y[1], z[0]),
            (y[1], x[0], z[0]),
            (z[1], x[1], y[0]),
            (z[1], y[1], x[1]),
            (z[1], x[0], y[1]),
            (y[1], x[1], z[1]),
            (x[0], y[1], z[1]),
            (y[0], x[0], z[1]),
            (z[0], x[1], y[1]),
            (z[0], y[1], x[0]),
            (z[0], x[0], y[0]),
            (x[1], z[0], y[0]),
            (y[1], z[0], x[1]),
            (x[0], z[0], y[1]),
            (x[1], z[1], y[1]),
            (y[1], z[1], x[0]),
            (x[0], z[1], y[0]),
        ]
    return [tuple(f"g{axis}" for axis in range(len(kernels)))]


def _lookup_for_kernels(kernels):
    lookup = {}
    for axis, kernel in enumerate(kernels):
        lookup[f"g{axis}"] = kernel
        lookup[f"jg{axis}"] = np.flip(kernel)
    return lookup


def _permuted_separable_kernel_pairs(kernels, pre_kernels, rotational_invariance):
    prefix = ()
    if kernels and kernels[0] is None:
        prefix = (None,)
        kernels = tuple(kernels[1:])
        pre_kernels = tuple(pre_kernels[1:])
    else:
        kernels = tuple(kernels)
        pre_kernels = tuple(pre_kernels)

    if not rotational_invariance:
        return [(prefix + kernels, prefix + pre_kernels)]

    filter_patterns = _separable_permutation_patterns(kernels)
    pre_patterns = _separable_permutation_patterns(pre_kernels)
    filter_lookup = _lookup_for_kernels(kernels)
    pre_lookup = _lookup_for_kernels(pre_kernels)

    pairs = []
    seen = set()
    for filter_pattern, pre_pattern in zip(filter_patterns, pre_patterns):
        current_filters = prefix + tuple(np.asarray(filter_lookup[symbol], dtype=np.float64) for symbol in filter_pattern)
        current_pre = prefix + tuple(np.asarray(pre_lookup[symbol], dtype=np.float64) for symbol in pre_pattern)
        key = (
            tuple(None if value is None else tuple(value) for value in current_filters),
            tuple(None if value is None else tuple(value) for value in current_pre),
        )
        if key not in seen:
            seen.add(key)
            pairs.append((current_filters, current_pre))
    return pairs


def _laws_kernel(name):
    kernels = {
        "l5": [1.0, 4.0, 6.0, 4.0, 1.0],
        "e5": [-1.0, -2.0, 0.0, 2.0, 1.0],
        "s5": [-1.0, 0.0, 2.0, 0.0, -1.0],
        "w5": [-1.0, 2.0, 0.0, -2.0, 1.0],
        "r5": [1.0, -4.0, 6.0, -4.0, 1.0],
        "l3": [1.0, 2.0, 1.0],
        "e3": [-1.0, 0.0, 1.0],
        "s3": [-1.0, 2.0, -1.0],
    }
    kernel = np.asarray(kernels[str(name).lower()], dtype=np.float64)
    return kernel / np.sqrt(np.sum(kernel * kernel))


def getLawsImage(inputImage, _inputMask, **kwargs):
    """Apply IBSI Laws response and optional texture-energy map."""

    arr = sitk.GetArrayFromImage(inputImage).astype(np.float64, copy=False)
    response_map = str(kwargs.get("lawsResponseMap", kwargs.get("response_map", "")))
    mode = _ibsi_boundary_mode(kwargs.get("lawsBoundaryCondition", kwargs.get("boundary_condition", "mirror")))
    cval = float(kwargs.get("lawsConstantValue", kwargs.get("constant_value", 0.0)))
    energy_map = bool(kwargs.get("lawsEnergyMap", kwargs.get("energy_map", False)))
    delta = int(kwargs.get("lawsDelta", kwargs.get("energy_map_distance_delta_voxels", 7)))
    rotational = bool(kwargs.get("lawsRotationalInvariance", kwargs.get("pseudo_rotational_invariance", False)))
    pooling = str(kwargs.get("lawsPoolingMethod", kwargs.get("response_map_pooling", "max"))).lower()

    parts = [response_map[index:index + 2] for index in range(0, len(response_map), 2)]
    if not parts:
        logger.warning("No Laws response map configured")
        return
    kernels_xyz = [_laws_kernel(part) for part in parts]
    if len(kernels_xyz) == 2:
        kernels_zyx = (None, kernels_xyz[1], kernels_xyz[0])
    else:
        kernels_zyx = tuple(reversed(kernels_xyz))

    response = None
    filter_sets = _permuted_separable_kernels(kernels_zyx, rotational)
    for kernels in filter_sets:
        response = _pool_arrays(response, _convolve_separable(arr, kernels, mode=mode, cval=cval), pooling)
    if pooling in {"mean", "average"}:
        response = response / float(len(filter_sets))

    if energy_map:
        response = np.abs(response)
        energy_kernel = np.ones(2 * delta + 1, dtype=np.float64) / float(2 * delta + 1)
        energy_kernels = tuple(None if kernel is None else energy_kernel for kernel in kernels_zyx)
        response = _convolve_separable(response, energy_kernels, mode=mode, cval=cval)

    name = f"laws-{response_map.lower()}"
    if energy_map:
        name += f"-energy-delta-{delta}"
    yield _as_sitk_image(response, inputImage), name, kwargs


def _gabor_response_2d(slice_array, sigma, gamma, wavelength, theta, mode, response_type):
    y_size, x_size = slice_array.shape
    x_size = int(1 + 2 * np.floor(x_size / 2.0))
    y_size = int(1 + 2 * np.floor(y_size / 2.0))
    y, x = np.mgrid[:y_size, :x_size].astype(float)
    y -= (y_size - 1.0) / 2.0
    x -= (x_size - 1.0) / 2.0
    rotation_matrix = np.array([[-np.cos(theta), np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    rotated = np.dot(rotation_matrix, np.array((y.flatten(), x.flatten())))
    y = rotated[0, :].reshape((y_size, x_size))
    x = rotated[1, :].reshape((y_size, x_size))
    gabor_filter = np.exp(
        -(x * x + gamma * gamma * y * y) / (2.0 * sigma * sigma)
        + 1.0j * (2.0 * np.pi * x) / wavelength
    )
    pad_y = int(np.floor(gabor_filter.shape[0] / 2.0))
    pad_x = int(np.floor(gabor_filter.shape[1] / 2.0))
    pad_mode = "reflect" if mode == "mirror" else ("edge" if mode == "nearest" else mode)
    if pad_mode == "constant":
        padded = np.pad(slice_array, ((pad_y, pad_y), (pad_x, pad_x)), mode=pad_mode, constant_values=0.0)
    else:
        padded = np.pad(slice_array, ((pad_y, pad_y), (pad_x, pad_x)), mode=pad_mode)
    f_image = sp_fft.fft2(padded)
    f_filter = sp_fft.fft2(gabor_filter, padded.shape)
    response = sp_fft.ifft2(f_image * f_filter)
    response = response[2 * pad_y:2 * pad_y + slice_array.shape[0], 2 * pad_x:2 * pad_x + slice_array.shape[1]]
    return _real_response(response, response_type)


def _apply_gabor_axis(arr, axis, sigma, gamma, wavelength, theta, mode, response_type):
    moved = np.moveaxis(arr, axis, 0)
    out = np.empty_like(moved, dtype=np.float64)
    for index in range(moved.shape[0]):
        out[index] = _gabor_response_2d(moved[index], sigma, gamma, wavelength, theta, mode, response_type)
    return np.moveaxis(out, 0, axis)


def getGaborImage(inputImage, _inputMask, **kwargs):
    """Apply IBSI 2D Gabor response maps."""

    arr = sitk.GetArrayFromImage(inputImage).astype(np.float64, copy=False)
    spacing = np.asarray(inputImage.GetSpacing(), dtype=np.float64)
    array_spacing = np.asarray([spacing[2], spacing[1], spacing[0]], dtype=np.float64)
    sigma_mm = float(kwargs.get("gaborSigma", kwargs.get("scale_sigma_star_mm", 5.0)))
    wavelength_mm = float(kwargs.get("gaborLambda", kwargs.get("wavelength_lambda_star_mm", 2.0)))
    gamma = float(kwargs.get("gaborGamma", kwargs.get("ellipticity_gamma", 1.5)))
    mode = _ibsi_boundary_mode(kwargs.get("gaborBoundaryCondition", kwargs.get("boundary_condition", "mirror")))
    response_type = str(kwargs.get("gaborResponse", kwargs.get("response_map", "modulus"))).lower()
    pooling = str(kwargs.get("gaborPoolingMethod", kwargs.get("response_map_pooling", "mean"))).lower()
    stack_axes = kwargs.get("gaborStackAxes", [0])
    if isinstance(stack_axes, int):
        stack_axes = [stack_axes]
    theta_values = kwargs.get("gaborThetaValues")
    if theta_values is None:
        delta = float(kwargs.get("gaborThetaStep", np.pi / 8.0))
        theta_values = list(np.arange(0.0, np.pi, delta))

    response = None
    count = 0
    for axis in stack_axes:
        in_plane_spacing = max(array_spacing[idx] for idx in range(arr.ndim) if idx != int(axis))
        sigma = sigma_mm / float(in_plane_spacing)
        wavelength = wavelength_mm / float(in_plane_spacing)
        for theta in theta_values:
            current = _apply_gabor_axis(arr, int(axis), sigma, gamma, wavelength, float(theta), mode, response_type)
            response = _pool_arrays(response, current, pooling)
            count += 1
    if pooling in {"mean", "average"} and count > 0:
        response = response / float(count)

    yield _as_sitk_image(response, inputImage), "gabor", kwargs


def _wavelet_kernel(wavelet_name, component):
    wavelet = pywt.Wavelet(wavelet_name)
    kernel = np.asarray(wavelet.dec_lo if component.lower() == "l" else wavelet.dec_hi, dtype=np.float64)
    if kernel.size % 2 == 0:
        kernel = np.append(kernel, 0.0)
    return kernel


def _dilate_filter(kernel):
    if kernel is None:
        return None
    out = np.zeros(kernel.size * 2 - 1, dtype=np.float64)
    out[::2] = kernel
    return out


def getIBSIWaveletImage(inputImage, _inputMask, **kwargs):
    """Apply IBSI undecimated separable wavelet maps."""

    arr = sitk.GetArrayFromImage(inputImage).astype(np.float64, copy=False)
    wavelet_name = str(kwargs.get("ibsiWaveletName", kwargs.get("wavelet", "db3")))
    combination = str(kwargs.get("ibsiWaveletCombination", kwargs.get("wavelet_filter_combination", "LH"))).lower()
    level = int(kwargs.get("ibsiWaveletLevel", kwargs.get("level", 1)))
    mode = _ibsi_boundary_mode(kwargs.get("ibsiWaveletBoundaryCondition", kwargs.get("boundary_condition", "mirror")))
    rotational = bool(kwargs.get("ibsiWaveletRotationalInvariance", kwargs.get("pseudo_rotational_invariance", False)))
    pooling = str(kwargs.get("ibsiWaveletPoolingMethod", kwargs.get("response_map_pooling", "mean"))).lower()

    kernels_xyz = [_wavelet_kernel(wavelet_name, component) for component in combination]
    if len(kernels_xyz) == 2:
        kernels_zyx = (None, kernels_xyz[1], kernels_xyz[0])
    else:
        kernels_zyx = tuple(reversed(kernels_xyz))
    pre_kernel = np.asarray(pywt.Wavelet(wavelet_name).dec_lo, dtype=np.float64)
    if pre_kernel.size % 2 == 0:
        pre_kernel = np.append(pre_kernel, 0.0)
    pre_kernels = tuple(None if kernel is None else pre_kernel.copy() for kernel in kernels_zyx)

    response = None
    filter_pairs = _permuted_separable_kernel_pairs(kernels_zyx, pre_kernels, rotational)
    for kernels, pre_filters in filter_pairs:
        current = arr
        current_kernels = tuple(None if kernel is None else np.array(kernel, copy=True) for kernel in kernels)
        current_pre = tuple(None if kernel is None else np.array(kernel, copy=True) for kernel in pre_filters)
        for decomposition_level in range(1, level + 1):
            if decomposition_level < level:
                current = _convolve_separable(current, current_pre, mode=mode)
                current_pre = tuple(_dilate_filter(kernel) for kernel in current_pre)
                current_kernels = tuple(_dilate_filter(kernel) for kernel in current_kernels)
            else:
                current = _convolve_separable(current, current_kernels, mode=mode)
        response = _pool_arrays(response, current, pooling)
    if pooling in {"mean", "average"}:
        response = response / float(len(filter_pairs))

    yield _as_sitk_image(response, inputImage), f"ibsiwavelet-{wavelet_name}-{combination}-level-{level}", kwargs


def _simoncelli_filter(shape, level):
    grid_center = (np.asarray(shape, dtype=np.float64) - 1.0) / 2.0
    coordinates = [(values - center) / center for values, center in zip(np.indices(shape, sparse=True), grid_center)]
    distance_grid = np.sqrt(sum(np.square(values) for values in coordinates))
    max_frequency = 1.0 / 2.0 ** (float(level) - 1.0)
    response = np.zeros(shape, dtype=np.float64)
    mask = np.logical_and(distance_grid >= max_frequency / 4.0, distance_grid <= max_frequency)
    response[mask] = np.cos(np.pi / 2.0 * np.log2(2.0 * distance_grid[mask] / max_frequency))
    return response


def _coordinate_grids(shape):
    grid_center = (np.asarray(shape, dtype=np.float64) - 1.0) / 2.0
    sparse = np.indices(shape, sparse=True)
    return [
        np.broadcast_to((values - center) / center, shape)
        for values, center in zip(sparse, grid_center)
    ]


def _riesz_prefactor(order):
    from math import factorial

    order = tuple(int(value) for value in order)
    order_sum = int(sum(order))
    denom = 1
    for value in order:
        denom *= factorial(value)
    return (-1.0j) ** order_sum * np.sqrt(float(factorial(order_sum)) / float(denom))


def _riesz_multiplier(shape, order):
    order = tuple(int(value) for value in order)
    order_sum = int(sum(order))
    if order_sum <= 0:
        return np.ones(shape, dtype=np.complex128)

    coordinates = _coordinate_grids(shape)
    distance = np.sqrt(sum(np.square(values) for values in coordinates))
    gradient = np.ones(shape, dtype=np.complex128)
    for axis, axis_order in enumerate(order):
        if axis_order:
            gradient *= np.power(coordinates[axis], axis_order)

    with np.errstate(divide="ignore", invalid="ignore"):
        multiplier = _riesz_prefactor(order) * gradient / np.power(distance, order_sum)
    multiplier = np.asarray(multiplier, dtype=np.complex128)
    multiplier[~np.isfinite(multiplier)] = 0.0
    return multiplier


def _multi_indices(order_sum, ndim):
    if ndim == 1:
        yield (order_sum,)
        return
    for value in range(order_sum + 1):
        for rest in _multi_indices(order_sum - value, ndim - 1):
            yield (value,) + rest


def _riesz_response(arr, level, order):
    order = tuple(int(value) for value in order)
    filt = sp_fft.ifftshift(_simoncelli_filter(arr.shape, level))
    multiplier = sp_fft.ifftshift(_riesz_multiplier(arr.shape, order))
    return sp_fft.ifftn(sp_fft.fftn(arr) * filt * multiplier)


def _principal_orientation(arr, sigma):
    gradients = [np.asarray(component, dtype=np.float64) for component in np.gradient(arr.astype(np.float64, copy=False))]
    if sigma and float(sigma) > 0.0:
        gradients = [
            ndi.gaussian_filter(component, sigma=float(sigma), mode="mirror")
            for component in gradients
        ]

    norm = np.sqrt(sum(component * component for component in gradients))
    orientation = np.zeros(arr.shape + (arr.ndim,), dtype=np.float64)
    valid = norm > 0.0
    for axis, component in enumerate(gradients):
        with np.errstate(divide="ignore", invalid="ignore"):
            orientation[..., axis] = np.where(valid, component / norm, 0.0)
    return orientation


def _steered_riesz_response(arr, level, order, sigma):
    order = tuple(int(value) for value in order)
    order_sum = int(sum(order))
    if order_sum <= 0:
        return _riesz_response(arr, level, order)

    orientation = _principal_orientation(arr, sigma)
    response = np.zeros(arr.shape, dtype=np.complex128)
    for component_order in _multi_indices(order_sum, arr.ndim):
        component = _riesz_response(arr, level, component_order)
        coeff = np.ones(arr.shape, dtype=np.float64) * np.real(_riesz_prefactor(component_order))
        for axis, axis_order in enumerate(component_order):
            if axis_order:
                coeff *= np.power(orientation[..., axis], axis_order)
        response += coeff * component
    return response


def _real_response(current, response_type):
    if response_type in {"modulus", "abs", "magnitude"}:
        return np.abs(current)
    if response_type == "real":
        return np.real(current)
    if response_type in {"imaginary", "imag"}:
        return np.imag(current)
    if response_type in {"angle", "phase", "argument"}:
        return np.angle(current)
    raise ValueError(f"Unknown complex response type: {response_type}")


def getSimoncelliWaveletImage(inputImage, _inputMask, **kwargs):
    """Apply IBSI Simoncelli non-separable wavelet maps."""

    arr = sitk.GetArrayFromImage(inputImage).astype(np.float64, copy=False)
    level = int(kwargs.get("simoncelliLevel", kwargs.get("wavelet_decomposition_level", 1)))
    response_type = str(kwargs.get("simoncelliResponse", "real")).lower()
    force2d = bool(kwargs.get("force2D", False))
    force_axis = int(kwargs.get("force2Ddimension", 0))

    if force2d:
        moved = np.moveaxis(arr, force_axis, 0)
        response = np.empty_like(moved, dtype=np.float64)
        filt = sp_fft.ifftshift(_simoncelli_filter(moved.shape[1:], level))
        for index in range(moved.shape[0]):
            current = sp_fft.ifftn(sp_fft.fftn(moved[index]) * filt)
            if response_type in {"modulus", "abs", "magnitude"}:
                response[index] = np.abs(current)
            elif response_type == "real":
                response[index] = np.real(current)
            else:
                response[index] = np.imag(current)
        response = np.moveaxis(response, 0, force_axis)
    else:
        filt = sp_fft.ifftshift(_simoncelli_filter(arr.shape, level))
        current = sp_fft.ifftn(sp_fft.fftn(arr) * filt)
        if response_type in {"modulus", "abs", "magnitude"}:
            response = np.abs(current)
        elif response_type == "real":
            response = np.real(current)
        else:
            response = np.imag(current)

    yield _as_sitk_image(response, inputImage), f"simoncelli-level-{level}", kwargs


def getRieszTransformedSimoncelliWaveletImage(inputImage, _inputMask, **kwargs):
    """Apply an IBSI Riesz-transformed Simoncelli non-separable wavelet map."""

    arr = sitk.GetArrayFromImage(inputImage).astype(np.float64, copy=False)
    level = int(kwargs.get("rieszSimoncelliLevel", kwargs.get("wavelet_decomposition_level", 1)))
    response_type = str(kwargs.get("rieszSimoncelliResponse", "real")).lower()
    order = kwargs.get("rieszOrder", kwargs.get("riesz_transformation_order"))
    if order is None:
        logger.warning("No Riesz transformation order configured")
        return
    order = tuple(int(value) for value in order)
    force2d = bool(kwargs.get("force2D", False))
    force_axis = int(kwargs.get("force2Ddimension", 0))
    steering = bool(kwargs.get("rieszSteering", kwargs.get("riesz_filter_steering", False)))
    tensor_sigma = float(kwargs.get("rieszTensorSigma", kwargs.get("riesz_structure_tensor_window_scale_sigma_star_mm", 1.0)))

    if force2d:
        moved = np.moveaxis(arr, force_axis, 0)
        response = np.empty_like(moved, dtype=np.float64)
        for index in range(moved.shape[0]):
            if steering:
                current = _steered_riesz_response(moved[index], level, order, tensor_sigma)
            else:
                current = _riesz_response(moved[index], level, order)
            response[index] = _real_response(current, response_type)
        response = np.moveaxis(response, 0, force_axis)
    else:
        if steering:
            current = _steered_riesz_response(arr, level, order, tensor_sigma)
        else:
            current = _riesz_response(arr, level, order)
        response = _real_response(current, response_type)

    name = f"riesz-simoncelli-level-{level}-order-{'-'.join(str(value) for value in order)}"
    if steering:
        name += "-steered"
    yield _as_sitk_image(response, inputImage), name, kwargs


def getWaveletImage(inputImage, _inputMask, **kwargs):
    """
    Applies wavelet filter to the input image and yields the decompositions and the approximation.

    Following settings are possible:

    - start_level [0]: integer, 0 based level of wavelet which should be used as first set of decompositions
      from which a signature is calculated
    - level [1]: integer, number of levels of wavelet decompositions from which a signature is calculated.
    - wavelet ["coif1"]: string, type of wavelet decomposition. Enumerated value, validated against possible values
      present in the ``pyWavelet.wavelist()``. Current possible values (pywavelet version 0.4.0) (where an
      additional number is needed, range of values is indicated in []):

      - haar
      - dmey
      - sym[2-20]
      - db[1-20]
      - coif[1-5]
      - bior[1.1, 1.3, 1.5, 2.2, 2.4, 2.6, 2.8, 3.1, 3.3, 3.5, 3.7, 3.9, 4.4, 5.5, 6.8]
      - rbio[1.1, 1.3, 1.5, 2.2, 2.4, 2.6, 2.8, 3.1, 3.3, 3.5, 3.7, 3.9, 4.4, 5.5, 6.8]

    Returned filter name reflects wavelet type:
    wavelet[level]-<decompositionName>

    N.B. only levels greater than the first level are entered into the name.

    :return: Yields each wavelet decomposition and final approximation, corresponding imaget type name and ``kwargs``
      (customized settings).
    """
    logger.debug("Generating Wavelet images")

    Nd = inputImage.GetDimension()
    axes = list(range(Nd - 1, -1, -1))
    if kwargs.get("force2D", False):
        axes.remove(kwargs.get("force2Ddimension", 0))

    approx, ret = _swt3(inputImage, tuple(axes), **kwargs)

    for idx, wl in enumerate(ret, start=1):
        for decompositionName, decompositionImage in wl.items():
            msg = f"Computing Wavelet {decompositionName}"
            logger.info(msg)

            if idx == 1:
                inputImageName = f"wavelet-{decompositionName}"
            else:
                inputImageName = f"wavelet{idx}-{decompositionName}"
            msg = f"Yielding {inputImageName} image"
            logger.debug(msg)
            yield decompositionImage, inputImageName, kwargs

    if len(ret) == 1:
        inputImageName = f"wavelet-{'L' * len(axes)}"
    else:
        inputImageName = f"wavelet{len(ret)}-{'L' * len(axes)}"
    msg = f"Yielding approximation ({inputImageName}) image"
    logger.debug(msg)
    yield approx, inputImageName, kwargs


def _swt3(inputImage, axes, **kwargs):  # Stationary Wavelet Transform 3D
    wavelet = kwargs.get("wavelet", "coif1")
    level = kwargs.get("level", 1)
    start_level = kwargs.get("start_level", 0)

    matrix = sitk.GetArrayFromImage(
        inputImage
    )  # This function gets a numpy array from the SimpleITK Image "inputImage"
    matrix = np.asarray(
        matrix
    )  # The function np.asarray converts "matrix" (which could be also a tuple) into an array.

    original_shape = matrix.shape
    padding = tuple([(0, 1 if dim % 2 != 0 else 0) for dim in original_shape])
    data = matrix.copy()  # creates a modifiable copy of "matrix" and we call it "data"
    data = np.pad(
        data, padding, "wrap"
    )  # padding the tuple "padding" previously computed

    if not isinstance(wavelet, pywt.Wavelet):
        wavelet = pywt.Wavelet(wavelet)

    for _i in range(
        start_level
    ):  # if start_level = 0 (default) this for loop never gets executed
        dec = pywt.swtn(data, wavelet, level=1, start_level=0, axes=axes)[0]
        data = dec["a" * len(axes)].copy()

    ret = []  # initialize empty list
    for _i in range(start_level, start_level + level):
        dec = pywt.swtn(data, wavelet, level=1, start_level=0, axes=axes)[0]
        data = dec["a" * len(axes)].copy()

        dec_im = {}  # initialize empty dict
        for decName, decImage in dec.items():
            if decName == "a" * len(axes):
                continue
            decTemp = decImage.copy()
            decTemp = decTemp[
                tuple(
                    slice(None, -1 if dim % 2 != 0 else None) for dim in original_shape
                )
            ]
            sitkImage = sitk.GetImageFromArray(decTemp)
            sitkImage.CopyInformation(inputImage)
            dec_im[str(decName).replace("a", "L").replace("d", "H")] = sitkImage

        ret.append(
            dec_im
        )  # appending all the filtered sitk images (stored in "dec_im") to the "ret" list

    data = data[
        tuple(slice(None, -1 if dim % 2 != 0 else None) for dim in original_shape)
    ]
    approximation = sitk.GetImageFromArray(data)
    approximation.CopyInformation(inputImage)

    return (
        approximation,
        ret,
    )  # returns the approximation and the detail (ret) coefficients of the stationary wavelet decomposition


def getSquareImage(inputImage, _inputMask, **kwargs):
    r"""
    Computes the square of the image intensities.

    Resulting values are rescaled on the range of the initial original image and negative intensities are made
    negative in resultant filtered image.

    :math:`f(x) = (cx)^2,\text{ where } c=\displaystyle\frac{1}{\sqrt{\max(|x|)}}`

    Where :math:`x` and :math:`f(x)` are the original and filtered intensity, respectively.

    :return: Yields square filtered image, 'square' and ``kwargs`` (customized settings).
    """
    im = sitk.GetArrayFromImage(inputImage)
    im = im.astype("float64")
    coeff = 1 / np.sqrt(np.max(np.abs(im)))
    im = (coeff * im) ** 2
    im = sitk.GetImageFromArray(im)
    im.CopyInformation(inputImage)

    logger.debug("Yielding square image")
    yield im, "square", kwargs


def getSquareRootImage(inputImage, _inputMask, **kwargs):
    r"""
  Computes the square root of the absolute value of image intensities.

  Resulting values are rescaled on the range of the initial original image and negative intensities are made
  negative in resultant filtered image.

  :math:`f(x) = \left\{ {\begin{array}{lcl}
  \sqrt{cx} & \mbox{for} & x \ge 0 \\
  -\sqrt{-cx} & \mbox{for} & x < 0\end{array}} \right.,\text{ where } c=\max(|x|)`

  Where :math:`x` and :math:`f(x)` are the original and filtered intensity, respectively.

  :return: Yields square root filtered image, 'squareroot' and ``kwargs`` (customized settings).
  """
    im = sitk.GetArrayFromImage(inputImage)
    im = im.astype("float64")
    coeff = np.max(np.abs(im))
    im[im > 0] = np.sqrt(im[im > 0] * coeff)
    im[im < 0] = -np.sqrt(-im[im < 0] * coeff)
    im = sitk.GetImageFromArray(im)
    im.CopyInformation(inputImage)

    logger.debug("Yielding squareroot image")
    yield im, "squareroot", kwargs


def getLogarithmImage(inputImage, _inputMask, **kwargs):
    r"""
  Computes the logarithm of the absolute value of the original image + 1.

  Resulting values are rescaled on the range of the initial original image and negative intensities are made
  negative in resultant filtered image.

  :math:`f(x) = \left\{ {\begin{array}{lcl}
  c\log{(x + 1)} & \mbox{for} & x \ge 0 \\
  -c\log{(-x + 1)} & \mbox{for} & x < 0\end{array}} \right. \text{, where } c=\frac{\max(|x|)}{\log(\max(|x|) + 1)}`

  Where :math:`x` and :math:`f(x)` are the original and filtered intensity, respectively.

  :return: Yields logarithm filtered image, 'logarithm' and ``kwargs`` (customized settings)
  """
    im = sitk.GetArrayFromImage(inputImage)
    im = im.astype("float64")
    im_max = np.max(np.abs(im))
    im[im > 0] = np.log(im[im > 0] + 1)
    im[im < 0] = -np.log(-(im[im < 0] - 1))
    im = im * (im_max / np.max(np.abs(im)))
    im = sitk.GetImageFromArray(im)
    im.CopyInformation(inputImage)

    logger.debug("Yielding logarithm image")
    yield im, "logarithm", kwargs


def getExponentialImage(inputImage, _inputMask, **kwargs):
    r"""
    Computes the exponential of the original image.

    Resulting values are rescaled on the range of the initial original image.

    :math:`f(x) = e^{cx},\text{ where } c=\displaystyle\frac{\log(\max(|x|))}{\max(|x|)}`

    Where :math:`x` and :math:`f(x)` are the original and filtered intensity, respectively.

    :return: Yields exponential filtered image, 'exponential' and ``kwargs`` (customized settings)
    """
    im = sitk.GetArrayFromImage(inputImage)
    im = im.astype("float64")
    im_max = np.max(np.abs(im))
    coeff = np.log(im_max) / im_max
    im = np.exp(coeff * im)
    im = sitk.GetImageFromArray(im)
    im.CopyInformation(inputImage)

    logger.debug("Yielding exponential image")
    yield im, "exponential", kwargs


def getGradientImage(inputImage, _inputMask, **kwargs):
    r"""
    Compute and return the Gradient Magnitude in the image.
    By default, takes into account the image spacing, this can be switched off by specifying
    ``gradientUseSpacing = False``.

    References:

    - `SimpleITK documentation
      <https://itk.org/SimpleITKDoxygen/html/classitk_1_1simple_1_1GradientMagnitudeImageFilter.html>`_
    - `<https://en.wikipedia.org/wiki/Image_gradient>`_
    """
    gmif = sitk.GradientMagnitudeImageFilter()
    gmif.SetUseImageSpacing(kwargs.get("gradientUseSpacing", True))
    im = gmif.Execute(inputImage)
    yield im, "gradient", kwargs


def getLBP2DImage(inputImage, _inputMask, **kwargs):
    """
    Compute and return the Local Binary Pattern (LBP) in 2D. If ``force2D`` is set to false (= feature extraction in 3D) a
    warning is logged, as this filter processes the image in a by-slice operation. The plane in which the LBP is
    applied can be controlled by the ``force2Ddimension`` parameter (see also :py:func:`generateAngles`).

    Following settings are possible (in addition to ``force2Ddimension``):

      - ``lbp2DRadius`` [1]: Float, specifies the radius in which the neighbours should be sampled
      - ``lbp2DSamples`` [9]: Integer, specifies the number of samples to use
      - ``lbp2DMethod`` ['uniform']: String, specifies the method for computing the LBP to use.

    For more information see `scikit documentation
    <http://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.local_binary_pattern>`_

    :return: Yields LBP filtered image, 'lbp-2D' and ``kwargs`` (customized settings)

    .. note::
      LBP can often return only a very small number of different gray levels. A customized bin width is often needed.
    .. warning::
      Requires package ``scikit-image`` to function. If not available, this filter logs a warning and does not yield an image.

    References:

    - T. Ojala, M. Pietikainen, and D. Harwood (1994), "Performance evaluation of texture measures with classification
      based on Kullback discrimination of distributions", Proceedings of the 12th IAPR International Conference on Pattern
      Recognition (ICPR 1994), vol. 1, pp. 582 - 585.
    - T. Ojala, M. Pietikainen, and D. Harwood (1996), "A Comparative Study of Texture Measures with Classification Based
      on Feature Distributions", Pattern Recognition, vol. 29, pp. 51-59.
    """
    try:
        from skimage.feature import local_binary_pattern
    except ImportError:
        logger.warning(
            'Could not load required package "skimage", cannot implement filter LBP 2D'
        )
        return

    lbp_radius = kwargs.get("lbp2DRadius", 1)
    lbp_samples = kwargs.get("lbp2DSamples", 8)
    lbp_method = kwargs.get("lbp2DMethod", "uniform")

    im_arr = sitk.GetArrayFromImage(inputImage)

    Nd = inputImage.GetDimension()
    if Nd == 3:
        if not kwargs.get("force2D", False):
            logger.warning(
                "Calculating Local Binary Pattern in 2D, but extracting features in 3D. Use with caution!"
            )
        lbp_axis = kwargs.get("force2Ddimension", 0)

        im_arr = im_arr.swapaxes(0, lbp_axis)
        for idx in range(im_arr.shape[0]):
            im_arr[idx, ...] = local_binary_pattern(
                im_arr[idx, ...], P=lbp_samples, R=lbp_radius, method=lbp_method
            )
        im_arr = im_arr.swapaxes(0, lbp_axis)
    elif Nd == 2:
        im_arr = local_binary_pattern(
            im_arr, P=lbp_samples, R=lbp_radius, method=lbp_method
        )
    else:
        logger.warning(
            "LBP 2D is only available for 2D or 3D with forced 2D extraction"
        )
        return

    im = sitk.GetImageFromArray(im_arr)
    im.CopyInformation(inputImage)

    yield im, "lbp-2D", kwargs


def getLBP3DImage(inputImage, inputMask, **kwargs):
    """
    Compute and return the Local Binary Pattern (LBP) in 3D using spherical harmonics.
    If ``force2D`` is set to true (= feature extraction in 2D) a warning is logged.

    LBP is only calculated for voxels segmented in the mask

    Following settings are possible:

      - ``lbp3DLevels`` [2]: integer, specifies the the number of levels in spherical harmonics to use.
      - ``lbp3DIcosphereRadius`` [1]: Float, specifies the radius in which the neighbours should be sampled
      - ``lbp3DIcosphereSubdivision`` [1]: Integer, specifies the number of subdivisions to apply in the icosphere

    :return: Yields LBP filtered image for each level, 'lbp-3D-m<level>' and ``kwargs`` (customized settings).
             Additionally yields the kurtosis image, 'lbp-3D-k' and ``kwargs``.

    .. note::
      LBP can often return only a very small number of different gray levels. A customized bin width is often needed.
    .. warning::
      Requires package ``scipy`` and ``trimesh`` to function. If not available, this filter logs a warning and does not
      yield an image.

    References:

    - Banerjee, J, Moelker, A, Niessen, W.J, & van Walsum, T.W. (2013), "3D LBP-based rotationally invariant region
      description." In: Park JI., Kim J. (eds) Computer Vision - ACCV 2012 Workshops. ACCV 2012. Lecture Notes in Computer
      Science, vol 7728. Springer, Berlin, Heidelberg. doi:10.1007/978-3-642-37410-4_3
    """
    Nd = inputImage.GetDimension()
    if Nd != 3:
        msg = f"LBP 3D only available for 3 dimensional images, found {Nd} dimensions"
        logger.warning(msg)
        return

    try:
        from scipy.ndimage.interpolation import map_coordinates
        from scipy.special import sph_harm
        from scipy.stats import kurtosis
        from trimesh.creation import icosphere
    except ImportError:
        msg = 'Could not load required package "scipy" or "trimesh", cannot implement filter LBP 3D'
        logger.warning(msg)
        return

    if kwargs.get("force2D", False):
        logger.warning(
            "Calculating Local Binary Pattern in 3D, but extracting features in 2D. Use with caution!"
        )

    label = kwargs.get("label", 1)

    lbp_levels = kwargs.get("lbp3DLevels", 2)
    lbp_icosphereRadius = kwargs.get("lbp3DIcosphereRadius", 1)
    lbp_icosphereSubdivision = kwargs.get("lbp3DIcosphereSubdivision", 1)

    im_arr = sitk.GetArrayFromImage(inputImage)
    ma_arr = sitk.GetArrayFromImage(inputMask)

    # Variables used in the shape comments:
    # Np Number of voxels
    # Nv Number of vertices

    # Vertices icosahedron for spherical sampling
    coords_icosahedron = np.array(
        icosphere(lbp_icosphereSubdivision, lbp_icosphereRadius).vertices
    )  # shape(Nv, 3)

    # Corresponding polar coordinates
    theta = np.arccos(np.true_divide(coords_icosahedron[:, 2], lbp_icosphereRadius))
    phi = np.arctan2(coords_icosahedron[:, 1], coords_icosahedron[:, 0])

    # Corresponding spherical harmonics coefficients Y_{m, n, theta, phi}
    Y = sph_harm(0, 0, theta, phi)  # shape(Nv,)
    n_ix = np.array(0)

    for n in range(1, lbp_levels):
        for m in range(-n, n + 1):
            n_ix = np.append(n_ix, n)
            Y = np.column_stack((Y, sph_harm(m, n, theta, phi)))
    # shape (Nv, x) where x is the number of iterations in the above loops + 1

    ROI_coords = np.where(ma_arr == label)  # shape(3, Np)

    # Interpolate f (samples on the spheres across the entire volume)
    coords = (
        np.array(ROI_coords).T[None, :, :] + coords_icosahedron[:, None, :]
    )  # shape(Nv, Np, 3)
    f = map_coordinates(
        im_arr, coords.T, order=3
    )  # Shape(Np, Nv)  Note that 'Np' and 'Nv' are swapped due to .T

    k = kurtosis(f, axis=1)  # shape(Np,)

    f_centroids = im_arr[ROI_coords]  # Shape(Np,)
    f = np.greater_equal(f, f_centroids[:, None]).astype(int)  # Shape(Np, Nv)

    # Compute c_{m,n} coefficients
    c = np.multiply(f[:, :, None], Y[None, :, :])  # Shape(Np, Nv, x)
    c = c.sum(axis=1)  # Shape(Np, x)

    f = np.multiply(c[:, None, n_ix == 0], Y[None, :, n_ix == 0])  # Shape (Np, Nv, 1)
    for n in range(1, lbp_levels):
        f = np.concatenate(
            (
                f,
                np.sum(
                    np.multiply(c[:, None, n_ix == n], Y[None, :, n_ix == n]),
                    axis=2,
                    keepdims=True,
                ),
            ),
            axis=2,
        )
    # Shape f (Np, Nv, levels)

    # Compute L2-Norm
    f = np.sqrt(np.sum(f**2, axis=1))  # shape(Np, levels)

    f = np.real(f)  # shape(Np, levels)
    k = np.real(k)  # shape(Np,)

    result = np.ndarray(im_arr.shape)
    for l_idx in range(lbp_levels):
        result[ROI_coords] = f[:, l_idx]

        im = sitk.GetImageFromArray(result)
        im.CopyInformation(inputImage)

        yield im, f"lbp-3D-m{int(l_idx + 1)}", kwargs

    result[ROI_coords] = k

    im = sitk.GetImageFromArray(result)
    im.CopyInformation(inputImage)

    yield im, "lbp-3D-k", kwargs
