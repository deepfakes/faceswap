#!/usr/bin python3
""" Handles retrieval and storage of Faceswap aligned masks """

from __future__ import annotations
import logging
import typing as T

from zlib import compress, decompress

import cv2
import numpy as np

from lib.logger import parse_class_init

from .alignments import MaskAlignmentsFileDict
from . import get_adjusted_center, get_centered_size

if T.TYPE_CHECKING:
    from collections.abc import Callable
    from .aligned_face import CenteringType

logger = logging.getLogger(__name__)


class Mask():
    """ Face Mask information and convenience methods

    Holds a Faceswap mask as generated from :mod:`plugins.extract.mask` and the information
    required to transform it to its original frame.

    Holds convenience methods to handle the warping, storing and retrieval of the mask.

    Parameters
    ----------
    storage_size: int, optional
        The size (in pixels) that the mask should be stored at. Default: 128.
    storage_centering, str (optional):
        The centering to store the mask at. One of `"legacy"`, `"face"`, `"head"`.
        Default: `"face"`

    Attributes
    ----------
    stored_size: int
        The size, in pixels, of the stored mask across its height and width.
    stored_centering: str
        The centering that the mask is stored at. One of `"legacy"`, `"face"`, `"head"`
    """
    def __init__(self,
                 storage_size: int = 128,
                 storage_centering: CenteringType = "face") -> None:
        logger.trace(parse_class_init(locals()))  # type:ignore[attr-defined]
        self.stored_size = storage_size
        self.stored_centering = storage_centering

        self._mask: bytes | None = None
        self._affine_matrix: np.ndarray | None = None
        self._interpolator: int | None = None

        self._blur_type: T.Literal["gaussian", "normalized"] | None = None
        self._blur_passes: int = 0
        self._blur_kernel: float | int = 0
        self._threshold = 0.0
        self._dilation: tuple[T.Literal["erode", "dilate"], np.ndarray | None] = ("erode", None)
        self._sub_crop_size = 0
        self._sub_crop_slices: dict[T.Literal["in", "out"], list[slice]] = {}

        self.set_blur_and_threshold()
        logger.trace("Initialized: %s", self.__class__.__name__)  # type:ignore[attr-defined]

    @property
    def mask(self) -> np.ndarray:
        """ :class:`numpy.ndarray`: The mask at the size of :attr:`stored_size` with any requested
        blurring, threshold amount and centering applied."""
        mask = self.stored_mask
        if self._dilation[-1] is not None or self._threshold != 0.0 or self._blur_kernel != 0:
            mask = mask.copy()
        self._dilate_mask(mask)
        if self._threshold != 0.0:
            mask[mask < self._threshold] = 0.0
            mask[mask > 255.0 - self._threshold] = 255.0
        if self._blur_kernel != 0 and self._blur_type is not None:
            mask = BlurMask(self._blur_type,
                            mask,
                            self._blur_kernel,
                            passes=self._blur_passes).blurred
        if self._sub_crop_size:  # Crop the mask to the given centering
            out = np.zeros((self._sub_crop_size, self._sub_crop_size, 1), dtype=mask.dtype)
            slice_in, slice_out = self._sub_crop_slices["in"], self._sub_crop_slices["out"]
            out[slice_out[0], slice_out[1], :] = mask[slice_in[0], slice_in[1], :]
            mask = out
        logger.trace("mask shape: %s", mask.shape)  # type:ignore[attr-defined]
        return mask

    @property
    def stored_mask(self) -> np.ndarray:
        """ :class:`numpy.ndarray`: The mask at the size of :attr:`stored_size` as it is stored
        (i.e. with no blurring/centering applied). """
        assert self._mask is not None
        dims = (self.stored_size, self.stored_size, 1)
        mask = np.frombuffer(decompress(self._mask), dtype="uint8").reshape(dims)
        logger.trace("stored mask shape: %s", mask.shape)  # type:ignore[attr-defined]
        return mask

    @property
    def original_roi(self) -> np.ndarray:
        """ :class: `numpy.ndarray`: The original region of interest of the mask in the
        source frame. """
        points = np.array([[0, 0],
                           [0, self.stored_size - 1],
                           [self.stored_size - 1, self.stored_size - 1],
                           [self.stored_size - 1, 0]], np.int32).reshape((-1, 1, 2))
        matrix = cv2.invertAffineTransform(self.affine_matrix)
        roi = cv2.transform(points, matrix).reshape((4, 2))
        logger.trace("Returning: %s", roi)  # type:ignore[attr-defined]
        return roi

    @property
    def affine_matrix(self) -> np.ndarray:
        """ :class: `numpy.ndarray`: The affine matrix to transpose the mask to a full frame. """
        assert self._affine_matrix is not None
        return self._affine_matrix

    @property
    def interpolator(self) -> int:
        """ int: The cv2 interpolator required to transpose the mask to a full frame. """
        assert self._interpolator is not None
        return self._interpolator

    def _dilate_mask(self, mask: np.ndarray) -> None:
        """ Erode/Dilate the mask. The action is performed in-place on the given mask.

        No action is performed if a dilation amount has not been set

        Parameters
        ----------
        mask: :class:`numpy.ndarray`
            The mask to be eroded/dilated
        """
        if self._dilation[-1] is None:
            return

        func = cv2.erode if self._dilation[0] == "erode" else cv2.dilate
        func(mask, self._dilation[-1], dst=mask, iterations=1)

    def get_full_frame_mask(self, width: int, height: int) -> np.ndarray:
        """ Return the stored mask in a full size frame of the given dimensions

        Parameters
        ----------
        width: int
            The width of the original frame that the mask was extracted from
        height: int
            The height of the original frame that the mask was extracted from

        Returns
        -------
        :class:`numpy.ndarray`: The mask affined to the original full frame of the given dimensions
        """
        frame = np.zeros((width, height, 1), dtype="uint8")
        mask = cv2.warpAffine(self.mask,
                              self.affine_matrix,
                              (width, height),
                              frame,
                              flags=cv2.WARP_INVERSE_MAP | self.interpolator,
                              borderMode=cv2.BORDER_CONSTANT)
        logger.trace("mask shape: %s, mask dtype: %s, mask min: %s, "  # type:ignore[attr-defined]
                     "mask max: %s", mask.shape, mask.dtype, mask.min(), mask.max())
        return mask

    def add(self, mask: np.ndarray, affine_matrix: np.ndarray, interpolator: int) -> None:
        """ Add a Faceswap mask to this :class:`Mask`.

        The mask should be the original output from  :mod:`plugins.extract.mask`

        Parameters
        ----------
        mask: :class:`numpy.ndarray`
            The mask that is to be added as output from :mod:`plugins.extract.mask`
            It should be in the range 0.0 - 1.0 ideally with a ``dtype`` of ``float32``
        affine_matrix: :class:`numpy.ndarray`
            The transformation matrix required to transform the mask to the original frame.
        interpolator, int:
            The CV2 interpolator required to transform this mask to it's original frame
        """
        logger.trace("mask shape: %s, mask dtype: %s, mask min: %s, "  # type:ignore[attr-defined]
                     "mask max: %s, affine_matrix: %s, interpolator: %s)",
                     mask.shape, mask.dtype, mask.min(), affine_matrix, mask.max(), interpolator)
        self._affine_matrix = self._adjust_affine_matrix(mask.shape[0], affine_matrix)
        self._interpolator = interpolator
        self.replace_mask(mask)

    def replace_mask(self, mask: np.ndarray) -> None:
        """ Replace the existing :attr:`_mask` with the given mask.

        Parameters
        ----------
        mask: :class:`numpy.ndarray`
            The mask that is to be added as output from :mod:`plugins.extract.mask`.
            It should be in the range 0.0 - 1.0 ideally with a ``dtype`` of ``float32``
        """
        mask = (cv2.resize(mask * 255.0,
                           (self.stored_size, self.stored_size),
                           interpolation=cv2.INTER_AREA)).astype("uint8")
        self._mask = compress(mask.tobytes())

    def set_dilation(self, amount: float) -> None:
        """ Set the internal dilation object for returned masks

        Parameters
        ----------
        amount: float
            The amount of erosion/dilation to apply as a percentage of the total mask size.
            Negative values erode the mask. Positive values dilate the mask
        """
        if amount == 0:
            self._dilation = ("erode", None)
            return

        action: T.Literal["erode", "dilate"] = "erode" if amount < 0 else "dilate"
        kernel = int(round(self.stored_size * abs(amount / 100.), 0))
        self._dilation = (action, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel)))

        logger.trace("action: '%s', amount: %s, kernel: %s, ",  # type:ignore[attr-defined]
                     action, amount, kernel)

    def set_blur_and_threshold(self,
                               blur_kernel: int = 0,
                               blur_type: T.Literal["gaussian", "normalized"] | None = "gaussian",
                               blur_passes: int = 1,
                               threshold: int = 0) -> None:
        """ Set the internal blur kernel and threshold amount for returned masks

        Parameters
        ----------
        blur_kernel: int, optional
            The kernel size, in pixels to apply gaussian blurring to the mask. Set to 0 for no
            blurring. Should be odd, if an even number is passed in (outside of 0) then it is
            rounded up to the next odd number. Default: 0
        blur_type: ["gaussian", "normalized"], optional
            The blur type to use. ``gaussian`` or ``normalized`` box filter. Default: ``gaussian``
        blur_passes: int, optional
            The number of passed to perform when blurring. Default: 1
        threshold: int, optional
            The threshold amount to minimize/maximize mask values to 0 and 100. Percentage value.
            Default: 0
        """
        logger.trace("blur_kernel: %s, blur_type: %s, "  # type:ignore[attr-defined]
                     "blur_passes: %s, threshold: %s",
                     blur_kernel, blur_type, blur_passes, threshold)
        if blur_type is not None:
            blur_kernel += 0 if blur_kernel == 0 or blur_kernel % 2 == 1 else 1
            self._blur_kernel = blur_kernel
            self._blur_type = blur_type
            self._blur_passes = blur_passes
        self._threshold = (threshold / 100.0) * 255.0

    def set_sub_crop(self,
                     source_offset: np.ndarray,
                     target_offset: np.ndarray,
                     centering: CenteringType,
                     coverage_ratio: float = 1.0) -> None:
        """ Set the internal crop area of the mask to be returned.

        This impacts the returned mask from :attr:`mask` if the requested mask is required for
        different face centering than what has been stored.

        Parameters
        ----------
        source_offset: :class:`numpy.ndarray`
            The (x, y) offset for the mask at its stored centering
        target_offset: :class:`numpy.ndarray`
            The (x, y) offset for the mask at the requested target centering
        centering: str
            The centering to set the sub crop area for. One of `"legacy"`, `"face"`. `"head"`
        coverage_ratio: float, optional
            The coverage ratio to be applied to the target image. ``None`` for default (1.0).
            Default: ``None``
        """
        if centering == self.stored_centering and coverage_ratio == 1.0:
            return

        center = get_adjusted_center(self.stored_size,
                                     source_offset,
                                     target_offset,
                                     self.stored_centering)
        crop_size = get_centered_size(self.stored_centering,
                                      centering,
                                      self.stored_size,
                                      coverage_ratio=coverage_ratio)
        roi = np.array([center - crop_size // 2, center + crop_size // 2]).ravel()

        self._sub_crop_size = crop_size
        self._sub_crop_slices["in"] = [slice(max(roi[1], 0), max(roi[3], 0)),
                                       slice(max(roi[0], 0), max(roi[2], 0))]
        self._sub_crop_slices["out"] = [
            slice(max(roi[1] * -1, 0),
                  crop_size - min(crop_size, max(0, roi[3] - self.stored_size))),
            slice(max(roi[0] * -1, 0),
                  crop_size - min(crop_size, max(0, roi[2] - self.stored_size)))]

        logger.trace("src_size: %s, coverage_ratio: %s, "  # type:ignore[attr-defined]
                     "sub_crop_size: %s, sub_crop_slices: %s",
                     roi, coverage_ratio, self._sub_crop_size, self._sub_crop_slices)

    def _adjust_affine_matrix(self, mask_size: int, affine_matrix: np.ndarray) -> np.ndarray:
        """ Adjust the affine matrix for the mask's storage size

        Parameters
        ----------
        mask_size: int
            The original size of the mask.
        affine_matrix: :class:`numpy.ndarray`
            The affine matrix to transform the mask at original size to the parent frame.

        Returns
        -------
        affine_matrix: :class:`numpy,ndarray`
            The affine matrix adjusted for the mask at its stored dimensions.
        """
        zoom = self.stored_size / mask_size
        zoom_mat = np.array([[zoom, 0, 0.], [0, zoom, 0.]])
        adjust_mat = np.dot(zoom_mat, np.concatenate((affine_matrix, np.array([[0., 0., 1.]]))))
        logger.trace("storage_size: %s, mask_size: %s, zoom: %s, "  # type:ignore[attr-defined]
                     "original matrix: %s, adjusted_matrix: %s", self.stored_size, mask_size, zoom,
                     affine_matrix.shape, adjust_mat.shape)
        return adjust_mat

    def to_dict(self, is_png=False) -> MaskAlignmentsFileDict:
        """ Convert the mask to a dictionary for saving to an alignments file

        Parameters
        ----------
        is_png: bool
            ``True`` if the dictionary is being created for storage in a png header otherwise
            ``False``. Default: ``False``

        Returns
        -------
        dict:
            The :class:`Mask` for saving to an alignments file. Contains the keys ``mask``,
            ``affine_matrix``, ``interpolator``, ``stored_size``, ``stored_centering``
        """
        assert self._mask is not None
        affine_matrix = self.affine_matrix.tolist() if is_png else self.affine_matrix
        retval = MaskAlignmentsFileDict(mask=self._mask,
                                        affine_matrix=affine_matrix,
                                        interpolator=self.interpolator,
                                        stored_size=self.stored_size,
                                        stored_centering=self.stored_centering)
        logger.trace({k: v if k != "mask" else type(v)  # type:ignore[attr-defined]
                      for k, v in retval.items()})
        return retval

    def to_png_meta(self) -> MaskAlignmentsFileDict:
        """ Convert the mask to a dictionary supported by png itxt headers.

        Returns
        -------
        dict:
            The :class:`Mask` for saving to an alignments file. Contains the keys ``mask``,
            ``affine_matrix``, ``interpolator``, ``stored_size``, ``stored_centering``
        """
        return self.to_dict(is_png=True)

    def from_dict(self, mask_dict: MaskAlignmentsFileDict) -> None:
        """ Populates the :class:`Mask` from a dictionary loaded from an alignments file.

        Parameters
        ----------
        mask_dict: dict
            A dictionary stored in an alignments file containing the keys ``mask``,
            ``affine_matrix``, ``interpolator``, ``stored_size``, ``stored_centering``
        """
        self._mask = mask_dict["mask"]
        affine_matrix = mask_dict["affine_matrix"]
        self._affine_matrix = (affine_matrix if isinstance(affine_matrix, np.ndarray)
                               else np.array(affine_matrix, dtype="float64"))
        self._interpolator = mask_dict["interpolator"]
        self.stored_size = mask_dict["stored_size"]
        centering = mask_dict.get("stored_centering")
        self.stored_centering = "face" if centering is None else centering
        logger.trace({k: v if k != "mask" else type(v)  # type:ignore[attr-defined]
                      for k, v in mask_dict.items()})


class LandmarksMask(Mask):
    """ Create a single channel mask from aligned landmark points.

    Landmarks masks are created on the fly, so the stored centering and size should be the same as
    the aligned face that the mask will be applied to. As the masks are created on the fly, blur +
    dilation is applied to the mask at creation (prior to compression) rather than after
    decompression when requested.

    Note
    ----
    Threshold is not used for Landmarks mask as the mask is binary

    Parameters
    ----------
    points: list
        A list of landmark points that correspond to the given storage_size to create
        the mask. Each item in the list should be a :class:`numpy.ndarray` that a filled
        convex polygon will be created from
    storage_size: int, optional
        The size (in pixels) that the compressed mask should be stored at. Default: 128.
    storage_centering, str (optional):
        The centering to store the mask at. One of `"legacy"`, `"face"`, `"head"`.
        Default: `"face"`
    dilation: float, optional
        The amount of dilation to apply to the mask. as a percentage of the mask size. Default: 0.0
    """
    def __init__(self,
                 points: list[np.ndarray],
                 storage_size: int = 128,
                 storage_centering: CenteringType = "face",
                 dilation: float = 0.0) -> None:
        super().__init__(storage_size=storage_size, storage_centering=storage_centering)
        self._points = points
        self.set_dilation(dilation)

    @property
    def mask(self) -> np.ndarray:
        """ :class:`numpy.ndarray`: Overrides the default mask property, creating the processed
        mask at first call and compressing it. The decompressed mask is returned from this
        property. """
        return self.stored_mask

    def generate_mask(self, affine_matrix: np.ndarray, interpolator: int) -> None:
        """ Generate the mask.

        Creates the mask applying any requested dilation and blurring and assigns compressed mask
        to :attr:`_mask`

        Parameters
        ----------
        affine_matrix: :class:`numpy.ndarray`
            The transformation matrix required to transform the mask to the original frame.
        interpolator, int:
            The CV2 interpolator required to transform this mask to it's original frame
        """
        mask = np.zeros((self.stored_size, self.stored_size, 1), dtype="float32")
        for landmarks in self._points:
            lms = np.rint(landmarks).astype("int")
            cv2.fillConvexPoly(mask, cv2.convexHull(lms), [1.0], lineType=cv2.LINE_AA)
        if self._dilation[-1] is not None:
            self._dilate_mask(mask)
        if self._blur_kernel != 0 and self._blur_type is not None:
            mask = BlurMask(self._blur_type,
                            mask,
                            self._blur_kernel,
                            passes=self._blur_passes).blurred
        logger.trace("mask: (shape: %s, dtype: %s)",  # type:ignore[attr-defined]
                     mask.shape, mask.dtype)
        self.add(mask, affine_matrix, interpolator)


class BlurMask():
    """ Factory class to return the correct blur object for requested blur type.

    Works for square images only. Currently supports Gaussian and Normalized Box Filters.

    Parameters
    ----------
    blur_type: ["gaussian", "normalized"]
        The type of blur to use
    mask: :class:`numpy.ndarray`
        The mask to apply the blur to
    kernel: int or float
        Either the kernel size (in pixels) or the size of the kernel as a ratio of mask size
    is_ratio: bool, optional
        Whether the given :attr:`kernel` parameter is a ratio or not. If ``True`` then the
        actual kernel size will be calculated from the given ratio and the mask size. If
        ``False`` then the kernel size will be set directly from the :attr:`kernel` parameter.
        Default: ``False``
    passes: int, optional
        The number of passes to perform when blurring. Default: ``1``

    Example
    -------
    >>> print(mask.shape)
    (128, 128, 1)
    >>> new_mask = BlurMask("gaussian", mask, 3, is_ratio=False, passes=1).blurred
    >>> print(new_mask.shape)
    (128, 128, 1)
    """
    def __init__(self,
                 blur_type: T.Literal["gaussian", "normalized"],
                 mask: np.ndarray,
                 kernel: int | float,
                 is_ratio: bool = False,
                 passes: int = 1) -> None:
        logger.trace(parse_class_init(locals()))  # type:ignore[attr-defined]
        self._blur_type = blur_type
        self._mask = mask
        self._passes = passes
        kernel_size = self._get_kernel_size(kernel, is_ratio)
        self._kernel_size = self._get_kernel_tuple(kernel_size)
        logger.trace("Initialized %s", self.__class__.__name__)  # type:ignore[attr-defined]

    @property
    def blurred(self) -> np.ndarray:
        """ :class:`numpy.ndarray`: The final mask with blurring applied. """
        func = self._func_mapping[self._blur_type]
        kwargs = self._get_kwargs()
        blurred = self._mask
        for i in range(self._passes):
            assert isinstance(kwargs["ksize"], tuple)
            ksize = int(kwargs["ksize"][0])
            logger.trace("Pass: %s, kernel_size: %s",  # type:ignore[attr-defined]
                         i + 1, (ksize, ksize))
            blurred = func(blurred, **kwargs)
            ksize = int(round(ksize * self._multipass_factor))
            kwargs["ksize"] = self._get_kernel_tuple(ksize)
        blurred = blurred[..., None]
        logger.trace("Returning blurred mask. Shape: %s",  # type:ignore[attr-defined]
                     blurred.shape)
        return blurred

    @property
    def _multipass_factor(self) -> float:
        """ For multiple passes the kernel must be scaled down. This value is
            different for box filter and gaussian """
        factor = {"gaussian": 0.8, "normalized": 0.5}
        return factor[self._blur_type]

    @property
    def _sigma(self) -> T.Literal[0]:
        """ int: The Sigma for Gaussian Blur. Returns 0 to force calculation from kernel size. """
        return 0

    @property
    def _func_mapping(self) -> dict[T.Literal["gaussian", "normalized"], Callable]:
        """ dict: :attr:`_blur_type` mapped to cv2 Function name. """
        return {"gaussian": cv2.GaussianBlur, "normalized": cv2.blur}

    @property
    def _kwarg_requirements(self) -> dict[T.Literal["gaussian", "normalized"], list[str]]:
        """ dict: :attr:`_blur_type` mapped to cv2 Function required keyword arguments. """
        return {"gaussian": ['ksize', 'sigmaX'], "normalized": ['ksize']}

    @property
    def _kwarg_mapping(self) -> dict[str, int | tuple[int, int]]:
        """ dict: cv2 function keyword arguments mapped to their parameters. """
        return {"ksize": self._kernel_size, "sigmaX": self._sigma}

    def _get_kernel_size(self, kernel: int | float, is_ratio: bool) -> int:
        """ Set the kernel size to absolute value.

        If :attr:`is_ratio` is ``True`` then the kernel size is calculated from the given ratio and
        the :attr:`_mask` size, otherwise the given kernel size is just returned.

        Parameters
        ----------
        kernel: int or float
            Either the kernel size (in pixels) or the size of the kernel as a ratio of mask size
        is_ratio: bool, optional
            Whether the given :attr:`kernel` parameter is a ratio or not. If ``True`` then the
            actual kernel size will be calculated from the given ratio and the mask size. If
            ``False`` then the kernel size will be set directly from the :attr:`kernel` parameter.

        Returns
        -------
        int
            The size (in pixels) of the blur kernel
        """
        if not is_ratio:
            return int(kernel)

        mask_diameter = np.sqrt(np.sum(self._mask))
        radius = round(max(1., mask_diameter * kernel / 100.))
        kernel_size = int(radius * 2 + 1)
        logger.trace("kernel_size: %s", kernel_size)  # type:ignore[attr-defined]
        return kernel_size

    @staticmethod
    def _get_kernel_tuple(kernel_size: int) -> tuple[int, int]:
        """ Make sure kernel_size is odd and return it as a tuple.

        Parameters
        ----------
        kernel_size: int
            The size in pixels of the blur kernel

        Returns
        -------
        tuple
            The kernel size as a tuple of ('int', 'int')
        """
        kernel_size += 1 if kernel_size % 2 == 0 else 0
        retval = (kernel_size, kernel_size)
        logger.trace(retval)  # type:ignore[attr-defined]
        return retval

    def _get_kwargs(self) -> dict[str, int | tuple[int, int]]:
        """ dict: the valid keyword arguments for the requested :attr:`_blur_type` """
        retval = {kword: self._kwarg_mapping[kword]
                  for kword in self._kwarg_requirements[self._blur_type]}
        logger.trace("BlurMask kwargs: %s", retval)  # type:ignore[attr-defined]
        return retval
