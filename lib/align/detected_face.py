#!/usr/bin python3
""" Face and landmarks detection for faceswap.py """

import logging
import sys
import os

from hashlib import sha1
from typing import Dict, List, TYPE_CHECKING

from zlib import compress, decompress

import cv2
import numpy as np

from lib.image import encode_image, read_image
from lib.utils import FaceswapError
from . import AlignedFace, get_adjusted_center, get_centered_size

if TYPE_CHECKING:
    from .aligned_face import CenteringType

if sys.version_info < (3, 8):
    from typing_extensions import Literal
else:
    from typing import Literal

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class DetectedFace():
    """ Detected face and landmark information

    Holds information about a detected face, it's location in a source image
    and the face's 68 point landmarks.

    Methods for aligning a face are also callable from here.

    Parameters
    ----------
    image: numpy.ndarray, optional
        Original frame that holds this face. Optional (not required if just storing coordinates)
    x: int
        The left most point (in pixels) of the face's bounding box as discovered in
        :mod:`plugins.extract.detect`
    w: int
        The width (in pixels) of the face's bounding box as discovered in
        :mod:`plugins.extract.detect`
    y: int
        The top most point (in pixels) of the face's bounding box as discovered in
        :mod:`plugins.extract.detect`
    h: int
        The height (in pixels) of the face's bounding box as discovered in
        :mod:`plugins.extract.detect`
    landmarks_xy: list
        The 68 point landmarks as discovered in :mod:`plugins.extract.align`. Should be a ``list``
        of 68 `(x, y)` ``tuples`` with each of the landmark co-ordinates.
    mask: dict
        The generated mask(s) for the face as generated in :mod:`plugins.extract.mask`. Must be a
        dict of {**name** (`str`): :class:`Mask`}.

    Attributes
    ----------
    image: numpy.ndarray, optional
        This is a generic image placeholder that should not be relied on to be holding a particular
        image. It may hold the source frame that holds the face, a cropped face or a scaled image
        depending on the method using this object.
    x: int
        The left most point (in pixels) of the face's bounding box as discovered in
        :mod:`plugins.extract.detect`
    w: int
        The width (in pixels) of the face's bounding box as discovered in
        :mod:`plugins.extract.detect`
    y: int
        The top most point (in pixels) of the face's bounding box as discovered in
        :mod:`plugins.extract.detect`
    h: int
        The height (in pixels) of the face's bounding box as discovered in
        :mod:`plugins.extract.detect`
    landmarks_xy: list
        The 68 point landmarks as discovered in :mod:`plugins.extract.align`.
    mask: dict
        The generated mask(s) for the face as generated in :mod:`plugins.extract.mask`. Is a
        dict of {**name** (`str`): :class:`Mask`}.
    """
    def __init__(self, image=None, x=None, w=None, y=None, h=None, landmarks_xy=None, mask=None,
                 filename=None):
        logger.trace("Initializing %s: (image: %s, x: %s, w: %s, y: %s, h:%s, landmarks_xy: %s, "
                     "mask: %s, filename: %s)",
                     self.__class__.__name__,
                     image.shape if image is not None and image.any() else image,
                     x, w, y, h, landmarks_xy,
                     {k: v.shape for k, v in mask} if mask is not None else mask,
                     filename)
        self.image = image
        self.x = x  # pylint:disable=invalid-name
        self.w = w  # pylint:disable=invalid-name
        self.y = y  # pylint:disable=invalid-name
        self.h = h  # pylint:disable=invalid-name
        self.landmarks_xy = landmarks_xy
        self.thumbnail = None
        self.mask = dict() if mask is None else mask

        self.aligned = None
        logger.trace("Initialized %s", self.__class__.__name__)

    @property
    def left(self):
        """int: Left point (in pixels) of face detection bounding box within the parent image """
        return self.x

    @property
    def top(self):
        """int: Top point (in pixels) of face detection bounding box within the parent image """
        return self.y

    @property
    def right(self):
        """int: Right point (in pixels) of face detection bounding box within the parent image """
        return self.x + self.w

    @property
    def bottom(self):
        """int: Bottom point (in pixels) of face detection bounding box within the parent image """
        return self.y + self.h

    def add_mask(self, name, mask, affine_matrix, interpolator,
                 storage_size=128, storage_centering="face"):
        """ Add a :class:`Mask` to this detected face

        The mask should be the original output from  :mod:`plugins.extract.mask`
        If a mask with this name already exists it will be overwritten by the given
        mask.

        Parameters
        ----------
        name: str
            The name of the mask as defined by the :attr:`plugins.extract.mask._base.name`
            parameter.
        mask: numpy.ndarray
            The mask that is to be added as output from :mod:`plugins.extract.mask`
            It should be in the range 0.0 - 1.0 ideally with a ``dtype`` of ``float32``
        affine_matrix: numpy.ndarray
            The transformation matrix required to transform the mask to the original frame.
        interpolator, int:
            The CV2 interpolator required to transform this mask to it's original frame.
        storage_size, int (optional):
            The size the mask is to be stored at. Default: 128
        storage_centering, str (optional):
            The centering to store the mask at. One of `"legacy"`, `"face"`, `"head"`.
            Default: `"face"`
        """
        logger.trace("name: '%s', mask shape: %s, affine_matrix: %s, interpolator: %s, "
                     "storage_size: %s, storage_centering: %s)", name, mask.shape, affine_matrix,
                     interpolator, storage_size, storage_centering)
        fsmask = Mask(storage_size=storage_size, storage_centering=storage_centering)
        fsmask.add(mask, affine_matrix, interpolator)
        self.mask[name] = fsmask

    def get_landmark_mask(self, size, area,
                          aligned=True, centering="face", dilation=0, blur_kernel=0, as_zip=False):
        """ Obtain a single channel mask based on the face's landmark points.

        Parameters
        ----------
        size: int or tuple
            The size of the aligned mask to retrieve. Should be an `int` if an aligned face is
            being requested, or a ('height', 'width') shape tuple if a full frame is being
            requested
        area: ["mouth", "eyes"]
            The type of mask to obtain. `face` is a full face mask the others are masks for those
            specific areas
        aligned: bool, optional
            ``True`` if the returned mask should be for an aligned face. ``False`` if a full frame
            mask should be returned. Default ``True``
        centering: ["legacy", "face", "head"], optional
            Only used if `aligned`=``True``. The centering for the landmarks based mask. Should be
            the same as the centering used for the extracted face that this mask will be applied
            to. "legacy" places the nose in the center of the image (the original method for
            aligning). "face" aligns for the nose to be in the center of the face (top to bottom)
            but the center of the skull for left to right. "head" aligns for the center of the
            skull (in 3D space) being the center of the extracted image, with the crop holding the
            full head. Default: `"face"`
        dilation: int, optional
            The amount of dilation to apply to the mask. `0` for none. Default: `0`
        blur_kernel: int, optional
            The kernel size for applying gaussian blur to apply to the mask. `0` for none.
            Default: `0`
        as_zip: bool, optional
            ``True`` if the mask should be returned zipped otherwise ``False``

        Returns
        -------
        :class:`numpy.ndarray` or zipped array
            The mask as a single channel image of the given :attr:`size` dimension. If
            :attr:`as_zip` is ``True`` then the :class:`numpy.ndarray` will be contained within a
            zipped container
        """
        # TODO Face mask generation from landmarks
        logger.trace("size: %s, area: %s, aligned: %s, dilation: %s, blur_kernel: %s, as_zip: %s",
                     size, area, aligned, dilation, blur_kernel, as_zip)
        areas = dict(mouth=[slice(48, 60)], eyes=[slice(36, 42), slice(42, 48)])
        if aligned:
            face = AlignedFace(self.landmarks_xy, centering=centering, size=size)
            landmarks = face.landmarks
            size = (size, size)
        else:
            landmarks = self.landmarks_xy
        points = [landmarks[zone] for zone in areas[area]]  # pylint:disable=unsubscriptable-object
        mask = _LandmarksMask(size, points, dilation=dilation, blur_kernel=blur_kernel)
        retval = mask.get(as_zip=as_zip)
        return retval

    def to_alignment(self):
        """  Return the detected face formatted for an alignments file

        returns
        -------
        alignment: dict
            The alignment dict will be returned with the keys ``x``, ``w``, ``y``, ``h``,
            ``landmarks_xy``, ``mask``. The additional key ``thumb`` will be provided if the
            detected face object contains a thumbnail.
        """
        alignment = dict(x=self.x,
                         w=self.w,
                         y=self.y,
                         h=self.h,
                         landmarks_xy=self.landmarks_xy,
                         mask={name: mask.to_dict() for name, mask in self.mask.items()})
        if self.thumbnail is not None:
            alignment["thumb"] = self.thumbnail
        logger.trace("Returning: %s", alignment)
        return alignment

    def from_alignment(self, alignment, image=None, with_thumb=False):
        """ Set the attributes of this class from an alignments file and optionally load the face
        into the ``image`` attribute.

        Parameters
        ----------
        alignment: dict
            A dictionary entry for a face from an alignments file containing the keys
            ``x``, ``w``, ``y``, ``h``, ``landmarks_xy``.
            Optionally the key ``thumb`` will be provided. This is for use in the manual tool and
            contains the compressed jpg thumbnail of the face to be allocated to :attr:`thumbnail.
            Optionally the key ``mask`` will be provided, but legacy alignments will not have
            this key.
        image: numpy.ndarray, optional
            If an image is passed in, then the ``image`` attribute will
            be set to the cropped face based on the passed in bounding box co-ordinates
        with_thumb: bool, optional
            Whether to load the jpg thumbnail into the detected face object, if provided.
            Default: ``False``
        """

        logger.trace("Creating from alignment: (alignment: %s, has_image: %s)",
                     alignment, bool(image is not None))
        self.x = alignment["x"]
        self.w = alignment["w"]
        self.y = alignment["y"]
        self.h = alignment["h"]
        landmarks = alignment["landmarks_xy"]
        if not isinstance(landmarks, np.ndarray):
            landmarks = np.array(landmarks, dtype="float32")
        self.landmarks_xy = landmarks.copy()

        if with_thumb:
            # Thumbnails currently only used for manual tool. Default to None
            self.thumbnail = alignment.get("thumb", None)
        # Manual tool and legacy alignments will not have a mask
        self.aligned = None

        if alignment.get("mask", None) is not None:
            self.mask = dict()
            for name, mask_dict in alignment["mask"].items():
                self.mask[name] = Mask()
                self.mask[name].from_dict(mask_dict)
        if image is not None and image.any():
            self._image_to_face(image)
        logger.trace("Created from alignment: (x: %s, w: %s, y: %s. h: %s, "
                     "landmarks: %s, mask: %s)",
                     self.x, self.w, self.y, self.h, self.landmarks_xy, self.mask)

    def to_png_meta(self):
        """ Return the detected face formatted for insertion into a png itxt header.

        returns: dict
            The alignments dict will be returned with the keys ``x``, ``w``, ``y``, ``h``,
            ``landmarks_xy`` and ``mask``
        """
        alignment = dict(x=self.x,
                         w=self.w,
                         y=self.y,
                         h=self.h,
                         landmarks_xy=self.landmarks_xy.tolist(),
                         mask={name: mask.to_png_meta() for name, mask in self.mask.items()})
        return alignment

    def from_png_meta(self, alignment):
        """ Set the attributes of this class from alignments stored in a png exif header.

        Parameters
        ----------
        alignment: dict
            A dictionary entry for a face from alignments stored in a png exif header containing
            the keys ``x``, ``w``, ``y``, ``h``, ``landmarks_xy`` and ``mask``
        """
        self.x = alignment["x"]
        self.w = alignment["w"]
        self.y = alignment["y"]
        self.h = alignment["h"]
        self.landmarks_xy = np.array(alignment["landmarks_xy"], dtype="float32")
        self.mask = dict()
        for name, mask_dict in alignment["mask"].items():
            self.mask[name] = Mask()
            self.mask[name].from_dict(mask_dict)
        logger.trace("Created from png exif header: (x: %s, w: %s, y: %s. h: %s, landmarks: %s, "
                     "mask: %s)", self.x, self.w, self.y, self.h, self.landmarks_xy, self.mask)

    def _image_to_face(self, image):
        """ set self.image to be the cropped face from detected bounding box """
        logger.trace("Cropping face from image")
        self.image = image[self.top: self.bottom,
                           self.left: self.right]

    # <<< Aligned Face methods and properties >>> #
    def load_aligned(self, image, size=256, dtype=None, centering="head", force=False):
        """ Align a face from a given image.

        Aligning a face is a relatively expensive task and is not required for all uses of
        the :class:`~lib.align.DetectedFace` object, so call this function explicitly to
        load an aligned face.

        This method plugs into :mod:`lib.align.AlignedFace` to perform face alignment based on this
        face's ``landmarks_xy``. If the face has already been aligned, then this function will
        return having performed no action.

        Parameters
        ----------
        image: numpy.ndarray
            The image that contains the face to be aligned
        size: int
            The size of the output face in pixels
        dtype: str, optional
            Optionally set a ``dtype`` for the final face to be formatted in. Default: ``None``
        centering: ["legacy", "face", "head"], optional
            The type of extracted face that should be loaded. "legacy" places the nose in the
            center of the image (the original method for aligning). "face" aligns for the nose to
            be in the center of the face (top to bottom) but the center of the skull for left to
            right. "head" aligns for the center of the skull (in 3D space) being the center of the
            extracted image, with the crop holding the full head.
            Default: `"head"`
        force: bool, optional
            Force an update of the aligned face, even if it is already loaded. Default: ``False``

        Notes
        -----
        This method must be executed to get access to the following an :class:`AlignedFace` object
        """
        if self.aligned and not force:
            # Don't reload an already aligned face
            logger.trace("Skipping alignment calculation for already aligned face")
        else:
            logger.trace("Loading aligned face: (size: %s, dtype: %s)", size, dtype)
            self.aligned = AlignedFace(self.landmarks_xy,
                                       image=image,
                                       centering=centering,
                                       size=size,
                                       coverage_ratio=1.0,
                                       dtype=dtype,
                                       is_aligned=False)


class _LandmarksMask():  # pylint:disable=too-few-public-methods
    """ Create a single channel mask from aligned landmark points.

    size: tuple
        The (height, width) shape tuple that the mask should be returned as
    points: list
        A list of landmark points that correspond to the given shape tuple to create
        the mask. Each item in the list should be a :class:`numpy.ndarray` that a filled
        convex polygon will be created from
    dilation: int, optional
        The amount of dilation to apply to the mask. `0` for none. Default: `0`
    blur_kernel: int, optional
        The kernel size for applying gaussian blur to apply to the mask. `0` for none. Default: `0`
    """
    def __init__(self, size, points, dilation=0, blur_kernel=0):
        logger.trace("Initializing: %s: (size: %s, points: %s, dilation: %s, blur_kernel: %s)",
                     self.__class__.__name__, size, points, dilation, blur_kernel)
        self._size = size
        self._points = points
        self._dilation = dilation
        self._blur_kernel = blur_kernel
        self._mask = None
        logger.trace("Initialized: %s", self.__class__.__name__)

    def get(self, as_zip=False):
        """ Obtain the mask.

        Parameters
        ----------
        as_zip: bool, optional
            ``True`` if the mask should be returned zipped otherwise ``False``

        Returns
        -------
        :class:`numpy.ndarray` or zipped array
            The mask as a single channel image of the given :attr:`size` dimension. If
            :attr:`as_zip` is ``True`` then the :class:`numpy.ndarray` will be contained within a
            zipped container
        """
        if not np.any(self._mask):
            self._generate_mask()
        retval = compress(self._mask) if as_zip else self._mask
        logger.trace("as_zip: %s, retval type: %s", as_zip, type(retval))
        return retval

    def _generate_mask(self):
        """ Generate the mask.

        Creates the mask applying any requested dilation and blurring and assigns to
        :attr:`_mask`

        Returns
        -------
        :class:`numpy.ndarray`
            The mask as a single channel image of the given :attr:`size` dimension.
        """
        mask = np.zeros((self._size) + (1, ), dtype="float32")
        for landmarks in self._points:
            lms = np.rint(landmarks).astype("int")
            cv2.fillConvexPoly(mask, cv2.convexHull(lms), 1.0, lineType=cv2.LINE_AA)
        if self._dilation != 0:
            mask = cv2.dilate(mask,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                        (self._dilation, self._dilation)),
                              iterations=1)
        if self._blur_kernel != 0:
            mask = BlurMask("gaussian", mask, self._blur_kernel).blurred
        logger.trace("mask: (shape: %s, dtype: %s)", mask.shape, mask.dtype)
        self._mask = (mask * 255.0).astype("uint8")


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
    def __init__(self, storage_size=128, storage_centering="face"):
        logger.trace("Initializing: %s (storage_size: %s, storage_centering: %s)",
                     self.__class__.__name__, storage_size, storage_centering)
        self.stored_size = storage_size
        self.stored_centering = storage_centering

        self._mask = None
        self._affine_matrix = None
        self._interpolator = None

        self._blur = dict()
        self._blur_kernel = 0
        self._threshold = 0.0
        self._sub_crop_size = 0
        self._sub_crop_slices: Dict[Literal["in", "out"], List[slice]] = {}
        self.set_blur_and_threshold()
        logger.trace("Initialized: %s", self.__class__.__name__)

    @property
    def mask(self):
        """ numpy.ndarray: The mask at the size of :attr:`stored_size` with any requested blurring,
        threshold amount and centering applied."""
        mask = self.stored_mask
        if self._threshold != 0.0 or self._blur["kernel"] != 0:
            mask = mask.copy()
        if self._threshold != 0.0:
            mask[mask < self._threshold] = 0.0
            mask[mask > 255.0 - self._threshold] = 255.0
        if self._blur["kernel"] != 0:
            mask = BlurMask(self._blur["type"],
                            mask,
                            self._blur["kernel"],
                            passes=self._blur["passes"]).blurred
        if self._sub_crop_size:  # Crop the mask to the given centering
            out = np.zeros((self._sub_crop_size, self._sub_crop_size, 1), dtype=mask.dtype)
            slice_in, slice_out = self._sub_crop_slices["in"], self._sub_crop_slices["out"]
            out[slice_out[0], slice_out[1], :] = mask[slice_in[0], slice_in[1], :]
            mask = out
        logger.trace("mask shape: %s", mask.shape)  # type: ignore
        return mask

    @property
    def stored_mask(self):
        """ :class:`numpy.ndarray`: The mask at the size of :attr:`stored_size` as it is stored
        (i.e. with no blurring/centering applied). """
        dims = (self.stored_size, self.stored_size, 1)
        mask = np.frombuffer(decompress(self._mask), dtype="uint8").reshape(dims)
        logger.trace("stored mask shape: %s", mask.shape)
        return mask

    @property
    def original_roi(self):
        """ :class: `numpy.ndarray`: The original region of interest of the mask in the
        source frame. """
        points = np.array([[0, 0],
                           [0, self.stored_size - 1],
                           [self.stored_size - 1, self.stored_size - 1],
                           [self.stored_size - 1, 0]], np.int32).reshape((-1, 1, 2))
        matrix = cv2.invertAffineTransform(self._affine_matrix)
        roi = cv2.transform(points, matrix).reshape((4, 2))
        logger.trace("Returning: %s", roi)
        return roi

    @property
    def affine_matrix(self):
        """ :class: `numpy.ndarray`: The affine matrix to transpose the mask to a full frame. """
        return self._affine_matrix

    @property
    def interpolator(self):
        """ int: The cv2 interpolator required to transpose the mask to a full frame. """
        return self._interpolator

    def get_full_frame_mask(self, width, height):
        """ Return the stored mask in a full size frame of the given dimensions

        Parameters
        ----------
        width: int
            The width of the original frame that the mask was extracted from
        height: int
            The height of the original frame that the mask was extracted from

        Returns
        -------
        numpy.ndarray: The mask affined to the original full frame of the given dimensions
        """
        frame = np.zeros((width, height, 1), dtype="uint8")
        mask = cv2.warpAffine(self.mask,
                              self._affine_matrix,
                              (width, height),
                              frame,
                              flags=cv2.WARP_INVERSE_MAP | self._interpolator,
                              borderMode=cv2.BORDER_CONSTANT)
        logger.trace("mask shape: %s, mask dtype: %s, mask min: %s, mask max: %s",
                     mask.shape, mask.dtype, mask.min(), mask.max())
        return mask

    def add(self, mask, affine_matrix, interpolator):
        """ Add a Faceswap mask to this :class:`Mask`.

        The mask should be the original output from  :mod:`plugins.extract.mask`

        Parameters
        ----------
        mask: numpy.ndarray
            The mask that is to be added as output from :mod:`plugins.extract.mask`
            It should be in the range 0.0 - 1.0 ideally with a ``dtype`` of ``float32``
        affine_matrix: numpy.ndarray
            The transformation matrix required to transform the mask to the original frame.
        interpolator, int:
            The CV2 interpolator required to transform this mask to it's original frame
        """
        logger.trace("mask shape: %s, mask dtype: %s, mask min: %s, mask max: %s, "
                     "affine_matrix: %s, interpolator: %s)", mask.shape, mask.dtype, mask.min(),
                     affine_matrix, mask.max(), interpolator)
        self._affine_matrix = self._adjust_affine_matrix(mask.shape[0], affine_matrix)
        self._interpolator = interpolator
        self.replace_mask(mask)

    def replace_mask(self, mask):
        """ Replace the existing :attr:`_mask` with the given mask.

        Parameters
        ----------
        mask: numpy.ndarray
            The mask that is to be added as output from :mod:`plugins.extract.mask`.
            It should be in the range 0.0 - 1.0 ideally with a ``dtype`` of ``float32``
        """
        mask = (cv2.resize(mask,
                           (self.stored_size, self.stored_size),
                           interpolation=cv2.INTER_AREA) * 255.0).astype("uint8")
        self._mask = compress(mask)

    def set_blur_and_threshold(self,
                               blur_kernel=0, blur_type="gaussian", blur_passes=1, threshold=0):
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
        logger.trace("blur_kernel: %s, threshold: %s", blur_kernel, threshold)
        if blur_type is not None:
            blur_kernel += 0 if blur_kernel == 0 or blur_kernel % 2 == 1 else 1
            self._blur["kernel"] = blur_kernel
            self._blur["type"] = blur_type
            self._blur["passes"] = blur_passes
        self._threshold = (threshold / 100.0) * 255.0

    def set_sub_crop(self,
                     source_offset: np.ndarray,
                     target_offset: np.ndarray,
                     centering: "CenteringType",
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

        logger.trace("src_size: %s, coverage_ratio: %s, sub_crop_size: %s, ",  # type: ignore
                     "sub_crop_slices: %s", roi, coverage_ratio, self._sub_crop_size,
                     self._sub_crop_slices)

    def _adjust_affine_matrix(self, mask_size, affine_matrix):
        """ Adjust the affine matrix for the mask's storage size

        Parameters
        ----------
        mask_size: int
            The original size of the mask.
        affine_matrix: numpy.ndarray
            The affine matrix to transform the mask at original size to the parent frame.

        Returns
        -------
        affine_matrix: numpy,ndarray
            The affine matrix adjusted for the mask at its stored dimensions.
        """
        zoom = self.stored_size / mask_size
        zoom_mat = np.array([[zoom, 0, 0.], [0, zoom, 0.]])
        adjust_mat = np.dot(zoom_mat, np.concatenate((affine_matrix, np.array([[0., 0., 1.]]))))
        logger.trace("storage_size: %s, mask_size: %s, zoom: %s, original matrix: %s, "
                     "adjusted_matrix: %s", self.stored_size, mask_size, zoom, affine_matrix.shape,
                     adjust_mat.shape)
        return adjust_mat

    def to_dict(self):
        """ Convert the mask to a dictionary for saving to an alignments file

        Returns
        -------
        dict:
            The :class:`Mask` for saving to an alignments file. Contains the keys ``mask``,
            ``affine_matrix``, ``interpolator``, ``stored_size``, ``stored_centering``
        """
        retval = dict()
        for key in ("mask", "affine_matrix", "interpolator", "stored_size", "stored_centering"):
            retval[key] = getattr(self, self._attr_name(key))
        logger.trace({k: v if k != "mask" else type(v) for k, v in retval.items()})
        return retval

    def to_png_meta(self):
        """ Convert the mask to a dictionary supported by png itxt headers.

        Returns
        -------
        dict:
            The :class:`Mask` for saving to an alignments file. Contains the keys ``mask``,
            ``affine_matrix``, ``interpolator``, ``stored_size``, ``stored_centering``
        """
        retval = dict()
        for key in ("mask", "affine_matrix", "interpolator", "stored_size", "stored_centering"):
            val = getattr(self, self._attr_name(key))
            if isinstance(val, np.ndarray):
                retval[key] = val.tolist()
            else:
                retval[key] = val
        logger.trace({k: v if k != "mask" else type(v) for k, v in retval.items()})
        return retval

    def from_dict(self, mask_dict):
        """ Populates the :class:`Mask` from a dictionary loaded from an alignments file.

        Parameters
        ----------
        mask_dict: dict
            A dictionary stored in an alignments file containing the keys ``mask``,
            ``affine_matrix``, ``interpolator``, ``stored_size``, ``stored_centering``
        """
        for key in ("mask", "affine_matrix", "interpolator", "stored_size", "stored_centering"):
            val = mask_dict.get(key)
            val = "face" if key == "stored_centering" and val is None else val
            if key == "affine_matrix" and not isinstance(val, np.ndarray):
                val = np.array(val, dtype="float64")
            setattr(self, self._attr_name(key), val)
            logger.trace("%s - %s", key, val if key != "mask" else type(val))

    @staticmethod
    def _attr_name(dict_key):
        """ The :class:`Mask` attribute name for the given dictionary key

        Parameters
        ----------
        dict_key: str
            The key name from an alignments dictionary

        Returns
        -------
        attribute_name: str
            The attribute name for the given key for :class:`Mask`
        """
        retval = "_{}".format(dict_key) if not dict_key.startswith("stored") else dict_key
        logger.trace("dict_key: %s, attribute_name: %s", dict_key, retval)
        return retval


class BlurMask():  # pylint:disable=too-few-public-methods
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
    def __init__(self, blur_type, mask, kernel, is_ratio=False, passes=1):
        logger.trace("Initializing %s: (blur_type: '%s', mask_shape: %s, kernel: %s, "
                     "is_ratio: %s, passes: %s)", self.__class__.__name__, blur_type, mask.shape,
                     kernel, is_ratio, passes)
        self._blur_type = blur_type.lower()
        self._mask = mask
        self._passes = passes
        kernel_size = self._get_kernel_size(kernel, is_ratio)
        self._kernel_size = self._get_kernel_tuple(kernel_size)
        logger.trace("Initialized %s", self.__class__.__name__)

    @property
    def blurred(self):
        """ :class:`numpy.ndarray`: The final mask with blurring applied. """
        func = self._func_mapping[self._blur_type]
        kwargs = self._get_kwargs()
        blurred = self._mask
        for i in range(self._passes):
            ksize = int(kwargs["ksize"][0])
            logger.trace("Pass: %s, kernel_size: %s", i + 1, (ksize, ksize))
            blurred = func(blurred, **kwargs)
            ksize = int(round(ksize * self._multipass_factor))
            kwargs["ksize"] = self._get_kernel_tuple(ksize)
        blurred = blurred[..., None]
        logger.trace("Returning blurred mask. Shape: %s", blurred.shape)
        return blurred

    @property
    def _multipass_factor(self):
        """ For multiple passes the kernel must be scaled down. This value is
            different for box filter and gaussian """
        factor = dict(gaussian=0.8, normalized=0.5)
        return factor[self._blur_type]

    @property
    def _sigma(self):
        """ int: The Sigma for Gaussian Blur. Returns 0 to force calculation from kernel size. """
        return 0

    @property
    def _func_mapping(self):
        """ dict: :attr:`_blur_type` mapped to cv2 Function name. """
        return dict(gaussian=cv2.GaussianBlur,  # pylint: disable = no-member
                    normalized=cv2.blur)  # pylint: disable = no-member

    @property
    def _kwarg_requirements(self):
        """ dict: :attr:`_blur_type` mapped to cv2 Function required keyword arguments. """
        return dict(gaussian=["ksize", "sigmaX"],
                    normalized=["ksize"])

    @property
    def _kwarg_mapping(self):
        """ dict: cv2 function keyword arguments mapped to their parameters. """
        return dict(ksize=self._kernel_size,
                    sigmaX=self._sigma)

    def _get_kernel_size(self, kernel, is_ratio):
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
            return kernel

        mask_diameter = np.sqrt(np.sum(self._mask))
        radius = round(max(1., mask_diameter * kernel / 100.))
        kernel_size = int(radius * 2 + 1)
        logger.trace("kernel_size: %s", kernel_size)
        return kernel_size

    @staticmethod
    def _get_kernel_tuple(kernel_size):
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
        logger.trace(retval)
        return retval

    def _get_kwargs(self):
        """ dict: the valid keyword arguments for the requested :attr:`_blur_type` """
        retval = {kword: self._kwarg_mapping[kword]
                  for kword in self._kwarg_requirements[self._blur_type]}
        logger.trace("BlurMask kwargs: %s", retval)
        return retval


_HASHES_SEEN = dict()


def update_legacy_png_header(filename, alignments):
    """ Update a legacy extracted face from pre v2.1 alignments by placing the alignment data for
    the face in the png exif header for the given filename with the given alignment data.

    If the given file is not a .png then a png is created and the original file is removed

    Parameters
    ----------
    filename: str
        The image file to update
    alignments: :class:`lib.align.alignments.Alignments`
        The alignments data the contains the information to store in the image header. This must be
        a v2.0 or less alignments file as later versions no longer store the face hash (not
        required)

    Returns
    -------
    dict
        The metadata that has been applied to the given image
    """
    if alignments.version > 2.0:
        raise FaceswapError("The faces being passed in do not correspond to the given Alignments "
                            "file. Please double check your sources and try again.")
    # Track hashes for multiple files with the same hash. Not the most robust but should be
    # effective enough
    folder = os.path.dirname(filename)
    if folder not in _HASHES_SEEN:
        _HASHES_SEEN[folder] = dict()
    hashes_seen = _HASHES_SEEN[folder]

    in_image = read_image(filename, raise_error=True)
    in_hash = sha1(in_image).hexdigest()
    hashes_seen[in_hash] = hashes_seen.get(in_hash, -1) + 1

    alignment = alignments.hashes_to_alignment.get(in_hash)
    if not alignment:
        logger.debug("Alignments not found for image: '%s'", filename)
        return None

    detected_face = DetectedFace()
    detected_face.from_alignment(alignment)
    # For dupe hash handling, make sure we get a different filename for repeat hashes
    src_fname, face_idx = list(alignments.hashes_to_frame[in_hash].items())[hashes_seen[in_hash]]
    orig_filename = "{}_{}.png".format(os.path.splitext(src_fname)[0], face_idx)
    meta = dict(alignments=detected_face.to_png_meta(),
                source=dict(alignments_version=alignments.version,
                            original_filename=orig_filename,
                            face_index=face_idx,
                            source_filename=src_fname,
                            source_is_video=False))  # Can't check so set false

    out_filename = f"{os.path.splitext(filename)[0]}.png"  # Make sure saved file is png
    out_image = encode_image(in_image, ".png", metadata=meta)

    with open(out_filename, "wb") as out_file:
        out_file.write(out_image)

    if filename != out_filename:  # Remove the old non-png:
        logger.debug("Removing replaced face with deprecated extension: '%s'", filename)
        os.remove(filename)

    return meta
