#!/usr/bin python3
""" Face and landmarks detection for faceswap.py """

import logging
import sys
import os

from hashlib import sha1
from typing import cast, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING, Union
from zlib import compress, decompress

import cv2
import numpy as np

from lib.image import encode_image, read_image
from lib.utils import FaceswapError
from .alignments import (Alignments, AlignmentFileDict, MaskAlignmentsFileDict,
                         PNGHeaderAlignmentsDict, PNGHeaderDict, PNGHeaderSourceDict)
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
    left: int
        The left most point (in pixels) of the face's bounding box as discovered in
        :mod:`plugins.extract.detect`
    width: int
        The width (in pixels) of the face's bounding box as discovered in
        :mod:`plugins.extract.detect`
    top: int
        The top most point (in pixels) of the face's bounding box as discovered in
        :mod:`plugins.extract.detect`
    height: int
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
    left: int
        The left most point (in pixels) of the face's bounding box as discovered in
        :mod:`plugins.extract.detect`
    width: int
        The width (in pixels) of the face's bounding box as discovered in
        :mod:`plugins.extract.detect`
    top: int
        The top most point (in pixels) of the face's bounding box as discovered in
        :mod:`plugins.extract.detect`
    height: int
        The height (in pixels) of the face's bounding box as discovered in
        :mod:`plugins.extract.detect`
    landmarks_xy: list
        The 68 point landmarks as discovered in :mod:`plugins.extract.align`.
    mask: dict
        The generated mask(s) for the face as generated in :mod:`plugins.extract.mask`. Is a
        dict of {**name** (`str`): :class:`Mask`}.
    """
    def __init__(self,
                 image: Optional[np.ndarray] = None,
                 left: Optional[int] = None,
                 width: Optional[int] = None,
                 top: Optional[int] = None,
                 height: Optional[int] = None,
                 landmarks_xy: Optional[np.ndarray] = None,
                 mask: Optional[Dict[str, "Mask"]] = None,
                 filename: Optional[str] = None) -> None:
        logger.trace("Initializing %s: (image: %s, left: %s, width: %s, top: %s, "  # type: ignore
                     "height: %s, landmarks_xy: %s, mask: %s, filename: %s)",
                     self.__class__.__name__,
                     image.shape if image is not None and image.any() else image, left, width, top,
                     height, landmarks_xy, mask, filename)
        self.image = image
        self.left = left
        self.width = width
        self.top = top
        self.height = height
        self._landmarks_xy = landmarks_xy
        self._identity: Dict[Literal["vggface2"], np.ndarray] = {}
        self.thumbnail: Optional[np.ndarray] = None
        self.mask = {} if mask is None else mask
        self._training_masks: Optional[Tuple[bytes, Tuple[int, int, int]]] = None

        self._aligned: Optional[AlignedFace] = None
        logger.trace("Initialized %s", self.__class__.__name__)  # type: ignore

    @property
    def aligned(self) -> AlignedFace:
        """ The aligned face connected to this detected face. """
        assert self._aligned is not None
        return self._aligned

    @property
    def landmarks_xy(self) -> np.ndarray:
        """ The aligned face connected to this detected face. """
        assert self._landmarks_xy is not None
        return self._landmarks_xy

    @property
    def right(self) -> int:
        """int: Right point (in pixels) of face detection bounding box within the parent image """
        assert self.left is not None and self.width is not None
        return self.left + self.width

    @property
    def bottom(self) -> int:
        """int: Bottom point (in pixels) of face detection bounding box within the parent image """
        assert self.top is not None and self.height is not None
        return self.top + self.height

    @property
    def identity(self) -> Dict[Literal["vggface2"], np.ndarray]:
        """ dict: Identity mechanism as key, identity embedding as value. """
        return self._identity

    def add_mask(self,
                 name: str,
                 mask: np.ndarray,
                 affine_matrix: np.ndarray,
                 interpolator: int,
                 storage_size: int = 128,
                 storage_centering: "CenteringType" = "face") -> None:
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
        logger.trace("name: '%s', mask shape: %s, affine_matrix: %s, "  # type: ignore
                     "interpolator: %s, storage_size: %s, storage_centering: %s)", name,
                     mask.shape, affine_matrix, interpolator, storage_size, storage_centering)
        fsmask = Mask(storage_size=storage_size, storage_centering=storage_centering)
        fsmask.add(mask, affine_matrix, interpolator)
        self.mask[name] = fsmask

    def add_landmarks_xy(self, landmarks: np.ndarray) -> None:
        """ Add landmarks to the detected face object. If landmarks alread exist, they will be
        overwritten.

        Parameters
        ----------
        landmarks: :class:`numpy.ndarray`
            The 68 point face landmarks to add for the face
        """
        logger.trace("landmarks shape: '%s'", landmarks.shape)  # type: ignore
        self._landmarks_xy = landmarks

    def add_identity(self, name: Literal["vggface2"], embedding: np.ndarray, ) -> None:
        """ Add an identity embedding to this detected face. If an identity already exists for the
        given :attr:`name` it will be overwritten

        Parameters
        ----------
        name: str
            The name of the mechanism that calculated the identity
        embedding: numpy.ndarray
            The identity embedding
        """
        logger.trace("name: '%s', embedding shape: %s",  # type: ignore
                     name, embedding.shape)
        assert name == "vggface2"
        assert embedding.shape[0] == 512
        self._identity[name] = embedding

    def get_landmark_mask(self,
                          area: Literal["eye", "face", "mouth"],
                          blur_kernel: int,
                          dilation: int) -> np.ndarray:
        """ Add a :class:`LandmarksMask` to this detected face

        Landmark based masks are generated from face Aligned Face landmark points. An aligned
        face must be loaded. As the data is coming from the already aligned face, no further mask
        cropping is required.

        Parameters
        ----------
        area: ["face", "mouth", "eye"]
            The type of mask to obtain. `face` is a full face mask the others are masks for those
            specific areas
        blur_kernel: int
            The size of the kernel for blurring the mask edges
        dilation: int
            The amount of dilation to apply to the mask. `0` for none. Default: `0`

        Returns
        -------
        :class:`numpy.ndarray`
            The generated landmarks mask for the selected area
        """
        # TODO Face mask generation from landmarks
        logger.trace("area: %s, dilation: %s", area, dilation)  # type: ignore
        areas = dict(mouth=[slice(48, 60)], eye=[slice(36, 42), slice(42, 48)])
        points = [self.aligned.landmarks[zone]
                  for zone in areas[area]]

        lmmask = LandmarksMask(points,
                               storage_size=self.aligned.size,
                               storage_centering=self.aligned.centering,
                               dilation=dilation)
        lmmask.set_blur_and_threshold(blur_kernel=blur_kernel)
        lmmask.generate_mask(
            self.aligned.adjusted_matrix,
            self.aligned.interpolators[1])
        return lmmask.mask

    def store_training_masks(self,
                             masks: List[Optional[np.ndarray]],
                             delete_masks: bool = False) -> None:
        """ Concatenate and compress the given training masks and store for retrieval.

        Parameters
        ----------
        masks: list
            A list of training mask. Must be all be uint-8 3D arrays of the same size in
            0-255 range
        delete_masks: bool, optional
            ``True`` to delete any of the :class:`Mask` objects owned by this detected face. Use to
            free up unrequired memory usage. Default: ``False``
        """
        if delete_masks:
            del self.mask
            self.mask = {}

        valid = [msk for msk in masks if msk is not None]
        if not valid:
            return
        combined = np.concatenate(valid, axis=-1)
        self._training_masks = (compress(combined), combined.shape)

    def get_training_masks(self) -> Optional[np.ndarray]:
        """ Obtain the decompressed combined training masks.

        Returns
        -------
        :class:`numpy.ndarray`
            A 3D array containing the decompressed training masks as uint8 in 0-255 range if
            training masks are present otherwise ``None``
        """
        if not self._training_masks:
            return None
        return np.frombuffer(decompress(self._training_masks[0]),
                             dtype="uint8").reshape(self._training_masks[1])

    def to_alignment(self) -> AlignmentFileDict:
        """  Return the detected face formatted for an alignments file

        returns
        -------
        alignment: dict
            The alignment dict will be returned with the keys ``x``, ``w``, ``y``, ``h``,
            ``landmarks_xy``, ``mask``. The additional key ``thumb`` will be provided if the
            detected face object contains a thumbnail.
        """
        if (self.left is None or self.width is None or self.top is None or self.height is None):
            raise AssertionError("Some detected face variables have not been initialized")
        alignment = AlignmentFileDict(x=self.left,
                                      w=self.width,
                                      y=self.top,
                                      h=self.height,
                                      landmarks_xy=self.landmarks_xy,
                                      mask={name: mask.to_dict()
                                            for name, mask in self.mask.items()},
                                      identity={k: v.tolist() for k, v in self._identity.items()},
                                      thumb=self.thumbnail)
        logger.trace("Returning: %s", alignment)  # type: ignore
        return alignment

    def from_alignment(self, alignment: AlignmentFileDict,
                       image: Optional[np.ndarray] = None, with_thumb: bool = False) -> None:
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

        logger.trace("Creating from alignment: (alignment: %s, has_image: %s)",  # type: ignore
                     alignment, bool(image is not None))
        self.left = alignment["x"]
        self.width = alignment["w"]
        self.top = alignment["y"]
        self.height = alignment["h"]
        landmarks = alignment["landmarks_xy"]
        if not isinstance(landmarks, np.ndarray):
            landmarks = np.array(landmarks, dtype="float32")
        self._identity = {cast(Literal["vggface2"], k): np.array(v, dtype="float32")
                          for k, v in alignment.get("identity", {}).items()}
        self._landmarks_xy = landmarks.copy()

        if with_thumb:
            # Thumbnails currently only used for manual tool. Default to None
            self.thumbnail = alignment.get("thumb")
        # Manual tool and legacy alignments will not have a mask
        self._aligned = None

        if alignment.get("mask", None) is not None:
            self.mask = {}
            for name, mask_dict in alignment["mask"].items():
                self.mask[name] = Mask()
                self.mask[name].from_dict(mask_dict)
        if image is not None and image.any():
            self._image_to_face(image)
        logger.trace("Created from alignment: (left: %s, width: %s, top: %s, "  # type: ignore
                     "height: %s, landmarks: %s, mask: %s)", self.left, self.width, self.top,
                     self.height, self.landmarks_xy, self.mask)

    def to_png_meta(self) -> PNGHeaderAlignmentsDict:
        """ Return the detected face formatted for insertion into a png itxt header.

        returns: dict
            The alignments dict will be returned with the keys ``x``, ``w``, ``y``, ``h``,
            ``landmarks_xy`` and ``mask``
        """
        if (self.left is None or self.width is None or self.top is None or self.height is None):
            raise AssertionError("Some detected face variables have not been initialized")
        alignment = PNGHeaderAlignmentsDict(
            x=self.left,
            w=self.width,
            y=self.top,
            h=self.height,
            landmarks_xy=self.landmarks_xy.tolist(),
            mask={name: mask.to_png_meta() for name, mask in self.mask.items()},
            identity={k: v.tolist() for k, v in self._identity.items()})
        return alignment

    def from_png_meta(self, alignment: PNGHeaderAlignmentsDict) -> None:
        """ Set the attributes of this class from alignments stored in a png exif header.

        Parameters
        ----------
        alignment: dict
            A dictionary entry for a face from alignments stored in a png exif header containing
            the keys ``x``, ``w``, ``y``, ``h``, ``landmarks_xy`` and ``mask``
        """
        self.left = alignment["x"]
        self.width = alignment["w"]
        self.top = alignment["y"]
        self.height = alignment["h"]
        self._landmarks_xy = np.array(alignment["landmarks_xy"], dtype="float32")
        self.mask = {}
        for name, mask_dict in alignment["mask"].items():
            self.mask[name] = Mask()
            self.mask[name].from_dict(mask_dict)
        self._identity = {}
        for key, val in alignment.get("identity", {}).items():
            assert key in ["vggface2"]
            self._identity[cast(Literal["vggface2"], key)] = np.array(val, dtype="float32")
        logger.trace("Created from png exif header: (left: %s, width: %s, top: %s "  # type: ignore
                     " height: %s, landmarks: %s, mask: %s, identity: %s)", self.left, self.width,
                     self.top, self.height, self.landmarks_xy, self.mask,
                     {k: v.shape for k, v in self._identity.items()})

    def _image_to_face(self, image: np.ndarray) -> None:
        """ set self.image to be the cropped face from detected bounding box """
        logger.trace("Cropping face from image")  # type: ignore
        self.image = image[self.top: self.bottom,
                           self.left: self.right]

    # <<< Aligned Face methods and properties >>> #
    def load_aligned(self,
                     image: Optional[np.ndarray],
                     size: int = 256,
                     dtype: Optional[str] = None,
                     centering: "CenteringType" = "head",
                     coverage_ratio: float = 1.0,
                     force: bool = False,
                     is_aligned: bool = False,
                     is_legacy: bool = False) -> None:
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
        coverage_ratio: float, optional
            The amount of the aligned image to return. A ratio of 1.0 will return the full contents
            of the aligned image. A ratio of 0.5 will return an image of the given size, but will
            crop to the central 50%% of the image. Default: `1.0`
        force: bool, optional
            Force an update of the aligned face, even if it is already loaded. Default: ``False``
        is_aligned: bool, optional
            Indicates that the :attr:`image` is an aligned face rather than a frame.
            Default: ``False``
        is_legacy: bool, optional
            Only used if `is_aligned` is ``True``. ``True`` indicates that the aligned image being
            loaded is a legacy extracted face rather than a current head extracted face
        Notes
        -----
        This method must be executed to get access to the following an :class:`AlignedFace` object
        """
        if self._aligned and not force:
            # Don't reload an already aligned face
            logger.trace("Skipping alignment calculation for already aligned face")  # type: ignore
        else:
            logger.trace("Loading aligned face: (size: %s, dtype: %s)",  # type: ignore
                         size, dtype)
            self._aligned = AlignedFace(self.landmarks_xy,
                                        image=image,
                                        centering=centering,
                                        size=size,
                                        coverage_ratio=coverage_ratio,
                                        dtype=dtype,
                                        is_aligned=is_aligned,
                                        is_legacy=is_aligned and is_legacy)


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
                 storage_centering: "CenteringType" = "face") -> None:
        logger.trace("Initializing: %s (storage_size: %s, storage_centering: %s)",  # type: ignore
                     self.__class__.__name__, storage_size, storage_centering)
        self.stored_size = storage_size
        self.stored_centering = storage_centering

        self._mask: Optional[bytes] = None
        self._affine_matrix: Optional[np.ndarray] = None
        self._interpolator: Optional[int] = None

        self._blur_type: Optional[Literal["gaussian", "normalized"]] = None
        self._blur_passes: int = 0
        self._blur_kernel: Union[float, int] = 0
        self._threshold = 0.0
        self._sub_crop_size = 0
        self._sub_crop_slices: Dict[Literal["in", "out"], List[slice]] = {}

        self.set_blur_and_threshold()
        logger.trace("Initialized: %s", self.__class__.__name__)  # type: ignore

    @property
    def mask(self) -> np.ndarray:
        """ :class:`numpy.ndarray`: The mask at the size of :attr:`stored_size` with any requested
        blurring, threshold amount and centering applied."""
        mask = self.stored_mask
        if self._threshold != 0.0 or self._blur_kernel != 0:
            mask = mask.copy()
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
        logger.trace("mask shape: %s", mask.shape)  # type: ignore
        return mask

    @property
    def stored_mask(self) -> np.ndarray:
        """ :class:`numpy.ndarray`: The mask at the size of :attr:`stored_size` as it is stored
        (i.e. with no blurring/centering applied). """
        assert self._mask is not None
        dims = (self.stored_size, self.stored_size, 1)
        mask = np.frombuffer(decompress(self._mask), dtype="uint8").reshape(dims)
        logger.trace("stored mask shape: %s", mask.shape)  # type: ignore
        return mask

    @property
    def original_roi(self) -> np.ndarray:
        """ :class: `numpy.ndarray`: The original region of interest of the mask in the
        source frame. """
        points = np.array([[0, 0],
                           [0, self.stored_size - 1],
                           [self.stored_size - 1, self.stored_size - 1],
                           [self.stored_size - 1, 0]], np.int32).reshape((-1, 1, 2))
        matrix = cv2.invertAffineTransform(self._affine_matrix)
        roi = cv2.transform(points, matrix).reshape((4, 2))
        logger.trace("Returning: %s", roi)  # type: ignore
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
                              self._affine_matrix,
                              (width, height),
                              frame,
                              flags=cv2.WARP_INVERSE_MAP | self._interpolator,
                              borderMode=cv2.BORDER_CONSTANT)
        logger.trace("mask shape: %s, mask dtype: %s, mask min: %s, mask max: %s",  # type: ignore
                     mask.shape, mask.dtype, mask.min(), mask.max())
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
        logger.trace("mask shape: %s, mask dtype: %s, mask min: %s, mask max: %s, "  # type: ignore
                     "affine_matrix: %s, interpolator: %s)", mask.shape, mask.dtype, mask.min(),
                     affine_matrix, mask.max(), interpolator)
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
        mask = (cv2.resize(mask,
                           (self.stored_size, self.stored_size),
                           interpolation=cv2.INTER_AREA) * 255.0).astype("uint8")
        self._mask = compress(mask.tobytes())

    def set_blur_and_threshold(self,
                               blur_kernel: int = 0,
                               blur_type: Optional[Literal["gaussian", "normalized"]] = "gaussian",
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
        logger.trace("blur_kernel: %s, blur_type: %s, blur_passes: %s, "  # type: ignore
                     "threshold: %s", blur_kernel, blur_type, blur_passes, threshold)
        if blur_type is not None:
            blur_kernel += 0 if blur_kernel == 0 or blur_kernel % 2 == 1 else 1
            self._blur_kernel = blur_kernel
            self._blur_type = blur_type
            self._blur_passes = blur_passes
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

        logger.trace("src_size: %s, coverage_ratio: %s, sub_crop_size: %s, "  # type: ignore
                     "sub_crop_slices: %s", roi, coverage_ratio, self._sub_crop_size,
                     self._sub_crop_slices)

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
        logger.trace("storage_size: %s, mask_size: %s, zoom: %s, "  # type: ignore
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
        logger.trace({k: v if k != "mask" else type(v) for k, v in retval.items()})  # type: ignore
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
        logger.trace({k: v if k != "mask" else type(v)  # type: ignore
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
    dilation: int, optional
        The amount of dilation to apply to the mask. `0` for none. Default: `0`
    """
    def __init__(self,
                 points: List[np.ndarray],
                 storage_size: int = 128,
                 storage_centering: "CenteringType" = "face",
                 dilation: int = 0) -> None:
        super().__init__(storage_size=storage_size, storage_centering=storage_centering)
        self._points = points
        self._dilation = dilation

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
            cv2.fillConvexPoly(mask, cv2.convexHull(lms), 1.0, lineType=cv2.LINE_AA)
        if self._dilation != 0:
            mask = cv2.dilate(mask,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                        (self._dilation, self._dilation)),
                              iterations=1)
        if self._blur_kernel != 0 and self._blur_type is not None:
            mask = BlurMask(self._blur_type,
                            mask,
                            self._blur_kernel,
                            passes=self._blur_passes).blurred
        logger.trace("mask: (shape: %s, dtype: %s)", mask.shape, mask.dtype)  # type: ignore
        self.add(mask, affine_matrix, interpolator)


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
    def __init__(self,
                 blur_type: Literal["gaussian", "normalized"],
                 mask: np.ndarray,
                 kernel: Union[int, float],
                 is_ratio: bool = False,
                 passes: int = 1) -> None:
        logger.trace("Initializing %s: (blur_type: '%s', mask_shape: %s, "  # type: ignore
                     "kernel: %s, is_ratio: %s, passes: %s)", self.__class__.__name__, blur_type,
                     mask.shape, kernel, is_ratio, passes)
        self._blur_type = blur_type
        self._mask = mask
        self._passes = passes
        kernel_size = self._get_kernel_size(kernel, is_ratio)
        self._kernel_size = self._get_kernel_tuple(kernel_size)
        logger.trace("Initialized %s", self.__class__.__name__)  # type: ignore

    @property
    def blurred(self) -> np.ndarray:
        """ :class:`numpy.ndarray`: The final mask with blurring applied. """
        func = self._func_mapping[self._blur_type]
        kwargs = self._get_kwargs()
        blurred = self._mask
        for i in range(self._passes):
            assert isinstance(kwargs["ksize"], tuple)
            ksize = int(kwargs["ksize"][0])
            logger.trace("Pass: %s, kernel_size: %s", i + 1, (ksize, ksize))  # type: ignore
            blurred = func(blurred, **kwargs)
            ksize = int(round(ksize * self._multipass_factor))
            kwargs["ksize"] = self._get_kernel_tuple(ksize)
        blurred = blurred[..., None]
        logger.trace("Returning blurred mask. Shape: %s", blurred.shape)  # type: ignore
        return blurred

    @property
    def _multipass_factor(self) -> float:
        """ For multiple passes the kernel must be scaled down. This value is
            different for box filter and gaussian """
        factor = dict(gaussian=0.8, normalized=0.5)
        return factor[self._blur_type]

    @property
    def _sigma(self) -> Literal[0]:
        """ int: The Sigma for Gaussian Blur. Returns 0 to force calculation from kernel size. """
        return 0

    @property
    def _func_mapping(self) -> Dict[Literal["gaussian", "normalized"], Callable]:
        """ dict: :attr:`_blur_type` mapped to cv2 Function name. """
        return dict(gaussian=cv2.GaussianBlur,  # pylint: disable = no-member
                    normalized=cv2.blur)  # pylint: disable = no-member

    @property
    def _kwarg_requirements(self) -> Dict[Literal["gaussian", "normalized"], List[str]]:
        """ dict: :attr:`_blur_type` mapped to cv2 Function required keyword arguments. """
        return dict(gaussian=["ksize", "sigmaX"],
                    normalized=["ksize"])

    @property
    def _kwarg_mapping(self) -> Dict[str, Union[int, Tuple[int, int]]]:
        """ dict: cv2 function keyword arguments mapped to their parameters. """
        return dict(ksize=self._kernel_size,
                    sigmaX=self._sigma)

    def _get_kernel_size(self, kernel: Union[int, float], is_ratio: bool) -> int:
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
        logger.trace("kernel_size: %s", kernel_size)  # type: ignore
        return kernel_size

    @staticmethod
    def _get_kernel_tuple(kernel_size: int) -> Tuple[int, int]:
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
        logger.trace(retval)  # type: ignore
        return retval

    def _get_kwargs(self) -> Dict[str, Union[int, Tuple[int, int]]]:
        """ dict: the valid keyword arguments for the requested :attr:`_blur_type` """
        retval = {kword: self._kwarg_mapping[kword]
                  for kword in self._kwarg_requirements[self._blur_type]}
        logger.trace("BlurMask kwargs: %s", retval)  # type: ignore
        return retval


_HASHES_SEEN: Dict[str, Dict[str, int]] = {}


def update_legacy_png_header(filename: str, alignments: Alignments
                             ) -> Optional[PNGHeaderDict]:
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
        _HASHES_SEEN[folder] = {}
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
    orig_filename = f"{os.path.splitext(src_fname)[0]}_{face_idx}.png"
    meta = PNGHeaderDict(alignments=detected_face.to_png_meta(),
                         source=PNGHeaderSourceDict(
                            alignments_version=alignments.version,
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
