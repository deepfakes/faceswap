#!/usr/bin python3
""" Face and landmarks detection for faceswap.py """
from __future__ import annotations
import logging
import os
import typing as T

from hashlib import sha1
from zlib import compress, decompress

import numpy as np

from lib.image import encode_image, read_image
from lib.logger import parse_class_init
from lib.utils import FaceswapError
from .alignments import (Alignments, AlignmentFileDict, PNGHeaderAlignmentsDict,
                         PNGHeaderDict, PNGHeaderSourceDict)
from .aligned_face import AlignedFace
from .aligned_mask import LandmarksMask, Mask
from .constants import LANDMARK_PARTS

if T.TYPE_CHECKING:
    from .aligned_face import CenteringType

logger = logging.getLogger(__name__)


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
        dict of {**name** (`str`): :class:`~lib.align.aligned_mask.Mask`}.

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
        dict of {**name** (`str`): :class:`~lib.align.aligned_mask.Mask`}.
    """
    def __init__(self,
                 image: np.ndarray | None = None,
                 left: int | None = None,
                 width: int | None = None,
                 top: int | None = None,
                 height: int | None = None,
                 landmarks_xy: np.ndarray | None = None,
                 mask: dict[str, Mask] | None = None) -> None:
        logger.trace(parse_class_init(locals()))  # type:ignore[attr-defined]
        self.image = image
        self.left = left
        self.width = width
        self.top = top
        self.height = height
        self._landmarks_xy = landmarks_xy
        self._identity: dict[str, np.ndarray] = {}
        self.thumbnail: np.ndarray | None = None
        self.mask = {} if mask is None else mask
        self._training_masks: tuple[bytes, tuple[int, int, int]] | None = None

        self._aligned: AlignedFace | None = None
        logger.trace("Initialized %s", self.__class__.__name__)  # type:ignore[attr-defined]

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
    def identity(self) -> dict[str, np.ndarray]:
        """ dict: Identity mechanism as key, identity embedding as value. """
        return self._identity

    def add_mask(self,
                 name: str,
                 mask: np.ndarray,
                 affine_matrix: np.ndarray,
                 interpolator: int,
                 storage_size: int = 128,
                 storage_centering: CenteringType = "face") -> None:
        """ Add a :class:`~lib.align.aligned_mask.Mask` to this detected face

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
        logger.trace("name: '%s', mask shape: %s, affine_matrix: %s, "  # type:ignore[attr-defined]
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
        logger.trace("landmarks shape: '%s'", landmarks.shape)  # type:ignore[attr-defined]
        self._landmarks_xy = landmarks

    def add_identity(self, name: str, embedding: np.ndarray, ) -> None:
        """ Add an identity embedding to this detected face. If an identity already exists for the
        given :attr:`name` it will be overwritten

        Parameters
        ----------
        name: str
            The name of the mechanism that calculated the identity
        embedding: numpy.ndarray
            The identity embedding
        """
        logger.trace("name: '%s', embedding shape: %s",  # type:ignore[attr-defined]
                     name, embedding.shape)
        assert name == "vggface2"
        assert embedding.shape[0] == 512
        self._identity[name] = embedding

    def clear_all_identities(self) -> None:
        """ Remove all stored identity embeddings """
        self._identity = {}

    def get_landmark_mask(self,
                          area: T.Literal["eye", "face", "mouth"],
                          blur_kernel: int,
                          dilation: float) -> np.ndarray:
        """ Add a :class:`L~lib.align.aligned_mask.LandmarksMask` to this detected face

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
        dilation: float
            The amount of dilation to apply to the mask. as a percentage of the mask size

        Returns
        -------
        :class:`numpy.ndarray`
            The generated landmarks mask for the selected area

        Raises
        ------
        FaceSwapError
            If the aligned face does not contain the correct landmarks to generate a landmark mask
        """
        # TODO Face mask generation from landmarks
        logger.trace("area: %s, dilation: %s", area, dilation)  # type:ignore[attr-defined]

        lm_type = self.aligned.landmark_type
        if lm_type not in LANDMARK_PARTS:
            raise FaceswapError(f"Landmark based masks cannot be created for {lm_type.name}")

        lm_parts = LANDMARK_PARTS[self.aligned.landmark_type]
        mapped = {"mouth": ["mouth_outer"], "eye": ["right_eye", "left_eye"]}
        if not all(part in lm_parts for parts in mapped.values() for part in parts):
            raise FaceswapError(f"Landmark based masks cannot be created for {lm_type.name}")

        areas = {key: [slice(*lm_parts[v][:2]) for v in val]for key, val in mapped.items()}
        points = [self.aligned.landmarks[zone] for zone in areas[area]]

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
                             masks: list[np.ndarray | None],
                             delete_masks: bool = False) -> None:
        """ Concatenate and compress the given training masks and store for retrieval.

        Parameters
        ----------
        masks: list
            A list of training mask. Must be all be uint-8 3D arrays of the same size in
            0-255 range
        delete_masks: bool, optional
            ``True`` to delete any of the :class:`~lib.align.aligned_mask.Mask` objects owned by
            this detected face. Use to free up unrequired memory usage. Default: ``False``
        """
        if delete_masks:
            del self.mask
            self.mask = {}

        valid = [msk for msk in masks if msk is not None]
        if not valid:
            return
        combined = np.concatenate(valid, axis=-1)
        self._training_masks = (compress(combined), combined.shape)

    def get_training_masks(self) -> np.ndarray | None:
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
        logger.trace("Returning: %s", alignment)  # type:ignore[attr-defined]
        return alignment

    def from_alignment(self, alignment: AlignmentFileDict,
                       image: np.ndarray | None = None, with_thumb: bool = False) -> None:
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

        logger.trace("Creating from alignment: (alignment: %s,"  # type:ignore[attr-defined]
                     " has_image: %s)", alignment, bool(image is not None))
        self.left = alignment["x"]
        self.width = alignment["w"]
        self.top = alignment["y"]
        self.height = alignment["h"]
        landmarks = alignment["landmarks_xy"]
        if not isinstance(landmarks, np.ndarray):
            landmarks = np.array(landmarks, dtype="float32")
        self._identity = {T.cast(T.Literal["vggface2"], k): np.array(v, dtype="float32")
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
        logger.trace("Created from alignment: (left: %s, width: %s, "  # type:ignore[attr-defined]
                     "top: %s, height: %s, landmarks: %s, mask: %s)",
                     self.left, self.width, self.top, self.height, self.landmarks_xy, self.mask)

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
            self._identity[T.cast(T.Literal["vggface2"], key)] = np.array(val, dtype="float32")
        logger.trace("Created from png exif header: (left: %s, "  # type:ignore[attr-defined]
                     "width: %s, top: %s  height: %s, landmarks: %s, mask: %s, identity: %s)",
                     self.left, self.width, self.top, self.height, self.landmarks_xy, self.mask,
                     {k: v.shape for k, v in self._identity.items()})

    def _image_to_face(self, image: np.ndarray) -> None:
        """ set self.image to be the cropped face from detected bounding box """
        logger.trace("Cropping face from image")  # type:ignore[attr-defined]
        self.image = image[self.top: self.bottom,
                           self.left: self.right]

    # <<< Aligned Face methods and properties >>> #
    def load_aligned(self,
                     image: np.ndarray | None,
                     size: int = 256,
                     dtype: str | None = None,
                     centering: CenteringType = "head",
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
            logger.trace("Skipping alignment calculation for already "  # type:ignore[attr-defined]
                         "aligned face")
        else:
            logger.trace("Loading aligned face: (size: %s, "  # type:ignore[attr-defined]
                         "dtype: %s)", size, dtype)
            self._aligned = AlignedFace(self.landmarks_xy,
                                        image=image,
                                        centering=centering,
                                        size=size,
                                        coverage_ratio=coverage_ratio,
                                        dtype=dtype,
                                        is_aligned=is_aligned,
                                        is_legacy=is_aligned and is_legacy)


_HASHES_SEEN: dict[str, dict[str, int]] = {}


def update_legacy_png_header(filename: str, alignments: Alignments
                             ) -> PNGHeaderDict | None:
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
                            source_is_video=False,  # Can't check so set false
                            source_frame_dims=None))

    out_filename = f"{os.path.splitext(filename)[0]}.png"  # Make sure saved file is png
    out_image = encode_image(in_image, ".png", metadata=meta)

    with open(out_filename, "wb") as out_file:
        out_file.write(out_image)

    if filename != out_filename:  # Remove the old non-png:
        logger.debug("Removing replaced face with deprecated extension: '%s'", filename)
        os.remove(filename)

    return meta
