#!/usr/bin/env python3
""" Aligner for faceswap.py """
from __future__ import annotations

from dataclasses import dataclass, field
import logging
import typing as T

from threading import Lock

import cv2
import numpy as np

from lib.logger import parse_class_init

from .constants import CenteringType, EXTRACT_RATIOS, LandmarkType, _MEAN_FACE
from .pose import PoseEstimate

logger = logging.getLogger(__name__)


def get_matrix_scaling(matrix: np.ndarray) -> tuple[int, int]:
    """ Given a matrix, return the cv2 Interpolation method and inverse interpolation method for
    applying the matrix on an image.

    Parameters
    ----------
    matrix: :class:`numpy.ndarray`
        The transform matrix to return the interpolator for

    Returns
    -------
    tuple
        The interpolator and inverse interpolator for the given matrix. This will be (Cubic, Area)
        for an upscale matrix and (Area, Cubic) for a downscale matrix
    """
    x_scale = np.sqrt(matrix[0, 0] * matrix[0, 0] + matrix[0, 1] * matrix[0, 1])
    y_scale = (matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0]) / x_scale
    avg_scale = (x_scale + y_scale) * 0.5
    if avg_scale >= 1.:
        interpolators = cv2.INTER_CUBIC, cv2.INTER_AREA
    else:
        interpolators = cv2.INTER_AREA, cv2.INTER_CUBIC
    logger.trace("interpolator: %s, inverse interpolator: %s",  # type:ignore[attr-defined]
                 interpolators[0], interpolators[1])
    return interpolators


def transform_image(image: np.ndarray,
                    matrix: np.ndarray,
                    size: int,
                    padding: int = 0) -> np.ndarray:
    """ Perform transformation on an image, applying the given size and padding to the matrix.

    Parameters
    ----------
    image: :class:`numpy.ndarray`
        The image to transform
    matrix: :class:`numpy.ndarray`
        The transformation matrix to apply to the image
    size: int
        The final size of the transformed image
    padding: int, optional
        The amount of padding to apply to the final image. Default: `0`

    Returns
    -------
    :class:`numpy.ndarray`
        The transformed image
    """
    logger.trace("image shape: %s, matrix: %s, size: %s. padding: %s",  # type:ignore[attr-defined]
                 image.shape, matrix, size, padding)
    # transform the matrix for size and padding
    mat = matrix * (size - 2 * padding)
    mat[:, 2] += padding

    # transform image
    interpolators = get_matrix_scaling(mat)
    retval = cv2.warpAffine(image, mat, (size, size), flags=interpolators[0])
    logger.trace("transformed matrix: %s, final image shape: %s",  # type:ignore[attr-defined]
                 mat, image.shape)
    return retval


def get_adjusted_center(image_size: int,
                        source_offset: np.ndarray,
                        target_offset: np.ndarray,
                        source_centering: CenteringType) -> np.ndarray:
    """ Obtain the correct center of a face extracted image to translate between two different
    extract centerings.

    Parameters
    ----------
    image_size: int
        The size of the image at the given :attr:`source_centering`
    source_offset: :class:`numpy.ndarray`
        The pose offset to translate a base extracted face to source centering
    target_offset: :class:`numpy.ndarray`
        The pose offset to translate a base extracted face to target centering
    source_centering: ["face", "head", "legacy"]
        The centering of the source image

    Returns
    -------
    :class:`numpy.ndarray`
        The center point of the image at the given size for the target centering
    """
    source_size = image_size - (image_size * EXTRACT_RATIOS[source_centering])
    offset = target_offset - source_offset
    offset *= source_size
    center = np.rint(offset + image_size / 2).astype("int32")
    logger.trace(  # type:ignore[attr-defined]
        "image_size: %s, source_offset: %s, target_offset: %s, source_centering: '%s', "
        "adjusted_offset: %s, center: %s",
        image_size, source_offset, target_offset, source_centering, offset, center)
    return center


def get_centered_size(source_centering: CenteringType,
                      target_centering: CenteringType,
                      size: int,
                      coverage_ratio: float = 1.0) -> int:
    """ Obtain the size of a cropped face from an aligned image.

    Given an image of a certain dimensions, returns the dimensions of the sub-crop within that
    image for the requested centering at the requested coverage ratio

    Notes
    -----
    `"legacy"` places the nose in the center of the image (the original method for aligning).
    `"face"` aligns for the nose to be in the center of the face (top to bottom) but the center
    of the skull for left to right. `"head"` places the center in the middle of the skull in 3D
    space.

    The ROI in relation to the source image is calculated by rounding the padding of one side
    to the nearest integer then applying this padding to the center of the crop, to ensure that
    any dimensions always have an even number of pixels.

    Parameters
    ----------
    source_centering: ["head", "face", "legacy"]
        The centering that the original image is aligned at
    target_centering: ["head", "face", "legacy"]
        The centering that the sub-crop size should be obtained for
    size: int
        The size of the source image to obtain the cropped size for
    coverage_ratio: float, optional
        The coverage ratio to be applied to the target image. Default: `1.0`

    Returns
    -------
    int
        The pixel size of a sub-crop image from a full head aligned image with the given coverage
        ratio
    """
    if source_centering == target_centering and coverage_ratio == 1.0:
        retval = size
    else:
        src_size = size - (size * EXTRACT_RATIOS[source_centering])
        retval = 2 * int(np.rint((src_size / (1 - EXTRACT_RATIOS[target_centering])
                                 * coverage_ratio) / 2))
    logger.trace(  # type:ignore[attr-defined]
        "source_centering: %s, target_centering: %s, size: %s, coverage_ratio: %s, "
        "source_size: %s, crop_size: %s",
        source_centering, target_centering, size, coverage_ratio, src_size, retval)
    return retval


@dataclass
class _FaceCache:  # pylint:disable=too-many-instance-attributes
    """ Cache for storing items related to a single aligned face.

    Items are cached so that they are only created the first time they are called.
    Each item includes a threading lock to make cache creation thread safe.

    Parameters
    ----------
    pose: :class:`lib.align.PoseEstimate`, optional
        The estimated pose in 3D space. Default: ``None``
    original_roi: :class:`numpy.ndarray`, optional
        The location of the extracted face box within the original frame. Default: ``None``
    landmarks: :class:`numpy.ndarray`, optional
        The 68 point facial landmarks aligned to the extracted face box. Default: ``None``
    landmarks_normalized: :class:`numpy.ndarray`:
        The 68 point facial landmarks normalized to 0.0 - 1.0 as aligned by Umeyama.
        Default: ``None``
    average_distance: float, optional
        The average distance of the core landmarks (18-67) from the mean face that was used for
        aligning the image.  Default: `0.0`
    relative_eye_mouth_position: float, optional
        A float value representing the relative position of the lowest eye/eye-brow point to the
        highest mouth point. Positive values indicate that eyes/eyebrows are aligned above the
        mouth, negative values indicate that eyes/eyebrows are misaligned below the mouth.
        Default: `0.0`
    adjusted_matrix: :class:`numpy.ndarray`, optional
        The 3x2 transformation matrix for extracting and aligning the core face area out of the
        original frame with padding and sizing applied. Default: ``None``
    interpolators: tuple, optional
        (`interpolator` and `reverse interpolator`) for the :attr:`adjusted matrix`.
        Default: `(0, 0)`
    cropped_roi, dict, optional
        The (`left`, `top`, `right`, `bottom` location of the region of interest within an
            aligned face centered for each centering. Default: `{}`
    cropped_slices: dict, optional
        The slices for an input full head image and output cropped image. Default: `{}`
    """
    pose: PoseEstimate | None = None
    original_roi: np.ndarray | None = None
    landmarks: np.ndarray | None = None
    landmarks_normalized: np.ndarray | None = None
    average_distance: float = 0.0
    relative_eye_mouth_position: float = 0.0
    adjusted_matrix: np.ndarray | None = None
    interpolators: tuple[int, int] = (0, 0)
    cropped_roi: dict[CenteringType, np.ndarray] = field(default_factory=dict)
    cropped_slices: dict[CenteringType, dict[T.Literal["in", "out"],
                                             tuple[slice, slice]]] = field(default_factory=dict)

    _locks: dict[str, Lock] = field(default_factory=dict)

    def __post_init__(self):
        """ Initialize the locks for the class parameters """
        self._locks = {name: Lock() for name in self.__dict__}

    def lock(self, name: str) -> Lock:
        """ Obtain the lock for the given property

        Parameters
        ----------
        name: str
            The name of a parameter within the cache

        Returns
        -------
        :class:`threading.Lock`
            The lock associated with the requested parameter
        """
        return self._locks[name]


class AlignedFace():
    """ Class to align a face.

    Holds the aligned landmarks and face image, as well as associated matrices and information
    about an aligned face.

    Parameters
    ----------
    landmarks: :class:`numpy.ndarray`
        The original 68 point landmarks that pertain to the given image for this face
    image: :class:`numpy.ndarray`, optional
        The original frame that contains the face that is to be aligned. Pass `None` if the aligned
        face is not to be generated, and just the co-ordinates should be calculated.
    centering: ["legacy", "face", "head"], optional
        The type of extracted face that should be loaded. "legacy" places the nose in the center of
        the image (the original method for aligning). "face" aligns for the nose to be in the
        center of the face (top to bottom) but the center of the skull for left to right. "head"
        aligns for the center of the skull (in 3D space) being the center of the extracted image,
        with the crop holding the full head. Default: `"face"`
    size: int, optional
        The size in pixels, of each edge of the final aligned face. Default: `64`
    coverage_ratio: float, optional
        The amount of the aligned image to return. A ratio of 1.0 will return the full contents of
        the aligned image. A ratio of 0.5 will return an image of the given size, but will crop to
        the central 50%% of the image.
    dtype: str, optional
        Set a data type for the final face to be returned as. Passing ``None`` will return a face
        with the same data type as the original :attr:`image`. Default: ``None``
    is_aligned_face: bool, optional
        Indicates that the :attr:`image` is an aligned face rather than a frame.
        Default: ``False``
    is_legacy: bool, optional
        Only used if `is_aligned` is ``True``. ``True`` indicates that the aligned image being
        loaded is a legacy extracted face rather than a current head extracted face
    """
    def __init__(self,
                 landmarks: np.ndarray,
                 image: np.ndarray | None = None,
                 centering: CenteringType = "face",
                 size: int = 64,
                 coverage_ratio: float = 1.0,
                 dtype: str | None = None,
                 is_aligned: bool = False,
                 is_legacy: bool = False) -> None:
        logger.trace(parse_class_init(locals()))  # type:ignore[attr-defined]
        self._frame_landmarks = landmarks
        self._landmark_type = LandmarkType.from_shape(landmarks.shape)
        self._centering = centering
        self._size = size
        self._coverage_ratio = coverage_ratio
        self._dtype = dtype
        self._is_aligned = is_aligned
        self._source_centering: CenteringType = "legacy" if is_legacy and is_aligned else "head"
        self._padding = self._padding_from_coverage(size, coverage_ratio)

        lookup = self._landmark_type
        self._mean_lookup = LandmarkType.LM_2D_51 if lookup == LandmarkType.LM_2D_68 else lookup

        self._cache = _FaceCache()
        self._matrices: dict[CenteringType, np.ndarray] = {"legacy": self._get_default_matrix()}

        self._face = self.extract_face(image)
        logger.trace("Initialized: %s (padding: %s, face shape: %s)",  # type:ignore[attr-defined]
                     self.__class__.__name__, self._padding,
                     self._face if self._face is None else self._face.shape)

    @property
    def centering(self) -> T.Literal["legacy", "head", "face"]:
        """ str: The centering of the Aligned Face. One of `"legacy"`, `"head"`, `"face"`. """
        return self._centering

    @property
    def size(self) -> int:
        """ int: The size (in pixels) of one side of the square extracted face image. """
        return self._size

    @property
    def padding(self) -> int:
        """ int: The amount of padding (in pixels) that is applied to each side of the
        extracted face image for the selected extract type. """
        return self._padding[self._centering]

    @property
    def matrix(self) -> np.ndarray:
        """ :class:`numpy.ndarray`: The 3x2 transformation matrix for extracting and aligning the
        core face area out of the original frame, with no padding or sizing applied. The returned
        matrix is offset for the given :attr:`centering`. """
        if self._centering not in self._matrices:
            matrix = self._matrices["legacy"].copy()
            matrix[:, 2] -= self.pose.offset[self._centering]
            self._matrices[self._centering] = matrix
            logger.trace("original matrix: %s, new matrix: %s",  # type:ignore[attr-defined]
                         self._matrices["legacy"], matrix)
        return self._matrices[self._centering]

    @property
    def pose(self) -> PoseEstimate:
        """ :class:`lib.align.PoseEstimate`: The estimated pose in 3D space. """
        with self._cache.lock("pose"):
            if self._cache.pose is None:
                lms = np.nan_to_num(cv2.transform(np.expand_dims(self._frame_landmarks, axis=1),
                                    self._matrices["legacy"]).squeeze())
                self._cache.pose = PoseEstimate(lms, self._landmark_type)
        return self._cache.pose

    @property
    def adjusted_matrix(self) -> np.ndarray:
        """ :class:`numpy.ndarray`: The 3x2 transformation matrix for extracting and aligning the
        core face area out of the original frame with padding and sizing applied. """
        with self._cache.lock("adjusted_matrix"):
            if self._cache.adjusted_matrix is None:
                matrix = self.matrix.copy()
                mat = matrix * (self._size - 2 * self.padding)
                mat[:, 2] += self.padding
                logger.trace("adjusted_matrix: %s", mat)  # type:ignore[attr-defined]
                self._cache.adjusted_matrix = mat
        return self._cache.adjusted_matrix

    @property
    def face(self) -> np.ndarray | None:
        """ :class:`numpy.ndarray`: The aligned face at the given :attr:`size` at the specified
        :attr:`coverage` in the given :attr:`dtype`. If an :attr:`image` has not been provided
        then an the attribute will return ``None``. """
        return self._face

    @property
    def original_roi(self) -> np.ndarray:
        """ :class:`numpy.ndarray`: The location of the extracted face box within the original
        frame. """
        with self._cache.lock("original_roi"):
            if self._cache.original_roi is None:
                roi = np.array([[0, 0],
                                [0, self._size - 1],
                                [self._size - 1, self._size - 1],
                                [self._size - 1, 0]])
                roi = np.rint(self.transform_points(roi, invert=True)).astype("int32")
                logger.trace("original roi: %s", roi)  # type:ignore[attr-defined]
                self._cache.original_roi = roi
        return self._cache.original_roi

    @property
    def landmarks(self) -> np.ndarray:
        """ :class:`numpy.ndarray`: The 68 point facial landmarks aligned to the extracted face
        box. """
        with self._cache.lock("landmarks"):
            if self._cache.landmarks is None:
                lms = self.transform_points(self._frame_landmarks)
                logger.trace("aligned landmarks: %s", lms)  # type:ignore[attr-defined]
                self._cache.landmarks = lms
            return self._cache.landmarks

    @property
    def landmark_type(self) -> LandmarkType:
        """:class:`~LandmarkType`: The type of landmarks that generated this aligned face """
        return self._landmark_type

    @property
    def normalized_landmarks(self) -> np.ndarray:
        """ :class:`numpy.ndarray`: The 68 point facial landmarks normalized to 0.0 - 1.0 as
        aligned by Umeyama. """
        with self._cache.lock("landmarks_normalized"):
            if self._cache.landmarks_normalized is None:
                lms = np.expand_dims(self._frame_landmarks, axis=1)
                lms = cv2.transform(lms, self._matrices["legacy"]).squeeze()
                logger.trace("normalized landmarks: %s", lms)  # type:ignore[attr-defined]
                self._cache.landmarks_normalized = lms
        return self._cache.landmarks_normalized

    @property
    def interpolators(self) -> tuple[int, int]:
        """ tuple: (`interpolator` and `reverse interpolator`) for the :attr:`adjusted matrix`. """
        with self._cache.lock("interpolators"):
            if not any(self._cache.interpolators):
                interpolators = get_matrix_scaling(self.adjusted_matrix)
                logger.trace("interpolators: %s", interpolators)  # type:ignore[attr-defined]
                self._cache.interpolators = interpolators
        return self._cache.interpolators

    @property
    def average_distance(self) -> float:
        """ float: The average distance of the core landmarks (18-67) from the mean face that was
        used for aligning the image. """
        with self._cache.lock("average_distance"):
            if not self._cache.average_distance:
                mean_face = _MEAN_FACE[self._mean_lookup]
                lms = self.normalized_landmarks
                if self._landmark_type == LandmarkType.LM_2D_68:
                    lms = lms[17:]  # 68 point landmarks only use core face items
                average_distance = np.mean(np.abs(lms - mean_face))
                logger.trace("average_distance: %s", average_distance)  # type:ignore[attr-defined]
                self._cache.average_distance = average_distance
        return self._cache.average_distance

    @property
    def relative_eye_mouth_position(self) -> float:
        """ float: Value representing the relative position of the lowest eye/eye-brow point to the
        highest mouth point. Positive values indicate that eyes/eyebrows are aligned above the
        mouth, negative values indicate that eyes/eyebrows are misaligned below the mouth. """
        with self._cache.lock("relative_eye_mouth_position"):
            if not self._cache.relative_eye_mouth_position:
                if self._landmark_type != LandmarkType.LM_2D_68:
                    position = 1.0  # arbitrary positive value
                else:
                    lowest_eyes = np.max(self.normalized_landmarks[np.r_[17:27, 36:48], 1])
                    highest_mouth = np.min(self.normalized_landmarks[48:68, 1])
                    position = highest_mouth - lowest_eyes
                logger.trace("lowest_eyes: %s, highest_mouth: %s, "  # type:ignore[attr-defined]
                             "relative_eye_mouth_position: %s", lowest_eyes, highest_mouth,
                             position)
                self._cache.relative_eye_mouth_position = position
        return self._cache.relative_eye_mouth_position

    @classmethod
    def _padding_from_coverage(cls, size: int, coverage_ratio: float) -> dict[CenteringType, int]:
        """ Return the image padding for a face from coverage_ratio set against a
            pre-padded training image.

        Parameters
        ----------
        size: int
            The final size of the aligned image in pixels
        coverage_ratio: float
            The ratio of the final image to pad to

        Returns
        -------
        dict
            The padding required, in pixels for 'head', 'face' and 'legacy' face types
        """
        retval = {_type: round((size * (coverage_ratio - (1 - EXTRACT_RATIOS[_type]))) / 2)
                  for _type in T.get_args(T.Literal["legacy", "face", "head"])}
        logger.trace(retval)  # type:ignore[attr-defined]
        return retval

    def _get_default_matrix(self) -> np.ndarray:
        """ Get the default (legacy) matrix. All subsequent matrices are calculated from this

        Returns
        -------
        :class:`numpy.ndarray`
            The default 'legacy' matrix
        """
        lms = self._frame_landmarks
        if self._landmark_type == LandmarkType.LM_2D_68:
            lms = lms[17:]  # 68 point landmarks only use core face items
        retval = _umeyama(lms, _MEAN_FACE[self._mean_lookup], True)[0:2]
        logger.trace("Default matrix: %s", retval)  # type:ignore[attr-defined]
        return retval

    def transform_points(self, points: np.ndarray, invert: bool = False) -> np.ndarray:
        """ Perform transformation on a series of (x, y) co-ordinates in world space into
        aligned face space.

        Parameters
        ----------
        points: :class:`numpy.ndarray`
            The points to transform
        invert: bool, optional
            ``True`` to reverse the transformation (i.e. transform the points into world space from
            aligned face space). Default: ``False``

        Returns
        -------
        :class:`numpy.ndarray`
            The transformed points
        """
        retval = np.expand_dims(points, axis=1)
        mat = cv2.invertAffineTransform(self.adjusted_matrix) if invert else self.adjusted_matrix
        retval = cv2.transform(retval, mat).squeeze()
        logger.trace(  # type:ignore[attr-defined]
            "invert: %s, Original points: %s, transformed points: %s", invert, points, retval)
        return retval

    def extract_face(self, image: np.ndarray | None) -> np.ndarray | None:
        """ Extract the face from a source image and populate :attr:`face`. If an image is not
        provided then ``None`` is returned.

        Parameters
        ----------
        image: :class:`numpy.ndarray` or ``None``
            The original frame to extract the face from. ``None`` if the face should not be
            extracted

        Returns
        -------
        :class:`numpy.ndarray` or ``None``
            The extracted face at the given size, with the given coverage of the given dtype or
            ``None`` if no image has been provided.
        """
        if image is None:
            logger.trace("_extract_face called without a loaded "  # type:ignore[attr-defined]
                         "image. Returning empty face.")
            return None

        if self._is_aligned and (self._centering != self._source_centering or
                                 self._coverage_ratio != 1.0):
            # Crop out the sub face from full head
            image = self._convert_centering(image)

        if self._is_aligned and image.shape[0] != self._size:  # Resize the given aligned face
            interp = cv2.INTER_CUBIC if image.shape[0] < self._size else cv2.INTER_AREA
            retval = cv2.resize(image, (self._size, self._size), interpolation=interp)
        elif self._is_aligned:
            retval = image
        else:
            retval = transform_image(image, self.matrix, self._size, self.padding)
        retval = retval if self._dtype is None else retval.astype(self._dtype)
        return retval

    def _convert_centering(self, image: np.ndarray) -> np.ndarray:
        """ When the face being loaded is pre-aligned, the loaded image will have 'head' centering
        so it needs to be cropped out to the appropriate centering.

        This function temporarily converts this object to a full head aligned face, extracts the
        sub-cropped face to the correct centering, reverse the sub crop and returns the cropped
        face at the selected coverage ratio.

        Parameters
        ----------
        image: :class:`numpy.ndarray`
            The original head-centered aligned image

        Returns
        -------
        :class:`numpy.ndarray`
            The aligned image with the correct centering, scaled to image input size
        """
        logger.trace(  # type:ignore[attr-defined]
            "image_size: %s, target_size: %s, coverage_ratio: %s",
            image.shape[0], self.size, self._coverage_ratio)

        img_size = image.shape[0]
        target_size = get_centered_size(self._source_centering,
                                        self._centering,
                                        img_size,
                                        self._coverage_ratio)
        out = np.zeros((target_size, target_size, image.shape[-1]), dtype=image.dtype)

        slices = self._get_cropped_slices(img_size, target_size)
        out[slices["out"][0], slices["out"][1], :] = image[slices["in"][0], slices["in"][1], :]
        logger.trace(  # type:ignore[attr-defined]
            "Cropped from aligned extract: (centering: %s, in shape: %s, out shape: %s)",
            self._centering, image.shape, out.shape)
        return out

    def _get_cropped_slices(self,
                            image_size: int,
                            target_size: int,
                            ) -> dict[T.Literal["in", "out"], tuple[slice, slice]]:
        """ Obtain the slices to turn a full head extract into an alternatively centered extract.

        Parameters
        ----------
        image_size: int
            The size of the full head extracted image loaded from disk
        target_size: int
            The size of the target centered face with coverage ratio applied in relation to the
            original image size

        Returns
        -------
        dict
            The slices for an input full head image and output cropped image
        """
        with self._cache.lock("cropped_slices"):
            if not self._cache.cropped_slices.get(self._centering):
                roi = self.get_cropped_roi(image_size, target_size, self._centering)
                slice_in = (slice(max(roi[1], 0), max(roi[3], 0)),
                            slice(max(roi[0], 0), max(roi[2], 0)))
                slice_out = (slice(max(roi[1] * -1, 0),
                                   target_size - min(target_size, max(0, roi[3] - image_size))),
                             slice(max(roi[0] * -1, 0),
                                   target_size - min(target_size, max(0, roi[2] - image_size))))
                self._cache.cropped_slices[self._centering] = {"in": slice_in, "out": slice_out}
                logger.trace("centering: %s, cropped_slices: %s",  # type:ignore[attr-defined]
                             self._centering, self._cache.cropped_slices[self._centering])
        return self._cache.cropped_slices[self._centering]

    def get_cropped_roi(self,
                        image_size: int,
                        target_size: int,
                        centering: CenteringType) -> np.ndarray:
        """ Obtain the region of interest within an aligned face set to centered coverage for
        an alternative centering

        Parameters
        ----------
        image_size: int
            The size of the full head extracted image loaded from disk
        target_size: int
            The size of the target centered face with coverage ratio applied in relation to the
            original image size

        centering: ["legacy", "face"]
            The type of centering to obtain the region of interest for. "legacy" places the nose
            in the center of the image (the original method for aligning). "face" aligns for the
            nose to be in the center of the face (top to bottom) but the center of the skull for
            left to right.

        Returns
        -------
        :class:`numpy.ndarray`
            The (`left`, `top`, `right`, `bottom` location of the region of interest within an
            aligned face centered on the head for the given centering
        """
        with self._cache.lock("cropped_roi"):
            if centering not in self._cache.cropped_roi:
                center = get_adjusted_center(image_size,
                                             self.pose.offset[self._source_centering],
                                             self.pose.offset[centering],
                                             self._source_centering)
                padding = target_size // 2
                roi = np.array([center - padding, center + padding]).ravel()
                logger.trace(  # type:ignore[attr-defined]
                    "centering: '%s', center: %s, padding: %s, sub roi: %s",
                    centering, center, padding, roi)
                self._cache.cropped_roi[centering] = roi
        return self._cache.cropped_roi[centering]

    def split_mask(self) -> np.ndarray:
        """ Remove the mask from the alpha channel of :attr:`face` and return the mask

        Returns
        -------
        :class:`numpy.ndarray`
            The mask that was stored in the :attr:`face`'s alpha channel

        Raises
        ------
        AssertionError
            If :attr:`face` does not contain a mask in the alpha channel
        """
        assert self._face is not None
        assert self._face.shape[-1] == 4, "No mask stored in the alpha channel"
        mask = self._face[..., 3]
        self._face = self._face[..., :3]
        return mask


def _umeyama(source: np.ndarray, destination: np.ndarray, estimate_scale: bool) -> np.ndarray:
    """Estimate N-D similarity transformation with or without scaling.

    Imported, and slightly adapted, directly from:
    https://github.com/scikit-image/scikit-image/blob/master/skimage/transform/_geometric.py


    Parameters
    ----------
    source: :class:`numpy.ndarray`
        (M, N) array source coordinates.
    destination: :class:`numpy.ndarray`
        (M, N) array destination coordinates.
    estimate_scale: bool
        Whether to estimate scaling factor.

    Returns
    -------
    :class:`numpy.ndarray`
        (N + 1, N + 1) The homogeneous similarity transformation matrix. The matrix contains
        NaN values only if the problem is not well-conditioned.

    References
    ----------
    .. [1] "Least-squares estimation of transformation parameters between two
            point patterns", Shinji Umeyama, PAMI 1991, :DOI:`10.1109/34.88573`
    """
    # pylint:disable=invalid-name,too-many-locals
    num = source.shape[0]
    dim = source.shape[1]

    # Compute mean of source and destination.
    src_mean = source.mean(axis=0)
    dst_mean = destination.mean(axis=0)

    # Subtract mean from source and destination.
    src_demean = source - src_mean
    dst_demean = destination - dst_mean

    # Eq. (38).
    A = dst_demean.T @ src_demean / num

    # Eq. (39).
    d = np.ones((dim,), dtype=np.double)
    if np.linalg.det(A) < 0:
        d[dim - 1] = -1

    retval = np.eye(dim + 1, dtype=np.double)

    U, S, V = np.linalg.svd(A)

    # Eq. (40) and (43).
    rank = np.linalg.matrix_rank(A)
    if rank == 0:
        return np.nan * retval
    if rank == dim - 1:
        if np.linalg.det(U) * np.linalg.det(V) > 0:
            retval[:dim, :dim] = U @ V
        else:
            s = d[dim - 1]
            d[dim - 1] = -1
            retval[:dim, :dim] = U @ np.diag(d) @ V
            d[dim - 1] = s
    else:
        retval[:dim, :dim] = U @ np.diag(d) @ V

    if estimate_scale:
        # Eq. (41) and (42).
        scale = 1.0 / src_demean.var(axis=0).sum() * (S @ d)
    else:
        scale = 1.0

    retval[:dim, dim] = dst_mean - scale * (retval[:dim, :dim] @ src_mean.T)
    retval[:dim, :dim] *= scale

    return retval
