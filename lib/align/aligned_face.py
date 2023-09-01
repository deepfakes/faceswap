#!/usr/bin/env python3
""" Aligner for faceswap.py """

from dataclasses import dataclass, field
import logging
import typing as T
from threading import Lock

import cv2
import numpy as np

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
CenteringType = T.Literal["face", "head", "legacy"]

_MEAN_FACE = np.array([[0.010086, 0.106454], [0.085135, 0.038915], [0.191003, 0.018748],
                       [0.300643, 0.034489], [0.403270, 0.077391], [0.596729, 0.077391],
                       [0.699356, 0.034489], [0.808997, 0.018748], [0.914864, 0.038915],
                       [0.989913, 0.106454], [0.500000, 0.203352], [0.500000, 0.307009],
                       [0.500000, 0.409805], [0.500000, 0.515625], [0.376753, 0.587326],
                       [0.435909, 0.609345], [0.500000, 0.628106], [0.564090, 0.609345],
                       [0.623246, 0.587326], [0.131610, 0.216423], [0.196995, 0.178758],
                       [0.275698, 0.179852], [0.344479, 0.231733], [0.270791, 0.245099],
                       [0.192616, 0.244077], [0.655520, 0.231733], [0.724301, 0.179852],
                       [0.803005, 0.178758], [0.868389, 0.216423], [0.807383, 0.244077],
                       [0.729208, 0.245099], [0.264022, 0.780233], [0.350858, 0.745405],
                       [0.438731, 0.727388], [0.500000, 0.742578], [0.561268, 0.727388],
                       [0.649141, 0.745405], [0.735977, 0.780233], [0.652032, 0.864805],
                       [0.566594, 0.902192], [0.500000, 0.909281], [0.433405, 0.902192],
                       [0.347967, 0.864805], [0.300252, 0.784792], [0.437969, 0.778746],
                       [0.500000, 0.785343], [0.562030, 0.778746], [0.699747, 0.784792],
                       [0.563237, 0.824182], [0.500000, 0.831803], [0.436763, 0.824182]])

_MEAN_FACE_3D = np.array([[4.056931, -11.432347, 1.636229],   # 8 chin LL
                          [1.833492, -12.542305, 4.061275],   # 7 chin L
                          [0.0, -12.901019, 4.070434],        # 6 chin C
                          [-1.833492, -12.542305, 4.061275],  # 5 chin R
                          [-4.056931, -11.432347, 1.636229],  # 4 chin RR
                          [6.825897, 1.275284, 4.402142],     # 33 L eyebrow L
                          [1.330353, 1.636816, 6.903745],     # 29 L eyebrow R
                          [-1.330353, 1.636816, 6.903745],    # 34 R eyebrow L
                          [-6.825897, 1.275284, 4.402142],    # 38 R eyebrow R
                          [1.930245, -5.060977, 5.914376],    # 54 nose LL
                          [0.746313, -5.136947, 6.263227],    # 53 nose L
                          [0.0, -5.485328, 6.76343],          # 52 nose C
                          [-0.746313, -5.136947, 6.263227],   # 51 nose R
                          [-1.930245, -5.060977, 5.914376],   # 50 nose RR
                          [5.311432, 0.0, 3.987654],          # 13 L eye L
                          [1.78993, -0.091703, 4.413414],     # 17 L eye R
                          [-1.78993, -0.091703, 4.413414],    # 25 R eye L
                          [-5.311432, 0.0, 3.987654],         # 21 R eye R
                          [2.774015, -7.566103, 5.048531],    # 43 mouth L
                          [0.509714, -7.056507, 6.566167],    # 42 mouth top L
                          [0.0, -7.131772, 6.704956],         # 41 mouth top C
                          [-0.509714, -7.056507, 6.566167],   # 40 mouth top R
                          [-2.774015, -7.566103, 5.048531],   # 39 mouth R
                          [-0.589441, -8.443925, 6.109526],   # 46 mouth bottom R
                          [0.0, -8.601736, 6.097667],         # 45 mouth bottom C
                          [0.589441, -8.443925, 6.109526]])   # 44 mouth bottom L

_EXTRACT_RATIOS = {"legacy": 0.375, "face": 0.5, "head": 0.625}


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
    logger.trace("interpolator: %s, inverse interpolator: %s",  # type: ignore
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
    logger.trace("image shape: %s, matrix: %s, size: %s. padding: %s",  # type: ignore
                 image.shape, matrix, size, padding)
    # transform the matrix for size and padding
    mat = matrix * (size - 2 * padding)
    mat[:, 2] += padding

    # transform image
    interpolators = get_matrix_scaling(mat)
    retval = cv2.warpAffine(image, mat, (size, size), flags=interpolators[0])
    logger.trace("transformed matrix: %s, final image shape: %s",  # type: ignore
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
    source_size = image_size - (image_size * _EXTRACT_RATIOS[source_centering])
    offset = target_offset - source_offset
    offset *= source_size
    center = np.rint(offset + image_size / 2).astype("int32")
    logger.trace("image_size: %s, source_offset: %s, target_offset: %s, "  # type: ignore
                 "source_centering: '%s', adjusted_offset: %s, center: %s", image_size,
                 source_offset, target_offset, source_centering, offset, center)
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
        src_size = size - (size * _EXTRACT_RATIOS[source_centering])
        retval = 2 * int(np.rint((src_size / (1 - _EXTRACT_RATIOS[target_centering])
                                 * coverage_ratio) / 2))
    logger.trace("source_centering: %s, target_centering: %s, size: %s, "  # type: ignore
                 "coverage_ratio: %s, source_size: %s, crop_size: %s", source_centering,
                 target_centering, size, coverage_ratio, src_size, retval)
    return retval


class PoseEstimate():
    """ Estimates pose from a generic 3D head model for the given 2D face landmarks.

    Parameters
    ----------
    landmarks: :class:`numpy.ndarry`
        The original 68 point landmarks aligned to 0.0 - 1.0 range

    References
    ----------
    Head Pose Estimation using OpenCV and Dlib - https://www.learnopencv.com/tag/solvepnp/
    3D Model points - http://aifi.isr.uc.pt/Downloads/OpenGL/glAnthropometric3DModel.cpp
    """
    def __init__(self, landmarks: np.ndarray) -> None:
        self._distortion_coefficients = np.zeros((4, 1))  # Assuming no lens distortion
        self._xyz_2d: np.ndarray | None = None

        self._camera_matrix = self._get_camera_matrix()
        self._rotation, self._translation = self._solve_pnp(landmarks)
        self._offset = self._get_offset()
        self._pitch_yaw_roll: tuple[float, float, float] = (0, 0, 0)

    @property
    def xyz_2d(self) -> np.ndarray:
        """ :class:`numpy.ndarray` projected (x, y) coordinates for each x, y, z point at a
        constant distance from adjusted center of the skull (0.5, 0.5) in the 2D space. """
        if self._xyz_2d is None:
            xyz = cv2.projectPoints(np.array([[6., 0., -2.3],
                                              [0., 6., -2.3],
                                              [0., 0., 3.7]]).astype("float32"),
                                    self._rotation,
                                    self._translation,
                                    self._camera_matrix,
                                    self._distortion_coefficients)[0].squeeze()
            self._xyz_2d = xyz - self._offset["head"]
        return self._xyz_2d

    @property
    def offset(self) -> dict[CenteringType, np.ndarray]:
        """ dict: The amount to offset a standard 0.0 - 1.0 umeyama transformation matrix for a
        from the center of the face (between the eyes) or center of the head (middle of skull)
        rather than the nose area. """
        return self._offset

    @property
    def pitch(self) -> float:
        """ float: The pitch of the aligned face in eular angles """
        if not any(self._pitch_yaw_roll):
            self._get_pitch_yaw_roll()
        return self._pitch_yaw_roll[0]

    @property
    def yaw(self) -> float:
        """ float: The yaw of the aligned face in eular angles """
        if not any(self._pitch_yaw_roll):
            self._get_pitch_yaw_roll()
        return self._pitch_yaw_roll[1]

    @property
    def roll(self) -> float:
        """ float: The roll of the aligned face in eular angles """
        if not any(self._pitch_yaw_roll):
            self._get_pitch_yaw_roll()
        return self._pitch_yaw_roll[2]

    def _get_pitch_yaw_roll(self) -> None:
        """ Obtain the yaw, roll and pitch from the :attr:`_rotation` in eular angles. """
        proj_matrix = np.zeros((3, 4), dtype="float32")
        proj_matrix[:3, :3] = cv2.Rodrigues(self._rotation)[0]
        euler = cv2.decomposeProjectionMatrix(proj_matrix)[-1]
        self._pitch_yaw_roll = T.cast(tuple[float, float, float], tuple(euler.squeeze()))
        logger.trace("yaw_pitch: %s", self._pitch_yaw_roll)  # type: ignore

    @classmethod
    def _get_camera_matrix(cls) -> np.ndarray:
        """ Obtain an estimate of the camera matrix based off the original frame dimensions.

        Returns
        -------
        :class:`numpy.ndarray`
            An estimated camera matrix
        """
        focal_length = 4
        camera_matrix = np.array([[focal_length, 0, 0.5],
                                  [0, focal_length, 0.5],
                                  [0, 0, 1]], dtype="double")
        logger.trace("camera_matrix: %s", camera_matrix)  # type: ignore
        return camera_matrix

    def _solve_pnp(self, landmarks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """ Solve the Perspective-n-Point for the given landmarks.

        Takes 2D landmarks in world space and estimates the rotation and translation vectors
        in 3D space.

        Parameters
        ----------
        landmarks: :class:`numpy.ndarry`
            The original 68 point landmark co-ordinates relating to the original frame

        Returns
        -------
        rotation: :class:`numpy.ndarray`
            The solved rotation vector
        translation: :class:`numpy.ndarray`
            The solved translation vector
        """
        points = landmarks[[6, 7, 8, 9, 10, 17, 21, 22, 26, 31, 32, 33, 34,
                            35, 36, 39, 42, 45, 48, 50, 51, 52, 54, 56, 57, 58]]
        _, rotation, translation = cv2.solvePnP(_MEAN_FACE_3D,
                                                points,
                                                self._camera_matrix,
                                                self._distortion_coefficients,
                                                flags=cv2.SOLVEPNP_ITERATIVE)
        logger.trace("points: %s, rotation: %s, translation: %s",  # type: ignore
                     points, rotation, translation)
        return rotation, translation

    def _get_offset(self) -> dict[CenteringType, np.ndarray]:
        """ Obtain the offset between the original center of the extracted face to the new center
        of the head in 2D space.

        Returns
        -------
        :class:`numpy.ndarray`
            The x, y offset of the new center from the old center.
        """
        offset: dict[CenteringType, np.ndarray] = {"legacy": np.array([0.0, 0.0])}
        points: dict[T.Literal["face", "head"], tuple[float, ...]] = {"head": (0.0, 0.0, -2.3),
                                                                      "face": (0.0, -1.5, 4.2)}

        for key, pnts in points.items():
            center = cv2.projectPoints(np.array([pnts]).astype("float32"),
                                       self._rotation,
                                       self._translation,
                                       self._camera_matrix,
                                       self._distortion_coefficients)[0].squeeze()
            logger.trace("center %s: %s", key, center)  # type: ignore
            offset[key] = center - (0.5, 0.5)
        logger.trace("offset: %s", offset)  # type: ignore
        return offset


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
        logger.trace("Initializing: %s (image shape: %s, centering: '%s', "  # type: ignore
                     "size: %s, coverage_ratio: %s, dtype: %s, is_aligned: %s, is_legacy: %s)",
                     self.__class__.__name__, image if image is None else image.shape,
                     centering, size, coverage_ratio, dtype, is_aligned, is_legacy)
        self._frame_landmarks = landmarks
        self._centering = centering
        self._size = size
        self._coverage_ratio = coverage_ratio
        self._dtype = dtype
        self._is_aligned = is_aligned
        self._source_centering: CenteringType = "legacy" if is_legacy and is_aligned else "head"
        self._matrices = {"legacy": _umeyama(landmarks[17:], _MEAN_FACE, True)[0:2],
                          "face": np.array([]),
                          "head": np.array([])}
        self._padding = self._padding_from_coverage(size, coverage_ratio)

        self._cache = _FaceCache()

        self._face = self.extract_face(image)
        logger.trace("Initialized: %s (matrix: %s, padding: %s, face shape: %s)",  # type: ignore
                     self.__class__.__name__, self._matrices["legacy"], self._padding,
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
        if not np.any(self._matrices[self._centering]):
            matrix = self._matrices["legacy"].copy()
            matrix[:, 2] -= self.pose.offset[self._centering]
            self._matrices[self._centering] = matrix
            logger.trace("original matrix: %s, new matrix: %s",  # type: ignore
                         self._matrices["legacy"], matrix)
        return self._matrices[self._centering]

    @property
    def pose(self) -> PoseEstimate:
        """ :class:`lib.align.PoseEstimate`: The estimated pose in 3D space. """
        with self._cache.lock("pose"):
            if self._cache.pose is None:
                lms = np.nan_to_num(cv2.transform(np.expand_dims(self._frame_landmarks, axis=1),
                                    self._matrices["legacy"]).squeeze())
                self._cache.pose = PoseEstimate(lms)
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
                logger.trace("adjusted_matrix: %s", mat)  # type: ignore
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
                logger.trace("original roi: %s", roi)  # type: ignore
                self._cache.original_roi = roi
        return self._cache.original_roi

    @property
    def landmarks(self) -> np.ndarray:
        """ :class:`numpy.ndarray`: The 68 point facial landmarks aligned to the extracted face
        box. """
        with self._cache.lock("landmarks"):
            if self._cache.landmarks is None:
                lms = self.transform_points(self._frame_landmarks)
                logger.trace("aligned landmarks: %s", lms)  # type: ignore
                self._cache.landmarks = lms
            return self._cache.landmarks

    @property
    def normalized_landmarks(self) -> np.ndarray:
        """ :class:`numpy.ndarray`: The 68 point facial landmarks normalized to 0.0 - 1.0 as
        aligned by Umeyama. """
        with self._cache.lock("landmarks_normalized"):
            if self._cache.landmarks_normalized is None:
                lms = np.expand_dims(self._frame_landmarks, axis=1)
                lms = cv2.transform(lms, self._matrices["legacy"], lms.shape).squeeze()
                logger.trace("normalized landmarks: %s", lms)  # type: ignore
                self._cache.landmarks_normalized = lms
        return self._cache.landmarks_normalized

    @property
    def interpolators(self) -> tuple[int, int]:
        """ tuple: (`interpolator` and `reverse interpolator`) for the :attr:`adjusted matrix`. """
        with self._cache.lock("interpolators"):
            if not any(self._cache.interpolators):
                interpolators = get_matrix_scaling(self.adjusted_matrix)
                logger.trace("interpolators: %s", interpolators)  # type: ignore
                self._cache.interpolators = interpolators
        return self._cache.interpolators

    @property
    def average_distance(self) -> float:
        """ float: The average distance of the core landmarks (18-67) from the mean face that was
        used for aligning the image. """
        with self._cache.lock("average_distance"):
            if not self._cache.average_distance:
                average_distance = np.mean(np.abs(self.normalized_landmarks[17:] - _MEAN_FACE))
                logger.trace("average_distance: %s", average_distance)  # type: ignore
                self._cache.average_distance = average_distance
        return self._cache.average_distance

    @property
    def relative_eye_mouth_position(self) -> float:
        """ float: Value representing the relative position of the lowest eye/eye-brow point to the
        highest mouth point. Positive values indicate that eyes/eyebrows are aligned above the
        mouth, negative values indicate that eyes/eyebrows are misaligned below the mouth. """
        with self._cache.lock("relative_eye_mouth_position"):
            if not self._cache.relative_eye_mouth_position:
                lowest_eyes = np.max(self.normalized_landmarks[np.r_[17:27, 36:48], 1])
                highest_mouth = np.min(self.normalized_landmarks[48:68, 1])
                position = highest_mouth - lowest_eyes
                logger.trace("lowest_eyes: %s, highest_mouth: %s, "  # type: ignore
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
        retval = {_type: round((size * (coverage_ratio - (1 - _EXTRACT_RATIOS[_type]))) / 2)
                  for _type in T.get_args(T.Literal["legacy", "face", "head"])}
        logger.trace(retval)  # type: ignore
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
        retval = cv2.transform(retval, mat, retval.shape).squeeze()
        logger.trace("invert: %s, Original points: %s, transformed points: %s",  # type: ignore
                     invert, points, retval)
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
            logger.trace("_extract_face called without a loaded image. "  # type: ignore
                         "Returning empty face.")
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
        logger.trace("image_size: %s, target_size: %s, coverage_ratio: %s",  # type: ignore
                     image.shape[0], self.size, self._coverage_ratio)

        img_size = image.shape[0]
        target_size = get_centered_size(self._source_centering,
                                        self._centering,
                                        img_size,
                                        self._coverage_ratio)
        out = np.zeros((target_size, target_size, image.shape[-1]), dtype=image.dtype)

        slices = self._get_cropped_slices(img_size, target_size)
        out[slices["out"][0], slices["out"][1], :] = image[slices["in"][0], slices["in"][1], :]
        logger.trace("Cropped from aligned extract: (centering: %s, in shape: %s, "  # type: ignore
                     "out shape: %s)", self._centering, image.shape, out.shape)
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
                logger.trace("centering: %s, cropped_slices: %s",  # type: ignore
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
                logger.trace("centering: '%s', center: %s, padding: %s, "  # type: ignore
                             "sub roi: %s", centering, center, padding, roi)
                self._cache.cropped_roi[centering] = roi
        return self._cache.cropped_roi[centering]


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
