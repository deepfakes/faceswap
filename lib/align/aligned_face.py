#!/usr/bin/env python3
""" Aligner for faceswap.py """

import logging
from threading import Lock

import cv2
import numpy as np
from skimage.transform._geometric import _umeyama as umeyama

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


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


def get_matrix_scaling(matrix):
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
    logger.trace("interpolator: %s, inverse interpolator: %s", interpolators[0], interpolators[1])
    return interpolators


def transform_image(image, matrix, size, padding=0):
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
    logger.trace("image shape: %s, matrix: %s, size: %s. padding: %s",
                 image.shape, matrix, size, padding)
    # transform the matrix for size and padding
    mat = matrix * (size - 2 * padding)
    mat[:, 2] += padding

    # transform image
    interpolators = get_matrix_scaling(mat)
    retval = cv2.warpAffine(image, mat, (size, size), flags=interpolators[0])
    logger.trace("transformed matrix: %s, final image shape: %s", mat, image.shape)
    return retval


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
        with the crop holding the full head. Default: `"legacy"`
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
    """
    def __init__(self, landmarks, image=None, centering="legacy", size=64, coverage_ratio=1.0,
                 dtype=None, is_aligned=False):
        logger.trace("Initializing: %s (image shape: %s, centering: '%s', size: %s, "
                     "coverage_ratio: %s, dtype: %s, is_aligned: %s)", self.__class__.__name__,
                     image if image is None else image.shape, centering, size, coverage_ratio,
                     dtype, is_aligned)
        self._frame_landmarks = landmarks
        self._centering = centering
        self._size = size
        self._dtype = dtype
        self._is_aligned = is_aligned
        self._ratios = dict(legacy=0.375, face=0.5, head=0.625)
        self._matrices = dict(legacy=umeyama(landmarks[17:], _MEAN_FACE, True)[0:2],
                              face=None,
                              head=None)
        self._padding = self._padding_from_coverage(size, coverage_ratio)

        self._cache = dict(pose=[None, Lock()],
                           original_roi=[None, Lock()],
                           landmarks=[None, Lock()],
                           adjusted_matrix=[None, Lock()],
                           interpolators=[None, Lock()],
                           cropped_roi=[dict(), Lock()],
                           cropped_size=[dict(), Lock()])

        self._face = self._extract_face(image)
        logger.trace("Initialized: %s (matrix: %s, padding: %s, face shape: %s)",
                     self.__class__.__name__, self._matrices["legacy"], self._padding,
                     self._face if self._face is None else self._face.shape)

    @property
    def size(self):
        """ int: The size (in pixels) of one side of the square extracted face image. """
        return self._size

    @property
    def padding(self):
        """ int: The amount of padding (in pixels) that is applied to each side of the
        extracted face image for the selected extract type. """
        return self._padding[self._centering]

    @property
    def matrix(self):
        """ :class:`numpy.ndarray`: The 3x2 transformation matrix for extracting and aligning the
        core face area out of the original frame, with no padding or sizing applied. The returned
        matrix is offset for the given :attr:`centering`. """
        if self._matrices[self._centering] is None:
            matrix = self._matrices["legacy"].copy()
            matrix[:, 2] -= self.pose.offset[self._centering]
            self._matrices[self._centering] = matrix
            logger.trace("original matrix: %s, new matrix: %s", self._matrices["legacy"], matrix)
        return self._matrices[self._centering]

    @property
    def pose(self):
        """ :class:`lib.align.PoseEstimate`: The estimated pose in 3D space. """
        with self._cache["pose"][1]:
            if self._cache["pose"][0] is None:
                lms = cv2.transform(np.expand_dims(self._frame_landmarks, axis=1),
                                    self._matrices["legacy"]).squeeze()
                self._cache["pose"][0] = PoseEstimate(lms)
        return self._cache["pose"][0]

    @property
    def adjusted_matrix(self):
        """ :class:`numpy.ndarray`: The 3x2 transformation matrix for extracting and aligning the
        core face area out of the original frame with padding and sizing applied. """
        with self._cache["adjusted_matrix"][1]:
            if self._cache["adjusted_matrix"][0] is None:
                matrix = self.matrix.copy()
                mat = matrix * (self._size - 2 * self.padding)
                mat[:, 2] += self.padding
                logger.trace("adjusted_matrix: %s", mat)
                self._cache["adjusted_matrix"][0] = mat
        return self._cache["adjusted_matrix"][0]

    @property
    def face(self):
        """ :class:`numpy.ndarray`: The aligned face at the given :attr:`size` at the specified
        :attr:`coverage` in the given :attr:`dtype`. If an :attr:`image` has not been provided
        then an the attribute will return ``None``. """
        return self._face

    @property
    def original_roi(self):
        """ :class:`numpy.ndarray`: The location of the extracted face box within the original
        frame. """
        with self._cache["original_roi"][1]:
            if self._cache["original_roi"][0] is None:
                roi = np.array([[0, 0],
                                [0, self._size - 1],
                                [self._size - 1, self._size - 1],
                                [self._size - 1, 0]])
                roi = np.rint(self.transform_points(roi, invert=True)).astype("int32")
                logger.trace("original roi: %s", roi)
                self._cache["original_roi"][0] = roi
        return self._cache["original_roi"][0]

    @property
    def landmarks(self):
        """ :class:`numpy.ndarray`: The 68 point facial landmarks aligned to the extracted face
        box. """
        with self._cache["landmarks"][1]:
            if self._cache["landmarks"][0] is None:
                lms = self.transform_points(self._frame_landmarks)
                logger.trace("aligned landmarks: %s", lms)
                self._cache["landmarks"][0] = lms
        return self._cache["landmarks"][0]

    @property
    def interpolators(self):
        """ tuple: (`interpolator` and `reverse interpolator`) for the :attr:`adjusted matrix`. """
        with self._cache["interpolators"][1]:
            if self._cache["interpolators"][0] is None:
                interpolators = get_matrix_scaling(self.adjusted_matrix)
                logger.trace("interpolators: %s", interpolators)
                self._cache["interpolators"][0] = interpolators
        return self._cache["interpolators"][0]

    def transform_points(self, points, invert=False):
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
        logger.trace("invert: %s, Original points: %s, transformed points: %s",
                     invert, points, retval)
        return retval

    def _extract_face(self, image):
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
            logger.debug("_extract_face called without a loaded image. Returning empty face.")
            return None
        if self._is_aligned:  # Resize the given aligned face
            original_size = image.shape[0]
            interp = cv2.INTER_CUBIC if original_size < self._size else cv2.INTER_AREA
            retval = cv2.resize(image, (self._size, self._size), interpolation=interp)
        else:
            retval = transform_image(image, self.matrix, self._size, self.padding)
        retval = retval if self._dtype is None else retval.astype(self._dtype)
        return retval

    def _padding_from_coverage(self, size, coverage_ratio):
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
        retval = {_type: round((size * (coverage_ratio - (1 - self._ratios[_type]))) / 2)
                  for _type in ("legacy", "face", "head")}
        logger.trace(retval)
        return retval

    def get_cropped_roi(self, centering):
        """ Obtain the region of interest within an aligned face set to centered coverage for
        an alternative centering

        Parameters
        ----------
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
        if self._centering != "head":
            raise ValueError("Sub ROI can only be obtained from an aligned face with 'head' "
                             "centering")
        with self._cache["cropped_roi"][1]:
            if centering not in self._cache["cropped_roi"][0]:
                # Get offset from center without padding
                if centering == "legacy":
                    offset = np.float32((0, 0))
                else:
                    offset = self.pose.offset[centering]
                offset -= self.pose.offset["head"]
                offset *= ((self._size - self._padding["head"]) / 2)

                # Get roi from sub image from adjusted center and correct padding
                center = np.rint(offset + self._size / 2).astype("int32")
                padding = np.rint((self._size / self._ratios["head"]) *
                                  self._ratios[centering] / 2).astype("int32")
                roi = np.array([center - padding, center + padding]).ravel()

                logger.trace("centering: '%s', center: %s, padding: %s, sub roi: %s",
                             centering, center, padding, roi)
                self._cache["cropped_roi"][0][centering] = roi
        return self._cache["cropped_roi"][0][centering]

    def get_cropped_size(self, centering):
        """ Obtain the size of a cropped face from a full head centered image.

        Parameters
        ----------
        centering: ["legacy", "face"]
            The type of centering to obtain the region of interest for. "legacy" places the nose
            in the center of the image (the original method for aligning). "face" aligns for the
            nose to be in the center of the face (top to bottom) but the center of the skull for
            left to right.

        Returns
        -------
        int
           The pixel size of a sub-crop image from a full head aligned image
        """
        if self._centering != "head":
            raise ValueError("Sub ROI can only be obtained from an aligned face with 'head' "
                             "centering")
        with self._cache["cropped_size"][1]:
            if not self._cache["cropped_size"][0].get(centering):
                size = int(np.rint(self._size / self._ratios["head"] * self._ratios[centering]))
                logger.trace("centering: %s, size: %s, crop_size: %s", centering, self._size, size)
                self._cache["cropped_size"][0][centering] = size
        return self._cache["cropped_size"][0][centering]


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
    def __init__(self, landmarks):
        self._distortion_coefficients = np.zeros((4, 1))  # Assuming no lens distortion
        self._xyz_2d = None

        self._camera_matrix = self._get_camera_matrix()
        self._rotation, self._translation = self._solve_pnp(landmarks)
        self._offset = self._get_offset()

    @property
    def xyz_2d(self):
        """ :class:`numpy.ndarray` projected (x, y) coordinates for each x, y, z point at a
        constant distance from adjusted center of the skull (0.5, 0.5) in the 2D space. """
        if self._xyz_2d is None:
            xyz = cv2.projectPoints(np.float32([[6, 0, -2.3], [0, 6, -2.3], [0, 0, 3.7]]),
                                    self._rotation,
                                    self._translation,
                                    self._camera_matrix,
                                    self._distortion_coefficients)[0].squeeze()
            self._xyz_2d = xyz - self._offset["head"]
        return self._xyz_2d

    @property
    def offset(self):
        """ dict: The amount to offset a standard 0.0 - 1.0 umeyama transformation matrix for a
        from the center of the face (between the eyes) or center of the head (middle of skull)
        rather than the nose area. """
        return self._offset

    @classmethod
    def _get_camera_matrix(cls):
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
        logger.trace("camera_matrix: %s", camera_matrix)
        return camera_matrix

    def _solve_pnp(self, landmarks):
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
        logger.trace("points: %s, rotation: %s, translation: %s", points, rotation, translation)
        return rotation, translation

    def _get_offset(self):
        """ Obtain the offset between the original center of the extracted face to the new center
        of the head in 2D space.

        Returns
        -------
        :class:`numpy.ndarray`
            The x, y offset of the new center from the old center.
        """
        points = dict(head=(0, 0, -2.3), face=(0, -1.5, 4.2))
        offset = dict()
        for key, pnts in points.items():
            center = cv2.projectPoints(np.float32([pnts]),
                                       self._rotation,
                                       self._translation,
                                       self._camera_matrix,
                                       self._distortion_coefficients)[0].squeeze()
            logger.trace("center %s: %s", key, center)
            offset[key] = center - (0.5, 0.5)
        logger.trace("offset: %s", offset)
        return offset
