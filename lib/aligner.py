#!/usr/bin/env python3
""" Aligner for faceswap.py """

import logging

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


class Extract():
    """ Alignment tools for transforming face and landmark points to and from a source frame/
    aligned face.

    Based on the original https://www.reddit.com/r/deepfakes/ code sample + contributions.
    """

    @staticmethod
    def transform_matrix(matrix, size, padding):
        """ Adjust the given matrix to the given size and padding.

        Parameters
        ----------
        matrix: :class:`numpy.ndarray`
            The original transformation matrix to be adjusted
        size: int
            The size that the matrix is to be adjusted for
        padding: int
            The amount of padding to be applied to each side of the image

        Returns
        -------
        :class:`numpy.ndarray`
            The original matrix adjusted for the given size and padding
        """
        logger.trace("size: %s. padding: %s", size, padding)
        retval = matrix * (size - 2 * padding)
        retval[:, 2] += padding
        logger.trace("Returning: %s", retval)
        return retval

    def transform(self, image, matrix, size, padding=0):
        """ Perform transformation on an image.

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
        logger.trace("matrix: %s, size: %s. padding: %s", matrix, size, padding)
        mat = self.transform_matrix(matrix, size, padding)
        interpolators = get_matrix_scaling(mat)
        retval = cv2.warpAffine(image, mat, (size, size), flags=interpolators[0])
        return retval


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


def get_align_matrix(landmarks):
    """ Get the Umeyama alignment Matrix for the core 52 face landmarks. for aligning a face

    Parameters
    ----------
    landmarks: :class:`numpy.ndarry`
        The original 68 point landmark co-ordinates relating to the original frame

    Returns
    -------
    :class:`numpy.ndarry`
        The alignment matrix
    """
    # indices = [27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 39, 40, 41, 42, 45, 46, 47]
    # mat_umeyama = umeyama(landmarks[indices], _MEAN_FACE, True)[0:2]
    mat_umeyama = umeyama(landmarks[17:], _MEAN_FACE, True)[0:2]
    return mat_umeyama


class PoseEstimate():
    """ Estimates pose from a generic 3D head model for the given global 2D face landmarks.

    Parameters
    ----------
    matrix: :class:`numpy.ndarray`
        The original umeyama transformation matrix with no adjustments applied
    frame_dimensions: tuple
        The (`height`, `width`) dimensions of the source frame, in pixels
    landmarks: :class:`numpy.ndarry`
        The original 68 point landmark co-ordinates relating to the original frame

    References
    ----------
    Head Pose Estimation using OpenCV and Dlib - https://www.learnopencv.com/tag/solvepnp/
    3D Model points - http://aifi.isr.uc.pt/Downloads/OpenGL/glAnthropometric3DModel.cpp
    """
    def __init__(self, matrix, frame_dimensions, landmarks):
        self._mean_face = np.array([
            [4.056931, -11.432347, 1.636229],   # 8 chin LL
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
        self._distortion_coefficients = np.zeros((4, 1))  # Assuming no lens distortion
        self._cache = dict(center_2d=None, xyz_2d=None)

        self._camera_matrix = self._get_camera_matrix(frame_dimensions)
        self._rotation, self._translation = self._solve_pnp(landmarks)
        self._matrix = self._adjust_matrix(matrix)

    @property
    def center_2d(self):
        """ :class:`numpy.ndarray`: The projected (x, y) coordinates of the center of the skull
        in 2D space"""
        if self._cache["center_2d"] is None:
            center, _ = cv2.projectPoints(np.float32([[0, 0, 0]]),
                                          self._rotation,
                                          self._translation,
                                          self._camera_matrix,
                                          self._distortion_coefficients)
            self._cache["center_2d"] = center
        return self._cache["center_2d"]

    @property
    def xyz_2d(self):
        """ :class:`numpy.ndarray` projected (x, y) coordinates for each x, y, z point at a
        constant distance from center of the skull in the 2D space. """
        if self._cache["xyz_2d"] is None:
            xyz, _ = cv2.projectPoints(np.float32([[12, 0, 0], [0, 12, 0], [0, 0, 12]]),
                                       self._rotation,
                                       self._translation,
                                       self._camera_matrix,
                                       self._distortion_coefficients)
            self._cache["xyz_2d"] = xyz.squeeze()
        return self._cache["xyz_2d"]

    @property
    def matrix(self):
        """:class:`numpy.ndarray`: The adjusted umeyama transformation matrix focused on the
        center of the skull. """
        return self._matrix

    @classmethod
    def _get_camera_matrix(cls, frame_dimensions):
        """ Obtain an estimate of the camera matrix based off the original frame dimensions.

        Parameters
        ----------
        frame_dimensions: tuple
            The (`height`, `width`) dimensions of the source frame, in pixels

        Returns
        -------
        :class:`numpy.ndarray`
            An estimated camera matrix
        """
        frame_center = (frame_dimensions[0]/2, frame_dimensions[1]/2)
        focal_length = frame_center[0] / np.tan(60/2 * np.pi / 180)
        camera_matrix = np.array([[focal_length, 0, frame_center[0]],
                                  [0, focal_length, frame_center[1]],
                                  [0, 0, 1]], dtype="double")
        logger.trace("frame_dimensions: %s, camera_matrix: %s", frame_dimensions, camera_matrix)
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
        _, rotation, translation = cv2.solvePnP(self._mean_face,
                                                points,
                                                self._camera_matrix,
                                                self._distortion_coefficients,
                                                flags=cv2.SOLVEPNP_ITERATIVE)
        logger.trace("points: %s, rotation: %s, translation: %s", points, rotation, translation)
        return rotation, translation

    def _adjust_matrix(self, matrix):
        """ Adjust a standard face umeyama transformation matrix to center on the full head rather
        than the face.

        Parameters
        ----------
        matrix: :class:`numpy.ndarray`
            The original umeyama transformation matrix with no adjustments applied

        Returns
        -------
        :class:`numpy.ndarray`
            The original umeyama transformation matrix adjusted to center on the middle of the
            skull
        """
        retval = matrix.copy()
        center = self.center_2d  # Project center of skull to 2D
        # Adjust matrix to new center
        retval[:, 2] -= (cv2.transform(center, matrix, center.shape).squeeze() - (0.5, 0.5))
        logger.trace("original matrix: %s, new center: %s, new matrix: %s", matrix, center, retval)
        return retval
