#!/usr/bin/env python3
""" Aligner for faceswap.py """

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


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
        _, rotation, translation = cv2.solvePnP(self._mean_face,
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
