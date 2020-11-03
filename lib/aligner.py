#!/usr/bin/env python3
""" Aligner for faceswap.py """

import logging

import cv2
import numpy as np

from skimage.transform._geometric import _umeyama as umeyama

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


_MEAN_FACE = np.stack([
    [0.000213256, 0.0752622, 0.18113, 0.29077, 0.393397, 0.586856, 0.689483,
     0.799124, 0.904991, 0.98004, 0.490127, 0.490127, 0.490127, 0.490127,
     0.36688, 0.426036, 0.490127, 0.554217, 0.613373, 0.121737, 0.187122,
     0.265825, 0.334606, 0.260918, 0.182743, 0.645647, 0.714428, 0.793132,
     0.858516, 0.79751, 0.719335, 0.254149, 0.340985, 0.428858, 0.490127,
     .551395, 0.639268, 0.726104, 0.642159, 0.556721, 0.490127, 0.423532,
     0.338094, 0.290379, 0.428096, 0.490127, 0.552157, 0.689874, 0.553364,
     0.490127, 0.42689],
    [0.106454, 0.038915, 0.0187482, 0.0344891, 0.0773906, 0.0773906, 0.0344891,
     0.0187482, 0.038915, 0.106454, 0.203352, 0.307009, 0.409805, 0.515625,
     0.587326, 0.609345, 0.628106, 0.609345, 0.587326, 0.216423, 0.178758,
     0.179852, 0.231733, 0.245099, 0.244077, 0.231733, 0.179852, 0.178758,
     0.216423, 0.244077, 0.245099, 0.780233, 0.745405, 0.727388, 0.742578,
     0.727388, 0.745405, 0.780233, 0.864805, 0.902192, 0.909281, 0.902192,
     0.864805, 0.784792, 0.778746, 0.785343, 0.778746, 0.784792, 0.824182,
     0.831803, 0.824182]], axis=1)


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

    def transform_points(self, points, matrix, size, padding=0):
        """ Perform transformation on a series of x, y co-ordinates.

        Parameters
        ----------
        points: :class:`numpy.ndarray`
            The points to transform
        matrix: :class:`numpy.ndarray`
            The transformation matrix to apply to the image
        size: int
            The final size of the transformed image
        padding: int, optional
            The amount of padding to apply to the final image. Default: `0`

        Returns
        -------
        :class:`numpy.ndarray`
            The transformed points
        """
        logger.trace("points: %s, matrix: %s, size: %s. padding: %s",
                     points, matrix, size, padding)
        mat = self.transform_matrix(matrix, size, padding)
        points = np.expand_dims(points, axis=1)
        points = cv2.transform(points, mat, points.shape)
        retval = np.squeeze(points)
        logger.trace("Returning: %s", retval)
        return retval

    def get_original_roi(self, matrix, size, padding=0):
        """ Return the square aligned box location on an original frame.

        Parameters
        ----------
        matrix: :class:`numpy.ndarray`
            The transformation matrix used to extract the image
        size: int
            The final size of the transformed image
        padding: int, optional
            The amount of padding applied to the final image. Default: `0`

        Returns
        -------
        :class:`numpy.ndarray`
            The original ROI points
        """
        logger.trace("matrix: %s, size: %s. padding: %s", matrix, size, padding)
        mat = self.transform_matrix(matrix, size, padding)
        points = np.array([[0, 0], [0, size - 1], [size - 1, size - 1], [size - 1, 0]], np.int32)
        points = points.reshape((-1, 1, 2))
        mat = cv2.invertAffineTransform(mat)
        logger.trace("Returning: (points: %s, matrix: %s", points, mat)
        return cv2.transform(points, mat)


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


def get_align_matrix(face):
    """ Get the Umeyama alignment Matrix for the core 52 face landmarks. for aligning a face

    Parameters
    ----------
    face: :class:`lib.faces_detect.DetectedFace`
        The detected face object to retrieve the alignment matrix

    Returns
    -------
    :class:`numpy.ndarry`
        The alignment matrix
    """
    mat_umeyama = umeyama(face.landmarks_xy[17:], _MEAN_FACE, True)[0:2]
    return mat_umeyama
