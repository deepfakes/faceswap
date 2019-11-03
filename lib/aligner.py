#!/usr/bin/env python3
""" Aligner for faceswap.py """

import logging

import cv2
import numpy as np

from lib.umeyama import umeyama

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class Extract():
    """ Based on the original https://www.reddit.com/r/deepfakes/
        code sample + contribs """

    def extract(self, image, face, size):
        """ Extract a face from an image """
        logger.trace("size: %s", size)
        padding = int(size * 0.1875)
        alignment = get_align_mat(face)
        extracted = self.transform(image, alignment, size, padding)
        logger.trace("Returning face and alignment matrix: (alignment_matrix: %s)", alignment)
        return extracted, alignment

    @staticmethod
    def transform_matrix(mat, size, padding):
        """ Transform the matrix for current size and padding """
        logger.trace("size: %s. padding: %s", size, padding)
        matrix = mat * (size - 2 * padding)
        matrix[:, 2] += padding
        logger.trace("Returning: %s", matrix)
        return matrix

    def transform(self, image, mat, size, padding=0):
        """ Transform Image """
        logger.trace("matrix: %s, size: %s. padding: %s", mat, size, padding)
        matrix = self.transform_matrix(mat, size, padding)
        interpolators = get_matrix_scaling(matrix)
        retval = cv2.warpAffine(image, matrix, (size, size), flags=interpolators[0])
        return retval

    def transform_points(self, points, mat, size, padding=0):
        """ Transform points along matrix """
        logger.trace("points: %s, matrix: %s, size: %s. padding: %s", points, mat, size, padding)
        matrix = self.transform_matrix(mat, size, padding)
        points = np.expand_dims(points, axis=1)
        points = cv2.transform(points, matrix, points.shape)
        retval = np.squeeze(points)
        logger.trace("Returning: %s", retval)
        return retval

    def get_original_roi(self, mat, size, padding=0):
        """ Return the square aligned box location on the original image """
        logger.trace("matrix: %s, size: %s. padding: %s", mat, size, padding)
        matrix = self.transform_matrix(mat, size, padding)
        points = np.array([[0, 0], [0, size - 1], [size - 1, size - 1], [size - 1, 0]], np.int32)
        points = points.reshape((-1, 1, 2))
        matrix = cv2.invertAffineTransform(matrix)
        logger.trace("Returning: (points: %s, matrix: %s", points, matrix)
        return cv2.transform(points, matrix)

    @staticmethod
    def get_feature_mask(aligned_landmarks_68, size, padding=0, dilation=30):
        """ Return the face feature mask """
        logger.trace("aligned_landmarks_68: %s, size: %s, padding: %s, dilation: %s",
                     aligned_landmarks_68, size, padding, dilation)
        scale = size - 2 * padding
        translation = padding
        pad_mat = np.matrix([[scale, 0.0, translation], [0.0, scale, translation]])
        aligned_landmarks_68 = np.expand_dims(aligned_landmarks_68, axis=1)
        aligned_landmarks_68 = cv2.transform(aligned_landmarks_68,
                                             pad_mat,
                                             aligned_landmarks_68.shape)
        aligned_landmarks_68 = np.squeeze(aligned_landmarks_68)
        l_eye_points = aligned_landmarks_68[42:48].tolist()
        l_brow_points = aligned_landmarks_68[22:27].tolist()
        r_eye_points = aligned_landmarks_68[36:42].tolist()
        r_brow_points = aligned_landmarks_68[17:22].tolist()
        nose_points = aligned_landmarks_68[27:36].tolist()
        chin_points = aligned_landmarks_68[8:11].tolist()
        mouth_points = aligned_landmarks_68[48:68].tolist()
        # TODO remove excessive reshapes and flattens

        l_eye = np.array(l_eye_points + l_brow_points).reshape((-1, 2)).astype('int32').flatten()
        r_eye = np.array(r_eye_points + r_brow_points).reshape((-1, 2)).astype('int32').flatten()
        mouth = np.array(mouth_points + nose_points + chin_points)
        mouth = mouth.reshape((-1, 2)).astype('int32').flatten()
        l_eye_hull = cv2.convexHull(l_eye.reshape((-1, 2)))
        r_eye_hull = cv2.convexHull(r_eye.reshape((-1, 2)))
        mouth_hull = cv2.convexHull(mouth.reshape((-1, 2)))

        mask = np.zeros((size, size, 3), dtype=float)
        cv2.fillConvexPoly(mask, l_eye_hull, (1, 1, 1))
        cv2.fillConvexPoly(mask, r_eye_hull, (1, 1, 1))
        cv2.fillConvexPoly(mask, mouth_hull, (1, 1, 1))

        if dilation > 0:
            kernel = np.ones((dilation, dilation), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1)

        logger.trace("Returning: %s", mask)
        return mask


def get_matrix_scaling(mat):
    """ Get the correct interpolator """
    x_scale = np.sqrt(mat[0, 0] * mat[0, 0] + mat[0, 1] * mat[0, 1])
    y_scale = (mat[0, 0] * mat[1, 1] - mat[0, 1] * mat[1, 0]) / x_scale
    avg_scale = (x_scale + y_scale) * 0.5
    if avg_scale >= 1.:
        interpolators = cv2.INTER_CUBIC, cv2.INTER_AREA
    else:
        interpolators = cv2.INTER_AREA, cv2.INTER_CUBIC
    logger.trace("interpolator: %s, inverse interpolator: %s", interpolators[0], interpolators[1])
    return interpolators


def get_align_mat(face):
    """ Return the alignment Matrix """
    mat_umeyama = umeyama(face.landmarks_xy[17:], True)[0:2]
    return mat_umeyama
