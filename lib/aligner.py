#!/usr/bin/env python3
""" Aligner for faceswap.py """

import logging

import cv2
import numpy as np

from lib.umeyama import umeyama
from lib.align_eyes import align_eyes as func_align_eyes, FACIAL_LANDMARKS_IDXS

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class Extract():
    """ Based on the original https://www.reddit.com/r/deepfakes/
        code sample + contribs """

    def extract(self, image, face, size, align_eyes):
        """ Extract a face from an image """
        logger.trace("size: %s. align_eyes: %s", size, align_eyes)
        padding = int(size * 0.1875)
        alignment = get_align_mat(face, size, align_eyes)
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
        return cv2.warpAffine(  # pylint: disable=no-member
            image, matrix, (size, size), flags=interpolators[0])

    def transform_points(self, points, mat, size, padding=0):
        """ Transform points along matrix """
        logger.trace("points: %s, matrix: %s, size: %s. padding: %s", points, mat, size, padding)
        matrix = self.transform_matrix(mat, size, padding)
        points = np.expand_dims(points, axis=1)
        points = cv2.transform(  # pylint: disable=no-member
            points, matrix, points.shape)
        retval = np.squeeze(points)
        logger.trace("Returning: %s", retval)
        return retval

    def get_original_roi(self, mat, size, padding=0):
        """ Return the square aligned box location on the original
            image """
        logger.trace("matrix: %s, size: %s. padding: %s", mat, size, padding)
        matrix = self.transform_matrix(mat, size, padding)
        points = np.array([[0, 0],
                           [0, size - 1],
                           [size - 1, size - 1],
                           [size - 1, 0]], np.int32)
        points = points.reshape((-1, 1, 2))
        matrix = cv2.invertAffineTransform(matrix)  # pylint: disable=no-member
        logger.trace("Returning: (points: %s, matrix: %s", points, matrix)
        return cv2.transform(points, matrix)  # pylint: disable=no-member

    @staticmethod
    def get_feature_mask(aligned_landmarks_68, size,
                         padding=0, dilation=30):
        """ Return the face feature mask """
        # pylint: disable=no-member
        logger.trace("aligned_landmarks_68: %s, size: %s, padding: %s, dilation: %s",
                     aligned_landmarks_68, size, padding, dilation)
        scale = size - 2 * padding
        translation = padding
        pad_mat = np.matrix([[scale, 0.0, translation],
                             [0.0, scale, translation]])
        aligned_landmarks_68 = np.expand_dims(aligned_landmarks_68, axis=1)
        aligned_landmarks_68 = cv2.transform(aligned_landmarks_68,
                                             pad_mat,
                                             aligned_landmarks_68.shape)
        aligned_landmarks_68 = np.squeeze(aligned_landmarks_68)

        (l_start, l_end) = FACIAL_LANDMARKS_IDXS["left_eye"]
        (r_start, r_end) = FACIAL_LANDMARKS_IDXS["right_eye"]
        (m_start, m_end) = FACIAL_LANDMARKS_IDXS["mouth"]
        (n_start, n_end) = FACIAL_LANDMARKS_IDXS["nose"]
        (lb_start, lb_end) = FACIAL_LANDMARKS_IDXS["left_eyebrow"]
        (rb_start, rb_end) = FACIAL_LANDMARKS_IDXS["right_eyebrow"]
        (c_start, c_end) = FACIAL_LANDMARKS_IDXS["chin"]

        l_eye_points = aligned_landmarks_68[l_start:l_end].tolist()
        l_brow_points = aligned_landmarks_68[lb_start:lb_end].tolist()
        r_eye_points = aligned_landmarks_68[r_start:r_end].tolist()
        r_brow_points = aligned_landmarks_68[rb_start:rb_end].tolist()
        nose_points = aligned_landmarks_68[n_start:n_end].tolist()
        chin_points = aligned_landmarks_68[c_start:c_end].tolist()
        mouth_points = aligned_landmarks_68[m_start:m_end].tolist()
        l_eye_points = l_eye_points + l_brow_points
        r_eye_points = r_eye_points + r_brow_points
        mouth_points = mouth_points + nose_points + chin_points

        l_eye_hull = cv2.convexHull(np.array(l_eye_points).reshape(
            (-1, 2)).astype(int)).flatten().reshape((-1, 2))
        r_eye_hull = cv2.convexHull(np.array(r_eye_points).reshape(
            (-1, 2)).astype(int)).flatten().reshape((-1, 2))
        mouth_hull = cv2.convexHull(np.array(mouth_points).reshape(
            (-1, 2)).astype(int)).flatten().reshape((-1, 2))

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
    if avg_scale >= 1.0:
        interpolators = cv2.INTER_CUBIC, cv2.INTER_AREA   # pylint: disable=no-member
    else:
        interpolators = cv2.INTER_AREA, cv2.INTER_CUBIC  # pylint: disable=no-member
    logger.trace("interpolator: %s, inverse interpolator: %s", interpolators[0], interpolators[1])
    return interpolators


def get_align_mat(face, size, should_align_eyes):
    """ Return the alignment Matrix """
    logger.trace("size: %s, should_align_eyes: %s", size, should_align_eyes)
    mat_umeyama = umeyama(np.array(face.landmarks_as_xy[17:]), True)[0:2]

    if should_align_eyes is False:
        return mat_umeyama

    mat_umeyama = mat_umeyama * size

    # Convert to matrix
    landmarks = np.matrix(face.landmarks_as_xy)

    # cv2 expects points to be in the form
    # np.array([ [[x1, y1]], [[x2, y2]], ... ]), we'll expand the dim
    landmarks = np.expand_dims(landmarks, axis=1)

    # Align the landmarks using umeyama
    umeyama_landmarks = cv2.transform(  # pylint: disable=no-member
        landmarks,
        mat_umeyama,
        landmarks.shape)

    # Determine a rotation matrix to align eyes horizontally
    mat_align_eyes = func_align_eyes(umeyama_landmarks, size)

    # Extend the 2x3 transform matrices to 3x3 so we can multiply them
    # and combine them as one
    mat_umeyama = np.matrix(mat_umeyama)
    mat_umeyama.resize((3, 3))
    mat_align_eyes = np.matrix(mat_align_eyes)
    mat_align_eyes.resize((3, 3))
    mat_umeyama[2] = mat_align_eyes[2] = [0, 0, 1]

    # Combine the umeyama transform with the extra rotation matrix
    transform_mat = mat_align_eyes * mat_umeyama

    # Remove the extra row added, shape needs to be 2x3
    transform_mat = np.delete(transform_mat, 2, 0)
    transform_mat = transform_mat / size
    logger.trace("Returning: %s", transform_mat)
    return transform_mat
