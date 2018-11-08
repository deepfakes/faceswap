#!/usr/bin/env python3
""" Aligner for faceswap.py """

import cv2
import numpy as np

from lib.umeyama import umeyama
from lib.align_eyes import align_eyes as func_align_eyes, FACIAL_LANDMARKS_IDXS

MEAN_FACE_X = np.array([
    0.000213256, 0.0752622, 0.18113, 0.29077, 0.393397, 0.586856, 0.689483,
    0.799124, 0.904991, 0.98004, 0.490127, 0.490127, 0.490127, 0.490127,
    0.36688, 0.426036, 0.490127, 0.554217, 0.613373, 0.121737, 0.187122,
    0.265825, 0.334606, 0.260918, 0.182743, 0.645647, 0.714428, 0.793132,
    0.858516, 0.79751, 0.719335, 0.254149, 0.340985, 0.428858, 0.490127,
    .551395, 0.639268, 0.726104, 0.642159, 0.556721, 0.490127, 0.423532,
    0.338094, 0.290379, 0.428096, 0.490127, 0.552157, 0.689874, 0.553364,
    0.490127, 0.42689])

MEAN_FACE_Y = np.array([
    0.106454, 0.038915, 0.0187482, 0.0344891, 0.0773906, 0.0773906, 0.0344891,
    0.0187482, 0.038915, 0.106454, 0.203352, 0.307009, 0.409805, 0.515625,
    0.587326, 0.609345, 0.628106, 0.609345, 0.587326, 0.216423, 0.178758,
    0.179852, 0.231733, 0.245099, 0.244077, 0.231733, 0.179852, 0.178758,
    0.216423, 0.244077, 0.245099, 0.780233, 0.745405, 0.727388, 0.742578,
    0.727388, 0.745405, 0.780233, 0.864805, 0.902192, 0.909281, 0.902192,
    0.864805, 0.784792, 0.778746, 0.785343, 0.778746, 0.784792, 0.824182,
    0.831803, 0.824182])

LANDMARKS_2D = np.stack([MEAN_FACE_X, MEAN_FACE_Y], axis=1)


class Extract():
    """ Based on the original https://www.reddit.com/r/deepfakes/
        code sample + contribs """

    def extract(self, image, face, size, align_eyes):
        """ Extract a face from an image """
        alignment = get_align_mat(face, size, align_eyes)
        extracted = self.transform(image, alignment, size, 48)
        return extracted, alignment

    @staticmethod
    def transform_matrix(mat, size, padding):
        """ Transform the matrix for current size and padding """
        matrix = mat * (size - 2 * padding)
        matrix[:, 2] += padding
        return matrix

    def transform(self, image, mat, size, padding=0):
        """ Transform Image """
        matrix = self.transform_matrix(mat, size, padding)
        return cv2.warpAffine(  # pylint: disable=no-member
            image, matrix, (size, size))

    def transform_points(self, points, mat, size, padding=0):
        """ Transform points along matrix """
        matrix = self.transform_matrix(mat, size, padding)
        points = np.expand_dims(points, axis=1)
        points = cv2.transform(  # pylint: disable=no-member
            points, matrix, points.shape)
        return np.squeeze(points)

    def get_original_roi(self, mat, size, padding=0):
        """ Return the square aligned box location on the original
            image """
        matrix = self.transform_matrix(mat, size, padding)
        points = np.array([[0, 0],
                           [0, size - 1],
                           [size - 1, size - 1],
                           [size - 1, 0]], np.int32)
        points = points.reshape((-1, 1, 2))
        matrix = cv2.invertAffineTransform(matrix)  # pylint: disable=no-member
        return cv2.transform(points, matrix)  # pylint: disable=no-member

    @staticmethod
    def get_feature_mask(aligned_landmarks_68, size,
                         padding=0, dilation=30):
        """ Return the face feature mask """
        # pylint: disable=no-member
        scale = size - 2*padding
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

        return mask


def get_align_mat(face, size, should_align_eyes):
    """ Return the alignment Matrix """
    mat_umeyama = umeyama(np.array(face.landmarks_as_xy()[17:]),
                          LANDMARKS_2D,
                          True)[0:2]

    if should_align_eyes is False:
        return mat_umeyama

    mat_umeyama = mat_umeyama * size

    # Convert to matrix
    landmarks = np.matrix(face.landmarks_as_xy())

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
    return transform_mat
