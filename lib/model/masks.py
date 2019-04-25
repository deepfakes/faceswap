#!/usr/bin/env python3
""" Masks functions for faceswap.py
    Masks from:
        dfaker: https://github.com/dfaker/df"""

import logging

import cv2
import numpy as np

from lib.umeyama import umeyama
from lib.aligner import LANDMARKS_2D

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def dfaker(landmarks, face, **kwargs):
    """ Dfaker model mask
        Embeds the mask into the face alpha channel """
    coverage = kwargs["coverage"]
    logger.trace("face_shape: %s, coverage: %s, landmarks: %s", face.shape, coverage, landmarks)
    size = face.shape[0] - 1

    mat = umeyama(landmarks[17:], LANDMARKS_2D, True)[0:2]
    mat = np.array(mat.ravel()).reshape(2, 3)
    mat = mat * coverage
    mat[:, 2] += 42

    points = np.array(landmarks).reshape((-1, 2))
    facepoints = np.array(points).reshape((-1, 2))

    mask = np.zeros_like(face, dtype=np.uint8)

    hull = cv2.convexHull(facepoints.astype(int))  # pylint: disable=no-member
    hull = cv2.transform(hull.reshape(1, -1, 2),  # pylint: disable=no-member
                         mat).reshape(-1, 2).astype(int)
    cv2.fillConvexPoly(mask, hull, (size, size, size))  # pylint: disable=no-member

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))  # pylint: disable=no-member
    mask = cv2.dilate(mask,  # pylint: disable=no-member
                      kernel,
                      iterations=1,
                      borderType=cv2.BORDER_REFLECT)  # pylint: disable=no-member
    mask = mask[:, :, 0]
    merged = np.dstack([face, mask]).astype(np.uint8)

    logger.trace("Returning: face_shape: %s", merged.shape)
    return merged


def dfl_full(landmarks, face, **kwargs):
    """ DFL Face Full Mask """
    logger.trace("face_shape: %s, landmarks: %s", face.shape, landmarks)
    hull_mask = np.zeros(face.shape[0:2] + (1, ), dtype=np.float32)
    hull1 = cv2.convexHull(np.concatenate((landmarks[0:17],  # pylint: disable=no-member
                                           landmarks[48:],
                                           [landmarks[0]],
                                           [landmarks[8]],
                                           [landmarks[16]])))
    hull2 = cv2.convexHull(np.concatenate((landmarks[27:31],  # pylint: disable=no-member
                                           [landmarks[33]])))
    hull3 = cv2.convexHull(np.concatenate((landmarks[17:27],  # pylint: disable=no-member
                                           [landmarks[0]],
                                           [landmarks[27]],
                                           [landmarks[16]],
                                           [landmarks[33]])))

    cv2.fillConvexPoly(hull_mask, hull1, (1, ))  # pylint: disable=no-member
    cv2.fillConvexPoly(hull_mask, hull2, (1, ))  # pylint: disable=no-member
    cv2.fillConvexPoly(hull_mask, hull3, (1, ))  # pylint: disable=no-member

    face = np.concatenate((face, hull_mask), -1)
    logger.trace("Returning: face_shape: %s", face.shape)
    return face
