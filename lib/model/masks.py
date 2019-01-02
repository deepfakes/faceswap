#!/usr/bin/env python3
""" Masks functions for faceswap.py
    Masks from:
        dfaker: https://github.com/dfaker/df"""

import cv2
import numpy as np

from lib.umeyama import umeyama
from lib.aligner import LANDMARKS_2D


def dfaker_mask(landmarks, coverage, face):
    """ Dfaker model mask """
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

    return merged
