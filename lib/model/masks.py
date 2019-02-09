#!/usr/bin/env python3
""" Masks functions for faceswap.py
    Masks from:
        dfaker: https://github.com/dfaker/df"""

import logging

import cv2
import numpy as np

from lib.umeyama import umeyama

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def dfaker(landmarks, face, channels=4):
    """ Dfaker model mask
        Embeds the mask into the face alpha channel

        channels: 1, 3 or 4:
                1 - Return a single channel mask
                3 - Return a 3 channel mask
                4 - Return the original image with the mask in the alpha channel
        """
    padding = int(face.shape[0] * 0.1875)
    coverage = face.shape[0] - (padding * 2)
    logger.trace("face_shape: %s, coverage: %s, landmarks: %s", face.shape, coverage, landmarks)

    mat = umeyama(landmarks[17:], True)[0:2]
    mat = np.array(mat.ravel()).reshape(2, 3)
    mat = mat * coverage
    mat[:, 2] += padding

    points = np.array(landmarks).reshape((-1, 2))
    facepoints = np.array(points).reshape((-1, 2))

    mask = np.zeros_like(face, dtype=np.uint8)

    hull = cv2.convexHull(facepoints.astype(int))  # pylint: disable=no-member
    hull = cv2.transform(hull.reshape(1, -1, 2),  # pylint: disable=no-member
                         mat).reshape(-1, 2).astype(int)
    cv2.fillConvexPoly(mask, hull, (255, 255, 255))  # pylint: disable=no-member

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))  # pylint: disable=no-member
    mask = cv2.dilate(mask,  # pylint: disable=no-member
                      kernel,
                      iterations=1,
                      borderType=cv2.BORDER_REFLECT)  # pylint: disable=no-member
    mask = mask[:, :, :1]

    return merge_mask(face, mask, channels)


def dfl_full(landmarks, face, channels=4):
    """ DFL Face Full Mask

        channels: 1, 3 or 4:
            1 - Return a single channel mask
            3 - Return a 3 channel mask
            4 - Return the original image with the mask in the alpha channel
        """
    logger.trace("face_shape: %s, landmarks: %s", face.shape, landmarks)
    mask = np.zeros(face.shape[0:2] + (1, ), dtype=np.float32)
    jaw = cv2.convexHull(np.concatenate((  # pylint: disable=no-member
                                         landmarks[0:17],   # jawline
                                         landmarks[48:68],  # mouth
                                         [landmarks[0]],    # temple
                                         [landmarks[8]],    # chin
                                         [landmarks[16]]))) # temple
    nose_ridge = cv2.convexHull(np.concatenate((  # pylint: disable=no-member
                                                landmarks[27:31],  # nose line
                                                [landmarks[33]]))) # nose point 
    eyes = cv2.convexHull(np.concatenate((  # pylint: disable=no-member
                                          landmarks[17:27],  # eyebrows
                                          [landmarks[0]],    # temple
                                          [landmarks[27]],   # nose top
                                          [landmarks[16]],   # temple
                                          [landmarks[33]]))) # nose point

    cv2.fillConvexPoly(mask, jaw, (255, 255, 255))  # pylint: disable=no-member
    cv2.fillConvexPoly(mask, nose_ridge, (255, 255, 255))  # pylint: disable=no-member
    cv2.fillConvexPoly(mask, eyes, (255, 255, 255))  # pylint: disable=no-member
    return merge_mask(face, mask, channels)


def merge_mask(image, mask, channels):
    """ Return the mask in requested shape """
    logger.trace("image_shape: %s, mask_shape: %s, channels: %s",
                 image.shape, mask.shape, channels)
    assert channels in (1, 3, 4), "Channels should be 1, 3 or 4"
    assert mask.shape[2] == 1 and mask.ndim == 3, "Input mask be 3 dimensions with 1 channel"

    if channels == 3:
        retval = np.tile(mask, 3)
    elif channels == 4:
        retval = np.concatenate((image, mask), -1)
    else:
        retval = mask

    logger.trace("Final mask shape: %s", retval.shape)
    return retval
