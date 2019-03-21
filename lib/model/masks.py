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
    mat = mat.reshape(-1).reshape(2, 3)
    mat = mat * coverage
    mat[:, 2] += padding

    mask = np.zeros(face.shape[0:2] + (1, ), dtype=np.float32)
    hull = cv2.convexHull(landmarks).reshape(1, -1, 2)  # pylint: disable=no-member
    hull = cv2.transform(hull, mat).reshape(-1, 2)  # pylint: disable=no-member
    cv2.fillConvexPoly(mask, hull, 255.)  # pylint: disable=no-member

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))  # pylint: disable=no-member
    mask = cv2.dilate(mask, kernel, borderType=cv2.BORDER_REFLECT)  # pylint: disable=no-member
    mask = np.expand_dims(mask, axis=-1)

    return merge_mask(face, mask, channels)


def dfl_full(landmarks, face, channels=4):
    """ DFL facial mask

        channels: 1, 3 or 4:
            1 - Return a single channel mask
            3 - Return a 3 channel mask
            4 - Return the original image with the mask in the alpha channel
        """
    logger.trace("face_shape: %s, landmarks: %s", face.shape, landmarks)
    mask = np.zeros(face.shape[0:2] + (1, ), dtype=np.float32)

    nose_ridge = (landmarks[27:31], landmarks[33:34])
    jaw = (landmarks[0:17], landmarks[48:68], landmarks[0:1],
           landmarks[8:9], landmarks[16:17])
    eyes = (landmarks[17:27], landmarks[0:1], landmarks[27:28],
            landmarks[16:17], landmarks[33:34])
    parts = [jaw, nose_ridge, eyes]

    for item in parts:
        merged = np.concatenate(item)
        cv2.fillConvexPoly(mask, cv2.convexHull(merged), 255.)  # pylint: disable=no-member

    return merge_mask(face, mask, channels)


def components(landmarks, face, channels=4):
    """ Component model mask

        channels: 1, 3 or 4:
            1 - Return a single channel mask
            3 - Return a 3 channel mask
            4 - Return the original image with the mask in the alpha channel
        """
    logger.trace("face_shape: %s, landmarks: %s", face.shape, landmarks)
    mask = np.zeros(face.shape[0:2] + (1, ), dtype=np.float32)

    r_jaw = (landmarks[0:9], landmarks[17:18])
    l_jaw = (landmarks[8:17], landmarks[26:27])
    r_cheek = (landmarks[17:20], landmarks[8:9])
    l_cheek = (landmarks[24:27], landmarks[8:9])
    nose_ridge = (landmarks[19:25], landmarks[8:9],)
    r_eye = (landmarks[17:22], landmarks[27:28],
             landmarks[31:36], landmarks[8:9])
    l_eye = (landmarks[22:27], landmarks[27:28],
             landmarks[31:36], landmarks[8:9])
    nose = (landmarks[27:31], landmarks[31:36])
    parts = [r_jaw, l_jaw, r_cheek, l_cheek, nose_ridge, r_eye, l_eye, nose]

    for item in parts:
        merged = np.concatenate(item)
        cv2.fillConvexPoly(mask, cv2.convexHull(merged), 255.)  # pylint: disable=no-member

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
