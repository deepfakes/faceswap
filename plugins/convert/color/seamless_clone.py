#!/usr/bin/env python3
""" Seamless clone adjustment plugin for faceswap.py converter
    NB: This probably isn't the best place for this, but it is independent of
        color adjustments and does not have a natural home, so here for now
        and called as an extra plugin from lib/convert.py
"""

import cv2
import numpy as np
from ._base import Adjustment


class Color(Adjustment):
    """ Seamless clone the swapped face into the old face with cv2
        NB: This probably isn't the best place for this, but it doesn't work well and
        and does not have a natural home, so here for now.
    """

    def process(self, old_face, new_face, raw_mask):
        height, width, _ = old_face.shape
        height = height // 2
        width = width // 2

        y_indices, x_indices, _ = np.nonzero(raw_mask)
        y_crop = slice(np.min(y_indices), np.max(y_indices))
        x_crop = slice(np.min(x_indices), np.max(x_indices))
        y_center = int(np.rint((np.max(y_indices) + np.min(y_indices)) / 2 + height))
        x_center = int(np.rint((np.max(x_indices) + np.min(x_indices)) / 2 + width))

        insertion = np.rint(new_face[y_crop, x_crop] * 255.0).astype("uint8")
        insertion_mask = np.rint(raw_mask[y_crop, x_crop] * 255.0).astype("uint8")
        insertion_mask[insertion_mask != 0] = 255
        prior = np.rint(np.pad(old_face * 255.0,
                               ((height, height), (width, width), (0, 0)),
                               'constant')).astype("uint8")

        blended = cv2.seamlessClone(insertion,  # pylint:disable=no-member
                                    prior,
                                    insertion_mask,
                                    (x_center, y_center),
                                    cv2.NORMAL_CLONE)  # pylint:disable=no-member
        blended = blended[height:-height, width:-width]

        return blended.astype("float32") / 255.0
