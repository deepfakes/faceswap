#!/usr/bin/env python3
""" Average colour adjustment color matching adjustment plugin for faceswap.py converter """

import numpy as np
from ._base import Adjustment


class Color(Adjustment):
    """ Adjust the mean of the color channels to be the same for the swap and old frame """

    @staticmethod
    def process(old_face, new_face, raw_mask):
        for _ in [0, 1]:
            diff = old_face - new_face
            avg_diff = np.sum(diff * raw_mask, axis=(0, 1))
            adjustment = avg_diff / np.sum(raw_mask, axis=(0, 1))
            new_face += adjustment
        return new_face
