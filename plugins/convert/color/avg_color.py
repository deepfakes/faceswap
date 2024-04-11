#!/usr/bin/env python3
""" Average colour adjustment color matching adjustment plugin for faceswap.py converter """

import numpy as np
from ._base import Adjustment


class Color(Adjustment):
    """ Adjust the mean of the color channels to be the same for the swap and old frame """

    def process(self,
                old_face: np.ndarray,
                new_face: np.ndarray,
                raw_mask: np.ndarray) -> np.ndarray:
        """ Adjust the mean of the original face and the new face to be the same

        Parameters
        ----------
        old_face: :class:`numpy.ndarray`
            The original face
        new_face: :class:`numpy.ndarray`
            The Faceswap generated face
        raw_mask: :class:`numpy.ndarray`
            A raw mask for including the face area only

        Returns
        -------
        :class:`numpy.ndarray`
            The adjusted face patch
        """
        for _ in [0, 1]:
            diff = old_face - new_face
            if np.any(raw_mask):
                avg_diff = np.sum(diff * raw_mask, axis=(0, 1))
                adjustment = avg_diff / np.sum(raw_mask, axis=(0, 1))
            else:
                adjustment = diff
            new_face += adjustment
        return new_face
