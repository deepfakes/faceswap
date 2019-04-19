#!/usr/bin/env python3
""" Match histogram colour adjustment color matching adjustment plugin
    for faceswap.py converter """

import numpy as np
from ._base import Adjustment


class Color(Adjustment):
    """ Match the histogram of the color intensity of each channel """

    def process(self, old_face, new_face, raw_mask):
        mask_indices = np.nonzero(raw_mask.squeeze())
        new_face = [self.hist_match(old_face[:, :, c],
                                    new_face[:, :, c],
                                    mask_indices,
                                    self.config["threshold"] / 100)
                    for c in range(3)]
        new_face = np.stack(new_face, axis=-1)
        return new_face

    @staticmethod
    def hist_match(old_channel, new_channel, mask_indices, threshold):
        """  Construct the histogram of the color intensity of a channel
             for the swap and the original. Match the histogram of the original
             by interpolation
        """
        if mask_indices[0].size == 0:
            return new_channel

        old_masked = old_channel[mask_indices]
        new_masked = new_channel[mask_indices]
        _, bin_idx, s_counts = np.unique(new_masked, return_inverse=True, return_counts=True)
        t_values, t_counts = np.unique(old_masked, return_counts=True)
        s_quants = np.cumsum(s_counts, dtype='float32')
        t_quants = np.cumsum(t_counts, dtype='float32')
        s_quants = threshold * s_quants / s_quants[-1]  # cdf
        t_quants /= t_quants[-1]  # cdf
        interp_s_values = np.interp(s_quants, t_quants, t_values)
        new_channel[mask_indices] = interp_s_values[bin_idx]
        return new_channel
