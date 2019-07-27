#!/usr/bin/env python3
""" Adjustments for the swap box for faceswap.py converter """

import numpy as np

from ._base import Adjustment, BlurMask, logger


class Mask(Adjustment):
    """ Manipulations that occur on the swap box
        Actions performed here occur prior to warping the face back to the background frame

        For actions that occur identically for each frame (e.g. blend_box), constants can
        be placed into self.func_constants to be compiled at launch, then referenced for
        each face. """
    def __init__(self, mask_type, output_size, predicted_available=False, **kwargs):
        super().__init__(mask_type, output_size, predicted_available, **kwargs)

    def process(self, new_face):
        """ The blend box function. Adds the created mask to the alpha channel """
        if self.skip:
            logger.trace("Skipping blend box")
        else:
            mask = self.get_mask()
            new_face[:, :, -1] = np.minimum(new_face[:, :, -1:], mask)
            logger.trace("Blended box")
        return new_face

    def get_mask(self):
        """ The box for every face will be identical, so set the mask just once
            As gaussian blur technically blurs both sides of the mask, reduce the mask ratio by
            half to give a more expected box """
        logger.debug("Building box mask")
        mask_ratio = self.config["distance"] / 200.
        erode_size = round(self.output_size * mask_ratio)
        erode = slice(erode_size, -erode_size)
        mask = np.zeros((self.output_size, self.output_size, 1), dtype='float32')
        mask[erode, erode] = 1.
        raw_mask = BlurMask(mask,
                            self.config["type"],
                            self.config["radius"],
                            self.config["passes"])
        mask = raw_mask.blurred
        logger.debug("Built box mask. Shape: %s", mask.shape)
        return mask
