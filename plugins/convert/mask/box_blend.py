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
        self.mask = self.get_mask() if not self.skip else None

    def get_mask(self):
        """ The box for every face will be identical, so set the mask just once
            As gaussian blur technically blurs both sides of the mask, reduce the mask ratio by
            half to give a more expected box """
        logger.debug("Building box mask")
        mask_ratio = self.config["distance"] / 200
        facesize = self.dummy.shape[0]
        erode = slice(round(facesize * mask_ratio), -round(facesize * mask_ratio))
        mask = self.dummy[:, :, -1]
        mask[erode, erode] = 1.0

        mask = BlurMask(self.config["type"],
                        mask,
                        self.config["radius"],
                        self.config["passes"]).blurred
        logger.debug("Built box mask. Shape: %s", mask.shape)
        return mask

    def process(self, new_face):
        """ The blend box function. Adds the created mask to the alpha channel """
        if self.skip:
            logger.trace("Skipping blend box")
            return new_face

        logger.trace("Blending box")
        mask = np.expand_dims(self.mask, axis=-1)
        new_face = np.clip(np.concatenate((new_face, mask), axis=-1), 0.0, 1.0)
        logger.trace("Blended box")
        return new_face
