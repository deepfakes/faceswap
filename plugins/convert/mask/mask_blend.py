#!/usr/bin/env python3
""" Adjustments for the mask for faceswap.py converter """

import cv2
import numpy as np

from ._base import Adjustment, BlurMask, logger


class Mask(Adjustment):
    """ Return the requested mask """
    def __init__(self, mask_type, output_size, predicted_available, **kwargs):
        super().__init__(mask_type, output_size, predicted_available, **kwargs)
        self.do_erode = self.config.get("erosion", 0) != 0
        self.do_blend = self.config.get("type", None) is not None

    def process(self, new_face):
        """ Return mask and perform processing """
        mask = new_face[..., 3:]
        print("mask11: ", mask.shape)
        if not self.skip:
            if self.do_erode:
                mask = self.erode(mask)
            if self.do_blend:
                mask = self.blend(mask)
        mask = np.expand_dims(mask, axis=-1) if mask.ndim != 3 else mask
        mask = np.clip(mask, 0., 1.)
        logger.trace("mask shape: %s", mask.shape)
        return mask

    # MASK MANIPULATIONS
    def erode(self, mask):
        """ Erode/dilate mask if requested """
        # pylint: disable=no-member
        kernel = self.get_erosion_kernel(mask)
        if self.config["erosion"] > 0.:
            logger.trace("Eroding mask")
            mask = cv2.erode(mask, kernel, iterations=1)
        else:
            logger.trace("Dilating mask")
            mask = cv2.dilate(mask, kernel, iterations=1)
        return mask

    def get_erosion_kernel(self, mask):
        """ Get the erosion kernel """
        # pylint: disable=no-member
        erosion_ratio = self.config["erosion"] / 100.
        mask_radius = np.sqrt(np.sum(mask)) / 2.
        kernel_size = int(max(1., abs(erosion_ratio * mask_radius)))
        erosion_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        logger.trace("erosion_kernel shape: %s", erosion_kernel.shape)
        return erosion_kernel

    def blend(self, mask):
        """ Blur mask if requested """
        logger.trace("Blending mask")
        raw_mask = BlurMask(self.config["type"],
                            mask,
                            self.config["radius"],
                            self.config["passes"])
        mask = raw_mask.blurred
        print("mask22: ", mask.shape)
        return mask
