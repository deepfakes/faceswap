#!/usr/bin/env python3
""" Adjustments for the mask for faceswap.py converter """

import cv2
import numpy as np

from lib.model import masks as model_masks
from ._base import Adjustment, BlurMask, logger


class Mask(Adjustment):
    """ Return the requested mask """
    def __init__(self, mask_type, output_size, predicted_available, **kwargs):
        super().__init__(mask_type, output_size, predicted_available, **kwargs)
        self.do_erode = self.config.get("erosion", 0) != 0
        self.do_blend = self.config.get("type", None) is not None

    def process(self, detected_face, predicted_mask=None):
        """ Return mask and perform processing """
        mask = self.get_mask(detected_face, predicted_mask)
        raw_mask = mask.copy()
        if not self.skip and self.do_erode:
            mask = self.erode(mask)
        if not self.skip and self.do_blend:
            mask = self.blend(mask)
        raw_mask = np.expand_dims(raw_mask, axis=-1) if raw_mask.ndim != 3 else raw_mask
        mask = np.expand_dims(mask, axis=-1) if mask.ndim != 3 else mask
        logger.trace("mask shape: %s, raw_mask shape: %s", mask.shape, raw_mask.shape)
        return mask, raw_mask

    def get_mask(self, detected_face, predicted_mask):
        """ Return the mask from lib/model/masks and intersect with box """
        if self.mask_type == "none":
            # Return a dummy mask if not using a mask
            mask = np.ones_like(self.dummy[:, :, 1])
        elif self.mask_type == "predicted":
            mask = predicted_mask
        else:
            landmarks = detected_face.reference_landmarks
            mask = getattr(model_masks, self.mask_type)(landmarks, self.dummy, channels=1).mask
        np.nan_to_num(mask, copy=False)
        np.clip(mask, 0.0, 1.0, out=mask)
        return mask

    # MASK MANIPULATIONS
    def erode(self, mask):
        """ Erode/dilate mask if requested """
        kernel = self.get_erosion_kernel(mask)
        if self.config["erosion"] > 0:
            logger.trace("Eroding mask")
            mask = cv2.erode(mask, kernel, iterations=1)  # pylint: disable=no-member
        else:
            logger.trace("Dilating mask")
            mask = cv2.dilate(mask, kernel, iterations=1)  # pylint: disable=no-member
        return mask

    def get_erosion_kernel(self, mask):
        """ Get the erosion kernel """
        erosion_ratio = self.config["erosion"] / 100
        mask_radius = np.sqrt(np.sum(mask)) / 2
        kernel_size = max(1, int(abs(erosion_ratio * mask_radius)))
        erosion_kernel = cv2.getStructuringElement(  # pylint: disable=no-member
            cv2.MORPH_ELLIPSE,  # pylint: disable=no-member
            (kernel_size, kernel_size))
        logger.trace("erosion_kernel shape: %s", erosion_kernel.shape)
        return erosion_kernel

    def blend(self, mask):
        """ Blur mask if requested """
        logger.trace("Blending mask")
        mask = BlurMask(self.config["type"],
                        mask,
                        self.config["radius"],
                        self.config["passes"]).blurred
        return mask
