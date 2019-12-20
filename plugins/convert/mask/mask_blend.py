#!/usr/bin/env python3
""" Adjustments for the mask for faceswap.py converter """

import cv2
import numpy as np

from ._base import Adjustment, logger


class Mask(Adjustment):
    """ Return the requested mask """
    def __init__(self, mask_type, output_size, coverage_ratio, **kwargs):
        super().__init__(mask_type, output_size, **kwargs)
        self.do_erode = self.config.get("erosion", 0) != 0
        self._coverage_ratio = coverage_ratio

    def process(self, detected_face, predicted_mask=None):
        """ Return mask and perform processing """
        mask = self.get_mask(detected_face, predicted_mask)
        raw_mask = mask.copy()
        if not self.skip and self.do_erode:
            mask = self.erode(mask)
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
            mask = detected_face.mask[self.mask_type]
            mask.set_blur_and_threshold(blur_kernel=self.config["kernel_size"],
                                        blur_type=self.config["type"],
                                        blur_passes=self.config["passes"],
                                        threshold=self.config["threshold"])
            mask = self._crop_to_coverage(mask.mask)

            mask_size = mask.shape[0]
            face_size = self.dummy.shape[0]
            if mask_size != face_size:
                interp = cv2.INTER_CUBIC if mask_size < face_size else cv2.INTER_AREA
                mask = cv2.resize(mask,
                                  self.dummy.shape[:2],
                                  interpolation=interp)[..., None]
        logger.trace(mask.shape)
        return mask

    def _crop_to_coverage(self, mask):
        """ Crap the mask to the correct dimensions based on coverage ratio.

        Parameters
        ----------
        mask: :class:`numpy.ndarray`
            The original mask to be cropped

        Returns
        -------
        :class:`numpy.ndarray`
            The cropped mask
        """
        if self._coverage_ratio == 1.0:
            return mask
        mask_size = mask.shape[0]
        padding = round((mask_size * (1 - self._coverage_ratio)) / 2)
        mask_slice = slice(padding, mask_size - padding)
        mask = mask[mask_slice, mask_slice, :]
        logger.trace("mask_size: %s, coverage: %s, padding: %s, final shape: %s",
                     mask_size, self._coverage_ratio, padding, mask.shape)
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
