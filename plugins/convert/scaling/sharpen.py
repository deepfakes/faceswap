#!/usr/bin/env python3
""" Sharpening for enlarged face for faceswap.py converter """
import cv2
import numpy as np

from ._base import Adjustment, logger


class Scaling(Adjustment):
    """ Sharpening Adjustments for the face applied after warp to final frame """

    def process(self, new_face):
        """ Sharpen using the requested technique """
        amount = self.config["amount"] / 100.0
        kernel_center = self.get_kernel_size(new_face, self.config["radius"])
        new_face = getattr(self, self.config["method"])(new_face, kernel_center, amount)
        return new_face

    @staticmethod
    def get_kernel_size(new_face, radius_percent):
        """ Return the kernel size and central point for the given radius
            relative to frame width """
        radius = max(1, round(new_face.shape[1] * radius_percent / 100))
        kernel_size = int((radius * 2) + 1)
        kernel_size = (kernel_size, kernel_size)
        logger.trace(kernel_size)
        return kernel_size, radius

    @staticmethod
    def box(new_face, kernel_center, amount):
        """ Sharpen using box filter """
        kernel_size, center = kernel_center
        kernel = np.zeros(kernel_size, dtype="float32")
        kernel[center, center] = 1.0
        box_filter = np.ones(kernel_size, dtype="float32") / kernel_size[0]**2
        kernel = kernel + (kernel - box_filter) * amount
        new_face = cv2.filter2D(new_face, -1, kernel)  # pylint:disable=no-member
        return new_face

    @staticmethod
    def gaussian(new_face, kernel_center, amount):
        """ Sharpen using gaussian filter """
        kernel_size = kernel_center[0]
        blur = cv2.GaussianBlur(new_face, kernel_size, 0)  # pylint:disable=no-member
        new_face = cv2.addWeighted(new_face,  # pylint:disable=no-member
                                   1.0 + (0.5 * amount),
                                   blur,
                                   -(0.5 * amount),
                                   0)
        return new_face

    def unsharp_mask(self, new_face, kernel_center, amount):
        """ Sharpen using unsharp mask """
        kernel_size = kernel_center[0]
        threshold = self.config["threshold"] / 255.0
        blur = cv2.GaussianBlur(new_face, kernel_size, 0)  # pylint:disable=no-member
        low_contrast_mask = (abs(new_face - blur) < threshold).astype("float32")
        sharpened = (new_face * (1.0 + amount)) + (blur * -amount)
        new_face = (new_face * (1.0 - low_contrast_mask)) + (sharpened * low_contrast_mask)
        return new_face
