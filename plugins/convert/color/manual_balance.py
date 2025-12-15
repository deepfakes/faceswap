#!/usr/bin/env python3
""" Manual Balance colour adjustment plugin for faceswap.py converter """

import cv2
import numpy as np
from lib.utils import get_module_objects
from ._base import Adjustment
from . import manual_balance_defaults as cfg


class Color(Adjustment):
    """ Adjust the mean of the color channels to be the same for the swap and old frame """

    def process(self, old_face, new_face, raw_mask):
        image = self.convert_colorspace(new_face * 255.0)
        adjustment = np.array([cfg.balance_1() / 100.0,
                               cfg.balance_2() / 100.0,
                               cfg.balance_3() / 100.0]).astype("float32")
        for idx in range(3):
            if adjustment[idx] >= 0:
                image[:, :, idx] = ((1 - image[:, :, idx]) * adjustment[idx]) + image[:, :, idx]
            else:
                image[:, :, idx] = image[:, :, idx] * (1 + adjustment[idx])

        image = self.convert_colorspace(image * 255.0, to_bgr=True)
        image = self.adjust_contrast(image)
        return image

    def adjust_contrast(self, image):
        """
        Adjust image contrast and brightness.
        """
        contrast = max(-126, int(round(cfg.contrast() * 1.27)))
        brightness = max(-126, int(round(cfg.brightness() * 1.27)))

        if not contrast and not brightness:
            return image

        image = np.rint(image * 255.0).astype("uint8")
        image = np.clip(image * (contrast/127+1) - contrast + brightness, 0, 255)
        image = np.clip(np.divide(image, 255, dtype=np.float32), .0, 1.0)
        return image

    def convert_colorspace(self, new_face, to_bgr=False):
        """ Convert colorspace based on mode or back to bgr """
        mode = cfg.colorspace().lower()
        colorspace = "YCrCb" if mode == "ycrcb" else mode.upper()
        conversion = f"{colorspace}2BGR" if to_bgr else f"BGR2{colorspace}"
        image = cv2.cvtColor(new_face.astype("uint8"),  # pylint:disable=no-member
                             getattr(cv2, f"COLOR_{conversion}")).astype("float32") / 255.0
        return image


__all__ = get_module_objects(__name__)
