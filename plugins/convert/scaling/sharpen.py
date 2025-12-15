#!/usr/bin/env python3
""" Sharpening for enlarged face for faceswap.py converter """
import cv2
import numpy as np

from lib.utils import get_module_objects

from ._base import Adjustment, logger
from . import sharpen_defaults as cfg


class Scaling(Adjustment):
    """ Sharpening Adjustments for the face applied after warp to final frame """

    def process(self, new_face: np.ndarray) -> np.ndarray:
        """ Sharpen using the requested technique

        Parameters
        ----------
        new_face : :class:`numpy.ndarray`
            A batch of swapped image patch that is to have sharpening applied

        Returns
        -------
        :class:`numpy.ndarray`
            The batch of swapped faces with sharpening applied
        """
        if cfg.method() == "none":
            return new_face
        amount = cfg.amount() / 100.0
        kernel, radius = self.get_kernel_size(new_face, cfg.radius())
        new_face = getattr(self, cfg.method())(new_face, kernel, radius, amount)
        return new_face

    @classmethod
    def get_kernel_size(cls,
                        new_face: np.ndarray,
                        radius_percent: float) -> tuple[tuple[int, int], int]:
        """ Return the kernel size and central point for the given radius
            relative to frame width.

        Parameters
        ----------
        new_face : :class:`numpy.ndarray`
            The swapped image patch that is to have sharpening applied

        radius_percent : float
            The percentage of the image size to use as the sharpening kernel

        Returns
        -------
        kernel_size : tuple[int, int]
            The sharpening kernel
        radius : int
            The pixel radius the kernel
        """
        radius = max(1, round(new_face.shape[1] * radius_percent / 100))
        kernel_size = int((radius * 2) + 1)
        full_kernel_size = (kernel_size, kernel_size)
        logger.trace(kernel_size)  # type:ignore[attr-defined]
        return full_kernel_size, radius

    @classmethod
    def box(cls,
            new_face: np.ndarray,
            kernel_size: tuple[int, int],
            radius: int,
            amount: float) -> np.ndarray:
        """ Sharpen using box filter

        Parameters
        ----------
        new_face : :class:`numpy.ndarray`
            The batch of swapped image patches that is to have sharpening applied
        kernel_size : tuple[int, int]
            The sharpening kernel size
        radius : int
            The pixel radius the kernel
        amount : float
            The amount of sharpening to apply

        Returns
        -------
        :class:`numpy.ndarray`
            The batch of swapped faces with box sharpening applied
        """
        kernel: np.ndarray = np.zeros(kernel_size, dtype="float32")
        kernel[radius, radius] = 1.0
        box_filter = np.ones(kernel_size, dtype="float32") / kernel_size[0]**2
        kernel = kernel + (kernel - box_filter) * amount
        new_face = cv2.filter2D(new_face, -1, kernel)
        return new_face

    @classmethod
    def gaussian(cls,
                 new_face: np.ndarray,
                 kernel_size: tuple[int, int],
                 radius: float,  # pylint:disable=unused-argument
                 amount: float) -> np.ndarray:
        """ Sharpen using gaussian filter

        Parameters
        ----------
        new_face : :class:`numpy.ndarray`
            The batch of swapped image patches that is to have sharpening applied
        kernel_size : tuple[int, int]
            The sharpening kernel size
        radius : int
            The pixel radius the kernel. Unused
        amount : float
            The amount of sharpening to apply

        Returns
        -------
        :class:`numpy.ndarray`
            The batch of swapped faces with gaussian sharpening applied
        """
        blur = cv2.GaussianBlur(new_face, kernel_size, 0)
        new_face = cv2.addWeighted(new_face,
                                   1.0 + (0.5 * amount),
                                   blur,
                                   -(0.5 * amount),
                                   0)
        return new_face

    @classmethod
    def unsharp_mask(cls,
                     new_face: np.ndarray,
                     kernel_size: tuple[int, int],
                     center: float,  # pylint:disable=unused-argument
                     amount: float) -> np.ndarray:
        """ Sharpen using unsharp mask

        Parameters
        ----------
        new_face : :class:`numpy.ndarray`
            The batch of swapped image patches that is to have sharpening applied
        kernel_size : tuple[int, int]
            The sharpening kernel size
        radius : int
            The pixel radius the kernel. Unused
        amount : float
            The amount of sharpening to apply

        Returns
        -------
        :class:`numpy.ndarray`
            The batch of swapped faces with unsharp-mask sharpening applied
        """
        threshold = cfg.threshold() / 255.0
        blur = cv2.GaussianBlur(new_face, kernel_size, 0)
        low_contrast_mask = (abs(new_face - blur) < threshold).astype("float32")
        sharpened = (new_face * (1.0 + amount)) + (blur * -amount)
        new_face = (new_face * (1.0 - low_contrast_mask)) + (sharpened * low_contrast_mask)
        return new_face


__all__ = get_module_objects(__name__)
