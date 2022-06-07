#!/usr/bin/env python3
""" Plugin to blend the edges of the face between the swap and the original face. """
from typing import List, Literal, Optional, Tuple, TYPE_CHECKING

import cv2
import numpy as np

from ._base import Adjustment, logger

if TYPE_CHECKING:
    from lib.align import DetectedFace


class Mask(Adjustment):
    """ Manipulations to perform to the mask that is to be applied to the output of the Faceswap
    model.

    Parameters
    ----------
    mask_type: str
        The mask type to use for this plugin
    output_size: int
        The size of the output from the Faceswap model.
    coverage_ratio: float
        The coverage ratio that the Faceswap model was trained at.
    **kwargs: dict, optional
        See the parent :class:`~plugins.convert.mask._base` for additional keyword arguments.
    """
    def __init__(self, mask_type: str, output_size: int, coverage_ratio: float, **kwargs):
        super().__init__(mask_type, output_size, **kwargs)

        erode_types = [f"erosion{f}" for f in ["", "_left", "_top", "_right", "_bottom"]]
        self._erodes = [self.config.get(erode, 0) / 100 for erode in erode_types]
        self._do_erode = any(amount != 0 for amount in self._erodes)

        self._coverage_ratio = coverage_ratio

    def process(self,  # type:ignore # pylint:disable=arguments-differ
                detected_face: "DetectedFace",
                sub_crop_offset: Optional[np.ndarray],
                centering: Literal["legacy", "face", "head"],
                predicted_mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """ Obtain the requested mask type and perform any defined mask manipulations.

        Parameters
        ----------
        detected_face: :class:`lib.align.DetectedFace`
            The DetectedFace object as returned from :class:`scripts.convert.Predictor`.
        sub_crop_offset: :class:`numpy.ndarray`, optional
            The (x, y) offset to crop the mask from the center point.
        centering: [`"legacy"`, `"face"`, `"head"`]
            The centering to obtain the mask for
        predicted_mask: :class:`numpy.ndarray`, optional
            The predicted mask as output from the Faceswap Model, if the model was trained
            with a mask, otherwise ``None``. Default: ``None``.

        Returns
        -------
        mask: :class:`numpy.ndarray`
            The mask with all requested manipulations applied
        raw_mask: :class:`numpy.ndarray`
            The mask with no erosion/dilation applied
        """
        logger.trace(  # type: ignore
            "detected_face: %s, sub_crop_offset: %s, centering: '%s', predicted_mask: %s",
            detected_face, sub_crop_offset, centering, predicted_mask is not None)
        mask = self._get_mask(detected_face, predicted_mask, centering, sub_crop_offset)
        raw_mask = mask.copy()
        if not self.skip and self._do_erode:
            mask = self._erode(mask)
        logger.trace(  # type: ignore
            "mask shape: %s, raw_mask shape: %s", mask.shape, raw_mask.shape)
        return mask, raw_mask

    def _get_mask(self,
                  detected_face: "DetectedFace",
                  predicted_mask: Optional[np.ndarray],
                  centering: Literal["legacy", "face", "head"],
                  sub_crop_offset: Optional[np.ndarray]) -> np.ndarray:
        """ Return the requested mask with any requested blurring applied.

        Parameters
        ----------
        detected_face: :class:`lib.align.DetectedFace`
            The DetectedFace object as returned from :class:`scripts.convert.Predictor`.
        predicted_mask: :class:`numpy.ndarray`
            The predicted mask as output from the Faceswap Model if the model was trained
            with a mask, otherwise ``None``
        centering: [`"legacy"`, `"face"`, `"head"`]
            The centering to obtain the mask for
        sub_crop_offset: :class:`numpy.ndarray`
            The (x, y) offset to crop the mask from the center point. Set to `None` if the mask
            does not need to be offset for alternative centering

        Returns
        -------
        :class:`numpy.ndarray`
            The mask sized to Faceswap model output with any requested blurring applied.
        """
        if self.mask_type == "none":
            # Return a dummy mask if not using a mask
            mask = np.ones_like(self.dummy[:, :, 1], dtype="float32")[..., None]
        elif self.mask_type == "predicted" and predicted_mask is not None:
            mask = predicted_mask[..., None]
        else:
            mask = detected_face.mask[self.mask_type]
            mask.set_blur_and_threshold(blur_kernel=self.config["kernel_size"],
                                        blur_type=self.config["type"],
                                        blur_passes=self.config["passes"],
                                        threshold=self.config["threshold"])
            if sub_crop_offset is not None and np.any(sub_crop_offset):
                mask.set_sub_crop(sub_crop_offset, centering)
            mask = self._crop_to_coverage(mask.mask)
            mask_size = mask.shape[0]
            face_size = self.dummy.shape[0]
            if mask_size != face_size:
                interp = cv2.INTER_CUBIC if mask_size < face_size else cv2.INTER_AREA
                mask = cv2.resize(mask,
                                  self.dummy.shape[:2],
                                  interpolation=interp)[..., None]
            mask = mask.astype("float32") / 255.0
        logger.trace(mask.shape)  # type: ignore
        return mask

    def _crop_to_coverage(self, mask: np.ndarray) -> np.ndarray:
        """ Crop the mask to the correct dimensions based on coverage ratio.

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
        logger.trace("mask_size: %s, coverage: %s, padding: %s, final shape: %s",  # type: ignore
                     mask_size, self._coverage_ratio, padding, mask.shape)
        return mask

    # MASK MANIPULATIONS
    def _erode(self, mask: np.ndarray) -> np.ndarray:
        """ Erode or dilate mask the mask based on configuration options.

        Parameters
        ----------
        mask: :class:`numpy.ndarray`
            The mask to be eroded or dilated

        Returns
        -------
        :class:`numpy.ndarray`
            The mask with erosion/dilation applied
        """
        kernels = self._get_erosion_kernels(mask)
        if not any(k.any() for k in kernels):
            return mask  # No kernels could be created from selected input res
        eroded = []
        for idx, (kernel, ratio) in enumerate(zip(kernels, self._erodes)):
            if not kernel.any():
                continue
            anchor = [-1, -1]
            if idx > 0:
                pos = 1 if idx % 2 == 0 else 0
                val = max(kernel.shape) - 1 if idx < 3 else 0
                anchor[pos] = val
            func = cv2.erode if ratio > 0 else cv2.dilate
            eroded.append(func(mask, kernel, iterations=1, anchor=anchor))

        mask = np.min(np.array(eroded), axis=0) if len(eroded) > 1 else eroded[0]
        return mask

    def _get_erosion_kernels(self, mask: np.ndarray) -> List[np.ndarray]:
        """ Get the erosion kernels for each of the center, left, top right and bottom erosions.

        An approximation is made based on the number of positive pixels within the mask to create
        an ellipse to act as kernel.

        Parameters
        ----------
        mask: :class:`numpy.ndarray`
            The mask to be eroded or dilated

        Returns
        -------
        list
            The erosion kernels to be used for erosion/dilation
        """
        mask_radius = np.sqrt(np.sum(mask)) / 2
        kernel_sizes = [max(0, int(abs(ratio * mask_radius))) for ratio in self._erodes]
        kernels = []
        for idx, size in enumerate(kernel_sizes):
            kernel = [size, size]
            shape = cv2.MORPH_ELLIPSE if idx == 0 else cv2.MORPH_RECT
            if idx > 1:
                pos = 0 if idx % 2 == 0 else 1
                kernel[pos] = 1  # Set x/y to 1px based on whether eroding top/bottom, left/right
            kernels.append(cv2.getStructuringElement(shape, kernel) if size else np.array(0))
        logger.trace("Erosion kernels: %s", [k.shape for k in kernels])  # type: ignore
        return kernels
