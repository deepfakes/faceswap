#!/usr/bin/env python3
"""Handles the creation of display images for preview window and timelapses """
from __future__ import annotations

import logging
import typing as T

import cv2
import numpy as np

from lib.logger import format_array, parse_class_init
from lib.image import hex_to_rgb
from lib.utils import get_module_objects
from lib.training.data import get_label

if T.TYPE_CHECKING:
    import numpy.typing as npt

logger = logging.getLogger(__name__)


class Samples():
    """Compile samples for display for preview and time-lapse

    Parameters
    ----------
    coverage_ratio
        Ratio of face to be cropped out of the training image.
    has_mask
        ``True`` if the model was trained with a mask
    mask_opacity
        The opacity (as a percentage) to use for the mask overlay
    mask_color
        The hex RGB value to use the mask overlay
    """
    def __init__(self,
                 coverage_ratio: float,
                 has_mask: bool,
                 mask_opacity: int,
                 mask_color: str) -> None:
        logger.debug(parse_class_init(locals()))
        self._coverage_ratio = coverage_ratio
        self._has_mask = has_mask
        self._mask_opacity = mask_opacity / 100.0
        self._mask_color = mask_color
        self._mask_color_array = (
            np.array(hex_to_rgb(mask_color),
                     dtype=np.float32)[..., 2::-1] / 255.).astype(np.float32)

        self._name = self.__class__.__name__
        self._display_mask = has_mask

    def __repr__(self) -> str:
        """Pretty print for logging"""
        params = ", ".join(f"{k[1:]}={repr(v)}" for k, v in self.__dict__.items()
                           if k in ("_coverage_ratio", "_has_mask", "_mask_opacity",
                                    "_mask_color"))
        return f"{self._name}({params})"

    def toggle_mask_display(self) -> None:
        """Toggle the mask overlay on or off depending on user input."""
        if not self._has_mask:
            return
        display_mask = not self._display_mask
        print("\x1b[2K", end="\r")  # Clear last line
        logger.info("Toggling mask display %s...", "on" if display_mask else "off")
        self._display_mask = display_mask

    def _get_background(self,
                        targets: npt.NDArray[np.float32],
                        patch_size: int,
                        padding: int) -> npt.NDArray[np.float32]:
        """Obtain the images that will hold the background stacked as (src>dst, samples, width,
        height, 3)

        For 100% coverage just the source (ground truth) images will be populated, otherwise all
        backgrounds are populated from the ground truth and the crop area box is created

        Parameters
        ----------
        targets
            The The (BGR) targets shape: (src_side, batch_size, height, width, channels)
        patch_size
            The size of each final face patch
        padding
            The padding required to place the prediction within the final patch

        Returns
        -------
        The background image patches shaped (src_side, num_src + 1, batch_size, height, width, 3)
        """
        num_swaps = targets.shape[0]
        assert self._coverage_ratio != 1.0, "Background only required for coverage != 1.0"
        retval = np.empty((num_swaps, num_swaps + 1, *targets.shape[1:4], 3), dtype=np.float32)
        length = patch_size // 4
        t_l, b_r = (padding - 1, patch_size - padding + 1)
        retval[:] = np.repeat(targets[:, None, ..., :3], 3, axis=1)
        retval[:, :, :, t_l:t_l + length, t_l:t_l + length] = self._mask_color_array
        retval[:, :, :, t_l:t_l + length, b_r - length:b_r] = self._mask_color_array
        retval[:, :, :, b_r - length:b_r, b_r - length:b_r] = self._mask_color_array
        retval[:, :, :, b_r - length:b_r, t_l:t_l + length] = self._mask_color_array
        logger.debug("[%s] Created background display patches: %s",
                     self._name, format_array(retval))
        return retval

    def _get_foreground(self,
                        predictions: npt.NDArray[np.float32],
                        targets: npt.NDArray[np.float32],
                        patch_size: int,
                        padding: int) -> npt.NDArray[np.float32]:
        """Obtain the foreground patches for overlaying on the backgrounds, with any mask
        application applied

        Parameters
        ----------
        predictions
            The The (BGR) predictions shape: (src_side, dst_side, batch_size, height, width,
            channels)
        targets
            The The (BGR) targets shape: (src_side, batch_size, height, width, channels)
        patch_size
            The size of each final face patch
        padding
            The padding required to place the prediction within the final patch

        Returns
        -------
        The foreground image patches shaped (src_side, num_src + 1, batch_size, height, width, 3)
        """
        num_swaps = predictions.shape[0]
        retval = np.empty((num_swaps, num_swaps + 1, *predictions.shape[2:5], 3),
                          dtype=np.float32)

        retval[:, 1:] = predictions[..., :3]

        if self._coverage_ratio == 1.:
            retval[:, 0] = targets[..., :3]
        else:
            retval[:, 0] = targets[:,
                                   :,
                                   padding:patch_size - padding,
                                   padding:patch_size - padding,
                                   :3]

        logger.debug("[%s] Created foreground display patches: %s",
                     self._name, format_array(retval))
        return retval

    def _apply_masks(self,
                     patches: npt.NDArray[np.float32],
                     predictions: npt.NDArray[np.float32],
                     targets: npt.NDArray[np.float32],
                     patch_size: int,
                     padding: int) -> npt.NDArray[np.float32]:
        """Apply the masks to the final patches, if requested

        Parameters
        ----------
        image
            The image patches shaped (src_side, num_src + 1, batch_size, height, width, 3) to have
            masks applied
        predictions
            The The (BGR) predictions shape: (src_side, dst_side, batch_size, height, width,
            channels)
        targets
            The The (BGR) targets shape: (src_side, batch_size, height, width, channels)
        patch_size
            The size of each final face patch
        padding
            The padding required to place the prediction within the final patch
        """
        if not self._display_mask:
            return patches

        if predictions.shape[-1] == 4:  # Learn mask is enabled
            masks = np.zeros(patches.shape[:-1], dtype=np.float32)
            masks[:, 0] = targets[..., -1]
            pred = predictions[..., -1]

            if self._coverage_ratio == 1.0:
                masks[:, 1:] = pred
            else:
                masks[:, 1:, :, padding:patch_size - padding, padding:patch_size - padding] = pred
        else:
            masks = np.repeat(targets[:, None, ..., -1], 3, axis=1)
        masks = 1. - masks
        overlay = np.ones_like(patches, dtype=np.float32) * self._mask_color_array
        masks *= self._mask_opacity
        overlay *= masks[..., None]
        patches *= (1. - masks[..., None])
        retval = patches + T.cast("npt.NDArray[np.float32]", overlay)
        logger.debug("[%s] Applied masks: %s", self._name, format_array(retval))
        return retval

    def _get_headers(self, num_swaps: int, patch_width: int  # pylint:disable=too-many-locals
                     ) -> npt.NDArray[np.uint8]:
        """Set header row for the final preview frame

        Parameters
        ----------
        num_swaps
            The number of swap instances exist within the model
        patch_width
            The width of each of the display patches

        Returns
        -------
        The column headings for the output image
        """
        labels = [
            get_label(i, num_swaps) + (f" > {get_label(i + j, num_swaps, next_identity=True)}"
                                       if j > 0 else "")
            for i in range(num_swaps)
            for j in range(num_swaps + 1)
        ]
        cols = len(labels)
        height = int(patch_width / 4.5)
        headers = np.zeros((cols, height, patch_width, 3), dtype="uint8") + 255
        font = cv2.FONT_HERSHEY_SIMPLEX
        scaling = patch_width / 140
        text_sizes = [cv2.getTextSize(labels[idx], font, scaling, 1)[0]
                      for idx in range(len(labels))]
        t_y = int((height + text_sizes[0][1]) / 2)
        t_x = [int((patch_width - text_sizes[i][0]) / 2) for i in range(cols)]
        thickness = max(1, patch_width // 64)
        logger.debug("[%s] labels: %s, text_sizes: %s, text_x: %s, text_y: %s, thickness: %s, "
                     "scaling: %s",
                     self._name, labels, text_sizes, t_x, t_y, thickness, scaling)
        for idx, (text, header) in enumerate(zip(labels, headers)):
            cv2.putText(header,
                        text,
                        (t_x[idx], t_y),
                        font,
                        scaling,
                        (0, 0, 0),
                        thickness,
                        lineType=cv2.LINE_AA)
        retval = headers.swapaxes(0, 1).reshape((height, patch_width * cols, 3))
        logger.debug("[%s] Headers: %s", self._name, format_array(retval))
        return retval

    def _create_image(self, patches: npt.NDArray[np.float32]) -> npt.NDArray[np.uint8]:
        """Create the final laid out image display with headers

        Parameters
        ----------
        patches
            The final image patches shaped (src_side, num_src + 1, batch_size, height, width, 3)

        Returns
        -------
        The final preview image
        """
        headers = self._get_headers(patches.shape[0], patches.shape[-2])
        src_side, img_count, identities, rows, cols, channels = patches.shape
        images = (patches.transpose(2, 3, 0, 1, 4, 5).reshape((rows * identities,
                                                               cols * src_side * img_count,
                                                               channels)) * 255.).astype(np.uint8)
        if images.shape[0] > images.shape[1]:
            height = len(images) // 2
            images = np.concatenate([images[:height], images[height:]], axis=1)
            headers = np.concatenate([headers, headers], axis=1)
        retval = np.concatenate([headers, images], axis=0)
        logger.debug("[%s] Created preview: %s", self._name, format_array(retval))
        return retval

    def get_preview(self, predictions: npt.NDArray[np.float32], targets: npt.NDArray[np.float32]
                    ) -> npt.NDArray[np.uint8]:
        """Compile a preview image.

        Predictions
            The (BGR) predictions shape: (src_side, dst_side, batch_size, height, width, channels)
        targets
            Full size BGR face patches at 100% coverage for patching predictions into in
            (A, B, ...) order

        Returns
        -------
        A compiled preview image ready for display or saving
        """
        patch_size = targets.shape[-2]
        pad = (patch_size - predictions.shape[-2]) // 2

        logger.debug("[%s] Showing sample. Predictions: %s, targets: %s, patch_size: %s, "
                     "padding: %s",
                     self._name, format_array(predictions), format_array(targets),
                     patch_size, pad)

        foreground = self._get_foreground(predictions, targets, patch_size, pad)

        if self._coverage_ratio != 1.0:
            patches = self._get_background(targets, patch_size, pad)
            patches[:, :, :, pad:patch_size - pad, pad:patch_size - pad] = foreground
        else:
            patches = foreground

        patches = self._apply_masks(patches, predictions, targets, patch_size, pad)
        return self._create_image(patches)


__all__ = get_module_objects(__name__)
