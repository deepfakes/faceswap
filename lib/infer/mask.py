#! /usr/env/bin/python3
"""Handles face masking plugins and runners """
from __future__ import annotations

import logging
import typing as T

import cv2
import numpy as np

from lib.logger import parse_class_init
from lib.utils import get_module_objects
from plugins.extract import extract_config as cfg

from .objects import ExtractBatchMask
from .handler import ExtractHandlerFace

if T.TYPE_CHECKING:
    import numpy.typing as npt
    from .objects import ExtractBatch

logger = logging.getLogger(__name__)


class Mask(ExtractHandlerFace):
    """Responsible for running Masking plugins within the extract pipeline

    Parameters
    ----------
    plugin
        The plugin that this runner is to use
    compile_model
        ``True`` to compile any PyTorch models
    config_file
        Full path to a custom config file to load. ``None`` for default config
    """
    def __init__(self,
                 plugin: str,
                 compile_model: bool = False,
                 config_file: str | None = None) -> None:
        logger.debug(parse_class_init(locals()))
        self._storage_size = cfg.mask_storage_size()
        super().__init__(plugin, compile_model=compile_model, config_file=config_file)
        if 0 < self._storage_size < 64:
            logger.warning("Updating mask storage size from %s to 64", self._storage_size)
            self._storage_size = 64

    # Pre-processing
    def _pre_process_aligned(self, batch: ExtractBatch, matrices: npt.NDArray[np.float32]
                             ) -> npt.NDArray[np.uint8]:
        """Pre-process the data when the input are aligned faces. Sub-crops the feed images from
        the aligned images and adds the ROI mask to the alpha channel

        Parameters
        ----------
        batch
            The inbound batch object containing aligned faces
        matrices
            The adjustment matrices for taking the image patch from the full frame for plugin input

        Returns
        -------
        The prepared images with ROI mask in the alpha channel
        """
        assert batch.frame_sizes is not None, (
            "[Mask] Frame sizes must be provided when input is aligned faces")

        dtype = batch.images[0].dtype
        retval = np.empty((len(batch.bboxes), self._input_size, self._input_size, 4), dtype=dtype)
        retval[..., :3] = self._get_faces_aligned(batch.images,
                                                  batch.frame_ids,
                                                  batch.aligned.offsets_head,
                                                  getattr(batch.aligned,
                                                          self._aligned_offsets_name))

        mats = matrices[:, :2]
        linear = mats[:, :, 0]
        scales = np.hypot(linear[:, 0], linear[:, 1])  # Always same x/y scaling
        interpolations = np.where(scales > 1.0, cv2.INTER_LINEAR, cv2.INTER_AREA)
        size = (self._input_size, self._input_size)
        for idx, (mat, interpolation) in enumerate(zip(mats, interpolations)):
            mask = np.ones((batch.frame_sizes[batch.frame_ids[idx]]), dtype=dtype) * 255
            retval[idx, :, :, 3] = cv2.warpAffine(mask, mat, size, flags=interpolation)

        return retval

    def pre_process(self, batch: ExtractBatch) -> None:
        """Obtain the aligned face images at the requested size, centering and image format.
        Perform any plugin specific pre-processing

        Parameters
        ----------
        batch
            The incoming ExtractBatch to use for pre-processing
        """
        self._maybe_log_warning(batch.landmark_type)
        matrices = self._get_matrices(getattr(batch.aligned, self._aligned_mat_name))

        if batch.is_aligned:
            data = self._pre_process_aligned(batch, matrices)
        else:
            data = self._get_faces(batch.images, batch.frame_ids, matrices, with_alpha=True)

        data = self._format_images(data)
        batch.matrices = data[..., -1]  # type:ignore[assignment]  # Hacky re-use for ROI
        batch.data = self.plugin.pre_process(data[..., :3])
        batch.masks[self.storage_name] = ExtractBatchMask(self._centering, matrices)

    # Post-processing
    @classmethod
    def _crop_out_of_bounds(cls, masks: npt.NDArray[np.float32], roi_masks: npt.NDArray[np.float32]
                            ) -> None:
        """Un-mask any area of the predicted mask that falls outside of the original frame.

        Parameters
        ----------
        masks
            The predicted masks from the plugin
        roi_mask
            The roi masks. In frame is white, out of frame is black
        """
        if np.all(roi_masks):
            return  # All of the masks are within the frame
        needs_crop = np.any(roi_masks < 1., axis=(1, 2))
        roi_masks = roi_masks[..., None] if masks.ndim == 4 else roi_masks
        masks[needs_crop] *= roi_masks[needs_crop]

    def post_process(self, batch: ExtractBatch) -> None:
        """Perform mask post processing.

        Obtains the final output from the mask plugins and masks any part of the face patch that
        goes out of bounds

        Parameters
        ----------
        batch
            The incoming ExtractBatch to use for post-processing
        """
        masks = batch.data
        if self._overridden["post_process"]:
            masks = self.plugin.post_process(masks)
        self._crop_out_of_bounds(masks, batch.matrices)

        if self._storage_size == 0:
            self._storage_size = masks.shape[1]
            logger.debug("[%s.post_process] Updated storage size to %s",
                         self.plugin.name, self._storage_size)

        batch.masks[self.storage_name].masks = (masks * 255.).astype(np.uint8)
        batch.masks[self.storage_name].storage_size = self._storage_size


__all__ = get_module_objects(__name__)
