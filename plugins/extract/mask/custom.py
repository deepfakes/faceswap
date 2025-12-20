#!/usr/bin/env python3
""" Components Mask for faceswap.py """
from __future__ import annotations
import logging
import typing as T

import numpy as np
from lib.utils import get_module_objects
from ._base import BatchType, Masker

from . import custom_defaults as cfg

if T.TYPE_CHECKING:
    from lib.align.constants import CenteringType

logger = logging.getLogger(__name__)


class Mask(Masker):
    """ A mask that fills the whole face area with 1s or 0s (depending on user selected settings)
    for custom editing. """
    # pylint:disable=duplicate-code
    def __init__(self, **kwargs):
        git_model_id = None
        model_filename = None
        super().__init__(git_model_id=git_model_id, model_filename=model_filename, **kwargs)
        self.input_size = 256
        self.name = "Custom"
        self.vram = 0  # Doesn't use GPU
        self.vram_per_batch = 0
        self.batchsize = cfg.batch_size()
        self._storage_centering = T.cast("CenteringType", cfg.centering())
        # Separate storage for face and head masks
        self._storage_name = f"{self._storage_name}_{self._storage_centering}"

    def init_model(self) -> None:
        logger.debug("No mask model to initialize")

    def process_input(self, batch: BatchType) -> None:
        """ Compile the detected faces for prediction """
        batch.feed = np.zeros((self.batchsize, self.input_size, self.input_size, 1),
                              dtype="float32")

    def predict(self, feed: np.ndarray) -> np.ndarray:
        """ Run model to get predictions """
        if cfg.fill():
            feed[:] = 1.0
        return feed

    def process_output(self, batch: BatchType) -> None:
        """ Compile found faces for output """
        return


__all__ = get_module_objects(__name__)
