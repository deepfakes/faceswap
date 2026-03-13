#!/usr/bin/env python3
"""Custom Mask for faceswap.py"""
from __future__ import annotations
import logging
import typing as T

import numpy as np
from lib.utils import get_module_objects
from plugins.extract.base import FacePlugin

from . import custom_defaults as cfg

if T.TYPE_CHECKING:
    from lib.align.constants import CenteringType

logger = logging.getLogger(__name__)


class Custom(FacePlugin):
    """A mask that fills the whole face area with 1s or 0s (depending on user selected settings)
    for custom editing."""
    # pylint:disable=duplicate-code

    def __init__(self):
        super().__init__(input_size=256,
                         batch_size=cfg.batch_size(),
                         is_rgb=False,
                         dtype="uint8",
                         scale=(0, 255),
                         centering=T.cast("CenteringType", cfg.centering()))
        # Separate storage for face and head masks
        self.storage_name = f"{self.storage_name}_{self.centering}"
        self._fill = cfg.fill()

    def load_model(self) -> None:
        """No model to load, just return"""
        logger.debug("No mask model to initialize")

    def pre_process(self, batch: np.ndarray) -> np.ndarray:
        """ Return a zero array of the same shape and dtype as the input array

        Parameters
        ----------
        batch
            The batch of aligned faces in the correct format for the model

        Returns
        -------
        A zero'd array of the same shape and dtype as the input
        """
        return np.zeros(batch.shape[:3], dtype="uint8")

    def process(self, batch: np.ndarray) -> np.ndarray:
        """Get the masks from the model

        Parameters
        ----------
        batch
            The batch to process

        Returns
        -------
        The processed empty masks
        """
        if self._fill:
            batch[:] = 255
        return batch


__all__ = get_module_objects(__name__)
