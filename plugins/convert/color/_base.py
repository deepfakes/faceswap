#!/usr/bin/env python3
""" Parent class for color Adjustments for faceswap.py converter """

import logging
import numpy as np

from plugins.convert import convert_config

logger = logging.getLogger(__name__)


class Adjustment():
    """ Parent class for adjustments """
    def __init__(self, configfile=None, config=None):
        logger.debug("Initializing %s: (configfile: %s, config: %s)",
                     self.__class__.__name__, configfile, config)
        convert_config.load_config(config_file=configfile)
        logger.debug("Initialized %s", self.__class__.__name__)

    def process(self, old_face, new_face, raw_mask):
        """ Override for specific color adjustment process """
        raise NotImplementedError

    def run(self, old_face, new_face, raw_mask):
        """ Perform selected adjustment on face """
        # pylint:disable=duplicate-code
        logger.trace("Performing color adjustment")  # type:ignore[attr-defined]
        # Remove Mask for processing
        reinsert_mask = False
        final_mask = None
        if new_face.shape[2] == 4:
            reinsert_mask = True
            final_mask = new_face[:, :, -1]
            new_face = new_face[:, :, :3]
        new_face = self.process(old_face, new_face, raw_mask)
        new_face = np.clip(new_face, 0.0, 1.0)
        if reinsert_mask and new_face.shape[2] != 4:
            # Reinsert Mask
            assert final_mask is not None
            new_face = np.concatenate((new_face, np.expand_dims(final_mask, axis=-1)), -1)
        logger.trace("Performed color adjustment")  # type:ignore[attr-defined]
        return new_face
