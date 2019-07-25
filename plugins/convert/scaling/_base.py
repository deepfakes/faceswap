#!/usr/bin/env python3
""" Parent class for scaling Adjustments for faceswap.py converter """

import logging
import numpy as np

from plugins.convert._config import Config

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def get_config(plugin_name):
    """ Return the config for the requested model """
    return Config(plugin_name).config_dict


class Adjustment():
    """ Parent class for scaling adjustments """
    def __init__(self):
        logger.debug("Initializing %s", self.__class__.__name__)
        self.config = get_config(".".join(self.__module__.split(".")[-2:]))
        logger.debug("config: %s", self.config)
        logger.debug("Initialized %s", self.__class__.__name__)

    def process(self, new_face):
        """ Override for specific scaling adjustment process """
        raise NotImplementedError

    def run(self, new_face):
        """ Perform selected adjustment on face """
        logger.trace("Performing scaling adjustment")
        # Remove Mask for processing
        reinsert_mask = False
        if new_face.shape[2] == 4:
            reinsert_mask = True
            final_mask = new_face[:, :, -1]
            new_face = new_face[:, :, :3]
        new_face = self.process(new_face)
        new_face = np.clip(new_face, 0.0, 1.0)
        if reinsert_mask and new_face.shape[2] != 4:
            # Reinsert Mask
            new_face = np.concatenate((new_face, np.expand_dims(final_mask, axis=-1)), -1)
        logger.trace("Performed scaling adjustment")
        return new_face
