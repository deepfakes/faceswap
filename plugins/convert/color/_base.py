#!/usr/bin/env python3
""" Parent class for color Adjustments for faceswap.py converter """

import logging
import numpy as np

from plugins.convert._config import Config

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def get_config(plugin_name, configfile=None):
    """ Return the config for the requested model """
    return Config(plugin_name, configfile=configfile).config_dict


class Adjustment():
    """ Parent class for adjustments """
    def __init__(self, configfile=None, config=None):
        logger.debug("Initializing %s: (configfile: %s, config: %s)",
                     self.__class__.__name__, configfile, config)
        self.config = self.set_config(configfile, config)
        logger.debug("config: %s", self.config)
        logger.debug("Initialized %s", self.__class__.__name__)

    def set_config(self, configfile, config):
        """ Set the config to either global config or passed in config """
        section = ".".join(self.__module__.split(".")[-2:])
        if config is None:
            retval = get_config(section, configfile)
        else:
            config.section = section
            retval = config.config_dict
            config.section = None
        logger.debug("Config: %s", retval)
        return retval

    def process(self, old_face, new_face, raw_mask):
        """ Override for specific color adjustment process """
        raise NotImplementedError

    def run(self, old_face, new_face, raw_mask):
        """ Perform selected adjustment on face """
        logger.trace("Performing color adjustment")
        # Remove mask for processing
        print("face shapes: ", old_face.shape, new_face.shape, raw_mask.shape)
        new_face = self.process(old_face, new_face, raw_mask)
        np.clip(new_face, 0.0, 1.0)
        new_face = np.concatenate((new_face, raw_mask), axis=-1)
        logger.trace("Performed color adjustment")
        return new_face
