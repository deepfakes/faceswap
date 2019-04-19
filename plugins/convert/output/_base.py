#!/usr/bin/env python3
""" Parent class for output writers for faceswap.py converter """

import logging

from plugins.convert._config import Config

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def get_config(plugin_name):
    """ Return the config for the requested model """
    return Config(plugin_name).config_dict


class Writer():
    """ Parent class for scaling adjustments """
    def __init__(self, output_folder):
        logger.debug("Initializing %s: (output_folder: '%s')",
                     self.__class__.__name__, output_folder)
        self.config = get_config(".".join(self.__module__.split(".")[-2:]))
        logger.debug("config: %s", self.config)
        self.output_folder = output_folder
        logger.debug("Initialized %s", self.__class__.__name__)

    def write(self, filename, image):
        """ Override for specific frame writing method """
        raise NotImplementedError

    def close(self):
        """ Override for specific frame writing close methods """
        raise NotImplementedError
