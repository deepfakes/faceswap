#!/usr/bin/env python3
""" Parent class for output writers for faceswap.py converter """

import logging
import os

from plugins.convert._config import Config

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def get_config(plugin_name):
    """ Return the config for the requested model """
    return Config(plugin_name).config_dict


class Output():
    """ Parent class for scaling adjustments """
    def __init__(self, scaling, output_folder):
        logger.debug("Initializing %s: (output_folder: '%s')",
                     self.__class__.__name__, output_folder)
        self.config = get_config(".".join(self.__module__.split(".")[-2:]))
        logger.debug("config: %s", self.config)
        self.output_folder = output_folder
        self.scaling_factor = scaling / 100
        logger.debug("Initialized %s", self.__class__.__name__)

    def output_filename(self, filename):
        """ Return the output filename with the correct folder and extension
            NB: The plugin must have a config item 'format' that contains the
                file extension to use this method """
        out_filename = "{}.{}".format(os.path.splitext(filename)[0], self.config["format"])
        out_filename = os.path.join(self.output_folder, out_filename)
        logger.trace("in filename: '%s', out filename: '%s'", filename, out_filename)
        return out_filename

    def write(self, filename, image):
        """ Override for specific frame writing method """
        raise NotImplementedError

    def close(self):
        """ Override for specific frame writing close methods """
        raise NotImplementedError
