#!/usr/bin/env python3
""" Parent class for output writers for faceswap.py converter """

import logging
import os
import re

from plugins.convert._config import Config

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def get_config(plugin_name, configfile=None):
    """ Return the config for the requested model """
    return Config(plugin_name, configfile=configfile).config_dict


class Output():
    """ Parent class for scaling adjustments """
    def __init__(self, output_folder, configfile=None):
        logger.debug("Initializing %s: (output_folder: '%s')",
                     self.__class__.__name__, output_folder)
        self.config = get_config(".".join(self.__module__.split(".")[-2:]), configfile=configfile)
        logger.debug("config: %s", self.config)
        self.output_folder = output_folder
        self.output_dimensions = None

        # Methods for making sure frames are written out in frame order
        self.re_search = re.compile(r"(\d+)(?=\.\w+$)")  # Identify frame numbers
        self.cache = dict()  # Cache for when frames must be written in correct order
        logger.debug("Initialized %s", self.__class__.__name__)

    @property
    def is_stream(self):
        """ Return whether the writer is a stream or images
            Writers that write to a stream have a frame_order paramater to dictate
            the order in which frames should be written out (eg. gif/ffmpeg) """
        retval = hasattr(self, "frame_order")
        return retval

    def output_filename(self, filename):
        """ Return the output filename with the correct folder and extension
            NB: The plugin must have a config item 'format' that contains the
                file extension to use this method """
        filename = os.path.splitext(os.path.basename(filename))[0]
        out_filename = "{}.{}".format(filename, self.config["format"])
        out_filename = os.path.join(self.output_folder, out_filename)
        logger.trace("in filename: '%s', out filename: '%s'", filename, out_filename)
        return out_filename

    def cache_frame(self, filename, image):
        """ Add the incoming frame to the cache """
        frame_no = int(re.search(self.re_search, filename).group())
        self.cache[frame_no] = image
        logger.trace("Added to cache. Frame no: %s", frame_no)
        logger.trace("Current cache: %s", sorted(self.cache.keys()))

    def write(self, filename, image):
        """ Override for specific frame writing method """
        raise NotImplementedError

    def pre_encode(self, image):  # pylint: disable=unused-argument,no-self-use
        """ If the writer supports pre-encoding then override this to pre-encode
            the image in lib/convert.py to speed up saving """
        return None

    def close(self):
        """ Override for specific frame writing close methods """
        raise NotImplementedError
