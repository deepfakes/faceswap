#!/usr/bin/env python3
""" Parent class for output writers for faceswap.py converter """

import logging
import os
import re

import cv2
import numpy as np

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
        self.interpolation = self.get_interpolation()
        self.output_dimensions = None

        # Methods for making sure frames are written out in frame order
        self.re_search = re.compile(r"(\d+)(?=\.\w+$)")  # Identify frame numbers
        self.cache = dict()  # Cache for when frames must be written in correct order
        logger.debug("Initialized %s", self.__class__.__name__)

    def get_interpolation(self):
        """ Return the correct interpolation method for the scaling factor """
        if self.scaling_factor > 1:
            interpolation = cv2.INTER_CUBIC  # pylint: disable=no-member
        else:
            interpolation = cv2.INTER_AREA  # pylint: disable=no-member
        logger.debug(interpolation)
        return interpolation

    def output_filename(self, filename):
        """ Return the output filename with the correct folder and extension
            NB: The plugin must have a config item 'format' that contains the
                file extension to use this method """
        out_filename = "{}.{}".format(os.path.splitext(filename)[0], self.config["format"])
        out_filename = os.path.join(self.output_folder, out_filename)
        logger.trace("in filename: '%s', out filename: '%s'", filename, out_filename)
        return out_filename

    def set_dimensions(self, frame_dims):
        """ Set the dimensions based on a given frame frame. This protects against different
            sized images coming in and ensure all images go out at the same size for writers
            that require it """
        logger.debug("frame_dims: %s", frame_dims)
        height, width = frame_dims
        dest_width = round((width * self.scaling_factor) / 2) * 2
        dest_height = round((height * self.scaling_factor) / 2) * 2
        self.output_dimensions = (dest_width, dest_height)
        logger.debug("dimensions: %s", self.output_dimensions)

    def scale_image_cv2(self, frame):
        """ Scale an incoming image with cv2 and return the scaled image """
        logger.trace("source frame: %s", frame.shape)
        if self.scaling_factor == 1 and self.output_dimensions is None:
            return frame
        frame_dims = (frame.shape[1], frame.shape[0])
        if self.output_dimensions is not None and frame_dims == self.output_dimensions:
            return frame

        # Image writing where size of frames doesn't need to be consistent
        if self.output_dimensions is None:
            interpolation = self.interpolation
            dimensions = (int(frame.shape[1] * self.scaling_factor),
                          int(frame.shape[0] * self.scaling_factor))
        # Video/GIF writing where size of frames muust be consistent
        else:
            dimensions = (self.output_dimensions[0], self.output_dimensions[1])
            if np.prod(frame_dims) < np.prod(self.output_dimensions):
                interpolation = cv2.INTER_CUBIC  # pylint: disable=no-member
            else:
                interpolation = cv2.INTER_AREA  # pylint: disable=no-member
        frame = cv2.resize(frame,  # pylint: disable=no-member
                           dimensions,
                           interpolation=interpolation)
        logger.trace("resized frame: %s", frame.shape)
        return frame

    def cache_frame(self, filename, image):
        """ Add the incoming frame to the cache """
        frame_no = int(re.search(self.re_search, filename).group())
        self.cache[frame_no] = image
        logger.trace("Added to cache. Frame no: %s", frame_no)

    def write(self, filename, image):
        """ Override for specific frame writing method """
        raise NotImplementedError

    def close(self):
        """ Override for specific frame writing close methods """
        raise NotImplementedError
