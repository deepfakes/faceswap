#!/usr/bin/env python3
""" Animated GIF writer for faceswap.py converter """
import os

import cv2
import imageio

from ._base import Output, logger


class Writer(Output):
    """ Video output writer using imageio """
    def __init__(self, output_folder, total_count, frame_ranges, **kwargs):
        logger.debug("total_count: %s, frame_ranges: %s", total_count, frame_ranges)
        super().__init__(output_folder, **kwargs)
        self.frame_order = self.set_frame_order(total_count, frame_ranges)
        self.output_dimensions = None  # Fix dims of 1st frame in case of different sized images
        self.writer = None  # Need to know dimensions of first frame, so set writer then
        self.gif_file = None  # Set filename based on first file seen

    @property
    def gif_params(self):
        """ Format the gif params """
        kwargs = {key: int(val) for key, val in self.config.items()}
        logger.debug(kwargs)
        return kwargs

    @staticmethod
    def set_frame_order(total_count, frame_ranges):
        """ Return the full list of frames to be converted in order """
        if frame_ranges is None:
            retval = list(range(1, total_count + 1))
        else:
            retval = list()
            for rng in frame_ranges:
                retval.extend(list(range(rng[0], rng[1] + 1)))
        logger.debug("frame_order: %s", retval)
        return retval

    def get_writer(self):
        """ Add the requested encoding options and return the writer """
        logger.debug("writer config: %s", self.config)
        return imageio.get_writer(self.gif_file,
                                  mode="i",
                                  **self.config)

    def write(self, filename, image):
        """ Frames come from the pool in arbitrary order, so cache frames
            for writing out in correct order """
        logger.trace("Received frame: (filename: '%s', shape: %s", filename, image.shape)
        if not self.gif_file:
            self.set_gif_filename(filename)
            self.set_dimensions(image.shape[:2])
            self.writer = self.get_writer()
        if (image.shape[1], image.shape[0]) != self.output_dimensions:
            image = cv2.resize(image, self.output_dimensions)  # pylint: disable=no-member
        self.cache_frame(filename, image)
        self.save_from_cache()

    def set_gif_filename(self, filename):
        """ Set the gif output filename """
        logger.debug("sample filename: '%s'", filename)
        filename = os.path.splitext(os.path.basename(filename))[0]
        idx = len(filename)
        for char in list(filename[::-1]):
            if not char.isdigit() and char not in ("_", "-"):
                break
            idx -= 1
        self.gif_file = os.path.join(self.output_folder, "{}_converted.gif".format(filename[:idx]))
        logger.info("Outputting to: '%s'", self.gif_file)

    def set_dimensions(self, frame_dims):
        """ Set the dimensions based on a given frame frame. This protects against different
            sized images coming in and ensure all images go out at the same size for writers
            that require it """
        logger.debug("input dimensions: %s", frame_dims)
        self.output_dimensions = (frame_dims[1], frame_dims[0])
        logger.debug("Set dimensions: %s", self.output_dimensions)

    def save_from_cache(self):
        """ Save all the frames that are ready to be output from cache """
        while self.frame_order:
            if self.frame_order[0] not in self.cache:
                logger.trace("Next frame not ready. Continuing")
                break
            save_no = self.frame_order.pop(0)
            save_image = self.cache.pop(save_no)
            logger.trace("Rendering from cache. Frame no: %s", save_no)
            self.writer.append_data(save_image[:, :, ::-1])
        logger.trace("Current cache size: %s", len(self.cache))

    def close(self):
        """ Close the ffmpeg writer and mux the audio """
        self.writer.close()
