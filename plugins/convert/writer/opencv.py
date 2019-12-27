#!/usr/bin/env python3
""" Image output writer for faceswap.py converter
    Uses cv2 for writing as in testing this was a lot faster than both Pillow and ImageIO
"""

import cv2
from ._base import Output, logger


class Writer(Output):
    """ Images output writer using cv2 """
    def __init__(self, output_folder, **kwargs):
        super().__init__(output_folder, **kwargs)
        self.extension = ".{}".format(self.config["format"])
        self.check_transparency_format()
        self.args = self.get_save_args()

    def check_transparency_format(self):
        """ Make sure that the output format is correct if draw_transparent is selected """
        transparent = self.config["draw_transparent"]
        if not transparent or (transparent and self.config["format"] == "png"):
            return
        logger.warning("Draw Transparent selected, but the requested format does not support "
                       "transparency. Changing output format to 'png'")
        self.config["format"] = "png"

    def get_save_args(self):
        """ Return the save parameters for the file format """
        filetype = self.config["format"]
        args = list()
        if filetype == "jpg" and self.config["jpg_quality"] > 0:
            args = (cv2.IMWRITE_JPEG_QUALITY,  # pylint: disable=no-member
                    self.config["jpg_quality"])
        if filetype == "png" and self.config["png_compress_level"] > -1:
            args = (cv2.IMWRITE_PNG_COMPRESSION,  # pylint: disable=no-member
                    self.config["png_compress_level"])
        logger.debug(args)
        return args

    def write(self, filename, image):
        logger.trace("Outputting: (filename: '%s'", filename)
        filename = self.output_filename(filename)
        try:
            with open(filename, "wb") as outfile:
                outfile.write(image)
        except Exception as err:  # pylint: disable=broad-except
            logger.error("Failed to save image '%s'. Original Error: %s", filename, err)

    def pre_encode(self, image):
        """ Pre_encode the image in lib/convert.py threads as it is a LOT quicker """
        logger.trace("Pre-encoding image")
        image = cv2.imencode(self.extension, image, self.args)[1]  # pylint: disable=no-member
        return image

    def close(self):
        """ Image writer does not need a close method """
        return
