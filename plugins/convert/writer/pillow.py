#!/usr/bin/env python3
""" Image output writer for faceswap.py converter """

import os
from PIL import Image

from ._base import Output, logger


class Writer(Output):
    """ Images output writer using cv2 """
    def __init__(self, scaling, output_folder):
        super().__init__(scaling, output_folder)
        self.check_transparency_format()
        self.kwargs = self.get_save_kwargs()

    def check_transparency_format(self):
        """ Make sure that the output format is correct if draw_transparent is selected """
        transparent = self.config["draw_transparent"]
        if not transparent or (transparent and self.config["format"] in ("png", "tif")):
            return
        logger.warning("Draw Transparent selected, but the requested format does not support "
                       "transparency. Changing output format to 'png'")
        self.config["format"] = "png"

    def get_save_kwargs(self):
        """ Return the save parameters for the file format """
        filetype = self.config["format"]
        kwargs = dict()
        if filetype in ("gif", "jpg", "png"):
            kwargs["optimize"] = self.config["optimize"]
        if filetype == "gif":
            kwargs["interlace"] = self.config["gif_interlace"]
        if filetype == "png":
            kwargs["compress_level"] = self.config["png_compress_level"]
        if filetype == "tif":
            kwargs["compression"] = self.config["tif_compression"]
        logger.debug(kwargs)
        return kwargs

    def write(self, filename, image):
        logger.trace("Outputting: (filename: '%s', shape: %s", filename, image.shape)
        filename = self.output_filename(filename)
        if self.scaling_factor != 1:
            image = self.scale_image_cv2(image)

        rgb = [2, 1, 0]
        if image.shape[2] == 4:
            rgb.append(3)

        out_image = Image.fromarray(image[..., rgb])
        try:
            out_image.save(filename, **self.kwargs, interlace=0)
        except Exception as err:  # pylint: disable=broad-except
            logger.error("Failed to save image '%s'. Original Error: %s", filename, err)

    def output_filename(self, filename):
        """ Return the output filename with the correct folder and extension """
        out_filename = "{}.{}".format(os.path.splitext(filename)[0], self.config["format"])
        out_filename = os.path.join(self.output_folder, out_filename)
        logger.trace("in filename: '%s', out filename: '%s'", filename, out_filename)
        return out_filename

    def close(self):
        """ Image writer does not need a close method """
        return
