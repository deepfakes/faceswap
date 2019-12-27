#!/usr/bin/env python3
""" Image output writer for faceswap.py converter """

from io import BytesIO
from PIL import Image

from ._base import Output, logger


class Writer(Output):
    """ Images output writer using cv2 """
    def __init__(self, output_folder, **kwargs):
        super().__init__(output_folder, **kwargs)
        self.check_transparency_format()
        # Correct format namings for writing to byte stream
        self.format_dict = dict(jpg="JPEG", jp2="JPEG 2000", tif="TIFF")
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
        logger.trace("Outputting: (filename: '%s'", filename)
        filename = self.output_filename(filename)
        try:
            with open(filename, "wb") as outfile:
                outfile.write(image.read())
        except Exception as err:  # pylint: disable=broad-except
            logger.error("Failed to save image '%s'. Original Error: %s", filename, err)

    def pre_encode(self, image):
        """ Pre_encode the image in lib/convert.py threads as it is a LOT quicker """
        logger.trace("Pre-encoding image")
        fmt = self.format_dict.get(self.config["format"], None)
        fmt = self.config["format"].upper() if fmt is None else fmt
        encoded = BytesIO()
        rgb = [2, 1, 0]
        if image.shape[2] == 4:
            rgb.append(3)
        out_image = Image.fromarray(image[..., rgb])
        out_image.save(encoded, fmt, **self.kwargs)
        encoded.seek(0)
        return encoded

    def close(self):
        """ Image writer does not need a close method """
        return
