#!/usr/bin/env python3
""" Image output writer for faceswap.py converter """
from io import BytesIO
from PIL import Image

import numpy as np

from ._base import Output, logger


class Writer(Output):
    """ Images output writer using Pillow

    Parameters
    ----------
    output_folder: str
        The full path to the output folder where the converted media should be saved
    configfile: str, optional
        The full path to a custom configuration ini file. If ``None`` is passed
        then the file is loaded from the default location. Default: ``None``.
    """
    def __init__(self, output_folder: str, **kwargs) -> None:
        super().__init__(output_folder, **kwargs)
        self._check_transparency_format()
        # Correct format namings for writing to byte stream
        self._format_dict = {"jpg": "JPEG", "jp2": "JPEG 2000", "tif": "TIFF"}
        self._separate_mask = self.config["draw_transparent"] and self.config["separate_mask"]
        self._kwargs = self._get_save_kwargs()

    def _check_transparency_format(self) -> None:
        """ Make sure that the output format is correct if draw_transparent is selected """
        transparent = self.config["draw_transparent"]
        if not transparent or (transparent and self.config["format"] in ("png", "tif")):
            return
        logger.warning("Draw Transparent selected, but the requested format does not support "
                       "transparency. Changing output format to 'png'")
        self.config["format"] = "png"

    def _get_save_kwargs(self) -> dict[str, bool | int | str]:
        """ Return the save parameters for the file format

        Returns
        -------
        dict
            The specific keyword arguments for the selected file format
        """
        filetype = self.config["format"]
        kwargs = {}
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

    def write(self, filename: str, image: list[BytesIO]) -> None:
        """ Write out the pre-encoded image to disk. If separate mask has been selected, write out
        the encoded mask to a sub-folder in the output directory.

        Parameters
        ----------
        filename: str
            The full path to write out the image to.
        image: list
            List of :class:`BytesIO` objects of length 1 (containing just the image to write out)
            or length 2 (containing the image and mask to write out)
        """
        logger.trace("Outputting: (filename: '%s'", filename)  # type:ignore
        filenames = self.output_filename(filename, self._separate_mask)
        try:
            for fname, img in zip(filenames, image):
                with open(fname, "wb") as outfile:
                    outfile.write(img.read())
        except Exception as err:  # pylint:disable=broad-except
            logger.error("Failed to save image '%s'. Original Error: %s", filename, err)

    def pre_encode(self, image: np.ndarray, **kwargs) -> list[BytesIO]:
        """ Pre_encode the image in lib/convert.py threads as it is a LOT quicker

        Parameters
        ----------
        image: :class:`numpy.ndarray`
            A 3 or 4 channel BGR swapped frame

        Returns
        -------
        list
            List of :class:`BytesIO` objects ready for writing. The list will be of length 1 with
            image bytes object as the only member unless separate mask has been requested, in which
            case it will be length 2 with the image in position 0 and mask in position 1
         """
        logger.trace("Pre-encoding image")  # type:ignore

        if self._separate_mask:
            encoded_mask = self._encode_image(image[..., -1])
            image = image[..., :3]

        rgb = [2, 1, 0, 3] if image.shape[2] == 4 else [2, 1, 0]
        encoded_image = self._encode_image(image[..., rgb])

        retval = [encoded_image]

        if self._separate_mask:
            retval.append(encoded_mask)

        return retval

    def _encode_image(self, image: np.ndarray) -> BytesIO:
        """ Encode an image in the correct format as a bytes object for saving

        Parameters
        ----------
        image: :class:`np.ndarray`
            The single channel mask to encode for saving

        Returns
        -------
        :class:`BytesIO`
            The image as a bytes object ready for writing to disk
        """
        fmt = self._format_dict.get(self.config["format"], self.config["format"].upper())
        encoded = BytesIO()
        out_image = Image.fromarray(image)
        out_image.save(encoded, fmt, **self._kwargs)
        encoded.seek(0)
        return encoded

    def close(self) -> None:
        """ Does nothing as Pillow writer does not need a close method """
        return
