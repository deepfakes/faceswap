#!/usr/bin/env python3
""" Image output writer for faceswap.py converter
    Uses cv2 for writing as in testing this was a lot faster than both Pillow and ImageIO
"""
import cv2
import numpy as np

from ._base import Output, logger


class Writer(Output):
    """ Images output writer using cv2

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
        self._extension = f".{self.config['format']}"
        self._check_transparency_format()
        self._separate_mask = self.config["draw_transparent"] and self.config["separate_mask"]
        self._args = self._get_save_args()

    def _check_transparency_format(self) -> None:
        """ Make sure that the output format is correct if draw_transparent is selected """
        transparent = self.config["draw_transparent"]
        if not transparent or (transparent and self.config["format"] == "png"):
            return
        logger.warning("Draw Transparent selected, but the requested format does not support "
                       "transparency. Changing output format to 'png'")
        self.config["format"] = "png"

    def _get_save_args(self) -> tuple[int, ...]:
        """ Obtain the save parameters for the file format.

        Returns
        -------
        tuple
            The OpenCV specific arguments for the selected file format
         """
        filetype = self.config["format"]
        args: tuple[int, ...] = tuple()
        if filetype == "jpg" and self.config["jpg_quality"] > 0:
            args = (cv2.IMWRITE_JPEG_QUALITY,
                    self.config["jpg_quality"])
        if filetype == "png" and self.config["png_compress_level"] > -1:
            args = (cv2.IMWRITE_PNG_COMPRESSION,
                    self.config["png_compress_level"])
        logger.debug(args)
        return args

    def write(self, filename: str, image: list[bytes]) -> None:
        """ Write out the pre-encoded image to disk. If separate mask has been selected, write out
        the encoded mask to a sub-folder in the output directory.

        Parameters
        ----------
        filename: str
            The full path to write out the image to.
        image: list
            List of :class:`bytes` objects of length 1 (containing just the image to write out)
            or length 2 (containing the image and mask to write out)
        """
        logger.trace("Outputting: (filename: '%s'", filename)  # type:ignore
        filenames = self.output_filename(filename, self._separate_mask)
        for fname, img in zip(filenames, image):
            try:
                with open(fname, "wb") as outfile:
                    outfile.write(img)
            except Exception as err:  # pylint:disable=broad-except
                logger.error("Failed to save image '%s'. Original Error: %s", filename, err)

    def pre_encode(self, image: np.ndarray, **kwargs) -> list[bytes]:
        """ Pre_encode the image in lib/convert.py threads as it is a LOT quicker.

        Parameters
        ----------
        image: :class:`numpy.ndarray`
            A 3 or 4 channel BGR swapped frame

        Returns
        -------
        list
            List of :class:`bytes` objects ready for writing. The list will be of length 1 with
            image bytes object as the only member unless separate mask has been requested, in which
            case it will be length 2 with the image in position 0 and mask in position 1
         """
        logger.trace("Pre-encoding image")  # type:ignore
        retval = []

        if self._separate_mask:
            mask = image[..., -1]
            image = image[..., :3]

            retval.append(cv2.imencode(self._extension,
                                       mask,
                                       self._args)[1])

        retval.insert(0, cv2.imencode(self._extension,
                                      image,
                                      self._args)[1])
        return retval

    def close(self) -> None:
        """ Does nothing as OpenCV writer does not need a close method """
        return
