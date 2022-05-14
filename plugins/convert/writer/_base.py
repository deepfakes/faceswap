#!/usr/bin/env python3
""" Parent class for output writers for faceswap.py converter """

import logging
import os
import re

from typing import Optional

from plugins.convert._config import Config

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def get_config(plugin_name: str, configfile: Optional[str] = None) -> dict:
    """ Obtain the configuration settings for the writer plugin.

    Parameters
    ----------
    plugin_name: str
        The name of the convert plugin to return configuration settings for
    configfile: str, optional
        The full path to a custom configuration ini file. If ``None`` is passed
        then the file is loaded from the default location. Default: ``None``.

    Returns
    -------
    dict
        The requested configuration dictionary
    """
    return Config(plugin_name, configfile=configfile).config_dict


class Output():
    """ Parent class for writer plugins.

    Parameters
    ----------
    output_folder: str
        The full path to the output folder where the converted media should be saved
    configfile: str, optional
        The full path to a custom configuration ini file. If ``None`` is passed
        then the file is loaded from the default location. Default: ``None``.
    """
    def __init__(self, output_folder: str, configfile: Optional[str] = None) -> None:
        logger.debug("Initializing %s: (output_folder: '%s')",
                     self.__class__.__name__, output_folder)
        self.config: dict = get_config(".".join(self.__module__.split(".")[-2:]),
                                       configfile=configfile)
        logger.debug("config: %s", self.config)
        self.output_folder: str = output_folder

        # Methods for making sure frames are written out in frame order
        self.re_search: re.Pattern = re.compile(r"(\d+)(?=\.\w+$)")  # Identify frame numbers
        self.cache: dict = {}  # Cache for when frames must be written in correct order
        logger.debug("Initialized %s", self.__class__.__name__)

    @property
    def is_stream(self) -> bool:
        """ bool: Whether the writer outputs a stream or a series images.

        Writers that write to a stream have a frame_order paramater to dictate
        the order in which frames should be written out (eg. gif/ffmpeg) """
        retval = hasattr(self, "frame_order")
        return retval

    def output_filename(self, filename: str) -> str:
        """ Obtain the full path for the output file, including the correct extension, for the
        given input filename.

        NB: The plugin must have a config item 'format' that contains the file extension to use
        this method.

        Parameters
        ----------
        filename: str
            The input frame filename to generate the output file name for

        Returns
        -------
        str
            The full path for the output converted frame to be saved to.
        """
        filename = os.path.splitext(os.path.basename(filename))[0]
        out_filename = f"{filename}.{self.config['format']}"
        out_filename = os.path.join(self.output_folder, out_filename)
        logger.trace("in filename: '%s', out filename: '%s'", filename, out_filename)
        return out_filename

    def cache_frame(self, filename, image) -> None:
        """ Add the incoming converted frame to the cache ready for writing out.

        Used for ffmpeg and gif writers to ensure that the frames are written out in the correct
        order.

        Parameters
        ----------
        filename: str
            The filename of the incoming frame, where the frame index can be extracted from
        image: class:`numpy.ndarray`
            The converted frame corresponding to the given filename
        """
        frame_no = int(re.search(self.re_search, filename).group())
        self.cache[frame_no] = image
        logger.trace("Added to cache. Frame no: %s", frame_no)
        logger.trace("Current cache: %s", sorted(self.cache.keys()))

    def write(self, filename: str, image) -> None:
        """ Override for specific frame writing method.

        Parameters
        ----------
        filename: str
            The incoming frame filename.
        image: :class:`numpy.ndarray`
            The converted image to be written
        """
        raise NotImplementedError

    def pre_encode(self, image) -> None:  # pylint: disable=unused-argument,no-self-use
        """ Some writer plugins support the pre-encoding of images prior to saving out. As
        patching is done in multiple threads, but writing is done in a single thread, it can
        speed up the process to do any pre-encoding as part of the converter process.

        If the writer supports pre-encoding then override this to pre-encode the image in
        :module:`lib.convert` to speed up saving.

        Parameters
        ----------
        image: :class:`numpy.ndarray`
            The converted image that is to be run through the pre-encoding function

        Returns
        -------
        python function or ``None``
            If ``None`` then the writer does not support pre-encoding, otherwise return the python
            function that will pre-encode the image
        """
        return None

    def close(self) -> None:
        """ Override for specific converted frame writing close methods """
        raise NotImplementedError
