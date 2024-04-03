#!/usr/bin/env python3
""" Parent class for output writers for faceswap.py converter """

import logging
import os
import re
import typing as T

import numpy as np

from plugins.convert._config import Config

logger = logging.getLogger(__name__)


def get_config(plugin_name: str, configfile: str | None = None) -> dict:
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
    def __init__(self, output_folder: str, configfile: str | None = None) -> None:
        logger.debug("Initializing %s: (output_folder: '%s')",
                     self.__class__.__name__, output_folder)
        self.config: dict = get_config(".".join(self.__module__.split(".")[-2:]),
                                       configfile=configfile)
        logger.debug("config: %s", self.config)
        self.output_folder: str = output_folder

        # For creating subfolders when separate mask is selected
        self._subfolders_created: bool = False

        # Methods for making sure frames are written out in frame order
        self.re_search = re.compile(r"(\d+)(?=\.\w+$)")  # Identify frame numbers
        self.cache: dict = {}  # Cache for when frames must be written in correct order
        logger.debug("Initialized %s", self.__class__.__name__)

    @property
    def is_stream(self) -> bool:
        """ bool: Whether the writer outputs a stream or a series images.

        Writers that write to a stream have a frame_order paramater to dictate
        the order in which frames should be written out (eg. gif/ffmpeg) """
        retval = hasattr(self, "_frame_order")
        return retval

    @classmethod
    def _set_frame_order(cls,
                         total_count: int,
                         frame_ranges: list[tuple[int, int]] | None) -> list[int]:
        """ Obtain the full list of frames to be converted in order.

        Used for FFMPEG and Gif writers to ensure correct frame order

        Parameters
        ----------
        total_count: int
            The total number of frames to be converted
        frame_ranges: list or ``None``
            List of tuples for starting and end values of each frame range to be converted or
            ``None`` if all frames are to be converted

        Returns
        -------
        list
            Full list of all frame indices to be converted
        """
        if frame_ranges is None:
            retval = list(range(1, total_count + 1))
        else:
            retval = []
            for rng in frame_ranges:
                retval.extend(list(range(rng[0], rng[1] + 1)))
        logger.debug("frame_order: %s", retval)
        return retval

    def output_filename(self, filename: str, separate_mask: bool = False) -> list[str]:
        """ Obtain the full path for the output file, including the correct extension, for the
        given input filename.

        NB: The plugin must have a config item 'format' that contains the file extension to use
        this method.

        Parameters
        ----------
        filename: str
            The input frame filename to generate the output file name for
        separate_mask: bool, optional
            ``True`` if the mask should be saved out to a sub-folder otherwise ``False``

        Returns
        -------
        list
            The full path for the output converted frame to be saved to in position 1. The full
            path for the mask to be output to in position 2 (if requested)
        """
        filename = os.path.splitext(os.path.basename(filename))[0]
        out_filename = f"{filename}.{self.config['format']}"
        retval = [os.path.join(self.output_folder, out_filename)]
        if separate_mask:
            retval.append(os.path.join(self.output_folder, "masks", out_filename))

        if separate_mask and not self._subfolders_created:
            locations = [os.path.dirname(loc) for loc in retval]
            logger.debug("Creating sub-folders: %s", locations)
            for location in locations:
                os.makedirs(location, exist_ok=True)

        logger.trace("in filename: '%s', out filename: '%s'", filename, retval)  # type:ignore
        return retval

    def cache_frame(self, filename: str, image: np.ndarray) -> None:
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
        re_frame = re.search(self.re_search, filename)
        assert re_frame is not None
        frame_no = int(re_frame.group())
        self.cache[frame_no] = image
        logger.trace("Added to cache. Frame no: %s", frame_no)  # type: ignore
        logger.trace("Current cache: %s", sorted(self.cache.keys()))  # type:ignore

    def write(self, filename: str, image: T.Any) -> None:
        """ Override for specific frame writing method.

        Parameters
        ----------
        filename: str
            The incoming frame filename.
        image: Any
            The converted image to be written. Could be a numpy array, a bytes encoded image or
            any other plugin specific format
        """
        raise NotImplementedError

    def pre_encode(self, image: np.ndarray, **kwargs) -> T.Any:  # pylint:disable=unused-argument
        """ Some writer plugins support the pre-encoding of images prior to saving out. As
        patching is done in multiple threads, but writing is done in a single thread, it can
        speed up the process to do any pre-encoding as part of the converter process.

        If the writer supports pre-encoding then override this to pre-encode the image in
        :mod:`lib.convert` to speed up saving.

        Parameters
        ----------
        image: :class:`numpy.ndarray`
            The converted image that is to be run through the pre-encoding function

        Returns
        -------
        Any or ``None``
            If ``None`` then the writer does not support pre-encoding, otherwise return output of
            the plugin specific pre-enccode function
        """
        return None

    def close(self) -> None:
        """ Override for specific converted frame writing close methods """
        raise NotImplementedError
