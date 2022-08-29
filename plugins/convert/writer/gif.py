#!/usr/bin/env python3
""" Animated GIF writer for faceswap.py converter """
import os
from typing import Optional, List, Tuple, TYPE_CHECKING

import cv2
import imageio

from ._base import Output, logger

if TYPE_CHECKING:
    from imageio.plugins.pillowmulti import GIFFormat


class Writer(Output):
    """ GIF output writer using imageio.


    Parameters
    ----------
    output_folder: str
        The folder to save the output gif to
    total_count: int
        The total number of frames to be converted
    frame_ranges: list or ``None``
        List of tuples for starting and end values of each frame range to be converted or ``None``
        if all frames are to be converted
    kwargs: dict
        Any additional standard :class:`plugins.convert.writer._base.Output` key word arguments.
    """
    def __init__(self,
                 output_folder: str,
                 total_count: int,
                 frame_ranges: Optional[List[Tuple[int, int]]],
                 **kwargs) -> None:
        logger.debug("total_count: %s, frame_ranges: %s", total_count, frame_ranges)
        super().__init__(output_folder, **kwargs)
        self.frame_order: List[int] = self._set_frame_order(total_count, frame_ranges)
        self._output_dimensions: Optional[Tuple[int, int]] = None  # Fix dims on 1st received frame
        # Need to know dimensions of first frame, so set writer then
        self._writer: Optional[imageio.plugins.pillowmulti.GIFFormat.Writer] = None
        self._gif_file: Optional[str] = None  # Set filename based on first file seen

    @property
    def _gif_params(self) -> dict:
        """ dict: The selected gif plugin configuration options. """
        kwargs = {key: int(val) for key, val in self.config.items()}
        logger.debug(kwargs)
        return kwargs

    @staticmethod
    def _set_frame_order(total_count: int,
                         frame_ranges: Optional[List[Tuple[int, int]]]) -> List[int]:
        """ Obtain the full list of frames to be converted in order.

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

    def _get_writer(self) -> "GIFFormat.Writer":
        """ Obtain the GIF writer with the requested GIF encoding options.

        Returns
        -------
        :class:`imageio.plugins.pillowmulti.GIFFormat.Writer`
            The imageio GIF writer
        """
        logger.debug("writer config: %s", self.config)
        return imageio.get_writer(self._gif_file,
                                  mode="i",
                                  **self._gif_params)

    def write(self, filename: str, image) -> None:
        """ Frames come from the pool in arbitrary order, so frames are cached for writing out
        in the correct order.

        Parameters
        ----------
        filename: str
            The incoming frame filename.
        image: :class:`numpy.ndarray`
            The converted image to be written
        """
        logger.trace("Received frame: (filename: '%s', shape: %s",  # type: ignore
                     filename, image.shape)
        if not self._gif_file:
            self._set_gif_filename(filename)
            self._set_dimensions(image.shape[:2])
            self._writer = self._get_writer()
        if (image.shape[1], image.shape[0]) != self._output_dimensions:
            image = cv2.resize(image, self._output_dimensions)  # pylint: disable=no-member
        self.cache_frame(filename, image)
        self._save_from_cache()

    def _set_gif_filename(self, filename: str) -> None:
        """ Set the full path to GIF output file to :attr:`_gif_file`

        The filename is the created from the source filename of the first input image received with
        `"_converted"` appended to the end and a .gif extension. If a file already exists with the
        given filename, then `"_1"` is appended to the end of the filename. This number iterates
        until a valid filename that does not exist is found.

        Parameters
        ----------
        filename: str
            The incoming frame filename.
        """

        logger.debug("sample filename: '%s'", filename)
        filename = os.path.splitext(os.path.basename(filename))[0]
        snip = len(filename)
        for char in list(filename[::-1]):
            if not char.isdigit() and char not in ("_", "-"):
                break
            snip -= 1
        filename = filename[:snip]

        idx = 0
        while True:
            out_file = f"{filename}_converted{'' if idx == 0 else f'_{idx}'}.gif"
            retval = os.path.join(self.output_folder, out_file)
            if not os.path.exists(retval):
                break
            idx += 1

        self._gif_file = retval
        logger.info("Outputting to: '%s'", self._gif_file)

    def _set_dimensions(self, frame_dims: Tuple[int, int]) -> None:
        """ Set the attribute :attr:`_output_dimensions` based on the first frame received. This
        protects against different sized images coming in and ensure all images get written to the
        Gif at the sema dimensions. """
        logger.debug("input dimensions: %s", frame_dims)
        self._output_dimensions = (frame_dims[1], frame_dims[0])
        logger.debug("Set dimensions: %s", self._output_dimensions)

    def _save_from_cache(self) -> None:
        """ Writes any consecutive frames to the GIF container that are ready to be output
        from the cache. """
        assert self._writer is not None
        while self.frame_order:
            if self.frame_order[0] not in self.cache:
                logger.trace("Next frame not ready. Continuing")  # type: ignore
                break
            save_no = self.frame_order.pop(0)
            save_image = self.cache.pop(save_no)
            logger.trace("Rendering from cache. Frame no: %s", save_no)  # type: ignore
            self._writer.append_data(save_image[:, :, ::-1])
        logger.trace("Current cache size: %s", len(self.cache))  # type: ignore

    def close(self) -> None:
        """ Close the GIF writer on completion. """
        if self._writer is not None:
            self._writer.close()
