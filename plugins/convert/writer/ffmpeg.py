#!/usr/bin/env python3
""" Video output writer for faceswap.py converter """
from __future__ import annotations

import logging
import typing as T
import os
from collections import deque

from lib.logger import parse_class_init
from lib.utils import get_module_objects
from lib.video import VideoMux

from ._base import Output
from . import ffmpeg_defaults as cfg

if T.TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

logger = logging.getLogger(__name__)


class Writer(Output):
    """ Video output writer using pyAV.

    Parameters
    ----------
    output_folder: str
        The folder to save the output video to
    total_count: int
        The total number of frames to be converted
    frame_ranges: list or ``None``
        List of tuples for starting and end values of each frame range to be converted or ``None``
        if all frames are to be converted
    source_video: str
        The full path to the source video for obtaining fps and audio
    kwargs: dict
        Any additional standard :class:`plugins.convert.writer._base.Output` key word arguments.
    """
    def __init__(self,
                 output_folder: str,
                 total_count: int,
                 frame_ranges: list[tuple[int, int]] | None,
                 source_video: str,
                 **kwargs) -> None:
        super().__init__(output_folder, **kwargs)
        logger.debug(parse_class_init(locals()))
        self._frame_ranges: list[tuple[int, int]] | None = frame_ranges
        self._frame_order: deque[int] = self._set_frame_order(total_count, frame_ranges)
        self._muxer = self._get_muxer(source_video)

    @property
    def _valid_tunes(self) -> dict:
        """ dict: Valid tune selections for libx264 and libx265 codecs. """
        return {"libx264": ["film", "animation", "grain", "stillimage", "fastdecode",
                            "zerolatency"],
                "libx265": ["grain", "fastdecode", "zerolatency"]}

    def _get_output_filename(self, source_filename: str) -> str:
        """ Return full path to video output file.

        The filename is the same as the input video with `"_converted"` appended to the end. The
        file extension is as selected in the plugin settings. If a file already exists with the
        given filename, then `"_1"` is appended to the end of the filename. This number iterates
        until a valid filename that does not exist is found.

        Parameters
        ----------
        The filename of the source/reference video

        Returns
        -------
        str
            The full path to the output video filename
        """
        filename = os.path.basename(source_filename)
        filename = os.path.splitext(filename)[0]
        ext = cfg.container()
        idx = 0
        while True:
            out_file = f"{filename}_converted{'' if idx == 0 else f'_{idx}'}.{ext}"
            retval = os.path.join(self.output_folder, out_file)
            if not os.path.exists(retval):
                break
            idx += 1
        logger.info("[FFMPEG] Outputting to: '%s'", retval)
        return retval

    def _get_codec_parameters(self) -> dict[str, str]:
        """Obtain the selected video codec parameters

        Returns
        -------
        Parameter option name to parameter value for the codec options
        """
        codec = cfg.codec()
        tune = cfg.tune()

        output_args = {"crf": str(cfg.crf()),
                       "preset": cfg.preset()}

        if tune is not None and tune in self._valid_tunes[codec]:
            output_args["tune"] = tune

        if codec == "libx264" and cfg.profile() != "auto":
            output_args["profile"] = cfg.profile()

        if codec == "libx264" and cfg.level() != "auto":
            output_args["level"] = cfg.level()

        logger.debug("[FFMPEG] codec_params: %s", output_args)
        return output_args

    def _should_mux_audio(self) -> bool:
        """Test if audio should be muxed based on selected parameters

        Returns
        -------
        ``True`` if audio should be muxed
        """
        if cfg.skip_mux():
            logger.info("Skipping audio muxing due to configuration settings.")
            return False

        if self._frame_ranges is not None:
            logger.warning("Muxing audio is not supported for limited frame ranges."
                           "The output video will be created but you will need to mux audio "
                           "manually.")
            return False

        logger.debug("[FFMPEG] Audio will be muxed")
        return True

    def _get_muxer(self, source_filename: str) -> VideoMux:
        """Obtain the VideoMux object for encoding the video

        Parameters
        ----------
        source_filename
            The filename of the reference source video
        """
        out_file = self._get_output_filename(source_filename)
        params = self._get_codec_parameters()
        mux_audio = self._should_mux_audio()
        codec = T.cast(T.Literal["libx264", "libx265"], cfg.codec())
        return VideoMux(source_filename, out_file, codec, params, mux_audio)

    def _save_from_cache(self) -> None:
        """Sends any any consecutive frames to the muxer that are ready to be output from cache."""
        while self._frame_order:
            if self._frame_order[0] not in self.cache:
                logger.trace("Next frame not ready. Continuing")  # type:ignore[attr-defined]
                break
            save_no = self._frame_order.popleft()
            save_image = self.cache.pop(save_no)
            logger.trace(  # type:ignore[attr-defined]
                "[FFMPEG] Rendering from cache. Frame no: %s", save_no)
            self._muxer.encode(save_image)
        logger.trace("[FFMPEG] Current cache size: %s",  # type:ignore[attr-defined]
                     len(self.cache))

    def write(self, filename: str, image: npt.NDArray[np.uint8]) -> None:
        """ Frames come from the pool in arbitrary order, so frames are cached for writing out
        in the correct order.

        Parameters
        ----------
        filename: str
            The incoming frame filename.
        image: :class:`numpy.ndarray`
            The converted image to be written
        """
        logger.trace("Received frame: (filename: '%s', shape: %s",  # type:ignore[attr-defined]
                     filename, image.shape)
        self.cache_frame(filename, image)
        self._save_from_cache()

    def close(self) -> None:
        """ Close the ffmpeg writer"""
        self._muxer.encode(None)


__all__ = get_module_objects(__name__)
