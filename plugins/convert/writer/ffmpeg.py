#!/usr/bin/env python3
""" Video output writer for faceswap.py converter """
from __future__ import annotations
import os
import typing as T

from math import ceil
from subprocess import CalledProcessError, check_output, STDOUT

import imageio
import imageio_ffmpeg as im_ffm
import numpy as np

from ._base import Output, logger

if T.TYPE_CHECKING:
    from collections.abc import Generator


class Writer(Output):
    """ Video output writer using imageio-ffmpeg.

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
        logger.debug("total_count: %s, frame_ranges: %s, source_video: '%s'",
                     total_count, frame_ranges, source_video)
        self._source_video: str = source_video
        self._output_filename: str = self._get_output_filename()
        self._frame_ranges: list[tuple[int, int]] | None = frame_ranges
        self._frame_order: list[int] = self._set_frame_order(total_count, frame_ranges)
        self._output_dimensions: str | None = None  # Fix dims on 1st received frame
        # Need to know dimensions of first frame, so set writer then
        self._writer: Generator[None, np.ndarray, None] | None = None

    @property
    def _valid_tunes(self) -> dict:
        """ dict: Valid tune selections for libx264 and libx265 codecs. """
        return {"libx264": ["film", "animation", "grain", "stillimage", "fastdecode",
                            "zerolatency"],
                "libx265": ["grain", "fastdecode", "zerolatency"]}

    @property
    def _video_fps(self) -> float:
        """ float: The fps of the source video. """
        reader = imageio.get_reader(self._source_video, "ffmpeg")  # type:ignore[arg-type]
        retval = reader.get_meta_data()["fps"]
        reader.close()
        logger.debug(retval)
        return retval

    @property
    def _output_params(self) -> list[str]:
        """ list: The FFMPEG Output parameters """
        codec = self.config["codec"]
        tune = self.config["tune"]
        # Force all frames to the same size
        output_args = ["-vf", f"scale={self._output_dimensions}"]

        output_args.extend(["-crf", str(self.config["crf"])])
        output_args.extend(["-preset", self.config["preset"]])

        if tune is not None and tune in self._valid_tunes[codec]:
            output_args.extend(["-tune", tune])

        if codec == "libx264" and self.config["profile"] != "auto":
            output_args.extend(["-profile:v", self.config["profile"]])

        if codec == "libx264" and self.config["level"] != "auto":
            output_args.extend(["-level", self.config["level"]])

        logger.debug(output_args)
        return output_args

    @property
    def _audio_codec(self) -> str | None:
        """ str or ``None``: The audio codec to use. This will either be ``"copy"`` (the default)
        or ``None`` if skip muxing has been selected in configuration options, or if frame ranges
        have been passed in the command line arguments. """
        retval: str | None = "copy"
        if self.config["skip_mux"]:
            logger.info("Skipping audio muxing due to configuration settings.")
            retval = None
        elif self._frame_ranges is not None:
            logger.warning("Muxing audio is not supported for limited frame ranges."
                           "The output video will be created but you will need to mux audio "
                           "manually.")
            retval = None
        elif not self._test_for_audio_stream():
            logger.warning("No audio stream could be found in the source video '%s'. Muxing audio "
                           "will be disabled.", self._source_video)
            retval = None
        logger.debug("Audio codec: %s", retval)
        return retval

    def _test_for_audio_stream(self) -> bool:
        """ Check whether the source video file contains an audio stream.

        If we attempt to mux audio from a source video that does not contain an audio stream
        ffmpeg will crash faceswap in a fairly ugly manner.

        Returns
        -------
        bool
            ``True`` if an audio stream is found in the source video file, otherwise ``False``

        Raises
        ------
        ValueError
            If a subprocess error is raised scanning the input video file
        """
        exe = im_ffm.get_ffmpeg_exe()
        cmd = [exe, "-hide_banner", "-i", self._source_video, "-f", "ffmetadata", "-"]

        try:
            out = check_output(cmd, stderr=STDOUT)
        except CalledProcessError as err:
            err_out = err.output.decode(errors="ignore")
            msg = f"Error checking audio stream. Status: {err.returncode}\n{err_out}"
            raise ValueError(msg) from err

        retval = False
        for line in out.splitlines():
            if not line.strip().startswith(b"Stream #"):
                continue
            logger.debug("scanning Stream line: %s", line.decode(errors="ignore").strip())
            if b"Audio" in line:
                retval = True
                break
        logger.debug("Audio found: %s", retval)
        return retval

    def _get_output_filename(self) -> str:
        """ Return full path to video output file.

        The filename is the same as the input video with `"_converted"` appended to the end. The
        file extension is as selected in the plugin settings. If a file already exists with the
        given filename, then `"_1"` is appended to the end of the filename. This number iterates
        until a valid filename that does not exist is found.

        Returns
        -------
        str
            The full path to the output video filename
        """
        filename = os.path.basename(self._source_video)
        filename = os.path.splitext(filename)[0]
        ext = self.config["container"]
        idx = 0
        while True:
            out_file = f"{filename}_converted{'' if idx == 0 else f'_{idx}'}.{ext}"
            retval = os.path.join(self.output_folder, out_file)
            if not os.path.exists(retval):
                break
            idx += 1
        logger.info("Outputting to: '%s'", retval)
        return retval

    def _get_writer(self, frame_dims: tuple[int, int]) -> Generator[None, np.ndarray, None]:
        """ Add the requested encoding options and return the writer.

        Parameters
        ----------
        frame_dims: tuple
            The (rows, colums) shape of the input image

        Returns
        -------
        generator
            The imageio ffmpeg writer
        """
        audio_codec = self._audio_codec
        audio_path = None if audio_codec is None else self._source_video
        logger.debug("writer config: %s, audio_path: '%s'", self.config, audio_path)

        retval = im_ffm.write_frames(self._output_filename,
                                     size=(frame_dims[1], frame_dims[0]),
                                     fps=self._video_fps,
                                     quality=None,
                                     codec=self.config["codec"],
                                     macro_block_size=8,
                                     ffmpeg_log_level="error",
                                     ffmpeg_timeout=10,
                                     output_params=self._output_params,
                                     audio_path=audio_path,
                                     audio_codec=audio_codec)
        logger.debug("FFMPEG Writer created: %s", retval)
        retval.send(None)

        return retval

    def write(self, filename: str, image: np.ndarray) -> None:
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
        if not self._output_dimensions:
            input_dims = T.cast(tuple[int, int], image.shape[:2])
            self._set_dimensions(input_dims)
            self._writer = self._get_writer(input_dims)
        self.cache_frame(filename, image)
        self._save_from_cache()

    def _set_dimensions(self, frame_dims: tuple[int, int]) -> None:
        """ Set the attribute :attr:`_output_dimensions` based on the first frame received.
        This protects against different sized images coming in and ensures all images are written
        to ffmpeg at the same size. Dimensions are mapped to a macro block size 8.

        Parameters
        ----------
        frame_dims: tuple
            The (rows, colums) shape of the input image
        """
        logger.debug("input dimensions: %s", frame_dims)
        self._output_dimensions = (f"{int(ceil(frame_dims[1] / 8) * 8)}:"
                                   f"{int(ceil(frame_dims[0] / 8) * 8)}")
        logger.debug("Set dimensions: %s", self._output_dimensions)

    def _save_from_cache(self) -> None:
        """ Writes any consecutive frames to the video container that are ready to be output
        from the cache. """
        assert self._writer is not None
        while self._frame_order:
            if self._frame_order[0] not in self.cache:
                logger.trace("Next frame not ready. Continuing")  # type:ignore[attr-defined]
                break
            save_no = self._frame_order.pop(0)
            save_image = self.cache.pop(save_no)
            logger.trace("Rendering from cache. Frame no: %s",  # type:ignore[attr-defined]
                         save_no)
            self._writer.send(np.ascontiguousarray(save_image[:, :, ::-1]))
        logger.trace("Current cache size: %s", len(self.cache))  # type:ignore[attr-defined]

    def close(self) -> None:
        """ Close the ffmpeg writer and mux the audio """
        if self._writer is not None:
            self._writer.close()
