#!/usr/bin/env python3
""" Video output writer for faceswap.py converter """
import os
from collections import OrderedDict
from math import ceil
from typing import Optional, List

import imageio
import imageio_ffmpeg as im_ffm
from ffmpy import FFmpeg, FFRuntimeError

from ._base import Output, logger


class Writer(Output):
    """ Video output writer using imageio-ffmpeg.

    Parameters
    ----------
    output_folder: str
        The folder to save the output video to
    total_count: int
        The total number of frames to be converted
    frame_ranges: list or ``None``
        List of integers for any explicit frame ranges to be converted or ``None`` if all frames
        are to be converted
    source_video: str
        The full path to the source video for obtaining fps and audio
    kwargs: dict
        Any additional standard :class:`plugins.convert.writer._base.Output` key word arguments.
    """
    def __init__(self,
                 output_folder: str,
                 total_count: int,
                 frame_ranges: Optional[List[int]],
                 source_video: str,
                 **kwargs) -> None:
        super().__init__(output_folder, **kwargs)
        logger.debug("total_count: %s, frame_ranges: %s, source_video: '%s'",
                     total_count, frame_ranges, source_video)
        self._source_video: str = source_video
        self._output_filename: str = self._get_output_filename()
        self._frame_ranges: Optional[List[int]] = frame_ranges
        self._frame_order: List[int] = self._set_frame_order(total_count)
        self._output_dimensions: Optional[str] = None  # Fix dims on 1st received frame
        # Need to know dimensions of first frame, so set writer then
        self._writer: Optional[imageio.plugins.ffmpeg.FfmpegFormat.Writer] = None

    @property
    def _video_tmp_file(self) -> str:
        """ str: Full path to the temporary video file that is generated prior to muxing final
        audio. """
        path, filename = os.path.split(self._output_filename)
        retval = os.path.join(path, f"__tmp_{filename}")
        logger.debug(retval)
        return retval

    @property
    def _valid_tunes(self) -> dict:
        """ dict: Valid tune selections for libx264 and libx265 codecs. """
        return {"libx264": ["film", "animation", "grain", "stillimage", "fastdecode",
                            "zerolatency"],
                "libx265": ["grain", "fastdecode", "zerolatency"]}

    @property
    def _video_fps(self) -> float:
        """ float: The fps of the source video. """
        reader = imageio.get_reader(self._source_video, "ffmpeg")
        retval = reader.get_meta_data()["fps"]
        reader.close()
        logger.debug(retval)
        return retval

    @property
    def _output_params(self) -> List[str]:
        """ list: The FFMPEG Output parameters """
        codec = self.config["codec"]
        tune = self.config["tune"]
        # Force all frames to the same size
        output_args = ["-vf", f"scale={self._output_dimensions}"]

        output_args.extend(["-c:v", codec])
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

    def _set_frame_order(self, total_count: int) -> List[int]:
        """ Obtain the full list of frames to be converted in order.

        Parameters
        ----------
        total_count: int
            The total number of frames to be converted

        Returns
        -------
        list
            Full list of all frame indices to be converted
        """
        if self._frame_ranges is None:
            retval = list(range(1, total_count + 1))
        else:
            retval = []
            for rng in self._frame_ranges:
                retval.extend(list(range(rng[0], rng[1] + 1)))
        logger.debug("frame_order: %s", retval)
        return retval

    def _get_writer(self) -> imageio.plugins.ffmpeg.FfmpegFormat.Writer:
        """ Add the requested encoding options and return the writer.

        Returns
        -------
        :class:`imageio.plugins.ffmpeg.FfmpegFormat.Writer`
            The imageio ffmpeg writer
        """
        logger.debug("writer config: %s", self.config)
        return imageio.get_writer(self._video_tmp_file,
                                  fps=self._video_fps,
                                  ffmpeg_log_level="error",
                                  quality=None,
                                  macro_block_size=8,
                                  output_params=self._output_params)

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
        logger.trace("Received frame: (filename: '%s', shape: %s", filename, image.shape)
        if not self._output_dimensions:
            self._set_dimensions(image.shape[:2])
            self._writer = self._get_writer()
        self.cache_frame(filename, image)
        self._save_from_cache()

    def _set_dimensions(self, frame_dims) -> None:
        """ Set the attribute :attr:`_output_dimensions` based on the first frame received.
        This protects against different sized images coming in and ensures all images are written
        to ffmpeg at the same size. Dimensions are mapped to a macro block size 16. """
        logger.debug("input dimensions: %s", frame_dims)
        self._output_dimensions = (f"{int(ceil(frame_dims[1] / 16) * 16)}:"
                                   f"{int(ceil(frame_dims[0] / 16) * 16)}")
        logger.debug("Set dimensions: %s", self._output_dimensions)

    def _save_from_cache(self) -> None:
        """ Writes any consecutive frames to the video container that are ready to be output
        from the cache. """
        while self._frame_order:
            if self._frame_order[0] not in self.cache:
                logger.trace("Next frame not ready. Continuing")
                break
            save_no = self._frame_order.pop(0)
            save_image = self.cache.pop(save_no)
            logger.trace("Rendering from cache. Frame no: %s", save_no)
            self._writer.append_data(save_image[:, :, ::-1])
        logger.trace("Current cache size: %s", len(self.cache))

    def close(self) -> None:
        """ Close the ffmpeg writer and mux the audio """
        self._writer.close()
        self._mux_audio()

    def _mux_audio(self) -> None:
        """ Mux audio the audio to the generated video temp file.

            ImageIO is a useful lib for frames > video as it also packages the ffmpeg binary
            however muxing audio is non-trivial, so this is done afterwards with ffmpy.

            # TODO A future fix could be implemented to mux audio with the frames """
        if self.config["skip_mux"]:
            logger.info("Skipping audio muxing due to configuration settings.")
            self._rename_tmp_file()
            return

        logger.info("Muxing Audio...")
        if self._frame_ranges is not None:
            logger.warning("Muxing audio is not currently supported for limited frame ranges."
                           "The output video has been created but you will need to mux audio "
                           "yourself")
            self._rename_tmp_file()
            return

        exe = im_ffm.get_ffmpeg_exe()
        inputs = OrderedDict([(self._video_tmp_file, None), (self._source_video, None)])
        outputs = {self._output_filename: "-map 0:v:0 -map 1:a:0 -c: copy"}
        ffm = FFmpeg(executable=exe,
                     global_options="-hide_banner -nostats -v 0 -y",
                     inputs=inputs,
                     outputs=outputs)
        logger.debug("Executing: %s", ffm.cmd)
        # Sometimes ffmpy exits for no discernible reason, but then works on a later attempt,
        # so take 5 shots at this
        attempts = 5
        for attempt in range(attempts):
            logger.debug("Muxing attempt: %s", attempt + 1)
            try:
                ffm.run()
            except FFRuntimeError as err:
                logger.debug("ffmpy runtime error: %s", str(err))
                if attempt != attempts - 1:
                    continue
                logger.error("There was a problem muxing audio. The output video has been "
                             "created but you will need to mux audio yourself either with the "
                             "EFFMpeg tool or an external application.")
                os.rename(self._video_tmp_file, self._output_filename)
            break
        logger.debug("Removing temp file")
        if os.path.isfile(self._video_tmp_file):
            os.remove(self._video_tmp_file)

    def _rename_tmp_file(self) -> None:
        """ Rename the temporary video file if not muxing audio. """
        os.rename(self._video_tmp_file, self._output_filename)
        logger.debug("Removing temp file")
        if os.path.isfile(self._video_tmp_file):
            os.remove(self._video_tmp_file)
