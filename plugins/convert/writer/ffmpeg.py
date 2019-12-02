#!/usr/bin/env python3
""" Video output writer for faceswap.py converter """
import os
from collections import OrderedDict
from math import ceil

import imageio
import imageio_ffmpeg as im_ffm
from ffmpy import FFmpeg, FFRuntimeError

from ._base import Output, logger


class Writer(Output):
    """ Video output writer using imageio """
    def __init__(self, output_folder, total_count, frame_ranges, source_video, **kwargs):
        super().__init__(output_folder, **kwargs)
        logger.debug("total_count: %s, frame_ranges: %s, source_video: '%s'",
                     total_count, frame_ranges, source_video)
        self.source_video = source_video
        self.frame_ranges = frame_ranges
        self.frame_order = self.set_frame_order(total_count)
        self.output_dimensions = None  # Fix dims of 1st frame in case of different sized images
        self.writer = None  # Need to know dimensions of first frame, so set writer then

    @property
    def video_file(self):
        """ Return full path to video output """
        filename = os.path.basename(self.source_video)
        filename = os.path.splitext(filename)[0]
        filename = "{}_converted.{}".format(filename, self.config["container"])
        retval = os.path.join(self.output_folder, filename)
        logger.debug(retval)
        return retval

    @property
    def video_tmp_file(self):
        """ Temporary video file, prior to muxing final audio """
        path, filename = os.path.split(self.video_file)
        retval = os.path.join(path, "__tmp_{}".format(filename))
        logger.debug(retval)
        return retval

    @property
    def valid_tune(self):
        """ Return whether selected tune is valid for selected codec """
        return {"libx264": ["film", "animation", "grain", "stillimage", "fastdecode",
                            "zerolatency"],
                "libx265": ["grain", "fastdecode", "zerolatency"]}

    @property
    def video_fps(self):
        """ Return the fps of source video """
        reader = imageio.get_reader(self.source_video, "ffmpeg")
        retval = reader.get_meta_data()["fps"]
        reader.close()
        logger.debug(retval)
        return retval

    @property
    def output_params(self):
        """ FFMPEG Output parameters """
        codec = self.config["codec"]
        tune = self.config["tune"]
        # Force all frames to the same size
        output_args = ["-vf", "scale={}".format(self.output_dimensions)]

        output_args.extend(["-c:v", codec])
        output_args.extend(["-crf", str(self.config["crf"])])
        output_args.extend(["-preset", self.config["preset"]])

        if tune is not None and tune in self.valid_tune[codec]:
            output_args.extend(["-tune", tune])

        if codec == "libx264" and self.config["profile"] != "auto":
            output_args.extend(["-profile:v", self.config["profile"]])

        if codec == "libx264" and self.config["level"] != "auto":
            output_args.extend(["-level", self.config["level"]])

        logger.debug(output_args)
        return output_args

    def set_frame_order(self, total_count):
        """ Return the full list of frames to be converted in order """
        if self.frame_ranges is None:
            retval = list(range(1, total_count + 1))
        else:
            retval = list()
            for rng in self.frame_ranges:
                retval.extend(list(range(rng[0], rng[1] + 1)))
        logger.debug("frame_order: %s", retval)
        return retval

    def get_writer(self):
        """ Add the requested encoding options and return the writer """
        logger.debug("writer config: %s", self.config)
        return imageio.get_writer(self.video_tmp_file,
                                  fps=self.video_fps,
                                  ffmpeg_log_level="error",
                                  quality=None,
                                  macro_block_size=8,
                                  output_params=self.output_params)

    def write(self, filename, image):
        """ Frames come from the pool in arbitrary order, so cache frames
            for writing out in correct order """
        logger.trace("Received frame: (filename: '%s', shape: %s", filename, image.shape)
        if not self.output_dimensions:
            logger.info("Outputting to: '%s'", self.video_file)
            self.set_dimensions(image.shape[:2])
            self.writer = self.get_writer()
        self.cache_frame(filename, image)
        self.save_from_cache()

    def set_dimensions(self, frame_dims):
        """ Set the dimensions based on a given frame frame. This protects against different
            sized images coming in and ensure all images go out at the same size for writers
            that require it and mapped to a macro block size 16"""
        logger.debug("input dimensions: %s", frame_dims)
        self.output_dimensions = "{}:{}".format(
            int(ceil(frame_dims[1] / 16) * 16),
            int(ceil(frame_dims[0] / 16) * 16))
        logger.debug("Set dimensions: %s", self.output_dimensions)

    def save_from_cache(self):
        """ Save all the frames that are ready to be output from cache """
        while self.frame_order:
            if self.frame_order[0] not in self.cache:
                logger.trace("Next frame not ready. Continuing")
                break
            save_no = self.frame_order.pop(0)
            save_image = self.cache.pop(save_no)
            logger.trace("Rendering from cache. Frame no: %s", save_no)
            self.writer.append_data(save_image[:, :, ::-1])
        logger.trace("Current cache size: %s", len(self.cache))

    def close(self):
        """ Close the ffmpeg writer and mux the audio """
        self.writer.close()
        self.mux_audio()

    def mux_audio(self):
        """ Mux audio
            ImageIO is a useful lib for frames > video as it also packages the ffmpeg binary
            however muxing audio is non-trivial, so this is done afterwards with ffmpy.
            A future fix could be implemented to mux audio with the frames """
        logger.info("Muxing Audio...")
        if self.frame_ranges is not None:
            logger.warning("Muxing audio is not currently supported for limited frame ranges."
                           "The output video has been created but you will need to mux audio "
                           "yourself")
            os.rename(self.video_tmp_file, self.video_file)
            logger.debug("Removing temp file")
            if os.path.isfile(self.video_tmp_file):
                os.remove(self.video_tmp_file)
            return

        exe = im_ffm.get_ffmpeg_exe()
        inputs = OrderedDict([(self.video_tmp_file, None), (self.source_video, None)])
        outputs = {self.video_file: "-map 0:v:0 -map 1:a:0 -c: copy"}
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
                os.rename(self.video_tmp_file, self.video_file)
            break
        logger.debug("Removing temp file")
        if os.path.isfile(self.video_tmp_file):
            os.remove(self.video_tmp_file)
