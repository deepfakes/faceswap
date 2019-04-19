#!/usr/bin/env python3
""" Video output writer for faceswap.py converter """
import os
import re
from collections import OrderedDict

import imageio
import imageio_ffmpeg as im_ffm
from ffmpy import FFmpeg

from ._base import Writer, logger


class Output(Writer):
    """ Video output writer using imageio """
    def __init__(self, output_folder, total_count, source_video):
        super().__init__(output_folder)
        self.source_video = source_video
        self.re_search = re.compile(r"(\d+)(?=\.\w+$)")  # Identify frame numbers
        self.frame_order = list(range(1, total_count + 1))
        self.writer = None  # Need to know dimensions of first frame, so set writer then
        self.cache = dict()
        self.dimensions = None
        self.scaling_factor = 1

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
        reader = imageio.get_reader(self.source_video)
        retval = reader.get_meta_data()["fps"]
        logger.debug(retval)
        return retval

    @property
    def output_params(self):
        """ FFMPEG Output parameters """
        codec = self.config["codec"]
        tune = self.config["tune"]
        output_args = ["-vf", "scale={}".format(self.dimensions)]
        if self.scaling_factor < 1:
            output_args.extend(["-sws_flags", "area"])
        elif self.scaling_factor > 1:
            output_args.extend(["-sws_flags", "spline"])

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

    def get_writer(self):
        """ Add the requested encoding options and return the writer """
        logger.debug("writer config: %s", self.config)
        return imageio.get_writer(self.video_tmp_file,
                                  fps=self.video_fps,
                                  quality=None,
                                  output_params=self.output_params)

    def write(self, filename, image):
        """ Frames come from the pool in arbitrary order, so cache frames
            for writing out in correct order """
        logger.trace("Received frame: (filename: '%s', shape: %s", filename, image.shape)
        if not self.dimensions:
            self.set_dimensions(image.shape[:2])
            self.writer = self.get_writer()
        self.cache_frame(filename, image)
        self.save_from_cache()

    def set_dimensions(self, frame_dims):
        """ Set the dimensions based on the first frame. This protects against different sized "
            images coming in and ensure all images go into the video at the same size """
        logger.debug("frame_dims: %s", frame_dims)
        height, width = frame_dims
        dest_width = self.config["width"]
        dest_height = self.config["height"]

        if dest_width == 0 and dest_height == 0:
            scale_width = width
            scale_height = height
        elif dest_width != 0 and dest_height != 0:
            self.scaling_factor = round((dest_height * dest_width) / (height * width))
            scale_width = dest_width
            scale_height = dest_height
        elif dest_height == 0:
            self.scaling_factor = dest_width / width
            scale_width = dest_width
            scale_height = round(height * self.scaling_factor)
        elif dest_width == 0:
            self.scaling_factor = dest_height / height
            scale_width = round(width * self.scaling_factor)
            scale_height = dest_height
        self.dimensions = "{}:{}".format(scale_width, scale_height)
        logger.debug("dimensions: %s, scale_factor: %s", self.dimensions, self.scaling_factor)

    def cache_frame(self, filename, image):
        """ Add the incoming frame to the cache """
        frame_no = int(re.search(self.re_search, filename).group())
        self.cache[frame_no] = image
        logger.trace("Added to cache. Frame no: %s", frame_no)

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
        exe = im_ffm.get_ffmpeg_exe()
        inputs = OrderedDict([(self.video_tmp_file, None), (self.source_video, None)])
        outputs = {self.video_file: "-map 0:0 -map 1:1 -c: copy"}
        ffm = FFmpeg(executable=exe,
                     global_options="-hide_banner -nostats -v 0 -y",
                     inputs=inputs,
                     outputs=outputs)
        logger.debug("Executing: %s", ffm.cmd)
        ffm.run()
        logger.debug("Removing temp file")
        os.remove(self.video_tmp_file)
