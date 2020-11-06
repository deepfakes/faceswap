#!/usr/bin/env python3
# vim: set fileencoding=utf-8 :
"""
Created on 2018-03-16 15:14

@author: Lev Velykoivanenko (velykoivanenko.lev@gmail.com)
"""
import logging
import os
import subprocess
import sys
import datetime
from collections import OrderedDict

import imageio
import imageio_ffmpeg as im_ffm
from ffmpy import FFmpeg, FFRuntimeError

# faceswap imports
from lib.utils import _image_extensions, _video_extensions

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class DataItem():
    """
    A simple class used for storing the media data items and directories that
    Effmpeg uses for 'input', 'output' and 'ref_vid'.
    """
    vid_ext = _video_extensions
    # future option in effmpeg to use audio file for muxing
    audio_ext = ['.aiff', '.flac', '.mp3', '.wav']
    img_ext = _image_extensions

    def __init__(self, path=None, name=None, item_type=None, ext=None,
                 fps=None):
        logger.debug("Initializing %s: (path: '%s', name: '%s', item_type: '%s', ext: '%s')",
                     self.__class__.__name__, path, name, item_type, ext)
        self.path = path
        self.name = name
        self.type = item_type
        self.ext = ext
        self.fps = fps
        self.dirname = None
        self.set_type_ext(path)
        self.set_dirname(self.path)
        self.set_name(name)
        if self.is_type("vid") and self.fps is None:
            self.set_fps()
        logger.debug("Initialized %s", self.__class__.__name__)

    def set_name(self, name=None):
        """ Set the name """
        if name is None and self.path is not None:
            self.name = os.path.basename(self.path)
        elif name is not None and self.path is None:
            self.name = os.path.basename(name)
        elif name is not None and self.path is not None:
            self.name = os.path.basename(name)
        else:
            self.name = None
        logger.debug(self.name)

    def set_type_ext(self, path=None):
        """ Set the extension """
        if path is not None:
            self.path = path
        if self.path is not None:
            item_ext = os.path.splitext(self.path)[1].lower()
            if item_ext in DataItem.vid_ext:
                item_type = 'vid'
            elif item_ext in DataItem.audio_ext:
                item_type = 'audio'
            else:
                item_type = 'dir'
            self.type = item_type
            self.ext = item_ext
            logger.debug("path: '%s', type: '%s', ext: '%s'", self.path, self.type, self.ext)
        else:
            return

    def set_dirname(self, path=None):
        """ Set the folder name """
        if path is None and self.path is not None:
            self.dirname = os.path.dirname(self.path)
        elif path is not None and self.path is None:
            self.dirname = os.path.dirname(path)
        elif path is not None and self.path is not None:
            self.dirname = os.path.dirname(path)
        else:
            self.dirname = None
        logger.debug("path: '%s', dirname: '%s'", path, self.dirname)

    def is_type(self, item_type=None):
        """ Get the type """
        if item_type == "media":
            chk_type = self.type in "vid audio"
        elif item_type == "dir":
            chk_type = self.type == "dir"
        elif item_type == "vid":
            chk_type = self.type == "vid"
        elif item_type == "audio":
            chk_type = self.type == "audio"
        elif item_type.lower() == "none":
            chk_type = self.type is None
        else:
            chk_type = False
        logger.debug("item_type: '%s', chk_type: '%s'", item_type, chk_type)
        return chk_type

    def set_fps(self):
        """ Set the Frames Per Second """
        try:
            self.fps = Effmpeg.get_fps(self.path)
        except FFRuntimeError:
            self.fps = None
        logger.debug(self.fps)


class Effmpeg():
    """
    Class that allows for "easy" ffmpeg use. It provides a nice cli interface
    for common video operations.
    """

    _actions_req_fps = ["extract", "gen_vid"]
    _actions_req_ref_video = ["mux_audio"]
    _actions_can_use_ref_video = ["gen_vid"]
    _actions_have_dir_output = ["extract"]
    _actions_have_vid_output = ["gen_vid", "mux_audio", "rescale", "rotate",
                                "slice"]
    _actions_have_print_output = ["get_fps", "get_info"]
    _actions_have_dir_input = ["gen_vid"]
    _actions_have_vid_input = ["extract", "get_fps", "get_info", "rescale",
                               "rotate", "slice"]

    # Class variable that stores the target executable (ffmpeg or ffplay)
    _executable = im_ffm.get_ffmpeg_exe()

    # Class variable that stores the common ffmpeg arguments based on verbosity
    __common_ffmpeg_args_dict = {"normal": "-hide_banner ",
                                 "quiet": "-loglevel panic -hide_banner ",
                                 "verbose": ''}

    # _common_ffmpeg_args is the class variable that will get used by various
    # actions and it will be set by the process_arguments() method based on
    # passed verbosity
    _common_ffmpeg_args = ''

    def __init__(self, arguments):
        logger.debug("Initializing %s: (arguments: %s)", self.__class__.__name__, arguments)
        self.args = arguments
        self.exe = im_ffm.get_ffmpeg_exe()
        self.input = DataItem()
        self.output = DataItem()
        self.ref_vid = DataItem()
        self.start = ""
        self.end = ""
        self.duration = ""
        self.print_ = False
        logger.debug("Initialized %s", self.__class__.__name__)

    def process(self):
        """ EFFMPEG Process """
        logger.debug("Running Effmpeg")
        # Format action to match the method name
        self.args.action = self.args.action.replace('-', '_')
        logger.debug("action: '%s", self.args.action)

        # Instantiate input DataItem object
        self.input = DataItem(path=self.args.input)

        # Instantiate output DataItem object
        if self.args.action in self._actions_have_dir_output:
            self.output = DataItem(path=self.__get_default_output())
        elif self.args.action in self._actions_have_vid_output:
            if self.__check_have_fps(self.args.fps) > 0:
                self.output = DataItem(path=self.__get_default_output(),
                                       fps=self.args.fps)
            else:
                self.output = DataItem(path=self.__get_default_output())

        if self.args.ref_vid is None \
                or self.args.ref_vid == '':
            self.args.ref_vid = None

        # Instantiate ref_vid DataItem object
        self.ref_vid = DataItem(path=self.args.ref_vid)

        # Check that correct input and output arguments were provided
        if self.args.action in self._actions_have_dir_input and not self.input.is_type("dir"):
            raise ValueError("The chosen action requires a directory as its "
                             "input, but you entered: "
                             "{}".format(self.input.path))
        if self.args.action in self._actions_have_vid_input and not self.input.is_type("vid"):
            raise ValueError("The chosen action requires a video as its "
                             "input, but you entered: "
                             "{}".format(self.input.path))
        if self.args.action in self._actions_have_dir_output and not self.output.is_type("dir"):
            raise ValueError("The chosen action requires a directory as its "
                             "output, but you entered: "
                             "{}".format(self.output.path))
        if self.args.action in self._actions_have_vid_output and not self.output.is_type("vid"):
            raise ValueError("The chosen action requires a video as its "
                             "output, but you entered: "
                             "{}".format(self.output.path))

        # Check that ref_vid is a video when it needs to be
        if self.args.action in self._actions_req_ref_video:
            if self.ref_vid.is_type("none"):
                raise ValueError("The file chosen as the reference video is "
                                 "not a video, either leave the field blank "
                                 "or type 'None': "
                                 "{}".format(self.ref_vid.path))
        elif self.args.action in self._actions_can_use_ref_video:
            if self.ref_vid.is_type("none"):
                logger.warning("Warning: no reference video was supplied, even though "
                               "one may be used with the chosen action. If this is "
                               "intentional then ignore this warning.")

        # Process start and duration arguments
        self.start = self.parse_time(self.args.start)
        self.end = self.parse_time(self.args.end)
        if not self.__check_equals_time(self.args.end, "00:00:00"):
            self.duration = self.__get_duration(self.start, self.end)
        else:
            self.duration = self.parse_time(str(self.args.duration))
        # If fps was left blank in gui, set it to default -1.0 value
        if self.args.fps == '':
            self.args.fps = str(-1.0)

        # Try to set fps automatically if needed and not supplied by user
        if self.args.action in self._actions_req_fps \
                and self.__convert_fps(self.args.fps) <= 0:
            if self.__check_have_fps(['r', 'i']):
                _error_str = "No fps, input or reference video was supplied, "
                _error_str += "hence it's not possible to "
                _error_str += "'{}'.".format(self.args.action)
                raise ValueError(_error_str)
            if self.output.fps is not None and self.__check_have_fps(['r', 'i']):
                self.args.fps = self.output.fps
            elif self.ref_vid.fps is not None and self.__check_have_fps(['i']):
                self.args.fps = self.ref_vid.fps
            elif self.input.fps is not None and self.__check_have_fps(['r']):
                self.args.fps = self.input.fps

        # Processing transpose
        if self.args.transpose is None or \
                self.args.transpose.lower() == "none":
            self.args.transpose = None
        else:
            self.args.transpose = self.args.transpose[1]

        # Processing degrees
        if self.args.degrees is None \
                or self.args.degrees.lower() == "none" \
                or self.args.degrees == '':
            self.args.degrees = None
        elif self.args.transpose is None:
            try:
                int(self.args.degrees)
            except ValueError:
                logger.error("You have entered an invalid value for degrees: %s",
                             self.args.degrees)
                sys.exit(1)

        # Set verbosity of output
        self.__set_verbosity(self.args.quiet, self.args.verbose)

        # Set self.print_ to True if output needs to be printed to stdout
        if self.args.action in self._actions_have_print_output:
            self.print_ = True

        self.effmpeg_process()
        logger.debug("Finished Effmpeg process")

    def effmpeg_process(self):
        """ The effmpeg process """
        kwargs = {"input_": self.input,
                  "output": self.output,
                  "ref_vid": self.ref_vid,
                  "fps": self.args.fps,
                  "extract_ext": self.args.extract_ext,
                  "start": self.start,
                  "duration": self.duration,
                  "mux_audio": self.args.mux_audio,
                  "degrees": self.args.degrees,
                  "transpose": self.args.transpose,
                  "scale": self.args.scale,
                  "print_": self.print_,
                  "exe": self.exe}
        action = getattr(self, self.args.action)
        action(**kwargs)

    @staticmethod
    def extract(input_=None, output=None, fps=None,  # pylint:disable=unused-argument
                extract_ext=None, start=None, duration=None, **kwargs):
        """ Extract video to image frames """
        logger.debug("input_: %s, output: %s, fps: %s, extract_ext: '%s', start: %s, duration: %s",
                     input_, output, fps, extract_ext, start, duration)
        _input_opts = Effmpeg._common_ffmpeg_args[:]
        if start is not None and duration is not None:
            _input_opts += '-ss {} -t {}'.format(start, duration)
        _input = {input_.path: _input_opts}
        _output_opts = '-y -vf fps="' + str(fps) + '" -q:v 1'
        _output_path = output.path + "/" + input_.name + "_%05d" + extract_ext
        _output = {_output_path: _output_opts}
        os.makedirs(output.path, exist_ok=True)
        logger.debug("_input: %s, _output: %s", _input, _output)
        Effmpeg.__run_ffmpeg(inputs=_input, outputs=_output)

    @staticmethod
    def gen_vid(input_=None, output=None, fps=None,  # pylint:disable=unused-argument
                mux_audio=False, ref_vid=None, exe=None, **kwargs):
        """ Generate Video """
        logger.debug("input: %s, output: %s, fps: %s, mux_audio: %s, ref_vid: '%s'exe: '%s'",
                     input, output, fps, mux_audio, ref_vid, exe)
        filename = Effmpeg.__get_extracted_filename(input_.path)
        _input_opts = Effmpeg._common_ffmpeg_args[:]
        _input_path = os.path.join(input_.path, filename)
        _fps_arg = '-r ' + str(fps) + ' '
        _input_opts += _fps_arg + "-f image2 "
        _output_opts = '-y ' + _fps_arg + ' -c:v libx264'
        if mux_audio:
            _ref_vid_opts = '-c copy -map 0:0 -map 1:1'
            _output_opts = _ref_vid_opts + ' ' + _output_opts
            _inputs = OrderedDict([(_input_path, _input_opts), (ref_vid.path, None)])
        else:
            _inputs = {_input_path: _input_opts}
        _outputs = {output.path: _output_opts}
        logger.debug("_inputs: %s, _outputs: %s", _inputs, _outputs)
        Effmpeg.__run_ffmpeg(exe=exe, inputs=_inputs, outputs=_outputs)

    @staticmethod
    def get_fps(input_=None, print_=False, **kwargs):
        """ Get Frames per Second """
        logger.debug("input_: %s, print_: %s, kwargs: %s", input_, print_, kwargs)
        input_ = input_ if isinstance(input_, str) else input_.path
        logger.debug("input: %s", input_)
        reader = imageio.get_reader(input_, "ffmpeg")
        _fps = reader.get_meta_data()["fps"]
        logger.debug(_fps)
        reader.close()
        if print_:
            logger.info("Video fps: %s", _fps)
        return _fps

    @staticmethod
    def get_info(input_=None, print_=False, **kwargs):
        """ Get video Info """
        logger.debug("input_: %s, print_: %s, kwargs: %s", input_, print_, kwargs)
        input_ = input_ if isinstance(input_, str) else input_.path
        logger.debug("input: %s", input_)
        reader = imageio.get_reader(input_, "ffmpeg")
        out = reader.get_meta_data()
        logger.debug(out)
        reader.close()
        if print_:
            logger.info("======== Video Info ========",)
            logger.info("path: %s", input_)
            for key, val in out.items():
                logger.info("%s: %s", key, val)
        return out

    @staticmethod
    def rescale(input_=None, output=None, scale=None,  # pylint:disable=unused-argument
                exe=None, **kwargs):
        """ Rescale Video """
        _input_opts = Effmpeg._common_ffmpeg_args[:]
        _output_opts = '-y -vf scale="' + str(scale) + '"'
        _inputs = {input_.path: _input_opts}
        _outputs = {output.path: _output_opts}
        Effmpeg.__run_ffmpeg(exe=exe, inputs=_inputs, outputs=_outputs)

    @staticmethod
    def rotate(input_=None, output=None, degrees=None,  # pylint:disable=unused-argument
               transpose=None, exe=None, **kwargs):
        """ Rotate Video """
        if transpose is None and degrees is None:
            raise ValueError("You have not supplied a valid transpose or "
                             "degrees value:\ntranspose: {}\ndegrees: "
                             "{}".format(transpose, degrees))

        _input_opts = Effmpeg._common_ffmpeg_args[:]
        _output_opts = '-y -c:a copy -vf '
        _bilinear = ''
        if transpose is not None:
            _output_opts += 'transpose="' + str(transpose) + '"'
        elif int(degrees) != 0:
            if int(degrees) % 90 == 0 and int(degrees) != 0:
                _bilinear = ":bilinear=0"
            _output_opts += 'rotate="' + str(degrees) + '*(PI/180)'
            _output_opts += _bilinear + '" '

        _inputs = {input_.path: _input_opts}
        _outputs = {output.path: _output_opts}
        Effmpeg.__run_ffmpeg(exe=exe, inputs=_inputs, outputs=_outputs)

    @staticmethod
    def mux_audio(input_=None, output=None, ref_vid=None,  # pylint:disable=unused-argument
                  exe=None, **kwargs):
        """ Mux Audio """
        _input_opts = Effmpeg._common_ffmpeg_args[:]
        _ref_vid_opts = None
        _output_opts = '-y -c copy -map 0:0 -map 1:1 -shortest'
        _inputs = OrderedDict([(input_.path, _input_opts), (ref_vid.path, _ref_vid_opts)])
        _outputs = {output.path: _output_opts}
        Effmpeg.__run_ffmpeg(exe=exe, inputs=_inputs, outputs=_outputs)

    @staticmethod
    def slice(input_=None, output=None, start=None,  # pylint:disable=unused-argument
              duration=None, exe=None, **kwargs):
        """ Slice Video """
        _input_opts = Effmpeg._common_ffmpeg_args[:]
        _input_opts += "-ss " + start
        _output_opts = "-t " + duration + " "
        _inputs = {input_.path: _input_opts}
        _output = {output.path: _output_opts}
        Effmpeg.__run_ffmpeg(exe=exe, inputs=_inputs, outputs=_output)

    # Various helper methods
    @classmethod
    def __set_verbosity(cls, quiet, verbose):
        if verbose:
            cls._common_ffmpeg_args = cls.__common_ffmpeg_args_dict["verbose"]
        elif quiet:
            cls._common_ffmpeg_args = cls.__common_ffmpeg_args_dict["quiet"]
        else:
            cls._common_ffmpeg_args = cls.__common_ffmpeg_args_dict["normal"]

    def __get_default_output(self):
        """ Set output to the same directory as input
            if the user didn't specify it. """
        if self.args.output == "":
            if self.args.action in self._actions_have_dir_output:
                retval = os.path.join(self.input.dirname, 'out')
            elif self.args.action in self._actions_have_vid_output:
                if self.input.is_type("media"):
                    # Using the same extension as input leads to very poor
                    # output quality, hence the default is mkv for now
                    retval = os.path.join(self.input.dirname, "out.mkv")  # + self.input.ext)
                else:  # case if input was a directory
                    retval = os.path.join(self.input.dirname, 'out.mkv')
        else:
            retval = self.args.output
        logger.debug(retval)
        return retval

    def __check_have_fps(self, items):
        items_to_check = list()
        for i in items:
            if i == 'r':
                items_to_check.append('ref_vid')
            elif i == 'i':
                items_to_check.append('input')
            elif i == 'o':
                items_to_check.append('output')

        return all(getattr(self, i).fps is None for i in items_to_check)

    @staticmethod
    def __run_ffmpeg(exe=im_ffm.get_ffmpeg_exe(), inputs=None, outputs=None):
        """ Run ffmpeg """
        logger.debug("Running ffmpeg: (exe: '%s', inputs: %s, outputs: %s", exe, inputs, outputs)
        ffm = FFmpeg(executable=exe, inputs=inputs, outputs=outputs)
        try:
            ffm.run(stderr=subprocess.STDOUT)
        except FFRuntimeError as ffe:
            # After receiving SIGINT ffmpeg has a 255 exit code
            if ffe.exit_code == 255:
                pass
            else:
                raise ValueError("An unexpected FFRuntimeError occurred: "
                                 "{}".format(ffe))
        except KeyboardInterrupt:
            pass  # Do nothing if voluntary interruption
        logger.debug("ffmpeg finished")

    @staticmethod
    def __convert_fps(fps):
        """ Convert to Frames per Second """
        if '/' in fps:
            _fps = fps.split('/')
            retval = float(_fps[0]) / float(_fps[1])
        else:
            retval = float(fps)
        logger.debug(retval)
        return retval

    @staticmethod
    def __get_duration(start_time, end_time):
        """ Get the duration """
        start = [int(i) for i in start_time.split(':')]
        end = [int(i) for i in end_time.split(':')]
        start = datetime.timedelta(hours=start[0], minutes=start[1], seconds=start[2])
        end = datetime.timedelta(hours=end[0], minutes=end[1], seconds=end[2])
        delta = end - start
        secs = delta.total_seconds()
        retval = '{:02}:{:02}:{:02}'.format(int(secs // 3600),
                                            int(secs % 3600 // 60),
                                            int(secs % 60))
        logger.debug(retval)
        return retval

    @staticmethod
    def __get_extracted_filename(path):
        """ Get the extracted filename """
        logger.debug("path: '%s'", path)
        filename = ''
        for file in os.listdir(path):
            if any(i in file for i in DataItem.img_ext):
                filename = file
                break
        logger.debug("sample filename: '%s'", filename)
        filename, img_ext = os.path.splitext(filename)
        zero_pad = Effmpeg.__get_zero_pad(filename)
        name = filename[:-zero_pad]
        retval = "{}%{}d{}".format(name, zero_pad, img_ext)
        logger.debug("filename: %s, img_ext: '%s', zero_pad: %s, name: '%s'",
                     filename, img_ext, zero_pad, name)
        logger.debug(retval)
        return retval

    @staticmethod
    def __get_zero_pad(filename):
        """ Return the starting position of zero padding from a filename """
        chkstring = filename[::-1]
        logger.trace("filename: %s, chkstring: %s", filename, chkstring)
        pos = 0
        for pos in range(len(chkstring)):
            if not chkstring[pos].isdigit():
                break
        logger.debug("filename: '%s', pos: %s", filename, pos)
        return pos

    @staticmethod
    def __check_is_valid_time(value):
        """ Check valid time """
        val = value.replace(':', '')
        retval = val.isdigit()
        logger.debug("value: '%s', retval: %s", value, retval)
        return retval

    @staticmethod
    def __check_equals_time(value, time):
        """ Check equals time """
        val = value.replace(':', '')
        tme = time.replace(':', '')
        retval = val.zfill(6) == tme.zfill(6)
        logger.debug("value: '%s', time: %s, retval: %s", value, time, retval)
        return retval

    @staticmethod
    def parse_time(txt):
        """ Parse Time """
        clean_txt = txt.replace(':', '')
        hours = clean_txt[0:2]
        minutes = clean_txt[2:4]
        seconds = clean_txt[4:6]
        retval = hours + ':' + minutes + ':' + seconds
        logger.debug("txt: '%s', retval: %s", txt, retval)
        return retval
