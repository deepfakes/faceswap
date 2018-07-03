#!/usr/bin/env python3
# vim: set fileencoding=utf-8 :
"""
Created on 2018-03-16 15:14

@author: Lev Velykoivanenko (velykoivanenko.lev@gmail.com)
"""
# TODO: integrate preview into gui window
# TODO: add preview support when muxing audio
#       -> figure out if ffmpeg | ffplay would work on windows and mac
import os
import sys
import subprocess
import datetime

from ffmpy import FFprobe, FFmpeg, FFRuntimeError

# faceswap imports
from lib.cli import FullHelpArgumentParser
from lib.utils import _image_extensions, _video_extensions
from . import cli

if sys.version_info[0] < 3:
    raise Exception("This program requires at least python3.2")
if sys.version_info[0] == 3 and sys.version_info[1] < 2:
    raise Exception("This program requires at least python3.2")


class DataItem(object):
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

    def set_name(self, name=None):
        if name is None and self.path is not None:
            self.name = os.path.basename(self.path)
        elif name is not None and self.path is None:
            self.name = os.path.basename(name)
        elif name is not None and self.path is not None:
            self.name = os.path.basename(name)
        else:
            self.name = None

    def set_type_ext(self, path=None):
        if path is not None:
            self.path = path
        if self.path is not None:
            item_ext = os.path.splitext(self.path)[1]
            if item_ext in DataItem.vid_ext:
                item_type = 'vid'
            elif item_ext in DataItem.audio_ext:
                item_type = 'audio'
            else:
                item_type = 'dir'
            self.type = item_type
            self.ext = item_ext
        else:
            return

    def set_dirname(self, path=None):
        if path is None and self.path is not None:
            self.dirname = os.path.dirname(self.path)
        elif path is not None and self.path is None:
            self.dirname = os.path.dirname(path)
        elif path is not None and self.path is not None:
            self.dirname = os.path.dirname(path)
        else:
            self.dirname = None

    def is_type(self, item_type=None):
        if item_type == "media":
            return self.type in "vid audio"
        elif item_type == "dir":
            return self.type == "dir"
        elif item_type == "vid":
            return self.type == "vid"
        elif item_type == "audio":
            return self.type == "audio"
        elif item_type.lower() == "none":
            return self.type is None
        else:
            return False

    def set_fps(self):
        try:
            self.fps = Effmpeg.get_fps(self.path)
        except FFRuntimeError:
            self.fps = None


class Effmpeg(object):
    """
    Class that allows for "easy" ffmpeg use. It provides a nice cli interface
    for common video operations.
    """

    _actions_req_fps = ["extract", "gen_vid"]
    _actions_req_ref_video = ["mux_audio"]
    _actions_can_preview = ["gen_vid", "mux_audio", "rescale", "rotate",
                            "slice"]
    _actions_can_use_ref_video = ["gen_vid"]
    _actions_have_dir_output = ["extract"]
    _actions_have_vid_output = ["gen_vid", "mux_audio", "rescale", "rotate",
                                "slice"]
    _actions_have_print_output = ["get_fps", "get_info"]
    _actions_have_dir_input = ["gen_vid"]
    _actions_have_vid_input = ["extract", "get_fps", "get_info", "rescale",
                               "rotate", "slice"]

    # Class variable that stores the target executable (ffmpeg or ffplay)
    _executable = 'ffmpeg'

    # Class variable that stores the common ffmpeg arguments based on verbosity
    __common_ffmpeg_args_dict = {"normal": "-hide_banner ",
                                 "quiet": "-loglevel panic -hide_banner ",
                                 "verbose": ''}

    # _common_ffmpeg_args is the class variable that will get used by various
    # actions and it will be set by the process_arguments() method based on
    # passed verbosity
    _common_ffmpeg_args = ''

    def __init__(self, arguments):
        self.args = arguments
        self.exe = "ffmpeg"
        self.input = DataItem()
        self.output = DataItem()
        self.ref_vid = DataItem()
        self.start = ""
        self.end = ""
        self.duration = ""
        self.print_ = False

    def process(self):
        # Format action to match the method name
        self.args.action = self.args.action.replace('-', '_')

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
                print("Warning: no reference video was supplied, even though "
                      "one may be used with the chosen action. If this is "
                      "intentional then ignore this warning.", file=sys.stderr)

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
            elif self.output.fps is not None and self.__check_have_fps(['r', 'i']):
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
            except ValueError as ve:
                print("You have entered an invalid value for degrees: "
                      "{}".format(self.args.degrees), file=sys.stderr)
                exit(1)

        # Set executable based on whether previewing or not
        if self.args.preview and self.args.action in self._actions_can_preview:
            self.exe = 'ffplay'
            self.output = DataItem()

        # Set verbosity of output
        self.__set_verbosity(self.args.quiet, self.args.verbose)

        # Set self.print_ to True if output needs to be printed to stdout
        if self.args.action in self._actions_have_print_output:
            self.print_ = True

        self.effmpeg_process()

    def effmpeg_process(self):
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
                  "preview": self.args.preview,
                  "exe": self.exe}
        action = getattr(self, self.args.action)
        action(**kwargs)

    @staticmethod
    def extract(input_=None, output=None, fps=None, extract_ext=None,
                **kwargs):
        _input_opts = Effmpeg._common_ffmpeg_args[:]
        _input = {input_.path: _input_opts}
        _output_opts = '-y -vf fps="' + str(fps) + '"'
        _output_path = output.path + "/" + input_.name + "_%05d" + extract_ext
        _output = {_output_path: _output_opts}
        os.makedirs(output.path, exist_ok=True)
        Effmpeg.__run_ffmpeg(inputs=_input, outputs=_output)

    @staticmethod
    def gen_vid(input_=None, output=None, fps=None, mux_audio=False,
                ref_vid=None, preview=False, exe=None, **kwargs):
        filename = Effmpeg.__get_extracted_filename(input_.path)
        _input_opts = Effmpeg._common_ffmpeg_args[:]
        _input_path = os.path.join(input_.path, filename)
        _output_opts = '-vf fps="' + str(fps) + '" '
        if not preview:
            _output_opts = '-y ' + _output_opts + ' -c:v libx264'
        if mux_audio:
            _ref_vid_opts = '-c copy -map 0:0 -map 1:1'
            if preview:
                raise ValueError("Preview for gen-vid with audio muxing is "
                                 "not supported.")
            _output_opts = _ref_vid_opts + ' ' + _output_opts
            _inputs = {_input_path: _input_opts, ref_vid.path: None}
        else:
            _inputs = {_input_path: _input_opts}
        _outputs = {output.path: _output_opts}
        Effmpeg.__run_ffmpeg(exe=exe, inputs=_inputs, outputs=_outputs)

    @staticmethod
    def get_fps(input_=None, print_=False, **kwargs):
        _input_opts = '-v error -select_streams v -of '
        _input_opts += 'default=noprint_wrappers=1:nokey=1 '
        _input_opts += '-show_entries stream=r_frame_rate'
        if type(input_) == str:
            _inputs = {input_: _input_opts}
        else:
            _inputs = {input_.path: _input_opts}
        ff = FFprobe(inputs=_inputs)
        _fps = ff.run(stdout=subprocess.PIPE)[0].decode("utf-8")
        _fps = _fps.strip()
        if print_:
            print("Video fps:", _fps)
        else:
            return _fps

    @staticmethod
    def get_info(input_=None, print_=False, **kwargs):
        _input_opts = Effmpeg._common_ffmpeg_args[:]
        _inputs = {input_.path: _input_opts}
        ff = FFprobe(inputs=_inputs)
        out = ff.run(stdout=subprocess.PIPE,
                     stderr=subprocess.STDOUT)[0].decode('utf-8')
        if print_:
            print(out)
        else:
            return out

    @staticmethod
    def rescale(input_=None, output=None, scale=None, preview=False, exe=None,
                **kwargs):
        _input_opts = Effmpeg._common_ffmpeg_args[:]
        _output_opts = '-vf scale="' + str(scale) + '"'
        if not preview:
            _output_opts = '-y ' + _output_opts
        _inputs = {input_.path: _input_opts}
        _outputs = {output.path: _output_opts}
        Effmpeg.__run_ffmpeg(exe=exe, inputs=_inputs, outputs=_outputs)

    @staticmethod
    def rotate(input_=None, output=None, degrees=None, transpose=None,
               preview=None, exe=None, **kwargs):
        if transpose is None and degrees is None:
            raise ValueError("You have not supplied a valid transpose or "
                             "degrees value:\ntranspose: {}\ndegrees: "
                             "{}".format(transpose, degrees))

        _input_opts = Effmpeg._common_ffmpeg_args[:]
        _output_opts = '-vf '
        if not preview:
            _output_opts = '-y -c:a copy ' + _output_opts
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
    def mux_audio(input_=None, output=None, ref_vid=None, preview=None,
                  exe=None, **kwargs):
        _input_opts = Effmpeg._common_ffmpeg_args[:]
        _ref_vid_opts = None
        _output_opts = '-y -c copy -map 0:0 -map 1:1 -shortest'
        if preview:
            raise ValueError("Preview with audio muxing is not supported.")
        """
        if not preview:
            _output_opts = '-y ' + _output_opts
        """
        _inputs = {input_.path: _input_opts, ref_vid.path: _ref_vid_opts}
        _outputs = {output.path: _output_opts}
        Effmpeg.__run_ffmpeg(exe=exe, inputs=_inputs, outputs=_outputs)

    @staticmethod
    def slice(input_=None, output=None, start=None, duration=None,
              preview=None, exe=None,  **kwargs):
        _input_opts = Effmpeg._common_ffmpeg_args[:]
        _input_opts += "-ss " + start
        _output_opts = "-t " + duration + " "
        if not preview:
            _output_opts = '-y ' + _output_opts + "-vcodec copy -acodec copy"
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
        # Set output to the same directory as input
        # if the user didn't specify it.
        if self.args.output == "":
            if self.args.action in self._actions_have_dir_output:
                return os.path.join(self.input.dirname, 'out')
            elif self.args.action in self._actions_have_vid_output:
                if self.input.is_type("media"):
                    # Using the same extension as input leads to very poor
                    # output quality, hence the default is mkv for now
                    return os.path.join(self.input.dirname,
                                        "out.mkv")  # + self.input.ext)
                else:  # case if input was a directory
                    return os.path.join(self.input.dirname, 'out.mkv')
        else:
            return self.args.output

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
    def __run_ffmpeg(exe="ffmpeg", inputs=None, outputs=None):
        ff = FFmpeg(executable=exe, inputs=inputs, outputs=outputs)
        try:
            ff.run(stderr=subprocess.STDOUT)
        except FFRuntimeError as ffe:
            # After receiving SIGINT ffmpeg has a 255 exit code
            if ffe.exit_code == 255:
                pass
            else:
                raise ValueError("An unexpected FFRuntimeError occurred: "
                                 "{}".format(ffe))
        except KeyboardInterrupt:
            pass  # Do nothing if voluntary interruption

    @staticmethod
    def __convert_fps(fps):
        if '/' in fps:
            _fps = fps.split('/')
            return float(_fps[0]) / float(_fps[1])
        else:
            return float(fps)

    @staticmethod
    def __get_duration(start_time, end_time):
        start = [int(i) for i in start_time.split(':')]
        end = [int(i) for i in end_time.split(':')]
        start = datetime.timedelta(hours=start[0], minutes=start[1], seconds=start[2])
        end = datetime.timedelta(hours=end[0], minutes=end[1], seconds=end[2])
        delta = end - start
        s = delta.total_seconds()
        return '{:02}:{:02}:{:02}'.format(int(s // 3600), int(s % 3600 // 60), int(s % 60))

    @staticmethod
    def __get_extracted_filename(path):
        filename = ''
        for file in os.listdir(path):
            if any(i in file for i in DataItem.img_ext):
                filename = file
                break
        filename = filename.split('.')
        img_ext = filename[-1]
        zero_pad = filename[-2]
        name = '.'.join(filename[:-2])

        vid_ext = ''
        underscore = ''
        for ve in [ve.replace('.', '') for ve in DataItem.vid_ext]:
            if ve in zero_pad:
                vid_ext = ve
                zero_pad = zero_pad.replace(ve, '')
                if '_' in zero_pad:
                    zero_pad = len(zero_pad.replace('_', ''))
                    underscore = '_'
                else:
                    zero_pad = len(zero_pad)
                break

        zero_pad = str(zero_pad).zfill(2)
        filename_list = [name, vid_ext + underscore + '%' + zero_pad + 'd',
                         img_ext]
        return '.'.join(filename_list)

    @staticmethod
    def __check_is_valid_time(value):
        val = value.replace(':', '')
        return val.isdigit()

    @staticmethod
    def __check_equals_time(value, time):
        v = value.replace(':', '')
        t = time.replace(':', '')
        return v.zfill(6) == t.zfill(6)

    @staticmethod
    def parse_time(txt):
        clean_txt = txt.replace(':', '')
        hours = clean_txt[0:2]
        minutes = clean_txt[2:4]
        seconds = clean_txt[4:6]
        return hours + ':' + minutes + ':' + seconds


def bad_args(args):
    """ Print help on bad arguments """
    PARSER.print_help()
    exit(0)


if __name__ == "__main__":
    print('"Easy"-ffmpeg wrapper.\n')

    PARSER = FullHelpArgumentParser()
    SUBPARSER = PARSER.add_subparsers()
    EFFMPEG = cli.EffmpegArgs(
        SUBPARSER, "effmpeg", "Wrapper for various common ffmpeg commands.")
    PARSER.set_defaults(func=bad_args)
    ARGUMENTS = PARSER.parse_args()
    ARGUMENTS.func(ARGUMENTS)

