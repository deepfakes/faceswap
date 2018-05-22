#!/usr/bin/env python3
# vim: set fileencoding=utf-8 :
"""
Created on 2018-03-16 15:14

@author: Lev Velykoivanenko (velykoivanenko.lev@gmail.com)
"""
import argparse
import os
import sys
import subprocess
import datetime

from ffmpy import FFprobe, FFmpeg, FFRuntimeError

# faceswap imports
from lib.cli import FileFullPaths, ComboFullPaths
from lib.utils import _image_extensions, _video_extensions


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
    _actions_can_use_ref_video = ["gen_vid"]
    _actions_have_dir_output = ["extract"]
    _actions_have_vid_output = ["gen_vid", "mux_audio", "rescale", "rotate",
                                "slice"]
    _actions_have_print_output = ["get_fps", "get_info"]
    _actions_have_dir_input = ["gen_vid"]
    _actions_have_vid_input = ["extract", "get_fps", "get_info", "rescale",
                               "rotate", "slice"]

    # Class variable that stores the common ffmpeg arguments based on verbosity
    __common_ffmpeg_args_dict = {"normal": "-hide_banner ",
                                 "quiet": "-loglevel panic -hide_banner ",
                                 "verbose": ''}

    # _common_ffmpeg_args is the class variable that will get used by various
    # actions and it will be set by the process_arguments() method based on
    # passed verbosity
    _common_ffmpeg_args = ''

    def __init__(self, subparser, command, description='default'):
        self.argument_list = self.get_argument_list()
        self.optional_arguments = list()
        self.args = None
        self.input = DataItem()
        self.output = DataItem()
        self.ref_vid = DataItem()
        self.start = ""
        self.end = ""
        self.duration = ""
        self.print_ = False
        self.parse_arguments(description, subparser, command)

    @staticmethod
    def get_argument_list():
        vid_files = FileFullPaths.prep_filetypes([["Video Files",
                                                  DataItem.vid_ext]])
        arguments_list = list()
        arguments_list.append({"opts": ('-a', '--action'),
                               "dest": "action",
                               "choices": ("extract", "gen-vid", "get-fps",
                                           "get-info", "mux-audio", "rescale",
                                           "rotate", "slice"),
                               "default": "extract",
                               "help": """Choose which action you want ffmpeg 
                                          ffmpeg to do.
                                          'slice' cuts a portion of the video
                                          into a separate video file.
                                          'get-fps' returns the chosen video's
                                          fps."""})

        arguments_list.append({"opts": ('-i', '--input'),
                               "action": ComboFullPaths,
                               "dest": "input",
                               "default": "input",
                               "help": "Input file.",
                               "required": True,
                               "actions_open_type": {
                                   "task_name": "effmpeg",
                                   "extract": "load",
                                   "gen-vid": "folder",
                                   "get-fps": "load",
                                   "get-info": "load",
                                   "mux-audio": "load",
                                   "rescale": "load",
                                   "rotate": "load",
                                   "slice": "load",
                               },
                               "filetypes": {
                                   "extract": vid_files,
                                   "gen-vid": None,
                                   "get-fps": vid_files,
                                   "get-info": vid_files,
                                   "mux-audio": vid_files,
                                   "rescale": vid_files,
                                   "rotate": vid_files,
                                   "slice": vid_files
                               }})

        arguments_list.append({"opts": ('-o', '--output'),
                               "action": ComboFullPaths,
                               "dest": "output",
                               "default": "",
                               "help": """Output file. If no output is 
                                          specified then: if the output is  
                                          meant to be a video then a video 
                                          called 'out.mkv' will be created in 
                                          the input directory; if the output is
                                          meant to be a directory then a 
                                          directory called 'out' will be 
                                          created inside the input 
                                          directory.
                                          Note: the chosen output file 
                                          extension will determine the file
                                          encoding.""",
                               "actions_open_type": {
                                   "task_name": "effmpeg",
                                   "extract": "save",
                                   "gen-vid": "save",
                                   "get-fps": "nothing",
                                   "get-info": "nothing",
                                   "mux-audio": "save",
                                   "rescale": "save",
                                   "rotate": "save",
                                   "slice": "save"
                               },
                               "filetypes": {
                                   "extract": None,
                                   "gen-vid": vid_files,
                                   "get-fps": None,
                                   "get-info": None,
                                   "mux-audio": vid_files,
                                   "rescale": vid_files,
                                   "rotate": vid_files,
                                   "slice": vid_files
                               }})

        arguments_list.append({"opts": ('-r', '--reference-video'),
                               "action": ComboFullPaths,
                               "dest": "ref_vid",
                               "default": "None",
                               "help": """Path to reference video if 'input' 
                                          was not a video.""",
                               "actions_open_type": {
                                   "task_name": "effmpeg",
                                   "extract": "nothing",
                                   "gen-vid": "load",
                                   "get-fps": "nothing",
                                   "get-info": "nothing",
                                   "mux-audio": "load",
                                   "rescale": "nothing",
                                   "rotate": "nothing",
                                   "slice": "nothing"
                               },
                               "filetypes": {
                                   "extract": None,
                                   "gen-vid": vid_files,
                                   "get-fps": None,
                                   "get-info": None,
                                   "mux-audio": vid_files,
                                   "rescale": None,
                                   "rotate": None,
                                   "slice": None
                               }})

        arguments_list.append({"opts": ('-fps', '--fps'),
                               "type": str,
                               "dest": "fps",
                               "default": "-1.0",
                               "help": """Provide video fps. Can be an integer,
                                          float or fraction. Negative values 
                                          will make the program try to get the 
                                          fps from the input or reference 
                                          videos."""})

        arguments_list.append({"opts": ("-ef", "--extract-filetype"),
                               "choices": DataItem.img_ext,
                               "dest": "extract_ext",
                               "default": ".png",
                               "help": """Image format that extracted images
                                          should be saved as. '.bmp' will offer
                                          the fastest extraction speed, but
                                          will take the most storage space.
                                          '.png' will be slower but will take
                                          less storage."""})

        arguments_list.append({"opts": ('-s', '--start'),
                               "type": str,
                               "dest": "start",
                               "default": "00:00:00",
                               "help": """Enter the start time from which an 
                                          action is to be applied.
                                          Default: 00:00:00, in HH:MM:SS 
                                          format. You can also enter the time
                                          with or without the colons, e.g. 
                                          00:0000 or 026010."""})

        arguments_list.append({"opts": ('-e', '--end'),
                               "type": str,
                               "dest": "end",
                               "default": "00:00:00",
                               "help": """Enter the end time to which an action
                                          is to be applied. If both an end time
                                          and duration are set, then the end 
                                          time will be used and the duration 
                                          will be ignored.
                                          Default: 00:00:00, in HH:MM:SS."""})

        arguments_list.append({"opts": ('-d', '--duration'),
                               "type": str,
                               "dest": "duration",
                               "default": "00:00:00",
                               "help": """Enter the duration of the chosen
                                          action, for example if you enter
                                          00:00:10 for slice, then the first 10 
                                          seconds after and including the start
                                          time will be cut out into a new
                                          video.
                                          Default: 00:00:00, in HH:MM:SS 
                                          format. You can also enter the time
                                          with or without the colons, e.g. 
                                          00:0000 or 026010."""})

        arguments_list.append({"opts": ('-m', '--mux-audio'),
                               "action": "store_true",
                               "dest": "mux_audio",
                               "default": False,
                               "help": """Mux the audio from the reference 
                                          video into the input video. This
                                          option is only used for the 'gen-vid'
                                          action. 'mux-audio' action has this
                                          turned on implicitly."""})

        arguments_list.append({"opts": ('-tr', '--transpose'),
                               "choices": ("(0, 90CounterClockwise&VerticalFlip)",
                                           "(1, 90Clockwise)",
                                           "(2, 90CounterClockwise)",
                                           "(3, 90Clockwise&VerticalFlip)",
                                           "None"),
                               "type": lambda v: Effmpeg.__parse_transpose(v),
                               "dest": "transpose",
                               "default": "None",
                               "help": """Transpose the video. If transpose is 
                                          set, then degrees will be ignored. For
                                          cli you can enter either the number
                                          or the long command name, 
                                          e.g. to use (1, 90Clockwise)
                                          -tr 1 or -tr 90Clockwise"""})

        arguments_list.append({"opts": ('-de', '--degrees'),
                               "type": str,
                               "dest": "degrees",
                               "default": "None",
                               "help": """Rotate the video clockwise by the 
                                          given number of degrees."""})

        arguments_list.append({"opts": ('-sc', '--scale'),
                               "type": str,
                               "dest": "scale",
                               "default": "1920x1080",
                               "help": """Set the new resolution scale if the
                                          chosen action is 'rescale'."""})

        arguments_list.append({"opts": ('-q', '--quiet'),
                               "action": "store_true",
                               "dest": "quiet",
                               "default": False,
                               "help": """Reduces output verbosity so that only
                                          serious errors are printed. If both
                                          quiet and verbose are set, verbose
                                          will override quiet."""})

        arguments_list.append({"opts": ('-v', '--verbose'),
                               "action": "store_true",
                               "dest": "verbose",
                               "default": False,
                               "help": """Increases output verbosity. If both
                                          quiet and verbose are set, verbose
                                          will override quiet."""})

        return arguments_list

    def parse_arguments(self, description, subparser, command):
        parser = subparser.add_parser(
                command,
                help="This command lets you easily invoke"
                     "common ffmpeg commands.",
                description=description,
                epilog="Questions and feedback: \
                        https://github.com/deepfakes/faceswap-playground"
        )

        for option in self.argument_list:
            args = option['opts']
            kwargs = {key: option[key] for key in option.keys() if key != 'opts'}
            parser.add_argument(*args, **kwargs)

        parser.set_defaults(func=self.process_arguments)

    def process_arguments(self, arguments):
        self.args = arguments

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

        if self.args.ref_vid.lower() == "none" or self.args.ref_vid == '':
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
        if self.args.transpose.lower() == "none":
            self.args.transpose = None
        else:
            self.args.transpose = self.args.transpose[1]

        # Processing degrees
        if self.args.degrees.lower() == "none" or self.args.degrees == '':
            self.args.degrees = None
        elif self.args.transpose is None:
            try:
                int(self.args.degrees)
            except ValueError as ve:
                print("You have entered an invalid value for degrees: "
                      "{}".format(self.args.degrees), file=sys.stderr)
                exit(1)

        # Set verbosity of output
        self.__set_verbosity(self.args.quiet, self.args.verbose)

        # Set self.print_ to True if output needs to be printed to stdout
        if self.args.action in self._actions_have_print_output:
            self.print_ = True

        self.process()

    def process(self):
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
                  "print_": self.print_}
        action = getattr(self, self.args.action)
        action(**kwargs)

    @staticmethod
    def extract(input_=None, output=None, fps=None, extract_ext=None,
                **kwargs):
        _input_opts = Effmpeg._common_ffmpeg_args[:]
        _input = {input_.path: _input_opts}
        _output_opts = '-y -vf fps="' + str(fps) + '"'
        _output_path = output.path + "/" + input_.name + "%05d" + extract_ext
        _output = {_output_path: _output_opts}
        ff = FFmpeg(inputs=_input, outputs=_output)
        os.makedirs(output.path, exist_ok=True)
        Effmpeg.__run_ffmpeg(ff)

    @staticmethod
    def gen_vid(input_=None, output=None, fps=None, mux_audio=False,
                ref_vid=None, **kwargs):
        filename = Effmpeg.__get_extracted_filename(input_.path)
        _input_opts = Effmpeg._common_ffmpeg_args[:]
        _input_path = os.path.join(input_.path, filename)
        _output_opts = '-y -c:v libx264 -vf fps="' + str(fps) + '" '
        if mux_audio:
            _ref_vid_opts = '-c copy -map 0:0 -map 1:1'
            _output_opts = _ref_vid_opts + ' ' + _output_opts
            _inputs = {_input_path: _input_opts, ref_vid.path: None}
        else:
            _inputs = {_input_path: _input_opts}
        _outputs = {output.path: _output_opts}
        ff = FFmpeg(inputs=_inputs, outputs=_outputs)
        Effmpeg.__run_ffmpeg(ff)

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
    def rescale(input_=None, output=None, scale=None, **kwargs):
        _input_opts = Effmpeg._common_ffmpeg_args[:]
        _output_opts = '-y -vf scale="' + str(scale) + '"'
        _inputs = {input_.path: _input_opts}
        _outputs = {output.path: _output_opts}
        ff = FFmpeg(inputs=_inputs, outputs=_outputs)
        Effmpeg.__run_ffmpeg(ff)

    @staticmethod
    def rotate(input_=None, output=None, degrees=None, transpose=None,
               **kwargs):
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
        ff = FFmpeg(inputs=_inputs, outputs=_outputs)
        Effmpeg.__run_ffmpeg(ff)

    @staticmethod
    def mux_audio(input_=None, output=None, ref_vid=None, **kwargs):
        _input_opts = Effmpeg._common_ffmpeg_args[:]
        _ref_vid_opts = None
        _output_opts = '-y -c copy -map 0:0 -map 1:1 -shortest'
        _inputs = {input_.path: _input_opts, ref_vid.path: _ref_vid_opts}
        _outputs = {output.path: _output_opts}
        ff = FFmpeg(inputs=_inputs, outputs=_outputs)
        Effmpeg.__run_ffmpeg(ff)

    @staticmethod
    def slice(input_=None, output=None, start=None, duration=None, **kwargs):
        _input_opts = Effmpeg._common_ffmpeg_args[:]
        _input_opts += "-ss " + start
        _output_opts = "-y -t " + duration + " "
        _output_opts += "-vcodec copy -acodec copy"
        _inputs = {input_.path: _input_opts}
        _output = {output.path: _output_opts}
        ff = FFmpeg(inputs=_inputs, outputs=_output)
        Effmpeg.__run_ffmpeg(ff)

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
    def __run_ffmpeg(ff):
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
        for ve in [ve.replace('.', '') for ve in DataItem.vid_ext]:
            if ve in zero_pad:
                vid_ext = ve
                zero_pad = len(zero_pad.replace(ve, ''))
                break

        zero_pad = str(zero_pad).zfill(2)
        filename_list = [name, vid_ext + '%0' + zero_pad + 'd', img_ext]
        return '.'.join(filename_list)

    @staticmethod
    def __parse_transpose(value):
        index = 0
        opts = ["(0, 90CounterClockwise&VerticalFlip)",
                "(1, 90Clockwise)",
                "(2, 90CounterClockwise)",
                "(3, 90Clockwise&VerticalFlip)",
                "None"]
        if len(value) == 1:
            index = int(value)
        else:
            for i in range(5):
                if value in opts[i]:
                    index = i
                    break
        return opts[index]

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
    parser.print_help()
    exit(0)


if __name__ == "__main__":
    print('"Easy"-ffmpeg wrapper.\n')

    parser = argparse.ArgumentParser()
    subparser = parser.add_subparsers()
    sort = Effmpeg(
            subparser, "effmpeg", "Wrapper for various common ffmpeg commands.")

    parser.set_defaults(func=bad_args)
    arguments = parser.parse_args()
    arguments.func(arguments)
