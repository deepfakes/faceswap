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
from lib.cli import FullPaths, DirFullPaths, FileFullPaths, ComboFullPaths

from ffmpy import FFprobe, FFmpeg, FFRuntimeError, FFExecutableNotFoundError

if sys.version_info[0] < 3:
    raise Exception("This program requires at least python3.2")
if sys.version_info[0] == 3 and sys.version_info[1] < 2:
    raise Exception("This program requires at least python3.2")


class DataItem(object):
    """
    A simple class used for storing the media data items and directories that
    Effmpeg uses for 'input', 'output' and 'ref_vid'.
    """
    vid_ext = ["mp4", "mpeg", "webm", "mkv"]
    audio_ext = ["mp3", "wav", "flac"]

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
            item_ext = self.path.split('.')[-1]
            if item_ext in DataItem.vid_ext:
                item_type = 'vid'
            elif item_ext in DataItem.audio_ext:
                item_type = 'audio'
            else:
                item_type = 'dir'
            self.type = item_type if self.type is None else self.type
            self.ext = item_ext if self.ext is None else self.ext
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
    _actions_have_dir_output = ["extract"]
    _actions_have_vid_output = ["gen_vid", "mux_audio", "rescale", "rotate",
                                "slice"]
    _actions_have_dir_input = ["gen_vid"]
    _actions_have_vid_input = ["extract", "get_fps", "get_info", "rescale",
                               "rotate", "slice"]

    def __init__(self, subparser, command, description='default'):
        self.argument_list = self.get_argument_list()
        self.optional_arguments = self.get_optional_arguments()
        self.args = None
        self.input = DataItem()
        self.output = DataItem()
        self.ref_vid = DataItem()
        self.start = ""
        self.end = ""
        self.duration = ""
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
                                          directory.""",
                               "actions_open_type": {
                                   "task_name": "effmpeg",
                                   "extract": "folder",
                                   "gen-vid": "load",
                                   "get-fps": "load",
                                   "get-info": "load",
                                   "mux-audio": "load",
                                   "rescale": "load",
                                   "rotate": "load",
                                   "slice": "load"
                               },
                               "filetypes": {
                                   "extract": None,
                                   "gen-vid": vid_files,
                                   "get-fps": vid_files,
                                   "get-info": vid_files,
                                   "mux-audio": vid_files,
                                   "rescale": vid_files,
                                   "rotate": vid_files,
                                   "slice": vid_files
                               }})

        arguments_list.append({"opts": ('-r', '--reference-video'),
                               "action": FileFullPaths,
                               "filetypes": vid_files,
                               "dest": "ref_vid",
                               "default": "None",
                               "help": """Path to reference video if 'input' 
                                          was not a video."""})

        arguments_list.append({"opts": ('-fps', '--fps'),
                               "type": str,
                               "dest": "fps",
                               "default": "-1.0",
                               "help": """Provide video fps. Can be an integer,
                                          float or fraction. Negative values 
                                          will make the program try to get the 
                                          fps from the input or reference 
                                          videos."""})

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
                               "choices": ("0", "1", "2", "3", "None"),
                               "dest": "transpose",
                               "default": "None",
                               "help": """Transpose the video. If transpose is 
                                          set, then rotate will be ignored.
                                          0 = 90 counter-clockwise and vertical
                                          flip
                                          1 = 90 clockwise
                                          2 = 90 counter clockwise
                                          3 = 90 clockwise and vertical flip"""})

        arguments_list.append({"opts": ('-ro', '--rotate'),
                               "type": str,
                               "dest": "rotate",
                               "default": "None",
                               "help": """Rotate the video clockwise by the 
                                          given number of degrees."""})

        arguments_list.append({"opts": ('-sc', '--scale'),
                               "type": str,
                               "dest": "scale",
                               "default": "1920x1080",
                               "help": """Set the new resolution scale if the
                                          chosen action is 'rescale'."""})

        return arguments_list

    @staticmethod
    def get_optional_arguments():
        """
        Put the arguments in a list so that they are accessible from both
        argparse and gui.
        """
        # Override this for custom arguments
        argument_list = []
        return argument_list

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

        parser = self.add_optional_arguments(parser)
        parser.set_defaults(func=self.process_arguments)

    @staticmethod
    def add_optional_arguments(parser):
        # Override this for custom arguments
        return parser

    def process_arguments(self, arguments):
        self.args = arguments

        self.args.action = self.args.action.replace('-', '_')

        # Instantiate input DataItem object
        self.input = DataItem(path=self.args.input)

        # Instantiate output DataItem object
        if self.args.action in self._actions_have_dir_output:
            self.output = DataItem(path=self.__get_default_vid_output())
        elif self.args.action in self._actions_have_vid_output:
            if self.__check_have_fps(self.args.fps) > 0:
                self.output = DataItem(path=self.__get_default_vid_output(),
                                       fps=self.args.fps)
            else:
                self.output = DataItem(path=self.__get_default_vid_output())

        if self.args.ref_vid == "None" or self.args.ref_vid == '':
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

        # Check that ref_vid is a video
        if self.ref_vid.is_type("none"):
            raise ValueError("The file chosen as the reference video is not a "
                             "video, either leave the field blank or type "
                             "'None': {}".format(self.ref_vid.path))

        # Process start and duration arguments
        self.start = self.parse_time(str(self.args.start))
        self.end = self.parse_time(str(self.args.end))
        if self.end != "00:00:00" or self.end != "":
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

        # Processing rotate
        if self.args.rotate.lower() == "none":
            self.args.rotate = 0

        self.process()

    def process(self):
        kwargs = {"input_": self.input,
                  "output": self.output,
                  "ref_vid": self.ref_vid,
                  "mux_audio": self.args.mux_audio,
                  "start": self.start,
                  "duration": self.duration,
                  "fps": self.args.fps,
                  "rotate": self.args.rotate,
                  "transpose": self.args.transpose,
                  "scale": self.args.scale}
        action = getattr(self, self.args.action)
        action(**kwargs)

    @staticmethod
    def extract(input_=None, output=None, fps=None, **kwargs):
        _input = {input_.path: None}
        _output_opts = '-y -vf fps="' + str(fps) + '"'
        _output_path = output.path + "/" + input_.name + "%05d.png"
        _output = {_output_path: _output_opts}
        ff = FFmpeg(inputs=_input, outputs=_output)
        os.makedirs(output.path, exist_ok=True)
        ff.run(stderr=subprocess.STDOUT)

    @staticmethod
    def gen_vid(input_=None, output=None, fps=None, mux_audio=False,
                ref_vid=None, **kwargs):
        _input_path = os.path.join(input_.path,
                                   os.listdir(input_.path)[0][:-9]
                                   + "%05d.png")
        _output_opts = '-y -c:v libx264 -vf fps="' + str(fps) + '"'
        if mux_audio:
            _ref_vid_opts = '-c copy -map 0:0 -map 1:1 -shortest'
            _output_opts = _ref_vid_opts + ' ' + _output_opts
            _inputs = {_input_path: None, ref_vid.path: None}
        else:
            _inputs = {_input_path: None}
        _outputs = {output.path: _output_opts}
        ff = FFmpeg(inputs=_inputs, outputs=_outputs)
        ff.run(stderr=subprocess.STDOUT)

    @staticmethod
    def get_fps(input_=None, **kwargs):
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
        return _fps

    @staticmethod
    def get_info(input_=None, print_=True, **kwargs):
        _input_opts = ''
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
        _input_opts = None
        _output_opts = '-y -vf scale="' + str(scale) + '"'
        _inputs = {input_.path: _input_opts}
        _outputs = {output.path: _output_opts}
        ff = FFmpeg(inputs=_inputs, outputs=_outputs)
        ff.run(stderr=subprocess.STDOUT)

    @staticmethod
    def rotate(input_=None, output=None, rotate=None, transpose=None,
               **kwargs):
        _input_opts = None
        _output_opts = '-y -c:a copy -vf '
        _bilinear = ''
        if transpose is not None and transpose != "None":
            _output_opts += 'transpose="' + str(transpose) + '"'
        elif int(rotate) != 0:
            if int(rotate) % 90 == 0 and int(rotate) != 0:
                _bilinear = ":bilinear=0"
            _output_opts += 'rotate="' + str(rotate) + '*(PI/180)'
            _output_opts += _bilinear + '" '
        else:
            raise ValueError("You have not supplied a valid rotate or "
                             "transpose value:\nrotate: {}\ntranspose: "
                             "{}".format(rotate, transpose))

        _inputs = {input_.path: _input_opts}
        _outputs = {output.path: _output_opts}
        ff = FFmpeg(inputs=_inputs, outputs=_outputs)
        ff.run(stderr=subprocess.STDOUT)

    @staticmethod
    def mux_audio(input_=None, output=None, ref_vid=None, **kwargs):
        _input_opts = None
        _output_opts = '-y -c copy -map 0:0 -map 1:1 -shortest'
        _inputs = {input_.path: _input_opts, ref_vid: _input_opts}
        _outputs = {output.path: _output_opts}
        ff = FFmpeg(inputs=_inputs, outputs=_outputs)
        ff.run(stderr=subprocess.STDOUT)

    @staticmethod
    def slice(input_=None, output=None, start='', duration=None, **kwargs):
        _input = {input_.path: None}
        _output_opts = "-y -ss " + start + " "
        _output_opts += "-t " + duration + " "
        _output_opts += "-vcodec copy -acodec copy -y"
        _output = {output.path: _output_opts}
        ff = FFmpeg(inputs=_input, outputs=_output)
        ff.run(stderr=subprocess.STDOUT)

    # Various helper methods
    def __get_default_vid_output(self):
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
    def zero_pad(text, num):
        string = "{:0" + str(num) + "d}"
        return string.format(text)

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
