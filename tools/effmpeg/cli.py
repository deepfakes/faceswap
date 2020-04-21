#!/usr/bin/env python3
""" Command Line Arguments for tools """
from lib.cli.args import FaceSwapArgs
from lib.cli.actions import ContextFullPaths, FileFullPaths, Radio
from lib.utils import _image_extensions

_HELPTEXT = "This command allows you to easily execute common ffmpeg tasks."


class EffmpegArgs(FaceSwapArgs):
    """ Class to parse the command line arguments for EFFMPEG tool """

    @staticmethod
    def get_info():
        """ Return command information """
        return "A wrapper for ffmpeg for performing image <> video converting."

    @staticmethod
    def __parse_transpose(value):
        index = 0
        opts = ["(0, 90CounterClockwise&VerticalFlip)",
                "(1, 90Clockwise)",
                "(2, 90CounterClockwise)",
                "(3, 90Clockwise&VerticalFlip)"]
        if len(value) == 1:
            index = int(value)
        else:
            for i in range(5):
                if value in opts[i]:
                    index = i
                    break
        return opts[index]

    def get_argument_list(self):
        argument_list = list()
        argument_list.append({"opts": ('-a', '--action'),
                              "action": Radio,
                              "dest": "action",
                              "choices": ("extract", "gen-vid", "get-fps",
                                          "get-info", "mux-audio", "rescale",
                                          "rotate", "slice"),
                              "default": "extract",
                              "help": "R|Choose which action you want ffmpeg "
                                      "ffmpeg to do."
                                      "\nL|'extract': turns videos into images "
                                      "\nL|'gen-vid': turns images into videos "
                                      "\nL|'get-fps' returns the chosen video's fps."
                                      "\nL|'get-info' returns information about a video."
                                      "\nL|'mux-audio' add audio from one video to another."
                                      "\nL|'rescale' resize video."
                                      "\nL|'rotate' rotate video."
                                      "\nL|'slice' cuts a portion of the video into a separate "
                                      "video file."})
        argument_list.append({"opts": ('-i', '--input'),
                              "action": ContextFullPaths,
                              "dest": "input",
                              "default": "input",
                              "help": "Input file.",
                              "group": "data",
                              "required": True,
                              "action_option": "-a",
                              "filetypes": "video"})
        argument_list.append({"opts": ('-o', '--output'),
                              "action": ContextFullPaths,
                              "group": "data",
                              "default": "",
                              "dest": "output",
                              "help": "Output file. If no output is "
                                      "specified then: if the output is "
                                      "meant to be a video then a video "
                                      "called 'out.mkv' will be created in "
                                      "the input directory; if the output is "
                                      "meant to be a directory then a "
                                      "directory called 'out' will be "
                                      "created inside the input "
                                      "directory."
                                      "Note: the chosen output file "
                                      "extension will determine the file "
                                      "encoding.",
                              "action_option": "-a",
                              "filetypes": "video"})
        argument_list.append({"opts": ('-r', '--reference-video'),
                              "action": FileFullPaths,
                              "dest": "ref_vid",
                              "group": "data",
                              "default": None,
                              "help": "Path to reference video if 'input' "
                                      "was not a video.",
                              "filetypes": "video"})
        argument_list.append({"opts": ('-fps', '--fps'),
                              "type": str,
                              "dest": "fps",
                              "group": "output",
                              "default": "-1.0",
                              "help": "Provide video fps. Can be an integer, "
                                      "float or fraction. Negative values "
                                      "will make the program try to get the "
                                      "fps from the input or reference "
                                      "videos."})
        argument_list.append({"opts": ("-ef", "--extract-filetype"),
                              "action": Radio,
                              "choices": _image_extensions,
                              "dest": "extract_ext",
                              "group": "output",
                              "default": ".png",
                              "help": "Image format that extracted images "
                                      "should be saved as. '.bmp' will offer "
                                      "the fastest extraction speed, but "
                                      "will take the most storage space. "
                                      "'.png' will be slower but will take "
                                      "less storage."})
        argument_list.append({"opts": ('-s', '--start'),
                              "type": str,
                              "dest": "start",
                              "group": "clip",
                              "default": "00:00:00",
                              "help": "Enter the start time from which an "
                                      "action is to be applied. "
                                      "Default: 00:00:00, in HH:MM:SS "
                                      "format. You can also enter the time "
                                      "with or without the colons, e.g. "
                                      "00:0000 or 026010."})
        argument_list.append({"opts": ('-e', '--end'),
                              "type": str,
                              "dest": "end",
                              "group": "clip",
                              "default": "00:00:00",
                              "help": "Enter the end time to which an action "
                                      "is to be applied. If both an end time "
                                      "and duration are set, then the end "
                                      "time will be used and the duration "
                                      "will be ignored. "
                                      "Default: 00:00:00, in HH:MM:SS."})
        argument_list.append({"opts": ('-d', '--duration'),
                              "type": str,
                              "dest": "duration",
                              "group": "clip",
                              "default": "00:00:00",
                              "help": "Enter the duration of the chosen "
                                      "action, for example if you enter "
                                      "00:00:10 for slice, then the first 10 "
                                      "seconds after and including the start "
                                      "time will be cut out into a new "
                                      "video. "
                                      "Default: 00:00:00, in HH:MM:SS "
                                      "format. You can also enter the time "
                                      "with or without the colons, e.g. "
                                      "00:0000 or 026010."})
        argument_list.append({"opts": ('-m', '--mux-audio'),
                              "action": "store_true",
                              "dest": "mux_audio",
                              "group": "output",
                              "default": False,
                              "help": "Mux the audio from the reference "
                                      "video into the input video. This "
                                      "option is only used for the 'gen-vid' "
                                      "action. 'mux-audio' action has this "
                                      "turned on implicitly."})
        argument_list.append(
            {"opts": ('-tr', '--transpose'),
             "choices": ("(0, 90CounterClockwise&VerticalFlip)",
                         "(1, 90Clockwise)",
                         "(2, 90CounterClockwise)",
                         "(3, 90Clockwise&VerticalFlip)"),
             "type": lambda v: self.__parse_transpose(v),
             "dest": "transpose",
             "group": "rotate",
             "default": None,
             "help": "Transpose the video. If transpose is "
                     "set, then degrees will be ignored. For "
                     "cli you can enter either the number "
                     "or the long command name, "
                     "e.g. to use (1, 90Clockwise) "
                     "-tr 1 or -tr 90Clockwise"})
        argument_list.append({"opts": ('-de', '--degrees'),
                              "type": str,
                              "dest": "degrees",
                              "default": None,
                              "group": "rotate",
                              "help": "Rotate the video clockwise by the "
                                      "given number of degrees."})
        argument_list.append({"opts": ('-sc', '--scale'),
                              "type": str,
                              "dest": "scale",
                              "group": "output",
                              "default": "1920x1080",
                              "help": "Set the new resolution scale if the "
                                      "chosen action is 'rescale'."})
        argument_list.append({"opts": ('-q', '--quiet'),
                              "action": "store_true",
                              "dest": "quiet",
                              "group": "settings",
                              "default": False,
                              "help": "Reduces output verbosity so that only "
                                      "serious errors are printed. If both "
                                      "quiet and verbose are set, verbose "
                                      "will override quiet."})
        argument_list.append({"opts": ('-v', '--verbose'),
                              "action": "store_true",
                              "dest": "verbose",
                              "group": "settings",
                              "default": False,
                              "help": "Increases output verbosity. If both "
                                      "quiet and verbose are set, verbose "
                                      "will override quiet."})
        return argument_list
