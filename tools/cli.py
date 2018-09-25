#!/usr/bin/env python3
""" Command Line Arguments for tools """
from lib.cli import FaceSwapArgs
from lib.cli import (ContextFullPaths, DirFullPaths,
                     FileFullPaths, SaveFileFullPaths)
from lib.utils import _image_extensions


class AlignmentsArgs(FaceSwapArgs):
    """ Class to parse the command line arguments for Aligments tool """

    def get_argument_list(self):
        frames_dir = "\n\tMust Pass in a frames folder (-fr)."
        faces_dir = "\n\tMust Pass in a faces folder (-fc)."
        frames_or_faces_dir = ("\n\tMust Pass in either a frames folder"
                               "\n\tOR a faces folder (-fr or -fc).")
        frames_and_faces_dir = ("\n\tMust Pass in a frames folder AND a faces"
                                "\n\tfolder (-fr and -fc).")
        output_opts = "\n\tUse the output option (-o) to process\n\tresults."
        align_eyes = "\n\tCan optionally use the align-eyes switch (-ae)."
        argument_list = list()
        argument_list.append({
            "opts": ("-j", "--job"),
            "type": str,
            "choices": ("draw", "extract", "extract-large", "manual",
                        "missing-alignments", "missing-frames",
                        "leftover-faces", "multi-faces", "no-faces",
                        "reformat", "remove-faces", "remove-frames",
                        "rotate", "sort-x", "sort-y"),
            "required": True,
            "help": "R|Choose which action you want to perform.\n"
                    "NB: All actions require an alignments file (-a) to"
                    "\nbe passed in."
                    "\n'draw': Draw landmarks on frames in the selected"
                    "\n\tfolder. A subfolder will be created within"
                    "\n\tthe frames folder to hold the output." +
                    frames_dir + align_eyes +
                    "\n'extract': Re-extract faces from the source frames"
                    "\n\tbased on alignment data. This is a"
                    "\n\tlot quicker than re-detecting faces." +
                    frames_and_faces_dir + align_eyes +
                    "\n'extract-large' - Extract all faces that have not been"
                    "\n\tupscaled. Useful for excluding low-res images from a"
                    "\n\ttraining set" + frames_and_faces_dir + align_eyes +
                    "\n'manual': Manually view and edit landmarks." +
                    frames_dir + align_eyes +
                    "\n'missing-alignments': Identify frames that do not"
                    "\n\texist in the alignments file." + output_opts +
                    frames_dir +
                    "\n'missing-frames': Identify frames in the alignments"
                    "\n\tfile that do not appear within the frames"
                    "\n\tfolder." + output_opts + frames_dir +
                    "\n'leftover-faces': Identify faces in the faces"
                    "\n\tfolder that do not exist in the alignments file."
                    + output_opts + faces_dir +
                    "\n'multi-faces': Identify where multiple faces exist"
                    "\n\twithin the alignments file." + output_opts +
                    frames_or_faces_dir +
                    "\n'no-faces': Identify frames that exist within the"
                    "\n\talignment file but no faces were detected." +
                    output_opts + frames_dir +
                    "\n'reformat': Save a copy of alignments file in a"
                    "\n\tdifferent format. Specify a format with"
                    "\n\tthe -fmt option."
                    "\n\tAlignments can be converted from"
                    "\n\tDeepFaceLab by specifing:"
                    "\n\t    -a dfl"
                    "\n\t    -fc <source faces folder>"
                    "\n'remove-faces': Remove deleted faces from an"
                    "\n\talignments file. The original alignments file"
                    "\n\t will be backed up. A different file format or"
                    "\n\tthe alignments file can optionally be specified"
                    "\n\t(-fmt)." + faces_dir +
                    "\n'remove-frames': Remove deleted frames from an"
                    "\n\talignments file. The original alignments file"
                    "\n\twill be backed up. A different file format for"
                    "\n\tthe alignments file can optionally be specified"
                    "\n\t(-fmt)." + frames_dir +
                    "\n'rotate' - Rotate landmarks and bounding boxes. Legacy"
                    "\n\talignments files hold an 'r' parameter indicating"
                    "\n\tthat the image needs to be rotated for convert. This"
                    "\n\tmeans that the stored landarks and bounding box do"
                    "\n\tnot correspond to the actual frame. This command"
                    "\n\tupdates the alignments file with the correct"
                    "\n\tlandmarks and bounding boxes and removes the 'r'"
                    "\n\tparameter." + frames_dir +
                    "\n'sort-x' - Re-index the alignments from left to"
                    "\n\tright. For alignments with multiple faces this will"
                    "\n\tensure that the left-most face is at index 0"
                    "\n\tOptionally pass in a faces folder (-fc) to also"
                    "\n\trename extracted faces."
                    "\n'sort-y' - Re-index the alignments from top to"
                    "\n\tbottom. For alignments with multiple faces this will"
                    "\n\tensure that the top-most face is at index 0"
                    "\n\tOptionally pass in a faces folder (-fc) to also"
                    "\n\trename extracted faces."})
        argument_list.append({"opts": ("-a", "--alignments_file"),
                              "action": FileFullPaths,
                              "dest": "alignments_file",
                              "required": True,
                              "filetypes": "alignments",
                              "help": "Full path to the alignments "
                                      "file to be processed."})
        argument_list.append({"opts": ("-fc", "-faces_folder"),
                              "action": DirFullPaths,
                              "dest": "faces_dir",
                              "help": "Directory containing extracted faces."})
        argument_list.append({"opts": ("-fr", "-frames_folder"),
                              "action": DirFullPaths,
                              "dest": "frames_dir",
                              "help": "Directory containing source frames "
                                      "that faces were extracted from."})
        argument_list.append({"opts": ("-fmt", "--alignment_format"),
                              "type": str,
                              "choices": ("json", "pickle", "yaml"),
                              "help": "The file format to save the alignment "
                                      "data in. Defaults to same as source."})
        argument_list.append({
            "opts": ("-o", "--output"),
            "type": str,
            "choices": ("console", "file", "move"),
            "default": "console",
            "help": "R|How to output discovered items ('faces' and"
                    "\n'frames' only):"
                    "\n'console': Print the list of frames to the screen."
                    "\n\t(DEFAULT)"
                    "\n'file': Output the list of frames to a text file"
                    "\n\t(stored within the source directory)."
                    "\n'move': Move the discovered items to a sub-folder"
                    "\n\twithin the source directory."})
        argument_list.append({"opts": ("-ae", "--align-eyes"),
                              "action": "store_true",
                              "dest": "align_eyes",
                              "default": False,
                              "help": "Perform extra alignment to ensure "
                                      "left/right eyes are  at the same "
                                      "height. (Draw, Extract and manual "
                                      "only)"})
        argument_list.append({"opts": ("-v", "--verbose"),
                              "action": "store_true",
                              "dest": "verbose",
                              "default": False,
                              "help": "Show verbose output"})
        return argument_list


class EffmpegArgs(FaceSwapArgs):
    """ Class to parse the command line arguments for EFFMPEG tool """

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
                              "dest": "action",
                              "choices": ("extract", "gen-vid", "get-fps",
                                          "get-info", "mux-audio", "rescale",
                                          "rotate", "slice"),
                              "default": "extract",
                              "help": "Choose which action you want ffmpeg "
                                      "ffmpeg to do.\n"
                                      "'slice' cuts a portion of the video "
                                      "into a separate video file.\n"
                                      "'get-fps' returns the chosen video's "
                                      "fps."})

        argument_list.append({"opts": ('-i', '--input'),
                              "action": ContextFullPaths,
                              "dest": "input",
                              "default": "input",
                              "help": "Input file.",
                              "required": True,
                              "action_option": "-a",
                              "filetypes": "video"})

        argument_list.append({"opts": ('-o', '--output'),
                              "action": ContextFullPaths,
                              "dest": "output",
                              "default": "",
                              "help": "Output file. If no output is "
                                      "specified then: if the output is "
                                      "meant to be a video then a video "
                                      "called 'out.mkv' will be created in "
                                      "the input directory; if the output is "
                                      "meant to be a directory then a "
                                      "directory called 'out' will be "
                                      "created inside the input "
                                      "directory.\n"
                                      "Note: the chosen output file "
                                      "extension will determine the file "
                                      "encoding.",
                              "action_option": "-a",
                              "filetypes": "video"})

        argument_list.append({"opts": ('-r', '--reference-video'),
                              "action": FileFullPaths,
                              "dest": "ref_vid",
                              "default": None,
                              "help": "Path to reference video if 'input' "
                                      "was not a video.",
                              "filetypes": "video"})

        argument_list.append({"opts": ('-fps', '--fps'),
                              "type": str,
                              "dest": "fps",
                              "default": "-1.0",
                              "help": "Provide video fps. Can be an integer, "
                                      "float or fraction. Negative values "
                                      "will make the program try to get the "
                                      "fps from the input or reference "
                                      "videos."})

        argument_list.append({"opts": ("-ef", "--extract-filetype"),
                              "choices": _image_extensions,
                              "dest": "extract_ext",
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
                              "default": "00:00:00",
                              "help": "Enter the start time from which an "
                                      "action is to be applied.\n"
                                      "Default: 00:00:00, in HH:MM:SS "
                                      "format. You can also enter the time "
                                      "with or without the colons, e.g. "
                                      "00:0000 or 026010."})

        argument_list.append({"opts": ('-e', '--end'),
                              "type": str,
                              "dest": "end",
                              "default": "00:00:00",
                              "help": "Enter the end time to which an action "
                                      "is to be applied. If both an end time "
                                      "and duration are set, then the end "
                                      "time will be used and the duration "
                                      "will be ignored.\n"
                                      "Default: 00:00:00, in HH:MM:SS."})

        argument_list.append({"opts": ('-d', '--duration'),
                              "type": str,
                              "dest": "duration",
                              "default": "00:00:00",
                              "help": "Enter the duration of the chosen "
                                      "action, for example if you enter "
                                      "00:00:10 for slice, then the first 10 "
                                      "seconds after and including the start "
                                      "time will be cut out into a new "
                                      "video.\n"
                                      "Default: 00:00:00, in HH:MM:SS "
                                      "format. You can also enter the time "
                                      "with or without the colons, e.g. "
                                      "00:0000 or 026010."})

        argument_list.append({"opts": ('-m', '--mux-audio'),
                              "action": "store_true",
                              "dest": "mux_audio",
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
                              "help": "Rotate the video clockwise by the "
                                      "given number of degrees."})

        argument_list.append({"opts": ('-sc', '--scale'),
                              "type": str,
                              "dest": "scale",
                              "default": "1920x1080",
                              "help": "Set the new resolution scale if the "
                                      "chosen action is 'rescale'."})

        argument_list.append({"opts": ('-pr', '--preview'),
                              "action": "store_true",
                              "dest": "preview",
                              "default": False,
                              "help": "Uses ffplay to preview the effects of "
                                      "actions that have a video output. "
                                      "Currently preview does not work when "
                                      "muxing audio."})

        argument_list.append({"opts": ('-q', '--quiet'),
                              "action": "store_true",
                              "dest": "quiet",
                              "default": False,
                              "help": "Reduces output verbosity so that only "
                                      "serious errors are printed. If both "
                                      "quiet and verbose are set, verbose "
                                      "will override quiet."})

        argument_list.append({"opts": ('-v', '--verbose'),
                              "action": "store_true",
                              "dest": "verbose",
                              "default": False,
                              "help": "Increases output verbosity. If both "
                                      "quiet and verbose are set, verbose "
                                      "will override quiet."})

        return argument_list


class SortArgs(FaceSwapArgs):
    """ Class to parse the command line arguments for sort tool """

    @staticmethod
    def get_argument_list():
        """ Put the arguments in a list so that they are accessible from both
        argparse and gui """
        argument_list = list()
        argument_list.append({"opts": ('-i', '--input'),
                              "action": DirFullPaths,
                              "dest": "input_dir",
                              "default": "input_dir",
                              "help": "Input directory of aligned faces.",
                              "required": True})

        argument_list.append({"opts": ('-o', '--output'),
                              "action": DirFullPaths,
                              "dest": "output_dir",
                              "default": "_output_dir",
                              "help": "Output directory for sorted aligned "
                                      "faces."})

        argument_list.append({"opts": ('-fp', '--final-process'),
                              "type": str,
                              "choices": ("folders", "rename"),
                              "dest": 'final_process',
                              "default": "rename",
                              "help": "R|\n'folders': files are sorted using "
                                      "the -s/--sort-by\n\tmethod, then they "
                                      "are organized into\n\tfolders using "
                                      "the -g/--group-by grouping\n\tmethod."
                                      "\n'rename': files are sorted using "
                                      "the -s/--sort-by\n\tthen they are "
                                      "renamed.\nDefault: rename"})

        argument_list.append({"opts": ('-k', '--keep'),
                              "action": 'store_true',
                              "dest": 'keep_original',
                              "default": False,
                              "help": "Keeps the original files in the input "
                                      "directory. Be careful when using this "
                                      "with rename grouping and no specified "
                                      "output directory as this would keep "
                                      "the original and renamed files in the "
                                      "same directory."})

        argument_list.append({"opts": ('-s', '--sort-by'),
                              "type": str,
                              "choices": ("blur", "face", "face-cnn",
                                          "face-cnn-dissim", "face-dissim",
                                          "face-yaw", "hist",
                                          "hist-dissim"),
                              "dest": 'sort_method',
                              "default": "hist",
                              "help": "Sort by method. "
                                      "Choose how images are sorted. "
                                      "Default: hist"})

        argument_list.append({"opts": ('-g', '--group-by'),
                              "type": str,
                              "choices": ("blur", "face", "face-cnn",
                                          "face-yaw", "hist"),
                              "dest": 'group_method',
                              "default": "hist",
                              "help": "Group by method. "
                                      "When -fp/--final-processing by "
                                      "folders choose the how the images are "
                                      "grouped after sorting. "
                                      "Default: hist"})

        argument_list.append({"opts": ('-t', '--ref_threshold'),
                              "type": float,
                              "dest": 'min_threshold',
                              "default": -1.0,
                              "help": "Float value. "
                                      "Minimum threshold to use for grouping "
                                      "comparison with 'face' and 'hist' "
                                      "methods. The lower the value the more "
                                      "discriminating the grouping is. "
                                      "Leaving -1.0 will make the program "
                                      "set the default value automatically. "
                                      "For face 0.6 should be enough, with "
                                      "0.5 being very discriminating. "
                                      "For face-cnn 7.2 should be enough, "
                                      "with 4 being very discriminating. "
                                      "For hist 0.3 should be enough, with "
                                      "0.2 being very discriminating. "
                                      "Be careful setting a value that's too "
                                      "low in a directory with many images, "
                                      "as this could result in a lot of "
                                      "directories being created. "
                                      "Defaults: face 0.6, face-cnn 7.2, "
                                      "hist 0.3"})

        argument_list.append({"opts": ('-b', '--bins'),
                              "type": int,
                              "dest": 'num_bins',
                              "default": 5,
                              "help": "Integer value. "
                                      "Number of folders that will be used "
                                      "to group by blur and face-yaw. "
                                      "For blur folder 0 will be the least "
                                      "blurry, while the last folder will be "
                                      "the blurriest. "
                                      "For face-yaw the number of bins is by "
                                      "how much 180 degrees is divided. So "
                                      "if you use 18, then each folder will "
                                      "be a 10 degree increment. Folder 0 "
                                      "will contain faces looking the most "
                                      "to the left whereas the last folder "
                                      "will contain the faces looking the "
                                      "most to the right. "
                                      "If the number of images doesn't "
                                      "divide evenly into the number of "
                                      "bins, the remaining images get put in "
                                      "the last bin."
                                      "Default value: 5"})

        argument_list.append({"opts": ('-l', '--log-changes'),
                              "action": 'store_true',
                              "dest": 'log_changes',
                              "default": False,
                              "help": "Logs file renaming changes if "
                                      "grouping by renaming, or it logs the "
                                      "file copying/movement if grouping by "
                                      "folders. If no log file is specified "
                                      "with '--log-file', then a "
                                      "'sort_log.json' file will be created "
                                      "in the input directory."})

        argument_list.append({"opts": ('-lf', '--log-file'),
                              "action": SaveFileFullPaths,
                              "filetypes": "alignments",
                              "dest": 'log_file_path',
                              "default": 'sort_log.json',
                              "help": "Specify a log file to use for saving "
                                      "the renaming or grouping information. "
                                      "If specified extension isn't 'json' "
                                      "or 'yaml', then json will be used as "
                                      "the serializer, with the supplied "
                                      "filename. "
                                      "Default: sort_log.json"})

        return argument_list
