#!/usr/bin/env python3
""" Command Line Arguments for tools """
from argparse import SUPPRESS

from lib.cli import FaceSwapArgs
from lib.cli import (ContextFullPaths, DirOrFileFullPaths, DirFullPaths, FileFullPaths,
                     FilesFullPaths, SaveFileFullPaths, Radio, Slider)
from lib.utils import _image_extensions
from plugins.plugin_loader import PluginLoader


class AlignmentsArgs(FaceSwapArgs):
    """ Class to parse the command line arguments for Alignments tool """

    @staticmethod
    def get_info():
        """ Return command information """
        return ("Alignments tool\nThis tool allows you to perform numerous actions on or using an "
                "alignments file against its corresponding faceset/frame source.")

    def get_argument_list(self):
        frames_dir = " Must Pass in a frames folder/source video file (-fr)."
        faces_dir = " Must Pass in a faces folder (-fc)."
        frames_or_faces_dir = (" Must Pass in either a frames folder/source video file OR a"
                               "faces folder (-fr or -fc).")
        frames_and_faces_dir = (" Must Pass in a frames folder/source video file AND a faces "
                                "folder (-fr and -fc).")
        output_opts = " Use the output option (-o) to process results."
        align_eyes = " Can optionally use the align-eyes switch (-ae)."
        argument_list = list()
        argument_list.append({
            "opts": ("-j", "--job"),
            "action": Radio,
            "type": str,
            "choices": ("dfl", "draw", "extract", "fix", "manual", "merge", "missing-alignments",
                        "missing-frames", "leftover-faces", "multi-faces", "no-faces",
                        "remove-faces", "remove-frames", "rename", "sort", "spatial",
                        "update-hashes"),
            "required": True,
            "help": "R|Choose which action you want to perform. "
                    "NB: All actions require an alignments file (-a) to be passed in."
                    "\nL|'dfl': Create an alignments file from faces extracted from DeepFaceLab. "
                    "Specify 'dfl' as the 'alignments file' entry and the folder containing the "
                    "dfl faces as the 'faces folder' ('-a dfl -fc <source faces folder>'"
                    "\nL|'draw': Draw landmarks on frames in the selected folder/video. A "
                    "subfolder will be created within the frames folder to hold the output." +
                    frames_dir +
                    "\nL|'extract': Re-extract faces from the source frames/video based on "
                    "alignment data. This is a lot quicker than re-detecting faces. Can pass in "
                    "the '-een' (--extract-every-n) parameter to only extract every nth frame." +
                    frames_and_faces_dir + align_eyes +
                    # TODO - Remove the fix job after a period of time. Implemented 2019/12/07
                    "\nL|'fix': There was a bug when extracting from video which would shift all "
                    "the faces out by 1 frame. This was a shortlived bug, but this job will fix "
                    "alignments files that have this issue. NB: Only run this on alignments files "
                    "that you know need fixing."
                    "\nL|'manual': Manually view and edit landmarks." + frames_dir +
                    "\nL|'merge': Merge multiple alignment files into one. Specify a space "
                    "separated list of alignments files with the -a flag. Optionally specify a "
                    "faces (-fc) folder to filter the final alignments file to only those faces "
                    "that appear within the provided folder."
                    "\nL|'missing-alignments': Identify frames that do not exist in the "
                    "alignments file." + output_opts + frames_dir +
                    "\nL|'missing-frames': Identify frames in the alignments file that do not "
                    "appear within the frames folder/video." + output_opts + frames_dir +
                    "\nL|'leftover-faces': Identify faces in the faces folder that do not exist "
                    "in the alignments file." + output_opts + faces_dir +
                    "\nL|'multi-faces': Identify where multiple faces exist within the alignments "
                    "file." + output_opts + frames_or_faces_dir +
                    "\nL|'no-faces': Identify frames that exist within the alignment file but no "
                    "faces were detected." + output_opts + frames_dir +
                    "\nL|'remove-faces': Remove deleted faces from an alignments file. The "
                    "original alignments file will be backed up." + faces_dir +
                    "\nL|'remove-frames': Remove deleted frames from an alignments file. The "
                    "original alignments file will be backed up." + frames_dir +
                    "\nL|'rename' - Rename faces to correspond with their parent frame and "
                    "position index in the alignments file (i.e. how they are named after running "
                    "extract)." + faces_dir +
                    "\nL|'sort': Re-index the alignments from left to right. For alignments "
                    "with multiple faces this will ensure that the left-most face is at index 0 "
                    "Optionally pass in a faces folder (-fc) to also rename extracted faces."
                    "\nL|'spatial': Perform spatial and temporal filtering to smooth alignments "
                    "(EXPERIMENTAL!)"
                    "\nL|'update-hashes': Recalculate the face hashes. Only use this if you have "
                    "altered the extracted faces (e.g. colour adjust). The files MUST be "
                    "named '<frame_name>_face index' (i.e. how they are named after running "
                    "extract)." + faces_dir})
        argument_list.append({"opts": ("-a", "--alignments_file"),
                              "action": FilesFullPaths,
                              "dest": "alignments_file",
                              "nargs": "+",
                              "group": "data",
                              "required": True,
                              "filetypes": "alignments",
                              "help": "Full path to the alignments file to be processed. If "
                                      "merging alignments, then multiple files can be selected, "
                                      "space separated"})
        argument_list.append({"opts": ("-fc", "-faces_folder"),
                              "action": DirFullPaths,
                              "dest": "faces_dir",
                              "group": "data",
                              "help": "Directory containing extracted faces."})
        argument_list.append({"opts": ("-fr", "-frames_folder"),
                              "action": DirOrFileFullPaths,
                              "dest": "frames_dir",
                              "filetypes": "video",
                              "group": "data",
                              "help": "Directory containing source frames "
                                      "that faces were extracted from."})
        argument_list.append({
            "opts": ("-o", "--output"),
            "action": Radio,
            "type": str,
            "choices": ("console", "file", "move"),
            "group": "processing",
            "default": "console",
            "help": "R|How to output discovered items ('faces' and 'frames' only):"
                    "\nL|'console': Print the list of frames to the screen. (DEFAULT)"
                    "\nL|'file': Output the list of frames to a text file (stored within the "
                    " source directory)."
                    "\nL|'move': Move the discovered items to a sub-folder within the source "
                    "directory."})
        argument_list.append({"opts": ("-een", "--extract-every-n"),
                              "type": int,
                              "action": Slider,
                              "dest": "extract_every_n",
                              "min_max": (1, 100),
                              "default": 1,
                              "rounding": 1,
                              "group": "extract",
                              "help": "[Extract only] Extract every 'nth' frame. This option will "
                                      "skip frames when extracting faces. For example a value of "
                                      "1 will extract faces from every frame, a value of 10 will "
                                      "extract faces from every 10th frame."})
        argument_list.append({"opts": ("-sz", "--size"),
                              "type": int,
                              "action": Slider,
                              "min_max": (128, 512),
                              "default": 256,
                              "group": "extract",
                              "rounding": 64,
                              "help": "[Extract only] The output size of extracted faces."})
        argument_list.append({"opts": ("-ae", "--align-eyes"),
                              "action": "store_true",
                              "dest": "align_eyes",
                              "group": "extract",
                              "default": False,
                              "help": "[Extract only] Perform extra alignment to ensure "
                                      "left/right eyes are at the same height."})
        argument_list.append({"opts": ("-l", "--large"),
                              "action": "store_true",
                              "group": "extract",
                              "default": False,
                              "help": "[Extract only] Only extract faces that have not been "
                                      "upscaled to the required size (`-sz`, `--size). Useful "
                                      "for excluding low-res images from a training set."})
        argument_list.append({"opts": ("-dm", "--disable-monitor"),
                              "action": "store_true",
                              "group": "manual tool",
                              "dest": "disable_monitor",
                              "default": False,
                              "help": "Enable this option if manual "
                                      "alignments window is closing "
                                      "instantly. (Manual only)"})
        return argument_list


class PreviewArgs(FaceSwapArgs):
    """ Class to parse the command line arguments for Preview (Convert Settings) tool """

    @staticmethod
    def get_info():
        """ Return command information """
        return "Preview tool\nAllows you to configure your convert settings with a live preview"

    def get_argument_list(self):

        argument_list = list()
        argument_list.append({"opts": ("-i", "--input-dir"),
                              "action": DirOrFileFullPaths,
                              "filetypes": "video",
                              "dest": "input_dir",
                              "group": "data",
                              "required": True,
                              "help": "Input directory or video. Either a directory containing "
                                      "the image files you wish to process or path to a video "
                                      "file."})
        argument_list.append({"opts": ("-al", "--alignments"),
                              "action": FileFullPaths,
                              "filetypes": "alignments",
                              "type": str,
                              "group": "data",
                              "dest": "alignments_path",
                              "help": "Path to the alignments file for the input, if not at the "
                                      "default location"})
        argument_list.append({"opts": ("-m", "--model-dir"),
                              "action": DirFullPaths,
                              "dest": "model_dir",
                              "group": "data",
                              "required": True,
                              "help": "Model directory. A directory containing the trained model "
                                      "you wish to process."})
        argument_list.append({"opts": ("-s", "--swap-model"),
                              "action": "store_true",
                              "dest": "swap_model",
                              "default": False,
                              "help": "Swap the model. Instead of A -> B, "
                                      "swap B -> A"})
        argument_list.append({"opts": ("-ag", "--allow-growth"),
                              "action": "store_true",
                              "dest": "allow_growth",
                              "default": False,
                              "backend": "nvidia",
                              "help": "Sets allow_growth option of Tensorflow to spare memory "
                                      "on some configurations."})

        return argument_list


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

        argument_list.append({"opts": ('-pr', '--preview'),
                              "action": "store_true",
                              "dest": "preview",
                              "default": False,
                              # TODO Fix preview or remove
                              "help": SUPPRESS,
                              # "help": "Uses ffplay to preview the effects of "
                              #         "actions that have a video output. "
                              #         "Currently preview does not work when "
                              #         "muxing audio."
                              })

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


class MaskArgs(FaceSwapArgs):
    """ Class to parse the command line arguments for Mask tool """

    @staticmethod
    def get_info():
        """ Return command information """
        return "Mask tool\nGenerate masks for existing alignments files."

    def get_argument_list(self):
        argument_list = list()
        argument_list.append({
            "opts": ("-a", "--alignments"),
            "action": FileFullPaths,
            "type": str,
            "group": "data",
            "required": True,
            "filetypes": "alignments",
            "help": "Full path to the alignments file to add the mask to. NB: if the mask already "
                    "exists in the alignments file it will be overwritten."})
        argument_list.append({
            "opts": ("-i", "--input"),
            "action": DirOrFileFullPaths,
            "type": str,
            "group": "data",
            "filetypes": "video",
            "required": True,
            "help": "Directory containing extracted faces, source frames, or a video file."})
        argument_list.append({
            "opts": ("-it", "--input-type"),
            "action": Radio,
            "type": str.lower,
            "choices": ("faces", "frames"),
            "dest": "input_type",
            "group": "data",
            "default": "frames",
            "help": "R|Whether the `input` is a folder of faces or a folder frames/video"
                    "\nL|faces: The input is a folder containing extracted faces."
                    "\nL|frames: The input is a folder containing frames or is a video"})
        argument_list.append({
            "opts": ("-M", "--masker"),
            "action": Radio,
            "type": str.lower,
            "choices": PluginLoader.get_available_extractors("mask"),
            "default": "extended",
            "group": "process",
            "help": "R|Masker to use."
                    "\nL|components: Mask designed to provide facial segmentation based on the "
                    "positioning of landmark locations. A convex hull is constructed around the "
                    "exterior of the landmarks to create a mask."
                    "\nL|extended: Mask designed to provide facial segmentation based on the "
                    "positioning of landmark locations. A convex hull is constructed around the "
                    "exterior of the landmarks and the mask is extended upwards onto the forehead."
                    "\nL|vgg-clear: Mask designed to provide smart segmentation of mostly frontal "
                    "faces clear of obstructions. Profile faces and obstructions may result in "
                    "sub-par performance."
                    "\nL|vgg-obstructed: Mask designed to provide smart segmentation of mostly "
                    "frontal faces. The mask model has been specifically trained to recognize "
                    "some facial obstructions (hands and eyeglasses). Profile faces may result in "
                    "sub-par performance."
                    "\nL|unet-dfl: Mask designed to provide smart segmentation of mostly frontal "
                    "faces. The mask model has been trained by community members and will need "
                    "testing for further description. Profile faces may result in sub-par "
                    "performance."})
        argument_list.append({
            "opts": ("-p", "--processing"),
            "action": Radio,
            "type": str.lower,
            "choices": ("all", "missing", "output"),
            "default": "missing",
            "group": "process",
            "help": "R|Whether to update all masks in the alignments files, only those faces "
                    "that do not already have a mask of the given `mask type` or just to output "
                    "the masks to the `output` location."
                    "\nL|all: Update the mask for all faces in the alignments file."
                    "\nL|missing: Create a mask for all faces in the alignments file where a mask "
                    "does not previously exist."
                    "\nL|output: Don't update the masks, just output them for review in the given "
                    "output folder."})
        argument_list.append({
            "opts": ("-o", "--output-folder"),
            "action": DirFullPaths,
            "dest": "output",
            "type": str,
            "group": "output",
            "help": "Optional output location. If provided, a preview of the masks created will "
                    "be output in the given folder."})
        argument_list.append({
            "opts": ("-b", "--blur_kernel"),
            "action": Slider,
            "type": int,
            "group": "output",
            "min_max": (0, 9),
            "default": 3,
            "rounding": 1,
            "help": "Apply gaussian blur to the mask output. Has the effect of smoothing the "
                    "edges of the mask giving less of a hard edge. the size is in pixels. This "
                    "value should be odd, if an even number is passed in then it will be rounded "
                    "to the next odd number. NB: Only effects the output preview. Set to 0 for "
                    "off"})
        argument_list.append({
            "opts": ("-t", "--threshold"),
            "action": Slider,
            "type": int,
            "group": "output",
            "min_max": (0, 50),
            "default": 4,
            "rounding": 1,
            "help": "Helps reduce 'blotchiness' on some masks by making light shades white "
                    "and dark shades black. Higher values will impact more of the mask. NB: "
                    "Only effects the output preview. Set to 0 for off"})
        argument_list.append({
            "opts": ("-ot", "--output-type"),
            "action": Radio,
            "type": str.lower,
            "choices": ("combined", "masked", "mask"),
            "default": "combined",
            "group": "output",
            "help": "R|How to format the output when processing is set to 'output'."
                    "\nL|combined: The image contains the face/frame, face mask and masked face."
                    "\nL|masked: Output the face/frame as rgba image with the face masked."
                    "\nL|mask: Only output the mask as a single channel image."})
        argument_list.append({
            "opts": ("-f", "--full-frame"),
            "action": "store_true",
            "default": False,
            "group": "output",
            "help": "R|Whether to output the whole frame or only the face box when using "
                    "output processing. Only has an effect when using frames as input."})

        return argument_list


class RestoreArgs(FaceSwapArgs):
    """ Class to restore model files from backup """

    @staticmethod
    def get_info():
        """ Return command information """
        return "A tool for restoring models from backup (.bk) files"

    @staticmethod
    def get_argument_list():
        """ Put the arguments in a list so that they are accessible from both argparse and gui """
        argument_list = list()
        argument_list.append({"opts": ("-m", "--model-dir"),
                              "action": DirFullPaths,
                              "dest": "model_dir",
                              "required": True,
                              "help": "Model directory. A directory containing the model "
                                      "you wish to restore from backup."})
        return argument_list


class SortArgs(FaceSwapArgs):
    """ Class to parse the command line arguments for sort tool """

    @staticmethod
    def get_info():
        """ Return command information """
        return "Sort faces using a number of different techniques"

    @staticmethod
    def get_argument_list():
        """ Put the arguments in a list so that they are accessible from both argparse and gui """
        argument_list = list()
        argument_list.append({"opts": ('-i', '--input'),
                              "action": DirFullPaths,
                              "dest": "input_dir",
                              "group": "data",
                              "help": "Input directory of aligned faces.",
                              "required": True})

        argument_list.append({"opts": ('-o', '--output'),
                              "action": DirFullPaths,
                              "dest": "output_dir",
                              "group": "data",
                              "help": "Output directory for sorted aligned "
                                      "faces."})

        argument_list.append({"opts": ('-s', '--sort-by'),
                              "action": Radio,
                              "type": str,
                              "choices": ("blur", "face", "face-cnn", "face-cnn-dissim",
                                          "face-yaw", "hist", "hist-dissim", "color-gray",
                                          "color-luma", "color-green", "color-orange"),
                              "dest": 'sort_method',
                              "group": "sort settings",
                              "default": "face",
                              "help": "R|Sort by method. Choose how images are sorted. "
                                      "\nL|'blur': Sort faces by blurriness."
                                      "\nL|'face': Use VGG Face to sort by face similarity. This "
                                      "uses a pairwise clustering algorithm to check the "
                                      "distances between 4096 features on every face in your set "
                                      "and order them appropriately. WARNING: On very large "
                                      "datasets it is possible to run out of memory performing "
                                      "this calculation."
                                      "\nL|'face-cnn': Sort faces by their landmarks. You can "
                                      "adjust the threshold with the '-t' (--ref_threshold) "
                                      "option."
                                      "\nL|'face-cnn-dissim': Like 'face-cnn' but sorts by "
                                      "dissimilarity."
                                      "\nL|'face-yaw': Sort faces by Yaw (rotation left to right)."
                                      "\nL|'hist': Sort faces by their color histogram. You can "
                                      "adjust the threshold with the '-t' (--ref_threshold) "
                                      "option."
                                      "\nL|'hist-dissim': Like 'hist' but sorts by dissimilarity."
                                      "\nL|'color-gray': Sort images by the average intensity of "
                                      "the converted grayscale color channel."
                                      "\nL|'color-luma': Sort images by the average intensity of "
                                      "the converted Y color channel. Bright lighting and "
                                      "oversaturated images will be ranked first."
                                      "\nL|'color-green': Sort images by the average intensity of "
                                      "the converted Cg color channel. Green images will be "
                                      "ranked first and red images will be last."
                                      "\nL|'color-orange': Sort images by the average intensity "
                                      "of the converted Co color channel. Orange images will be "
                                      "ranked first and blue images will be last."
                                      "\nDefault: hist"})
        argument_list.append({"opts": ('-k', '--keep'),
                              "action": 'store_true',
                              "dest": 'keep_original',
                              "default": False,
                              "group": "output",
                              "help": "Keeps the original files in the input "
                                      "directory. Be careful when using this "
                                      "with rename grouping and no specified "
                                      "output directory as this would keep "
                                      "the original and renamed files in the "
                                      "same directory."})
        argument_list.append({"opts": ('-t', '--ref_threshold'),
                              "action": Slider,
                              "min_max": (-1.0, 10.0),
                              "rounding": 2,
                              "type": float,
                              "dest": 'min_threshold',
                              "group": "sort settings",
                              "default": -1.0,
                              "help": "Float value. "
                                      "Minimum threshold to use for grouping comparison with "
                                      "'face-cnn' and 'hist' methods. The lower the value the "
                                      "more discriminating the grouping is. Leaving -1.0 will "
                                      "allow the program set the default value automatically. "
                                      "For face-cnn 7.2 should be enough, with 4 being very "
                                      "discriminating. For hist 0.3 should be enough, with 0.2 "
                                      "being very discriminating. Be careful setting a value "
                                      "that's too low in a directory with many images, as this "
                                      "could result in a lot of directories being created. "
                                      "Defaults: face-cnn 7.2, hist 0.3"})

        argument_list.append({"opts": ('-fp', '--final-process'),
                              "action": Radio,
                              "type": str,
                              "choices": ("folders", "rename"),
                              "dest": 'final_process',
                              "default": "rename",
                              "group": "output",
                              "help": "R|Default: rename."
                                      "\nL|'folders': files are sorted using "
                                      "the -s/--sort-by method, then they "
                                      "are organized into folders using "
                                      "the -g/--group-by grouping method."
                                      "\nL|'rename': files are sorted using "
                                      "the -s/--sort-by then they are "
                                      "renamed."})

        argument_list.append({"opts": ('-g', '--group-by'),
                              "action": Radio,
                              "type": str,
                              "choices": ("blur", "face-cnn", "face-yaw", "hist"),
                              "dest": 'group_method',
                              "group": "output",
                              "default": "hist",
                              "help": "Group by method. "
                                      "When -fp/--final-processing by "
                                      "folders choose the how the images are "
                                      "grouped after sorting. "
                                      "Default: hist"})

        argument_list.append({"opts": ('-b', '--bins'),
                              "action": Slider,
                              "min_max": (1, 100),
                              "rounding": 1,
                              "type": int,
                              "dest": 'num_bins',
                              "group": "output",
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

        argument_list.append({"opts": ("-be", "--backend"),
                              "action": Radio,
                              "type": str.upper,
                              "choices": ("CPU", "GPU"),
                              "default": "GPU",
                              "group": "settings",
                              "help": "Backend to use for VGG Face inference."
                                      "Only used for sort by 'face'."})

        argument_list.append({"opts": ('-l', '--log-changes'),
                              "action": 'store_true',
                              "group": "settings",
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
                              "group": "settings",
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
