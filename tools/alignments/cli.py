#!/usr/bin/env python3
""" Command Line Arguments for tools """
from lib.cli.args import FaceSwapArgs
from lib.cli.actions import DirOrFileFullPaths, DirFullPaths, FilesFullPaths, Radio, Slider

_HELPTEXT = "This command lets you perform various tasks pertaining to an alignments file."


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
