#!/usr/bin/env python3
""" Command Line Arguments for tools """
import argparse
import sys
import gettext
import typing as T

from lib.cli.args import FaceSwapArgs
from lib.cli.actions import DirOrFileFullPaths, DirFullPaths, FileFullPaths, Radio, Slider

# LOCALES
_LANG = gettext.translation("tools.alignments.cli", localedir="locales", fallback=True)
_ = _LANG.gettext


_HELPTEXT = _("This command lets you perform various tasks pertaining to an alignments file.")


class AlignmentsArgs(FaceSwapArgs):
    """ Class to parse the command line arguments for Alignments tool """

    @staticmethod
    def get_info() -> str:
        """ Obtain command information.

        Returns
        -------
        str
            The help text for displaying in argparses help output
         """
        return _("Alignments tool\nThis tool allows you to perform numerous actions on or using "
                 "an alignments file against its corresponding faceset/frame source.")

    @staticmethod
    def get_argument_list() -> list[dict[str, T.Any]]:
        """ Collect the argparse argument options.

        Returns
        -------
        dict
            The argparse command line options for processing by argparse
        """
        frames_dir = _(" Must Pass in a frames folder/source video file (-r).")
        faces_dir = _(" Must Pass in a faces folder (-c).")
        frames_or_faces_dir = _(" Must Pass in either a frames folder/source video file OR a "
                                "faces folder (-r or -c).")
        frames_and_faces_dir = _(" Must Pass in a frames folder/source video file AND a faces "
                                 "folder (-r and -c).")
        output_opts = _(" Use the output option (-o) to process results.")
        argument_list = []
        argument_list.append({
            "opts": ("-j", "--job"),
            "action": Radio,
            "type": str,
            "choices": ("draw", "extract", "export", "from-faces", "missing-alignments",
                        "missing-frames", "multi-faces", "no-faces", "remove-faces", "rename",
                        "sort", "spatial"),
            "group": _("processing"),
            "required": True,
            "help": _(
                "R|Choose which action you want to perform. NB: All actions require an "
                "alignments file (-a) to be passed in."
                "\nL|'draw': Draw landmarks on frames in the selected folder/video. A "
                "subfolder will be created within the frames folder to hold the output.{0}"
                "\nL|'export': Export the contents of an alignments file to a json file. Can be "
                "used for editing alignment information in external tools and then re-importing "
                "by using Faceswap's Extract 'Import' plugins. Note: masks and identity vectors "
                "will not be included in the exported file, so will be re-generated when the json "
                "file is imported back into Faceswap. All data is exported with the origin (0, 0) "
                "at the top left of the canvas."
                "\nL|'extract': Re-extract faces from the source frames/video based on "
                "alignment data. This is a lot quicker than re-detecting faces. Can pass in "
                "the '-een' (--extract-every-n) parameter to only extract every nth frame.{1}"
                "\nL|'from-faces': Generate alignment file(s) from a folder of extracted "
                "faces. if the folder of faces comes from multiple sources, then multiple "
                "alignments files will be created. NB: for faces which have been extracted "
                "from folders of source images, rather than a video, a single alignments file "
                "will be created as there is no way for the process to know how many folders "
                "of images were originally used. You do not need to provide an alignments file "
                "path to run this job. {3}"
                "\nL|'missing-alignments': Identify frames that do not exist in the alignments "
                "file.{2}{0}"
                "\nL|'missing-frames': Identify frames in the alignments file that do not "
                "appear within the frames folder/video.{2}{0}"
                "\nL|'multi-faces': Identify where multiple faces exist within the alignments "
                "file.{2}{4}"
                "\nL|'no-faces': Identify frames that exist within the alignment file but no "
                "faces were detected.{2}{0}"
                "\nL|'remove-faces': Remove deleted faces from an alignments file. The "
                "original alignments file will be backed up.{3}"
                "\nL|'rename' - Rename faces to correspond with their parent frame and "
                "position index in the alignments file (i.e. how they are named after running "
                "extract).{3}"
                "\nL|'sort': Re-index the alignments from left to right. For alignments with "
                "multiple faces this will ensure that the left-most face is at index 0."
                "\nL|'spatial': Perform spatial and temporal filtering to smooth alignments "
                "(EXPERIMENTAL!)").format(frames_dir, frames_and_faces_dir, output_opts,
                                          faces_dir, frames_or_faces_dir)})
        argument_list.append({
            "opts": ("-o", "--output"),
            "action": Radio,
            "type": str,
            "choices": ("console", "file", "move"),
            "group": _("processing"),
            "default": "console",
            "help": _(
                "R|How to output discovered items ('faces' and 'frames' only):"
                "\nL|'console': Print the list of frames to the screen. (DEFAULT)"
                "\nL|'file': Output the list of frames to a text file (stored within the "
                "source directory)."
                "\nL|'move': Move the discovered items to a sub-folder within the source "
                "directory.")})
        argument_list.append({
            "opts": ("-a", "--alignments_file"),
            "action": FileFullPaths,
            "dest": "alignments_file",
            "type": str,
            "group": _("data"),
            # hacky solution to not require alignments file if creating alignments from faces:
            "required": not any(val in sys.argv for val in ["from-faces",
                                                            "-r",
                                                            "-frames_folder"]),
            "filetypes": "alignments",
            "help": _(
                "Full path to the alignments file to be processed. If you have input a "
                "'frames_dir' and don't provide this option, the process will try to find the "
                "alignments file at the default location. All jobs require an alignments file "
                "with the exception of 'from-faces' when the alignments file will be generated "
                "in the specified faces folder.")})
        argument_list.append({
            "opts": ("-c", "-faces_folder"),
            "action": DirFullPaths,
            "dest": "faces_dir",
            "group": ("data"),
            "help": ("Directory containing extracted faces.")})
        argument_list.append({
            "opts": ("-r", "-frames_folder"),
            "action": DirOrFileFullPaths,
            "dest": "frames_dir",
            "filetypes": "video",
            "group": _("data"),
            "help": _("Directory containing source frames that faces were extracted from.")})
        argument_list.append({
            "opts": ("-B", "--batch-mode"),
            "action": "store_true",
            "dest": "batch_mode",
            "default": False,
            "group": _("data"),
            "help": _(
                "R|Run the aligmnents tool on multiple sources. The following jobs support "
                "batch mode:"
                "\nL|draw, extract, from-faces, missing-alignments, missing-frames, no-faces, "
                "sort, spatial."
                "\nIf batch mode is selected then the other options should be set as follows:"
                "\nL|alignments_file: For 'sort' and 'spatial' this should point to the parent "
                "folder containing the alignments files to be processed. For all other jobs "
                "this option is ignored, and the alignments files must exist at their default "
                "location relative to the original frames folder/video."
                "\nL|faces_dir: For 'from-faces' this should be a parent folder, containing "
                "sub-folders of extracted faces from which to generate alignments files. For "
                "'extract' this should be a parent folder where sub-folders will be created "
                "for each extraction to be run. For all other jobs this option is ignored."
                "\nL|frames_dir: For 'draw', 'extract', 'missing-alignments', 'missing-frames' "
                "and 'no-faces' this should be a parent folder containing video files or sub-"
                "folders of images to perform the alignments job on. The alignments file "
                "should exist at the default location. For all other jobs this option is "
                "ignored.")})
        argument_list.append({
            "opts": ("-N", "--extract-every-n"),
            "type": int,
            "action": Slider,
            "dest": "extract_every_n",
            "min_max": (1, 100),
            "default": 1,
            "rounding": 1,
            "group": _("extract"),
            "help": _(
                "[Extract only] Extract every 'nth' frame. This option will skip frames when "
                "extracting faces. For example a value of 1 will extract faces from every frame, "
                "a value of 10 will extract faces from every 10th frame.")})
        argument_list.append({
            "opts": ("-z", "--size"),
            "type": int,
            "action": Slider,
            "min_max": (256, 1024),
            "rounding": 64,
            "default": 512,
            "group": _("extract"),
            "help": _("[Extract only] The output size of extracted faces.")})
        argument_list.append({
            "opts": ("-m", "--min-size"),
            "type": int,
            "action": Slider,
            "min_max": (0, 200),
            "rounding": 1,
            "default": 0,
            "dest": "min_size",
            "group": _("extract"),
            "help": _(
                "[Extract only] Only extract faces that have been resized by this percent or "
                "more to meet the specified extract size (`-sz`, `--size`). Useful for "
                "excluding low-res images from a training set. Set to 0 to extract all faces. "
                "Eg: For an extract size of 512px, A setting of 50 will only include faces "
                "that have been resized from 256px or above. Setting to 100 will only extract "
                "faces that have been resized from 512px or above. A setting of 200 will only "
                "extract faces that have been downscaled from 1024px or above.")})
        # Deprecated multi-character switches
        argument_list.append({
            "opts": ("-fc", ),
            "type": str,
            "dest": "depr_faces_folder_fc_c",
            "help": argparse.SUPPRESS})
        argument_list.append({
            "opts": ("-fr", ),
            "type": str,
            "dest": "depr_extract-every-n_een_N",
            "help": argparse.SUPPRESS})
        argument_list.append({
            "opts": ("-een", ),
            "type": int,
            "dest": "depr_faces_folder_fr_r",
            "help": argparse.SUPPRESS})
        argument_list.append({
            "opts": ("-sz", ),
            "type": int,
            "dest": "depr_size_sz_z",
            "help": argparse.SUPPRESS})
        return argument_list
