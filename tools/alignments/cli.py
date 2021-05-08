#!/usr/bin/env python3
""" Command Line Arguments for tools """
import gettext

from lib.cli.args import FaceSwapArgs
from lib.cli.actions import DirOrFileFullPaths, DirFullPaths, FileFullPaths, Radio, Slider


# LOCALES
_LANG = gettext.translation("tools.alignments.cli", localedir="locales", fallback=True)
_ = _LANG.gettext


_HELPTEXT = _("This command lets you perform various tasks pertaining to an alignments file.")


class AlignmentsArgs(FaceSwapArgs):
    """ Class to parse the command line arguments for Alignments tool """

    @staticmethod
    def get_info():
        """ Return command information """
        return _("Alignments tool\nThis tool allows you to perform numerous actions on or using "
                 "an alignments file against its corresponding faceset/frame source.")

    def get_argument_list(self):
        frames_dir = _(" Must Pass in a frames folder/source video file (-fr).")
        faces_dir = _(" Must Pass in a faces folder (-fc).")
        frames_or_faces_dir = _(" Must Pass in either a frames folder/source video file OR a"
                                "faces folder (-fr or -fc).")
        frames_and_faces_dir = _(" Must Pass in a frames folder/source video file AND a faces "
                                 "folder (-fr and -fc).")
        output_opts = _(" Use the output option (-o) to process results.")
        argument_list = list()
        argument_list.append(dict(
            opts=("-j", "--job"),
            action=Radio,
            type=str,
            choices=("draw", "extract", "missing-alignments", "missing-frames", "multi-faces",
                     "no-faces", "remove-faces", "rename", "sort", "spatial"),
            group=_("processing"),
            required=True,
            help=_("R|Choose which action you want to perform. NB: All actions require an "
                   "alignments file (-a) to be passed in."
                   "\nL|'draw': Draw landmarks on frames in the selected folder/video. A "
                   "subfolder will be created within the frames folder to hold the output.{0}"
                   "\nL|'extract': Re-extract faces from the source frames/video based on "
                   "alignment data. This is a lot quicker than re-detecting faces. Can pass in "
                   "the '-een' (--extract-every-n) parameter to only extract every nth frame.{1}"
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
                                             faces_dir, frames_or_faces_dir)))
        argument_list.append(dict(
            opts=("-o", "--output"),
            action=Radio,
            type=str,
            choices=("console", "file", "move"),
            group=_("processing"),
            default="console",
            help=_("R|How to output discovered items ('faces' and 'frames' only):"
                   "\nL|'console': Print the list of frames to the screen. (DEFAULT)"
                   "\nL|'file': Output the list of frames to a text file (stored within the "
                   "source directory)."
                   "\nL|'move': Move the discovered items to a sub-folder within the source "
                   "directory.")))
        argument_list.append(dict(
            opts=("-a", "--alignments_file"),
            action=FileFullPaths,
            dest="alignments_file",
            type=str,
            group=_("data"),
            required=True,
            filetypes="alignments",
            help=_("Full path to the alignments file to be processed.")))
        argument_list.append(dict(
            opts=("-fc", "-faces_folder"),
            action=DirFullPaths,
            dest="faces_dir",
            group=_("data"),
            help=_("Directory containing extracted faces.")))
        argument_list.append(dict(
            opts=("-fr", "-frames_folder"),
            action=DirOrFileFullPaths,
            dest="frames_dir",
            filetypes="video",
            group=_("data"),
            help=_("Directory containing source frames that faces were extracted from.")))
        argument_list.append(dict(
            opts=("-een", "--extract-every-n"),
            type=int,
            action=Slider,
            dest="extract_every_n",
            min_max=(1, 100),
            default=1,
            rounding=1,
            group=_("extract"),
            help=_("[Extract only] Extract every 'nth' frame. This option will skip frames when "
                   "extracting faces. For example a value of 1 will extract faces from every "
                   "frame, a value of 10 will extract faces from every 10th frame.")))
        argument_list.append(dict(
            opts=("-sz", "--size"),
            type=int,
            action=Slider,
            min_max=(256, 1024),
            default=512,
            group=_("extract"),
            rounding=64,
            help=_("[Extract only] The output size of extracted faces.")))
        argument_list.append(dict(
            opts=("-l", "--large"),
            action="store_true",
            group=_("extract"),
            default=False,
            help=_("[Extract only] Only extract faces that have not been upscaled to the required "
                   "size (`-sz`, `--size). Useful for excluding low-res images from a training "
                   "set.")))
        return argument_list
