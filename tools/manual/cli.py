#!/usr/bin/env python3
""" The Command Line Arguments for the Manual Editor tool. """
from lib.cli.args import FaceSwapArgs, DirOrFileFullPaths, FileFullPaths

_HELPTEXT = ("This command lets you perform various actions on frames, "
             "faces and alignments files using visual tools.")


class ManualArgs(FaceSwapArgs):
    """ Generate the command line options for the Manual Editor Tool."""

    @staticmethod
    def get_info():
        """ Obtain the information about what the Manual Tool does. """
        return ("A tool to perform various actions on frames, faces and alignments files using "
                "visual tools")

    @staticmethod
    def get_argument_list():
        """ Generate the command line argument list for the Manual Tool. """
        argument_list = list()
        argument_list.append(dict(
            opts=("-al", "--alignments"),
            action=FileFullPaths,
            filetypes="alignments",
            type=str,
            group="data",
            dest="alignments_path",
            help="Path to the alignments file for the input, if not at the default location"))
        argument_list.append(dict(
            opts=("-fr", "--frames"),
            action=DirOrFileFullPaths,
            filetypes="video",
            required=True,
            group="data",
            help="Video file or directory containing source frames that faces were extracted "
                 "from."))
        argument_list.append(dict(
            opts=("-t", "--thumb-regen"),
            action="store_true",
            dest="thumb_regen",
            default=False,
            group="options",
            help="Force regeneration of the low resolution jpg thumbnails in the alignments "
                 "file."))
        argument_list.append(dict(
            opts=("-s", "--single-process"),
            action="store_true",
            dest="single_process",
            default=False,
            group="options",
            help="The process attempts to speed up generation of thumbnails by extracting from "
                 "the video in parallel threads. For some videos, this causes the caching "
                 "process to hang. If this happens, then set this option to generate the "
                 "thumbnails in a slower, but more stable single thread."))
        return argument_list
