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
        argument_list.append({
            "opts": ("-al", "--alignments"),
            "action": FileFullPaths,
            "filetypes": "alignments",
            "type": str,
            "group": "data",
            "dest": "alignments_path",
            "help": "Path to the alignments file for the input, if not at the default location"})
        argument_list.append({
            "opts": ("-fr", "--frames"),
            "action": DirOrFileFullPaths,
            "filetypes": "video",
            "required": True,
            "group": "data",
            "help": "Video file or directory containing source frames that faces were extracted "
                    "from."})
        return argument_list
