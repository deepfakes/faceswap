#!/usr/bin/env python3
""" Command Line Arguments for tools """
from lib.cli import FaceSwapArgs, DirOrFileFullPaths, FileFullPaths

_HELPTEXT = ("This command lets you perform various actions on frames, "
             "faces and alignments files using visual tools.")


class ManualArgs(FaceSwapArgs):
    """ Class to perform visual actions on an alignments file """

    @staticmethod
    def get_info():
        """ Return command information """
        return ("A tool to perform various actions on frames, faces and alignments files using "
                "visual tools")

    @staticmethod
    def get_argument_list():
        """ Put the arguments in a list so that they are accessible from both argparse and gui """
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
