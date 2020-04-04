#!/usr/bin/env python3
""" Command Line Arguments for tools """
from lib.cli import FaceSwapArgs, DirOrFileFullPaths, FileFullPaths, Slider

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
        argument_list.append({
            "opts": ("-s", "--face-size"),
            "action": Slider,
            "type": int,
            "min_max": (32, 128),
            "default": 96,
            "rounding": 16,
            "dest": "face_size",
            "help": "Set the thumbnail size for the Faces Viewer. NB: All faces within the video "
                    "are loaded into RAM, which can take a lot of space. The larger the thumbnail "
                    "size, the more RAM will be used. The value given here will be scaled to your "
                    "monitor's DPI. Optional. Default: 96"})
        return argument_list
