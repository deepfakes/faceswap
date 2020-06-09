#!/usr/bin/env python3
""" Command Line Arguments for tools """
from lib.cli.args import FaceSwapArgs
from lib.cli.actions import DirOrFileFullPaths, DirFullPaths, FileFullPaths

_HELPTEXT = "This command allows you to preview swaps to tweak convert settings."


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
