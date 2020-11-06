#!/usr/bin/env python3
""" Command Line Arguments for tools """
from lib.cli.args import FaceSwapArgs
from lib.cli.actions import DirFullPaths

_HELPTEXT = "This command lets you restore models from backup."


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
