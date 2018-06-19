#!/usr/bin/env python3
""" The master faceswap.py script """
import sys

import lib.cli as cli

if sys.version_info[0] < 3:
    raise Exception("This program requires at least python3.2")
if sys.version_info[0] == 3 and sys.version_info[1] < 2:
    raise Exception("This program requires at least python3.2")


def bad_args(args):
    """ Print help on bad arguments """
    PARSER.print_help()
    exit(0)


if __name__ == "__main__":
    PARSER = cli.FullHelpArgumentParser()
    SUBPARSER = PARSER.add_subparsers()
    EXTRACT = cli.ExtractArgs(SUBPARSER,
                              "extract",
                              "Extract the faces from pictures")
    TRAIN = cli.TrainArgs(SUBPARSER,
                          "train",
                          "This command trains the model for the two faces A and B")
    CONVERT = cli.ConvertArgs(SUBPARSER,
                              "convert",
                              "Convert a source image to a new one with the face swapped")
    GUI = cli.GuiArgs(SUBPARSER,
                      "gui",
                      "Launch the Faceswap Graphical User Interface")
    PARSER.set_defaults(func=bad_args)
    ARGUMENTS = PARSER.parse_args()
    ARGUMENTS.func(ARGUMENTS)
