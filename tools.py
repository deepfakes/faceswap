#!/usr/bin/env python3
""" The master tools.py script """
import sys
# Importing the various tools
import tools.cli as cli
from lib.cli import FullHelpArgumentParser, GuiArgs

# Python version check
if sys.version_info[0] < 3:
    raise Exception("This program requires at least python3.2")
if sys.version_info[0] == 3 and sys.version_info[1] < 2:
    raise Exception("This program requires at least python3.2")


def bad_args(args):
    """ Print help on bad arguments """
    PARSER.print_help()
    exit(0)


if __name__ == "__main__":
    _tools_warning = "Please backup your data and/or test the tool you want "
    _tools_warning += "to use with a smaller data set to make sure you "
    _tools_warning += "understand how it works."
    print(_tools_warning)

    PARSER = FullHelpArgumentParser()
    SUBPARSER = PARSER.add_subparsers()
    ALIGN = cli.AlignmentsArgs(SUBPARSER,
                               "alignments",
                               "This command lets you perform various tasks "
                               "pertaining to an alignments file.")
    EFFMPEG = cli.EffmpegArgs(SUBPARSER,
                              "effmpeg",
                              "This command allows you to easily execute "
                              "common ffmpeg tasks.")
    SORT = cli.SortArgs(SUBPARSER,
                        "sort",
                        "This command lets you sort images using various "
                        "methods.")
    GUI = GuiArgs(SUBPARSER,
                  "gui",
                  "Launch the Faceswap Tools Graphical User Interface.")
    PARSER.set_defaults(func=bad_args)
    ARGUMENTS = PARSER.parse_args()
    ARGUMENTS.func(ARGUMENTS)
