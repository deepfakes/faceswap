#!/usr/bin/env python3
""" The master tools.py script """
import sys
# Importing the various tools
import tools.cli as cli
from lib.cli import FullHelpArgumentParser

# Python version check
if sys.version_info[0] < 3:
    raise Exception("This program requires at least python3.2")
if sys.version_info[0] == 3 and sys.version_info[1] < 2:
    raise Exception("This program requires at least python3.2")


def bad_args(args):  # pylint:disable=unused-argument
    """ Print help on bad arguments """
    PARSER.print_help()
    exit(0)


if __name__ == "__main__":
    _TOOLS_WARNING = "Please backup your data and/or test the tool you want "
    _TOOLS_WARNING += "to use with a smaller data set to make sure you "
    _TOOLS_WARNING += "understand how it works."
    print(_TOOLS_WARNING)

    PARSER = FullHelpArgumentParser()
    SUBPARSER = PARSER.add_subparsers()
    ALIGN = cli.AlignmentsArgs(SUBPARSER,
                               "alignments",
                               "This command lets you perform various tasks pertaining to an "
                               "alignments file.")
    PREVIEW = cli.PreviewArgs(SUBPARSER,
                              "preview",
                              "This command allows you to preview swaps to tweak convert "
                              "settings.")
    EFFMPEG = cli.EffmpegArgs(SUBPARSER,
                              "effmpeg",
                              "This command allows you to easily execute common ffmpeg tasks.")
    RESTORE = cli.RestoreArgs(SUBPARSER,
                              "restore",
                              "This command lets you restore models from backup.")
    SORT = cli.SortArgs(SUBPARSER,
                        "sort",
                        "This command lets you sort images using various methods.")
    PARSER.set_defaults(func=bad_args)
    ARGUMENTS = PARSER.parse_args()
    ARGUMENTS.func(ARGUMENTS)
