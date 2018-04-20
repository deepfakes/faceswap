#!/usr/bin/env python3
import sys
from lib.cli import FullHelpArgumentParser
# Importing the various tools
from tools.sort import SortProcessor
from tools.effmpeg import Effmpeg
from scripts.gui import TKGui

# Python version check
if sys.version_info[0] < 3:
    raise Exception("This program requires at least python3.2")
if sys.version_info[0] == 3 and sys.version_info[1] < 2:
    raise Exception("This program requires at least python3.2")


def bad_args(args):
    parser.print_help()
    exit(0)


if __name__ == "__main__":
    _tools_warning = "Please backup your data and/or test the tool you want "
    _tools_warning += "to use with a smaller data set to make sure you "
    _tools_warning += "understand how it works."
    print(_tools_warning)

    parser = FullHelpArgumentParser()
    subparser = parser.add_subparsers()
    effmpeg = Effmpeg(
            subparser, "effmpeg",
            "This command allows you to easily execute common ffmpeg tasks.")
    sort = SortProcessor(
            subparser, "sort",
            "This command lets you sort images using various methods.")
    guiparsers = {'effmpeg': effmpeg, 'sort': sort}
    gui = TKGui(
            subparser, guiparsers, "gui",
            "Launch the Faceswap Tools Graphical User Interface.")
    parser.set_defaults(func=bad_args)
    arguments = parser.parse_args()
    arguments.func(arguments)
