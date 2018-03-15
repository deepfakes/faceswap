#!/usr/bin/env python3
import sys
from lib.cli import FullHelpArgumentParser
# Importing the various tools
from tools.sort import SortProcessor

# Python version check
if sys.version_info[0] < 3:
    raise Exception("This program requires at least python3.2")
if sys.version_info[0] == 3 and sys.version_info[1] < 2:
    raise Exception("This program requires at least python3.2")


def bad_args(args):
    parser.print_help()
    exit(0)


if __name__ == "__main__":
    parser = FullHelpArgumentParser()
    subparser = parser.add_subparsers()
    sort = SortProcessor(
        subparser, "sort", "Sort images using various methods.")
    parser.set_defaults(func=bad_args)
    arguments = parser.parse_args()
    arguments.func(arguments)
