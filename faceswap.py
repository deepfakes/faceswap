#!/usr/bin/env python3
""" The master faceswap.py script """
import sys

from lib.cli import args
from lib.config import generate_configs

if sys.version_info[0] < 3:
    raise Exception("This program requires at least python3.6")
if sys.version_info[0] == 3 and sys.version_info[1] < 6:
    raise Exception("This program requires at least python3.6")


_PARSER = args.FullHelpArgumentParser()


def _bad_args():
    """ Print help to console when bad arguments are provided. """
    _PARSER.print_help()
    sys.exit(0)


def _main():
    """ The main entry point into Faceswap.

    - Generates the config files, if they don't pre-exist.
    - Compiles the :class:`~lib.cli.args.FullHelpArgumentParser` objects for each section of
      Faceswap.
    - Sets the default values and launches the relevant script.
    - Outputs help if invalid parameters are provided.
    """
    generate_configs()

    subparser = _PARSER.add_subparsers()
    args.ExtractArgs(subparser, "extract", "Extract the faces from pictures")
    args.TrainArgs(subparser, "train", "This command trains the model for the two faces A and B")
    args.ConvertArgs(subparser,
                     "convert",
                     "Convert a source image to a new one with the face swapped")
    args.GuiArgs(subparser, "gui", "Launch the Faceswap Graphical User Interface")
    _PARSER.set_defaults(func=_bad_args)
    arguments = _PARSER.parse_args()
    arguments.func(arguments)


if __name__ == "__main__":
    _main()
