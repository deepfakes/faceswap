#!/usr/bin/env python3
""" The master faceswap.py script """
import gettext
import sys

from lib.cli import args as cli_args
from lib.config import generate_configs
from lib.utils import get_backend


# LOCALES
_LANG = gettext.translation("faceswap", localedir="locales", fallback=True)
_ = _LANG.gettext


if sys.version_info < (3, 7):
    raise Exception("This program requires at least python3.7")
if get_backend() == "amd" and sys.version_info >= (3, 9):
    raise Exception("The AMD version of Faceswap cannot run on versions of Python higher than 3.8")


_PARSER = cli_args.FullHelpArgumentParser()


def _bad_args(*args) -> None:  # pylint:disable=unused-argument
    """ Print help to console when bad arguments are provided. """
    print(cli_args)
    _PARSER.print_help()
    sys.exit(0)


def _main() -> None:
    """ The main entry point into Faceswap.

    - Generates the config files, if they don't pre-exist.
    - Compiles the :class:`~lib.cli.args.FullHelpArgumentParser` objects for each section of
      Faceswap.
    - Sets the default values and launches the relevant script.
    - Outputs help if invalid parameters are provided.
    """
    generate_configs()

    subparser = _PARSER.add_subparsers()
    cli_args.ExtractArgs(subparser, "extract", _("Extract the faces from pictures or a video"))
    cli_args.TrainArgs(subparser, "train", _("Train a model for the two faces A and B"))
    cli_args.ConvertArgs(subparser,
                         "convert",
                         _("Convert source pictures or video to a new one with the face swapped"))
    cli_args.GuiArgs(subparser, "gui", _("Launch the Faceswap Graphical User Interface"))
    _PARSER.set_defaults(func=_bad_args)
    arguments = _PARSER.parse_args()
    arguments.func(arguments)


if __name__ == "__main__":
    _main()
