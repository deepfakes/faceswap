#!/usr/bin/env python3
"""The master faceswap.py script."""
import gettext
import locale
import os
import sys

# Ensure the appropriate environment variable for Windows locale
if sys.platform.startswith("win"):
    os.environ["LANG"], _ = locale.getdefaultlocale()

from lib.cli import args as cli_args  # pylint:disable=wrong-import-position
from lib.cli.args_train import TrainArgs  # pylint:disable=wrong-import-position
from lib.cli.args_extract_convert import ConvertArgs, ExtractArgs  # noqa:E501 pylint:disable=wrong-import-position
from lib.config import generate_configs  # pylint:disable=wrong-import-position

# LOCALES
_LANG = gettext.translation("faceswap", localedir="locales", fallback=True)
_ = _LANG.gettext

# Ensure Python version is at least 3.10
if sys.version_info < (3, 10):
    raise ValueError("This program requires at least Python 3.10")

# Initialize the argument parser
_PARSER = cli_args.FullHelpArgumentParser()

def _bad_args(*args) -> None:
    """Print help when bad arguments are provided."""
    print(cli_args)
    _PARSER.print_help()
    sys.exit(1)  # Exit with a non-zero status to indicate an error

def _main() -> None:
    """Main entry point into Faceswap.

    - Generates the config files, if they don't pre-exist.
    - Sets up the CLI argument parser and its subcommands.
    - Launches the relevant script based on the arguments.
    """
    try:
        # Generate configuration files if needed
        generate_configs()

        # Set up subcommands for the Faceswap CLI
        subparser = _PARSER.add_subparsers()
        ExtractArgs(subparser, "extract", _("Extract faces from pictures or videos"))
        TrainArgs(subparser, "train", _("Train a model for faces A and B"))
        ConvertArgs(subparser,
                    "convert",
                    _("Convert source images or videos to a new one with swapped faces"))
        cli_args.GuiArgs(subparser, "gui", _("Launch the Faceswap Graphical User Interface"))

        # Default to showing help on invalid arguments
        _PARSER.set_defaults(func=_bad_args)

        # Parse arguments and run the corresponding function
        arguments = _PARSER.parse_args()
        arguments.func(arguments)
    
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    _main()
