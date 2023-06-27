#!/usr/bin/env python3
""" The master tools.py script """
import gettext
import os
import sys

from importlib import import_module

# Importing the various tools
from lib.cli.args import FullHelpArgumentParser

# LOCALES
_LANG = gettext.translation("tools", localedir="locales", fallback=True)
_ = _LANG.gettext

# Python version check
if sys.version_info < (3, 10):
    raise ValueError("This program requires at least python 3.10")


def bad_args(*args):  # pylint:disable=unused-argument
    """ Print help on bad arguments """
    PARSER.print_help()
    sys.exit(0)


def _get_cli_opts():
    """ Optain the subparsers and cli options for available tools """
    base_path = os.path.realpath(os.path.dirname(sys.argv[0]))
    tools_dir = os.path.join(base_path, "tools")
    for tool_name in sorted(os.listdir(tools_dir)):
        cli_file = os.path.join(tools_dir, tool_name, "cli.py")
        if os.path.exists(cli_file):
            mod = ".".join(("tools", tool_name, "cli"))
            module = import_module(mod)
            cliarg_class = getattr(module, f"{tool_name.title()}Args")
            help_text = getattr(module, "_HELPTEXT")
            yield tool_name, help_text, cliarg_class


if __name__ == "__main__":
    PARSER = FullHelpArgumentParser()
    SUBPARSER = PARSER.add_subparsers()
    for tool, helptext, cli_args in _get_cli_opts():
        cli_args(SUBPARSER, tool, helptext)
    PARSER.set_defaults(func=bad_args)
    ARGUMENTS = PARSER.parse_args()
    ARGUMENTS.func(ARGUMENTS)
