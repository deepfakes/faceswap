#!/usr/bin/env python3
""" The master tools.py script """
import os
import sys

from importlib import import_module

# Importing the various tools
from lib.cli.args import FullHelpArgumentParser

# Python version check
if sys.version_info[0] < 3:
    raise Exception("This program requires at least python3.2")
if sys.version_info[0] == 3 and sys.version_info[1] < 2:
    raise Exception("This program requires at least python3.2")


def bad_args(args):  # pylint:disable=unused-argument
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
            cliarg_class = getattr(module, "{}Args".format(tool_name.title()))
            help_text = getattr(module, "_HELPTEXT")
            yield tool_name, help_text, cliarg_class


if __name__ == "__main__":
    _TOOLS_WARNING = "Please backup your data and/or test the tool you want "
    _TOOLS_WARNING += "to use with a smaller data set to make sure you "
    _TOOLS_WARNING += "understand how it works."
    print(_TOOLS_WARNING)

    PARSER = FullHelpArgumentParser()
    SUBPARSER = PARSER.add_subparsers()
    for tool, helptext, cli_args in _get_cli_opts():
        cli_args(SUBPARSER, tool, helptext)
    PARSER.set_defaults(func=bad_args)
    ARGUMENTS = PARSER.parse_args()
    ARGUMENTS.func(ARGUMENTS)
