#!/usr/bin/env python3
"""The master tools.py script."""
import gettext
import os
import sys
from importlib import import_module

# Importing the argument parser for CLI
from lib.cli.args import FullHelpArgumentParser

# LOCALES
_LANG = gettext.translation("tools", localedir="locales", fallback=True)
_ = _LANG.gettext

# Python version check
if sys.version_info < (3, 10):
    raise ValueError("This program requires at least Python 3.10")

def bad_args(*args):  # pylint:disable=unused-argument
    """Print help when invalid arguments are provided."""
    PARSER.print_help()
    sys.exit(1)  # Exit with a non-zero status to indicate an error

def _get_cli_opts():
    """Obtain the subparsers and CLI options for available tools."""
    base_path = os.path.realpath(os.path.dirname(sys.argv[0]))
    tools_dir = os.path.join(base_path, "tools")
    
    # Loop through each tool in the tools directory
    for tool_name in sorted(os.listdir(tools_dir)):
        tool_path = os.path.join(tools_dir, tool_name)
        cli_file = os.path.join(tool_path, "cli.py")
        
        if os.path.exists(cli_file):
            try:
                # Dynamically import the CLI module for the tool
                mod = f"tools.{tool_name}.cli"
                module = import_module(mod)
                
                # Retrieve the necessary class and help text
                cliarg_class = getattr(module, f"{tool_name.title()}Args")
                help_text = getattr(module, "_HELPTEXT")
                yield tool_name, help_text, cliarg_class
            except ImportError as e:
                print(f"Error importing module {mod}: {e}")
                continue
            except AttributeError as e:
                print(f"Error retrieving arguments for {tool_name}: {e}")
                continue

if __name__ == "__main__":
    PARSER = FullHelpArgumentParser()
    SUBPARSER = PARSER.add_subparsers()
    
    # Add arguments for each tool dynamically
    for tool, helptext, cli_args in _get_cli_opts():
        cli_args(SUBPARSER, tool, helptext)
    
    # Set the default function for invalid arguments
    PARSER.set_defaults(func=bad_args)
    
    # Parse the arguments and execute the corresponding function
    ARGUMENTS = PARSER.parse_args()
    ARGUMENTS.func(ARGUMENTS)
