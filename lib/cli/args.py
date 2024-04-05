#!/usr/bin/env python3
""" The global and GUI Command Line Argument options for faceswap.py """

import argparse
import gettext
import logging
import re
import sys
import textwrap
import typing as T

from lib.utils import get_backend
from lib.gpu_stats import GPUStats

from .actions import FileFullPaths, MultiOption, SaveFileFullPaths
from .launcher import ScriptExecutor

logger = logging.getLogger(__name__)
_GPUS = GPUStats().cli_devices

# LOCALES
_LANG = gettext.translation("lib.cli.args", localedir="locales", fallback=True)
_ = _LANG.gettext


class FullHelpArgumentParser(argparse.ArgumentParser):
    """ Extends :class:`argparse.ArgumentParser` to output full help on bad arguments. """
    def error(self, message: str) -> T.NoReturn:
        self.print_help(sys.stderr)
        self.exit(2, f"{self.prog}: error: {message}\n")


class SmartFormatter(argparse.HelpFormatter):
    """ Extends the class :class:`argparse.HelpFormatter` to allow custom formatting in help text.

    Adapted from: https://stackoverflow.com/questions/3853722

    Notes
    -----
    Prefix help text with "R|" to override default formatting and use explicitly defined formatting
    within the help text.
    Prefixing a new line within the help text with "L|" will turn that line into a list item in
    both the cli help text and the GUI.
    """
    def __init__(self,
                 prog: str,
                 indent_increment: int = 2,
                 max_help_position: int = 24,
                 width: int | None = None) -> None:
        super().__init__(prog, indent_increment, max_help_position, width)
        self._whitespace_matcher_limited = re.compile(r'[ \r\f\v]+', re.ASCII)

    def _split_lines(self, text: str, width: int) -> list[str]:
        """ Split the given text by the given display width.

        If the text is not prefixed with "R|" then the standard
        :func:`argparse.HelpFormatter._split_lines` function is used, otherwise raw
        formatting is processed,

        Parameters
        ----------
        text: str
            The help text that is to be formatted for display
        width: int
            The display width, in characters, for the help text

        Returns
        -------
        list
            A list of split strings
        """
        if text.startswith("R|"):
            text = self._whitespace_matcher_limited.sub(' ', text).strip()[2:]
            output = []
            for txt in text.splitlines():
                indent = ""
                if txt.startswith("L|"):
                    indent = "    "
                    txt = f"  - {txt[2:]}"
                output.extend(textwrap.wrap(txt, width, subsequent_indent=indent))
            return output
        return argparse.HelpFormatter._split_lines(self,  # pylint:disable=protected-access
                                                   text,
                                                   width)


class FaceSwapArgs():
    """ Faceswap argument parser functions that are universal to all commands.

    This is the parent class to all subsequent argparsers which holds global arguments that pertain
    to all commands.

    Process the incoming command line arguments, validates then launches the relevant faceswap
    script with the given arguments.

    Parameters
    ----------
    subparser: :class:`argparse._SubParsersAction` | None
        The subparser for the given command. ``None`` if the class is being called for reading
        rather than processing
    command: str
        The faceswap command that is to be executed
    description: str, optional
        The description for the given command. Default: "default"
    """
    def __init__(self,
                 subparser: argparse._SubParsersAction | None,
                 command: str,
                 description: str = "default") -> None:
        self.global_arguments = self._get_global_arguments()
        self.info: str = self.get_info()
        self.argument_list = self.get_argument_list()
        self.optional_arguments = self.get_optional_arguments()
        self._process_suppressions()
        if not subparser:
            return
        self.parser = self._create_parser(subparser, command, description)
        self._add_arguments()
        script = ScriptExecutor(command)
        self.parser.set_defaults(func=script.execute_script)

    @staticmethod
    def get_info() -> str:
        """ Returns the information text for the current command.

        This function should be overridden with the actual command help text for each
        commands' parser.

        Returns
        -------
        str
            The information text for this command.
        """
        return ""

    @staticmethod
    def get_argument_list() -> list[dict[str, T.Any]]:
        """ Returns the argument list for the current command.

        The argument list should be a list of dictionaries pertaining to each option for a command.
        This function should be overridden with the actual argument list for each command's
        argument list.

        See existing parsers for examples.

        Returns
        -------
        list
            The list of command line options for the given command
        """
        argument_list: list[dict[str, T.Any]] = []
        return argument_list

    @staticmethod
    def get_optional_arguments() -> list[dict[str, T.Any]]:
        """ Returns the optional argument list for the current command.

        The optional arguments list is not always required, but is used when there are shared
        options between multiple commands (e.g. convert and extract). Only override if required.

        Returns
        -------
        list
            The list of optional command line options for the given command
        """
        argument_list: list[dict[str, T.Any]] = []
        return argument_list

    @staticmethod
    def _get_global_arguments() -> list[dict[str, T.Any]]:
        """ Returns the global Arguments list that are required for ALL commands in Faceswap.

        This method should NOT be overridden.

        Returns
        -------
        list
            The list of global command line options for all Faceswap commands.
        """
        global_args: list[dict[str, T.Any]] = []
        if _GPUS:
            global_args.append({
                "opts": ("-X", "--exclude-gpus"),
                "dest": "exclude_gpus",
                "action": MultiOption,
                "type": str.lower,
                "nargs": "+",
                "choices": [str(idx) for idx in range(len(_GPUS))],
                "group": _("Global Options"),
                "help": _(
                    "R|Exclude GPUs from use by Faceswap. Select the number(s) which correspond "
                    "to any GPU(s) that you do not wish to be made available to Faceswap. "
                    "Selecting all GPUs here will force Faceswap into CPU mode."
                    "\nL|{}".format(' \nL|'.join(_GPUS)))})
        global_args.append({
            "opts": ("-C", "--configfile"),
            "action": FileFullPaths,
            "filetypes": "ini",
            "type": str,
            "group": _("Global Options"),
            "help": _(
                "Optionally overide the saved config with the path to a custom config file.")})
        global_args.append({
            "opts": ("-L", "--loglevel"),
            "type": str.upper,
            "dest": "loglevel",
            "default": "INFO",
            "choices": ("INFO", "VERBOSE", "DEBUG", "TRACE"),
            "group": _("Global Options"),
            "help": _(
                "Log level. Stick with INFO or VERBOSE unless you need to file an error report. "
                "Be careful with TRACE as it will generate a lot of data")})
        global_args.append({
            "opts": ("-F", "--logfile"),
            "action": SaveFileFullPaths,
            "filetypes": 'log',
            "type": str,
            "dest": "logfile",
            "default": None,
            "group": _("Global Options"),
            "help": _("Path to store the logfile. Leave blank to store in the faceswap folder")})
        # These are hidden arguments to indicate that the GUI/Colab is being used
        global_args.append({
            "opts": ("-gui", "--gui"),
            "action": "store_true",
            "dest": "redirect_gui",
            "default": False,
            "help": argparse.SUPPRESS})
        # Deprecated multi-character switches
        global_args.append({
            "opts": ("-LF",),
            "action": SaveFileFullPaths,
            "filetypes": 'log',
            "type": str,
            "dest": "depr_logfile_LF_F",
            "help": argparse.SUPPRESS})

        return global_args

    @staticmethod
    def _create_parser(subparser: argparse._SubParsersAction,
                       command: str,
                       description: str) -> argparse.ArgumentParser:
        """ Create the parser for the selected command.

        Parameters
        ----------
        subparser: :class:`argparse._SubParsersAction`
            The subparser for the given command
        command: str
            The faceswap command that is to be executed
        description: str
            The description for the given command


        Returns
        -------
        :class:`~lib.cli.args.FullHelpArgumentParser`
            The parser for the given command
        """
        parser = subparser.add_parser(command,
                                      help=description,
                                      description=description,
                                      epilog="Questions and feedback: https://faceswap.dev/forum",
                                      formatter_class=SmartFormatter)
        return parser

    def _add_arguments(self) -> None:
        """ Parse the list of dictionaries containing the command line arguments and convert to
        argparse parser arguments. """
        options = self.global_arguments + self.argument_list + self.optional_arguments
        for option in options:
            args = option["opts"]
            kwargs = {key: option[key] for key in option.keys() if key not in ("opts", "group")}
            self.parser.add_argument(*args, **kwargs)

    def _process_suppressions(self) -> None:
        """ Certain options are only available for certain backends.

        Suppresses command line options that are not available for the running backend.
        """
        fs_backend = get_backend()
        for opt_list in [self.global_arguments, self.argument_list, self.optional_arguments]:
            for opts in opt_list:
                if opts.get("backend", None) is None:
                    continue
                opt_backend = opts.pop("backend")
                if isinstance(opt_backend, (list, tuple)):
                    opt_backend = [backend.lower() for backend in opt_backend]
                else:
                    opt_backend = [opt_backend.lower()]
                if fs_backend not in opt_backend:
                    opts["help"] = argparse.SUPPRESS


class GuiArgs(FaceSwapArgs):
    """ Creates the command line arguments for the GUI. """

    @staticmethod
    def get_argument_list() -> list[dict[str, T.Any]]:
        """ Returns the argument list for GUI arguments.

        Returns
        -------
        list
            The list of command line options for the GUI
        """
        argument_list: list[dict[str, T.Any]] = []
        argument_list.append({
            "opts": ("-d", "--debug"),
            "action": "store_true",
            "dest": "debug",
            "default": False,
            "help": _("Output to Shell console instead of GUI console")})
        return argument_list
