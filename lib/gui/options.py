#!/usr/bin python3
""" Cli Options for the GUI """
from __future__ import annotations

import inspect
from argparse import SUPPRESS
from dataclasses import dataclass
from importlib import import_module
import logging
import os
import re
import sys
import typing as T

from lib.cli import actions
from .utils import get_images
from .control_helper import ControlPanelOption

if T.TYPE_CHECKING:
    from tkinter import Variable
    from types import ModuleType
    from lib.cli.args import FaceSwapArgs

logger = logging.getLogger(__name__)


@dataclass
class CliOption:
    """ A parsed command line option

    Parameters
    ----------
    cpanel_option: :class:`~lib.gui.control_helper.ControlPanelOption`:
        Object to hold information of a command line item for displaying in a GUI
        :class:`~lib.gui.control_helper.ControlPanel`
    opts: tuple[str, ...]:
        The short switch and long name (if exists) of the command line option
    nargs: Literal["+"] | None:
        ``None`` for not used. "+" for at least 1 argument required with values to be contained
        in a list
    """
    cpanel_option: ControlPanelOption
    """:class:`~lib.gui.control_helper.ControlPanelOption`: Object to hold information of a command
    line item for displaying in a GUI :class:`~lib.gui.control_helper.ControlPanel`"""
    opts: tuple[str, ...]
    """tuple[str, ...]: The short switch and long name (if exists) of cli option """
    nargs: T.Literal["+"] | None
    """Literal["+"] | None: ``None`` for not used. "+" for at least 1 argument required with
    values to be contained in a list """


class CliOptions():
    """ Class and methods for the command line options """
    def __init__(self) -> None:
        logger.debug("Initializing %s", self.__class__.__name__)
        self._base_path = os.path.realpath(os.path.dirname(sys.argv[0]))
        self._commands: dict[T.Literal["faceswap", "tools"], list[str]] = {"faceswap": [],
                                                                           "tools": []}
        self._opts: dict[str, dict[str, CliOption | str]] = {}
        self._build_options()
        logger.debug("Initialized %s", self.__class__.__name__)

    @property
    def categories(self) -> tuple[T.Literal["faceswap", "tools"], ...]:
        """tuple[str, str] The categories for faceswap's GUI """
        return tuple(self._commands)

    @property
    def commands(self) -> dict[T.Literal["faceswap", "tools"], list[str]]:
        """dict[str, ]"""
        return self._commands

    @property
    def opts(self) -> dict[str, dict[str, CliOption | str]]:
        """dict[str, dict[str, CliOption | str]] The command line options collected from faceswap's
        cli files """
        return self._opts

    def _get_modules_tools(self) -> list[ModuleType]:
        """ Parse the tools cli python files for the modules that contain the command line
        arguments

        Returns
        -------
        list[`types.ModuleType`]
            The modules for each faceswap tool that exists in the project
        """
        tools_dir = os.path.join(self._base_path, "tools")
        logger.debug("Scanning '%s' for cli files", tools_dir)
        retval: list[ModuleType] = []
        for tool_name in sorted(os.listdir(tools_dir)):
            cli_file = os.path.join(tools_dir, tool_name, "cli.py")
            if not os.path.exists(cli_file):
                logger.debug("File does not exist. Skipping: '%s'", cli_file)
                continue

            mod = ".".join(("tools", tool_name, "cli"))
            retval.append(import_module(mod))
            logger.debug("Collected: %s", retval[-1])
        return retval

    def _get_modules_faceswap(self) -> list[ModuleType]:
        """ Parse the faceswap cli python files for the modules that contain the command line
        arguments

        Returns
        -------
        list[`types.ModuleType`]
            The modules for each faceswap command line argument file that exists in the project
        """
        base_dir = ["lib", "cli"]
        cli_dir = os.path.join(self._base_path, *base_dir)
        logger.debug("Scanning '%s' for cli files", cli_dir)
        retval: list[ModuleType] = []

        for fname in os.listdir(cli_dir):
            if not fname.startswith("args"):
                logger.debug("Skipping file '%s'", fname)
                continue
            mod = ".".join((*base_dir, os.path.splitext(fname)[0]))
            retval.append(import_module(mod))
            logger.debug("Collected: '%s", retval[-1])
        return retval

    def _get_modules(self, category: T.Literal["faceswap", "tools"]) -> list[ModuleType]:
        """ Parse the cli files for faceswap and tools and return the imported module

        Parameters
        ----------
        category: Literal["faceswap", "tools"]
            The faceswap category to obtain the cli modules

        Returns
        -------
        list[`types.ModuleType`]
            The modules for each faceswap command/tool that exists in the project for the given
            category
        """
        logger.debug("Getting '%s' cli modules", category)
        if category == "tools":
            return self._get_modules_tools()
        return self._get_modules_faceswap()

    @classmethod
    def _get_classes(cls, module: ModuleType) -> list[T.Type[FaceSwapArgs]]:
        """ Obtain the classes from the given module that contain the command line
        arguments

        Parameters
        ----------
        module: :class:`types.ModuleType`
            The imported module to parse for command line argument classes

        Returns
        -------
        list[:class:`~lib.cli.args.FaceswapArgs`]
            The command line argument class objects that exist in the module
        """
        retval = []
        for name, obj in inspect.getmembers(module):
            if not inspect.isclass(obj) or not name.lower().endswith("args"):
                logger.debug("Skipping non-cli class object '%s'", name)
                continue
            if name.lower() in (("faceswapargs", "extractconvertargs", "guiargs")):
                logger.debug("Skipping uneeded object '%s'", name)
                continue
            logger.debug("Collecting %s", obj)
            retval.append(obj)
        logger.debug("Collected from '%s': %s", module.__name__, [c.__name__ for c in retval])
        return retval

    def _get_all_classes(self, modules: list[ModuleType]) -> list[T.Type[FaceSwapArgs]]:
        """Obtain the  the command line options classes for the given modules

        Parameters
        ----------
        modules : list[:class:`types.ModuleType`]
            The imported modules to extract the command line argument classes from

        Returns
        -------
        list[:class:`~lib.cli.args.FaceSwapArgs`]
            The valid command line class objects for the given modules
        """
        retval = []
        for module in modules:
            mod_classes = self._get_classes(module)
            if not mod_classes:
                logger.debug("module '%s' contains no cli classes. Skipping", module)
                continue
            retval.extend(mod_classes)
        logger.debug("Obtained %s cli classes from %s modules", len(retval), len(modules))
        return retval

    @classmethod
    def _class_name_to_command(cls, class_name: str) -> str:
        """ Format a FaceSwapArgs class name to a standardized command name

        Parameters
        ----------
        class_name: str
            The name of the class to convert to a command name

        Returns
        -------
        str
            The formatted command name
        """
        return class_name.lower()[:-4]

    def _store_commands(self,
                        category: T.Literal["faceswap", "tools"],
                        classes: list[T.Type[FaceSwapArgs]]) -> None:
        """ Format classes into command names and sort. Store in :attr:`commands`.
        Sorting is in specific workflow order for faceswap and alphabetical for all others

        Parameters
        ----------
        category: Literal["faceswap", "tools"]
            The category to store the command names for
        classes: list[:class:`~lib.cli.args.FaceSwapArgs`]
            The valid command line class objects for the category
        """
        class_names = [c.__name__ for c in classes]
        commands = sorted(self._class_name_to_command(n) for n in class_names)

        if category == "faceswap":
            ordered = ["extract", "train", "convert"]
            commands = ordered + [command for command in commands
                                  if command not in ordered]
        self._commands[category].extend(commands)
        logger.debug("Set '%s' commands: %s", category, self._commands[category])

    @classmethod
    def _get_cli_arguments(cls,
                           arg_class: T.Type[FaceSwapArgs],
                           command: str) -> tuple[str, list[dict[str, T.Any]]]:
        """ Extract the command line options from the given cli class

        Parameters
        ----------
        arg_class: :class:`~lib.cli.args.FaceSwapArgs`
            The class to extract the options from
        command: str
            The command name to extract the options for

        Returns
        -------
        info: str
            The helptext information for given command
        options: list[dict. str, Any]
            The command line options for the given command
        """
        args = arg_class(None, command)
        arg_list = args.argument_list + args.optional_arguments + args.global_arguments
        logger.debug("Obtain options for '%s'. Info: '%s', options: %s",
                     command, args.info, len(arg_list))
        return args.info, arg_list

    @classmethod
    def _set_control_title(cls, opts: tuple[str, ...]) -> str:
        """ Take the option switch and format it nicely

        Parameters
        ----------
        opts: tuple[str, ...]
            The option switch for a command line option

        Returns
        -------
        str
            The option switch formatted for display
        """
        ctltitle = opts[1] if len(opts) == 2 else opts[0]
        retval = ctltitle.replace("-", " ").replace("_", " ").strip().title()
        logger.debug("Formatted '%s' to '%s'",  ctltitle, retval)
        return retval

    @classmethod
    def _get_data_type(cls, opt: dict[str, T.Any]) -> type:
        """ Return a data type for passing into control_helper.py to get the correct control

        Parameters
        ----------
        option: dict[str, Any]
            The option to extract the data type from

        Returns
        -------
        :class:`type`
            The Python type for the option
        """
        type_ = opt.get("type")
        if type_ is not None and isinstance(opt["type"], type):
            retval = type_
        elif opt.get("action", "") in ("store_true", "store_false"):
            retval = bool
        else:
            retval = str
        logger.debug("Setting type to %s for %s", retval, type_)
        return retval

    @classmethod
    def _get_rounding(cls, opt: dict[str, T.Any]) -> int | None:
        """ Return rounding for the given option

        Parameters
        ----------
        option: dict[str, Any]
            The option to extract the rounding from

        Returns
        -------
        int | None
            int if the data type supports rounding otherwise ``None``
        """
        dtype = opt.get("type")
        if dtype == float:
            retval = opt.get("rounding", 2)
        elif dtype == int:
            retval = opt.get("rounding", 1)
        else:
            retval = None
        logger.debug("Setting rounding to %s for type %s", retval, dtype)
        return retval

    @classmethod
    def _expand_action_option(cls,
                              option: dict[str, T.Any],
                              options: list[dict[str, T.Any]]) -> None:
        """ Expand the action option to the full command name

        Parameters
        ----------
        option: dict[str, Any]
            The option to expand the action for
        options: list[dict[str, Any]]
            The full list of options for the command
        """
        opts = {opt["opts"][0]: opt["opts"][-1]
                for opt in options}
        old_val = option["action_option"]
        new_val = opts[old_val]
        logger.debug("Updating action option from '%s' to '%s'", old_val, new_val)
        option["action_option"] = new_val

    def _get_sysbrowser(self,
                        option: dict[str, T.Any],
                        options: list[dict[str, T.Any]],
                        command: str) -> dict[T.Literal["filetypes",
                                                        "browser",
                                                        "command",
                                                        "destination",
                                                        "action_option"], str | list[str]] | None:
        """ Return the system file browser and file types if required

        Parameters
        ----------
        option: dict[str, Any]
            The option to obtain the system browser for
        options: list[dict[str, Any]]
            The full list of options for the command
        command: str
            The command that the options belong to

        Returns
        -------
        dict[Literal["filetypes", "browser", "command",
                     "destination", "action_option"], list[str]] | None
            The browser information, if valid, or ``None`` if browser not required
        """
        action = option.get("action", None)
        if action not in (actions.DirFullPaths,
                          actions.FileFullPaths,
                          actions.FilesFullPaths,
                          actions.DirOrFileFullPaths,
                          actions.DirOrFilesFullPaths,
                          actions.SaveFileFullPaths,
                          actions.ContextFullPaths):
            return None

        retval: dict[T.Literal["filetypes",
                               "browser",
                               "command",
                               "destination",
                               "action_option"], str | list[str]] = {}
        action_option = None
        if option.get("action_option", None) is not None:
            self._expand_action_option(option, options)
            action_option = option["action_option"]
        retval["filetypes"] = option.get("filetypes", "default")
        if action == actions.FileFullPaths:
            retval["browser"] = ["load"]
        elif action == actions.FilesFullPaths:
            retval["browser"] = ["multi_load"]
        elif action == actions.SaveFileFullPaths:
            retval["browser"] = ["save"]
        elif action == actions.DirOrFileFullPaths:
            retval["browser"] = ["folder", "load"]
        elif action == actions.DirOrFilesFullPaths:
            retval["browser"] = ["folder", "multi_load"]
        elif action == actions.ContextFullPaths and action_option:
            retval["browser"] = ["context"]
            retval["command"] = command
            retval["action_option"] = action_option
            retval["destination"] = option.get("dest", option["opts"][1].replace("--", ""))
        else:
            retval["browser"] = ["folder"]
        logger.debug(retval)
        return retval

    def _process_options(self, command_options: list[dict[str, T.Any]], command: str
                         ) -> dict[str, CliOption]:
        """ Process the options for a single command

        Parameters
        ----------
        command_options: list[dict. str, Any]
            The command line options for the given command
        command: str
            The command name to process

        Returns
        -------
        dict[str, :class:`CliOption`]
            The collected command line options for handling by the GUI
        """
        retval: dict[str, CliOption] = {}
        for opt in command_options:
            logger.debug("Processing: cli option: %s", opt["opts"])
            if opt.get("help", "") == SUPPRESS:
                logger.debug("Skipping suppressed option: %s", opt)
                continue
            title = self._set_control_title(opt["opts"])
            cpanel_option = ControlPanelOption(
                title,
                self._get_data_type(opt),
                group=opt.get("group", None),
                default=opt.get("default", None),
                choices=opt.get("choices", None),
                is_radio=opt.get("action", "") == actions.Radio,
                is_multi_option=opt.get("action", "") == actions.MultiOption,
                rounding=self._get_rounding(opt),
                min_max=opt.get("min_max", None),
                sysbrowser=self._get_sysbrowser(opt, command_options, command),
                helptext=opt["help"],
                track_modified=True,
                command=command)
            retval[title] = CliOption(cpanel_option=cpanel_option,
                                      opts=opt["opts"],
                                      nargs=opt.get("nargs"))
            logger.debug("Processed: %s", retval)
        return retval

    def _extract_options(self, arguments: list[T.Type[FaceSwapArgs]]):
        """ Extract the collected command line FaceSwapArg options into master options
        :attr:`opts` dictionary

        Parameters
        ----------
        arguments: list[:class:`~lib.cli.args.FaceSwapArgs`]
            The command line class objects to process
        """
        retval = {}
        for arg_class in arguments:
            logger.debug("Processing: '%s'", arg_class.__name__)
            command = self._class_name_to_command(arg_class.__name__)
            info, options = self._get_cli_arguments(arg_class, command)
            opts = T.cast(dict[str, CliOption | str], self._process_options(options, command))
            opts["helptext"] = info
            retval[command] = opts
        self._opts.update(retval)

    def _build_options(self) -> None:
        """ Parse the command line argument modules and populate :attr:`commands` and :attr:`opts`
        for each category """
        for category in self.categories:
            modules = self._get_modules(category)
            classes = self._get_all_classes(modules)
            self._store_commands(category, classes)
            self._extract_options(classes)
            logger.debug("Built '%s'", category)

    def _gen_command_options(self, command: str
                             ) -> T.Generator[tuple[str, CliOption], None, None]:
        """ Yield each option for specified command

        Parameters
        ----------
        command: str
            The faceswap command to generate the options for

        Yields
        ------
        str
            The option name for display
        :class:`CliOption`:
            The option object
        """
        for key, val in self._opts.get(command, {}).items():
            if not isinstance(val, CliOption):
                continue
            yield key, val

    def _options_to_process(self, command: str | None = None) -> list[CliOption]:
        """ Return a consistent object for processing regardless of whether processing all commands
        or just one command for reset and clear. Removes helptext from return value

        Parameters
        ----------
        command: str | None, optional
            The command to return the options for. ``None`` for all commands. Default ``None``

        Returns
        -------
        list[:class:`CliOption`]
            The options to be processed
        """
        if command is None:
            return [opt for opts in self._opts.values()
                    for opt in opts if isinstance(opt, CliOption)]
        return [opt for opt in self._opts[command] if isinstance(opt, CliOption)]

    def reset(self, command: str | None = None) -> None:
        """ Reset the options for all or passed command back to default value

        Parameters
        ----------
        command: str | None, optional
            The command to reset the options for. ``None`` to reset for all commands.
            Default: ``None``
        """
        logger.debug("Resetting options to default. (command: '%s'", command)
        for option in self._options_to_process(command):
            cp_opt = option.cpanel_option
            default = "" if cp_opt.default is None else cp_opt.default
            if option.nargs is not None and isinstance(default, (list, tuple)):
                default = ' '.join(str(val) for val in default)
            cp_opt.set(default)

    def clear(self, command: str | None = None) -> None:
        """ Clear the options values for all or passed commands

        Parameters
        ----------
        command: str | None, optional
            The command to clear the options for. ``None`` to clear options for all commands.
            Default: ``None``
        """
        logger.debug("Clearing options. (command: '%s'", command)
        for option in self._options_to_process(command):
            cp_opt = option.cpanel_option
            if isinstance(cp_opt.get(), bool):
                cp_opt.set(False)
            elif isinstance(cp_opt.get(), (int, float)):
                cp_opt.set(0)
            else:
                cp_opt.set("")

    def get_option_values(self, command: str | None = None
                          ) -> dict[str, dict[str, bool | int | float | str]]:
        """ Return all or single command control titles with the associated tk_var value

        Parameters
        ----------
        command: str | None, optional
            The command to get the option values for. ``None`` to get all option values.
            Default: ``None``

        Returns
        -------
        dict[str, dict[str, bool | int | float | str]]
            option values in the format {command: {option_name: option_value}}
        """
        ctl_dict: dict[str, dict[str, bool | int | float | str]] = {}
        for cmd, opts in self._opts.items():
            if command and command != cmd:
                continue
            cmd_dict: dict[str, bool | int | float | str] = {}
            for key, val in opts.items():
                if not isinstance(val, CliOption):
                    continue
                cmd_dict[key] = val.cpanel_option.get()
            ctl_dict[cmd] = cmd_dict
        logger.debug("command: '%s', ctl_dict: %s", command, ctl_dict)
        return ctl_dict

    def get_one_option_variable(self, command: str, title: str) -> Variable | None:
        """ Return a single :class:`tkinter.Variable` tk_var for the specified command and
        control_title

        Parameters
        ----------
        command: str
            The command to return the variable from
        title: str
            The option title to return the variable for

        Returns
        -------
        :class:`tkinter.Variable` | None
            The requested tkinter variable, or ``None`` if it could not be found
        """
        for opt_title, option in self._gen_command_options(command):
            if opt_title == title:
                return option.cpanel_option.tk_var
        return None

    def gen_cli_arguments(self, command: str) -> T.Generator[tuple[str, ...], None, None]:
        """ Yield the generated cli arguments for the selected command

        Parameters
        ----------
        command: str
            The command to generate the command line arguments for

        Yields
        ------
        tuple[str, ...]
            The generated command line arguments
        """
        output_dir = None
        for _, option in self._gen_command_options(command):
            str_val = str(option.cpanel_option.get())
            switch = option.opts[0]
            batch_mode = command == "extract" and switch == "-b"  # Check for batch mode
            if command in ("extract", "convert") and switch == "-o":  # Output location for preview
                output_dir = str_val

            if str_val in ("False", ""):  # skip no value opts
                continue

            if str_val == "True":  # store_true just output the switch
                yield (switch, )
                continue

            if option.nargs is not None:
                if "\"" in str_val:
                    val = [arg[1:-1] for arg in re.findall(r"\".+?\"", str_val)]
                else:
                    val = str_val.split(" ")
                retval = (switch, *val)
            else:
                retval = (switch, str_val)
            yield retval

        if command in ("extract", "convert") and output_dir is not None:
            get_images().preview_extract.set_faceswap_output_path(output_dir,
                                                                  batch_mode=batch_mode)
