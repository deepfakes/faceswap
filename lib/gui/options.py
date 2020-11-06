#!/usr/bin python3
""" Cli Options for the GUI """
import inspect
from argparse import SUPPRESS
from importlib import import_module
import logging
import os
import re
import sys
from collections import OrderedDict

from lib.cli import actions, args as cli
from .utils import get_images
from .control_helper import ControlPanelOption

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class CliOptions():
    """ Class and methods for the command line options """
    def __init__(self):
        logger.debug("Initializing %s", self.__class__.__name__)
        self.categories = ("faceswap", "tools")
        self.commands = dict()
        self.opts = dict()
        self.build_options()
        logger.debug("Initialized %s", self.__class__.__name__)

    def build_options(self):
        """ Get the commands that belong to each category """
        for category in self.categories:
            logger.debug("Building '%s'", category)
            if category == "tools":
                mod_classes = self._get_tools_cli_classes()
                self.commands[category] = self.sort_commands(category, mod_classes)
                for tool in sorted(mod_classes):
                    self.opts.update(self.extract_options(mod_classes[tool], [tool]))
            else:
                mod_classes = self.get_cli_classes(cli)
                self.commands[category] = self.sort_commands(category, mod_classes)
                self.opts.update(self.extract_options(cli, mod_classes))
            logger.debug("Built '%s'", category)

    @staticmethod
    def get_cli_classes(cli_source):
        """ Parse the cli scripts for the argument classes """
        mod_classes = []
        for name, obj in inspect.getmembers(cli_source):
            if inspect.isclass(obj) and name.lower().endswith("args") \
                    and name.lower() not in (("faceswapargs",
                                              "extractconvertargs",
                                              "guiargs")):
                mod_classes.append(name)
        logger.debug(mod_classes)
        return mod_classes

    @staticmethod
    def _get_tools_cli_classes():
        """ Parse the tools cli scripts for the argument classes """
        base_path = os.path.realpath(os.path.dirname(sys.argv[0]))
        tools_dir = os.path.join(base_path, "tools")
        mod_classes = dict()
        for tool_name in sorted(os.listdir(tools_dir)):
            cli_file = os.path.join(tools_dir, tool_name, "cli.py")
            if os.path.exists(cli_file):
                mod = ".".join(("tools", tool_name, "cli"))
                mod_classes["{}Args".format(tool_name.title())] = import_module(mod)
        return mod_classes

    def sort_commands(self, category, classes):
        """ Format classes into command names and sort:
            Specific workflow order for faceswap.
            Alphabetical for all others """
        commands = sorted(self.format_command_name(command)
                          for command in classes)
        if category == "faceswap":
            ordered = ["extract", "train", "convert"]
            commands = ordered + [command for command in commands
                                  if command not in ordered]
        logger.debug(commands)
        return commands

    @staticmethod
    def format_command_name(classname):
        """ Format args class name to command """
        return classname.lower()[:-4]

    def extract_options(self, cli_source, mod_classes):
        """ Extract the existing ArgParse Options
            into master options Dictionary """
        subopts = dict()
        for classname in mod_classes:
            logger.debug("Processing: (classname: '%s')", classname)
            command = self.format_command_name(classname)
            info, options = self.get_cli_arguments(cli_source, classname, command)
            options = self.process_options(options, command)
            options["helptext"] = info
            logger.debug("Processed: (classname: '%s', command: '%s', options: %s)",
                         classname, command, options)
            subopts[command] = options
        return subopts

    @staticmethod
    def get_cli_arguments(cli_source, classname, command):
        """ Extract the options from the main and tools cli files """
        meth = getattr(cli_source, classname)(None, command)
        return meth.info, meth.argument_list + meth.optional_arguments + meth.global_arguments

    def process_options(self, command_options, command):
        """ Process the options for a single command """
        gui_options = OrderedDict()
        for opt in command_options:
            logger.trace("Processing: %s", opt)
            if opt.get("help", "") == SUPPRESS:
                logger.trace("Skipping suppressed option: %s", opt)
                continue
            title = self.set_control_title(opt["opts"])
            cpanel_option = ControlPanelOption(
                title,
                self.get_data_type(opt),
                group=opt.get("group", None),
                default=opt.get("default", None),
                choices=opt.get("choices", None),
                is_radio=opt.get("action", "") == actions.Radio,
                is_multi_option=opt.get("action", "") == actions.MultiOption,
                rounding=self.get_rounding(opt),
                min_max=opt.get("min_max", None),
                sysbrowser=self.get_sysbrowser(opt, command_options, command),
                helptext=opt["help"],
                track_modified=True,
                command=command)
            gui_options[title] = dict(cpanel_option=cpanel_option,
                                      opts=opt["opts"],
                                      nargs=opt.get("nargs", None))
            logger.trace("Processed: %s", gui_options)
        return gui_options

    @staticmethod
    def set_control_title(opts):
        """ Take the option switch and format it nicely """
        ctltitle = opts[1] if len(opts) == 2 else opts[0]
        ctltitle = ctltitle.replace("-", " ").replace("_", " ").strip().title()
        return ctltitle

    @staticmethod
    def get_data_type(opt):
        """ Return a datatype for passing into control_helper.py to get the correct control """
        if opt.get("type", None) is not None and isinstance(opt["type"], type):
            retval = opt["type"]
        elif opt.get("action", "") in ("store_true", "store_false"):
            retval = bool
        else:
            retval = str
        return retval

    @staticmethod
    def get_rounding(opt):
        """ Return rounding if correct data type, else None """
        dtype = opt.get("type", None)
        if dtype == float:
            retval = opt.get("rounding", 2)
        elif dtype == int:
            retval = opt.get("rounding", 1)
        else:
            retval = None
        return retval

    def get_sysbrowser(self, option, options, command):
        """ Return the system file browser and file types if required else None """
        action = option.get("action", None)
        if action not in (actions.DirFullPaths,
                          actions.FileFullPaths,
                          actions.FilesFullPaths,
                          actions.DirOrFileFullPaths,
                          actions.SaveFileFullPaths,
                          actions.ContextFullPaths):
            return None

        retval = dict()
        action_option = None
        if option.get("action_option", None) is not None:
            self.expand_action_option(option, options)
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
        elif action == actions.ContextFullPaths and action_option:
            retval["browser"] = ["context"]
            retval["command"] = command
            retval["action_option"] = action_option
            retval["destination"] = option.get("dest", option["opts"][1].replace("--", ""))
        else:
            retval["browser"] = ["folder"]
        logger.debug(retval)
        return retval

    @staticmethod
    def expand_action_option(option, options):
        """ Expand the action option to the full command name """
        opts = {opt["opts"][0]: opt["opts"][-1]
                for opt in options}
        old_val = option["action_option"]
        new_val = opts[old_val]
        logger.debug("Updating action option from '%s' to '%s'", old_val, new_val)
        option["action_option"] = new_val

    def gen_command_options(self, command):
        """ Yield each option for specified command """
        for key, val in self.opts[command].items():
            if not isinstance(val, dict):
                continue
            yield key, val

    def options_to_process(self, command=None):
        """ Return a consistent object for processing regardless of whether processing all commands
            or just one command for reset and clear. Removes helptext from return value """
        if command is None:
            options = [opt for opts in self.opts.values()
                       for opt in opts.values() if isinstance(opt, dict)]
        else:
            options = [opt for opt in self.opts[command].values() if isinstance(opt, dict)]
        return options

    def reset(self, command=None):
        """ Reset the options for all or passed command
            back to default value """
        logger.debug("Resetting options to default. (command: '%s'", command)
        for option in self.options_to_process(command):
            cp_opt = option["cpanel_option"]
            default = "" if cp_opt.default is None else cp_opt.default
            if (option.get("nargs", None)
                    and isinstance(default, (list, tuple))):
                default = ' '.join(str(val) for val in default)
            cp_opt.set(default)

    def clear(self, command=None):
        """ Clear the options values for all or passed commands """
        logger.debug("Clearing options. (command: '%s'", command)
        for option in self.options_to_process(command):
            cp_opt = option["cpanel_option"]
            if isinstance(cp_opt.get(), bool):
                cp_opt.set(False)
            elif isinstance(cp_opt.get(), (int, float)):
                cp_opt.set(0)
            else:
                cp_opt.set("")

    def get_option_values(self, command=None):
        """ Return all or single command control titles with the associated tk_var value """
        ctl_dict = dict()
        for cmd, opts in self.opts.items():
            if command and command != cmd:
                continue
            cmd_dict = dict()
            for key, val in opts.items():
                if not isinstance(val, dict):
                    continue
                cmd_dict[key] = val["cpanel_option"].get()
            ctl_dict[cmd] = cmd_dict
        logger.debug("command: '%s', ctl_dict: %s", command, ctl_dict)
        return ctl_dict

    def get_one_option_variable(self, command, title):
        """ Return a single tk_var for the specified
            command and control_title """
        for opt_title, option in self.gen_command_options(command):
            if opt_title == title:
                return option["cpanel_option"].tk_var
        return None

    def gen_cli_arguments(self, command):
        """ Return the generated cli arguments for the selected command """
        for _, option in self.gen_command_options(command):
            optval = str(option["cpanel_option"].get())
            opt = option["opts"][0]
            if command in ("extract", "convert") and opt == "-o":
                get_images().set_faceswap_output_path(optval)
            if optval in ("False", ""):
                continue
            if optval == "True":
                yield (opt, )
            else:
                if option.get("nargs", None):
                    if "\"" in optval:
                        optval = [arg[1:-1] for arg in re.findall(r"\".+?\"", optval)]
                    else:
                        optval = optval.split(" ")
                    opt = [opt] + optval
                else:
                    opt = (opt, optval)
                yield opt
