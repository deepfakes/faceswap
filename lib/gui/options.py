#!/usr/bin python3
""" Cli Options for the GUI """
import inspect
from argparse import SUPPRESS
import logging
import re
from collections import OrderedDict

from lib import cli
import tools.cli as ToolsCli
from .utils import get_images

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
            src = ToolsCli if category == "tools" else cli
            mod_classes = self.get_cli_classes(src)
            self.commands[category] = self.sort_commands(category, mod_classes)
            self.opts.update(self.extract_options(src, mod_classes))
            logger.debug("Built '%s'", category)

    @staticmethod
    def get_cli_classes(cli_source):
        """ Parse the cli scripts for the arg classes """
        mod_classes = list()
        for name, obj in inspect.getmembers(cli_source):
            if inspect.isclass(obj) and name.lower().endswith("args") \
                    and name.lower() not in (("faceswapargs",
                                              "extractconvertargs",
                                              "guiargs")):
                mod_classes.append(name)
        logger.debug(mod_classes)
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
            options = self.get_cli_arguments(cli_source, classname, command)
            options = self.process_options(options, command)
            logger.debug("Processed: (classname: '%s', command: '%s', options: %s)",
                         classname, command, options)
            subopts[command] = options
        return subopts

    @staticmethod
    def get_cli_arguments(cli_source, classname, command):
        """ Extract the options from the main and tools cli files """
        meth = getattr(cli_source, classname)(None, command)
        return meth.argument_list + meth.optional_arguments + meth.global_arguments

    def process_options(self, command_options, command):
        """ Process the options for a single command """
        gui_options = OrderedDict()
        for opt in command_options:
            logger.trace("Processing: %s", opt)
            if opt.get("help", "") == SUPPRESS:
                logger.trace("Skipping suppressed option: %s", opt)
                continue
            title = self.set_control_title(opt["opts"])
            gui_options[title] = {
                "type": self.get_data_type(opt),
                "default": opt.get("default", None),
                "value": opt.get("default", ""),
                "choices": opt.get("choices", None),
                "gui_radio": opt.get("action", "") == cli.Radio,
                "rounding": self.get_rounding(opt),
                "min_max": opt.get("min_max", None),
                "sysbrowser": self.get_sysbrowser(opt, command),
                "group": opt.get("group", None),
                "helptext": opt["help"],
                "opts": opt["opts"],
                "nargs": opt.get("nargs", None)}
            logger.trace("Processed: %s", opt)
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

    @staticmethod
    def get_sysbrowser(option, command):
        """ Return the system file browser and file types if required else None """
        action = option.get("action", None)
        if action not in (cli.FullPaths,
                          cli.DirFullPaths,
                          cli.FileFullPaths,
                          cli.FilesFullPaths,
                          cli.DirOrFileFullPaths,
                          cli.SaveFileFullPaths,
                          cli.ContextFullPaths):
            return None

        retval = dict()
        action_option = option.get("action_option", None)
        retval["filetypes"] = option.get("filetypes", "default")
        if action == cli.FileFullPaths:
            retval["browser"] = ["load"]
        elif action == cli.FilesFullPaths:
            retval["browser"] = ["load_multi"]
        elif action == cli.SaveFileFullPaths:
            retval["browser"] = ["save"]
        elif action == cli.DirOrFileFullPaths:
            retval["browser"] = ["folder", "load"]
        elif action == cli.ContextFullPaths and action_option:
            retval["browser"] = ["context"]
            retval["command"] = command
            retval["action_option"] = action_option
            retval["destination"] = option.get("dest", option["opts"][1].replace("--", ""))
        else:
            retval["browser"] = ["folder"]
        logger.debug(retval)
        return retval

    def gen_command_options(self, command):
        """ Yield each option for specified command """
        for key, val in self.opts[command].items():
            yield key, val

    def options_to_process(self, command=None):
        """ Return a consistent object for processing
            regardless of whether processing all commands
            or just one command for reset and clear """
        if command is None:
            options = [opt for opts in self.opts.values() for opt in opts.values()]
        else:
            options = [opt for opt in self.opts[command].values()]
        return options

    def reset(self, command=None):
        """ Reset the options for all or passed command
            back to default value """
        logger.debug("Resetting options to default. (command: '%s'", command)
        for option in self.options_to_process(command):
            default = option.get("default", "")
            default = "" if default is None else default
            if (option.get("nargs", None)
                    and isinstance(default, (list, tuple))):
                default = ' '.join(str(val) for val in default)
            option["selected"].set(default)

    def clear(self, command=None):
        """ Clear the options values for all or passed
            commands """
        logger.debug("Clearing options. (command: '%s'", command)
        for option in self.options_to_process(command):
            if isinstance(option["selected"].get(), bool):
                option["selected"].set(False)
            elif isinstance(option["selected"].get(), int):
                option["selected"].set(0)
            else:
                option["selected"].set("")

    def get_option_values(self, command=None):
        """ Return all or single command control titles
            with the associated tk_var value """
        ctl_dict = dict()
        for cmd, opts in self.opts.items():
            if command and command != cmd:
                continue
            cmd_dict = dict()
            for key, val in opts.items():
                cmd_dict[key] = val["selected"].get()
            ctl_dict[cmd] = cmd_dict
        logger.debug("command: '%s', ctl_dict: '%s'", command, ctl_dict)
        return ctl_dict

    def get_one_option_variable(self, command, title):
        """ Return a single tk_var for the specified
            command and control_title """
        for opt_title, option in self.gen_command_options(command):
            if opt_title == title:
                return option["selected"]
        return None

    def gen_cli_arguments(self, command):
        """ Return the generated cli arguments for
            the selected command """
        for _, option in self.gen_command_options(command):
            optval = str(option.get("selected", "").get())
            opt = option["opts"][0]
            if command in ("extract", "convert") and opt == "-o":
                get_images().pathoutput = optval
            if optval in ("False", ""):
                continue
            elif optval == "True":
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
