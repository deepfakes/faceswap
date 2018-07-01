#!/usr/bin python3
""" Cli Options and Config functions for the GUI """
import inspect
from argparse import SUPPRESS
from tkinter import ttk

import lib.cli as cli
from lib.Serializer import JSONSerializer
import tools.cli as ToolsCli
from .utils import FileHandler, Images


class CliOptions(object):
    """ Class and methods for the command line options """
    def __init__(self):
        self.categories = ("faceswap", "tools")
        self.commands = dict()
        self.opts = dict()
        self.build_options()

    def build_options(self):
        """ Get the commands that belong to each category """
        for category in self.categories:
            src = ToolsCli if category == "tools" else cli
            mod_classes = self.get_cli_classes(src)
            self.commands[category] = self.sort_commands(category, mod_classes)
            self.opts.update(self.extract_options(src, mod_classes))

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
            command = self.format_command_name(classname)
            options = self.get_cli_arguments(cli_source, classname, command)
            self.process_options(options)
            subopts[command] = options
        return subopts

    @staticmethod
    def get_cli_arguments(cli_source, classname, command):
        """ Extract the options from the main and tools cli files """
        meth = getattr(cli_source, classname)(None, command)
        return meth.argument_list + meth.optional_arguments

    def process_options(self, command_options):
        """ Process the options for a single command """
        for opt in command_options:
            if opt.get("help", "") == SUPPRESS:
                command_options.remove(opt)
            ctl, sysbrowser, filetypes, action_option = self.set_control(opt)
            opt["control_title"] = self.set_control_title(
                opt.get("opts", ""))
            opt["control"] = ctl
            opt["filesystem_browser"] = sysbrowser
            opt["filetypes"] = filetypes
            opt["action_option"] = action_option

    @staticmethod
    def set_control_title(opts):
        """ Take the option switch and format it nicely """
        ctltitle = opts[1] if len(opts) == 2 else opts[0]
        ctltitle = ctltitle.replace("-", " ").replace("_", " ").strip().title()
        return ctltitle

    def set_control(self, option):
        """ Set the control and filesystem browser to use for each option """
        sysbrowser = None
        action = option.get("action", None)
        action_option = option.get("action_option", None)
        filetypes = option.get("filetypes", None)
        ctl = ttk.Entry
        if action in (cli.FullPaths,
                      cli.DirFullPaths,
                      cli.FileFullPaths,
                      cli.SaveFileFullPaths,
                      cli.ContextFullPaths):
            sysbrowser, filetypes = self.set_sysbrowser(action,
                                                        filetypes,
                                                        action_option)
        elif option.get("choices", "") != "":
            ctl = ttk.Combobox
        elif option.get("action", "") == "store_true":
            ctl = ttk.Checkbutton
        return ctl, sysbrowser, filetypes, action_option

    @staticmethod
    def set_sysbrowser(action, filetypes, action_option):
        """ Set the correct file system browser and filetypes
            for the passed in action """
        sysbrowser = "folder"
        filetypes = "default" if not filetypes else filetypes
        if action == cli.FileFullPaths:
            sysbrowser = "load"
        elif action == cli.SaveFileFullPaths:
            sysbrowser = "save"
        elif action == cli.ContextFullPaths and action_option:
            sysbrowser = "context"
        return sysbrowser, filetypes

    def set_context_option(self, command):
        """ Set the tk_var for the source action option
            that dictates the context sensitive file browser. """
        actions = {item["opts"][0]: item["value"]
                   for item in self.gen_command_options(command)}
        for opt in self.gen_command_options(command):
            if opt["filesystem_browser"] == "context":
                opt["action_option"] = actions[opt["action_option"]]

    def gen_command_options(self, command):
        """ Yield each option for specified command """
        for option in self.opts[command]:
            yield option

    def options_to_process(self, command=None):
        """ Return a consistent object for processing
            regardless of whether processing all commands
            or just one command for reset and clear """
        if command is None:
            options = [opt for opts in self.opts.values() for opt in opts]
        else:
            options = [opt for opt in self.gen_command_options(command)]
        return options

    def reset(self, command=None):
        """ Reset the options for all or passed command
            back to default value """
        for option in self.options_to_process(command):
            default = option.get("default", "")
            default = "" if default is None else default
            if (option.get("nargs", None)
                    and isinstance(default, (list, tuple))):
                default = ' '.join(str(val) for val in default)
            option["value"].set(default)

    def clear(self, command=None):
        """ Clear the options values for all or passed
            commands """
        for option in self.options_to_process(command):
            if isinstance(option["value"].get(), bool):
                option["value"].set(False)
            elif isinstance(option["value"].get(), int):
                option["value"].set(0)
            else:
                option["value"].set("")

    def get_option_values(self, command=None):
        """ Return all or single command control titles
            with the associated tk_var value """
        ctl_dict = dict()
        for cmd, opts in self.opts.items():
            if command and command != cmd:
                continue
            cmd_dict = dict()
            for opt in opts:
                cmd_dict[opt["control_title"]] = opt["value"].get()
            ctl_dict[cmd] = cmd_dict
        return ctl_dict

    def get_one_option_variable(self, command, title):
        """ Return a single tk_var for the specified
            command and control_title """
        for option in self.gen_command_options(command):
            if option["control_title"] == title:
                return option["value"]
        return None

    def gen_cli_arguments(self, command):
        """ Return the generated cli arguments for
            the selected command """
        for option in self.gen_command_options(command):
            optval = str(option.get("value", "").get())
            opt = option["opts"][0]
            if command in ("extract", "convert") and opt == "-o":
                Images().pathoutput = optval
            if optval == "False" or optval == "":
                continue
            elif optval == "True":
                yield (opt, )
            else:
                if option.get("nargs", None):
                    optval = optval.split(" ")
                    opt = [opt] + optval
                else:
                    opt = (opt, optval)
                yield opt


class Config(object):
    """ Actions for loading and saving Faceswap GUI command configurations """

    def __init__(self, cli_opts, tk_vars):
        self.cli_opts = cli_opts
        self.serializer = JSONSerializer
        self.tk_vars = tk_vars

    def load(self, command=None):
        """ Load a saved config file """
        cfgfile = FileHandler("open", "config").retfile
        if not cfgfile:
            return
        cfg = self.serializer.unmarshal(cfgfile.read())
        opts = self.get_command_options(cfg, command) if command else cfg
        for cmd, opts in opts.items():
            self.set_command_args(cmd, opts)

    def get_command_options(self, cfg, command):
        """ return the saved options for the requested
            command, if not loading global options """
        opts = cfg.get(command, None)
        if not opts:
            self.tk_vars["consoleclear"].set(True)
            print("No " + command + " section found in file")
        return {command: opts}

    def set_command_args(self, command, options):
        """ Pass the saved config items back to the CliOptions """
        if not options:
            return
        for srcopt, srcval in options.items():
            optvar = self.cli_opts.get_one_option_variable(command, srcopt)
            if not optvar:
                continue
            optvar.set(srcval)

    def save(self, command=None):
        """ Save the current GUI state to a config file in json format """
        cfgfile = FileHandler("save", "config").retfile
        if not cfgfile:
            return
        cfg = self.cli_opts.get_option_values(command)
        cfgfile.write(self.serializer.marshal(cfg))
        cfgfile.close()
