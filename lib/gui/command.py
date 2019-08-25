#!/usr/bin python3
""" The command frame for Faceswap GUI """

import logging
import tkinter as tk
from tkinter import ttk

from .control_helper import set_slider_rounding, ControlPanel
from .tooltip import Tooltip
from .utils import get_images, get_config

logger = logging.getLogger(__name__)  # pylint:disable=invalid-name


class CommandNotebook(ttk.Notebook):  # pylint:disable=too-many-ancestors
    """ Frame to hold each individual tab of the command notebook """

    def __init__(self, parent):
        logger.debug("Initializing %s: (parent: %s)", self.__class__.__name__, parent)
        scaling_factor = get_config().scaling_factor
        width = int(420 * scaling_factor)
        height = int(500 * scaling_factor)

        self.actionbtns = dict()
        super().__init__(parent, width=width, height=height)
        parent.add(self)

        self.tools_notebook = ToolsNotebook(self)
        self.set_running_task_trace()
        self.build_tabs()
        get_config().command_notebook = self
        logger.debug("Initialized %s", self.__class__.__name__)

    def set_running_task_trace(self):
        """ Set trigger action for the running task
            to change the action buttons text and command """
        logger.debug("Set running trace")
        tk_vars = get_config().tk_vars
        tk_vars["runningtask"].trace("w", self.change_action_button)

    def build_tabs(self):
        """ Build the tabs for the relevant command """
        logger.debug("Build Tabs")
        cli_opts = get_config().cli_opts
        for category in cli_opts.categories:
            book = self.tools_notebook if category == "tools" else self
            cmdlist = cli_opts.commands[category]
            for command in cmdlist:
                title = command.title()
                commandtab = CommandTab(book, category, command)
                book.add(commandtab, text=title)
        self.add(self.tools_notebook, text="Tools")
        logger.debug("Built Tabs")

    def change_action_button(self, *args):
        """ Change the action button to relevant control """
        logger.debug("Update Action Buttons: (args: %s", args)
        tk_vars = get_config().tk_vars

        for cmd in self.actionbtns.keys():
            btnact = self.actionbtns[cmd]
            if tk_vars["runningtask"].get():
                ttl = "Terminate"
                hlp = "Exit the running process"
            else:
                ttl = cmd.title()
                hlp = "Run the {} script".format(cmd.title())
            logger.debug("Updated Action Button: '%s'", ttl)
            btnact.config(text=ttl)
            Tooltip(btnact, text=hlp, wraplength=200)


class ToolsNotebook(ttk.Notebook):  # pylint:disable=too-many-ancestors
    """ Tools sit in their own tab, but need to inherit objects from the main command notebook """
    def __init__(self, parent):
        super().__init__(parent)
        self.actionbtns = parent.actionbtns


class CommandTab(ttk.Frame):  # pylint:disable=too-many-ancestors
    """ Frame to hold each individual tab of the command notebook """

    def __init__(self, parent, category, command):
        logger.debug("Initializing %s: (category: '%s', command: '%s')",
                     self.__class__.__name__, category, command)
        super().__init__(parent)

        self.category = category
        self.actionbtns = parent.actionbtns
        self.command = command

        self.build_tab()
        logger.debug("Initialized %s", self.__class__.__name__)

    def build_tab(self):
        """ Build the tab """
        logger.debug("Build Tab: '%s'", self.command)
        options = get_config().cli_opts.opts[self.command]
        ControlPanel(self, options, label_width=16, radio_columns=2, columns=2)
        self.add_frame_separator()

        ActionFrame(self)
        logger.debug("Built Tab: '%s'", self.command)

    def add_frame_separator(self):
        """ Add a separator between top and bottom frames """
        logger.debug("Add frame seperator")
        sep = ttk.Frame(self, height=2, relief=tk.RIDGE)
        sep.pack(fill=tk.X, pady=(5, 0), side=tk.TOP)
        logger.debug("Added frame seperator")


class ActionFrame(ttk.Frame):  # pylint:disable=too-many-ancestors
    """Action Frame - Displays action controls for the command tab """

    def __init__(self, parent):
        logger.debug("Initializing %s: (command: '%s')", self.__class__.__name__, parent.command)
        super().__init__(parent)
        self.pack(fill=tk.BOTH, padx=5, pady=5, side=tk.BOTTOM, anchor=tk.N)

        self.command = parent.command
        self.title = self.command.title()

        self.add_action_button(parent.category,
                               parent.actionbtns)
        self.add_util_buttons()
        logger.debug("Initialized %s", self.__class__.__name__)

    def add_action_button(self, category, actionbtns):
        """ Add the action buttons for page """
        logger.debug("Add action buttons: '%s'", self.title)
        actframe = ttk.Frame(self)
        actframe.pack(fill=tk.X, side=tk.LEFT)
        tk_vars = get_config().tk_vars

        var_value = "{},{}".format(category, self.command)

        btnact = ttk.Button(actframe,
                            text=self.title,
                            width=10,
                            command=lambda: tk_vars["action"].set(var_value))
        btnact.pack(side=tk.LEFT)
        Tooltip(btnact,
                text="Run the {} script".format(self.title),
                wraplength=200)
        actionbtns[self.command] = btnact

        btngen = ttk.Button(actframe,
                            text="Generate",
                            width=10,
                            command=lambda: tk_vars["generate"].set(var_value))
        btngen.pack(side=tk.LEFT, padx=5)
        if self.command == "train":
            self.add_timeout(actframe)
        Tooltip(btngen,
                text="Output command line options to the console",
                wraplength=200)
        logger.debug("Added action buttons: '%s'", self.title)

    def add_timeout(self, actframe):
        """ Add a timeout option for training """
        logger.debug("Adding timeout box for %s", self.command)
        tk_var = get_config().tk_vars["traintimeout"]
        min_max = (10, 600)

        frameto = ttk.Frame(actframe)
        frameto.pack(padx=5, pady=5, side=tk.RIGHT, fill=tk.X, expand=True)
        lblto = ttk.Label(frameto, text="Timeout:", anchor=tk.W)
        lblto.pack(side=tk.LEFT)
        sldto = ttk.Scale(frameto,
                          variable=tk_var,
                          from_=min_max[0],
                          to=min_max[1],
                          command=lambda val, var=tk_var, dt=int, rn=10, mm=min_max:
                          set_slider_rounding(val, var, dt, rn, mm))
        sldto.pack(padx=5, side=tk.LEFT, fill=tk.X, expand=True)
        tboxto = ttk.Entry(frameto, width=3, textvariable=tk_var, justify=tk.RIGHT)
        tboxto.pack(side=tk.RIGHT)
        helptxt = ("Training can take some time to save and shutdown. "
                   "Set the timeout in seconds before giving up and force quitting.")
        Tooltip(sldto,
                text=helptxt,
                wraplength=200)
        Tooltip(tboxto,
                text=helptxt,
                wraplength=200)
        logger.debug("Added timeout box for %s", self.command)

    def add_util_buttons(self):
        """ Add the section utility buttons """
        logger.debug("Add util buttons")
        utlframe = ttk.Frame(self)
        utlframe.pack(side=tk.RIGHT)

        config = get_config()
        for utl in ("load", "save", "clear", "reset"):
            logger.debug("Adding button: '%s'", utl)
            img = get_images().icons[utl]
            action_cls = config if utl in (("save", "load")) else config.cli_opts
            action = getattr(action_cls, utl)
            btnutl = ttk.Button(utlframe,
                                image=img,
                                command=lambda cmd=action: cmd(self.command))
            btnutl.pack(padx=2, side=tk.LEFT)
            Tooltip(btnutl,
                    text=utl.capitalize() + " " + self.title + " config",
                    wraplength=200)
        logger.debug("Added util buttons")
