#!/usr/bin python3
""" The command frame for Faceswap GUI """

import logging
import gettext
import tkinter as tk
from tkinter import ttk

from .control_helper import ControlPanel
from .custom_widgets import Tooltip
from .utils import get_images, get_config
from .options import CliOption

logger = logging.getLogger(__name__)

# LOCALES
_LANG = gettext.translation("gui.tooltips", localedir="locales", fallback=True)
_ = _LANG.gettext


class CommandNotebook(ttk.Notebook):  # pylint:disable=too-many-ancestors
    """ Frame to hold each individual tab of the command notebook """

    def __init__(self, parent):
        logger.debug("Initializing %s: (parent: %s)", self.__class__.__name__, parent)
        self.actionbtns = {}
        super().__init__(parent)
        parent.add(self)

        self.tools_notebook = ToolsNotebook(self)
        self.set_running_task_trace()
        self.build_tabs()
        self.modified_vars = self._set_modified_vars()
        get_config().set_command_notebook(self)
        logger.debug("Initialized %s", self.__class__.__name__)

    @property
    def tab_names(self):
        """ dict: Command tab titles with their IDs """
        return {self.tab(tab_id, "text").lower(): tab_id
                for tab_id in range(0, self.index("end"))}

    @property
    def tools_tab_names(self):
        """ dict: Tools tab titles with their IDs """
        return {self.tools_notebook.tab(tab_id, "text").lower(): tab_id
                for tab_id in range(0, self.tools_notebook.index("end"))}

    def set_running_task_trace(self):
        """ Set trigger action for the running task
            to change the action buttons text and command """
        logger.debug("Set running trace")
        tk_vars = get_config().tk_vars
        tk_vars.running_task.trace("w", self.change_action_button)

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

        for cmd, action in self.actionbtns.items():
            btnact = action
            if tk_vars.running_task.get():
                ttl = " Stop"
                img = get_images().icons["stop"]
                hlp = "Exit the running process"
            else:
                ttl = f" {cmd.title()}"
                img = get_images().icons["start"]
                hlp = f"Run the {cmd.title()} script"
            logger.debug("Updated Action Button: '%s'", ttl)
            btnact.config(text=ttl, image=img)
            Tooltip(btnact, text=hlp, wrap_length=200)

    def _set_modified_vars(self):
        """ Set the tkinter variable for each tab to indicate whether contents
        have been modified """
        tkvars = {}
        for tab in self.tab_names:
            if tab == "tools":
                for ttab in self.tools_tab_names:
                    var = tk.BooleanVar()
                    var.set(False)
                    tkvars[ttab] = var
                continue
            var = tk.BooleanVar()
            var.set(False)
            tkvars[tab] = var
        logger.debug("Set modified vars: %s", tkvars)
        return tkvars


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
        super().__init__(parent, name=f"tab_{command.lower()}")

        self.category = category
        self.actionbtns = parent.actionbtns
        self.command = command

        self.build_tab()
        logger.debug("Initialized %s", self.__class__.__name__)

    def build_tab(self):
        """ Build the tab """
        logger.debug("Build Tab: '%s'", self.command)
        options = get_config().cli_opts.opts[self.command]
        cp_opts = [val.cpanel_option for val in options.values() if isinstance(val, CliOption)]
        ControlPanel(self,
                     cp_opts,
                     label_width=16,
                     option_columns=3,
                     columns=1,
                     header_text=options.get("helptext", None),
                     style="CPanel")
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
        logger.debug("Initialized %s", self.__class__.__name__)

    def add_action_button(self, category, actionbtns):
        """ Add the action buttons for page """
        logger.debug("Add action buttons: '%s'", self.title)
        actframe = ttk.Frame(self)
        actframe.pack(fill=tk.X, side=tk.RIGHT)

        tk_vars = get_config().tk_vars
        var_value = f"{category},{self.command}"

        btngen = ttk.Button(actframe,
                            image=get_images().icons["generate"],
                            text=" Generate",
                            compound=tk.LEFT,
                            width=14,
                            command=lambda: tk_vars.generate_command.set(var_value))
        btngen.pack(side=tk.LEFT, padx=5)
        Tooltip(btngen,
                text=_("Output command line options to the console"),
                wrap_length=200)

        btnact = ttk.Button(actframe,
                            image=get_images().icons["start"],
                            text=f" {self.title}",
                            compound=tk.LEFT,
                            width=14,
                            command=lambda: tk_vars.action_command.set(var_value))
        btnact.pack(side=tk.LEFT, fill=tk.X, expand=True)
        Tooltip(btnact,
                text=_("Run the {} script").format(self.title),
                wrap_length=200)
        actionbtns[self.command] = btnact

        logger.debug("Added action buttons: '%s'", self.title)
