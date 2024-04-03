#!/usr/bin python3
""" The Menu Bars for faceswap GUI """
from __future__ import annotations
import gettext
import logging
import os
import tkinter as tk
import typing as T
from tkinter import ttk
import webbrowser

from lib.git import git
from lib.multithreading import MultiThread
from lib.serializer import get_serializer, Serializer
from lib.utils import FaceswapError
import update_deps

from .popup_configure import open_popup
from .custom_widgets import Tooltip
from .utils import get_config, get_images

if T.TYPE_CHECKING:
    from scripts.gui import FaceswapGui

logger = logging.getLogger(__name__)

# LOCALES
_LANG = gettext.translation("gui.menu", localedir="locales", fallback=True)
_ = _LANG.gettext

_RESOURCES: list[tuple[str, str]] = [
    (_("faceswap.dev - Guides and Forum"), "https://www.faceswap.dev"),
    (_("Patreon - Support this project"), "https://www.patreon.com/faceswap"),
    (_("Discord - The FaceSwap Discord server"), "https://discord.gg/VasFUAy"),
    (_("Github - Our Source Code"), "https://github.com/deepfakes/faceswap")]


class MainMenuBar(tk.Menu):  # pylint:disable=too-many-ancestors
    """ GUI Main Menu Bar

    Parameters
    ----------
    master: :class:`tkinter.Tk`
        The root tkinter object
    """
    def __init__(self, master: FaceswapGui) -> None:
        logger.debug("Initializing %s", self.__class__.__name__)
        super().__init__(master)
        self.root = master

        self.file_menu = FileMenu(self)
        self.settings_menu = SettingsMenu(self)
        self.help_menu = HelpMenu(self)

        self.add_cascade(label=_("File"), menu=self.file_menu, underline=0)
        self.add_cascade(label=_("Settings"), menu=self.settings_menu, underline=0)
        self.add_cascade(label=_("Help"), menu=self.help_menu, underline=0)
        logger.debug("Initialized %s", self.__class__.__name__)


class SettingsMenu(tk.Menu):  # pylint:disable=too-many-ancestors
    """ Settings menu items and functions

    Parameters
    ----------
    parent: :class:`tkinter.Menu`
        The main menu bar to hold this menu item
    """
    def __init__(self, parent: MainMenuBar) -> None:
        logger.debug("Initializing %s", self.__class__.__name__)
        super().__init__(parent, tearoff=0)
        self.root = parent.root
        self._build()
        logger.debug("Initialized %s", self.__class__.__name__)

    def _build(self) -> None:
        """ Add the settings menu to the menu bar """
        # pylint:disable=cell-var-from-loop
        logger.debug("Building settings menu")
        self.add_command(label=_("Configure Settings..."),
                         underline=0,
                         command=open_popup)
        logger.debug("Built settings menu")


class FileMenu(tk.Menu):  # pylint:disable=too-many-ancestors
    """ File menu items and functions

    Parameters
    ----------
    parent: :class:`tkinter.Menu`
        The main menu bar to hold this menu item
    """
    def __init__(self, parent: MainMenuBar) -> None:
        logger.debug("Initializing %s", self.__class__.__name__)
        super().__init__(parent, tearoff=0)
        self.root = parent.root
        self._config = get_config()
        self.recent_menu = tk.Menu(self, tearoff=0, postcommand=self._refresh_recent_menu)
        self._build()
        logger.debug("Initialized %s", self.__class__.__name__)

    def _refresh_recent_menu(self) -> None:
        """ Refresh recent menu on save/load of files """
        self.recent_menu.delete(0, "end")
        self._build_recent_menu()

    def _build(self) -> None:
        """ Add the file menu to the menu bar """
        logger.debug("Building File menu")
        self.add_command(label=_("New Project..."),
                         underline=0,
                         accelerator="Ctrl+N",
                         command=self._config.project.new)
        self.root.bind_all("<Control-n>", self._config.project.new)
        self.add_command(label=_("Open Project..."),
                         underline=0,
                         accelerator="Ctrl+O",
                         command=self._config.project.load)
        self.root.bind_all("<Control-o>", self._config.project.load)
        self.add_command(label=_("Save Project"),
                         underline=0,
                         accelerator="Ctrl+S",
                         command=lambda: self._config.project.save(save_as=False))
        self.root.bind_all("<Control-s>", lambda e: self._config.project.save(e, save_as=False))
        self.add_command(label=_("Save Project as..."),
                         underline=13,
                         accelerator="Ctrl+Alt+S",
                         command=lambda: self._config.project.save(save_as=True))
        self.root.bind_all("<Control-Alt-s>", lambda e: self._config.project.save(e, save_as=True))
        self.add_command(label=_("Reload Project from Disk"),
                         underline=0,
                         accelerator="F5",
                         command=self._config.project.reload)
        self.root.bind_all("<F5>", self._config.project.reload)
        self.add_command(label=_("Close Project"),
                         underline=0,
                         accelerator="Ctrl+W",
                         command=self._config.project.close)
        self.root.bind_all("<Control-w>", self._config.project.close)
        self.add_separator()
        self.add_command(label=_("Open Task..."),
                         underline=5,
                         accelerator="Ctrl+Alt+T",
                         command=lambda: self._config.tasks.load(current_tab=False))
        self.root.bind_all("<Control-Alt-t>",
                           lambda e: self._config.tasks.load(e, current_tab=False))
        self.add_separator()
        self.add_cascade(label=_("Open recent"), underline=6, menu=self.recent_menu)
        self.add_separator()
        self.add_command(label=_("Quit"),
                         underline=0,
                         accelerator="Alt+F4",
                         command=self.root.close_app)
        self.root.bind_all("<Alt-F4>", self.root.close_app)
        logger.debug("Built File menu")

    @classmethod
    def _clear_recent_files(cls, serializer: Serializer, menu_file: str) -> None:
        """ Creates or clears recent file list

        Parameters
        ----------
        serializer: :class:`~lib.serializer.Serializer`
            The serializer to use for storing files
        menu_file: str
            The file name holding the recent files
        """
        logger.debug("clearing recent files list: '%s'", menu_file)
        serializer.save(menu_file, [])

    def _build_recent_menu(self) -> None:
        """ Load recent files into menu bar """
        logger.debug("Building Recent Files menu")
        serializer = get_serializer("json")
        menu_file = os.path.join(self._config.pathcache, ".recent.json")
        if not os.path.isfile(menu_file) or os.path.getsize(menu_file) == 0:
            self._clear_recent_files(serializer, menu_file)
        try:
            recent_files = serializer.load(menu_file)
        except FaceswapError as err:
            if "Error unserializing data for type" in str(err):
                # Some reports of corruption breaking menus
                logger.warning("There was an error opening the recent files list so it has been "
                               "reset.")
                self._clear_recent_files(serializer, menu_file)
                recent_files = []

        logger.debug("Loaded recent files: %s", recent_files)
        removed_files = []
        for recent_item in recent_files:
            filename, command = recent_item
            if not os.path.isfile(filename):
                logger.debug("File does not exist. Flagging for removal: '%s'", filename)
                removed_files.append(recent_item)
                continue
            # Legacy project files didn't have a command stored
            command = command if command else "project"
            logger.debug("processing: ('%s', %s)", filename, command)
            if command.lower() == "project":
                load_func = self._config.project.load
                lbl = command
                kwargs = {"filename": filename}
            else:
                load_func = self._config.tasks.load  # type:ignore
                lbl = _("{} Task").format(command)
                kwargs = {"filename": filename, "current_tab": False}
            self.recent_menu.add_command(
                label=f"{filename} ({lbl.title()})",
                command=lambda kw=kwargs, fn=load_func: fn(**kw))  # type:ignore
        if removed_files:
            for recent_item in removed_files:
                logger.debug("Removing from recent files: `%s`", recent_item[0])
                recent_files.remove(recent_item)
            serializer.save(menu_file, recent_files)
        self.recent_menu.add_separator()
        self.recent_menu.add_command(
            label=_("Clear recent files"),
            underline=0,
            command=lambda srl=serializer, mnu=menu_file: self._clear_recent_files(  # type:ignore
                srl, mnu))

        logger.debug("Built Recent Files menu")


class HelpMenu(tk.Menu):  # pylint:disable=too-many-ancestors
    """ Help menu items and functions

    Parameters
    ----------
    parent: :class:`tkinter.Menu`
        The main menu bar to hold this menu item
    """
    def __init__(self, parent: MainMenuBar) -> None:
        logger.debug("Initializing %s", self.__class__.__name__)
        super().__init__(parent, tearoff=0)
        self.root = parent.root
        self.recources_menu = tk.Menu(self, tearoff=0)
        self._branches_menu = tk.Menu(self, tearoff=0)
        self._build()
        logger.debug("Initialized %s", self.__class__.__name__)

    def _in_thread(self, action: str):
        """ Perform selected action inside a thread

        Parameters
        ----------
        action: str
            The action to be performed. The action corresponds to the function name to be called
        """
        logger.debug("Performing help action: %s", action)
        thread = MultiThread(getattr(self, action), thread_count=1)
        thread.start()
        logger.debug("Performed help action: %s", action)

    def _output_sysinfo(self):
        """ Output system information to console """
        logger.debug("Obtaining system information")
        self.root.config(cursor="watch")
        self._clear_console()
        try:
            from lib.sysinfo import sysinfo  # pylint:disable=import-outside-toplevel
            info = sysinfo
        except Exception as err:  # pylint:disable=broad-except
            info = f"Error obtaining system info: {str(err)}"
        self._clear_console()
        logger.debug("Obtained system information: %s", info)
        print(info)
        self.root.config(cursor="")

    @classmethod
    def _process_status_output(cls, status: list[str]) -> bool:
        """ Process the output of a git status call and output information

        Parameters
        ----------
        status : list[str]
            The lines returned from a git status call

        Returns
        -------
        bool
            ``True`` if the repo can be updated otherwise ``False``
        """
        for line in status:
            if line.lower().startswith("your branch is ahead"):
                logger.warning("Your branch is ahead of the remote repo. Not updating")
                return False
            if line.lower().startswith("your branch is up to date"):
                logger.info("Faceswap is up to date.")
                return False
            if "have diverged" in line.lower():
                logger.warning("Your branch has diverged from the remote repo. Not updating")
                return False
            if line.lower().startswith("your branch is behind"):
                return True

        logger.warning("Unable to retrieve status of branch")
        return False

    def _check_for_updates(self, check: bool = False) -> bool:
        """ Check whether an update is required

        Parameters
        ----------
        check: bool
            ``True`` if we are just checking for updates ``False`` if a check and update is to be
            performed. Default: ``False``

        Returns
        -------
        bool
            ``True`` if an update is required
        """
        # Do the check
        logger.info("Checking for updates...")
        msg = ("Git is not installed or you are not running a cloned repo. "
               "Unable to check for updates")

        sync = git.update_remote()
        if not sync:
            logger.warning(msg)
            return False

        status = git.status
        if not status:
            logger.warning(msg)
            return False

        retval = self._process_status_output(status)
        if retval and check:
            logger.info("There are updates available")
        return retval

    def _check(self) -> None:
        """ Check for updates and clone repository """
        logger.debug("Checking for updates...")
        self.root.config(cursor="watch")
        self._check_for_updates(check=True)
        self.root.config(cursor="")

    def _do_update(self) -> bool:
        """ Update Faceswap

        Returns
        -------
        bool
            ``True`` if update was successful
        """
        logger.info("A new version is available. Updating...")
        success = git.pull()
        if not success:
            logger.info("An error occurred during update")
        return success

    def _update(self) -> None:
        """ Check for updates and clone repository """
        logger.debug("Updating Faceswap...")
        self.root.config(cursor="watch")
        success = False
        if self._check_for_updates():
            success = self._do_update()
        update_deps.main(is_gui=True)
        if success:
            logger.info("Please restart Faceswap to complete the update.")
        self.root.config(cursor="")

    def _build(self) -> None:
        """ Build the help menu """
        logger.debug("Building Help menu")

        self.add_command(label=_("Check for updates..."),
                         underline=0,
                         command=lambda action="_check": self._in_thread(action))  # type:ignore
        self.add_command(label=_("Update Faceswap..."),
                         underline=0,
                         command=lambda action="_update": self._in_thread(action))  # type:ignore
        if self._build_branches_menu():
            self.add_cascade(label=_("Switch Branch"), underline=7, menu=self._branches_menu)
        self.add_separator()
        self._build_recources_menu()
        self.add_cascade(label=_("Resources"), underline=0, menu=self.recources_menu)
        self.add_separator()
        self.add_command(
            label=_("Output System Information"),
            underline=0,
            command=lambda action="_output_sysinfo": self._in_thread(action))  # type:ignore
        logger.debug("Built help menu")

    def _build_branches_menu(self) -> bool:
        """ Build branch selection menu.

        Queries git for available branches and builds a menu based on output.

        Returns
        -------
        bool
            ``True`` if menu was successfully built otherwise ``False``
        """
        branches = git.branches
        if not branches:
            return False

        branches = self._filter_branches(branches)
        if not branches:
            return False

        for branch in branches:
            self._branches_menu.add_command(
                label=branch,
                command=lambda b=branch: self._switch_branch(b))  # type:ignore
        return True

    @classmethod
    def _filter_branches(cls, branches: list[str]) -> list[str]:
        """ Filter the branches, remove any non-local branches

        Parameters
        ----------
        branches: list[str]
            list of available git branches

        Returns
        -------
        list[str]
            Unique list of available branches sorted in alphabetical order
        """
        current = None
        unique = set()
        for line in branches:
            branch = line.strip()
            if branch.startswith("remotes"):
                continue
            if branch.startswith("*"):
                branch = branch.replace("*", "").strip()
                current = branch
                continue
            unique.add(branch)
        logger.debug("Found branches: %s", unique)
        if current in unique:
            logger.debug("Removing current branch from output: %s", current)
            unique.remove(current)

        retval = sorted(list(unique), key=str.casefold)
        logger.debug("Final branches: %s", retval)
        return retval

    @classmethod
    def _switch_branch(cls, branch: str) -> None:
        """ Change the currently checked out branch, and return a notification.

        Parameters
        ----------
        str
            The branch to switch to
        """
        logger.info("Switching branch to '%s'...", branch)
        if not git.checkout(branch):
            logger.error("Unable to switch branch to '%s'", branch)
            return
        logger.info("Succesfully switched to '%s'. You may want to check for updates to make sure "
                    "that you have the latest code.", branch)
        logger.info("Please restart Faceswap to complete the switch.")

    def _build_recources_menu(self) -> None:
        """ Build resources menu """
        # pylint:disable=cell-var-from-loop
        logger.debug("Building Resources Files menu")
        for resource in _RESOURCES:
            self.recources_menu.add_command(
                label=resource[0],
                command=lambda link=resource[1]: webbrowser.open_new(link))  # type:ignore
        logger.debug("Built resources menu")

    @classmethod
    def _clear_console(cls) -> None:
        """ Clear the console window """
        get_config().tk_vars.console_clear.set(True)


class TaskBar(ttk.Frame):  # pylint:disable=too-many-ancestors
    """ Task bar buttons

    Parameters
    ----------
    parent: :class:`tkinter.ttk.Frame`
        The frame that holds the task bar
    """
    def __init__(self, parent: ttk.Frame) -> None:
        super().__init__(parent)
        self._config = get_config()
        self.pack(side=tk.TOP, anchor=tk.W, fill=tk.X, expand=False)
        self._btn_frame = ttk.Frame(self)
        self._btn_frame.pack(side=tk.TOP, pady=2, anchor=tk.W, fill=tk.X, expand=False)

        self._project_btns()
        self._group_separator()
        self._task_btns()
        self._group_separator()
        self._settings_btns()
        self._section_separator()

    @classmethod
    def _loader_and_kwargs(cls, btntype: str) -> tuple[str, dict[str, bool]]:
        """ Get the loader name and key word arguments for the given button type

        Parameters
        ----------
        btntype: str
            The button type to obtain the information for

        Returns
        -------
        loader: str
            The name of the loader to use for the given button type
        kwargs: dict[str, bool]
            The keyword arguments to use for the returned loader
        """
        if btntype == "save":
            loader = btntype
            kwargs = {"save_as": False}
        elif btntype == "save_as":
            loader = "save"
            kwargs = {"save_as": True}
        else:
            loader = btntype
            kwargs = {}
        logger.debug("btntype: %s, loader: %s, kwargs: %s", btntype, loader, kwargs)
        return loader, kwargs

    @classmethod
    def _set_help(cls, btntype: str) -> str:
        """ Set the helptext for option buttons

        Parameters
        ----------
        btntype: str
            The button type to set the help text for
        """
        logger.debug("Setting help")
        hlp = ""
        task = _("currently selected Task") if btntype[-1] == "2" else _("Project")
        if btntype.startswith("reload"):
            hlp = _("Reload {} from disk").format(task)
        if btntype == "new":
            hlp = _("Create a new {}...").format(task)
        if btntype.startswith("clear"):
            hlp = _("Reset {} to default").format(task)
        elif btntype.startswith("save") and "_" not in btntype:
            hlp = _("Save {}").format(task)
        elif btntype.startswith("save_as"):
            hlp = _("Save {} as...").format(task)
        elif btntype.startswith("load"):
            msg = task
            if msg.endswith("Task"):
                msg += _(" from a task or project file")
            hlp = _("Load {}...").format(msg)
        return hlp

    def _project_btns(self) -> None:
        """ Place the project buttons """
        frame = ttk.Frame(self._btn_frame)
        frame.pack(side=tk.LEFT, anchor=tk.W, expand=False, padx=2)

        for btntype in ("new", "load", "save", "save_as", "reload"):
            logger.debug("Adding button: '%s'", btntype)

            loader, kwargs = self._loader_and_kwargs(btntype)
            cmd = getattr(self._config.project, loader)
            btn = ttk.Button(frame,
                             image=get_images().icons[btntype],
                             command=lambda fn=cmd, kw=kwargs: fn(**kw))  # type:ignore
            btn.pack(side=tk.LEFT, anchor=tk.W)
            hlp = self._set_help(btntype)
            Tooltip(btn, text=hlp, wrap_length=200)

    def _task_btns(self) -> None:
        """ Place the task buttons """
        frame = ttk.Frame(self._btn_frame)
        frame.pack(side=tk.LEFT, anchor=tk.W, expand=False, padx=2)

        for loadtype in ("load", "save", "save_as", "clear", "reload"):
            btntype = f"{loadtype}2"
            logger.debug("Adding button: '%s'", btntype)

            loader, kwargs = self._loader_and_kwargs(loadtype)
            if loadtype == "load":
                kwargs["current_tab"] = True
            cmd = getattr(self._config.tasks, loader)
            btn = ttk.Button(
                frame,
                image=get_images().icons[btntype],
                command=lambda fn=cmd, kw=kwargs: fn(**kw))  # type:ignore
            btn.pack(side=tk.LEFT, anchor=tk.W)
            hlp = self._set_help(btntype)
            Tooltip(btn, text=hlp, wrap_length=200)

    def _settings_btns(self) -> None:
        """ Place the settings buttons """
        # pylint:disable=cell-var-from-loop
        frame = ttk.Frame(self._btn_frame)
        frame.pack(side=tk.LEFT, anchor=tk.W, expand=False, padx=2)
        for name in ("extract", "train", "convert"):
            btntype = f"settings_{name}"
            btntype = btntype if btntype in get_images().icons else "settings"
            logger.debug("Adding button: '%s'", btntype)
            btn = ttk.Button(
                frame,
                image=get_images().icons[btntype],
                command=lambda n=name: open_popup(name=n))  # type:ignore
            btn.pack(side=tk.LEFT, anchor=tk.W)
            hlp = _("Configure {} settings...").format(name.title())
            Tooltip(btn, text=hlp, wrap_length=200)

    def _group_separator(self) -> None:
        """ Place a group separator """
        separator = ttk.Separator(self._btn_frame, orient="vertical")
        separator.pack(padx=(2, 1), fill=tk.Y, side=tk.LEFT)

    def _section_separator(self) -> None:
        """ Place a section separator """
        frame = ttk.Frame(self)
        frame.pack(side=tk.BOTTOM, fill=tk.X)
        separator = ttk.Separator(frame, orient="horizontal")
        separator.pack(fill=tk.X, side=tk.LEFT, expand=True)
