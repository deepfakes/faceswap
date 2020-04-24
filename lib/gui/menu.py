#!/usr/bin python3
""" The Menu Bars for faceswap GUI """

import locale
import logging
import os
import sys
import tkinter as tk
from tkinter import ttk
import webbrowser

from importlib import import_module
from subprocess import Popen, PIPE, STDOUT

from lib.multithreading import MultiThread
from lib.serializer import get_serializer
import update_deps

from .popup_configure import popup_config
from .custom_widgets import Tooltip
from .utils import get_config, get_images

_RESOURCES = [("faceswap.dev - Guides and Forum", "https://www.faceswap.dev"),
              ("Patreon - Support this project", "https://www.patreon.com/faceswap"),
              ("Discord - The FaceSwap Discord server", "https://discord.gg/VasFUAy"),
              ("Github - Our Source Code", "https://github.com/deepfakes/faceswap")]

_CONFIG_FILES = []
_CONFIGS = dict()
_WORKING_DIR = os.path.dirname(os.path.realpath(sys.argv[0]))


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class MainMenuBar(tk.Menu):  # pylint:disable=too-many-ancestors
    """ GUI Main Menu Bar """
    def __init__(self, master=None):
        logger.debug("Initializing %s", self.__class__.__name__)
        super().__init__(master)
        self.root = master

        self.file_menu = FileMenu(self)
        self.settings_menu = SettingsMenu(self)
        self.help_menu = HelpMenu(self)

        self.add_cascade(label="File", menu=self.file_menu, underline=0)
        self.add_cascade(label="Settings", menu=self.settings_menu, underline=0)
        self.add_cascade(label="Help", menu=self.help_menu, underline=0)
        logger.debug("Initialized %s", self.__class__.__name__)


class SettingsMenu(tk.Menu):  # pylint:disable=too-many-ancestors
    """ Settings menu items and functions """
    def __init__(self, parent):
        logger.debug("Initializing %s", self.__class__.__name__)
        super().__init__(parent, tearoff=0)
        self.root = parent.root
        self.configs = self.scan_for_plugin_configs()
        self.build()
        logger.debug("Initialized %s", self.__class__.__name__)

    def scan_for_plugin_configs(self):
        """ Scan for config.ini file locations """
        global _CONFIGS, _CONFIG_FILES  # pylint:disable=global-statement
        root_path = os.path.abspath(os.path.dirname(sys.argv[0]))
        plugins_path = os.path.join(root_path, "plugins")
        logger.debug("Scanning path: '%s'", plugins_path)
        configs = dict()
        for dirpath, _, filenames in os.walk(plugins_path):
            if "_config.py" in filenames:
                plugin_type = os.path.split(dirpath)[-1]
                config = self.load_config(plugin_type)
                configs[plugin_type] = config
        logger.debug("Configs loaded: %s", sorted(list(configs.keys())))
        keys = list(configs.keys())
        for key in ("extract", "train", "convert"):
            if key in keys:
                _CONFIG_FILES.append(keys.pop(keys.index(key)))
        _CONFIG_FILES.extend([key for key in sorted(keys)])
        _CONFIGS = configs
        return configs

    @staticmethod
    def load_config(plugin_type):
        """ Load the config to generate config file if it doesn't exist and get filename """
        # Load config to generate default if doesn't exist
        mod = ".".join(("plugins", plugin_type, "_config"))
        module = import_module(mod)
        config = module.Config(None)
        logger.debug("Found '%s' config at '%s'", plugin_type, config.configfile)
        return config

    def build(self):
        """ Add the settings menu to the menu bar """
        # pylint: disable=cell-var-from-loop
        logger.debug("Building settings menu")
        for name in _CONFIG_FILES:
            label = "Configure {} Plugins...".format(name.title())
            config = self.configs[name]
            self.add_command(
                label=label,
                underline=10,
                command=lambda conf=(name, config), root=self.root: popup_config(conf, root))
        self.add_separator()
        conf = get_config().user_config
        self.add_command(
            label="GUI Settings...",
            underline=10,
            command=lambda conf=("GUI", conf), root=self.root: popup_config(conf, root))
        logger.debug("Built settings menu")


class FileMenu(tk.Menu):  # pylint:disable=too-many-ancestors
    """ File menu items and functions """
    def __init__(self, parent):
        logger.debug("Initializing %s", self.__class__.__name__)
        super().__init__(parent, tearoff=0)
        self.root = parent.root
        self._config = get_config()
        self.recent_menu = tk.Menu(self, tearoff=0, postcommand=self.refresh_recent_menu)
        self.build()
        logger.debug("Initialized %s", self.__class__.__name__)

    def build(self):
        """ Add the file menu to the menu bar """
        logger.debug("Building File menu")
        self.add_command(label="New Project...",
                         underline=0,
                         accelerator="Ctrl+N",
                         command=self._config.project.new)
        self.root.bind_all("<Control-n>", self._config.project.new)
        self.add_command(label="Open Project...",
                         underline=0,
                         accelerator="Ctrl+O",
                         command=self._config.project.load)
        self.root.bind_all("<Control-o>", self._config.project.load)
        self.add_command(label="Save Project",
                         underline=0,
                         accelerator="Ctrl+S",
                         command=lambda: self._config.project.save(save_as=False))
        self.root.bind_all("<Control-s>", lambda e: self._config.project.save(e, save_as=False))
        self.add_command(label="Save Project as...",
                         underline=13,
                         accelerator="Ctrl+Alt+S",
                         command=lambda: self._config.project.save(save_as=True))
        self.root.bind_all("<Control-Alt-s>", lambda e: self._config.project.save(e, save_as=True))
        self.add_command(label="Reload Project from Disk",
                         underline=0,
                         accelerator="F5",
                         command=self._config.project.reload)
        self.root.bind_all("<F5>", self._config.project.reload)
        self.add_command(label="Close Project",
                         underline=0,
                         accelerator="Ctrl+W",
                         command=self._config.project.close)
        self.root.bind_all("<Control-w>", self._config.project.close)
        self.add_separator()
        self.add_command(label="Open Task...",
                         underline=5,
                         accelerator="Ctrl+Alt+T",
                         command=lambda: self._config.tasks.load(current_tab=False))
        self.root.bind_all("<Control-Alt-t>",
                           lambda e: self._config.tasks.load(e, current_tab=False))
        self.add_separator()
        self.add_cascade(label="Open recent", underline=6, menu=self.recent_menu)
        self.add_separator()
        self.add_command(label="Quit",
                         underline=0,
                         accelerator="Alt+F4",
                         command=self.root.close_app)
        self.root.bind_all("<Alt-F4>", self.root.close_app)
        logger.debug("Built File menu")

    def build_recent_menu(self):
        """ Load recent files into menu bar """
        logger.debug("Building Recent Files menu")
        serializer = get_serializer("json")
        menu_file = os.path.join(self._config.pathcache, ".recent.json")
        if not os.path.isfile(menu_file) or os.path.getsize(menu_file) == 0:
            self.clear_recent_files(serializer, menu_file)
        recent_files = serializer.load(menu_file)
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
                kwargs = dict(filename=filename)
            else:
                load_func = self._config.tasks.load
                lbl = "{} Task".format(command)
                kwargs = dict(filename=filename, current_tab=False)
            self.recent_menu.add_command(
                label="{} ({})".format(filename, lbl.title()),
                command=lambda kw=kwargs, fn=load_func: fn(**kw))
        if removed_files:
            for recent_item in removed_files:
                logger.debug("Removing from recent files: `%s`", recent_item[0])
                recent_files.remove(recent_item)
            serializer.save(menu_file, recent_files)
        self.recent_menu.add_separator()
        self.recent_menu.add_command(
            label="Clear recent files",
            underline=0,
            command=lambda srl=serializer, mnu=menu_file: self.clear_recent_files(srl, mnu))

        logger.debug("Built Recent Files menu")

    @staticmethod
    def clear_recent_files(serializer, menu_file):
        """ Creates or clears recent file list """
        logger.debug("clearing recent files list: '%s'", menu_file)
        serializer.save(menu_file, list())

    def refresh_recent_menu(self):
        """ Refresh recent menu on save/load of files """
        self.recent_menu.delete(0, "end")
        self.build_recent_menu()


class HelpMenu(tk.Menu):  # pylint:disable=too-many-ancestors
    """ Help menu items and functions """
    def __init__(self, parent):
        logger.debug("Initializing %s", self.__class__.__name__)
        super().__init__(parent, tearoff=0)
        self.root = parent.root
        self.recources_menu = tk.Menu(self, tearoff=0)
        self._branches_menu = tk.Menu(self, tearoff=0)
        self.build()
        logger.debug("Initialized %s", self.__class__.__name__)

    def build(self):
        """ Build the help menu """
        logger.debug("Building Help menu")

        self.add_command(label="Check for updates...",
                         underline=0,
                         command=lambda action="check": self.in_thread(action))
        self.add_command(label="Update Faceswap...",
                         underline=0,
                         command=lambda action="update": self.in_thread(action))
        if self._build_branches_menu():
            self.add_cascade(label="Switch Branch", underline=7, menu=self._branches_menu)
        self.add_separator()
        self._build_recources_menu()
        self.add_cascade(label="Resources", underline=0, menu=self.recources_menu)
        self.add_separator()
        self.add_command(label="Output System Information",
                         underline=0,
                         command=lambda action="output_sysinfo": self.in_thread(action))
        logger.debug("Built help menu")

    def _build_branches_menu(self):
        """ Build branch selection menu.

        Queries git for available branches and builds a menu based on output.

        Returns
        -------
        bool
            ``True`` if menu was successfully built otherwise ``False``
        """
        stdout = self._get_branches()
        if stdout is None:
            return False

        branches = self._filter_branches(stdout)
        if not branches:
            return False

        for branch in branches:
            self._branches_menu.add_command(
                label=branch,
                command=lambda b=branch: self._switch_branch(b))
        return True

    @staticmethod
    def _get_branches():
        """ Get the available github branches

        Returns
        -------
        str
            The list of branches available. If no branches were found or there was an
            error then `None` is returned
        """
        gitcmd = "git branch -a"
        cmd = Popen(gitcmd, shell=True, stdout=PIPE, stderr=STDOUT, cwd=_WORKING_DIR)
        stdout, _ = cmd.communicate()
        retcode = cmd.poll()
        if retcode != 0:
            logger.debug("Unable to list git branches. return code: %s, message: %s",
                         retcode, stdout.decode().strip().replace("\n", " - "))
            return None
        return stdout.decode(locale.getpreferredencoding())

    @staticmethod
    def _filter_branches(stdout):
        """ Filter the branches, remove duplicates and the current branch and return a sorted
        list.

        Parameters
        ----------
        stdout: str
            The output from the git branch query converted to a string

        Returns
        -------
        list
            Unique list of available branches sorted in alphabetical order
        """
        current = None
        branches = set()
        for line in stdout.splitlines():
            branch = line[line.rfind("/") + 1:] if "/" in line else line.strip()
            if branch.startswith("*"):
                branch = branch.replace("*", "").strip()
                current = branch
                continue
            branches.add(branch)
        logger.debug("Found branches: %s", branches)
        if current in branches:
            logger.debug("Removing current branch from output: %s", current)
            branches.remove(current)

        branches = sorted(list(branches), key=str.casefold)
        logger.debug("Final branches: %s", branches)
        return branches

    @staticmethod
    def _switch_branch(branch):
        """ Change the currently checked out branch, and return a notification.

        Parameters
        ----------
        str
            The branch to switch to
        """
        logger.info("Switching branch to '%s'...", branch)
        gitcmd = "git checkout {}".format(branch)
        cmd = Popen(gitcmd, shell=True, stdout=PIPE, stderr=STDOUT, cwd=_WORKING_DIR)
        stdout, _ = cmd.communicate()
        retcode = cmd.poll()
        if retcode != 0:
            logger.error("Unable to switch branch. return code: %s, message: %s",
                         retcode, stdout.decode().strip().replace("\n", " - "))
            return
        logger.info("Succesfully switched to '%s'. You may want to check for updates to make sure "
                    "that you have the latest code.", branch)
        logger.info("Please restart Faceswap to complete the switch.")

    def _build_recources_menu(self):
        """ Build resources menu """
        # pylint: disable=cell-var-from-loop
        logger.debug("Building Resources Files menu")
        for resource in _RESOURCES:
            self.recources_menu.add_command(
                label=resource[0],
                command=lambda link=resource[1]: webbrowser.open_new(link))
        logger.debug("Built resources menu")

    def in_thread(self, action):
        """ Perform selected action inside a thread """
        logger.debug("Performing help action: %s", action)
        thread = MultiThread(getattr(self, action), thread_count=1)
        thread.start()
        logger.debug("Performed help action: %s", action)

    @staticmethod
    def clear_console():
        """ Clear the console window """
        get_config().tk_vars["consoleclear"].set(True)

    def output_sysinfo(self):
        """ Output system information to console """
        logger.debug("Obtaining system information")
        self.root.config(cursor="watch")
        self.clear_console()
        try:
            from lib.sysinfo import sysinfo
            info = sysinfo
        except Exception as err:  # pylint:disable=broad-except
            info = "Error obtaining system info: {}".format(str(err))
        self.clear_console()
        logger.debug("Obtained system information: %s", info)
        print(info)
        self.root.config(cursor="")

    def check(self):
        """ Check for updates and clone repository """
        logger.debug("Checking for updates...")
        self.root.config(cursor="watch")
        encoding = locale.getpreferredencoding()
        logger.debug("Encoding: %s", encoding)
        self.check_for_updates(encoding, check=True)
        self.root.config(cursor="")

    def update(self):
        """ Check for updates and clone repository """
        logger.debug("Updating Faceswap...")
        self.root.config(cursor="watch")
        encoding = locale.getpreferredencoding()
        logger.debug("Encoding: %s", encoding)
        success = False
        if self.check_for_updates(encoding):
            success = self.do_update(encoding)
        update_deps.main(logger=logger)
        if success:
            logger.info("Please restart Faceswap to complete the update.")
        self.root.config(cursor="")

    @staticmethod
    def check_for_updates(encoding, check=False):
        """ Check whether an update is required """
        # Do the check
        logger.info("Checking for updates...")
        update = False
        msg = ""
        gitcmd = "git remote update && git status -uno"
        cmd = Popen(gitcmd, shell=True, stdout=PIPE, stderr=STDOUT, cwd=_WORKING_DIR)
        stdout, _ = cmd.communicate()
        retcode = cmd.poll()
        if retcode != 0:
            msg = ("Git is not installed or you are not running a cloned repo. "
                   "Unable to check for updates")
        else:
            chk = stdout.decode(encoding).splitlines()
            for line in chk:
                if line.lower().startswith("your branch is ahead"):
                    msg = "Your branch is ahead of the remote repo. Not updating"
                    break
                if line.lower().startswith("your branch is up to date"):
                    msg = "Faceswap is up to date."
                    break
                if line.lower().startswith("your branch is behind"):
                    msg = "There are updates available"
                    update = True
                    break
                if "have diverged" in line.lower():
                    msg = "Your branch has diverged from the remote repo. Not updating"
                    break
        if not update or check:
            logger.info(msg)
        logger.debug("Checked for update. Update required: %s", update)
        return update

    @staticmethod
    def do_update(encoding):
        """ Update Faceswap """
        logger.info("A new version is available. Updating...")
        gitcmd = "git pull"
        cmd = Popen(gitcmd, shell=True, stdout=PIPE, stderr=STDOUT, bufsize=1, cwd=_WORKING_DIR)
        while True:
            output = cmd.stdout.readline().decode(encoding)
            if output == "" and cmd.poll() is not None:
                break
            if output:
                logger.debug("'%s' output: '%s'", gitcmd, output.strip())
                print(output.strip())
        retcode = cmd.poll()
        logger.debug("'%s' returncode: %s", gitcmd, retcode)
        if retcode != 0:
            logger.info("An error occurred during update. return code: %s", retcode)
            retval = False
        else:
            retval = True
        return retval


class TaskBar(ttk.Frame):  # pylint: disable=too-many-ancestors
    """ Task bar buttons """
    def __init__(self, parent):
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

    def _project_btns(self):
        frame = ttk.Frame(self._btn_frame)
        frame.pack(side=tk.LEFT, anchor=tk.W, expand=False, padx=2)

        for btntype in ("new", "load", "save", "save_as", "reload"):
            logger.debug("Adding button: '%s'", btntype)

            loader, kwargs = self._loader_and_kwargs(btntype)
            cmd = getattr(self._config.project, loader)
            btn = ttk.Button(frame,
                             image=get_images().icons[btntype],
                             command=lambda fn=cmd, kw=kwargs: fn(**kw))
            btn.pack(side=tk.LEFT, anchor=tk.W)
            hlp = self.set_help(btntype)
            Tooltip(btn, text=hlp, wraplength=200)

    def _task_btns(self):
        frame = ttk.Frame(self._btn_frame)
        frame.pack(side=tk.LEFT, anchor=tk.W, expand=False, padx=2)

        for loadtype in ("load", "save", "save_as", "clear", "reload"):
            btntype = "{}2".format(loadtype)
            logger.debug("Adding button: '%s'", btntype)

            loader, kwargs = self._loader_and_kwargs(loadtype)
            if loadtype == "load":
                kwargs["current_tab"] = True
            cmd = getattr(self._config.tasks, loader)
            btn = ttk.Button(
                frame,
                image=get_images().icons[btntype],
                command=lambda fn=cmd, kw=kwargs: fn(**kw))
            btn.pack(side=tk.LEFT, anchor=tk.W)
            hlp = self.set_help(btntype)
            Tooltip(btn, text=hlp, wraplength=200)

    @staticmethod
    def _loader_and_kwargs(btntype):
        if btntype == "save":
            loader = btntype
            kwargs = dict(save_as=False)
        elif btntype == "save_as":
            loader = "save"
            kwargs = dict(save_as=True)
        else:
            loader = btntype
            kwargs = dict()
        logger.debug("btntype: %s, loader: %s, kwargs: %s", btntype, loader, kwargs)
        return loader, kwargs

    def _settings_btns(self):
        # pylint: disable=cell-var-from-loop
        frame = ttk.Frame(self._btn_frame)
        frame.pack(side=tk.LEFT, anchor=tk.W, expand=False, padx=2)
        root = get_config().root
        for name in _CONFIG_FILES:
            config = _CONFIGS[name]
            btntype = "settings_{}".format(name)
            btntype = btntype if btntype in get_images().icons else "settings"
            logger.debug("Adding button: '%s'", btntype)
            btn = ttk.Button(
                frame,
                image=get_images().icons[btntype],
                command=lambda conf=(name, config), root=root: popup_config(conf, root))
            btn.pack(side=tk.LEFT, anchor=tk.W)
            hlp = "Configure {} settings...".format(name.title())
            Tooltip(btn, text=hlp, wraplength=200)

    @staticmethod
    def set_help(btntype):
        """ Set the helptext for option buttons """
        logger.debug("Setting help")
        hlp = ""
        task = "currently selected Task" if btntype[-1] == "2" else "Project"
        if btntype.startswith("reload"):
            hlp = "Reload {} from disk".format(task)
        if btntype == "new":
            hlp = "Create a new {}...".format(task)
        if btntype.startswith("clear"):
            hlp = "Reset {} to default".format(task)
        elif btntype.startswith("save") and "_" not in btntype:
            hlp = "Save {}".format(task)
        elif btntype.startswith("save_as"):
            hlp = "Save {} as...".format(task)
        elif btntype.startswith("load"):
            msg = task
            if msg.endswith("Task"):
                msg += " from a task or project file"
            hlp = "Load {}...".format(msg)
        return hlp

    def _group_separator(self):
        separator = ttk.Separator(self._btn_frame, orient="vertical")
        separator.pack(padx=(2, 1), fill=tk.Y, side=tk.LEFT)

    def _section_separator(self):
        frame = ttk.Frame(self)
        frame.pack(side=tk.BOTTOM, fill=tk.X)
        separator = ttk.Separator(frame, orient="horizontal")
        separator.pack(fill=tk.X, side=tk.LEFT, expand=True)
