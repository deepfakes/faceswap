#!/usr/bin python3
""" The Menu Bars for faceswap GUI """

import locale
import logging
import os
import sys
import tkinter as tk

from importlib import import_module
from subprocess import Popen, PIPE, STDOUT

from lib.multithreading import MultiThread
from lib.Serializer import JSONSerializer

import update_deps
from .utils import get_config
from .popup_configure import popup_config


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class MainMenuBar(tk.Menu):  # pylint:disable=too-many-ancestors
    """ GUI Main Menu Bar """
    def __init__(self, master=None):
        logger.debug("Initializing %s", self.__class__.__name__)
        super().__init__(master)
        self.root = master

        self.file_menu = FileMenu(self)
        self.edit_menu = tk.Menu(self, tearoff=0)
        self.tools_menu = ToolsMenu(self)

        self.add_cascade(label="File", menu=self.file_menu, underline=0)
        self.build_edit_menu()
        self.add_cascade(label="Tools", menu=self.tools_menu, underline=0)
        logger.debug("Initialized %s", self.__class__.__name__)

    def build_edit_menu(self):
        """ Add the edit menu to the menu bar """
        logger.debug("Building Edit menu")
        configs = self.scan_for_configs()
        for name in sorted(list(configs.keys())):
            label = "Configure {} Plugins...".format(name.title())
            config = configs[name]
            self.edit_menu.add_command(
                label=label,
                underline=10,
                command=lambda conf=(name, config), root=self.root: popup_config(conf, root))
        self.add_cascade(label="Edit", menu=self.edit_menu, underline=0)
        logger.debug("Built Edit menu")

    def scan_for_configs(self):
        """ Scan for config.ini file locations """
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


class FileMenu(tk.Menu):  # pylint:disable=too-many-ancestors
    """ File menu items and functions """
    def __init__(self, parent):
        logger.debug("Initializing %s", self.__class__.__name__)
        super().__init__(parent, tearoff=0)
        self.root = parent.root
        self.config = get_config()
        self.recent_menu = tk.Menu(self, tearoff=0, postcommand=self.refresh_recent_menu)
        self.build()
        logger.debug("Initialized %s", self.__class__.__name__)

    def build(self):
        """ Add the file menu to the menu bar """
        logger.debug("Building File menu")
        self.add_command(label="Load full config...", underline=0, command=self.config.load)
        self.add_command(label="Save full config...", underline=0, command=self.config.save)
        self.add_separator()
        self.add_cascade(label="Open recent", underline=6, menu=self.recent_menu)
        self.add_separator()
        self.add_command(label="Reset all to default",
                         underline=0,
                         command=self.config.cli_opts.reset)
        self.add_command(label="Clear all", underline=0, command=self.config.cli_opts.clear)
        self.add_separator()
        self.add_command(label="Quit", underline=0, command=self.root.close_app)
        logger.debug("Built File menu")

    def build_recent_menu(self):
        """ Load recent files into menu bar """
        logger.debug("Building Recent Files menu")
        serializer = JSONSerializer
        menu_file = os.path.join(self.config.pathcache, ".recent.json")
        if not os.path.isfile(menu_file) or os.path.getsize(menu_file) == 0:
            self.clear_recent_files(serializer, menu_file)
        with open(menu_file, "rb") as inp:
            recent_files = serializer.unmarshal(inp.read().decode("utf-8"))
            logger.debug("Loaded recent files: %s", recent_files)
        for recent_item in recent_files:
            filename, command = recent_item
            logger.debug("processing: ('%s', %s)", filename, command)
            lbl_command = command if command else "All"
            self.recent_menu.add_command(
                label="{} ({})".format(filename, lbl_command.title()),
                command=lambda fnm=filename, cmd=command: self.config.load(cmd, fnm))
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
        recent_files = serializer.marshal(list())
        with open(menu_file, "wb") as out:
            out.write(recent_files.encode("utf-8"))

    def refresh_recent_menu(self):
        """ Refresh recent menu on save/load of files """
        self.recent_menu.delete(0, "end")
        self.build_recent_menu()


class ToolsMenu(tk.Menu):  # pylint:disable=too-many-ancestors
    """ Tools menu items and functions """
    def __init__(self, parent):
        logger.debug("Initializing %s", self.__class__.__name__)
        super().__init__(parent, tearoff=0)
        self.root = parent.root
        self.build()
        logger.debug("Initialized %s", self.__class__.__name__)

    def build(self):
        """ Build the tools menu """
        logger.debug("Building Tools menu")
        self.add_command(label="Check for updates...",
                         underline=0,
                         command=lambda action="update": self.in_thread(action))
        self.add_command(label="Output System Information",
                         underline=0,
                         command=lambda action="output_sysinfo": self.in_thread(action))
        logger.debug("Built Tools menu")

    def in_thread(self, action):
        """ Perform selected action inside a thread """
        logger.debug("Performing tools action: %s", action)
        thread = MultiThread(getattr(self, action), thread_count=1)
        thread.start()
        logger.debug("Performed tools action: %s", action)

    @staticmethod
    def clear_console():
        """ Clear the console window """
        get_config().tk_vars["consoleclear"].set(True)

    def output_sysinfo(self):
        """ Output system information to console """
        logger.debug("Obtaining system information")
        self.root.config(cursor="watch")
        self.clear_console()
        print("Obtaining system information...")
        from lib.sysinfo import sysinfo
        info = sysinfo
        self.clear_console()
        logger.debug("Obtained system information: %s", info)
        print(info)
        self.root.config(cursor="")

    def update(self):
        """ Check for updates and clone repo """
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
    def check_for_updates(encoding):
        """ Check whether an update is required """
        # Do the check
        logger.info("Checking for updates...")
        update = False
        msg = ""
        gitcmd = "git remote update && git status -uno"
        cmd = Popen(gitcmd, shell=True, stdout=PIPE, stderr=STDOUT)
        stdout, _ = cmd.communicate()
        retcode = cmd.poll()
        logger.debug("'%s' output: %s", gitcmd, stdout.decode(encoding))
        logger.debug("'%s' returncode: %s", gitcmd, retcode)
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
                    update = True
                    break
                if "have diverged" in line.lower():
                    msg = "Your branch has diverged from the remote repo. Not updating"
                    break
        if not update:
            logger.info(msg)
        logger.debug("Checked for update. Update required: %s", update)
        return update

    @staticmethod
    def do_update(encoding):
        """ Update Faceswap """
        logger.info("A new version is available. Updating...")
        gitcmd = "git pull"
        cmd = Popen(gitcmd, shell=True, stdout=PIPE, stderr=STDOUT, bufsize=1)
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
