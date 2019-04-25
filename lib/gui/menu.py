#!/usr/bin python3
""" The Menu Bars for faceswap GUI """

import locale
import logging
import os
import sys
import tkinter as tk

from importlib import import_module
from subprocess import Popen, PIPE, STDOUT

from lib.Serializer import JSONSerializer

from .utils import get_config
from .popup_configure import popup_config


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class MainMenuBar(tk.Menu):
    """ GUI Main Menu Bar """
    def __init__(self, master=None):
        logger.debug("Initializing %s", self.__class__.__name__)
        super().__init__(master)
        self.root = master
        self.config = get_config()

        self.file_menu = tk.Menu(self, tearoff=0)
        self.recent_menu = tk.Menu(self.file_menu, tearoff=0, postcommand=self.refresh_recent_menu)
        self.edit_menu = tk.Menu(self, tearoff=0)
        self.tools_menu = tk.Menu(self, tearoff=0)

        self.build_file_menu()
        self.build_edit_menu()
        self.build_tools_menu()
        logger.debug("Initialized %s", self.__class__.__name__)

    def build_file_menu(self):
        """ Add the file menu to the menu bar """
        logger.debug("Building File menu")
        self.file_menu.add_command(
            label="Load full config...", underline=0, command=self.config.load)
        self.file_menu.add_command(
            label="Save full config...", underline=0, command=self.config.save)
        self.file_menu.add_separator()
        self.file_menu.add_cascade(label="Open recent", underline=6, menu=self.recent_menu)
        self.file_menu.add_separator()
        self.file_menu.add_command(
            label="Reset all to default", underline=0, command=self.config.cli_opts.reset)
        self.file_menu.add_command(
            label="Clear all", underline=0, command=self.config.cli_opts.clear)
        self.file_menu.add_separator()
        self.file_menu.add_command(label="Quit", underline=0, command=self.root.close_app)
        self.add_cascade(label="File", menu=self.file_menu, underline=0)
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
            if not os.path.isfile(filename):
                logger.debug("File does not exist")
                continue
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

    def build_tools_menu(self):
        """ Add the file menu to the menu bar """
        logger.debug("Building Tools menu")
        self.tools_menu.add_command(
            label="Check for updates...", underline=0, command=self.check_updates)
        self.tools_menu.add_command(
            label="Output System Information", underline=0, command=self.output_sysinfo)
        self.add_cascade(label="Tools", menu=self.tools_menu, underline=0)
        logger.debug("Built Tools menu")

    @staticmethod
    def output_sysinfo():
        """ Output system information to console """
        get_config().tk_vars["consoleclear"].set(True)
        from lib.sysinfo import SysInfo
        print(SysInfo().full_info())

    @staticmethod
    def check_updates():
        """ Check for updates and clone repo """
        get_config().tk_vars["consoleclear"].set(True)
        encoding = locale.getpreferredencoding()

        # Do the check
        cmd = Popen("git remote update && git status -uno", shell=True, stdout=PIPE, stderr=STDOUT)
        stdout, _ = cmd.communicate()
        retcode = cmd.poll()
        logger.debug("Check output: %s", stdout.decode(encoding))
        logger.debug("Check returncode: %s", retcode)
        if retcode != 0:
            print("Git is not installed or you are not running a cloned repo. "
                  "Unable to check for updates")
            return
        chk = stdout.decode(encoding).splitlines()

        update = False
        for line in chk:
            if line.lower().startswith("your branch is ahead"):
                print("Your branch is ahead of the remote repo. Not updating")
                break
            if line.lower().startswith("your branch is up to date"):
                print("Faceswap is up to date.")
                break
            if line.lower().startswith("your branch is behind"):
                update = True
                break
            if "have diverged" in line.lower():
                print("Your branch has diverged from the remote repo. Not updating")
                break
        logger.debug("update: %s", update)
        if not update:
            return

        # Do the update
        print("A new version is available. Updating...")
        cmd = Popen("git pull", shell=True, stdout=PIPE, stderr=STDOUT, bufsize=1)
        while True:
            output = cmd.stdout.readline().decode(encoding)
            if output == "" and cmd.poll() is not None:
                break
            if output:
                logger.debug("Update output: '%s'", output.strip())
                print(output.strip())
        retcode = cmd.poll()
        logger.debug("Update returncode: %s", retcode)
        if retcode != 0:
            print("An error occurred during update. return code: {}".format(retcode))
            return

        print("\nPlease restart Faceswap to complete the update.")
