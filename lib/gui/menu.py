#!/usr/bin python3
""" The Menu Bars for faceswap GUI """

import logging
import os
import sys
import tkinter as tk

from importlib import import_module

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
        self.build_file_menu()
        self.build_edit_menu()
        logger.debug("Initialized %s", self.__class__.__name__)

    def build_file_menu(self):
        """ Add the file menu to the menu bar """
        logger.debug("Building File menu")
        config = get_config()

        file_menu = tk.Menu(self, tearoff=0)
        recent_menu = self.build_recent_menu(file_menu, config)
        
        file_menu.add_command(label="Load full config...", underline=0, command=config.load)
        file_menu.add_command(label="Save full config...", underline=0, command=config.save)
        file_menu.add_separator()
        file_menu.add_cascade(label="Open recent", underline=6,  menu=recent_menu)
        file_menu.add_separator()
        file_menu.add_command(
            label="Reset all to default", underline=0, command=config.cli_opts.reset)
        file_menu.add_command(label="Clear all", underline=0, command=config.cli_opts.clear)
        file_menu.add_separator()
        file_menu.add_command(label="Quit", underline=0, command=self.root.close_app)
        self.add_cascade(label="File", menu=file_menu, underline=0)
        logger.debug("Built File menu")

    def build_recent_menu(self, file_menu, config):
        """ Load recent files into menu bar """
        logger.debug("Building Recent Files menu")
        recent_menu = tk.Menu(file_menu, tearoff=False)
        serializer = JSONSerializer
        menu_file = os.path.join(config.pathcache, ".recent.json")
        if not os.path.isfile(menu_file):
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
            recent_menu.add_command(
                label="{} ({})".format(filename, lbl_command.title()),
                command=lambda fnm=filename, cmd=command: config.load_recent(fnm, cmd))
        recent_menu.add_separator()
        recent_menu.add_command(
            label="Clear recent files",
            underline=0,
            command = lambda srl=serializer, mnu=menu_file: self.clear_recent_files(srl, mnu))
        logger.debug("Built Recent Files menu")
        return recent_menu

    @staticmethod
    def clear_recent_files(serializer, menu_file):
        """ Creates or clears recent file list """
        logger.debug("clearing recent files list: '%s'", menu_file)
        recent_files = serializer.marshal(list())
        with open(menu_file, "wb") as out:
            out.write(recent_files.encode("utf-8"))

    def build_edit_menu(self):
        """ Add the edit menu to the menu bar """
        logger.debug("Building Edit menu")
        edit_menu = tk.Menu(self, tearoff=0)

        configs = self.scan_for_configs()
        for name in sorted(list(configs.keys())):
            label = "Configure {} Plugins...".format(name.title())
            config = configs[name]
            edit_menu.add_command(
                label=label,
                underline=10,
                command=lambda conf=(name, config), root=self.root: popup_config(conf, root))
        self.add_cascade(label="Edit", menu=edit_menu, underline=0)
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
