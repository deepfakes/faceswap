#!/usr/bin python3
""" The optional GUI for faceswap """

import logging
import os
import sys
import tkinter as tk
from tkinter import messagebox, ttk

from importlib import import_module

from lib.gui import (CliOptions, CommandNotebook, Config, ConsoleOut,
                     CurrentSession, DisplayNotebook, Images, ProcessWrapper,
                     StatusBar, popup_config)

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class FaceswapGui(tk.Tk):
    """ The Graphical User Interface """

    def __init__(self, pathscript):
        logger.debug("Initializing %s", self.__class__.__name__)
        tk.Tk.__init__(self)
        self.scaling_factor = self.get_scaling()
        self.set_geometry()

        pathcache = os.path.join(pathscript, "lib", "gui", ".cache")
        self.images = Images(pathcache)
        self.cliopts = CliOptions()
        self.session = CurrentSession()
        statusbar = StatusBar(self)
        self.wrapper = ProcessWrapper(statusbar,
                                      self.session,
                                      pathscript,
                                      self.cliopts)

        self.images.delete_preview()
        self.protocol("WM_DELETE_WINDOW", self.close_app)
        logger.debug("Initialized %s", self.__class__.__name__)

    def get_scaling(self):
        """ Get the display DPI """
        dpi = self.winfo_fpixels("1i")
        scaling = dpi / 72.0
        logger.debug("dpi: %s, scaling: %s'", dpi, scaling)
        return scaling

    def set_geometry(self):
        """ Set GUI geometry """
        self.tk.call("tk", "scaling", self.scaling_factor)
        width = int(1200 * self.scaling_factor)
        height = int(640 * self.scaling_factor)
        logger.debug("Geometry: %sx%s", width, height)
        self.geometry("{}x{}+80+80".format(str(width), str(height)))

    def build_gui(self, debug_console):
        """ Build the GUI """
        logger.debug("Building GUI")
        self.title("Faceswap.py")
        self.menu()

        topcontainer, bottomcontainer = self.add_containers()

        CommandNotebook(topcontainer,
                        self.cliopts,
                        self.wrapper.tk_vars,
                        self.scaling_factor)
        DisplayNotebook(topcontainer,
                        self.session,
                        self.wrapper.tk_vars,
                        self.scaling_factor)
        ConsoleOut(bottomcontainer, debug_console, self.wrapper.tk_vars)
        logger.debug("Built GUI")

    def menu(self):
        """ Menu bar for loading and saving configs """
        logger.debug("Adding menu bar")
        menu_bar = tk.Menu(self)
        self.build_file_menu(menu_bar)
        self.build_edit_menu(menu_bar)
        self.config(menu=menu_bar)
        logger.debug("Added menu bar")

    def build_file_menu(self, menu_bar):
        """ Add the file menu to the menu bar """
        logger.debug("Building File menu")
        file_menu = tk.Menu(menu_bar, tearoff=0)
        config = Config(self.cliopts, self.wrapper.tk_vars)

        file_menu.add_command(label="Load full config...", underline=0, command=config.load)
        file_menu.add_command(label="Save full config...", underline=0, command=config.save)
        file_menu.add_separator()
        file_menu.add_command(
            label="Reset all to default", underline=0, command=self.cliopts.reset)
        file_menu.add_command(label="Clear all", underline=0, command=self.cliopts.clear)
        file_menu.add_separator()
        file_menu.add_command(label="Quit", underline=0, command=self.close_app)
        menu_bar.add_cascade(label="File", menu=file_menu, underline=0)
        logger.debug("Built File menu")

    def build_edit_menu(self, menu_bar):
        """ Add the edit menu to the menu bar """
        logger.debug("Building Edit menu")
        edit_menu = tk.Menu(menu_bar, tearoff=0)

        configs = self.scan_for_configs()
        for name in sorted(list(configs.keys())):
            label = "Configure {} Plugins...".format(name.title())
            config = configs[name]
            edit_menu.add_command(
                label=label,
                underline=10,
                command=lambda conf=(name, config), root=self: popup_config(conf, root))
        menu_bar.add_cascade(label="Edit", menu=edit_menu, underline=0)
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

    def add_containers(self):
        """ Add the paned window containers that
            hold each main area of the gui """
        logger.debug("Adding containers")
        maincontainer = tk.PanedWindow(self,
                                       sashrelief=tk.RAISED,
                                       orient=tk.VERTICAL)
        maincontainer.pack(fill=tk.BOTH, expand=True)

        topcontainer = tk.PanedWindow(maincontainer,
                                      sashrelief=tk.RAISED,
                                      orient=tk.HORIZONTAL)
        maincontainer.add(topcontainer)

        bottomcontainer = ttk.Frame(maincontainer, height=150)
        maincontainer.add(bottomcontainer)

        logger.debug("Added containers")
        return topcontainer, bottomcontainer

    def close_app(self):
        """ Close Python. This is here because the graph
            animation function continues to run even when
            tkinter has gone away """
        logger.debug("Close Requested")
        confirm = messagebox.askokcancel
        confirmtxt = "Processes are still running. Are you sure...?"
        if (self.wrapper.tk_vars["runningtask"].get()
                and not confirm("Close", confirmtxt)):
            logger.debug("Close Cancelled")
            return
        if self.wrapper.tk_vars["runningtask"].get():
            self.wrapper.task.terminate()
        self.images.delete_preview()
        self.quit()
        logger.debug("Closed GUI")
        exit()


class Gui():  # pylint: disable=too-few-public-methods
    """ The GUI process. """
    def __init__(self, arguments):
        cmd = sys.argv[0]
        pathscript = os.path.realpath(os.path.dirname(cmd))
        self.args = arguments
        self.root = FaceswapGui(pathscript)

    def process(self):
        """ Builds the GUI """
        self.root.build_gui(self.args.debug)
        self.root.mainloop()
