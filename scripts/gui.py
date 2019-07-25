#!/usr/bin python3
""" The optional GUI for faceswap """

import logging
import os
import sys
import tkinter as tk
from tkinter import messagebox, ttk

from lib.gui import (CliOptions, CommandNotebook, ConsoleOut, Session, DisplayNotebook,
                     get_config, get_images, initialize_images, initialize_config, MainMenuBar,
                     ProcessWrapper, StatusBar)

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class FaceswapGui(tk.Tk):
    """ The Graphical User Interface """

    def __init__(self, pathscript):
        logger.debug("Initializing %s", self.__class__.__name__)
        super().__init__()

        self.initialize_globals(pathscript)
        self.set_geometry()
        self.wrapper = ProcessWrapper(pathscript)

        get_images().delete_preview()
        self.protocol("WM_DELETE_WINDOW", self.close_app)
        logger.debug("Initialized %s", self.__class__.__name__)

    def initialize_globals(self, pathscript):
        """ Initialize config and images global constants """
        cliopts = CliOptions()
        scaling_factor = self.get_scaling()
        pathcache = os.path.join(pathscript, "lib", "gui", ".cache")
        statusbar = StatusBar(self)
        session = Session()
        initialize_config(cliopts, scaling_factor, pathcache, statusbar, session)
        initialize_images()

    def get_scaling(self):
        """ Get the display DPI """
        dpi = self.winfo_fpixels("1i")
        scaling = dpi / 72.0
        logger.debug("dpi: %s, scaling: %s'", dpi, scaling)
        return scaling

    def set_geometry(self):
        """ Set GUI geometry """
        scaling_factor = get_config().scaling_factor
        self.tk.call("tk", "scaling", scaling_factor)
        width = int(1200 * scaling_factor)
        height = int(640 * scaling_factor)
        logger.debug("Geometry: %sx%s", width, height)
        self.geometry("{}x{}+80+80".format(str(width), str(height)))

    def build_gui(self, debug_console):
        """ Build the GUI """
        logger.debug("Building GUI")
        self.title("Faceswap.py")
        self.tk.call('wm', 'iconphoto', self._w, get_images().icons["favicon"])
        self.configure(menu=MainMenuBar(self))

        topcontainer, bottomcontainer = self.add_containers()

        CommandNotebook(topcontainer)
        DisplayNotebook(topcontainer)
        ConsoleOut(bottomcontainer, debug_console)
        logger.debug("Built GUI")

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
        tk_vars = get_config().tk_vars
        if (tk_vars["runningtask"].get()
                and not confirm("Close", confirmtxt)):
            logger.debug("Close Cancelled")
            return
        if tk_vars["runningtask"].get():
            self.wrapper.task.terminate()
        get_images().delete_preview()
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
