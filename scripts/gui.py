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
        self.set_fonts()
        self.set_styles()
        self.set_geometry()

        self.wrapper = ProcessWrapper(pathscript)
        self.objects = dict()

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
        initialize_config(self, cliopts, scaling_factor, pathcache, statusbar, session)
        initialize_images()

    @staticmethod
    def set_fonts():
        """ Set global default font """
        tk.font.nametofont("TkFixedFont").configure(size=get_config().default_font[1])
        for font in ("TkDefaultFont", "TkHeadingFont", "TkMenuFont"):
            tk.font.nametofont(font).configure(family=get_config().default_font[0],
                                               size=get_config().default_font[1])

    @staticmethod
    def set_styles():
        """ Set global custom styles """
        gui_style = ttk.Style()
        gui_style.configure('TLabelframe.Label', foreground="#0046D5", relief=tk.SOLID)

    def get_scaling(self):
        """ Get the display DPI """
        dpi = self.winfo_fpixels("1i")
        scaling = dpi / 72.0
        logger.debug("dpi: %s, scaling: %s'", dpi, scaling)
        return scaling

    def set_geometry(self):
        """ Set GUI geometry """
        fullscreen = get_config().user_config_dict["fullscreen"]
        scaling_factor = get_config().scaling_factor

        if fullscreen:
            initial_dimensions = (self.winfo_screenwidth(), self.winfo_screenheight())
        else:
            initial_dimensions = (round(1200 * scaling_factor), round(640 * scaling_factor))

        if fullscreen and sys.platform == "win32":
            self.state('zoomed')
        elif fullscreen:
            self.attributes('-zoomed', True)
        else:
            self.geometry("{}x{}+80+80".format(str(initial_dimensions[0]),
                                               str(initial_dimensions[1])))
        logger.debug("Geometry: %sx%s", *initial_dimensions)

    def build_gui(self, debug_console):
        """ Build the GUI """
        logger.debug("Building GUI")
        self.title("Faceswap.py")
        self.tk.call('wm', 'iconphoto', self._w, get_images().icons["favicon"])
        self.configure(menu=MainMenuBar(self))

        self.add_containers()

        self.objects["command"] = CommandNotebook(self.objects["containers"]["top"])
        self.objects["display"] = DisplayNotebook(self.objects["containers"]["top"])
        self.objects["console"] = ConsoleOut(self.objects["containers"]["bottom"], debug_console)
        self.set_initial_focus()
        self.set_layout()
        logger.debug("Built GUI")

    def add_containers(self):
        """ Add the paned window containers that
            hold each main area of the gui """
        logger.debug("Adding containers")
        maincontainer = tk.PanedWindow(self,
                                       sashrelief=tk.RIDGE,
                                       sashwidth=4,
                                       sashpad=8,
                                       orient=tk.VERTICAL,
                                       name="pw_main")
        maincontainer.pack(fill=tk.BOTH, expand=True)

        topcontainer = tk.PanedWindow(maincontainer,
                                      sashrelief=tk.RIDGE,
                                      sashwidth=4,
                                      sashpad=8,
                                      orient=tk.HORIZONTAL,
                                      name="pw_top")
        maincontainer.add(topcontainer)

        bottomcontainer = ttk.Frame(maincontainer, name="frame_bottom")
        maincontainer.add(bottomcontainer)
        self.objects["containers"] = dict(main=maincontainer,
                                          top=topcontainer,
                                          bottom=bottomcontainer)

        logger.debug("Added containers")

    @staticmethod
    def set_initial_focus():
        """ Set the tab focus from settings """
        config = get_config()
        tab = config.user_config_dict["tab"]
        logger.debug("Setting focus for tab: %s", tab)
        tabs = config.command_tabs
        if tab in tabs:
            config.command_notebook.select(tabs[tab])
        else:
            tool_tabs = config.tools_command_tabs
            if tab in tool_tabs:
                config.command_notebook.select(tabs["tools"])
                config.command_notebook.tools_notebook.select(tool_tabs[tab])
        logger.debug("Focus set to: %s", tab)

    def set_layout(self):
        """ Set initial layout """
        self.update_idletasks()
        root = get_config().root
        config = get_config().user_config_dict
        r_width = root.winfo_width()
        r_height = root.winfo_height()
        w_ratio = config["options_panel_width"] / 100.0
        h_ratio = 1 - (config["console_panel_height"] / 100.0)
        width = round(r_width * w_ratio)
        height = round(r_height * h_ratio)
        logger.debug("Setting Initial Layout: (root_width: %s, root_height: %s, width_ratio: %s, "
                     "height_ratio: %s, width: %s, height: %s", r_width, r_height, w_ratio,
                     h_ratio, width, height)
        self.objects["containers"]["top"].sash_place(0, width, 1)
        self.objects["containers"]["main"].sash_place(0, 1, height)
        self.update_idletasks()

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
