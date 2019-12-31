#!/usr/bin python3
""" The optional GUI for faceswap """

import logging
import sys
import tkinter as tk
from tkinter import messagebox, ttk

from lib.gui import (TaskBar, CliOptions, CommandNotebook, ConsoleOut, Session, DisplayNotebook,
                     get_images, initialize_images, initialize_config, LastSession,
                     MainMenuBar, ProcessWrapper, StatusBar)

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class FaceswapGui(tk.Tk):
    """ The Graphical User Interface """

    def __init__(self, debug):
        logger.debug("Initializing %s", self.__class__.__name__)
        super().__init__()

        self._init_args = dict(debug=debug)
        self._config = self.initialize_globals()
        self.set_fonts()
        self.set_styles()
        self._config.set_geometry(1200, 640, self._config.user_config_dict["fullscreen"])

        self.wrapper = ProcessWrapper()
        self.objects = dict()

        get_images().delete_preview()
        self.protocol("WM_DELETE_WINDOW", self.close_app)
        self.build_gui()
        self._last_session = LastSession(self._config)
        logger.debug("Initialized %s", self.__class__.__name__)

    def initialize_globals(self):
        """ Initialize config and images global constants """
        cliopts = CliOptions()
        statusbar = StatusBar(self)
        session = Session()
        config = initialize_config(self, cliopts, statusbar, session)
        initialize_images()
        return config

    def set_fonts(self):
        """ Set global default font """
        tk.font.nametofont("TkFixedFont").configure(size=self._config.default_font[1])
        for font in ("TkDefaultFont", "TkHeadingFont", "TkMenuFont"):
            tk.font.nametofont(font).configure(family=self._config.default_font[0],
                                               size=self._config.default_font[1])

    @staticmethod
    def set_styles():
        """ Set global custom styles """
        gui_style = ttk.Style()
        gui_style.configure('TLabelframe.Label', foreground="#0046D5", relief=tk.SOLID)

    def build_gui(self, rebuild=False):
        """ Build the GUI """
        logger.debug("Building GUI")
        if not rebuild:
            self.tk.call('wm', 'iconphoto', self._w, get_images().icons["favicon"])
            self.configure(menu=MainMenuBar(self))

        if rebuild:
            objects = list(self.objects.keys())
            for obj in objects:
                self.objects[obj].destroy()
                del self.objects[obj]

        self.objects["taskbar"] = TaskBar(self)
        self.add_containers()

        self.objects["command"] = CommandNotebook(self.objects["container_top"])
        self.objects["display"] = DisplayNotebook(self.objects["container_top"])
        self.objects["console"] = ConsoleOut(self.objects["container_bottom"],
                                             self._init_args["debug"])
        self.set_initial_focus()
        self.set_layout()
        self._config.set_default_options()
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
        self.objects["container_main"] = maincontainer
        self.objects["container_top"] = topcontainer
        self.objects["container_bottom"] = bottomcontainer

        logger.debug("Added containers")

    def set_initial_focus(self):
        """ Set the tab focus from settings """
        tab = self._config.user_config_dict["tab"]
        logger.debug("Setting focus for tab: %s", tab)
        self._config.set_active_tab_by_name(tab)
        logger.debug("Focus set to: %s", tab)

    def set_layout(self):
        """ Set initial layout """
        self.update_idletasks()
        config_opts = self._config.user_config_dict
        r_width = self.winfo_width()
        r_height = self.winfo_height()
        w_ratio = config_opts["options_panel_width"] / 100.0
        h_ratio = 1 - (config_opts["console_panel_height"] / 100.0)
        width = round(r_width * w_ratio)
        height = round(r_height * h_ratio)
        logger.debug("Setting Initial Layout: (root_width: %s, root_height: %s, width_ratio: %s, "
                     "height_ratio: %s, width: %s, height: %s", r_width, r_height, w_ratio,
                     h_ratio, width, height)
        self.objects["container_top"].sash_place(0, width, 1)
        self.objects["container_main"].sash_place(0, 1, height)
        self.update_idletasks()

    def rebuild(self):
        """ Rebuild the GUI on config change """
        logger.debug("Redrawing GUI")
        session_state = self._last_session.to_dict()
        self._config.refresh_config()
        get_images().__init__()
        self.set_fonts()
        self.build_gui(rebuild=True)
        if session_state is not None:
            self._last_session.from_dict(session_state)
        logger.debug("GUI Redrawn")

    def close_app(self, *args):  # pylint: disable=unused-argument
        """ Close Python. This is here because the graph
            animation function continues to run even when
            tkinter has gone away """
        logger.debug("Close Requested")

        if not self._confirm_close_on_running_task():
            return
        if not self._config.project.confirm_close():
            return

        if self._config.tk_vars["runningtask"].get():
            self.wrapper.task.terminate()

        self._last_session.save()
        get_images().delete_preview()
        self.quit()
        logger.debug("Closed GUI")
        sys.exit(0)

    def _confirm_close_on_running_task(self):
        """ Pop a confirmation box to close the GUI if a task is running

        Returns
        -------
        bool: ``True`` if user confirms close, ``False`` if user cancels close
        """
        if not self._config.tk_vars["runningtask"].get():
            logger.debug("No tasks currently running")
            return True

        confirmtxt = "Processes are still running.\n\nAre you sure you want to exit?"
        if not messagebox.askokcancel("Close", confirmtxt, default="cancel", icon="warning"):
            logger.debug("Close Cancelled")
            return True
        logger.debug("Close confirmed")
        return False


class Gui():  # pylint: disable=too-few-public-methods
    """ The GUI process. """
    def __init__(self, arguments):
        self.root = FaceswapGui(arguments.debug)

    def process(self):
        """ Builds the GUI """
        self.root.mainloop()
