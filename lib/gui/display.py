#!/usr/bin python3
""" Display Frame of the Faceswap GUI

    What is displayed in the Display Frame varies
    depending on what tasked is being run """

import logging
import tkinter as tk
from tkinter import ttk

from .display_analysis import Analysis
from .display_command import GraphDisplay, PreviewExtract, PreviewTrain
from .utils import get_config

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class DisplayNotebook(ttk.Notebook):  # pylint: disable=too-many-ancestors
    """ The display tabs """

    def __init__(self, parent):
        logger.debug("Initializing %s", self.__class__.__name__)
        super().__init__(parent)
        parent.add(self)
        tk_vars = get_config().tk_vars
        self.wrapper_var = tk_vars["display"]
        self.runningtask = tk_vars["runningtask"]

        self.set_wrapper_var_trace()
        self.add_static_tabs()
        self.static_tabs = [child for child in self.tabs()]
        logger.debug("Initialized %s", self.__class__.__name__)

    def set_wrapper_var_trace(self):
        """ Set the trigger actions for the display vars
            when they have been triggered in the Process Wrapper """
        logger.debug("Setting wrapper var trace")
        self.wrapper_var.trace("w", self.update_displaybook)

    def add_static_tabs(self):
        """ Add tabs that are permanently available """
        logger.debug("Adding static tabs")
        for tab in ("job queue", "analysis"):
            if tab == "job queue":
                continue    # Not yet implemented
            if tab == "analysis":
                helptext = {"stats":
                            "Summary statistics for each training session"}
                frame = Analysis(self, tab, helptext)
            else:
                frame = self.add_frame()
                self.add(frame, text=tab.title())

    def add_frame(self):
        """ Add a single frame for holding tab's contents """
        logger.debug("Adding frame")
        frame = ttk.Frame(self)
        frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        return frame

    def command_display(self, command):
        """ Select what to display based on incoming
            command """
        build_tabs = getattr(self, "{}_tabs".format(command))
        build_tabs()

    def extract_tabs(self, command="extract"):
        """ Build the extract tabs """
        logger.debug("Build extract tabs")
        helptext = ("Updates preview from output every 5 "
                    "seconds to limit disk contention")
        PreviewExtract(self, "preview", helptext, 5000, command)
        logger.debug("Built extract tabs")

    def train_tabs(self):
        """ Build the train tabs """
        logger.debug("Build train tabs")
        for tab in ("graph", "preview"):
            if tab == "graph":
                helptext = "Graph showing Loss vs Iterations"
                GraphDisplay(self, "graph", helptext, 5000)
            elif tab == "preview":
                helptext = "Training preview. Updated on every save iteration"
                PreviewTrain(self, "preview", helptext, 1000)
        logger.debug("Built train tabs")

    def convert_tabs(self):
        """ Build the convert tabs
            Currently identical to Extract, so just call that """
        logger.debug("Build convert tabs")
        self.extract_tabs(command="convert")
        logger.debug("Built convert tabs")

    def remove_tabs(self):
        """ Remove all command specific tabs """
        for child in self.tabs():
            if child in self.static_tabs:
                continue
            logger.debug("removing child: %s", child)
            child_name = child.split(".")[-1]
            child_object = self.children[child_name]  # returns the OptionalDisplayPage object
            child_object.close()  # Call the OptionalDisplayPage close() method
            self.forget(child)

    def update_displaybook(self, *args):  # pylint: disable=unused-argument
        """ Set the display tabs based on executing task """
        command = self.wrapper_var.get()
        self.remove_tabs()
        if not command or command not in ("extract", "train", "convert"):
            return
        self.command_display(command)
