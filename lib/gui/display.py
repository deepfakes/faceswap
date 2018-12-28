#!/usr/bin python3
""" Display Frame of the Faceswap GUI

    What is displayed in the Display Frame varies
    depending on what tasked is being run """

import tkinter as tk
from tkinter import ttk

from .display_analysis import Analysis
from .display_command import GraphDisplay, PreviewExtract, PreviewTrain


class DisplayNotebook(ttk.Notebook):
    """ The display tabs """

    def __init__(self, parent, session, tk_vars, scaling_factor):
        ttk.Notebook.__init__(self, parent, width=780)
        parent.add(self)

        self.wrapper_var = tk_vars["display"]
        self.runningtask = tk_vars["runningtask"]
        self.session = session

        self.set_wrapper_var_trace()
        self.add_static_tabs(scaling_factor)
        self.static_tabs = [child for child in self.tabs()]

    def set_wrapper_var_trace(self):
        """ Set the trigger actions for the display vars
            when they have been triggered in the Process Wrapper """
        self.wrapper_var.trace("w", self.update_displaybook)

    def add_static_tabs(self, scaling_factor):
        """ Add tabs that are permanently available """
        for tab in ("job queue", "analysis"):
            if tab == "job queue":
                continue    # Not yet implemented
            if tab == "analysis":
                helptext = {"stats":
                            "Summary statistics for each training session"}
                frame = Analysis(self, tab, helptext, scaling_factor)
            else:
                frame = self.add_frame()
                self.add(frame, text=tab.title())

    def add_frame(self):
        """ Add a single frame for holding tab's contents """
        frame = ttk.Frame(self)
        frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        return frame

    def command_display(self, command):
        """ Select what to display based on incoming
            command """
        build_tabs = getattr(self, "{}_tabs".format(command))
        build_tabs()

    def extract_tabs(self):
        """ Build the extract tabs """
        helptext = ("Updates preview from output every 5 "
                    "seconds to limit disk contention")
        PreviewExtract(self, "preview", helptext, 5000)

    def train_tabs(self):
        """ Build the train tabs """
        for tab in ("graph", "preview"):
            if tab == "graph":
                helptext = "Graph showing Loss vs Iterations"
                GraphDisplay(self, "graph", helptext, 5000)
            elif tab == "preview":
                helptext = "Training preview. Updated on every save iteration"
                PreviewTrain(self, "preview", helptext, 5000)

    def convert_tabs(self):
        """ Build the convert tabs
            Currently identical to Extract, so just call that """
        self.extract_tabs()

    def remove_tabs(self):
        """ Remove all command specific tabs """
        for child in self.tabs():
            if child not in self.static_tabs:
                self.forget(child)

    def update_displaybook(self, *args):
        """ Set the display tabs based on executing task """
        command = self.wrapper_var.get()
        self.remove_tabs()
        if not command or command not in ("extract", "train", "convert"):
            return
        self.command_display(command)
