#!/usr/bin python3
""" Display Frame of the Faceswap GUI

This is the large right hand area of the GUI. At default, the Analysis tab is always displayed
here. Further optional tabs will also be displayed depending on the currently executing Faceswap
task. """

import logging
import gettext
import tkinter as tk
from tkinter import ttk

from lib.logger import parse_class_init

from .display_analysis import Analysis
from .display_command import GraphDisplay, PreviewExtract, PreviewTrain
from .utils import get_config

logger = logging.getLogger(__name__)

# LOCALES
_LANG = gettext.translation("gui.tooltips", localedir="locales", fallback=True)
_ = _LANG.gettext


class DisplayNotebook(ttk.Notebook):  # pylint:disable=too-many-ancestors
    """ The tkinter Notebook that holds the display items.

    Parameters
    ----------
    parent: :class:`tk.PanedWindow`
        The paned window that holds the Display Notebook
    """

    def __init__(self, parent):
        logger.debug(parse_class_init(locals()))
        super().__init__(parent)
        parent.add(self)
        tk_vars = get_config().tk_vars
        self._wrapper_var = tk_vars.display
        self._running_task = tk_vars.running_task

        self._set_wrapper_var_trace()
        self._add_static_tabs()
        # pylint:disable=unnecessary-comprehension
        self._static_tabs = [child for child in self.tabs()]
        self.bind("<<NotebookTabChanged>>", self._on_tab_change)
        logger.debug("Initialized %s", self.__class__.__name__)

    @property
    def running_task(self):
        """ :class:`tkinter.BooleanVar`: The global tkinter variable that indicates whether a
        Faceswap task is currently running or not. """
        return self._running_task

    def _set_wrapper_var_trace(self):
        """ Sets the trigger to update the displayed notebook's pages when the global tkinter
        variable `display` is updated in the :class:`~lib.gui.wrapper.ProcessWrapper`. """
        logger.debug("Setting wrapper var trace")
        self._wrapper_var.trace("w", self._update_displaybook)

    def _add_static_tabs(self):
        """ Add the tabs to the Display Notebook that are permanently displayed.

        Currently this is just the `Analysis` tab.
        """
        logger.debug("Adding static tabs")
        for tab in ("job queue", "analysis"):
            if tab == "job queue":
                continue    # Not yet implemented
            if tab == "analysis":
                helptext = {"stats":
                            _("Summary statistics for each training session")}
                frame = Analysis(self, tab, helptext)
            else:
                frame = self._add_frame()
                self.add(frame, text=tab.title())

    def _add_frame(self):
        """ Add a single frame for holding a static tab's contents.

        Returns
        -------
        ttk.Frame
            The frame, packed into position
        """
        logger.debug("Adding frame")
        frame = ttk.Frame(self)
        frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        return frame

    def _command_display(self, command):
        """ Build the relevant command specific tabs based on the incoming Faceswap command.

        Parameters
        ----------
        command: str
            The Faceswap command that is being executed
        """
        build_tabs = getattr(self, f"_{command}_tabs")
        build_tabs()

    def _extract_tabs(self, command="extract"):
        """ Build the display tabs that are used for Faceswap extract and convert tasks.

        Notes
        -----
        The same display tabs are used for both convert and extract tasks.

        command: [`"extract"`, `"convert"`], optional
            The command that the display tabs are being built for. Default: `"extract"`

        """
        logger.debug("Build extract tabs")
        helptext = _("Preview updates every 5 seconds")
        PreviewExtract(self, "preview", helptext, 5000, command)
        logger.debug("Built extract tabs")

    def _train_tabs(self):
        """ Build the display tabs that are used for the Faceswap train task."""
        logger.debug("Build train tabs")
        for tab in ("graph", "preview"):
            if tab == "graph":
                helptext = _("Graph showing Loss vs Iterations")
                GraphDisplay(self, "graph", helptext, 5000)
            elif tab == "preview":
                helptext = _("Training preview. Updated on every save iteration")
                PreviewTrain(self, "preview", helptext, 1000)
        logger.debug("Built train tabs")

    def _convert_tabs(self):
        """ Build the display tabs that are used for the Faceswap convert task.

        Notes
        -----
        The tabs displayed are the same as used for extract, so :func:`_extract_tabs` is called.
        """
        logger.debug("Build convert tabs")
        self._extract_tabs(command="convert")
        logger.debug("Built convert tabs")

    def _remove_tabs(self):
        """ Remove all optional displayed command specific tabs from the notebook. """
        for child in self.tabs():
            if child in self._static_tabs:
                continue
            logger.debug("removing child: %s", child)
            child_name = child.split(".")[-1]
            child_object = self.children.get(child_name)  # returns the OptionalDisplayPage object
            if not child_object:
                continue
            child_object.close()  # Call the OptionalDisplayPage close() method
            self.forget(child)

    def _update_displaybook(self, *args):  # pylint:disable=unused-argument
        """ Callback to be executed when the global tkinter variable `display`
        (:attr:`wrapper_var`) is updated when a Faceswap task is executed.

        Currently only updates when a core faceswap task (extract, train or convert) is executed.

        Parameters
        ----------
        args: tuple
            Required for tkinter callback events, but unused.

        """
        command = self._wrapper_var.get()
        self._remove_tabs()
        if not command or command not in ("extract", "train", "convert"):
            return
        self._command_display(command)

    def _on_tab_change(self, event):  # pylint:disable=unused-argument
        """ Event trigger for tab change events.

        Calls the selected tabs :func:`on_tab_select` method, if it exists, otherwise returns.

        Parameters
        ----------
        event: tkinter callback event
            Required, but unused
        """
        selected = self.select().split(".")[-1]
        logger.debug("Selected tab: %s", selected)
        selected_object = self.children[selected]
        if hasattr(selected_object, "on_tab_select"):
            logger.debug("Calling on_tab_select for '%s'", selected_object)
            selected_object.on_tab_select()
        else:
            logger.debug("Object does not have on_tab_select method. Returning: '%s'",
                         selected_object)
