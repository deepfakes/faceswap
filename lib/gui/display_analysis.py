#!/usr/bin python3
""" Analysis tab of Display Frame of the Faceswap GUI """

import csv
import gettext
import logging
import os
import tkinter as tk
from tkinter import ttk

from lib.logger import parse_class_init

from .custom_widgets import Tooltip
from .display_page import DisplayPage
from .popup_session import SessionPopUp
from .analysis import Session
from .utils import FileHandler, get_config, get_images, LongRunningTask

logger = logging.getLogger(__name__)

# LOCALES
_LANG = gettext.translation("gui.tooltips", localedir="locales", fallback=True)
_ = _LANG.gettext


class Analysis(DisplayPage):  # pylint:disable=too-many-ancestors
    """ Session Analysis Tab.

    The area of the GUI that holds the session summary stats for model training sessions.

    Parameters
    ----------
    parent: :class:`lib.gui.display.DisplayNotebook`
        The :class:`ttk.Notebook` that holds this session summary statistics page
    tab_name: str
        The name of the tab to be displayed in the notebook
    helptext: str
        The help text to display for the summary statistics page
    """
    def __init__(self, parent, tab_name, helptext):
        logger.debug(parse_class_init(locals()))
        super().__init__(parent, tab_name, helptext)
        self._summary = None

        self._reset_session_info()
        _Options(self)
        self._stats = self._get_main_frame()

        self._thread = None  # Thread for compiling stats data in background
        self._set_callbacks()
        logger.debug("Initialized: %s", self.__class__.__name__)

    def set_vars(self):
        """ Set the analysis specific tkinter variables to :attr:`vars`.

        The tracked variables are the global variables that:
            * Trigger when a graph refresh has been requested.
            * Trigger training is commenced or halted
            * The variable holding the location of the current Tensorboard log folder.

        Returns
        -------
        dict
            The dictionary of variable names to tkinter variables
        """
        return {"selected_id": tk.StringVar(),
                "refresh_graph": get_config().tk_vars.refresh_graph,
                "is_training": get_config().tk_vars.is_training,
                "analysis_folder": get_config().tk_vars.analysis_folder}

    def on_tab_select(self):
        """ Callback for when the analysis tab is selected.

        If Faceswap is currently training a model, then update the statistics with the latest
        values.
        """
        if not self.vars["is_training"].get():
            return
        logger.debug("Analysis update callback received")
        self._reset_session()

    def _get_main_frame(self):
        """ Get the main frame to the sub-notebook to hold stats and session data.

        Returns
        -------
        :class:`StatsData`
            The frame that holds the analysis statistics for the Analysis notebook page
        """
        logger.debug("Getting main stats frame")
        mainframe = self.subnotebook_add_page("stats")
        retval = StatsData(mainframe, self.vars["selected_id"], self.helptext["stats"])
        logger.debug("got main frame: %s", retval)
        return retval

    def _set_callbacks(self):
        """ Adds callbacks to update the analysis summary statistics and add them to :attr:`vars`

        Training graph refresh - Updates the stats for the current training session when the graph
        has been updated.

        When the analysis folder has been populated - Updates the stats from that folder.
        """
        self.vars["refresh_graph"].trace("w", self._update_current_session)
        self.vars["analysis_folder"].trace("w", self._populate_from_folder)

    def _update_current_session(self, *args):  # pylint:disable=unused-argument
        """ Update the currently training session data on a graph update callback. """
        if not self.vars["refresh_graph"].get():
            return
        if not self._tab_is_active:
            logger.debug("Analyis tab not selected. Not updating stats")
            return
        logger.debug("Analysis update callback received")
        self._reset_session()

    def _reset_session_info(self):
        """ Reset the session info status to default """
        logger.debug("Resetting session info")
        self.set_info("No session data loaded")

    def _populate_from_folder(self, *args):  # pylint:disable=unused-argument
        """ Populate the Analysis tab from a model folder.

        Triggered when :attr:`vars` ``analysis_folder`` variable is is set.
        """
        if Session.is_training:
            return

        folder = self.vars["analysis_folder"].get()
        if not folder or not os.path.isdir(folder):
            logger.debug("Not a valid folder")
            self._clear_session()
            return

        state_files = [fname
                       for fname in os.listdir(folder)
                       if fname.endswith("_state.json")]
        if not state_files:
            logger.debug("No state files found in folder: '%s'", folder)
            self._clear_session()
            return

        state_file = state_files[0]
        if len(state_files) > 1:
            logger.debug("Multiple models found. Selecting: '%s'", state_file)

        if self._thread is None:
            self._load_session(full_path=os.path.join(folder, state_file))

    @classmethod
    def _get_model_name(cls, model_dir, state_file):
        """ Obtain the model name from a state file's file name.

        Parameters
        ----------
        model_dir: str
            The folder that the model's state file resides in
        state_file: str
            The filename of the model's state file

        Returns
        -------
        str or ``None``
            The name of the model extracted from the state file's file name or ``None`` if no
            log folders were found in the model folder
        """
        logger.debug("Getting model name")
        model_name = state_file.replace("_state.json", "")
        logger.debug("model_name: %s", model_name)
        logs_dir = os.path.join(model_dir, f"{model_name}_logs")
        if not os.path.isdir(logs_dir):
            logger.warning("No logs folder found in folder: '%s'", logs_dir)
            return None
        return model_name

    def _set_session_summary(self, message):
        """ Set the summary data and info message.

        Parameters
        ----------
        message: str
            The information message to set
        """
        if self._thread is None:
            logger.debug("Setting session summary. (message: '%s')", message)
            self._thread = LongRunningTask(target=self._summarise_data,
                                           args=(Session, ),
                                           widget=self)
            self._thread.start()
            self.after(1000, lambda msg=message: self._set_session_summary(msg))
        elif not self._thread.complete.is_set():
            logger.debug("Data not yet available")
            self.after(1000, lambda msg=message: self._set_session_summary(msg))
        else:
            logger.debug("Retrieving data from thread")
            result = self._thread.get_result()
            if result is None:
                logger.debug("No result from session summary. Clearing analysis view")
                self._clear_session()
                return
            self._summary = result
            self._thread = None
            self.set_info(f"Session: {message}")
            self._stats.tree_insert_data(self._summary)

    @classmethod
    def _summarise_data(cls, session):
        """ Summarize data in a LongRunningThread as it can take a while.

        Parameters
        ----------
        session: :class:`lib.gui.analysis.Session`
            The session object to generate the summary for
        """
        return session.full_summary

    def _clear_session(self):
        """ Clear the currently displayed analysis data from the Tree-View. """
        logger.debug("Clearing session")
        if not Session.is_loaded:
            logger.trace("No session loaded. Returning")
            return
        self._summary = None
        self._stats.tree_clear()
        if not Session.is_training:
            self._reset_session_info()
            Session.clear()

    def _load_session(self, full_path=None):
        """ Load the session statistics from a model's state file into the Analysis tab of the GUI
        display window.

        If a model's log files cannot be found within the model folder then the session is cleared.

        Parameters
        ----------
        full_path: str, optional
            The path to the state file to load session information from. If this is ``None`` then
            a file dialog is popped to enable the user to choose a state file. Default: ``None``
         """
        logger.debug("Loading session")
        if full_path is None:
            full_path = FileHandler("filename", "state").return_file
            if not full_path:
                return
        self._clear_session()
        logger.debug("state_file: '%s'", full_path)
        model_dir, state_file = os.path.split(full_path)
        logger.debug("model_dir: '%s'", model_dir)
        model_name = self._get_model_name(model_dir, state_file)
        if not model_name:
            return
        Session.initialize_session(model_dir, model_name, is_training=False)
        msg = full_path
        if len(msg) > 70:
            msg = f"...{msg[-70:]}"
        self._set_session_summary(msg)

    def _reset_session(self):
        """ Reset currently training sessions. Clears the current session and loads in the latest
        data. """
        logger.debug("Reset current training session")
        if not Session.is_training:
            logger.debug("Training not running")
            return
        if Session.logging_disabled:
            logger.trace("Logging disabled. Not triggering analysis update")
            return
        self._clear_session()
        self._set_session_summary("Currently running training session")

    def _save_session(self):
        """ Launch a file dialog pop-up to save the current analysis data to a CSV file. """
        logger.debug("Saving session")
        if not self._summary:
            logger.debug("No summary data loaded. Nothing to save")
            print("No summary data loaded. Nothing to save")
            return
        savefile = FileHandler("save", "csv").return_file
        if not savefile:
            logger.debug("No save file. Returning")
            return

        logger.debug("Saving to: '%s'", savefile)
        fieldnames = sorted(key for key in self._summary[0].keys())
        with savefile as outfile:
            csvout = csv.DictWriter(outfile, fieldnames)
            csvout.writeheader()
            for row in self._summary:
                csvout.writerow(row)


class _Options():  # pylint:disable=too-few-public-methods
    """ Options buttons for the Analysis tab.

    Parameters
    ----------
    parent: :class:`Analysis`
        The Analysis Display Tab that holds the options buttons
    """
    def __init__(self, parent):
        logger.debug(parse_class_init(locals()))
        self._parent = parent
        self._buttons = self._add_buttons()
        self._add_training_callback()
        logger.debug("Initialized: %s", self.__class__.__name__)

    def _add_buttons(self):
        """ Add the option buttons.

        Returns
        -------
        dict
            The button names to button objects
        """
        buttons = {}
        for btntype in ("clear", "save", "load"):
            logger.debug("Adding button: '%s'", btntype)
            cmd = getattr(self._parent, f"_{btntype}_session")
            btn = ttk.Button(self._parent.optsframe,
                             image=get_images().icons[btntype],
                             command=cmd)
            btn.pack(padx=2, side=tk.RIGHT)
            hlp = self._set_help(btntype)
            Tooltip(btn, text=hlp, wrap_length=200)
            buttons[btntype] = btn
        logger.debug("buttons: %s", buttons)
        return buttons

    @classmethod
    def _set_help(cls, button_type):
        """ Set the help text for option buttons.

        Parameters
        ----------
        button_type: {"reload", "clear", "save", "load"}
            The type of button to set the help text for
        """
        logger.debug("Setting help")
        hlp = ""
        if button_type == "reload":
            hlp = _("Load/Refresh stats for the currently training session")
        elif button_type == "clear":
            hlp = _("Clear currently displayed session stats")
        elif button_type == "save":
            hlp = _("Save session stats to csv")
        elif button_type == "load":
            hlp = _("Load saved session stats")
        return hlp

    def _add_training_callback(self):
        """ Add a callback to the training tkinter variable to disable save and clear buttons
        when a model is training. """
        var = self._parent.vars["is_training"]
        var.trace("w", self._set_buttons_state)

    def _set_buttons_state(self, *args):  # pylint:disable=unused-argument
        """ Callback to enable/disable button when training is commenced and stopped. """
        is_training = self._parent.vars["is_training"].get()
        state = "disabled" if is_training else "!disabled"
        for name, button in self._buttons.items():
            if name not in ("load", "clear"):
                continue
            logger.debug("Setting %s button state to %s", name, state)
            button.state([state])


class StatsData(ttk.Frame):  # pylint:disable=too-many-ancestors
    """ Stats frame of analysis tab.

    Holds the tree-view containing the summarized session statistics in the Analysis tab.

    Parameters
    ----------
    parent: :class:`tkinter.Frame`
        The frame within the Analysis Notebook that will hold the statistics
    selected_id: :class:`tkinter.IntVar`
        The tkinter variable that holds the currently selected session ID
    helptext: str
        The help text to display for the summary statistics page
    """
    def __init__(self, parent, selected_id, helptext):
        logger.debug(parse_class_init(locals()))
        super().__init__(parent)
        self._selected_id = selected_id

        self._canvas = tk.Canvas(self, bd=0, highlightthickness=0)
        tree_frame = ttk.Frame(self._canvas)
        self._tree_canvas = self._canvas.create_window((0, 0), window=tree_frame, anchor=tk.NW)
        self._sub_frame = ttk.Frame(tree_frame)

        self._add_label()

        self._tree = ttk.Treeview(self._sub_frame, height=1, selectmode=tk.BROWSE)
        self._scrollbar = ttk.Scrollbar(tree_frame, orient="vertical", command=self._tree.yview)

        self._columns = self._tree_configure(helptext)
        self._canvas.bind("<Configure>", self._resize_frame)

        self._scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self._tree.pack(side=tk.TOP, fill=tk.X)
        self._sub_frame.pack(side=tk.LEFT, fill=tk.X, anchor=tk.N, expand=True)
        self._canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.pack(side=tk.TOP, padx=5, pady=5, fill=tk.BOTH, expand=True)

        logger.debug("Initialized: %s", self.__class__.__name__)

    def _add_label(self):
        """ Add the title above the tree-view. """
        logger.debug("Adding Treeview title")
        lbl = ttk.Label(self._sub_frame, text="Session Stats", anchor=tk.CENTER)
        lbl.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

    def _resize_frame(self, event):
        """ Resize the options frame to fit the canvas.

        Parameters
        ----------
        event: `tkinter.Event`
            The tkinter resize event
        """
        logger.debug("Resize Analysis Frame")
        canvas_width = event.width
        canvas_height = event.height
        self._canvas.itemconfig(self._tree_canvas, width=canvas_width, height=canvas_height)
        logger.debug("Resized Analysis Frame")

    def _tree_configure(self, helptext):
        """ Build a tree-view widget to hold the sessions stats.

        Parameters
        ----------
        helptext: str
            The helptext to display when the mouse is over the tree-view

        Returns
        -------
        list
            The list of tree-view columns
        """
        logger.debug("Configuring Treeview")
        self._tree.configure(yscrollcommand=self._scrollbar.set)
        self._tree.tag_configure("total", background="black", foreground="white")
        self._tree.bind("<ButtonRelease-1>", self._select_item)
        Tooltip(self._tree, text=helptext, wrap_length=200)
        return self._tree_columns()

    def _tree_columns(self):
        """ Add the columns to the totals tree-view.

        Returns
        -------
        list
            The list of tree-view columns
        """
        logger.debug("Adding Treeview columns")
        columns = (("session", 40, "#"),
                   ("start", 130, None),
                   ("end", 130, None),
                   ("elapsed", 90, None),
                   ("batch", 50, None),
                   ("iterations", 90, None),
                   ("rate", 60, "EGs/sec"))
        self._tree["columns"] = [column[0] for column in columns]

        for column in columns:
            text = column[2] if column[2] else column[0].title()
            logger.debug("Adding heading: '%s'", text)
            self._tree.heading(column[0], text=text)
            self._tree.column(column[0], width=column[1], anchor=tk.E, minwidth=40)
        self._tree.column("#0", width=40)
        self._tree.heading("#0", text="Graphs")

        return [column[0] for column in columns]

    def tree_insert_data(self, sessions_summary):
        """ Insert the summary data into the statistics tree-view.

        Parameters
        ----------
        sessions_summary: list
            List of session summary dicts for populating into the tree-view
        """
        logger.debug("Inserting treeview data")
        self._tree.configure(height=len(sessions_summary))

        for item in sessions_summary:
            values = [item[column] for column in self._columns]
            kwargs = {"values": values}
            if self._check_valid_data(values):
                # Don't show graph icon for non-existent sessions
                kwargs["image"] = get_images().icons["graph"]
            if values[0] == "Total":
                kwargs["tags"] = "total"
            self._tree.insert("", "end", **kwargs)

    def tree_clear(self):
        """ Clear all of the summary data from the tree-view. """
        logger.debug("Clearing treeview data")
        try:
            self._tree.delete(* self._tree.get_children())
            self._tree.configure(height=1)
        except tk.TclError:
            # Catch non-existent tree view when rebuilding the GUI
            pass

    def _select_item(self, event):
        """ Update the session summary info with the selected item or launch graph.

        If the mouse is clicked on the graph icon, then the session summary pop-up graph is
        launched. Otherwise the selected ID is stored.

        Parameters
        ----------
        event: :class:`tkinter.Event`
            The tkinter mouse button release event
        """
        region = self._tree.identify("region", event.x, event.y)
        selection = self._tree.focus()
        values = self._tree.item(selection, "values")
        if values:
            logger.debug("Selected values: %s", values)
            self._selected_id.set(values[0])
            if region == "tree" and self._check_valid_data(values):
                data_points = int(values[self._columns.index("iterations")])
                self._data_popup(data_points)

    def _check_valid_data(self, values):
        """ Check there is valid data available for popping up a graph.

        Parameters
        ----------
        values: list
            The values that exist for a single session that are to be validated
        """
        col_indices = [self._columns.index("batch"), self._columns.index("iterations")]
        for idx in col_indices:
            if (isinstance(values[idx], int) or values[idx].isdigit()) and int(values[idx]) == 0:
                logger.warning("No data to graph for selected session")
                return False
        return True

    def _data_popup(self, data_points):
        """ Pop up a window and control it's position

        The default view is rolling average over 500 points. If there are fewer data points than
        this, switch the default to smoothed,

        Parameters
        ----------
        data_points: int
            The number of iterations that are to be plotted
        """
        logger.debug("Popping up data window")
        scaling_factor = get_config().scaling_factor
        toplevel = SessionPopUp(self._selected_id.get(),
                                data_points)
        toplevel.title(self._data_popup_title())
        toplevel.tk.call(
            'wm',
            'iconphoto',
            toplevel._w, get_images().icons["favicon"])  # pylint:disable=protected-access

        root = get_config().root
        offset = (root.winfo_x() + 20, root.winfo_y() + 20)
        height = int(900 * scaling_factor)
        width = int(480 * scaling_factor)
        toplevel.geometry(f"{height}x{width}+{offset[0]}+{offset[1]}")

        toplevel.update()

    def _data_popup_title(self):
        """ Get the summary graph popup title.

        Returns
        -------
        str
            The title to display at the top of the pop-up graph window
        """
        logger.debug("Setting poup title")
        selected_id = self._selected_id.get()
        model_dir, model_name = os.path.split(Session.model_filename)
        title = "All Sessions"
        if selected_id != "Total":
            title = f"{model_name.title()} Model: Session #{selected_id}"
        logger.debug("Title: '%s'", title)
        return f"{title} - {model_dir}"
