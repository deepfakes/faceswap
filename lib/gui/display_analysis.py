#!/usr/bin python3
""" Analysis tab of Display Frame of the Faceswap GUI """

import csv
import logging
import os
import tkinter as tk
from tkinter import ttk

from .control_helper import ControlBuilder, ControlPanelOption
from .display_graph import SessionGraph
from .display_page import DisplayPage
from .stats import Calculations, Session
from .custom_widgets import Tooltip
from .utils import FileHandler, get_config, get_images, LongRunningTask

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class Analysis(DisplayPage):  # pylint: disable=too-many-ancestors
    """ Session analysis tab """
    def __init__(self, parent, tabname, helptext):
        logger.debug("Initializing: %s: (parent, %s, tabname: '%s', helptext: '%s')",
                     self.__class__.__name__, parent, tabname, helptext)
        super().__init__(parent, tabname, helptext)

        self.summary = None
        self.session = None
        self.add_options()
        self.add_main_frame()
        self.thread = None  # Thread for compiling stats data in background
        self.set_callbacks()
        logger.debug("Initialized: %s", self.__class__.__name__)

    def set_callbacks(self):
        """ Add a callback to update analysis when the training graph is updated """
        tkv = get_config().tk_vars
        tkv["refreshgraph"].trace("w", self.update_current_session)
        tkv["istraining"].trace("w", self.remove_current_session)
        tkv["analysis_folder"].trace("w", self.populate_from_folder)

    def update_current_session(self, *args):  # pylint:disable=unused-argument
        """ Update the current session data on a graph update callback """
        if not get_config().tk_vars["refreshgraph"].get():
            return
        logger.debug("Analysis update callback received")
        self.reset_session()

    def remove_current_session(self, *args):  # pylint:disable=unused-argument
        """ Remove the current session data on a istraining=False callback """
        if get_config().tk_vars["istraining"].get():
            return
        logger.debug("Remove current training Analysis callback received")
        self.clear_session()

    def set_vars(self):
        """ Analysis specific vars """
        selected_id = tk.StringVar()
        return {"selected_id": selected_id}

    def add_main_frame(self):
        """ Add the main frame to the sub-notebook
            to hold stats and session data """
        logger.debug("Adding main frame")
        mainframe = self.subnotebook_add_page("stats")
        self.stats = StatsData(mainframe,
                               self.vars["selected_id"],
                               self.helptext["stats"])
        logger.debug("Added main frame")

    def add_options(self):
        """ Add the options bar """
        logger.debug("Adding options")
        self.reset_session_info()
        options = Options(self)
        options.add_options()
        logger.debug("Added options")

    def reset_session_info(self):
        """ Reset the session info status to default """
        logger.debug("Resetting session info")
        self.set_info("No session data loaded")

    def populate_from_folder(self, *args):  # pylint:disable=unused-argument
        """ Populate the Analysis tab from just a model folder. Triggered
        when tkinter variable ``analysis_folder`` is set.
        """
        folder = get_config().tk_vars["analysis_folder"].get()
        if not folder or not os.path.isdir(folder):
            logger.debug("Not a valid folder")
            self.clear_session()
            return

        state_files = [fname
                       for fname in os.listdir(folder)
                       if fname.endswith("_state.json")]
        if not state_files:
            logger.debug("No state files found in folder: '%s'", folder)
            self.clear_session()
            return

        state_file = state_files[0]
        if len(state_files) > 1:
            logger.debug("Multiple models found. Selecting: '%s'", state_file)

        if self.thread is None:
            self.load_session(fullpath=os.path.join(folder, state_file))

    def load_session(self, fullpath=None):
        """ Load previously saved sessions """
        logger.debug("Loading session")
        if fullpath is None:
            fullpath = FileHandler("filename", "state").retfile
            if not fullpath:
                return
        self.clear_session()
        logger.debug("state_file: '%s'", fullpath)
        model_dir, state_file = os.path.split(fullpath)
        logger.debug("model_dir: '%s'", model_dir)
        model_name = self.get_model_name(model_dir, state_file)
        if not model_name:
            return
        self.session = Session(model_dir=model_dir, model_name=model_name)
        self.session.initialize_session(is_training=False)
        msg = fullpath
        if len(msg) > 70:
            msg = "...{}".format(msg[-70:])
        self.set_session_summary(msg)

    @staticmethod
    def get_model_name(model_dir, state_file):
        """ Get the state file from the model directory """
        logger.debug("Getting model name")
        model_name = state_file.replace("_state.json", "")
        logger.debug("model_name: %s", model_name)
        logs_dir = os.path.join(model_dir, "{}_logs".format(model_name))
        if not os.path.isdir(logs_dir):
            logger.warning("No logs folder found in folder: '%s'", logs_dir)
            return None
        return model_name

    def reset_session(self):
        """ Reset currently training sessions """
        logger.debug("Reset current training session")
        self.clear_session()
        session = get_config().session
        if not session.initialized:
            logger.debug("Training not running")
            return
        if session.logging_disabled:
            logger.trace("Logging disabled. Not triggering analysis update")
            return
        msg = "Currently running training session"
        self.session = session
        # Reload the state file to get approx currently training iterations
        self.session.load_state_file()
        self.set_session_summary(msg)

    def set_session_summary(self, message):
        """ Set the summary data and info message """
        if self.thread is None:
            logger.debug("Setting session summary. (message: '%s')", message)
            self.thread = LongRunningTask(target=self.summarise_data,
                                          args=(self.session, ),
                                          widget=self)
            self.thread.start()
            self.after(1000, lambda msg=message: self.set_session_summary(msg))
        elif not self.thread.complete.is_set():
            logger.debug("Data not yet available")
            self.after(1000, lambda msg=message: self.set_session_summary(msg))
        else:
            logger.debug("Retrieving data from thread")
            result = self.thread.get_result()
            if result is None:
                logger.debug("No result from session summary. Clearing analysis view")
                self.clear_session()
                return
            self.summary = result
            self.thread = None
            self.set_info("Session: {}".format(message))
            self.stats.session = self.session
            self.stats.tree_insert_data(self.summary)

    @staticmethod
    def summarise_data(session):
        """ Summarize data in a LongRunningThread as it can take a while """
        return session.full_summary

    def clear_session(self):
        """ Clear sessions stats """
        logger.debug("Clearing session")
        if self.session is None:
            logger.trace("No session loaded. Returning")
            return
        self.summary = None
        self.stats.session = None
        self.stats.tree_clear()
        self.reset_session_info()
        self.session = None

    def save_session(self):
        """ Save sessions stats to csv """
        logger.debug("Saving session")
        if not self.summary:
            logger.debug("No summary data loaded. Nothing to save")
            print("No summary data loaded. Nothing to save")
            return
        savefile = FileHandler("save", "csv").retfile
        if not savefile:
            logger.debug("No save file. Returning")
            return

        logger.debug("Saving to: '%s'", savefile)
        fieldnames = sorted(key for key in self.summary[0].keys())
        with savefile as outfile:
            csvout = csv.DictWriter(outfile, fieldnames)
            csvout.writeheader()
            for row in self.summary:
                csvout.writerow(row)


class Options():
    """ Options bar of Analysis tab """
    def __init__(self, parent):
        logger.debug("Initializing: %s", self.__class__.__name__)
        self.optsframe = parent.optsframe
        self.parent = parent
        logger.debug("Initialized: %s", self.__class__.__name__)

    def add_options(self):
        """ Add the display tab options """
        self.add_buttons()

    def add_buttons(self):
        """ Add the option buttons """
        for btntype in ("clear", "save", "load"):
            logger.debug("Adding button: '%s'", btntype)
            cmd = getattr(self.parent, "{}_session".format(btntype))
            btn = ttk.Button(self.optsframe,
                             image=get_images().icons[btntype],
                             command=cmd)
            btn.pack(padx=2, side=tk.RIGHT)
            hlp = self.set_help(btntype)
            Tooltip(btn, text=hlp, wraplength=200)

    @staticmethod
    def set_help(btntype):
        """ Set the help text for option buttons """
        logger.debug("Setting help")
        hlp = ""
        if btntype == "reload":
            hlp = "Load/Refresh stats for the currently training session"
        elif btntype == "clear":
            hlp = "Clear currently displayed session stats"
        elif btntype == "save":
            hlp = "Save session stats to csv"
        elif btntype == "load":
            hlp = "Load saved session stats"
        return hlp


class StatsData(ttk.Frame):  # pylint: disable=too-many-ancestors
    """ Stats frame of analysis tab """
    def __init__(self, parent, selected_id, helptext):
        logger.debug("Initializing: %s: (parent, %s, selected_id: %s, helptext: '%s')",
                     self.__class__.__name__, parent, selected_id, helptext)
        super().__init__(parent)
        self.pack(side=tk.TOP, padx=5, pady=5, fill=tk.BOTH, expand=True)
        self.session = None  # set when loading or clearing from parent
        self.thread = None  # Thread for loading data popup
        self.selected_id = selected_id
        self.popup_positions = list()

        self.canvas = tk.Canvas(self, bd=0, highlightthickness=0)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.tree_frame = ttk.Frame(self.canvas)
        self.tree_canvas = self.canvas.create_window((0, 0), window=self.tree_frame, anchor=tk.NW)
        self.sub_frame = ttk.Frame(self.tree_frame)
        self.sub_frame.pack(side=tk.LEFT, fill=tk.X, anchor=tk.N, expand=True)

        self.add_label()
        self.tree = ttk.Treeview(self.sub_frame, height=1, selectmode=tk.BROWSE)
        self.scrollbar = ttk.Scrollbar(self.tree_frame, orient="vertical", command=self.tree.yview)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.columns = self.tree_configure(helptext)
        self.canvas.bind("<Configure>", self.resize_frame)
        logger.debug("Initialized: %s", self.__class__.__name__)

    def add_label(self):
        """ Add tree-view Title """
        logger.debug("Adding Treeview title")
        lbl = ttk.Label(self.sub_frame, text="Session Stats", anchor=tk.CENTER)
        lbl.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

    def resize_frame(self, event):
        """ Resize the options frame to fit the canvas """
        logger.debug("Resize Analysis Frame")
        canvas_width = event.width
        canvas_height = event.height
        self.canvas.itemconfig(self.tree_canvas, width=canvas_width, height=canvas_height)
        logger.debug("Resized Analysis Frame")

    def tree_configure(self, helptext):
        """ Build a tree-view widget to hold the sessions stats """
        logger.debug("Configuring Treeview")
        self.tree.configure(yscrollcommand=self.scrollbar.set)
        self.tree.tag_configure("total", background="black", foreground="white")
        self.tree.pack(side=tk.TOP, fill=tk.X)
        self.tree.bind("<ButtonRelease-1>", self.select_item)
        Tooltip(self.tree, text=helptext, wraplength=200)
        return self.tree_columns()

    def tree_columns(self):
        """ Add the columns to the totals tree-view """
        logger.debug("Adding Treeview columns")
        columns = (("session", 40, "#"),
                   ("start", 130, None),
                   ("end", 130, None),
                   ("elapsed", 90, None),
                   ("batch", 50, None),
                   ("iterations", 90, None),
                   ("rate", 60, "EGs/sec"))
        self.tree["columns"] = [column[0] for column in columns]

        for column in columns:
            text = column[2] if column[2] else column[0].title()
            logger.debug("Adding heading: '%s'", text)
            self.tree.heading(column[0], text=text)
            self.tree.column(column[0], width=column[1], anchor=tk.E, minwidth=40)
        self.tree.column("#0", width=40)
        self.tree.heading("#0", text="Graphs")

        return [column[0] for column in columns]

    def tree_insert_data(self, sessions_summary):
        """ Insert the data into the totals tree-view """
        logger.debug("Inserting treeview data")
        self.tree.configure(height=len(sessions_summary))

        for item in sessions_summary:
            values = [item[column] for column in self.columns]
            kwargs = {"values": values}
            if self.check_valid_data(values):
                # Don't show graph icon for non-existent sessions
                kwargs["image"] = get_images().icons["graph"]
            if values[0] == "Total":
                kwargs["tags"] = "total"
            self.tree.insert("", "end", **kwargs)

    def tree_clear(self):
        """ Clear the totals tree """
        logger.debug("Clearing treeview data")
        try:
            self.tree.delete(* self.tree.get_children())
            self.tree.configure(height=1)
        except tk.TclError:
            # Catch non-existent tree view when rebuilding the GUI
            pass

    def select_item(self, event):
        """ Update the session summary info with
            the selected item or launch graph """
        region = self.tree.identify("region", event.x, event.y)
        selection = self.tree.focus()
        values = self.tree.item(selection, "values")
        if values:
            logger.debug("Selected values: %s", values)
            self.selected_id.set(values[0])
            if region == "tree" and self.check_valid_data(values):
                datapoints = int(values[self.columns.index("iterations")])
                self.data_popup(datapoints)

    def check_valid_data(self, values):
        """ Check there is valid data available for popping up a graph """
        col_indices = [self.columns.index("batch"), self.columns.index("iterations")]
        for idx in col_indices:
            if (isinstance(values[idx], int) or values[idx].isdigit()) and int(values[idx]) == 0:
                logger.warning("No data to graph for selected session")
                return False
        return True

    def data_popup(self, datapoints):
        """ Pop up a window and control it's position

            The default view is rolling average over 500 points.
            If there are fewer data points than this, switch the default
            to smoothed
        """
        logger.debug("Popping up data window")
        scaling_factor = get_config().scaling_factor
        toplevel = SessionPopUp(self.session.modeldir,
                                self.session.modelname,
                                self.selected_id.get(),
                                datapoints)
        toplevel.title(self.data_popup_title())
        toplevel.tk.call(
            'wm',
            'iconphoto',
            toplevel._w, get_images().icons["favicon"])  # pylint:disable=protected-access
        position = self.data_popup_get_position()
        height = int(900 * scaling_factor)
        width = int(480 * scaling_factor)
        toplevel.geometry("{}x{}+{}+{}".format(str(height),
                                               str(width),
                                               str(position[0]),
                                               str(position[1])))
        toplevel.update()

    def data_popup_title(self):
        """ Set the data popup title """
        logger.debug("Setting poup title")
        selected_id = self.selected_id.get()
        title = "All Sessions"
        if selected_id != "Total":
            title = "{} Model: Session #{}".format(self.session.modelname.title(), selected_id)
        logger.debug("Title: '%s'", title)
        return "{} - {}".format(title, self.session.modeldir)

    def data_popup_get_position(self):
        """ Get the position of the next window """
        logger.debug("getting poup position")
        init_pos = [120, 120]
        pos = init_pos
        while True:
            if pos not in self.popup_positions:
                self.popup_positions.append(pos)
                break
            pos = [item + 200 for item in pos]
            init_pos, pos = self.data_popup_check_boundaries(init_pos, pos)
        logger.debug("Position: %s", pos)
        return pos

    def data_popup_check_boundaries(self, initial_position, position):
        """ Check that the popup remains within the screen boundaries """
        logger.debug("Checking poup boundaries: (initial_position: %s, position: %s)",
                     initial_position, position)
        boundary_x = self.winfo_screenwidth() - 120
        boundary_y = self.winfo_screenheight() - 120
        if position[0] >= boundary_x or position[1] >= boundary_y:
            initial_position = [initial_position[0] + 50, initial_position[1]]
            position = initial_position
        logger.debug("Returning poup boundaries: (initial_position: %s, position: %s)",
                     initial_position, position)
        return initial_position, position


class SessionPopUp(tk.Toplevel):
    """ Pop up for detailed graph/stats for selected session """
    def __init__(self, model_dir, model_name, session_id, datapoints):
        logger.debug("Initializing: %s: (model_dir: %s, model_name: %s, session_id: %s, "
                     "datapoints: %s)", self.__class__.__name__, model_dir, model_name, session_id,
                     datapoints)
        super().__init__()
        self.thread = None  # Thread for loading data in a background task
        self.default_avg = 500
        self.default_view = "avg" if datapoints > self.default_avg * 2 else "smoothed"
        self.session_id = session_id
        self.session = Session(model_dir=model_dir, model_name=model_name)
        self.initialize_session()

        self.graph_frame = None
        self.graph = None
        self.display_data = None

        self.vars = {"status": tk.StringVar()}
        self.graph_initialised = False
        self.build()
        logger.debug("Initialized: %s", self.__class__.__name__)

    @property
    def is_totals(self):
        """ Return True if these are totals else False """
        return bool(self.session_id == "Total")

    def initialize_session(self):
        """ Initialize the session """
        logger.debug("Initializing session")
        kwargs = dict(is_training=False)
        if not self.is_totals:
            kwargs["session_id"] = int(self.session_id)
        logger.debug("Session kwargs: %s", kwargs)
        self.session.initialize_session(**kwargs)

    def build(self):
        """ Build the popup window """
        logger.debug("Building popup")
        optsframe = self.layout_frames()
        self.set_callback()
        self.opts_build(optsframe)
        self.compile_display_data()
        logger.debug("Built popup")

    def set_callback(self):
        """ Set a tkinter Boolean var to callback when graph is ready to build """
        logger.debug("Setting tk graph build variable")
        var = tk.BooleanVar()
        var.set(False)
        var.trace("w", self.graph_build)
        self.vars["buildgraph"] = var

    def layout_frames(self):
        """ Top level container frames """
        logger.debug("Layout frames")
        leftframe = ttk.Frame(self)
        leftframe.pack(side=tk.LEFT, expand=False, fill=tk.BOTH, pady=5)

        sep = ttk.Frame(self, width=2, relief=tk.RIDGE)
        sep.pack(fill=tk.Y, side=tk.LEFT)

        self.graph_frame = ttk.Frame(self)
        self.graph_frame.pack(side=tk.RIGHT, fill=tk.BOTH, pady=5, expand=True)
        logger.debug("Laid out frames")

        return leftframe

    def opts_build(self, frame):
        """ Build Options into the options frame """
        logger.debug("Building Options")
        self.opts_combobox(frame)
        self.opts_checkbuttons(frame)
        self.opts_loss_keys(frame)
        self.opts_slider(frame)
        self.opts_buttons(frame)
        sep = ttk.Frame(frame, height=2, relief=tk.RIDGE)
        sep.pack(fill=tk.X, pady=(5, 0), side=tk.BOTTOM)
        logger.debug("Built Options")

    def opts_combobox(self, frame):
        """ Add the options combo boxes """
        logger.debug("Building Combo boxes")
        choices = {"Display": ("Loss", "Rate"),
                   "Scale": ("Linear", "Log")}

        for item in ["Display", "Scale"]:
            var = tk.StringVar()

            cmbframe = ttk.Frame(frame)
            cmbframe.pack(fill=tk.X, pady=5, padx=5, side=tk.TOP)
            lblcmb = ttk.Label(cmbframe,
                               text="{}:".format(item),
                               width=7,
                               anchor=tk.W)
            lblcmb.pack(padx=(0, 2), side=tk.LEFT)

            cmb = ttk.Combobox(cmbframe, textvariable=var, width=10)
            cmb["values"] = choices[item]
            cmb.current(0)
            cmb.pack(fill=tk.X, side=tk.RIGHT)

            cmd = self.optbtn_reload if item == "Display" else self.graph_scale
            var.trace("w", cmd)
            self.vars[item.lower().strip()] = var

            hlp = self.set_help(item)
            Tooltip(cmbframe, text=hlp, wraplength=200)
        logger.debug("Built Combo boxes")

    @staticmethod
    def add_section(frame, title):
        """ Add a separator and section title """
        sep = ttk.Frame(frame, height=2, relief=tk.SOLID)
        sep.pack(fill=tk.X, pady=(5, 0), side=tk.TOP)
        lbl = ttk.Label(frame, text=title)
        lbl.pack(side=tk.TOP, padx=5, pady=0, anchor=tk.CENTER)

    def opts_checkbuttons(self, frame):
        """ Add the options check buttons """
        logger.debug("Building Check Buttons")

        self.add_section(frame, "Display")
        for item in ("raw", "trend", "avg", "smoothed", "outliers"):
            if item == "avg":
                text = "Show Rolling Average"
            elif item == "outliers":
                text = "Flatten Outliers"
            else:
                text = "Show {}".format(item.title())
            var = tk.BooleanVar()

            if item == self.default_view:
                var.set(True)

            self.vars[item] = var

            ctl = ttk.Checkbutton(frame, variable=var, text=text)
            ctl.pack(side=tk.TOP, padx=5, pady=5, anchor=tk.W)

            hlp = self.set_help(item)
            Tooltip(ctl, text=hlp, wraplength=200)
        logger.debug("Built Check Buttons")

    def opts_loss_keys(self, frame):
        """ Add loss key selections """
        logger.debug("Building Loss Key Check Buttons")
        loss_keys = self.session.loss_keys
        lk_vars = dict()
        section_added = False
        for loss_key in sorted(loss_keys):
            text = loss_key.replace("_", " ").title()
            helptext = "Display {}".format(text)
            var = tk.BooleanVar()
            if loss_key.startswith("total"):
                var.set(True)
            lk_vars[loss_key] = var

            if len(loss_keys) == 1:
                # Don't display if there's only one item
                var.set(True)
                break

            if not section_added:
                self.add_section(frame, "Keys")
                section_added = True

            ctl = ttk.Checkbutton(frame, variable=var, text=text)
            ctl.pack(side=tk.TOP, padx=5, pady=5, anchor=tk.W)
            Tooltip(ctl, text=helptext, wraplength=200)

        self.vars["loss_keys"] = lk_vars
        logger.debug("Built Loss Key Check Buttons")

    def opts_slider(self, frame):
        """ Add the options entry boxes """

        self.add_section(frame, "Parameters")
        logger.debug("Building Slider Controls")
        for item in ("avgiterations", "smoothamount"):
            if item == "avgiterations":
                dtype = int
                text = "Iterations to Average:"
                default = 500
                rounding = 25
                min_max = (25, 2500)
            elif item == "smoothamount":
                dtype = float
                text = "Smoothing Amount:"
                default = 0.90
                rounding = 2
                min_max = (0, 0.99)
            slider = ControlPanelOption(text,
                                        dtype,
                                        default=default,
                                        rounding=rounding,
                                        min_max=min_max,
                                        helptext=self.set_help(item))
            self.vars[item] = slider.tk_var
            ControlBuilder(frame, slider, 1, 19, None, True)
        logger.debug("Built Sliders")

    def opts_buttons(self, frame):
        """ Add the option buttons """
        logger.debug("Building Buttons")
        btnframe = ttk.Frame(frame)
        btnframe.pack(fill=tk.X, pady=5, padx=5, side=tk.BOTTOM)

        lblstatus = ttk.Label(btnframe,
                              width=40,
                              textvariable=self.vars["status"],
                              anchor=tk.W)
        lblstatus.pack(side=tk.LEFT, anchor=tk.W, fill=tk.X, expand=True)

        for btntype in ("reload", "save"):
            cmd = getattr(self, "optbtn_{}".format(btntype))
            btn = ttk.Button(btnframe,
                             image=get_images().icons[btntype],
                             command=cmd)
            btn.pack(padx=2, side=tk.RIGHT)
            hlp = self.set_help(btntype)
            Tooltip(btn, text=hlp, wraplength=200)
        logger.debug("Built Buttons")

    def optbtn_save(self):
        """ Action for save button press """
        logger.debug("Saving File")
        savefile = FileHandler("save", "csv").retfile
        if not savefile:
            logger.debug("Save Cancelled")
            return
        logger.debug("Saving to: %s", savefile)
        save_data = self.display_data.stats
        fieldnames = sorted(key for key in save_data.keys())

        with savefile as outfile:
            csvout = csv.writer(outfile, delimiter=",")
            csvout.writerow(fieldnames)
            csvout.writerows(zip(*[save_data[key] for key in fieldnames]))

    def optbtn_reload(self, *args):  # pylint: disable=unused-argument
        """ Action for reset button press and checkbox changes"""
        logger.debug("Refreshing Graph")
        if not self.graph_initialised:
            return
        valid = self.compile_display_data()
        if not valid:
            logger.debug("Invalid data")
            return
        self.graph.refresh(self.display_data,
                           self.vars["display"].get(),
                           self.vars["scale"].get())
        logger.debug("Refreshed Graph")

    def graph_scale(self, *args):  # pylint: disable=unused-argument
        """ Action for changing graph scale """
        if not self.graph_initialised:
            return
        self.graph.set_yscale_type(self.vars["scale"].get())

    @staticmethod
    def set_help(control):
        """ Set the help text for option buttons """
        hlp = ""
        control = control.lower()
        if control == "reload":
            hlp = "Refresh graph"
        elif control == "save":
            hlp = "Save display data to csv"
        elif control == "avgiterations":
            hlp = "Number of data points to sample for rolling average"
        elif control == "smoothamount":
            hlp = "Set the smoothing amount. 0 is no smoothing, 0.99 is maximum smoothing"
        elif control == "outliers":
            hlp = "Flatten data points that fall more than 1 standard " \
                  "deviation from the mean to the mean value."
        elif control == "avg":
            hlp = "Display rolling average of the data"
        elif control == "smoothed":
            hlp = "Smooth the data"
        elif control == "raw":
            hlp = "Display raw data"
        elif control == "trend":
            hlp = "Display polynormal data trend"
        elif control == "display":
            hlp = "Set the data to display"
        elif control == "scale":
            hlp = "Change y-axis scale"
        return hlp

    def compile_display_data(self):
        """ Compile the data to be displayed """
        if self.thread is None:
            logger.debug("Compiling Display Data in background thread")
            loss_keys = [key for key, val in self.vars["loss_keys"].items()
                         if val.get()]
            logger.debug("Selected loss_keys: %s", loss_keys)

            selections = self.selections_to_list()

            if not self.check_valid_selection(loss_keys, selections):
                logger.warning("No data to display. Not refreshing")
                return False
            self.vars["status"].set("Loading Data...")
            kwargs = dict(session=self.session,
                          display=self.vars["display"].get(),
                          loss_keys=loss_keys,
                          selections=selections,
                          avg_samples=self.vars["avgiterations"].get(),
                          smooth_amount=self.vars["smoothamount"].get(),
                          flatten_outliers=self.vars["outliers"].get(),
                          is_totals=self.is_totals)
            self.thread = LongRunningTask(target=self.get_display_data, kwargs=kwargs, widget=self)
            self.thread.start()
            self.after(1000, self.compile_display_data)
            return True
        if not self.thread.complete.is_set():
            logger.debug("Popup Data not yet available")
            self.after(1000, self.compile_display_data)
            return True

        logger.debug("Getting Popup from background Thread")
        self.display_data = self.thread.get_result()
        self.thread = None
        if not self.check_valid_data():
            logger.warning("No valid data to display. Not refreshing")
            self.vars["status"].set("")
            return False
        logger.debug("Compiled Display Data")
        self.vars["buildgraph"].set(True)
        return True

    @staticmethod
    def get_display_data(**kwargs):
        """ Get the display data in a LongRunningTask """
        return Calculations(**kwargs)

    def check_valid_selection(self, loss_keys, selections):
        """ Check that there will be data to display """
        display = self.vars["display"].get().lower()
        logger.debug("Validating selection. (loss_keys: %s, selections: %s, display: %s)",
                     loss_keys, selections, display)
        if not selections or (display == "loss" and not loss_keys):
            return False
        return True

    def check_valid_data(self):
        """ Check that the selections holds valid data to display
            NB: len-as-condition is used as data could be a list or a numpy array
        """
        logger.debug("Validating data. %s",
                     {key: len(val) for key, val in self.display_data.stats.items()})
        if any(len(val) == 0  # pylint:disable=len-as-condition
               for val in self.display_data.stats.values()):
            return False
        return True

    def selections_to_list(self):
        """ Compile checkbox selections to list """
        logger.debug("Compiling selections to list")
        selections = list()
        for key, val in self.vars.items():
            if (isinstance(val, tk.BooleanVar)
                    and key != "outliers"
                    and val.get()):
                selections.append(key)
        logger.debug("Compiling selections to list: %s", selections)
        return selections

    def graph_build(self, *args):  # pylint:disable=unused-argument
        """ Build the graph in the top right paned window """
        if not self.vars["buildgraph"].get():
            return
        self.vars["status"].set("Loading Data...")
        logger.debug("Building Graph")
        if self.graph is None:
            self.graph = SessionGraph(self.graph_frame,
                                      self.display_data,
                                      self.vars["display"].get(),
                                      self.vars["scale"].get())
            self.graph.pack(expand=True, fill=tk.BOTH)
            self.graph.build()
            self.graph_initialised = True
        else:
            self.graph.refresh(self.display_data,
                               self.vars["display"].get(),
                               self.vars["scale"].get())
        self.vars["status"].set("")
        self.vars["buildgraph"].set(False)
        logger.debug("Built Graph")
