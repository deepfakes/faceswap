#!/usr/bin python3
""" Pop-up Graph launched from the Analysis tab of the Faceswap GUI """

import csv
import logging
import tkinter as tk
from tkinter import ttk

from .control_helper import ControlBuilder, ControlPanelOption
from .custom_widgets import Tooltip
from .display_graph import SessionGraph
from .stats import Calculations, Session
from .utils import FileHandler, get_images, LongRunningTask

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class SessionPopUp(tk.Toplevel):
    """ Pop up for detailed graph/stats for selected session.

    session_id: int or `"Total"`
        The session id number for the selected session from the Analysis tab. Should be the string
        `"Total"` if all sessions are being graphed
    data_points: int
        The number of iterations in the selected session
    """
    def __init__(self, session_id, data_points):
        logger.debug("Initializing: %s: (session_id: %s, data_points: %s)",
                     self.__class__.__name__, session_id, data_points)
        super().__init__()
        self._thread = None  # Thread for loading data in a background task
        self._default_view = "avg" if data_points > 1000 else "smoothed"
        self._session_id = None if session_id == "Total" else int(session_id)

        self._graph_frame = None
        self._graph = None
        self._display_data = None

        self._vars = self._set_vars()
        self._graph_initialised = False

        optsframe = self._layout_frames()
        self._build_options(optsframe)

        self._lbl_loading = ttk.Label(self._graph_frame, text="Loading Data...", anchor=tk.CENTER)
        self._lbl_loading.pack(fill=tk.BOTH, expand=True)
        self.update_idletasks()

        self._compile_display_data()

        logger.debug("Initialized: %s", self.__class__.__name__)

    def _set_vars(self):
        """ Set status tkinter String variable and tkinter Boolean variable to callback when the
        graph is ready to build.

        Returns
        -------
        dict
            The tkinter Variables for the pop up graph
        """
        logger.debug("Setting tk graph build variable and internal variables")

        retval = dict(status=tk.StringVar())

        var = tk.BooleanVar()
        var.set(False)
        var.trace("w", self._graph_build)

        retval["buildgraph"] = var
        return retval

    def _layout_frames(self):
        """ Top level container frames """
        logger.debug("Layout frames")

        leftframe = ttk.Frame(self)
        sep = ttk.Frame(self, width=2, relief=tk.RIDGE)
        self._graph_frame = ttk.Frame(self)

        self._graph_frame.pack(side=tk.RIGHT, fill=tk.BOTH, pady=5, expand=True)
        sep.pack(fill=tk.Y, side=tk.LEFT)
        leftframe.pack(side=tk.LEFT, expand=False, fill=tk.BOTH, pady=5)

        logger.debug("Laid out frames")

        return leftframe

    def _build_options(self, frame):
        """ Build Options into the options frame.

        Parameters
        ----------
        frame: `tkinter.ttk.Frame`
            The frame that the options reside in
        """
        logger.debug("Building Options")
        self._opts_combobox(frame)
        self._opts_checkbuttons(frame)
        self._opts_loss_keys(frame)
        self._opts_slider(frame)
        self._opts_buttons(frame)
        sep = ttk.Frame(frame, height=2, relief=tk.RIDGE)
        sep.pack(fill=tk.X, pady=(5, 0), side=tk.BOTTOM)
        logger.debug("Built Options")

    def _opts_combobox(self, frame):
        """ Add the options combo boxes.

        Parameters
        ----------
        frame: `tkinter.ttk.Frame`
            The frame that the options reside in
        """
        logger.debug("Building Combo boxes")
        choices = dict(Display=("Loss", "Rate"), Scale=("Linear", "Log"))

        for item in ["Display", "Scale"]:
            var = tk.StringVar()

            cmbframe = ttk.Frame(frame)
            lblcmb = ttk.Label(cmbframe, text="{}:".format(item), width=7, anchor=tk.W)
            cmb = ttk.Combobox(cmbframe, textvariable=var, width=10)
            cmb["values"] = choices[item]
            cmb.current(0)

            cmd = self._option_button_reload if item == "Display" else self._graph_scale
            var.trace("w", cmd)
            self._vars[item.lower().strip()] = var

            hlp = self._set_help(item)
            Tooltip(cmbframe, text=hlp, wraplength=200)

            cmb.pack(fill=tk.X, side=tk.RIGHT)
            lblcmb.pack(padx=(0, 2), side=tk.LEFT)
            cmbframe.pack(fill=tk.X, pady=5, padx=5, side=tk.TOP)
        logger.debug("Built Combo boxes")

    def _opts_checkbuttons(self, frame):
        """ Add the options check buttons.

        Parameters
        ----------
        frame: `tkinter.ttk.Frame`
            The frame that the options reside in
        """
        logger.debug("Building Check Buttons")
        self._add_section(frame, "Display")
        for item in ("raw", "trend", "avg", "smoothed", "outliers"):
            if item == "avg":
                text = "Show Rolling Average"
            elif item == "outliers":
                text = "Flatten Outliers"
            else:
                text = "Show {}".format(item.title())

            var = tk.BooleanVar()
            if item == self._default_view:
                var.set(True)
            self._vars[item] = var

            ctl = ttk.Checkbutton(frame, variable=var, text=text)
            hlp = self._set_help(item)
            Tooltip(ctl, text=hlp, wraplength=200)
            ctl.pack(side=tk.TOP, padx=5, pady=5, anchor=tk.W)

        logger.debug("Built Check Buttons")

    def _opts_loss_keys(self, frame):
        """ Add loss key selections.

        Parameters
        ----------
        frame: `tkinter.ttk.Frame`
            The frame that the options reside in
        """
        logger.debug("Building Loss Key Check Buttons")
        loss_keys = Session.get_loss_keys(self._session_id)
        lk_vars = dict()
        section_added = False
        for loss_key in sorted(loss_keys):
            if loss_key.startswith("total"):
                continue

            text = loss_key.replace("_", " ").title()
            helptext = "Display {}".format(text)

            var = tk.BooleanVar()
            var.set(True)
            lk_vars[loss_key] = var

            if len(loss_keys) == 1:
                # Don't display if there's only one item
                break

            if not section_added:
                self._add_section(frame, "Keys")
                section_added = True

            ctl = ttk.Checkbutton(frame, variable=var, text=text)
            Tooltip(ctl, text=helptext, wraplength=200)
            ctl.pack(side=tk.TOP, padx=5, pady=5, anchor=tk.W)

        self._vars["loss_keys"] = lk_vars
        logger.debug("Built Loss Key Check Buttons")

    def _opts_slider(self, frame):
        """ Add the options entry boxes.

        Parameters
        ----------
        frame: `tkinter.ttk.Frame`
            The frame that the options reside in
        """

        self._add_section(frame, "Parameters")
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
                                        helptext=self._set_help(item))
            self._vars[item] = slider.tk_var
            ControlBuilder(frame, slider, 1, 19, None, True)
        logger.debug("Built Sliders")

    def _opts_buttons(self, frame):
        """ Add the option buttons.

        Parameters
        ----------
        frame: `tkinter.ttk.Frame`
            The frame that the options reside in
        """
        logger.debug("Building Buttons")
        btnframe = ttk.Frame(frame)
        lblstatus = ttk.Label(btnframe,
                              width=40,
                              textvariable=self._vars["status"],
                              anchor=tk.W)

        for btntype in ("reload", "save"):
            cmd = getattr(self, "_option_button_{}".format(btntype))
            btn = ttk.Button(btnframe,
                             image=get_images().icons[btntype],
                             command=cmd)
            hlp = self._set_help(btntype)
            Tooltip(btn, text=hlp, wraplength=200)
            btn.pack(padx=2, side=tk.RIGHT)

        lblstatus.pack(side=tk.LEFT, anchor=tk.W, fill=tk.X, expand=True)
        btnframe.pack(fill=tk.X, pady=5, padx=5, side=tk.BOTTOM)
        logger.debug("Built Buttons")

    @staticmethod
    def _add_section(frame, title):
        """ Add a separator and section title between options

        Parameters
        ----------
        title: str
            The section title to display
        """
        sep = ttk.Frame(frame, height=2, relief=tk.SOLID)
        lbl = ttk.Label(frame, text=title)

        lbl.pack(side=tk.TOP, padx=5, pady=0, anchor=tk.CENTER)
        sep.pack(fill=tk.X, pady=(5, 0), side=tk.TOP)

    def _option_button_save(self):
        """ Action for save button press. """
        logger.debug("Saving File")
        savefile = FileHandler("save", "csv").retfile
        if not savefile:
            logger.debug("Save Cancelled")
            return
        logger.debug("Saving to: %s", savefile)
        save_data = self._display_data.stats
        fieldnames = sorted(key for key in save_data.keys())

        with savefile as outfile:
            csvout = csv.writer(outfile, delimiter=",")
            csvout.writerow(fieldnames)
            csvout.writerows(zip(*[save_data[key] for key in fieldnames]))

    def _option_button_reload(self, *args):  # pylint: disable=unused-argument
        """ Action for reset button press and checkbox changes. """
        logger.debug("Refreshing Graph")
        if not self._graph_initialised:
            return
        valid = self._compile_display_data()
        if not valid:
            logger.debug("Invalid data")
            return
        self._graph.refresh(self._display_data,
                            self._vars["display"].get(),
                            self._vars["scale"].get())
        logger.debug("Refreshed Graph")

    def _graph_scale(self, *args):  # pylint: disable=unused-argument
        """ Action for changing graph scale. """
        if not self._graph_initialised:
            return
        self._graph.set_yscale_type(self._vars["scale"].get())

    @classmethod
    def _set_help(cls, action):
        """ Set the help text for option buttons.

        Parameters
        ----------
        action: string
            The action to get the help text for

        Returns
        -------
        str
            The help text for the given action
        """
        hlp = ""
        action = action.lower()
        if action == "reload":
            hlp = "Refresh graph"
        elif action == "save":
            hlp = "Save display data to csv"
        elif action == "avgiterations":
            hlp = "Number of data points to sample for rolling average"
        elif action == "smoothamount":
            hlp = "Set the smoothing amount. 0 is no smoothing, 0.99 is maximum smoothing"
        elif action == "outliers":
            hlp = "Flatten data points that fall more than 1 standard " \
                  "deviation from the mean to the mean value."
        elif action == "avg":
            hlp = "Display rolling average of the data"
        elif action == "smoothed":
            hlp = "Smooth the data"
        elif action == "raw":
            hlp = "Display raw data"
        elif action == "trend":
            hlp = "Display polynormal data trend"
        elif action == "display":
            hlp = "Set the data to display"
        elif action == "scale":
            hlp = "Change y-axis scale"
        return hlp

    def _compile_display_data(self):
        """ Compile the data to be displayed. """
        if self._thread is None:
            logger.debug("Compiling Display Data in background thread")
            loss_keys = [key for key, val in self._vars["loss_keys"].items()
                         if val.get()]
            logger.debug("Selected loss_keys: %s", loss_keys)

            selections = self._selections_to_list()

            if not self._check_valid_selection(loss_keys, selections):
                logger.warning("No data to display. Not refreshing")
                return False
            self._vars["status"].set("Loading Data...")

            if self._graph is not None:
                self._graph.pack_forget()
            self._lbl_loading.pack(fill=tk.BOTH, expand=True)
            self.update_idletasks()

            kwargs = dict(session_id=self._session_id,
                          display=self._vars["display"].get(),
                          loss_keys=loss_keys,
                          selections=selections,
                          avg_samples=self._vars["avgiterations"].get(),
                          smooth_amount=self._vars["smoothamount"].get(),
                          flatten_outliers=self._vars["outliers"].get())
            self._thread = LongRunningTask(target=self._get_display_data,
                                           kwargs=kwargs,
                                           widget=self)
            self._thread.start()
            self.after(1000, self._compile_display_data)
            return True
        if not self._thread.complete.is_set():
            logger.debug("Popup Data not yet available")
            self.after(1000, self._compile_display_data)
            return True

        logger.debug("Getting Popup from background Thread")
        self._display_data = self._thread.get_result()
        self._thread = None
        if not self._check_valid_data():
            logger.warning("No valid data to display. Not refreshing")
            self._vars["status"].set("")
            return False
        logger.debug("Compiled Display Data")
        self._vars["buildgraph"].set(True)
        return True

    @staticmethod
    def _get_display_data(**kwargs):
        """ Get the display data in a LongRunningTask.

        Parameters
        ----------
        kwargs: dict
            The keyword arguments to pass to `lib.gui.stats.Calculations`

        Returns
        -------
        :class:`lib.gui.stats.Calculations`
            The summarized results for the given session
        """
        return Calculations(**kwargs)

    def _check_valid_selection(self, loss_keys, selections):
        """ Check that there will be data to display.

        Parameters
        ----------
        loss_keys: list
            The selected loss to display
        selections: list
            The selected checkbox options

        Returns
        -------
        bool
            ``True` if there is data to be displayed, otherwise ``False``
        """
        display = self._vars["display"].get().lower()
        logger.debug("Validating selection. (loss_keys: %s, selections: %s, display: %s)",
                     loss_keys, selections, display)
        if not selections or (display == "loss" and not loss_keys):
            return False
        return True

    def _check_valid_data(self):
        """ Check that the selections holds valid data to display
            NB: len-as-condition is used as data could be a list or a numpy array
        """
        logger.debug("Validating data. %s",
                     {key: len(val) for key, val in self._display_data.stats.items()})
        if any(len(val) == 0  # pylint:disable=len-as-condition
               for val in self._display_data.stats.values()):
            return False
        return True

    def _selections_to_list(self):
        """ Compile checkbox selections to a list.

        Returns
        -------
        list
            The selected options from the check-boxes
        """
        logger.debug("Compiling selections to list")
        selections = list()
        for key, val in self._vars.items():
            if (isinstance(val, tk.BooleanVar)
                    and key != "outliers"
                    and val.get()):
                selections.append(key)
        logger.debug("Compiling selections to list: %s", selections)
        return selections

    def _graph_build(self, *args):  # pylint:disable=unused-argument
        """ Build the graph in the top right paned window """
        if not self._vars["buildgraph"].get():
            return
        self._vars["status"].set("Loading Data...")
        logger.debug("Building Graph")
        self._lbl_loading.pack_forget()
        self.update_idletasks()
        if self._graph is None:
            self._graph = SessionGraph(self._graph_frame,
                                       self._display_data,
                                       self._vars["display"].get(),
                                       self._vars["scale"].get())
            self._graph.pack(expand=True, fill=tk.BOTH)
            self._graph.build()
            self._graph_initialised = True
        else:
            self._graph.refresh(self._display_data,
                                self._vars["display"].get(),
                                self._vars["scale"].get())
            self._graph.pack(fill=tk.BOTH, expand=True)
        self._vars["status"].set("")
        self._vars["buildgraph"].set(False)
        logger.debug("Built Graph")
