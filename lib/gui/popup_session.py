#!/usr/bin python3
""" Pop-up Graph launched from the Analysis tab of the Faceswap GUI """

import csv
import gettext
import logging
import tkinter as tk

from dataclasses import dataclass, field
from tkinter import ttk

from .control_helper import ControlBuilder, ControlPanelOption
from .custom_widgets import Tooltip
from .display_graph import SessionGraph
from .analysis import Calculations, Session
from .utils import FileHandler, get_images, LongRunningTask

logger = logging.getLogger(__name__)

# LOCALES
_LANG = gettext.translation("gui.tooltips", localedir="locales", fallback=True)
_ = _LANG.gettext


@dataclass
class SessionTKVars:
    """ Dataclass for holding the tk variables required for the session popup

    Parameters
    ----------
    buildgraph: :class:`tkinter.BooleanVar`
        Trigger variable to indicate the graph should be rebuilt
    status: :class:`tkinter.StringVar`
        The variable holding the current status of the popup window
    display: :class:`tkinter.StringVar`
        Variable indicating the type of information to be displayed
    scale: :class:`tkinter.StringVar`
        Variable indicating whether to display as log or linear data
    raw: :class:`tkinter.BooleanVar`
        Variable to indicate raw data should be displayed
    trend: :class:`tkinter.BooleanVar`
        Variable to indicate that trend data should be displayed
    avg: :class:`tkinter.BooleanVar`
        Variable to indicate that rolling average data should be displayed
    smoothed: :class:`tkinter.BooleanVar`
        Variable to indicate that smoothed data should be displayed
    outliers: :class:`tkinter.BooleanVar`
        Variable to indicate that outliers should be displayed
    loss_keys: dict
        Dictionary of names to :class:`tkinter.BooleanVar` indicating whether specific loss items
        should be displayed
    avgiterations: :class:`tkinter.IntVar`
        The number of iterations to use for rolling average
    smoothamount: :class:`tkinter.DoubleVar`
        The amount of smoothing to apply for smoothed data
    """
    buildgraph: tk.BooleanVar
    status: tk.StringVar
    display: tk.StringVar
    scale: tk.StringVar
    raw: tk.BooleanVar
    trend: tk.BooleanVar
    avg: tk.BooleanVar
    smoothed: tk.BooleanVar
    outliers: tk.BooleanVar
    avgiterations: tk.IntVar
    smoothamount: tk.DoubleVar
    loss_keys: dict[str, tk.BooleanVar] = field(default_factory=dict)


class SessionPopUp(tk.Toplevel):
    """ Pop up for detailed graph/stats for selected session.

    session_id: int or `"Total"`
        The session id number for the selected session from the Analysis tab. Should be the string
        `"Total"` if all sessions are being graphed
    data_points: int
        The number of iterations in the selected session
    """
    def __init__(self, session_id: int, data_points: int) -> None:
        logger.debug("Initializing: %s: (session_id: %s, data_points: %s)",
                     self.__class__.__name__, session_id, data_points)
        super().__init__()
        self._thread: LongRunningTask | None = None  # Thread for loading data in background
        self._default_view = "avg" if data_points > 1000 else "smoothed"
        self._session_id = None if session_id == "Total" else int(session_id)

        self._graph_frame = ttk.Frame(self)
        self._graph: SessionGraph | None = None
        self._display_data: Calculations | None = None

        self._vars = self._set_vars()

        self._graph_initialised = False

        optsframe = self._layout_frames()
        self._build_options(optsframe)

        self._lbl_loading = ttk.Label(self._graph_frame, text="Loading Data...", anchor=tk.CENTER)
        self._lbl_loading.pack(fill=tk.BOTH, expand=True)
        self.update_idletasks()

        self._compile_display_data()

        logger.debug("Initialized: %s", self.__class__.__name__)

    def _set_vars(self) -> SessionTKVars:
        """ Set status tkinter String variable and tkinter Boolean variable to callback when the
        graph is ready to build.

        Returns
        -------
        :class:`SessionTKVars`
            The tkinter Variables for the pop up graph
        """
        logger.debug("Setting tk graph build variable and internal variables")
        retval = SessionTKVars(buildgraph=tk.BooleanVar(),
                               status=tk.StringVar(),
                               display=tk.StringVar(),
                               scale=tk.StringVar(),
                               raw=tk.BooleanVar(),
                               trend=tk.BooleanVar(),
                               avg=tk.BooleanVar(),
                               smoothed=tk.BooleanVar(),
                               outliers=tk.BooleanVar(),
                               avgiterations=tk.IntVar(),
                               smoothamount=tk.DoubleVar())
        retval.buildgraph.set(False)
        retval.buildgraph.trace("w", self._graph_build)
        return retval

    def _layout_frames(self) -> ttk.Frame:
        """ Top level container frames """
        logger.debug("Layout frames")

        leftframe = ttk.Frame(self)
        sep = ttk.Frame(self, width=2, relief=tk.RIDGE)

        self._graph_frame.pack(side=tk.RIGHT, fill=tk.BOTH, pady=5, expand=True)
        sep.pack(fill=tk.Y, side=tk.LEFT)
        leftframe.pack(side=tk.LEFT, expand=False, fill=tk.BOTH, pady=5)

        logger.debug("Laid out frames")

        return leftframe

    def _build_options(self, frame: ttk.Frame) -> None:
        """ Build Options into the options frame.

        Parameters
        ----------
        frame: :class:`tkinter.ttk.Frame`
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

    def _opts_combobox(self, frame: ttk.Frame) -> None:
        """ Add the options combo boxes.

        Parameters
        ----------
        frame: :class:`tkinter.ttk.Frame`
            The frame that the options reside in
        """
        logger.debug("Building Combo boxes")
        choices = {"Display": ("Loss", "Rate"), "Scale": ("Linear", "Log")}

        for item in ["Display", "Scale"]:
            var: tk.StringVar = getattr(self._vars, item.lower())

            cmbframe = ttk.Frame(frame)
            lblcmb = ttk.Label(cmbframe, text=f"{item}:", width=7, anchor=tk.W)
            cmb = ttk.Combobox(cmbframe, textvariable=var, width=10)
            cmb["values"] = choices[item]
            cmb.current(0)

            cmd = self._option_button_reload if item == "Display" else self._graph_scale
            var.trace("w", cmd)
            hlp = self._set_help(item)
            Tooltip(cmbframe, text=hlp, wrap_length=200)

            cmb.pack(fill=tk.X, side=tk.RIGHT)
            lblcmb.pack(padx=(0, 2), side=tk.LEFT)
            cmbframe.pack(fill=tk.X, pady=5, padx=5, side=tk.TOP)
        logger.debug("Built Combo boxes")

    def _opts_checkbuttons(self, frame: ttk.Frame) -> None:
        """ Add the options check buttons.

        Parameters
        ----------
        frame: :class:`tkinter.ttk.Frame`
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
                text = f"Show {item.title()}"

            var: tk.BooleanVar = getattr(self._vars, item)
            if item == self._default_view:
                var.set(True)

            ctl = ttk.Checkbutton(frame, variable=var, text=text)
            hlp = self._set_help(item)
            Tooltip(ctl, text=hlp, wrap_length=200)
            ctl.pack(side=tk.TOP, padx=5, pady=5, anchor=tk.W)

        logger.debug("Built Check Buttons")

    def _opts_loss_keys(self, frame: ttk.Frame) -> None:
        """ Add loss key selections.

        Parameters
        ----------
        frame: :class:`tkinter.ttk.Frame`
            The frame that the options reside in
        """
        logger.debug("Building Loss Key Check Buttons")
        loss_keys = Session.get_loss_keys(self._session_id)
        lk_vars = {}
        section_added = False
        for loss_key in sorted(loss_keys):
            if loss_key.startswith("total"):
                continue

            text = loss_key.replace("_", " ").title()
            helptext = _("Display {}").format(text)

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
            Tooltip(ctl, text=helptext, wrap_length=200)
            ctl.pack(side=tk.TOP, padx=5, pady=5, anchor=tk.W)

        self._vars.loss_keys = lk_vars
        logger.debug("Built Loss Key Check Buttons")

    def _opts_slider(self, frame: ttk.Frame) -> None:
        """ Add the options entry boxes.

        Parameters
        ----------
        frame: :class:`tkinter.ttk.Frame`
            The frame that the options reside in
        """

        self._add_section(frame, "Parameters")
        logger.debug("Building Slider Controls")
        for item in ("avgiterations", "smoothamount"):
            if item == "avgiterations":
                dtype: type[int] | type[float] = int
                text = "Iterations to Average:"
                default: int | float = 500
                rounding = 25
                min_max: tuple[int, int | float] = (25, 2500)
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
            setattr(self._vars, item, slider.tk_var)
            ControlBuilder(frame, slider, 1, 19, None, "Analysis.", True)
        logger.debug("Built Sliders")

    def _opts_buttons(self, frame: ttk.Frame) -> None:
        """ Add the option buttons.

        Parameters
        ----------
        frame: :class:`tkinter.ttk.Frame`
            The frame that the options reside in
        """
        logger.debug("Building Buttons")
        btnframe = ttk.Frame(frame)
        lblstatus = ttk.Label(btnframe,
                              width=40,
                              textvariable=self._vars.status,
                              anchor=tk.W)

        for btntype in ("reload", "save"):
            cmd = getattr(self, f"_option_button_{btntype}")
            btn = ttk.Button(btnframe,
                             image=get_images().icons[btntype],
                             command=cmd)
            hlp = self._set_help(btntype)
            Tooltip(btn, text=hlp, wrap_length=200)
            btn.pack(padx=2, side=tk.RIGHT)

        lblstatus.pack(side=tk.LEFT, anchor=tk.W, fill=tk.X, expand=True)
        btnframe.pack(fill=tk.X, pady=5, padx=5, side=tk.BOTTOM)
        logger.debug("Built Buttons")

    @classmethod
    def _add_section(cls, frame: ttk.Frame, title: str) -> None:
        """ Add a separator and section title between options

        Parameters
        ----------
        frame: :class:`tkinter.ttk.Frame`
            The frame that the options reside in
        title: str
            The section title to display
        """
        sep = ttk.Frame(frame, height=2, relief=tk.SOLID)
        lbl = ttk.Label(frame, text=title)

        lbl.pack(side=tk.TOP, padx=5, pady=0, anchor=tk.CENTER)
        sep.pack(fill=tk.X, pady=(5, 0), side=tk.TOP)

    def _option_button_save(self) -> None:
        """ Action for save button press. """
        logger.debug("Saving File")
        savefile = FileHandler("save", "csv").return_file
        if not savefile:
            logger.debug("Save Cancelled")
            return
        logger.debug("Saving to: %s", savefile)
        assert self._display_data is not None
        save_data = self._display_data.stats
        fieldnames = sorted(key for key in save_data.keys())

        with savefile as outfile:
            csvout = csv.writer(outfile, delimiter=",")
            csvout.writerow(fieldnames)
            csvout.writerows(zip(*[save_data[key] for key in fieldnames]))

    def _option_button_reload(self, *args) -> None:  # pylint:disable=unused-argument
        """ Action for reset button press and checkbox changes.

        Parameters
        ----------
        args: tuple
            Required for TK Callback but unused
        """
        logger.debug("Refreshing Graph")
        if not self._graph_initialised:
            return
        valid = self._compile_display_data()
        if not valid:
            logger.debug("Invalid data")
            return
        assert self._graph is not None
        self._graph.refresh(self._display_data,
                            self._vars.display.get(),
                            self._vars.scale.get())
        logger.debug("Refreshed Graph")

    def _graph_scale(self, *args) -> None:  # pylint:disable=unused-argument
        """ Action for changing graph scale.

        Parameters
        ----------
        args: tuple
            Required for TK Callback but unused
        """
        assert self._graph is not None
        if not self._graph_initialised:
            return
        self._graph.set_yscale_type(self._vars.scale.get())

    @classmethod
    def _set_help(cls, action: str) -> str:
        """ Set the help text for option buttons.

        Parameters
        ----------
        action: str
            The action to get the help text for

        Returns
        -------
        str
            The help text for the given action
        """
        lookup = {
            "reload": _("Refresh graph"),
            "save": _("Save display data to csv"),
            "avgiterations": _("Number of data points to sample for rolling average"),
            "smoothamount": _("Set the smoothing amount. 0 is no smoothing, 0.99 is maximum "
                              "smoothing"),
            "outliers": _("Flatten data points that fall more than 1 standard deviation from the "
                          "mean to the mean value."),
            "avg": _("Display rolling average of the data"),
            "smoothed": _("Smooth the data"),
            "raw": _("Display raw data"),
            "trend": _("Display polynormal data trend"),
            "display": _("Set the data to display"),
            "scale": _("Change y-axis scale")}
        return lookup.get(action.lower(), "")

    def _compile_display_data(self) -> bool:
        """ Compile the data to be displayed.

        Returns
        -------
        bool
            ``True`` if there is valid data to display, ``False`` if not
        """
        if self._thread is None:
            logger.debug("Compiling Display Data in background thread")
            loss_keys = [key for key, val in self._vars.loss_keys.items()
                         if val.get()]
            logger.debug("Selected loss_keys: %s", loss_keys)

            selections = self._selections_to_list()

            if not self._check_valid_selection(loss_keys, selections):
                logger.warning("No data to display. Not refreshing")
                return False
            self._vars.status.set("Loading Data...")

            if self._graph is not None:
                self._graph.pack_forget()
            self._lbl_loading.pack(fill=tk.BOTH, expand=True)
            self.update_idletasks()

            kwargs = {"session_id": self._session_id,
                      "display": self._vars.display.get(),
                      "loss_keys": loss_keys,
                      "selections": selections,
                      "avg_samples": self._vars.avgiterations.get(),
                      "smooth_amount": self._vars.smoothamount.get(),
                      "flatten_outliers": self._vars.outliers.get()}
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
            self._vars.status.set("")
            return False
        logger.debug("Compiled Display Data")
        self._vars.buildgraph.set(True)
        return True

    @classmethod
    def _get_display_data(cls, **kwargs) -> Calculations:
        """ Get the display data in a LongRunningTask.

        Parameters
        ----------
        kwargs: dict
            The keyword arguments to pass to `lib.gui.analysis.Calculations`

        Returns
        -------
        :class:`lib.gui.analysis.Calculations`
            The summarized results for the given session
        """
        return Calculations(**kwargs)

    def _check_valid_selection(self, loss_keys: list[str], selections: list[str]) -> bool:
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
        display = self._vars.display.get().lower()
        logger.debug("Validating selection. (loss_keys: %s, selections: %s, display: %s)",
                     loss_keys, selections, display)
        if not selections or (display == "loss" and not loss_keys):
            return False
        return True

    def _check_valid_data(self) -> bool:
        """ Check that the selections holds valid data to display
            NB: len-as-condition is used as data could be a list or a numpy array

        Returns
        -------
        bool
            ``True` if there is data to be displayed, otherwise ``False``
        """
        assert self._display_data is not None
        logger.debug("Validating data. %s",
                     {key: len(val) for key, val in self._display_data.stats.items()})
        if any(len(val) == 0  # pylint:disable=len-as-condition
               for val in self._display_data.stats.values()):
            return False
        return True

    def _selections_to_list(self) -> list[str]:
        """ Compile checkbox selections to a list.

        Returns
        -------
        list
            The selected options from the check-boxes
        """
        logger.debug("Compiling selections to list")
        selections = []
        for item in ("raw", "trend", "avg", "smoothed"):
            var: tk.BooleanVar = getattr(self._vars, item)
            if var.get():
                selections.append(item)
        logger.debug("Compiling selections to list: %s", selections)
        return selections

    def _graph_build(self, *args) -> None:  # pylint:disable=unused-argument
        """ Build the graph in the top right paned window

        Parameters
        ----------
        args: tuple
            Required for TK Callback but unused
        """
        if not self._vars.buildgraph.get():
            return
        self._vars.status.set("Loading Data...")
        logger.debug("Building Graph")
        self._lbl_loading.pack_forget()
        self.update_idletasks()
        if self._graph is None:
            graph = SessionGraph(self._graph_frame,
                                 self._display_data,
                                 self._vars.display.get(),
                                 self._vars.scale.get())
            graph.pack(expand=True, fill=tk.BOTH)
            graph.build()
            self._graph = graph
            self._graph_initialised = True
        else:
            self._graph.refresh(self._display_data,
                                self._vars.display.get(),
                                self._vars.scale.get())
            self._graph.pack(fill=tk.BOTH, expand=True)
        self._vars.status.set("")
        self._vars.buildgraph.set(False)
        logger.debug("Built Graph")
