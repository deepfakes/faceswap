#!/usr/bin python3
""" Command specific tabs of Display Frame of the Faceswap GUI """
import datetime
import gettext
import logging
import os
import tkinter as tk
import typing as T

from tkinter import ttk

from lib.logger import parse_class_init
from lib.training.preview_tk import PreviewTk

from .display_graph import TrainingGraph
from .display_page import DisplayOptionalPage
from .custom_widgets import Tooltip
from .analysis import Calculations, Session
from .control_helper import set_slider_rounding
from .utils import FileHandler, get_config, get_images, preview_trigger

logger = logging.getLogger(__name__)

# LOCALES
_LANG = gettext.translation("gui.tooltips", localedir="locales", fallback=True)
_ = _LANG.gettext


class PreviewExtract(DisplayOptionalPage):  # pylint:disable=too-many-ancestors
    """ Tab to display output preview images for extract and convert """
    def __init__(self, *args, **kwargs) -> None:
        logger.debug(parse_class_init(locals()))
        self._preview = get_images().preview_extract
        super().__init__(*args, **kwargs)
        logger.debug("Initialized %s", self.__class__.__name__)

    def display_item_set(self) -> None:
        """ Load the latest preview if available """
        logger.trace("Loading latest preview")  # type:ignore[attr-defined]
        size = int(256 if self.command == "convert" else 128 * get_config().scaling_factor)
        if not self._preview.load_latest_preview(thumbnail_size=size,
                                                 frame_dims=(self.winfo_width(),
                                                             self.winfo_height())):
            logger.trace("Preview not updated")  # type:ignore[attr-defined]
            return
        logger.debug("Preview loaded")
        self.display_item = True

    def display_item_process(self) -> None:
        """ Display the preview """
        logger.trace("Displaying preview")  # type:ignore[attr-defined]
        if not self.subnotebook.children:
            self.add_child()
        else:
            self.update_child()

    def add_child(self) -> None:
        """ Add the preview label child """
        logger.debug("Adding child")
        preview = self.subnotebook_add_page(self.tabname, widget=None)
        lblpreview = ttk.Label(preview, image=self._preview.image)
        lblpreview.pack(side=tk.TOP, anchor=tk.NW)
        Tooltip(lblpreview, text=self.helptext, wrap_length=200)

    def update_child(self) -> None:
        """ Update the preview image on the label """
        logger.trace("Updating preview")  # type:ignore[attr-defined]
        for widget in self.subnotebook_get_widgets():
            widget.configure(image=self._preview.image)

    def save_items(self) -> None:
        """ Open save dialogue and save preview """
        location = FileHandler("dir", None).return_file
        if not location:
            return
        filename = "extract_convert_preview"
        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(location, f"{filename}_{now}.png")
        self._preview.save(filename)
        print(f"Saved preview to {filename}")


class PreviewTrain(DisplayOptionalPage):  # pylint:disable=too-many-ancestors
    """ Training preview image(s) """
    def __init__(self, *args, **kwargs) -> None:
        logger.debug(parse_class_init(locals()))
        self._preview = get_images().preview_train
        self._display: PreviewTk | None = None
        super().__init__(*args, **kwargs)
        logger.debug("Initialized %s", self.__class__.__name__)

    def add_options(self) -> None:
        """ Add the additional options """
        self._add_option_refresh()
        self._add_option_mask_toggle()
        super().add_options()

    def subnotebook_hide(self) -> None:
        """ Override default subnotebook hide action to also remove the embedded option bar
        control and reset the training image buffer """
        if self.subnotebook and self.subnotebook.winfo_ismapped():
            logger.debug("Removing preview controls from options bar")
            if self._display is not None:
                self._display.remove_option_controls()
            super().subnotebook_hide()
            del self._display
            self._display = None
            self._preview.reset()

    def _add_option_refresh(self) -> None:
        """ Add refresh button to refresh preview immediately """
        logger.debug("Adding refresh option")
        btnrefresh = ttk.Button(self.optsframe,
                                image=get_images().icons["reload"],
                                command=lambda x="update": preview_trigger().set(x))  # type:ignore
        btnrefresh.pack(padx=2, side=tk.RIGHT)
        Tooltip(btnrefresh,
                text=_("Preview updates at every model save. Click to refresh now."),
                wrap_length=200)
        logger.debug("Added refresh option")

    def _add_option_mask_toggle(self) -> None:
        """ Add button to toggle mask display on and off """
        logger.debug("Adding mask toggle option")
        btntoggle = ttk.Button(
            self.optsframe,
            image=get_images().icons["mask2"],
            command=lambda x="mask_toggle": preview_trigger().set(x))  # type:ignore
        btntoggle.pack(padx=2, side=tk.RIGHT)
        Tooltip(btntoggle,
                text=_("Click to toggle mask overlay on and off."),
                wrap_length=200)
        logger.debug("Added mask toggle option")

    def display_item_set(self) -> None:
        """ Load the latest preview if available """
        # TODO This seems to be triggering faster than the waittime
        logger.trace("Loading latest preview")  # type:ignore[attr-defined]
        if not self._preview.load():
            logger.trace("Preview not updated")  # type:ignore[attr-defined]
            return
        logger.debug("Preview loaded")
        self.display_item = True

    def display_item_process(self) -> None:
        """ Display the preview(s) resized as appropriate """
        if self.subnotebook.children:
            return

        logger.debug("Displaying preview")
        self._display = PreviewTk(self._preview.buffer, self.subnotebook, self.optsframe, None)
        self.subnotebook_add_page(self.tabname, widget=self._display.master_frame)

    def save_items(self) -> None:
        """ Open save dialogue and save preview """
        if self._display is None:
            return

        location = FileHandler("dir", None).return_file
        if not location:
            return

        self._display.save(location)


class GraphDisplay(DisplayOptionalPage):  # pylint:disable=too-many-ancestors
    """ The Graph Tab of the Display section """
    def __init__(self,
                 parent: ttk.Notebook,
                 tab_name: str,
                 helptext: str,
                 wait_time: int,
                 command: str | None = None) -> None:
        logger.debug(parse_class_init(locals()))
        self._trace_vars: dict[T.Literal["smoothgraph", "display_iterations"],
                               tuple[tk.BooleanVar, str]] = {}
        super().__init__(parent, tab_name, helptext, wait_time, command)
        logger.debug("Initialized %s", self.__class__.__name__)

    def set_vars(self) -> None:
        """ Add graphing specific variables to the default variables.

        Overrides original method.

        Returns
        -------
        dict
            The variable names with their corresponding tkinter variable
        """
        tk_vars = super().set_vars()

        smoothgraph = tk.DoubleVar()
        smoothgraph.set(0.900)
        tk_vars["smoothgraph"] = smoothgraph

        raw_var = tk.BooleanVar()
        raw_var.set(True)
        tk_vars["raw_data"] = raw_var

        smooth_var = tk.BooleanVar()
        smooth_var.set(True)
        tk_vars["smooth_data"] = smooth_var

        iterations_var = tk.IntVar()
        iterations_var.set(10000)
        tk_vars["display_iterations"] = iterations_var

        logger.debug(tk_vars)
        return tk_vars

    def on_tab_select(self) -> None:
        """ Callback for when the graph tab is selected.

        Pull latest data and run the tab's update code when the tab is selected.
        """
        logger.debug("Callback received for '%s' tab (display_item: %s)",
                     self.tabname, self.display_item)
        if self.display_item is not None:
            get_config().tk_vars.refresh_graph.set(True)
        self._update_page()

    def add_options(self) -> None:
        """ Add the additional options """
        self._add_option_refresh()
        super().add_options()
        self._add_option_raw()
        self._add_option_smoothed()
        self._add_option_smoothing()
        self._add_option_iterations()

    def _add_option_refresh(self) -> None:
        """ Add refresh button to refresh graph immediately """
        logger.debug("Adding refresh option")
        tk_var = get_config().tk_vars.refresh_graph
        btnrefresh = ttk.Button(self.optsframe,
                                image=get_images().icons["reload"],
                                command=lambda: tk_var.set(True))
        btnrefresh.pack(padx=2, side=tk.RIGHT)
        Tooltip(btnrefresh,
                text=_("Graph updates at every model save. Click to refresh now."),
                wrap_length=200)
        logger.debug("Added refresh option")

    def _add_option_raw(self) -> None:
        """ Add check-button to hide/display raw data """
        logger.debug("Adding display raw option")
        tk_var = self.vars["raw_data"]
        chkbtn = ttk.Checkbutton(
            self.optsframe,
            variable=tk_var,
            text="Raw",
            command=lambda v=tk_var: self._display_data_callback("raw", v))  # type:ignore
        chkbtn.pack(side=tk.RIGHT, padx=5, anchor=tk.W)
        Tooltip(chkbtn, text=_("Display the raw loss data"), wrap_length=200)

    def _add_option_smoothed(self) -> None:
        """ Add check-button to hide/display smoothed data """
        logger.debug("Adding display smoothed option")
        tk_var = self.vars["smooth_data"]
        chkbtn = ttk.Checkbutton(
            self.optsframe,
            variable=tk_var,
            text="Smoothed",
            command=lambda v=tk_var: self._display_data_callback("smoothed", v))  # type:ignore
        chkbtn.pack(side=tk.RIGHT, padx=5, anchor=tk.W)
        Tooltip(chkbtn, text=_("Display the smoothed loss data"), wrap_length=200)

    def _add_option_smoothing(self) -> None:
        """ Add a slider to adjust the smoothing amount """
        logger.debug("Adding Smoothing Slider")
        tk_var = self.vars["smoothgraph"]
        min_max = (0, 0.999)
        hlp = _("Set the smoothing amount. 0 is no smoothing, 0.99 is maximum smoothing.")

        ctl_frame = ttk.Frame(self.optsframe)
        ctl_frame.pack(padx=2, side=tk.RIGHT)

        lbl = ttk.Label(ctl_frame, text="Smoothing:", anchor=tk.W)
        lbl.pack(pady=5, side=tk.LEFT, anchor=tk.N, expand=True)

        tbox = ttk.Entry(ctl_frame, width=6, textvariable=tk_var, justify=tk.RIGHT)
        tbox.pack(padx=(0, 5), side=tk.RIGHT)

        ctl = ttk.Scale(
            ctl_frame,
            variable=tk_var,
            command=lambda val, var=tk_var, dt=float, rn=3, mm=min_max:  # type:ignore
            set_slider_rounding(val, var, dt, rn, mm))
        ctl["from_"] = min_max[0]
        ctl["to"] = min_max[1]
        ctl.pack(padx=5, pady=5, fill=tk.X, expand=True)
        for item in (tbox, ctl):
            Tooltip(item,
                    text=hlp,
                    wrap_length=200)
        logger.debug("Added Smoothing Slider")

    def _add_option_iterations(self) -> None:
        """ Add a slider to adjust the amount if iterations to display """
        logger.debug("Adding Iterations Slider")
        tk_var = self.vars["display_iterations"]
        min_max = (0, 100000)
        hlp = _("Set the number of iterations to display. 0 displays the full session.")

        ctl_frame = ttk.Frame(self.optsframe)
        ctl_frame.pack(padx=2, side=tk.RIGHT)

        lbl = ttk.Label(ctl_frame, text="Iterations:", anchor=tk.W)
        lbl.pack(pady=5, side=tk.LEFT, anchor=tk.N, expand=True)

        tbox = ttk.Entry(ctl_frame, width=6, textvariable=tk_var, justify=tk.RIGHT)
        tbox.pack(padx=(0, 5), side=tk.RIGHT)

        ctl = ttk.Scale(
            ctl_frame,
            variable=tk_var,
            command=lambda val, var=tk_var, dt=int, rn=1000, mm=min_max:  # type:ignore
            set_slider_rounding(val, var, dt, rn, mm))
        ctl["from_"] = min_max[0]
        ctl["to"] = min_max[1]
        ctl.pack(padx=5, pady=5, fill=tk.X, expand=True)
        for item in (tbox, ctl):
            Tooltip(item,
                    text=hlp,
                    wrap_length=200)
        logger.debug("Added Iterations Slider")

    def display_item_set(self) -> None:
        """ Load the graph(s) if available """
        if Session.is_training and Session.logging_disabled:
            logger.trace("Logs disabled. Hiding graph")  # type:ignore[attr-defined]
            self.set_info("Graph is disabled as 'no-logs' has been selected")
            self.display_item = None
            self._clear_trace_variables()
        elif Session.is_training and self.display_item is None:
            logger.trace("Loading graph")  # type:ignore[attr-defined]
            self.display_item = Session
            self._add_trace_variables()
        elif Session.is_training and self.display_item is not None:
            logger.trace("Graph already displayed. Nothing to do.")  # type:ignore[attr-defined]
        else:
            logger.trace("Clearing graph")  # type:ignore[attr-defined]
            self.display_item = None
            self._clear_trace_variables()

    def display_item_process(self) -> None:
        """ Add a single graph to the graph window """
        if not Session.is_training:
            logger.debug("Waiting for Session Data to become available to graph")
            self.after(1000, self.display_item_process)
            return

        existing = list(self.subnotebook_get_titles_ids().keys())

        loss_keys = self.display_item.get_loss_keys(Session.session_ids[-1])
        if not loss_keys:
            # Reload if we attempt to get loss keys before data is written
            logger.debug("Waiting for Session Data to become available to graph")
            self.after(1000, self.display_item_process)
            return

        loss_keys = [key for key in loss_keys if key != "total"]
        display_tabs = sorted(set(key[:-1].rstrip("_") for key in loss_keys))

        for loss_key in display_tabs:
            tabname = loss_key.replace("_", " ").title()
            if tabname in existing:
                continue
            logger.debug("Adding graph '%s'", tabname)

            display_keys = [key for key in loss_keys if key.startswith(loss_key)]
            data = Calculations(session_id=Session.session_ids[-1],
                                display="loss",
                                loss_keys=display_keys,
                                selections=["raw", "smoothed"],
                                smooth_amount=self.vars["smoothgraph"].get())
            self.add_child(tabname, data)

    def _smooth_amount_callback(self, *args) -> None:
        """ Update each graph's smooth amount on variable change """
        try:
            smooth_amount = self.vars["smoothgraph"].get()
        except tk.TclError:
            # Don't update when there is no value in the variable
            return
        logger.debug("Updating graph smooth_amount: (new_value: %s, args: %s)",
                     smooth_amount, args)
        for graph in self.subnotebook.children.values():
            graph.calcs.set_smooth_amount(smooth_amount)

    def _iteration_limit_callback(self, *args) -> None:
        """ Limit the amount of data displayed in the live graph on a iteration slider
        variable change. """
        try:
            limit = self.vars["display_iterations"].get()
        except tk.TclError:
            # Don't update when there is no value in the variable
            return
        logger.debug("Updating graph iteration limit: (new_value: %s, args: %s)",
                     limit, args)
        for graph in self.subnotebook.children.values():
            graph.calcs.set_iterations_limit(limit)

    def _display_data_callback(self, line: str, variable: tk.BooleanVar) -> None:
        """ Update the displayed graph lines based on option check button selection.

        Parameters
        ----------
        line: str
            The line to hide or display
        variable: :class:`tkinter.BooleanVar`
            The tkinter variable containing the ``True`` or ``False`` data for this display item
        """
        var = variable.get()
        logger.debug("Updating display %s to %s", line, var)
        for graph in self.subnotebook.children.values():
            graph.calcs.update_selections(line, var)

    def add_child(self, name: str, data: Calculations) -> None:
        """ Add the graph for the selected keys.

        Parameters
        ----------
        name: str
            The name of the graph to add to the notebook
        data: :class:`~lib.gui.analysis.stats.Calculations`
            The object holding the data to be graphed
        """
        logger.debug("Adding child: %s", name)
        graph = TrainingGraph(self.subnotebook, data, "Loss")
        graph.build()
        graph = self.subnotebook_add_page(name, widget=graph)
        Tooltip(graph, text=self.helptext, wrap_length=200)

    def save_items(self) -> None:
        """ Open save dialogue and save graphs """
        graphlocation = FileHandler("dir", None).return_file
        if not graphlocation:
            return
        for graph in self.subnotebook.children.values():
            graph.save_fig(graphlocation)

    def _add_trace_variables(self) -> None:
        """ Add tracing for when the option sliders are updated, for updating the graph. """
        for name, action in zip(T.get_args(T.Literal["smoothgraph", "display_iterations"]),
                                (self._smooth_amount_callback, self._iteration_limit_callback)):
            var = self.vars[name]
            if name not in self._trace_vars:
                self._trace_vars[name] = (var, var.trace("w", action))

    def _clear_trace_variables(self) -> None:
        """ Clear all of the trace variables from :attr:`_trace_vars` and reset the dictionary. """
        if self._trace_vars:
            for name, (var, trace) in self._trace_vars.items():
                logger.debug("Clearing trace from variable: %s", name)
                var.trace_vdelete("w", trace)
            self._trace_vars = {}

    def close(self) -> None:
        """ Clear the plots from RAM """
        self._clear_trace_variables()
        if self.subnotebook is None:
            logger.debug("No graphs to clear. Returning")
            return

        for name, graph in self.subnotebook.children.items():
            logger.debug("Clearing: %s", name)
            graph.clear()
        super().close()
