#!/usr/bin python3
""" Command specific tabs of Display Frame of the Faceswap GUI """
import datetime
import logging
import os
import tkinter as tk

from tkinter import ttk


from .display_graph import TrainingGraph
from .display_page import DisplayOptionalPage
from .custom_widgets import Tooltip
from .stats import Calculations, Session
from .control_helper import set_slider_rounding
from .utils import FileHandler, get_config, get_images, preview_trigger

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class PreviewExtract(DisplayOptionalPage):  # pylint: disable=too-many-ancestors
    """ Tab to display output preview images for extract and convert """

    def display_item_set(self):
        """ Load the latest preview if available """
        logger.trace("Loading latest preview")
        size = 256 if self.command == "convert" else 128
        get_images().load_latest_preview(thumbnail_size=int(size * get_config().scaling_factor),
                                         frame_dims=(self.winfo_width(), self.winfo_height()))
        self.display_item = get_images().previewoutput

    def display_item_process(self):
        """ Display the preview """
        logger.trace("Displaying preview")
        if not self.subnotebook.children:
            self.add_child()
        else:
            self.update_child()

    def add_child(self):
        """ Add the preview label child """
        logger.debug("Adding child")
        preview = self.subnotebook_add_page(self.tabname, widget=None)
        lblpreview = ttk.Label(preview, image=get_images().previewoutput[1])
        lblpreview.pack(side=tk.TOP, anchor=tk.NW)
        Tooltip(lblpreview, text=self.helptext, wraplength=200)

    def update_child(self):
        """ Update the preview image on the label """
        logger.trace("Updating preview")
        for widget in self.subnotebook_get_widgets():
            widget.configure(image=get_images().previewoutput[1])

    def save_items(self):
        """ Open save dialogue and save preview """
        location = FileHandler("dir", None).retfile
        if not location:
            return
        filename = "extract_convert_preview"
        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(location,
                                "{}_{}.{}".format(filename,
                                                  now,
                                                  "png"))
        get_images().previewoutput[0].save(filename)
        logger.debug("Saved preview to %s", filename)
        print("Saved preview to {}".format(filename))


class PreviewTrain(DisplayOptionalPage):  # pylint: disable=too-many-ancestors
    """ Training preview image(s) """
    def __init__(self, *args, **kwargs):
        self.update_preview = get_config().tk_vars["updatepreview"]
        super().__init__(*args, **kwargs)

    def add_options(self):
        """ Add the additional options """
        self.add_option_refresh()
        super().add_options()

    def add_option_refresh(self):
        """ Add refresh button to refresh preview immediately """
        logger.debug("Adding refresh option")
        btnrefresh = ttk.Button(self.optsframe,
                                image=get_images().icons["reload"],
                                command=preview_trigger().set)
        btnrefresh.pack(padx=2, side=tk.RIGHT)
        Tooltip(btnrefresh,
                text="Preview updates at every model save. Click to refresh now.",
                wraplength=200)
        logger.debug("Added refresh option")

    def display_item_set(self):
        """ Load the latest preview if available """
        logger.trace("Loading latest preview")
        if not self.update_preview.get():
            logger.trace("Preview not updated")
            return
        get_images().load_training_preview()
        self.display_item = get_images().previewtrain

    def display_item_process(self):
        """ Display the preview(s) resized as appropriate """
        logger.trace("Displaying preview")
        sortednames = sorted(list(get_images().previewtrain.keys()))
        existing = self.subnotebook_get_titles_ids()
        should_update = self.update_preview.get()

        for name in sortednames:
            if name not in existing.keys():
                self.add_child(name)
            elif should_update:
                tab_id = existing[name]
                self.update_child(tab_id, name)

        if should_update:
            self.update_preview.set(False)

    def add_child(self, name):
        """ Add the preview canvas child """
        logger.debug("Adding child")
        preview = PreviewTrainCanvas(self.subnotebook, name)
        preview = self.subnotebook_add_page(name, widget=preview)
        Tooltip(preview, text=self.helptext, wraplength=200)
        self.vars["modified"].set(get_images().previewtrain[name][2])

    def update_child(self, tab_id, name):
        """ Update the preview canvas """
        logger.debug("Updating preview")
        if self.vars["modified"].get() != get_images().previewtrain[name][2]:
            self.vars["modified"].set(get_images().previewtrain[name][2])
            widget = self.subnotebook_page_from_id(tab_id)
            widget.reload()

    def save_items(self):
        """ Open save dialogue and save preview """
        location = FileHandler("dir", None).retfile
        if not location:
            return
        for preview in self.subnotebook.children.values():
            preview.save_preview(location)


class PreviewTrainCanvas(ttk.Frame):  # pylint: disable=too-many-ancestors
    """ Canvas to hold a training preview image """
    def __init__(self, parent, previewname):
        logger.debug("Initializing %s: (previewname: '%s')", self.__class__.__name__, previewname)
        ttk.Frame.__init__(self, parent)

        self.name = previewname
        get_images().resize_image(self.name, None)
        self.previewimage = get_images().previewtrain[self.name][1]

        self.canvas = tk.Canvas(self, bd=0, highlightthickness=0)
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.imgcanvas = self.canvas.create_image(0,
                                                  0,
                                                  image=self.previewimage,
                                                  anchor=tk.NW)
        self.bind("<Configure>", self.resize)
        logger.debug("Initialized %s:", self.__class__.__name__)

    def resize(self, event):
        """  Resize the image to fit the frame, maintaining aspect ratio """
        logger.trace("Resizing preview image")
        framesize = (event.width, event.height)
        # Sometimes image is resized before frame is drawn
        framesize = None if framesize == (1, 1) else framesize
        get_images().resize_image(self.name, framesize)
        self.reload()

    def reload(self):
        """ Reload the preview image """
        logger.trace("Reloading preview image")
        self.previewimage = get_images().previewtrain[self.name][1]
        self.canvas.itemconfig(self.imgcanvas, image=self.previewimage)

    def save_preview(self, location):
        """ Save the figure to file """
        filename = self.name
        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(location,
                                "{}_{}.{}".format(filename,
                                                  now,
                                                  "png"))
        get_images().previewtrain[self.name][0].save(filename)
        logger.debug("Saved preview to %s", filename)
        print("Saved preview to {}".format(filename))


class GraphDisplay(DisplayOptionalPage):  # pylint: disable=too-many-ancestors
    """ The Graph Tab of the Display section """
    def __init__(self, parent, tab_name, helptext, waittime, command=None):
        self._trace_vars = dict()
        super().__init__(parent, tab_name, helptext, waittime, command)

    def set_vars(self):
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

    def on_tab_select(self):
        """ Callback for when the graph tab is selected.

        Pull latest data and run the tab's update code when the tab is selected.
        """
        logger.debug("Callback received for '%s' tab", self.tabname)
        if self.display_item is not None:
            get_config().tk_vars["refreshgraph"].set(True)
        self._update_page()

    def add_options(self):
        """ Add the additional options """
        self._add_option_refresh()
        super().add_options()
        self._add_option_raw()
        self._add_option_smoothed()
        self._add_option_smoothing()
        self._add_option_iterations()

    def _add_option_refresh(self):
        """ Add refresh button to refresh graph immediately """
        logger.debug("Adding refresh option")
        tk_var = get_config().tk_vars["refreshgraph"]
        btnrefresh = ttk.Button(self.optsframe,
                                image=get_images().icons["reload"],
                                command=lambda: tk_var.set(True))
        btnrefresh.pack(padx=2, side=tk.RIGHT)
        Tooltip(btnrefresh,
                text="Graph updates at every model save. Click to refresh now.",
                wraplength=200)
        logger.debug("Added refresh option")

    def _add_option_raw(self):
        """ Add check-button to hide/display raw data """
        logger.debug("Adding display raw option")
        tk_var = self.vars["raw_data"]
        chkbtn = ttk.Checkbutton(
            self.optsframe,
            variable=tk_var,
            text="Raw",
            command=lambda v=tk_var: self._display_data_callback("raw", v))
        chkbtn.pack(side=tk.RIGHT, padx=5, anchor=tk.W)
        Tooltip(chkbtn, text="Display the raw loss data", wraplength=200)

    def _add_option_smoothed(self):
        """ Add check-button to hide/display smoothed data """
        logger.debug("Adding display smoothed option")
        tk_var = self.vars["smooth_data"]
        chkbtn = ttk.Checkbutton(
            self.optsframe,
            variable=tk_var,
            text="Smoothed",
            command=lambda v=tk_var: self._display_data_callback("smoothed", v))
        chkbtn.pack(side=tk.RIGHT, padx=5, anchor=tk.W)
        Tooltip(chkbtn, text="Display the smoothed loss data", wraplength=200)

    def _add_option_smoothing(self):
        """ Add a slider to adjust the smoothing amount """
        logger.debug("Adding Smoothing Slider")
        tk_var = self.vars["smoothgraph"]
        min_max = (0, 0.999)
        hlp = "Set the smoothing amount. 0 is no smoothing, 0.99 is maximum smoothing."

        ctl_frame = ttk.Frame(self.optsframe)
        ctl_frame.pack(padx=2, side=tk.RIGHT)

        lbl = ttk.Label(ctl_frame, text="Smoothing:", anchor=tk.W)
        lbl.pack(pady=5, side=tk.LEFT, anchor=tk.N, expand=True)

        tbox = ttk.Entry(ctl_frame, width=6, textvariable=tk_var, justify=tk.RIGHT)
        tbox.pack(padx=(0, 5), side=tk.RIGHT)

        ctl = ttk.Scale(
            ctl_frame,
            variable=tk_var,
            command=lambda val, var=tk_var, dt=float, rn=3, mm=min_max:
            set_slider_rounding(val, var, dt, rn, mm))
        ctl["from_"] = min_max[0]
        ctl["to"] = min_max[1]
        ctl.pack(padx=5, pady=5, fill=tk.X, expand=True)
        for item in (tbox, ctl):
            Tooltip(item,
                    text=hlp,
                    wraplength=200)
        logger.debug("Added Smoothing Slider")

    def _add_option_iterations(self):
        """ Add a slider to adjust the amount if iterations to display """
        logger.debug("Adding Iterations Slider")
        tk_var = self.vars["display_iterations"]
        min_max = (0, 100000)
        hlp = "Set the number of iterations to display. 0 displays the full session."

        ctl_frame = ttk.Frame(self.optsframe)
        ctl_frame.pack(padx=2, side=tk.RIGHT)

        lbl = ttk.Label(ctl_frame, text="Iterations:", anchor=tk.W)
        lbl.pack(pady=5, side=tk.LEFT, anchor=tk.N, expand=True)

        tbox = ttk.Entry(ctl_frame, width=6, textvariable=tk_var, justify=tk.RIGHT)
        tbox.pack(padx=(0, 5), side=tk.RIGHT)

        ctl = ttk.Scale(
            ctl_frame,
            variable=tk_var,
            command=lambda val, var=tk_var, dt=int, rn=1000, mm=min_max:
            set_slider_rounding(val, var, dt, rn, mm))
        ctl["from_"] = min_max[0]
        ctl["to"] = min_max[1]
        ctl.pack(padx=5, pady=5, fill=tk.X, expand=True)
        for item in (tbox, ctl):
            Tooltip(item,
                    text=hlp,
                    wraplength=200)
        logger.debug("Added Iterations Slider")

    def display_item_set(self):
        """ Load the graph(s) if available """
        if Session.is_training and Session.logging_disabled:
            logger.trace("Logs disabled. Hiding graph")
            self.set_info("Graph is disabled as 'no-logs' has been selected")
            self.display_item = None
            self._clear_trace_variables()
        elif Session.is_training and self.display_item is None:
            logger.trace("Loading graph")
            self.display_item = Session
            self._add_trace_variables()
        else:
            logger.trace("Clearing graph")
            self.display_item = None
            self._clear_trace_variables()

    def display_item_process(self):
        """ Add a single graph to the graph window """
        if not Session.is_training:
            logger.debug("Waiting for Session Data to become available to graph")
            self.after(1000, self.display_item_process)
            return

        logger.debug("Adding graph")
        existing = list(self.subnotebook_get_titles_ids().keys())
        loss_keys = [key
                     for key in self.display_item.get_loss_keys(Session.session_ids[-1])
                     if key != "total"]
        display_tabs = sorted(set(key[:-1].rstrip("_") for key in loss_keys))

        for loss_key in display_tabs:
            tabname = loss_key.replace("_", " ").title()
            if tabname in existing:
                continue

            display_keys = [key for key in loss_keys if key.startswith(loss_key)]
            data = Calculations(session_id=Session.session_ids[-1],
                                display="loss",
                                loss_keys=display_keys,
                                selections=["raw", "smoothed"],
                                smooth_amount=self.vars["smoothgraph"].get())
            self.add_child(tabname, data)

    def _smooth_amount_callback(self, *args):
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

    def _iteration_limit_callback(self, *args):
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

    def _display_data_callback(self, line, variable):
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

    def add_child(self, name, data):
        """ Add the graph for the selected keys """
        logger.debug("Adding child: %s", name)
        graph = TrainingGraph(self.subnotebook, data, "Loss")
        graph.build()
        graph = self.subnotebook_add_page(name, widget=graph)
        Tooltip(graph, text=self.helptext, wraplength=200)

    def save_items(self):
        """ Open save dialogue and save graphs """
        graphlocation = FileHandler("dir", None).retfile
        if not graphlocation:
            return
        for graph in self.subnotebook.children.values():
            graph.save_fig(graphlocation)

    def _add_trace_variables(self):
        """ Add tracing for when the option sliders are updated, for updating the graph. """
        for name, action in zip(("smoothgraph", "display_iterations"),
                                (self._smooth_amount_callback, self._iteration_limit_callback)):
            var = self.vars[name]
            if name not in self._trace_vars:
                self._trace_vars[name] = (var, var.trace("w", action))

    def _clear_trace_variables(self):
        """ Clear all of the trace variables from :attr:`_trace_vars` and reset the dictionary. """
        if self._trace_vars:
            for name, (var, trace) in self._trace_vars.items():
                logger.debug("Clearing trace from variable: %s", name)
                var.trace_vdelete("w", trace)
            self._trace_vars = dict()

    def close(self):
        """ Clear the plots from RAM """
        self._clear_trace_variables()
        if self.subnotebook is None:
            logger.debug("No graphs to clear. Returning")
            return

        for name, graph in self.subnotebook.children.items():
            logger.debug("Clearing: %s", name)
            graph.clear()
        super().close()
