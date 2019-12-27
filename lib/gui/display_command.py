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
from .stats import Calculations
from .control_helper import set_slider_rounding
from .utils import FileHandler, get_config, get_images

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
    def __init__(self, parent, tabname, helptext, waittime, command=None):
        self.trace_var = None
        super().__init__(parent, tabname, helptext, waittime, command)

    def add_options(self):
        """ Add the additional options """
        self.add_option_refresh()
        super().add_options()
        self.add_option_smoothing()

    def add_option_refresh(self):
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

    def add_option_smoothing(self):
        """ Add refresh button to refresh graph immediately """
        logger.debug("Adding Smoothing Slider")
        tk_var = get_config().tk_vars["smoothgraph"]
        min_max = (0, 0.99)
        hlp = "Set the smoothing amount. 0 is no smoothing, 0.99 is maximum smoothing."

        ctl_frame = ttk.Frame(self.optsframe)
        ctl_frame.pack(padx=2, side=tk.RIGHT)

        lbl = ttk.Label(ctl_frame, text="Smoothing Amount:", anchor=tk.W)
        lbl.pack(pady=5, side=tk.LEFT, anchor=tk.N, expand=True)

        tbox = ttk.Entry(ctl_frame, width=6, textvariable=tk_var, justify=tk.RIGHT)
        tbox.pack(padx=(0, 5), side=tk.RIGHT)

        ctl = ttk.Scale(
            ctl_frame,
            variable=tk_var,
            command=lambda val, var=tk_var, dt=float, rn=2, mm=(0, 0.99):
            set_slider_rounding(val, var, dt, rn, mm))
        ctl["from_"] = min_max[0]
        ctl["to"] = min_max[1]
        ctl.pack(padx=5, pady=5, fill=tk.X, expand=True)
        for item in (tbox, ctl):
            Tooltip(item,
                    text=hlp,
                    wraplength=200)
        logger.debug("Added Smoothing Slider")

    def display_item_set(self):
        """ Load the graph(s) if available """
        session = get_config().session
        smooth_amount_var = get_config().tk_vars["smoothgraph"]
        if session.initialized and session.logging_disabled:
            logger.trace("Logs disabled. Hiding graph")
            self.set_info("Graph is disabled as 'no-logs' or 'pingpong' has been selected")
            self.display_item = None
            if self.trace_var is not None:
                smooth_amount_var.trace_vdelete("w", self.trace_var)
                self.trace_var = None
        elif session.initialized:
            logger.trace("Loading graph")
            self.display_item = session
            if self.trace_var is None:
                self.trace_var = smooth_amount_var.trace("w", self.smooth_amount_callback)
        else:
            self.display_item = None
            if self.trace_var is not None:
                smooth_amount_var.trace_vdelete("w", self.trace_var)
                self.trace_var = None

    def display_item_process(self):
        """ Add a single graph to the graph window """
        logger.trace("Adding graph")
        existing = list(self.subnotebook_get_titles_ids().keys())
        display_tabs = sorted(self.display_item.loss_keys)
        if any(key.startswith("total") for key in display_tabs):
            total_idx = [idx for idx, key in enumerate(display_tabs) if key.startswith("total")][0]
            display_tabs.insert(0, display_tabs.pop(total_idx))
        for loss_key in display_tabs:
            tabname = loss_key.replace("_", " ").title()
            if tabname in existing:
                continue

            data = Calculations(session=get_config().session,
                                display="loss",
                                loss_keys=[loss_key],
                                selections=["raw", "smoothed"],
                                smooth_amount=get_config().tk_vars["smoothgraph"].get())
            self.add_child(tabname, data)

    def smooth_amount_callback(self, *args):
        """ Update each graph's smooth amount on variable change """
        smooth_amount = get_config().tk_vars["smoothgraph"].get()
        logger.debug("Updating graph smooth_amount: (new_value: %s, args: %s)",
                     smooth_amount, args)
        for graph in self.subnotebook.children.values():
            graph.calcs.args["smooth_amount"] = smooth_amount

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

    def close(self):
        """ Clear the plots from RAM """
        if self.trace_var is not None:
            get_config().tk_vars["smoothgraph"].trace_vdelete("w", self.trace_var)
            self.trace_var = None
        if self.subnotebook is None:
            logger.debug("No graphs to clear. Returning")
            return
        for name, graph in self.subnotebook.children.items():
            logger.debug("Clearing: %s", name)
            graph.clear()
        super().close()
