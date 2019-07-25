#!/usr/bin python3
""" Command specific tabs of Display Frame of the Faceswap GUI """
import datetime
import logging
import os
import tkinter as tk

from tkinter import ttk


from .display_graph import TrainingGraph
from .display_page import DisplayOptionalPage
from .tooltip import Tooltip
from .stats import Calculations
from .utils import FileHandler, get_config, get_images

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class PreviewExtract(DisplayOptionalPage):  # pylint: disable=too-many-ancestors
    """ Tab to display output preview images for extract and convert """

    def display_item_set(self):
        """ Load the latest preview if available """
        logger.trace("Loading latest preview")
        get_images().load_latest_preview()
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

    def add_options(self):
        """ Add the additional options """
        self.add_option_refresh()
        super().add_options()

    def add_option_refresh(self):
        """ Add refresh button to refresh graph immediately """
        logger.debug("Adding refresh option")
        tk_var = get_config().tk_vars["refreshgraph"]
        btnrefresh = ttk.Button(self.optsframe,
                                image=get_images().icons["reset"],
                                command=lambda: tk_var.set(True))
        btnrefresh.pack(padx=2, side=tk.RIGHT)
        Tooltip(btnrefresh,
                text="Graph updates every 100 iterations. Click to refresh now.",
                wraplength=200)

    def display_item_set(self):
        """ Load the graph(s) if available """
        session = get_config().session
        if session.initialized and session.logging_disabled:
            logger.trace("Logs disabled. Hiding graph")
            self.set_info("Graph is disabled as 'no-logs' or 'pingpong' has been selected")
            self.display_item = None
        elif session.initialized:
            logger.trace("Loading graph")
            self.display_item = session
        else:
            self.display_item = None

    def display_item_process(self):
        """ Add a single graph to the graph window """
        logger.trace("Adding graph")
        existing = list(self.subnotebook_get_titles_ids().keys())

        for loss_key in self.display_item.loss_keys:
            tabname = loss_key.replace("_", " ").title()
            if tabname in existing:
                continue

            data = Calculations(session=get_config().session,
                                display="loss",
                                loss_keys=[loss_key],
                                selections=["raw", "trend"])
            self.add_child(tabname, data)

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
        for name, graph in self.subnotebook.children.items():
            logger.debug("Clearing: %s", name)
            graph.clear()
        super().close()
