#!/usr/bin python3
""" Graph functions for Display Frame of the Faceswap GUI """
import datetime
import logging
import os
import tkinter as tk

from tkinter import ttk
from math import ceil, floor

import matplotlib
# pylint: disable=wrong-import-position
matplotlib.use("TkAgg")

from matplotlib import style  # noqa
from matplotlib.figure import Figure  # noqa
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                               NavigationToolbar2Tk)  # noqa

from .custom_widgets import Tooltip  # noqa
from .utils import get_config, get_images, LongRunningTask  # noqa

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class NavigationToolbar(NavigationToolbar2Tk):  # pylint: disable=too-many-ancestors
    """ Same as default, but only including buttons we need
        with custom icons and layout
        From: https://stackoverflow.com/questions/12695678 """
    toolitems = [t for t in NavigationToolbar2Tk.toolitems if
                 t[0] in ("Home", "Pan", "Zoom", "Save")]

    @staticmethod
    def _Button(frame, text, file, command, extension=".gif"):  # pylint: disable=arguments-differ
        """ Map Buttons to their own frame.
            Use custom button icons, Use ttk buttons pack to the right """
        iconmapping = {"home": "reload",
                       "filesave": "save",
                       "zoom_to_rect": "zoom"}
        icon = iconmapping[file] if iconmapping.get(file, None) else file
        img = get_images().icons[icon]
        btn = ttk.Button(frame, text=text, image=img, command=command)
        btn.pack(side=tk.RIGHT, padx=2)
        return btn

    def _init_toolbar(self):
        """ Same as original but ttk widgets and standard tool-tips used. Separator added and
            message label packed to the left """
        xmin, xmax = self.canvas.figure.bbox.intervalx
        height, width = 50, xmax-xmin
        ttk.Frame.__init__(self, master=self.window, width=int(width), height=int(height))

        sep = ttk.Frame(self, height=2, relief=tk.RIDGE)
        sep.pack(fill=tk.X, pady=(5, 0), side=tk.TOP)

        self.update()  # Make axes menu

        btnframe = ttk.Frame(self)
        btnframe.pack(fill=tk.X, padx=5, pady=5, side=tk.RIGHT)

        for text, tooltip_text, image_file, callback in self.toolitems:
            if text is None:
                # Add a spacer; return value is unused.
                self._Spacer()
            else:
                button = self._Button(btnframe, text=text, file=image_file,
                                      command=getattr(self, callback))
                if tooltip_text is not None:
                    Tooltip(button, text=tooltip_text, wraplength=200)

        self.message = tk.StringVar(master=self)
        self._message_label = ttk.Label(master=self, textvariable=self.message)
        self._message_label.pack(side=tk.LEFT, padx=5)
        self.pack(side=tk.BOTTOM, fill=tk.X)


class GraphBase(ttk.Frame):  # pylint: disable=too-many-ancestors
    """ Base class for matplotlib line graphs """
    def __init__(self, parent, data, ylabel):
        logger.debug("Initializing %s", self.__class__.__name__)
        super().__init__(parent)
        style.use("ggplot")

        self.calcs = data
        self.ylabel = ylabel
        self.colourmaps = ["Reds", "Blues", "Greens", "Purples", "Oranges", "Greys", "copper",
                           "summer", "bone", "hot", "cool", "pink", "Wistia", "spring", "winter"]
        self.lines = list()
        self.toolbar = None
        self.fig = Figure(figsize=(4, 4), dpi=75)

        self.ax1 = self.fig.add_subplot(1, 1, 1)
        self.plotcanvas = FigureCanvasTkAgg(self.fig, self)

        self.initiate_graph()
        self.update_plot(initiate=True)
        logger.debug("Initialized %s", self.__class__.__name__)

    def initiate_graph(self):
        """ Place the graph canvas """
        logger.debug("Setting plotcanvas")
        self.plotcanvas.get_tk_widget().pack(side=tk.TOP, padx=5, fill=tk.BOTH, expand=True)
        self.fig.subplots_adjust(left=0.100,
                                 bottom=0.100,
                                 right=0.95,
                                 top=0.95,
                                 wspace=0.2,
                                 hspace=0.2)
        logger.debug("Set plotcanvas")

    def update_plot(self, initiate=True):
        """ Update the plot with incoming data """
        logger.trace("Updating plot")
        if initiate:
            logger.debug("Initializing plot")
            self.lines = list()
            self.ax1.clear()
            self.axes_labels_set()
            logger.debug("Initialized plot")

        fulldata = [item for item in self.calcs.stats.values()]
        self.axes_limits_set(fulldata)

        xrng = [x for x in range(self.calcs.iterations)]
        keys = list(self.calcs.stats.keys())
        for idx, item in enumerate(self.lines_sort(keys)):
            if initiate:
                self.lines.extend(self.ax1.plot(xrng, self.calcs.stats[item[0]],
                                                label=item[1], linewidth=item[2], color=item[3]))
            else:
                self.lines[idx].set_data(xrng, self.calcs.stats[item[0]])

        if initiate:
            self.legend_place()
        logger.trace("Updated plot")

    def axes_labels_set(self):
        """ Set the axes label and range """
        logger.debug("Setting axes labels. y-label: '%s'", self.ylabel)
        self.ax1.set_xlabel("Iterations")
        self.ax1.set_ylabel(self.ylabel)

    def axes_limits_set_default(self):
        """ Set default axes limits """
        logger.debug("Setting default axes ranges")
        self.ax1.set_ylim(0.00, 100.0)
        self.ax1.set_xlim(0, 1)

    def axes_limits_set(self, data):
        """ Set the axes limits """
        xmax = self.calcs.iterations - 1 if self.calcs.iterations > 1 else 1
        if data:
            ymin, ymax = self.axes_data_get_min_max(data)
            self.ax1.set_ylim(ymin, ymax)
            self.ax1.set_xlim(0, xmax)
            logger.trace("axes ranges: (y: (%s, %s), x:(0, %s)", ymin, ymax, xmax)
        else:
            self.axes_limits_set_default()

    @staticmethod
    def axes_data_get_min_max(data):
        """ Return the minimum and maximum values from list of lists """
        ymin, ymax = list(), list()
        for item in data:
            dataset = list(filter(lambda x: x is not None, item))
            if not dataset:
                continue
            ymin.append(min(dataset) * 1000)
            ymax.append(max(dataset) * 1000)
        ymin = floor(min(ymin)) / 1000
        ymax = ceil(max(ymax)) / 1000
        logger.trace("ymin: %s, ymax: %s", ymin, ymax)
        return ymin, ymax

    def axes_set_yscale(self, scale):
        """ Set the Y-Scale to log or linear """
        logger.debug("yscale: '%s'", scale)
        self.ax1.set_yscale(scale)

    def lines_sort(self, keys):
        """ Sort the data keys into consistent order
            and set line color map and line width """
        logger.trace("Sorting lines")
        raw_lines = list()
        sorted_lines = list()
        for key in sorted(keys):
            title = key.replace("_", " ").title()
            if key.startswith("raw"):
                raw_lines.append([key, title])
            else:
                sorted_lines.append([key, title])

        groupsize = self.lines_groupsize(raw_lines, sorted_lines)
        sorted_lines = raw_lines + sorted_lines
        lines = self.lines_style(sorted_lines, groupsize)
        return lines

    @staticmethod
    def lines_groupsize(raw_lines, sorted_lines):
        """ Get the number of items in each group.
            If raw data isn't selected, then check the length of
            remaining groups until something is found """
        groupsize = 1
        if raw_lines:
            groupsize = len(raw_lines)
        elif sorted_lines:
            keys = [key[0][:key[0].find("_")] for key in sorted_lines]
            distinct_keys = set(keys)
            groupsize = len(keys) // len(distinct_keys)
        logger.trace(groupsize)
        return groupsize

    def lines_style(self, lines, groupsize):
        """ Set the color map and line width for each group """
        logger.trace("Setting lines style")
        groups = int(len(lines) / groupsize)
        colours = self.lines_create_colors(groupsize, groups)
        for idx, item in enumerate(lines):
            linewidth = ceil((idx + 1) / groupsize)
            item.extend((linewidth, colours[idx]))
        return lines

    def lines_create_colors(self, groupsize, groups):
        """ Create the colors """
        colours = list()
        for i in range(1, groups + 1):
            for colour in self.colourmaps[0:groupsize]:
                cmap = matplotlib.cm.get_cmap(colour)
                cpoint = 1 - (i / 5)
                colours.append(cmap(cpoint))
        logger.trace(colours)
        return colours

    def legend_place(self):
        """ Place and format legend """
        logger.debug("Placing legend")
        self.ax1.legend(loc="upper right", ncol=2)

    def toolbar_place(self, parent):
        """ Add Graph Navigation toolbar """
        logger.debug("Placing toolbar")
        self.toolbar = NavigationToolbar(self.plotcanvas, parent)
        self.toolbar.pack(side=tk.BOTTOM)
        self.toolbar.update()

    def clear(self):
        """ Clear the plots from RAM """
        logger.debug("Clearing graph from RAM: %s", self)
        self.fig.clf()
        del self.fig


class TrainingGraph(GraphBase):  # pylint: disable=too-many-ancestors
    """ Live graph to be displayed during training. """

    def __init__(self, parent, data, ylabel):
        GraphBase.__init__(self, parent, data, ylabel)
        self.thread = None  # Thread for LongRunningTask
        self.add_callback()

    def add_callback(self):
        """ Add the variable trace to update graph on recent button or save iteration """
        get_config().tk_vars["refreshgraph"].trace("w", self.refresh)

    def build(self):
        """ Update the plot area with loss values """
        logger.debug("Building training graph")
        self.plotcanvas.draw()
        logger.debug("Built training graph")

    def refresh(self, *args):  # pylint: disable=unused-argument
        """ Read loss data and apply to graph """
        refresh_var = get_config().tk_vars["refreshgraph"]
        if not refresh_var.get() and self.thread is None:
            return

        if self.thread is None:
            logger.debug("Updating plot data")
            self.thread = LongRunningTask(target=self.calcs.refresh)
            self.thread.start()
            self.after(1000, self.refresh)
        elif not self.thread.complete.is_set():
            logger.debug("Graph Data not yet available")
            self.after(1000, self.refresh)
        else:
            logger.debug("Updating plot with data from background thread")
            self.calcs = self.thread.get_result()  # Terminate the LongRunningTask object
            self.thread = None
            self.update_plot(initiate=False)
            self.plotcanvas.draw()
            refresh_var.set(False)

    def save_fig(self, location):
        """ Save the figure to file """
        logger.debug("Saving graph: '%s'", location)
        keys = sorted([key.replace("raw_", "") for key in self.calcs.stats.keys()
                       if key.startswith("raw_")])
        filename = " - ".join(keys)
        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(location, "{}_{}.{}".format(filename, now, "png"))
        self.fig.set_size_inches(16, 9)
        self.fig.savefig(filename, bbox_inches="tight", dpi=120)
        print("Saved graph to {}".format(filename))
        logger.debug("Saved graph: '%s'", filename)
        self.resize_fig()

    def resize_fig(self):
        """ Resize the figure back to the canvas """
        class Event():  # pylint: disable=too-few-public-methods
            """ Event class that needs to be passed to plotcanvas.resize """
            pass  # pylint: disable=unnecessary-pass
        Event.width = self.winfo_width()
        Event.height = self.winfo_height()
        self.plotcanvas.resize(Event)  # pylint: disable=no-value-for-parameter


class SessionGraph(GraphBase):  # pylint: disable=too-many-ancestors
    """ Session Graph for session pop-up """
    def __init__(self, parent, data, ylabel, scale):
        GraphBase.__init__(self, parent, data, ylabel)
        self.scale = scale

    def build(self):
        """ Build the session graph """
        logger.debug("Building session graph")
        self.toolbar_place(self)
        self.plotcanvas.draw()
        logger.debug("Built session graph")

    def refresh(self, data, ylabel, scale):
        """ Refresh graph data """
        logger.debug("Refreshing session graph: (ylabel: '%s', scale: '%s')", ylabel, scale)
        self.calcs = data
        self.ylabel = ylabel
        self.set_yscale_type(scale)
        logger.debug("Refreshed session graph")

    def set_yscale_type(self, scale):
        """ switch the y-scale and redraw """
        logger.debug("Updating scale type: '%s'", scale)
        self.scale = scale
        self.update_plot(initiate=True)
        self.axes_set_yscale(self.scale)
        self.plotcanvas.draw()
        logger.debug("Updated scale type")
