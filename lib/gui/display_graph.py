#!/usr/bin python3
"""Graph functions for Display Frame area of the Faceswap GUI"""
from __future__ import annotations
import datetime
import logging
import os
import tkinter as tk
import typing as T

from tkinter import ttk
from math import ceil, floor

import numpy as np
import matplotlib
from matplotlib import style
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)  # pyright:ignore[reportPrivateImportUsage]
from matplotlib.backend_bases import NavigationToolbar2

from lib.logger import parse_class_init
from lib.utils import get_module_objects

from .custom_widgets import Tooltip
from .utils import get_config, get_images, LongRunningTask

if T.TYPE_CHECKING:
    from matplotlib.lines import Line2D

logger: logging.Logger = logging.getLogger(__name__)


class GraphBase(ttk.Frame):  # pylint:disable=too-many-ancestors
    """Base class for matplotlib line graphs.

    Parameters
    ----------
    parent
        The parent frame that holds the graph
    data
        The statistics class that holds the data to be displayed
    ylabel
        The data label for the y-axis
    """
    def __init__(self, parent, data, ylabel: str) -> None:
        super().__init__(parent)
        matplotlib.use("TkAgg")  # Can't be at module level as breaks Github CI
        style.use("ggplot")

        self._calcs = data
        self._ylabel = ylabel
        self._color_maps = ["Reds", "Blues", "Greens", "Purples", "Oranges", "Greys", "copper",
                            "summer", "bone", "hot", "cool", "pink", "Wistia", "spring", "winter"]
        self._lines: list[Line2D] = []
        self._toolbar: NavigationToolbar | None = None
        self._fig = Figure(figsize=(4, 4), dpi=75)

        self._ax1 = self._fig.add_subplot(1, 1, 1)
        self._plot_canvas = FigureCanvasTkAgg(self._fig, self)

        self._initiate_graph()
        self._update_plot(initiate=True)

    @property
    def calcs(self):
        """The calculated statistics associated with this graph."""
        return self._calcs

    def _initiate_graph(self) -> None:
        """Place the graph canvas"""
        logger.debug("[GraphBase] Setting plot canvas")
        self._plot_canvas.get_tk_widget().pack(side=tk.TOP, padx=5, fill=tk.BOTH, expand=True)
        self._fig.subplots_adjust(left=0.100,
                                  bottom=0.100,
                                  right=0.95,
                                  top=0.95,
                                  wspace=0.2,
                                  hspace=0.2)
        logger.debug("[GraphBase] Set plot canvas")

    def _update_plot(self, initiate: bool = True) -> None:
        """Update the plot with incoming data

        Parameters
        ----------
        initiate
            Whether the graph should be initialized for the first time (``True``) or data is being
            updated for an existing graph (``False``). Default: ``True``
        """
        logger.trace("[GraphBase] Updating plot")  # type:ignore[attr-defined]
        if initiate:
            logger.debug("[GraphBase] Initializing plot")
            self._lines = []
            self._ax1.clear()
            self._axes_labels_set()
            logger.debug("[GraphBase] Initialized plot")

        full_data = list(self._calcs.stats.values())
        self._axes_limits_set(full_data)

        if self._calcs.start_iteration > 0:
            end_iteration = self._calcs.start_iteration + self._calcs.iterations
            x_rng = list(range(self._calcs.start_iteration, end_iteration))
        else:
            x_rng = list(range(self._calcs.iterations))

        keys = list(self._calcs.stats.keys())

        for idx, item in enumerate(self._lines_sort(keys)):
            if initiate:
                self._lines.extend(self._ax1.plot(x_rng, self._calcs.stats[item[0]],
                                                  label=item[1], linewidth=item[2], color=item[3]))
            else:
                self._lines[idx].set_data(x_rng, self._calcs.stats[item[0]])

        if initiate:
            self._legend_place()
        logger.trace("[GraphBase] Updated plot")  # type:ignore[attr-defined]

    def _axes_labels_set(self) -> None:
        """Set the X and Y axes labels."""
        logger.debug("[GraphBase] Setting axes labels. y-label: '%s'", self._ylabel)
        self._ax1.set_xlabel("Iterations")
        self._ax1.set_ylabel(self._ylabel)

    def _axes_limits_set_default(self) -> None:
        """Set the default axes limits for the X and Y axes."""
        logger.debug("[GraphBase] Setting default axes ranges")
        self._ax1.set_ylim(0.00, 100.0)
        self._ax1.set_xlim(0, 1)

    def _axes_limits_set(self, data: list[float]) -> None:
        """Set the axes limits.

        Parameters
        ----------
        data
            The data points for the Y Axis
        """
        xmin = self._calcs.start_iteration
        if self._calcs.start_iteration > 0:
            xmax = self._calcs.iterations + self._calcs.start_iteration
        else:
            xmax = self._calcs.iterations
        xmax = max(1, xmax - 1)

        if data:
            ymin, ymax = self._axes_data_get_min_max(data)
            self._ax1.set_ylim(ymin, ymax)
            self._ax1.set_xlim(xmin, xmax)
            logger.trace(  # type:ignore[attr-defined]
                "[GraphBase] axes ranges: (y: (%s, %s), x:(0, %s)", ymin, ymax, xmax)
        else:
            self._axes_limits_set_default()

    @staticmethod
    def _axes_data_get_min_max(data: list[float]) -> tuple[float, float]:
        """Obtain the minimum and maximum values for the y-axis from the given data points.

        Parameters
        ----------
        data
            The data points for the Y Axis

        Returns
        -------
        The minimum and maximum values for the y axis
        """
        y_mins, y_maxes = [], []

        for item in data:  # TODO Handle as array not loop
            y_mins.append(np.nanmin(item) * 1000)
            y_maxes.append(np.nanmax(item) * 1000)
        ymin = floor(min(y_mins)) / 1000
        ymax = ceil(max(y_maxes)) / 1000
        logger.trace("[GraphBase] ymin: %s, ymax: %s", ymin, ymax)  # type:ignore[attr-defined]
        return ymin, ymax

    def _axes_set_y_scale(self, scale: str) -> None:
        """Set the Y-Scale to log or linear

        Parameters
        ----------
        scale
            Should be one of ``"log"`` or ``"linear"``
        """
        logger.debug("[GraphBase] y_scale: '%s'", scale)
        self._ax1.set_yscale(scale)

    def _lines_sort(self,
                    keys: list[str]) -> list[list[str | int | tuple[float, float, float, float]]]:
        """Sort the data keys into consistent order and set line color map and line width.

        Parameters
        ----------
        keys
            The list of data point keys

        Returns
        -------
        The sorted data keys
        """
        logger.trace("[GraphBase] Sorting lines")  # type:ignore[attr-defined]
        raw_lines: list[list[str]] = []
        sorted_lines: list[list[str]] = []
        for key in sorted(keys):
            title = key.replace("_", " ")
            if key.startswith("raw"):
                raw_lines.append([key, title])
            else:
                sorted_lines.append([key, title])

        group_size = self._lines_group_size(raw_lines, sorted_lines)
        sorted_lines = raw_lines + sorted_lines
        lines = self._lines_style(sorted_lines, group_size)
        return lines

    @staticmethod
    def _lines_group_size(raw_lines: list[list[str]], sorted_lines: list[list[str]]) -> int:
        """Get the number of items in each group.

        If raw data isn't selected, then check the length of remaining groups until something is
        found.

        Parameters
        ----------
        raw_lines
            The list of keys for the raw data points
        sorted_lines
            The list of sorted line keys to display on the graph

        Returns
        -------
        The size of each group that exist within the data set.
        """
        group_size = 1
        if raw_lines:
            group_size = len(raw_lines)
        elif sorted_lines:
            keys = [key[0][:key[0].find("_")] for key in sorted_lines]
            distinct_keys = set(keys)
            group_size = len(keys) // len(distinct_keys)
        logger.trace("[GraphBase] %s", group_size)  # type:ignore[attr-defined]
        return group_size

    def _lines_create_colors(self,
                             group_size: int,
                             groups: int) -> list[tuple[float, float, float, float]]:
        """Create the color maps.

        Parameters
        ----------
        group_size
            The size of each group to display in the graph
        groups
            The total number of groups to graph

        Returns
        -------
        The colour map for each group
        """
        colors = []
        for i in range(1, groups + 1):
            for colour in self._color_maps[0:group_size]:
                c_map = matplotlib.cm.get_cmap(  # pyright:ignore[reportAttributeAccessIssue]
                    colour
                    )
                c_point = 1 - (i / 5)
                colors.append(c_map(c_point))
        logger.trace("[GraphBase] %s",  colors)  # type:ignore[attr-defined]
        return colors

    def _lines_style(self,
                     lines: list[list[str]],
                     group_size: int) -> list[list[str | int | tuple[float, float, float, float]]]:
        """Obtain the color map and line width for each group.

        Parameters
        ----------
        lines
            The list of sorted line keys to display on the graph
        group_size
            The size of each group to display in the graph

        Returns
        -------
        A list of loss keys with their corresponding line formatting and color information
        """
        logger.trace("[GraphBase] Setting lines style")  # type:ignore[attr-defined]
        groups = int(len(lines) / group_size)
        colors = self._lines_create_colors(group_size, groups)
        widths = list(range(1, groups + 1))
        retval = T.cast(list[list[str | int | tuple[float, float, float, float]]], lines)
        for idx, item in enumerate(retval):
            linewidth = widths[idx // group_size]
            item.extend((linewidth, colors[idx]))
        return retval

    def _legend_place(self) -> None:
        """Place and format the graph legend"""
        logger.debug("[GraphBase] Placing legend")
        self._ax1.legend(loc="upper right", ncol=2)

    def _toolbar_place(self, parent) -> None:
        """Add Graph Navigation toolbar.

        Parameters
        ----------
        parent
            The parent graph frame to place the toolbar onto
        """
        logger.debug("[GraphBase] Placing toolbar")
        self._toolbar = NavigationToolbar(self._plot_canvas, parent)
        self._toolbar.pack(side=tk.BOTTOM)
        self._toolbar.update()

    def clear(self) -> None:
        """Clear the graph plots from RAM """
        logger.debug("[GraphBase] Clearing graph from RAM: %s", self)
        self._fig.clf()
        del self._fig


class TrainingGraph(GraphBase):  # pylint:disable=too-many-ancestors
    """Live graph to be displayed during training.

    Parameters
    ----------
    parent
        The parent frame that holds the graph
    data
        The statistics class that holds the data to be displayed
    ylabel
        The data label for the y-axis
    """
    def __init__(self, parent, data, ylabel: str) -> None:
        logger.debug(parse_class_init(locals()))
        super().__init__(parent, data, ylabel)
        self._thread: LongRunningTask | None = None  # Thread for LongRunningTask
        self._displayed_keys: list[str] = []
        self._add_callback()

    def _add_callback(self) -> None:
        """Add the variable trace to update graph on refresh button press or save iteration."""
        get_config().tk_vars.refresh_graph.trace("w", self.refresh)  # type:ignore

    def build(self) -> None:
        """Build the Training graph."""
        logger.debug("[TrainingGraph] Building training graph")
        self._plot_canvas.draw()
        logger.debug("[TrainingGraph] Built training graph")

    def refresh(self, *args) -> None:  # pylint:disable=unused-argument
        """Read the latest loss data and apply to current graph"""
        refresh_var = T.cast(tk.BooleanVar, get_config().tk_vars.refresh_graph)
        if not refresh_var.get() and self._thread is None:
            return

        if self._thread is None:
            logger.debug("[TrainingGraph] Updating plot data")
            self._thread = LongRunningTask(target=self._calcs.refresh)
            self._thread.start()
            self.after(1000, self.refresh)
        elif not self._thread.complete.is_set():
            logger.debug("[TrainingGraph] Graph Data not yet available")
            self.after(1000, self.refresh)
        else:
            logger.debug("[TrainingGraph] Updating plot with data from background thread")
            self._calcs = self._thread.get_result()  # Terminate the LongRunningTask object
            self._thread = None

            dsp_keys = list(sorted(self._calcs.stats))
            if dsp_keys != self._displayed_keys:
                logger.debug("[TrainingGraph] Reinitializing graph for keys change. "
                             "Old keys: %s New keys: %s",
                             self._displayed_keys, dsp_keys)
                initiate = True
                self._displayed_keys = dsp_keys
            else:
                initiate = False

            self._update_plot(initiate=initiate)
            self._plot_canvas.draw()
            refresh_var.set(False)

    def save_fig(self, location: str) -> None:
        """Save the current graph to file

        Parameters
        ----------
        location
            The full path to the folder where the current graph should be saved
        """
        logger.debug("[TrainingGraph] Saving graph: '%s'", location)
        keys = sorted([key.replace("raw_", "") for key in self._calcs.stats.keys()
                       if key.startswith("raw_")])
        filename = " - ".join(keys)
        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(location, f"{filename}_{now}.png")
        self._fig.set_size_inches(16, 9)
        self._fig.savefig(filename, bbox_inches="tight", dpi=120)
        print(f"Saved graph to {filename}")
        logger.debug("[TrainingGraph] Saved graph: '%s'", filename)
        self._resize_fig()

    def _resize_fig(self) -> None:
        """Resize the figure to the current canvas size."""
        class Event():  # pylint:disable=too-few-public-methods
            """Event class that needs to be passed to plot_canvas.resize"""
            pass  # pylint:disable=unnecessary-pass
        setattr(Event, "width", self.winfo_width())
        setattr(Event, "height", self.winfo_height())
        self._plot_canvas.resize(Event)  # pylint:disable=no-value-for-parameter


class SessionGraph(GraphBase):  # pylint:disable=too-many-ancestors
    """Session Graph for session pop-up.

    Parameters
    ----------
    parent
        The parent frame that holds the graph
    data
        The statistics class that holds the data to be displayed
    ylabel
        The data label for the y-axis
    scale
        Should be one of ``"log"`` or ``"linear"``
    """
    def __init__(self, parent, data, ylabel: str, scale: str) -> None:
        logger.debug(parse_class_init(locals()))
        super().__init__(parent, data, ylabel)
        self._scale = scale

    def build(self) -> None:
        """Build the session graph"""
        logger.debug("[SessionGraph] Building session graph")
        self._toolbar_place(self)
        self._plot_canvas.draw()
        logger.debug("[SessionGraph] Built session graph")

    def refresh(self, data, ylabel: str, scale: str) -> None:
        """Refresh the Session Graph's data.

        Parameters
        ----------
        data
            The statistics class that holds the data to be displayed
        ylabel
            The data label for the y-axis
        scale
            Should be one of ``"log"`` or ``"linear"``
        """
        logger.debug("[SessionGraph] Refreshing session graph: (ylabel: '%s', scale: '%s')",
                     ylabel, scale)
        self._calcs = data
        self._ylabel = ylabel
        self.set_yscale_type(scale)
        logger.debug("[SessionGraph] Refreshed session graph")

    def set_yscale_type(self, scale: str) -> None:
        """Set the scale type for the y-axis and redraw.

        Parameters
        ----------
        scale
            Should be one of ``"log"`` or ``"linear"``
        """
        scale = scale.lower()
        logger.debug("[SessionGraph] Updating scale type: '%s'", scale)
        self._scale = scale
        self._update_plot(initiate=True)
        self._axes_set_y_scale(self._scale)
        self._plot_canvas.draw()
        logger.debug("[SessionGraph] Updated scale type")


class NavigationToolbar(NavigationToolbar2Tk):  # pylint:disable=too-many-ancestors
    """Overrides the default Navigation Toolbar to provide only the buttons we require
    and to layout the items in a consistent manner with the rest of the GUI for the Analysis
    Session Graph pop up Window.

    Parameters
    ----------
    canvas
        The canvas that holds the displayed graph and will hold the toolbar
    window
        The Session Graph canvas
    pack_toolbar
        Whether to pack the Tool bar or not. Default: ``True``
    """
    toolitems = tuple(t for t in NavigationToolbar2Tk.toolitems if
                      t[0] in ("Home", "Pan", "Zoom", "Save"))

    def __init__(self,  # pylint:disable=super-init-not-called
                 canvas: FigureCanvasTkAgg,
                 window,
                 *,
                 pack_toolbar: bool = True) -> None:
        logger.debug(parse_class_init(locals()))
        # Avoid using self.window (prefer self.canvas.get_tk_widget().master),
        # so that Tool implementations can reuse the methods.

        ttk.Frame.__init__(T.cast(ttk.Frame, self),  # pylint:disable=non-parent-init-called
                           master=window,
                           width=int(canvas.figure.bbox.width),
                           height=50)

        sep = ttk.Frame(self, height=2, relief=tk.RIDGE)
        sep.pack(fill=tk.X, pady=(5, 0), side=tk.TOP)

        btn_frame = ttk.Frame(self)  # Add a button frame to consistently line up GUI
        btn_frame.pack(fill=tk.X, padx=5, pady=5, side=tk.RIGHT)

        self._buttons = {}
        for text, tooltip_text, image_file, callback in self.toolitems:
            assert isinstance(text, str)
            assert isinstance(image_file, str)
            assert isinstance(callback, str)
            self._buttons[text] = button = self._Button(
                btn_frame,
                text,
                image_file,
                toggle=callback in ["zoom", "pan"],
                command=getattr(self, callback),
            )
            if tooltip_text is not None:
                Tooltip(button, text=tooltip_text, wrap_length=200)

        self.message = tk.StringVar(master=self)
        self._message_label = ttk.Label(master=self, textvariable=self.message)
        self._message_label.pack(side=tk.LEFT, padx=5)  # Additional left padding

        NavigationToolbar2.__init__(self, canvas)  # pylint:disable=non-parent-init-called
        if pack_toolbar:
            self.pack(side=tk.BOTTOM, fill=tk.X)

    @staticmethod
    def _Button(frame,  # type:ignore[override] # pylint:disable=arguments-differ,arguments-renamed  # noqa:E501
                text: str,
                image_file: str,
                toggle: bool,
                command) -> ttk.Button | ttk.Checkbutton:
        """Override the default button method to use our icons and ttk widgets for
        consistent GUI layout.

        Parameters
        ----------
        frame
            The frame that holds the buttons
        text
            The display text for the button
        image_file
            The name of the image file to use
        toggle
            Whether to use a checkbutton (``True``) or a regular button (``False``)
        command
            The Navigation Toolbar callback method

        Returns
        -------
        The widget to use. A button if the option can not be toggled, a checkbutton if the option
        can be toggled.
        """
        icon_mapping = {"home": "reload",
                        "filesave": "save",
                        "zoom_to_rect": "zoom"}
        icon = icon_mapping[image_file] if icon_mapping.get(image_file, None) else image_file
        img = get_images().icons[icon]

        if not toggle:
            btn: ttk.Button | ttk.Checkbutton = ttk.Button(frame,
                                                           text=text,
                                                           image=img,  # type:ignore[arg-type]
                                                           command=command)
        else:
            var = tk.IntVar(master=frame)
            btn = ttk.Checkbutton(frame,
                                  text=text,
                                  image=img,  # type:ignore[arg-type]
                                  command=command, variable=var)

            # Original implementation uses tk Checkbuttons which have a select and deselect
            # method. These aren't available in ttk Checkbuttons, so we monkey patch the methods
            # to update the underlying variable.
            setattr(btn, "select", lambda i=1: var.set(i))
            setattr(btn, "deselect", lambda i=0: var.set(i))

        btn.pack(side=tk.RIGHT, padx=2)
        return btn


__all__ = get_module_objects(__name__)
