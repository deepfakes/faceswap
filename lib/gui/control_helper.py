#!/usr/bin/env python3
""" Helper functions and classes for GUI controls """
import logging
import tkinter as tk
from tkinter import ttk

from .tooltip import Tooltip
from .utils import ContextMenu

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def set_slider_rounding(value, var, d_type, round_to, min_max):
    """ Set the underlying variable to correct number based on slider rounding """
    if d_type == float:
        var.set(round(float(value), round_to))
    else:
        steps = range(min_max[0], min_max[1] + round_to, round_to)
        value = min(steps, key=lambda x: abs(x - int(float(value))))
        var.set(value)


def adjust_wraplength(event):
    """ dynamically adjust the wraplength of a label on event """
    label = event.widget
    label.configure(wraplength=event.width - 1)


class ControlPanel(ttk.Frame):  # pylint:disable=too-many-ancestors
    """ A Panel for holding controls
        Also keeps tally if groups passed in, so that any options with special
        processing needs are processed in the correct group frame """

    def __init__(self, parent, options, items_per_row=1, radio_columns=4, header_text=None):
        logger.debug("Initializing %s: (parent: '%s', options: %s, items_per_row: %s, "
                     "radio_columns: %s, header_text: %s)",
                     self.__class__.__name__, parent, options, items_per_row, radio_columns,
                     header_text)
        super().__init__(parent)
        self.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.options = options

        self.header_text = header_text
        self.group_frames = dict()

        self.canvas = tk.Canvas(self, bd=0, highlightthickness=0)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.optsframe = ttk.Frame(self.canvas)
        self.optscanvas = self.canvas.create_window((0, 0), window=self.optsframe, anchor=tk.NW)

        self.build_panel(radio_columns)
        logger.debug("Initialized %s", self.__class__.__name__)

    def build_panel(self, radio_columns):
        """ Build the options frame for this command """
        logger.debug("Add Config Frame")
        self.add_scrollbar()
        self.canvas.bind("<Configure>", self.resize_frame)

        self.add_info()
        for key, val in self.options.items():
            if key == "helptext":
                continue
            frame = self.get_holding_frame(val["group"])
            ctl = ControlBuilder(frame,
                                 key,
                                 val["type"],
                                 val["default"],
                                 selected_value=val["value"],
                                 choices=val["choices"],
                                 is_radio=val["gui_radio"],
                                 rounding=val["rounding"],
                                 min_max=val["min_max"],
                                 helptext=val["helptext"],
                                 radio_columns=radio_columns)
            val["selected"] = ctl.tk_var
        logger.debug("Added Config Frame")

    def get_holding_frame(self, group):
        """ Return either the main options frame or a group frame """
        if group is None:
            return self.optsframe
        group = group.lower()
        if self.group_frames.get(group, None) is None:
            logger.debug("Creating new group frame for: %s", group)
            group_frame = ttk.LabelFrame(self.optsframe, text=group.title())
            group_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
            self.group_frames[group] = group_frame
        return self.group_frames[group]

    def add_scrollbar(self):
        """ Add a scrollbar to the options frame """
        logger.debug("Add Config Scrollbar")
        scrollbar = ttk.Scrollbar(self, command=self.canvas.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.config(yscrollcommand=scrollbar.set)
        self.optsframe.bind("<Configure>", self.update_scrollbar)
        logger.debug("Added Config Scrollbar")

    def update_scrollbar(self, event):  # pylint: disable=unused-argument
        """ Update the options frame scrollbar """
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def resize_frame(self, event):
        """ Resize the options frame to fit the canvas """
        logger.debug("Resize Config Frame")
        canvas_width = event.width
        self.canvas.itemconfig(self.optscanvas, width=canvas_width)
        logger.debug("Resized Config Frame")

    def add_info(self):
        """ Plugin information """
        info_frame = ttk.Frame(self.optsframe)
        info_frame.pack(fill=tk.X, expand=True)
        lbl = ttk.Label(info_frame, text="About:", width=20, anchor=tk.W)
        lbl.pack(padx=5, pady=5, side=tk.LEFT, anchor=tk.N)
        info = ttk.Label(info_frame, text=self.header_text)
        info.pack(padx=5, pady=5, fill=tk.X, expand=True)
        info.bind("<Configure>", adjust_wraplength)


class ControlBuilder():
    # TODO Expand out for cli options
    """
    Builds and returns a frame containing a tkinter control with label

    Currently only setup for config items

    Parameters
    ----------
    parent: tkinter object
        Parent tkinter object
    title: str
        Title of the control. Will be used for label text
    dtype: datatype object
        Datatype of the control
    default: str
        Default value for the control
    selected_value: str, optional
        Selected value for the control. If None, default will be used
    choices: list or tuple, object
        Used for combo boxes and radio control option setting
    is_radio: bool, optional
        Specifies to use a Radio control instead of combobox if choices are passed
    rounding: int or float, optional
        For slider controls. Sets the stepping
    min_max: int or float, optional
        For slider controls. Sets the min and max values
    helptext: str, optional
        Sets the tooltip text
    radio_columns: int, optional
        Sets the number of columns to use for grouping radio buttons
    label_width: int, optional
        Sets the width of the control label. Defaults to 20
    control_width: int, optional
        Sets the width of the control. Default is to auto expand
    """
    def __init__(self, parent, title, dtype, default,
                 selected_value=None, choices=None, is_radio=False, rounding=None,
                 min_max=None, helptext=None, radio_columns=3, label_width=20, control_width=None):
        logger.debug("Initializing %s: (parent: %s, title: %s, dtype: %s, default: %s, "
                     "selected_value: %s, choices: %s, is_radio: %s, rounding: %s, min_max: %s, "
                     "helptext: %s, radio_columns: %s, label_width: %s, control_width: %s)",
                     self.__class__.__name__, parent, title, dtype, default, selected_value,
                     choices, is_radio, rounding, min_max, helptext, radio_columns, label_width,
                     control_width)

        self.title = title
        self.default = default

        self.frame = self.control_frame(parent, helptext)
        self.control = self.set_control(dtype, choices, is_radio)
        self.tk_var = self.set_tk_var(dtype, selected_value)

        self.build_control(choices,
                           dtype,
                           rounding,
                           min_max,
                           radio_columns,
                           label_width,
                           control_width)
        logger.debug("Initialized: %s", self.__class__.__name__)

    # Frame, control type and varable
    def control_frame(self, parent, helptext):
        """ Frame to hold control and it's label """
        logger.debug("Build control frame")
        frame = ttk.Frame(parent)
        frame.pack(side=tk.TOP, fill=tk.X)
        if helptext is not None:
            helptext = self.format_helptext(helptext)
            Tooltip(frame, text=helptext, wraplength=720)
        logger.debug("Built control frame")
        return frame

    def format_helptext(self, helptext):
        """ Format the help text for tooltips """
        logger.debug("Format control help: '%s'", self.title)
        helptext = helptext.replace("\n\t", "\n  - ").replace("%%", "%")
        helptext = self.title + " - " + helptext
        logger.debug("Formatted control help: (title: '%s', help: '%s'", self.title, helptext)
        return helptext

    def set_control(self, dtype, choices, is_radio):
        """ Set the correct control type based on the datatype or for this option """
        if choices and is_radio:
            control = ttk.Radiobutton
        elif choices:
            control = ttk.Combobox
        elif dtype == bool:
            control = ttk.Checkbutton
        elif dtype in (int, float):
            control = ttk.Scale
        else:
            control = ttk.Entry
        logger.debug("Setting control '%s' to %s", self.title, control)
        return control

    def set_tk_var(self, dtype, selected_value):
        """ Correct variable type for control """
        logger.debug("Setting tk variable: (title: '%s', dtype: %s, selected_value: %s)",
                     self.title, dtype, selected_value)
        if dtype == bool:
            var = tk.BooleanVar
        elif dtype == int:
            var = tk.IntVar
        elif dtype == float:
            var = tk.DoubleVar
        else:
            var = tk.StringVar
        var = var(self.frame)
        val = self.default if selected_value is None else selected_value
        var.set(val)
        logger.debug("Set tk variable: (title: '%s', type: %s, value: '%s')",
                     self.title, type(var), val)
        return var

    # Build the full control
    def build_control(self, choices, dtype, rounding, min_max, radio_columns,
                      label_width, control_width):
        """ Build the correct control type for the option passed through """
        logger.debug("Build confog option control")
        self.build_control_label(label_width)
        self.build_one_control(choices, dtype, rounding, min_max, radio_columns, control_width)
        logger.debug("Built option control")

    def build_control_label(self, label_width):
        """ Label for control """
        logger.debug("Build control label: (title: '%s', label_width: %s)",
                     self.title, label_width)
        title = self.title.replace("_", " ").title()
        lbl = ttk.Label(self.frame, text=title, width=label_width, anchor=tk.W)
        lbl.pack(padx=5, pady=5, side=tk.LEFT, anchor=tk.N)
        logger.debug("Built control label: '%s'", self.title)

    def build_one_control(self, choices, dtype, rounding, min_max, radio_columns, control_width):
        """ Build and place the option controls """
        logger.debug("Build control: (title: '%s', control: %s, choices: %s, dtype: %s, "
                     "rounding: %s, min_max: %s: radio_columns: %s, control_width: %s)",
                     self.title, self.control, choices, dtype, rounding, min_max, radio_columns,
                     control_width)
        if self.control == ttk.Scale:
            ctl = self.slider_control(dtype, rounding, min_max)
        elif self.control == ttk.Radiobutton:
            ctl = self.radio_control(choices, radio_columns)
        else:
            ctl = self.control_to_optionsframe(choices)
        self.set_control_width(ctl, control_width)
        ctl.pack(padx=5, pady=5, fill=tk.X, expand=True)
        logger.debug("Built control: '%s'", self.title)

    @staticmethod
    def set_control_width(ctl, control_width):
        """ Set the control width if required """
        if control_width is not None:
            ctl.config(width=control_width)

    def radio_control(self, choices, columns):
        """ Create a group of radio buttons """
        logger.debug("Adding radio group: %s", self.title)
        ctl = ttk.Frame(self.frame)
        frames = list()
        for _ in range(columns):
            frame = ttk.Frame(ctl)
            frame.pack(padx=5, pady=5, fill=tk.X, expand=True, side=tk.LEFT, anchor=tk.N)
            frames.append(frame)

        for idx, choice in enumerate(choices):
            frame_id = idx % columns
            radio = ttk.Radiobutton(frames[frame_id],
                                    text=choice.title(),
                                    value=choice,
                                    variable=self.tk_var)
            radio.pack(anchor=tk.W)
            logger.debug("Adding radio option %s to column %s", choice, frame_id)
        logger.debug("Added radio group: '%s'", self.title)
        return ctl

    def slider_control(self, dtype, rounding, min_max):
        """ A slider control with corresponding Entry box """
        logger.debug("Add slider control to Options Frame: (title: '%s', dtype: %s, rounding: %s, "
                     "min_max: %s)", self.title, dtype, rounding, min_max)
        tbox = ttk.Entry(self.frame, width=8, textvariable=self.tk_var, justify=tk.RIGHT)
        tbox.pack(padx=(0, 5), side=tk.RIGHT)
        ctl = self.control(
            self.frame,
            variable=self.tk_var,
            command=lambda val, var=self.tk_var, dt=dtype, rn=rounding, mm=min_max:
            set_slider_rounding(val, var, dt, rn, mm))
        rc_menu = ContextMenu(tbox)
        rc_menu.cm_bind()
        ctl["from_"] = min_max[0]
        ctl["to"] = min_max[1]
        logger.debug("Added slider control to Options Frame: %s", self.title)
        return ctl

    def control_to_optionsframe(self, choices):
        """ Standard non-check buttons sit in the main options frame """
        logger.debug("Add control to Options Frame: (title: '%s', control: %s, choices: %s)",
                     self.title, self.control, choices)
        if self.control == ttk.Checkbutton:
            ctl = self.control(self.frame, variable=self.tk_var, text=None)
        else:
            ctl = self.control(self.frame, textvariable=self.tk_var)
            rc_menu = ContextMenu(ctl)
            rc_menu.cm_bind()
        if choices:
            logger.debug("Adding combo choices: %s", choices)
            ctl["values"] = [choice for choice in choices]
        logger.debug("Added control to Options Frame: %s", self.title)
        return ctl
