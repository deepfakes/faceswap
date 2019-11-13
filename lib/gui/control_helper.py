#!/usr/bin/env python3
""" Helper functions and classes for GUI controls """
import logging
import re

import tkinter as tk
from tkinter import ttk
from itertools import zip_longest
from functools import partial

from _tkinter import Tcl_Obj

from .tooltip import Tooltip
from .utils import ContextMenu, FileHandler, get_config, get_images

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

# We store Tooltips, ContextMenus and Commands globally when they are created
# Because we need to add them back to newly cloned widgets (they are not easily accessible from
# original config or are prone to getting destroyed when the original widget is destroyed)
_RECREATE_OBJECTS = dict(tooltips=dict(), commands=dict(), contextmenus=dict())


def get_tooltip(widget, text, wraplength=600):
    """ Store the tooltip layout and widget id in _TOOLTIPS and return a tooltip """
    _RECREATE_OBJECTS["tooltips"][str(widget)] = {"text": text,
                                                  "wraplength": wraplength}
    logger.debug("Adding to tooltips dict: (widget: %s. text: '%s', wraplength: %s)",
                 widget, text, wraplength)
    return Tooltip(widget, text=text, wraplength=wraplength)


def get_contextmenu(widget):
    """ Create a context menu, store its mapping and return """
    rc_menu = ContextMenu(widget)
    _RECREATE_OBJECTS["contextmenus"][str(widget)] = rc_menu
    logger.debug("Adding to Context menu: (widget: %s. rc_menu: %s)",
                 widget, rc_menu)
    return rc_menu


def add_command(name, func):
    """ For controls that execute commands, the command must be added to the _COMMAND list so that
        it can be added back to the widget during cloning """
    logger.debug("Adding to commands: %s - %s", name, func)
    _RECREATE_OBJECTS["commands"][str(name)] = func


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


class ControlPanelOption():
    """
    A class to hold a control panel option. A list of these is expected
    to be passed to the ControlPanel object.

    Parameters
    ----------
    title: str
        Title of the control. Will be used for label text and control naming
    dtype: datatype object
        Datatype of the control.
    group: str, optional
        The group that this control should sit with. If provided, all controls in the same
        group will be placed together. Default: None
    default: str, optional
        Default value for the control. If None is provided, then action will be dictated by
        whether "blank_nones" is set in ControlPanel
    initial_value: str, optional
        Initial value for the control. If None, default will be used
    choices: list or tuple, object
        Used for combo boxes and radio control option setting
    is_radio: bool, optional
        Specifies to use a Radio control instead of combobox if choices are passed
    rounding: int or float, optional
        For slider controls. Sets the stepping
    min_max: int or float, optional
        For slider controls. Sets the min and max values
    sysbrowser: dict, optional
        Adds Filesystem browser buttons to ttk.Entry options.
        Expects a dict: {sysbrowser: str, filetypes: str}
    helptext: str, optional
        Sets the tooltip text
    """

    def __init__(self, title, dtype,  # pylint:disable=too-many-arguments
                 group=None, default=None, initial_value=None, choices=None, is_radio=False,
                 rounding=None, min_max=None, sysbrowser=None, helptext=None):
        logger.debug("Initializing %s: (title: '%s', dtype: %s, group: %s, default: %s, "
                     "initial_value: %s, choices: %s, is_radio: %s, rounding: %s, min_max: %s, "
                     "sysbrowser: %s, helptext: '%s')", self.__class__.__name__, title, dtype,
                     group, default, initial_value, choices, is_radio, rounding, min_max,
                     sysbrowser, helptext)

        self.dtype = dtype
        self.sysbrowser = sysbrowser
        self._options = dict(title=title,
                             group=group,
                             default=default,
                             initial_value=initial_value,
                             choices=choices,
                             is_radio=is_radio,
                             rounding=rounding,
                             min_max=min_max,
                             helptext=helptext)
        self.control = self.get_control()
        self.tk_var = self.get_tk_var()
        logger.debug("Initialized %s", self.__class__.__name__)

    @property
    def name(self):
        """ Lowered title for naming """
        return self._options["title"].lower()

    @property
    def title(self):
        """ Title case title for naming with underscores removed """
        return self._options["title"].replace("_", " ").title()

    @property
    def group(self):
        """ Return group or _master if no group set """
        group = self._options["group"]
        group = "_master" if group is None else group
        return group

    @property
    def default(self):
        """ Return either selected value or default """
        return self._options["default"]

    @property
    def value(self):
        """ Return either selected value or default """
        val = self._options["initial_value"]
        val = self.default if val is None else val
        return val

    @property
    def choices(self):
        """ Return choices """
        return self._options["choices"]

    @property
    def is_radio(self):
        """ Return is_radio """
        return self._options["is_radio"]

    @property
    def rounding(self):
        """ Return rounding """
        return self._options["rounding"]

    @property
    def min_max(self):
        """ Return min_max """
        return self._options["min_max"]

    @property
    def helptext(self):
        """ Format and return help text for tooltips """
        helptext = self._options["helptext"]
        if helptext is None:
            return helptext
        logger.debug("Format control help: '%s'", self.name)
        if helptext.startswith("R|"):
            helptext = helptext[2:].replace("\nL|", "\n - ").replace("\n", "\n\n")
        else:
            helptext = helptext.replace("\n\t", "\n - ").replace("%%", "%")
        helptext = ". ".join(i.capitalize() for i in helptext.split(". "))
        helptext = self.title + " - " + helptext
        logger.debug("Formatted control help: (name: '%s', help: '%s'", self.name, helptext)
        return helptext

    def get(self):
        """ Return the value from the tk_var """
        return self.tk_var.get()

    def set(self, value):
        """ Set the tk_var to a new value """
        self.tk_var.set(value)

    def get_control(self):
        """ Set the correct control type based on the datatype or for this option """
        if self.choices and self.is_radio:
            control = ttk.Radiobutton
        elif self.choices:
            control = ttk.Combobox
        elif self.dtype == bool:
            control = ttk.Checkbutton
        elif self.dtype in (int, float):
            control = ttk.Scale
        else:
            control = ttk.Entry
        logger.debug("Setting control '%s' to %s", self.title, control)
        return control

    def get_tk_var(self):
        """ Correct variable type for control """
        if self.dtype == bool:
            var = tk.BooleanVar()
        elif self.dtype == int:
            var = tk.IntVar()
        elif self.dtype == float:
            var = tk.DoubleVar()
        else:
            var = tk.StringVar()
        logger.debug("Setting tk variable: (name: '%s', dtype: %s, tk_var: %s)",
                     self.name, self.dtype, var)
        return var


class ControlPanel(ttk.Frame):  # pylint:disable=too-many-ancestors
    """
    A Control Panel to hold control panel options.
    This class handles all of the formatting, placing and TK_Variables
    in a consistent manner.

    It can also provide dynamic columns for resizing widgets

    Parameters
    ----------
    parent: tk object
        Parent widget that should hold this control panel
    options: list of  ControlPanelOptions objects
        The list of controls that are to be built into this control panel
    label_width: int, optional
        The width that labels for controls should be set to.
        Defaults to 20
    columns: int, optional
        The maximum number of columns that this control panel should be able
        to accomodate. Setting to 1 means that there will only be 1 column
        regardless of how wide the control panel is. Higher numbers will
        dynamically fill extra columns if space permits. Defaults to 1
    option_columns: int, optional
        For checkbutton and radiobutton containers, how many options should
        be displayed on each row. Defaults to 4
    header_text: str, optional
        If provided, will place an information box at the top of the control
        panel with these contents.
    blank_nones: bool, optional
        How the control panel should handle Nones. If set to True then Nones
        will be converted to empty strings. Default: False
    """

    def __init__(self, parent, options,  # pylint:disable=too-many-arguments
                 label_width=20, columns=1, option_columns=4, header_text=None, blank_nones=True):
        logger.debug("Initializing %s: (parent: '%s', options: %s, label_width: %s, columns: %s, "
                     "option_columns: %s, header_text: %s, blank_nones: %s)",
                     self.__class__.__name__, parent, options, label_width, columns,
                     option_columns, header_text, blank_nones)
        super().__init__(parent)

        self.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.options = options
        self.controls = []
        self.label_width = label_width
        self.columns = columns
        self.option_columns = option_columns

        self.header_text = header_text
        self.group_frames = dict()

        self.canvas = tk.Canvas(self, bd=0, highlightthickness=0)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.mainframe, self.optsframe = self.get_opts_frame()
        self.optscanvas = self.canvas.create_window((0, 0), window=self.mainframe, anchor=tk.NW)

        self.build_panel(blank_nones)

        logger.debug("Initialized %s", self.__class__.__name__)

    def get_opts_frame(self):
        """ Return an autofill container for the options inside a main frame """
        mainframe = ttk.Frame(self.canvas)
        if self.header_text is not None:
            self.add_info(mainframe)
        optsframe = ttk.Frame(mainframe, name="opts_frame")
        optsframe.pack(expand=True, fill=tk.BOTH)
        holder = AutoFillContainer(optsframe, self.columns)
        logger.debug("Opts frames: '%s'", holder)
        return mainframe, holder

    def add_info(self, frame):
        """ Plugin information """
        gui_style = ttk.Style()
        gui_style.configure('White.TFrame', background='#FFFFFF')
        gui_style.configure('Header.TLabel',
                            background='#FFFFFF',
                            font=get_config().default_font + ("bold", ))
        gui_style.configure('Body.TLabel',
                            background='#FFFFFF')

        info_frame = ttk.Frame(frame, style='White.TFrame', relief=tk.SOLID)
        info_frame.pack(fill=tk.X, side=tk.TOP, expand=True, padx=10, pady=10)
        label_frame = ttk.Frame(info_frame, style='White.TFrame')
        label_frame.pack(padx=5, pady=5, fill=tk.X, expand=True)
        for idx, line in enumerate(self.header_text.splitlines()):
            if not line:
                continue
            style = "Header.TLabel" if idx == 0 else "Body.TLabel"
            info = ttk.Label(label_frame, text=line, style=style, anchor=tk.W)
            info.bind("<Configure>", adjust_wraplength)
            info.pack(fill=tk.X, padx=0, pady=0, expand=True, side=tk.TOP)

    def build_panel(self, blank_nones):
        """ Build the options frame for this command """
        logger.debug("Add Config Frame")
        self.add_scrollbar()
        self.canvas.bind("<Configure>", self.resize_frame)

        for option in self.options:
            group_frame = self.get_group_frame(option.group)
            ctl = ControlBuilder(group_frame["frame"],
                                 option,
                                 label_width=self.label_width,
                                 checkbuttons_frame=group_frame["chkbtns"],
                                 option_columns=self.option_columns,
                                 blank_nones=blank_nones)
            if group_frame["chkbtns"].items > 0:
                group_frame["chkbtns"].parent.pack(side=tk.BOTTOM, fill=tk.X, anchor=tk.NW)

            self.controls.append(ctl)
        for control in self.controls:
            filebrowser = control.filebrowser
            if filebrowser is not None:
                filebrowser.set_context_action_option(self.options)
        logger.debug("Added Config Frame")

    def get_group_frame(self, group):
        """ Return a new group frame """
        group = group.lower()
        if self.group_frames.get(group, None) is None:
            logger.debug("Creating new group frame for: %s", group)
            is_master = group == "_master"
            opts_frame = self.optsframe.subframe
            if is_master:
                group_frame = ttk.Frame(opts_frame, name=group.lower())
            else:
                group_frame = ttk.LabelFrame(opts_frame,
                                             text="" if is_master else group.title(),
                                             name=group.lower())

            group_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5, anchor=tk.NW)

            self.group_frames[group] = dict(frame=group_frame,
                                            chkbtns=self.checkbuttons_frame(group_frame))
        group_frame = self.group_frames[group]
        return group_frame

    def add_scrollbar(self):
        """ Add a scrollbar to the options frame """
        logger.debug("Add Config Scrollbar")
        scrollbar = ttk.Scrollbar(self, command=self.canvas.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.config(yscrollcommand=scrollbar.set)
        self.mainframe.bind("<Configure>", self.update_scrollbar)
        logger.debug("Added Config Scrollbar")

    def update_scrollbar(self, event):  # pylint: disable=unused-argument
        """ Update the options frame scrollbar """
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def resize_frame(self, event):
        """ Resize the options frame to fit the canvas """
        logger.debug("Resize Config Frame")
        canvas_width = event.width
        self.canvas.itemconfig(self.optscanvas, width=canvas_width)
        self.optsframe.rearrange_columns(canvas_width)
        logger.debug("Resized Config Frame")

    def checkbuttons_frame(self, frame):
        """ Build and format frame for holding the check buttons
            if is_master then check buttons will be placed in a LabelFrame
            otherwise in a standard frame """
        logger.debug("Add Options CheckButtons Frame")
        chk_frame = ttk.Frame(frame, name="chkbuttons")
        holder = AutoFillContainer(chk_frame, self.option_columns)
        logger.debug("Added Options CheckButtons Frame")
        return holder


class AutoFillContainer():
    """ A container object that autofills columns """
    def __init__(self, parent, columns):
        logger.debug("Initializing: %s: (parent: %s, columns: %s)", self.__class__.__name__,
                     parent, columns)
        self.max_columns = 4
        self.single_column_width = self.scale_column_width(288, 9)
        self.max_width = self.max_columns * self.single_column_width
        self.parent = parent
        self.columns = min(columns, self.max_columns)
        self._items = 0
        self._idx = 0
        self._widget_config = []  # Master list of all children in order
        self.subframes = self.set_subframes()
        logger.debug("Initialized: %s", self.__class__.__name__)

    @staticmethod
    def scale_column_width(original_size, original_fontsize):
        """ Scale the column width based on selected font size """
        font_size = get_config().user_config_dict["font_size"]
        if font_size == original_fontsize:
            return original_size
        scale = 1 + (((font_size / original_fontsize) - 1) / 2)
        retval = round(original_size * scale)
        logger.debug("scaled column width: (old_width: %s, scale: %s, new_width:%s)",
                     original_size, scale, retval)
        return retval

    @property
    def items(self):
        """ Returns the number of items held in this containter """
        return self._items

    @property
    def subframe(self):
        """ Returns the next subframe to be populated """
        frame = self.subframes[self._idx]
        next_idx = self._idx + 1 if self._idx + 1 < self.columns else 0
        logger.debug("current_idx: %s, next_idx: %s", self._idx, next_idx)
        self._idx = next_idx
        self._items += 1
        return frame

    def set_subframes(self):
        """ Set a subrame for each possible column """
        subframes = []
        for idx in range(self.max_columns):
            name = "af_subframe_{}".format(idx)
            subframe = ttk.Frame(self.parent, name=name)
            if idx < self.columns:
                # Only pack visible columns
                subframe.pack(padx=5, pady=5, side=tk.LEFT, anchor=tk.N, expand=True, fill=tk.X)
            subframes.append(subframe)
            logger.debug("Added subframe: %s", name)
        return subframes

    def rearrange_columns(self, width):
        """ On column number change redistribute widgets """
        if not self.validate(width):
            return

        new_columns = min(self.max_columns, max(1, width // self.single_column_width))
        logger.debug("Rearranging columns: (width: %s, old_columns: %s, new_columns: %s)",
                     width, self.columns, new_columns)
        self.columns = new_columns
        if not self._widget_config:
            self.compile_widget_config()
        self.destroy_children()
        self.repack_columns()
        # Reset counters
        self._items = 0
        self._idx = 0
        self.pack_widget_clones(self._widget_config)

    def validate(self, width):
        """ Validate that passed in width should trigger column re-arranging """
        if ((width < self.single_column_width and self.columns == 1) or
                (width > self.max_width and self.columns == self.max_columns)):
            logger.debug("width outside min/max thresholds: (min: %s, width: %s, max: %s)",
                         self.single_column_width, width, self.max_width)
            return False
        range_min = self.columns * self.single_column_width
        range_max = (self.columns + 1) * self.single_column_width
        if range_min < width < range_max:
            logger.debug("width outside next step refresh threshold: (step down: %s, width: %s,"
                         "step up: %s)", range_min, width, range_max)
            return False
        return True

    def compile_widget_config(self):
        """ Compile all children recursively in correct order if not already compiled """
        zipped = zip_longest(*(subframe.winfo_children() for subframe in self.subframes))
        children = [child for group in zipped for child in group if child is not None]
        self._widget_config = [{"class": child.__class__,
                                "id": str(child),
                                "tooltip": _RECREATE_OBJECTS["tooltips"].get(str(child), None),
                                "rc_menu": _RECREATE_OBJECTS["contextmenus"].get(str(child), None),
                                "pack_info": self.pack_config_cleaner(child),
                                "name": child.winfo_name(),
                                "config": self.config_cleaner(child),
                                "children": self.get_all_children_config(child, [])}
                               for idx, child in enumerate(children)]
        logger.debug("Compiled AutoFillContainer children: %s", self._widget_config)

    def get_all_children_config(self, widget, child_list):
        """ Return all children, recursively, of given widget """
        for child in widget.winfo_children():
            if child.winfo_ismapped():
                id_ = str(child)
                child_list.append({
                    "class": child.__class__,
                    "id": id_,
                    "tooltip": _RECREATE_OBJECTS["tooltips"].get(id_, None),
                    "rc_menu": _RECREATE_OBJECTS["contextmenus"].get(str(id_), None),
                    "pack_info": self.pack_config_cleaner(child),
                    "name": child.winfo_name(),
                    "config": self.config_cleaner(child),
                    "parent": child.winfo_parent()})
            self.get_all_children_config(child, child_list)
        return child_list

    @staticmethod
    def config_cleaner(widget):
        """ Some options don't like to be copied, so this returns a cleaned
            configuration from a widget
            We use config() instead of configure() because some items (TScale) do
            not populate configure()"""
        new_config = dict()
        for key in widget.config():
            if key == "class":
                continue
            val = widget.cget(key)
            if key in ("anchor", "justify") and val == "":
                continue
            val = str(val) if isinstance(val, Tcl_Obj) else val
            # Return correct command from master command dict
            val = _RECREATE_OBJECTS["commands"][val] if key == "command" and val != "" else val
            new_config[key] = val
        return new_config

    @staticmethod
    def pack_config_cleaner(widget):
        """ Some options don't like to be copied, so this returns a cleaned
            configuration from a widget """
        return {key: val for key, val in widget.pack_info().items() if key != "in"}

    def destroy_children(self):
        """ Destroy the currently existing widgets """
        for subframe in self.subframes:
            for child in subframe.winfo_children():
                child.destroy()

    def repack_columns(self):
        """ Repack or unpack columns based on display columns """
        for idx, subframe in enumerate(self.subframes):
            logger.trace("Processing subframe: %s", subframe)
            if idx < self.columns and not subframe.winfo_ismapped():
                logger.trace("Packing subframe: %s", subframe)
                subframe.pack(padx=5, pady=5, side=tk.LEFT, anchor=tk.N, expand=True, fill=tk.X)
            elif idx >= self.columns and subframe.winfo_ismapped():
                logger.trace("Forgetting subframe: %s", subframe)
                subframe.pack_forget()

    def pack_widget_clones(self, widget_dicts, old_children=None, new_children=None):
        """ Widgets cannot be given a new parent so we need to clone
            them and then pack the new widget """
        for widget_dict in widget_dicts:
            logger.debug("Cloning widget: %s", widget_dict)
            old_children = [] if old_children is None else old_children
            new_children = [] if new_children is None else new_children
            if widget_dict.get("parent", None) is not None:
                parent = new_children[old_children.index(widget_dict["parent"])]
                logger.trace("old parent: '%s', new_parent: '%s'", widget_dict["parent"], parent)
            else:
                # Get the next subframe if this doesn't have a logged parent
                parent = self.subframe
            clone = widget_dict["class"](parent, name=widget_dict["name"])
            if widget_dict["config"] is not None:
                clone.configure(**widget_dict["config"])
            if widget_dict["tooltip"] is not None:
                Tooltip(clone, **widget_dict["tooltip"])
            rc_menu = widget_dict["rc_menu"]
            if rc_menu is not None:
                # Re-initialize for new widget and bind
                rc_menu.__init__(widget=clone)
                rc_menu.cm_bind()
            clone.pack(**widget_dict["pack_info"])
            old_children.append(widget_dict["id"])
            new_children.append(clone)
            if widget_dict.get("children", None) is not None:
                self.pack_widget_clones(widget_dict["children"], old_children, new_children)


class ControlBuilder():
    """
    Builds and returns a frame containing a tkinter control with label
    This should only be called from the ControlPanel class

    Parameters
    ----------
    parent: tkinter object
        Parent tkinter object
    option: ControlPanelOption object
        Holds all of the required option information
    option_columns: int
        Number of options to put on a single row for checkbuttons/radiobuttons
    label_width: int
        Sets the width of the control label
    checkbuttons_frame: tk.frame
        If a checkbutton frame is passed in, then checkbuttons will be placed in this frame
        rather than the main options frame
    blank_nones: bool
        Sets selected values to an empty string rather than None if this is true.
    """
    def __init__(self, parent, option, option_columns,  # pylint: disable=too-many-arguments
                 label_width, checkbuttons_frame, blank_nones):
        logger.debug("Initializing %s: (parent: %s, option: %s, option_columns: %s, "
                     "label_width: %s, checkbuttons_frame: %s, blank_nones: %s)",
                     self.__class__.__name__, parent, option, option_columns, label_width,
                     checkbuttons_frame, blank_nones)

        self.option = option
        self.option_columns = option_columns
        self.helpset = False
        self.label_width = label_width
        self.filebrowser = None

        self.frame = self.control_frame(parent)
        self.chkbtns = checkbuttons_frame

        self.set_tk_var(blank_nones)
        self.build_control()
        logger.debug("Initialized: %s", self.__class__.__name__)

    # Frame, control type and varable
    def control_frame(self, parent):
        """ Frame to hold control and it's label """
        logger.debug("Build control frame")
        frame = ttk.Frame(parent, name="fr_{}".format(self.option.name))
        frame.pack(fill=tk.X)
        logger.debug("Built control frame")
        return frame

    def set_tk_var(self, blank_nones):
        """ Correct variable type for control """
        val = "" if self.option.value is None and blank_nones else self.option.value
        self.option.tk_var.set(val)
        logger.debug("Set tk variable: (option: '%s', variable: %s, value: '%s')",
                     self.option.name, self.option.tk_var, val)

    # Build the full control
    def build_control(self):
        """ Build the correct control type for the option passed through """
        logger.debug("Build config option control")
        if self.option.control not in (ttk.Checkbutton, ttk.Radiobutton):
            self.build_control_label()
        self.build_one_control()
        logger.debug("Built option control")

    def build_control_label(self):
        """ Label for control """
        logger.debug("Build control label: (option: '%s')", self.option.name)
        lbl = ttk.Label(self.frame, text=self.option.title, width=self.label_width, anchor=tk.W)
        lbl.pack(padx=5, pady=5, side=tk.LEFT, anchor=tk.N)
        if self.option.helptext is not None:
            get_tooltip(lbl, text=self.option.helptext, wraplength=600)
        logger.debug("Built control label: (widget: '%s', title: '%s'",
                     self.option.name, self.option.title)

    def build_one_control(self):
        """ Build and place the option controls """
        logger.debug("Build control: '%s')", self.option.name)
        if self.option.control == ttk.Scale:
            ctl = self.slider_control()
        elif self.option.control == ttk.Radiobutton:
            ctl = self.radio_control()
        elif self.option.control == ttk.Checkbutton:
            ctl = self.control_to_checkframe()
        else:
            ctl = self.control_to_optionsframe()
        if self.option.control != ttk.Checkbutton:
            ctl.pack(padx=5, pady=5, fill=tk.X, expand=True)
            if self.option.helptext is not None and not self.helpset:
                get_tooltip(ctl, text=self.option.helptext, wraplength=600)

        logger.debug("Built control: '%s'", self.option.name)

    def radio_control(self):
        """ Create a group of radio buttons """
        logger.debug("Adding radio group: %s", self.option.name)
        all_help = [line for line in self.option.helptext.splitlines()]
        if any(line.startswith(" - ") for line in all_help):
            intro = all_help[0]
        helpitems = {re.sub(r'[^A-Za-z0-9\-]+', '',
                            line.split()[1].lower()): " ".join(line.split()[1:])
                     for line in all_help
                     if line.startswith(" - ")}
        ctl = ttk.LabelFrame(self.frame,
                             text=self.option.title,
                             name="radio_labelframe")
        radio_holder = AutoFillContainer(ctl, self.option_columns)
        for choice in self.option.choices:
            radio = ttk.Radiobutton(radio_holder.subframe,
                                    text=choice.replace("_", " ").title(),
                                    value=choice,
                                    variable=self.option.tk_var)
            if choice.lower() in helpitems:
                self.helpset = True
                helptext = helpitems[choice.lower()].capitalize()
                helptext = "{}\n\n - {}".format(
                    '. '.join(item.capitalize() for item in helptext.split('. ')),
                    intro)
                get_tooltip(radio, text=helptext, wraplength=600)
            radio.pack(anchor=tk.W)
            logger.debug("Added radio option %s", choice)
        return radio_holder.parent

    def slider_control(self):
        """ A slider control with corresponding Entry box """
        logger.debug("Add slider control to Options Frame: (widget: '%s', dtype: %s, "
                     "rounding: %s, min_max: %s)", self.option.name, self.option.dtype,
                     self.option.rounding, self.option.min_max)
        tbox = ttk.Entry(self.frame,
                         width=8,
                         textvariable=self.option.tk_var,
                         justify=tk.RIGHT,
                         font=get_config().default_font)
        tbox.pack(padx=(0, 5), side=tk.RIGHT)
        cmd = partial(set_slider_rounding,
                      var=self.option.tk_var,
                      d_type=self.option.dtype,
                      round_to=self.option.rounding,
                      min_max=self.option.min_max)
        ctl = self.option.control(self.frame, variable=self.option.tk_var, command=cmd)
        add_command(ctl.cget("command"), cmd)
        rc_menu = get_contextmenu(tbox)
        rc_menu.cm_bind()
        ctl["from_"] = self.option.min_max[0]
        ctl["to"] = self.option.min_max[1]
        logger.debug("Added slider control to Options Frame: %s", self.option.name)
        return ctl

    def control_to_optionsframe(self):
        """ Standard non-check buttons sit in the main options frame """
        logger.debug("Add control to Options Frame: (widget: '%s', control: %s, choices: %s)",
                     self.option.name, self.option.control, self.option.choices)
        if self.option.control == ttk.Checkbutton:
            ctl = self.option.control(self.frame, variable=self.option.tk_var, text=None)
        else:
            if self.option.sysbrowser is not None:
                self.filebrowser = FileBrowser(self.option.tk_var,
                                               self.frame,
                                               self.option.sysbrowser)
            ctl = self.option.control(self.frame,
                                      textvariable=self.option.tk_var,
                                      font=get_config().default_font)
            rc_menu = get_contextmenu(ctl)
            rc_menu.cm_bind()
        if self.option.choices:
            logger.debug("Adding combo choices: %s", self.option.choices)
            ctl["values"] = [choice for choice in self.option.choices]
        logger.debug("Added control to Options Frame: %s", self.option.name)
        return ctl

    def control_to_checkframe(self):
        """ Add checkbuttons to the checkbutton frame """
        logger.debug("Add control checkframe: '%s'", self.option.name)
        chkframe = self.chkbtns.subframe
        ctl = self.option.control(chkframe,
                                  variable=self.option.tk_var,
                                  text=self.option.title,
                                  name=self.option.name)
        get_tooltip(ctl, text=self.option.helptext, wraplength=600)
        ctl.pack(side=tk.TOP, anchor=tk.W)
        logger.debug("Added control checkframe: '%s'", self.option.name)
        return ctl


class FileBrowser():
    """ Add FileBrowser buttons to control and handle routing """
    def __init__(self, tk_var, control_frame, sysbrowser_dict):
        logger.debug("Initializing: %s: (tk_var: %s, control_frame: %s, sysbrowser_dict: %s)",
                     self.__class__.__name__, tk_var, control_frame, sysbrowser_dict)
        self.tk_var = tk_var
        self.frame = control_frame
        self.browser = sysbrowser_dict["browser"]
        self.filetypes = sysbrowser_dict["filetypes"]
        self.action_option = self.format_action_option(sysbrowser_dict.get("action_option", None))
        self.command = sysbrowser_dict.get("command", None)
        self.destination = sysbrowser_dict.get("destination", None)
        self.add_browser_buttons()
        logger.debug("Initialized: %s", self.__class__.__name__)

    @property
    def helptext(self):
        """ Dict containing tooltip text for buttons """
        retval = dict(folder="Select a folder...",
                      load="Select a file...",
                      load_multi="Select one or more files...",
                      context="Select a file or folder...",
                      save="Select a save location...")
        return retval

    @staticmethod
    def format_action_option(action_option):
        """ Format the action option to remove any dashes at the start """
        if action_option is None:
            return action_option
        if action_option.startswith("--"):
            return action_option[2:]
        if action_option.startswith("-"):
            return action_option[1:]
        return action_option

    def add_browser_buttons(self):
        """ Add correct file browser button for control """
        logger.debug("Adding browser buttons: (sysbrowser: %s", self.browser)
        for browser in self.browser:
            img = get_images().icons[browser]
            action = getattr(self, "ask_" + browser)
            cmd = partial(action, filepath=self.tk_var, filetypes=self.filetypes)
            fileopn = ttk.Button(self.frame, image=img, command=cmd)
            add_command(fileopn.cget("command"), cmd)
            fileopn.pack(padx=(0, 5), side=tk.RIGHT)
            get_tooltip(fileopn, text=self.helptext[browser], wraplength=600)
            logger.debug("Added browser buttons: (action: %s, filetypes: %s",
                         action, self.filetypes)

    def set_context_action_option(self, options):
        """ Set the tk_var for the source action option
            that dictates the context sensitive file browser. """
        if self.browser != ["context"]:
            return
        actions = {opt.name: opt.tk_var for opt in options}
        logger.debug("Settiong action option for opt %s", self.action_option)
        self.action_option = actions[self.action_option]

    @staticmethod
    def ask_folder(filepath, filetypes=None):
        """ Pop-up to get path to a directory
            :param filepath: tkinter StringVar object
            that will store the path to a directory.
            :param filetypes: Unused argument to allow
            filetypes to be given in ask_load(). """
        dirname = FileHandler("dir", filetypes).retfile
        if dirname:
            logger.debug(dirname)
            filepath.set(dirname)

    @staticmethod
    def ask_load(filepath, filetypes):
        """ Pop-up to get path to a file """
        filename = FileHandler("filename", filetypes).retfile
        if filename:
            logger.debug(filename)
            filepath.set(filename)

    @staticmethod
    def ask_load_multi(filepath, filetypes):
        """ Pop-up to get path to a file """
        filenames = FileHandler("filename_multi", filetypes).retfile
        if filenames:
            final_names = " ".join("\"{}\"".format(fname) for fname in filenames)
            logger.debug(final_names)
            filepath.set(final_names)

    @staticmethod
    def ask_save(filepath, filetypes=None):
        """ Pop-up to get path to save a new file """
        filename = FileHandler("savefilename", filetypes).retfile
        if filename:
            logger.debug(filename)
            filepath.set(filename)

    @staticmethod
    def ask_nothing(filepath, filetypes=None):  # pylint:disable=unused-argument
        """ Method that does nothing, used for disabling open/save pop up """
        return

    def ask_context(self, filepath, filetypes):
        """ Method to pop the correct dialog depending on context """
        logger.debug("Getting context filebrowser")
        selected_action = self.action_option.get()
        selected_variable = self.destination
        filename = FileHandler("context",
                               filetypes,
                               command=self.command,
                               action=selected_action,
                               variable=selected_variable).retfile
        if filename:
            logger.debug(filename)
            filepath.set(filename)
