#!/usr/bin/env python3
""" Helper functions and classes for GUI controls """
import logging
import re

import tkinter as tk
from tkinter import colorchooser, ttk
from itertools import zip_longest
from functools import partial

from _tkinter import Tcl_Obj, TclError

from .custom_widgets import ContextMenu, MultiOption, Tooltip
from .utils import FileHandler, get_config, get_images

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

# We store Tooltips, ContextMenus and Commands globally when they are created
# Because we need to add them back to newly cloned widgets (they are not easily accessible from
# original config or are prone to getting destroyed when the original widget is destroyed)
_RECREATE_OBJECTS = dict(tooltips=dict(), commands=dict(), contextmenus=dict())


def _get_tooltip(widget, text, wraplength=600):
    """ Store the tooltip layout and widget id in _TOOLTIPS and return a tooltip """
    _RECREATE_OBJECTS["tooltips"][str(widget)] = {"text": text,
                                                  "wraplength": wraplength}
    logger.debug("Adding to tooltips dict: (widget: %s. text: '%s', wraplength: %s)",
                 widget, text, wraplength)
    return Tooltip(widget, text=text, wraplength=wraplength)


def _get_contextmenu(widget):
    """ Create a context menu, store its mapping and return """
    rc_menu = ContextMenu(widget)
    _RECREATE_OBJECTS["contextmenus"][str(widget)] = rc_menu
    logger.debug("Adding to Context menu: (widget: %s. rc_menu: %s)",
                 widget, rc_menu)
    return rc_menu


def _add_command(name, func):
    """ For controls that execute commands, the command must be added to the _COMMAND list so that
        it can be added back to the widget during cloning """
    logger.debug("Adding to commands: %s - %s", name, func)
    _RECREATE_OBJECTS["commands"][str(name)] = func


def set_slider_rounding(value, var, d_type, round_to, min_max):
    """ Set the value of sliders underlying variable based on their datatype,
    rounding value and min/max.

    Parameters
    ----------
    var: tkinter.Var
        The variable to set the value for
    d_type: [:class:`int`, :class:`float`]
        The type of value that is stored in :attr:`var`
    round_to: int
        If :attr:`dtype` is :class:`float` then this is the decimal place rounding for :attr:`var`.
        If :attr:`dtype` is :class:`int` then this is the number of steps between each increment
        for :attr:`var`
    min_max: tuple (`int`, `int`)
        The (``min``, ``max``) values that this slider accepts
    """
    if d_type == float:
        var.set(round(float(value), round_to))
    else:
        steps = range(min_max[0], min_max[1] + round_to, round_to)
        value = min(steps, key=lambda x: abs(x - int(float(value))))
        var.set(value)


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
    subgroup: str, optional
        The subgroup that this option belongs to. If provided, will group options in the same
        subgroups together for the same layout as option/check boxes. Default: ``None``
    default: str, optional
        Default value for the control. If None is provided, then action will be dictated by
        whether "blank_nones" is set in ControlPanel
    initial_value: str, optional
        Initial value for the control. If None, default will be used
    choices: list or tuple, object
        Used for combo boxes and radio control option setting. Set to `"colorchooser"` for a color
        selection dialog.
    is_radio: bool, optional
        Specifies to use a Radio control instead of combobox if choices are passed
    is_multi_option:
        Specifies to use a Multi Check Button option group for the specified control
    rounding: int or float, optional
        For slider controls. Sets the stepping
    min_max: int or float, optional
        For slider controls. Sets the min and max values
    sysbrowser: dict, optional
        Adds Filesystem browser buttons to ttk.Entry options.
        Expects a dict: {sysbrowser: str, filetypes: str}
    helptext: str, optional
        Sets the tooltip text
    track_modified: bool, optional
        Set whether to set a callback trace indicating that the parameter has been modified.
        Default: False
    command: str, optional
        Required if tracking modified. The command that this option belongs to. Default: None
    """

    def __init__(self, title, dtype,  # pylint:disable=too-many-arguments
                 group=None, subgroup=None, default=None, initial_value=None, choices=None,
                 is_radio=False, is_multi_option=False, rounding=None, min_max=None,
                 sysbrowser=None, helptext=None, track_modified=False, command=None):
        logger.debug("Initializing %s: (title: '%s', dtype: %s, group: %s, subgroup: %s, "
                     "default: %s, initial_value: %s, choices: %s, is_radio: %s, "
                     "is_multi_option: %s, rounding: %s, min_max: %s, sysbrowser: %s, "
                     "helptext: '%s', track_modified: %s, command: '%s')", self.__class__.__name__,
                     title, dtype, group, subgroup, default, initial_value, choices, is_radio,
                     is_multi_option, rounding, min_max, sysbrowser, helptext, track_modified,
                     command)

        self.dtype = dtype
        self.sysbrowser = sysbrowser
        self._command = command
        self._options = dict(title=title,
                             subgroup=subgroup,
                             group=group,
                             default=default,
                             initial_value=initial_value,
                             choices=choices,
                             is_radio=is_radio,
                             is_multi_option=is_multi_option,
                             rounding=rounding,
                             min_max=min_max,
                             helptext=helptext)
        self.control = self.get_control()
        self.tk_var = self.get_tk_var(track_modified)
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
    def subgroup(self):
        """ str: The subgroup for the option, or ``None`` if none provided. """
        return self._options["subgroup"]

    @property
    def default(self):
        """ Return either selected value or default """
        return self._options["default"]

    @property
    def value(self):
        """ Return either initial value or default """
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
    def is_multi_option(self):
        """ bool: ``True`` if the control should be contained in a multi check button group,
        otherwise ``False``. """
        return self._options["is_multi_option"]

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
        """ Return the value from the tk_var

        Notes
        -----
        tk variables don't like empty values if it's not a stringVar. This seems to be pretty
        much the only reason that a get() call would fail, so replace any numerical variable
        with it's numerical zero equivalent on a TCL Error. Only impacts variables linked
        to Entry widgets.
        """
        try:
            val = self.tk_var.get()
        except TclError:
            if isinstance(self.tk_var, tk.IntVar):
                val = 0
            elif isinstance(self.tk_var, tk.DoubleVar):
                val = 0.0
            else:
                raise
        return val

    def set(self, value):
        """ Set the tk_var to a new value """
        self.tk_var.set(value)

    def set_initial_value(self, value):
        """ Set the initial_value to the given value

        Parameters
        ----------
        value: varies
            The value to set the initial value attribute to
        """
        logger.debug("Setting inital value for %s to %s", self.name, value)
        self._options["initial_value"] = value

    def get_control(self):
        """ Set the correct control type based on the datatype or for this option """
        if self.choices and self.is_radio:
            control = "radio"
        elif self.choices and self.is_multi_option:
            control = "multi"
        elif self.choices and self.choices == "colorchooser":
            control = "colorchooser"
        elif self.choices:
            control = ttk.Combobox
        elif self.dtype == bool:
            control = ttk.Checkbutton
        elif self.dtype in (int, float):
            control = "scale"
        else:
            control = ttk.Entry
        logger.debug("Setting control '%s' to %s", self.title, control)
        return control

    def get_tk_var(self, track_modified):
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
        if track_modified and self._command is not None:
            logger.debug("Tracking variable modification: %s", self.name)
            var.trace("w",
                      lambda name, index, mode, cmd=self._command: self._modified_callback(cmd))

        if track_modified and self._command in ("train", "convert") and self.title == "Model Dir":
            var.trace("w", lambda name, index, mode, v=var: self._model_callback(v))

        return var

    @staticmethod
    def _modified_callback(command):
        """ Set the modified variable for this tab to TRUE

        On initial setup the notebook won't yet exist, and we don't want to track the changes
        for initial variables anyway, so make sure notebook exists prior to performing the callback
        """
        config = get_config()
        if config.command_notebook is None:
            return
        config.set_modified_true(command)

    @staticmethod
    def _model_callback(var):
        """ Set a callback to load model stats for existing models when a model
        folder is selected """
        config = get_config()
        if not config.user_config_dict["auto_load_model_stats"]:
            logger.debug("Session updating disabled by user config")
            return
        if config.tk_vars["runningtask"].get():
            logger.debug("Task running. Not updating session")
            return
        folder = var.get()
        logger.debug("Setting analysis model folder callback: '%s'", folder)
        get_config().tk_vars["analysis_folder"].set(folder)


class ControlPanel(ttk.Frame):  # pylint:disable=too-many-ancestors
    """
    A Control Panel to hold control panel options.
    This class handles all of the formatting, placing and TK_Variables
    in a consistent manner.

    It can also provide dynamic columns for resizing widgets

    Parameters
    ----------
    parent: tkinter object
        Parent widget that should hold this control panel
    options: list of  ControlPanelOptions objects
        The list of controls that are to be built into this control panel
    label_width: int, optional
        The width that labels for controls should be set to.
        Defaults to 20
    columns: int, optional
        The initial number of columns to set the layout for. Default: 1
    max_columns: int, optional
        The maximum number of columns that this control panel should be able
        to accommodate. Setting to 1 means that there will only be 1 column
        regardless of how wide the control panel is. Higher numbers will
        dynamically fill extra columns if space permits. Defaults to 4
    option_columns: int, optional
        For check-button and radio-button containers, how many options should
        be displayed on each row. Defaults to 4
    header_text: str, optional
        If provided, will place an information box at the top of the control
        panel with these contents.
    blank_nones: bool, optional
        How the control panel should handle None values. If set to True then None values will be
        converted to empty strings. Default: False
    scrollbar: bool, optional
        ``True`` if a scrollbar should be added to the control panel, otherwise ``False``.
        Default: ``True``
    """

    def __init__(self, parent, options,  # pylint:disable=too-many-arguments
                 label_width=20, columns=1, max_columns=4, option_columns=4, header_text=None,
                 blank_nones=True, scrollbar=True):
        logger.debug("Initializing %s: (parent: '%s', options: %s, label_width: %s, columns: %s, "
                     "max_columns: %s, option_columns: %s, header_text: %s, blank_nones: %s, "
                     "scrollbar: %s)",
                     self.__class__.__name__, parent, options, label_width, columns, max_columns,
                     option_columns, header_text, blank_nones, scrollbar)
        super().__init__(parent)

        self.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.options = options
        self.controls = []
        self.label_width = label_width
        self.columns = columns
        self.max_columns = max_columns
        self.option_columns = option_columns

        self.header_text = header_text
        self.group_frames = dict()
        self._sub_group_frames = dict()

        self._canvas = tk.Canvas(self, bd=0, highlightthickness=0)
        self._canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.mainframe, self.optsframe = self.get_opts_frame()
        self._optscanvas = self._canvas.create_window((0, 0), window=self.mainframe, anchor=tk.NW)
        self.build_panel(blank_nones, scrollbar)

        logger.debug("Initialized %s", self.__class__.__name__)

    @staticmethod
    def _adjust_wraplength(event):
        """ dynamically adjust the wrap length of a label on event """
        label = event.widget
        label.configure(wraplength=event.width - 1)

    def get_opts_frame(self):
        """ Return an auto-fill container for the options inside a main frame """
        mainframe = ttk.Frame(self._canvas)
        if self.header_text is not None:
            self.add_info(mainframe)
        optsframe = ttk.Frame(mainframe, name="opts_frame")
        optsframe.pack(expand=True, fill=tk.BOTH)
        holder = AutoFillContainer(optsframe, self.columns, self.max_columns)
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
            info.bind("<Configure>", self._adjust_wraplength)
            info.pack(fill=tk.X, padx=0, pady=0, expand=True, side=tk.TOP)

    def build_panel(self, blank_nones, scrollbar):
        """ Build the options frame for this command """
        logger.debug("Add Config Frame")
        if scrollbar:
            self.add_scrollbar()
        self._canvas.bind("<Configure>", self.resize_frame)

        for option in self.options:
            group_frame = self.get_group_frame(option.group)
            sub_group_frame = self._get_subgroup_frame(group_frame["frame"], option.subgroup)
            frame = group_frame["frame"] if sub_group_frame is None else sub_group_frame.subframe

            ctl = ControlBuilder(frame,
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
        scrollbar = ttk.Scrollbar(self, command=self._canvas.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self._canvas.config(yscrollcommand=scrollbar.set)
        self.mainframe.bind("<Configure>", self.update_scrollbar)
        logger.debug("Added Config Scrollbar")

    def update_scrollbar(self, event):  # pylint: disable=unused-argument
        """ Update the options frame scrollbar """
        self._canvas.configure(scrollregion=self._canvas.bbox("all"))

    def resize_frame(self, event):
        """ Resize the options frame to fit the canvas """
        logger.debug("Resize Config Frame")
        canvas_width = event.width
        self._canvas.itemconfig(self._optscanvas, width=canvas_width)
        self.optsframe.rearrange_columns(canvas_width)
        logger.debug("Resized Config Frame")

    def checkbuttons_frame(self, frame):
        """ Build and format frame for holding the check buttons
            if is_master then check buttons will be placed in a LabelFrame
            otherwise in a standard frame """
        logger.debug("Add Options CheckButtons Frame")
        chk_frame = ttk.Frame(frame, name="chkbuttons")
        holder = AutoFillContainer(chk_frame, self.option_columns, self.option_columns)
        logger.debug("Added Options CheckButtons Frame")
        return holder

    def _get_subgroup_frame(self, parent, subgroup):
        if subgroup is None:
            return subgroup
        if subgroup not in self._sub_group_frames:
            sub_frame = ttk.Frame(parent, name="subgroup_{}".format(subgroup))
            self._sub_group_frames[subgroup] = AutoFillContainer(sub_frame,
                                                                 self.option_columns,
                                                                 self.option_columns)
            sub_frame.pack(anchor=tk.W, expand=True, fill=tk.X)
            logger.debug("Added Subgroup Frame: %s", subgroup)
        return self._sub_group_frames[subgroup]


class AutoFillContainer():
    """ A container object that auto-fills columns """
    def __init__(self, parent, initial_columns, max_columns):
        logger.debug("Initializing: %s: (parent: %s, initial_columns: %s, max_columns: %s)",
                     self.__class__.__name__, parent, initial_columns, max_columns)
        self.max_columns = max_columns
        self.columns = initial_columns
        self.parent = parent
#        self.columns = min(columns, self.max_columns)
        self.single_column_width = self.scale_column_width(288, 9)
        self.max_width = self.max_columns * self.single_column_width
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
        """ Returns the number of items held in this container """
        return self._items

    @property
    def subframe(self):
        """ Returns the next sub-frame to be populated """
        frame = self.subframes[self._idx]
        next_idx = self._idx + 1 if self._idx + 1 < self.columns else 0
        logger.debug("current_idx: %s, next_idx: %s", self._idx, next_idx)
        self._idx = next_idx
        self._items += 1
        return frame

    def set_subframes(self):
        """ Set a sub-frame for each possible column """
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
                                "children": self.get_all_children_config(child, []),
                                # Some children have custom kwargs, so keep dicts in sync
                                "custom_kwargs": dict()}
                               for idx, child in enumerate(children)]
        logger.debug("Compiled AutoFillContainer children: %s", self._widget_config)

    def get_all_children_config(self, widget, child_list):
        """ Return all children, recursively, of given widget """
        for child in widget.winfo_children():
            if child.winfo_ismapped():
                id_ = str(child)
                if child.__class__.__name__ == "MultiOption":
                    # MultiOption checkbox groups are a custom object with additional parameter
                    # requirements.
                    custom_kwargs = dict(
                        value=child._value,  # pylint:disable=protected-access
                        variable=child._master_variable)  # pylint:disable=protected-access
                else:
                    custom_kwargs = dict()

                child_list.append({
                    "class": child.__class__,
                    "id": id_,
                    "tooltip": _RECREATE_OBJECTS["tooltips"].get(id_, None),
                    "rc_menu": _RECREATE_OBJECTS["contextmenus"].get(str(id_), None),
                    "pack_info": self.pack_config_cleaner(child),
                    "name": child.winfo_name(),
                    "config": self.config_cleaner(child),
                    "parent": child.winfo_parent(),
                    "custom_kwargs": custom_kwargs})
            self.get_all_children_config(child, child_list)
        return child_list

    @staticmethod
    def config_cleaner(widget):
        """ Some options don't like to be copied, so this returns a cleaned
            configuration from a widget
            We use config() instead of configure() because some items (ttk Scale) do
            not populate configure()"""
        new_config = dict()
        for key in widget.config():
            if key == "class":
                continue
            val = widget.cget(key)
            # Some keys default to "" but tkinter doesn't like to set config to this value
            # so skip them to use default value.
            if key in ("anchor", "justify", "compound") and val == "":
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
                # Get the next sub-frame if this doesn't have a logged parent
                parent = self.subframe
            clone = widget_dict["class"](parent,
                                         name=widget_dict["name"],
                                         **widget_dict["custom_kwargs"])
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
        Number of options to put on a single row for check-buttons/radio-buttons
    label_width: int
        Sets the width of the control label
    checkbuttons_frame: tkinter.frame
        If a check-button frame is passed in, then check-buttons will be placed in this frame
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

    # Frame, control type and variable
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
        if self.option.control not in (ttk.Checkbutton, "radio", "multi", "colorchooser"):
            self.build_control_label()
        self.build_one_control()
        logger.debug("Built option control")

    def build_control_label(self):
        """ Label for control """
        logger.debug("Build control label: (option: '%s')", self.option.name)
        lbl = ttk.Label(self.frame, text=self.option.title, width=self.label_width, anchor=tk.W)
        lbl.pack(padx=5, pady=5, side=tk.LEFT, anchor=tk.N)
        if self.option.helptext is not None:
            _get_tooltip(lbl, text=self.option.helptext, wraplength=600)
        logger.debug("Built control label: (widget: '%s', title: '%s'",
                     self.option.name, self.option.title)

    def build_one_control(self):
        """ Build and place the option controls """
        logger.debug("Build control: '%s')", self.option.name)
        if self.option.control == "scale":
            ctl = self.slider_control()
        elif self.option.control in ("radio", "multi"):
            ctl = self._multi_option_control(self.option.control)
        elif self.option.control == "colorchooser":
            ctl = self._color_control()
        elif self.option.control == ttk.Checkbutton:
            ctl = self.control_to_checkframe()
        else:
            ctl = self.control_to_optionsframe()
        if self.option.control != ttk.Checkbutton:
            ctl.pack(padx=5, pady=5, fill=tk.X, expand=True)
            if self.option.helptext is not None and not self.helpset:
                _get_tooltip(ctl, text=self.option.helptext, wraplength=600)

        logger.debug("Built control: '%s'", self.option.name)

    def _multi_option_control(self, option_type):
        """ Create a group of buttons for single or multi-select

        Parameters
        ----------
        option_type: {"radio", "multi"}
            The type of boxes that this control should hold. "radio" for single item select,
            "multi" for multi item select.

        """
        logger.debug("Adding %s group: %s", option_type, self.option.name)
        help_intro, help_items = self._get_multi_help_items(self.option.helptext)
        ctl = ttk.LabelFrame(self.frame,
                             text=self.option.title,
                             name="{}_labelframe".format(option_type))
        holder = AutoFillContainer(ctl, self.option_columns, self.option_columns)
        for choice in self.option.choices:
            ctl = ttk.Radiobutton if option_type == "radio" else MultiOption
            ctl = ctl(holder.subframe,
                      text=choice.replace("_", " ").title(),
                      value=choice,
                      variable=self.option.tk_var)
            if choice.lower() in help_items:
                self.helpset = True
                helptext = help_items[choice.lower()].capitalize()
                helptext = "{}\n\n - {}".format(
                    '. '.join(item.capitalize() for item in helptext.split('. ')),
                    help_intro)
                _get_tooltip(ctl, text=helptext, wraplength=600)
            ctl.pack(anchor=tk.W)
            logger.debug("Added %s option %s", option_type, choice)
        return holder.parent

    @staticmethod
    def _get_multi_help_items(helptext):
        """ Split the help text up, for formatted help text, into the individual options
        for multi/radio buttons.

        Parameters
        ----------
        helptext: str
            The raw help text for this cli. option

        Returns
        -------
        tuple (`str`, `dict`)
            The help text intro and a dictionary containing the help text split into separate
            entries for each option choice
        """
        logger.debug("raw help: %s", helptext)
        all_help = helptext.splitlines()
        intro = ""
        if any(line.startswith(" - ") for line in all_help):
            intro = all_help[0]
        retval = (intro, {re.sub(r'[^A-Za-z0-9\-]+', '',
                                 line.split()[1].lower()): " ".join(line.split()[1:])
                          for line in all_help if line.startswith(" - ")})
        logger.debug("help items: %s", retval)
        return retval

    def slider_control(self):
        """ A slider control with corresponding Entry box """
        logger.debug("Add slider control to Options Frame: (widget: '%s', dtype: %s, "
                     "rounding: %s, min_max: %s)", self.option.name, self.option.dtype,
                     self.option.rounding, self.option.min_max)
        validate = self.slider_check_int if self.option.dtype == int else self.slider_check_float
        vcmd = (self.frame.register(validate))
        tbox = ttk.Entry(self.frame,
                         width=8,
                         textvariable=self.option.tk_var,
                         justify=tk.RIGHT,
                         font=get_config().default_font,
                         validate="all",
                         validatecommand=(vcmd, "%P"))
        tbox.pack(padx=(0, 5), side=tk.RIGHT)
        cmd = partial(set_slider_rounding,
                      var=self.option.tk_var,
                      d_type=self.option.dtype,
                      round_to=self.option.rounding,
                      min_max=self.option.min_max)
        ctl = ttk.Scale(self.frame, variable=self.option.tk_var, command=cmd)
        _add_command(ctl.cget("command"), cmd)
        rc_menu = _get_contextmenu(tbox)
        rc_menu.cm_bind()
        ctl["from_"] = self.option.min_max[0]
        ctl["to"] = self.option.min_max[1]
        logger.debug("Added slider control to Options Frame: %s", self.option.name)
        return ctl

    @staticmethod
    def slider_check_int(value):
        """ Validate a slider's text entry box for integer values.

        Parameters
        ----------
        value: str
            The slider text entry value to validate
        """
        if value.isdigit() or value == "":
            return True
        return False

    @staticmethod
    def slider_check_float(value):
        """ Validate a slider's text entry box for float values.
        Parameters
        ----------
        value: str
            The slider text entry value to validate
        """
        if value:
            try:
                float(value)
            except ValueError:
                return False
        return True

    def control_to_optionsframe(self):
        """ Standard non-check buttons sit in the main options frame """
        logger.debug("Add control to Options Frame: (widget: '%s', control: %s, choices: %s)",
                     self.option.name, self.option.control, self.option.choices)
        if self.option.sysbrowser is not None:
            self.filebrowser = FileBrowser(self.option.name,
                                           self.option.tk_var,
                                           self.frame,
                                           self.option.sysbrowser)

        ctl = self.option.control(self.frame,
                                  textvariable=self.option.tk_var,
                                  font=get_config().default_font)
        rc_menu = _get_contextmenu(ctl)
        rc_menu.cm_bind()

        if self.option.choices:
            logger.debug("Adding combo choices: %s", self.option.choices)
            ctl["values"] = self.option.choices
            ctl["state"] = "readonly"
        logger.debug("Added control to Options Frame: %s", self.option.name)
        return ctl

    def _color_control(self):
        """ Clickable label holding the currently selected color """
        logger.debug("Add control to Options Frame: (widget: '%s', control: %s, choices: %s)",
                     self.option.name, self.option.control, self.option.choices)
        frame = ttk.Frame(self.frame)
        ctl = tk.Frame(frame,
                       bg=self.option.default,
                       bd=2,
                       cursor="hand1",
                       relief=tk.SUNKEN,
                       width=round(int(20 * get_config().scaling_factor)),
                       height=round(int(12 * get_config().scaling_factor)))
        ctl.bind("<Button-1>", lambda *e, c=ctl, t=self.option.title: self._ask_color(c, t))
        ctl.pack(side=tk.LEFT, anchor=tk.W)
        lbl = ttk.Label(frame, text=self.option.title, width=self.label_width, anchor=tk.W)
        lbl.pack(padx=2, pady=5, side=tk.RIGHT, anchor=tk.N)
        frame.pack(side=tk.LEFT, anchor=tk.W)
        if self.option.helptext is not None:
            _get_tooltip(lbl, text=self.option.helptext, wraplength=600)
        logger.debug("Added control to Options Frame: %s", self.option.name)
        return ctl

    def _ask_color(self, frame, title):
        """ Pop ask color dialog set to variable and change frame color """
        color = self.option.tk_var.get()
        chosen = colorchooser.askcolor(color=color, title="{} Color".format(title))[1]
        if chosen is None:
            return
        frame.config(bg=chosen)
        self.option.tk_var.set(chosen)

    def control_to_checkframe(self):
        """ Add check-buttons to the check-button frame """
        logger.debug("Add control checkframe: '%s'", self.option.name)
        chkframe = self.chkbtns.subframe
        ctl = self.option.control(chkframe,
                                  variable=self.option.tk_var,
                                  text=self.option.title,
                                  name=self.option.name)
        _get_tooltip(ctl, text=self.option.helptext, wraplength=600)
        ctl.pack(side=tk.TOP, anchor=tk.W)
        logger.debug("Added control checkframe: '%s'", self.option.name)
        return ctl


class FileBrowser():
    """ Add FileBrowser buttons to control and handle routing """
    def __init__(self, opt_name, tk_var, control_frame, sysbrowser_dict):
        logger.debug("Initializing: %s: (tk_var: %s, control_frame: %s, sysbrowser_dict: %s)",
                     self.__class__.__name__, tk_var, control_frame, sysbrowser_dict)
        self._opt_name = opt_name
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
                      load2="Select a file...",
                      picture="Select a folder of images...",
                      video="Select a video...",
                      model="Select a model folder...",
                      multi_load="Select one or more files...",
                      context="Select a file or folder...",
                      save_as="Select a save location...")
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
        frame = ttk.Frame(self.frame)
        frame.pack(side=tk.RIGHT, padx=(0, 5))

        for browser in self.browser:
            if browser == "save":
                lbl = "save_as"
            elif browser == "load" and self.filetypes == "video":
                lbl = self.filetypes
            elif browser == "load":
                lbl = "load2"
            elif browser == "folder" and (self._opt_name.startswith(("frames", "faces"))
                                          or "input" in self._opt_name):
                lbl = "picture"
            elif browser == "folder" and "model" in self._opt_name:
                lbl = "model"
            else:
                lbl = browser
            img = get_images().icons[lbl]
            action = getattr(self, "ask_" + browser)
            cmd = partial(action, filepath=self.tk_var, filetypes=self.filetypes)
            fileopn = ttk.Button(frame, image=img, command=cmd)
            _add_command(fileopn.cget("command"), cmd)
            fileopn.pack(padx=0, side=tk.RIGHT)
            _get_tooltip(fileopn, text=self.helptext[lbl], wraplength=600)
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
    def ask_multi_load(filepath, filetypes):
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
