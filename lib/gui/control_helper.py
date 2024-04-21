#!/usr/bin/env python3
""" Helper functions and classes for GUI controls """
import gettext
import logging
import re

import tkinter as tk
import typing as T
from tkinter import colorchooser, ttk
from itertools import zip_longest
from functools import partial

from _tkinter import Tcl_Obj, TclError

from .custom_widgets import ContextMenu, MultiOption, ToggledFrame, Tooltip
from .utils import FileHandler, get_config, get_images

logger = logging.getLogger(__name__)

# LOCALES
_LANG = gettext.translation("gui.tooltips", localedir="locales", fallback=True)
_ = _LANG.gettext

# We store Tooltips, ContextMenus and Commands globally when they are created
# Because we need to add them back to newly cloned widgets (they are not easily accessible from
# original config or are prone to getting destroyed when the original widget is destroyed)
_RECREATE_OBJECTS: dict[str, dict[str, T.Any]] = {"tooltips": {},
                                                  "commands": {},
                                                  "contextmenus": {}}


def _get_tooltip(widget, text=None, text_variable=None):
    """ Store the tooltip layout and widget id in _TOOLTIPS and return a tooltip.

    Auto adjust tooltip width based on amount of text.

    """
    _RECREATE_OBJECTS["tooltips"][str(widget)] = {"text": text,
                                                  "text_variable": text_variable}
    logger.debug("Adding to tooltips dict: (widget: %s. text: '%s')", widget, text)

    wrap_length = 400
    if text is not None:
        while True:
            if len(text) < wrap_length * 5:
                break
            if wrap_length > 800:
                break
            wrap_length = int(wrap_length * 1.10)

    return Tooltip(widget, text=text, text_variable=text_variable, wrap_length=wrap_length)


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
    round_to: int or list
        If :attr:`d_type` is :class:`float` then this is the decimal place rounding for
        :attr:`var`. If :attr:`d_type` is :class:`int` then this is the number of steps between
        each increment for :attr:`var`. If a list is provided, then this must be a list of
        discreet values that are of the correct :attr:`d_type`.
    min_max: tuple (`int`, `int`)
        The (``min``, ``max``) values that this slider accepts
    """
    if isinstance(round_to, list):
        # Lock to nearest item
        var.set(min(round_to, key=lambda x: abs(x-float(value))))
    elif d_type == float:
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
        self._options = {"title": title,
                         "subgroup": subgroup,
                         "group": group,
                         "default": default,
                         "initial_value": initial_value,
                         "choices": choices,
                         "is_radio": is_radio,
                         "is_multi_option": is_multi_option,
                         "rounding": rounding,
                         "min_max": min_max,
                         "helptext": helptext}
        self.control = self.get_control()
        self.tk_var = self.get_tk_var(initial_value, track_modified)
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
            control = tk.Entry
        logger.debug("Setting control '%s' to %s", self.title, control)
        return control

    def get_tk_var(self, initial_value, track_modified):
        """ Correct variable type for control """
        if self.dtype == bool:
            var = tk.BooleanVar()
        elif self.dtype == int:
            var = tk.IntVar()
        elif self.dtype == float:
            var = tk.DoubleVar()
        else:
            var = tk.StringVar()
        if initial_value is not None:
            var.set(initial_value)
        logger.debug("Setting tk variable: (name: '%s', dtype: %s, tk_var: %s, initial_value: %s)",
                     self.name, self.dtype, var, initial_value)
        if track_modified and self._command is not None:
            logger.debug("Tracking variable modification: %s", self.name)
            var.trace("w",
                      lambda name, index, mode, cmd=self._command: self._modified_callback(cmd))

        if track_modified and self._command == "train" and self.title == "Model Dir":
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
        if config.tk_vars.running_task.get():
            logger.debug("Task running. Not updating session")
            return
        folder = var.get()
        logger.debug("Setting analysis model folder callback: '%s'", folder)
        get_config().tk_vars.analysis_folder.set(folder)


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
    style: str, optional
        The name of the style to use for the control panel. Styles are configured when TkInter
        initializes. The style name is the common prefix prior to the widget name. Default:
        ``None`` (use the OS style)
    blank_nones: bool, optional
        How the control panel should handle None values. If set to True then None values will be
        converted to empty strings. Default: False
    scrollbar: bool, optional
        ``True`` if a scrollbar should be added to the control panel, otherwise ``False``.
        Default: ``True``
    """

    def __init__(self, parent, options,  # pylint:disable=too-many-arguments
                 label_width=20, columns=1, max_columns=4, option_columns=4, header_text=None,
                 style=None, blank_nones=True, scrollbar=True):
        logger.debug("Initializing %s: (parent: '%s', options: %s, label_width: %s, columns: %s, "
                     "max_columns: %s, option_columns: %s, header_text: %s, style: %s, "
                     "blank_nones: %s, scrollbar: %s)",
                     self.__class__.__name__, parent, options, label_width, columns, max_columns,
                     option_columns, header_text, style, blank_nones, scrollbar)
        self._style = "" if style is None else f"{style}."
        super().__init__(parent, style=f"{self._style}.Group.TFrame")

        self.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.options = options
        self.controls = []
        self.label_width = label_width
        self.columns = columns
        self.max_columns = max_columns
        self.option_columns = option_columns

        self.header_text = header_text
        self._theme = get_config().user_theme["group_panel"]
        if self._style.startswith("SPanel"):
            self._theme = {**self._theme, **get_config().user_theme["group_settings"]}

        self.group_frames = {}
        self._sub_group_frames = {}

        canvas_kwargs = {"bd": 0, "highlightthickness": 0, "bg": self._theme["panel_background"]}

        self._canvas = tk.Canvas(self, **canvas_kwargs)
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
        style = f"{self._style}Holder."
        mainframe = ttk.Frame(self._canvas, style=f"{style}TFrame")
        if self.header_text is not None:
            self.add_info(mainframe)
        optsframe = ttk.Frame(mainframe, name="opts_frame", style=f"{style}TFrame")
        optsframe.pack(expand=True, fill=tk.BOTH)
        holder = AutoFillContainer(optsframe, self.columns, self.max_columns, style=style)
        logger.debug("Opts frames: '%s'", holder)
        return mainframe, holder

    def add_info(self, frame):
        """ Plugin information """
        info_frame = ttk.Frame(frame, style=f"{self._style}InfoHeader.TFrame")
        info_frame.pack(fill=tk.X, side=tk.TOP, expand=True, padx=10, pady=(10, 0))
        label_frame = ttk.Frame(info_frame, style=f"{self._style}InfoHeader.TFrame")
        label_frame.pack(padx=5, pady=5, fill=tk.X, expand=True)
        for idx, line in enumerate(self.header_text.splitlines()):
            if not line:
                continue
            style = f"{self._style}InfoHeader" if idx == 0 else f"{self._style}InfoBody"
            info = ttk.Label(label_frame, text=line, style=f"{style}.TLabel", anchor=tk.W)
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
                                 style=self._style,
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
        """ Return a group frame.

        If a group frame has already been created for the given group, then it will be returned,
        otherwise it will be created and returned.

        Parameters
        ----------
        group: str
            The name of the group to obtain the group frame for

        Returns
        -------
        :class:`ttk.Frame` or :class:`ToggledFrame`
            If this is a 'master' group frame then returns a standard frame. If this is any
            other group, then will return the ToggledFrame for that group
        """
        group = group.lower()

        if self.group_frames.get(group, None) is None:
            logger.debug("Creating new group frame for: %s", group)
            is_master = group == "_master"
            opts_frame = self.optsframe.subframe
            if is_master:
                group_frame = ttk.Frame(opts_frame, style=f"{self._style}.Group.TFrame")
                retval = group_frame
            else:
                group_frame = ToggledFrame(opts_frame, text=group.title(), theme=self._style)
                retval = group_frame.sub_frame

            group_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5, anchor=tk.NW)

            self.group_frames[group] = {"frame": retval,
                                        "chkbtns": self.checkbuttons_frame(retval)}
        group_frame = self.group_frames[group]
        return group_frame

    def add_scrollbar(self):
        """ Add a scrollbar to the options frame """
        logger.debug("Add Config Scrollbar")
        scrollbar = ttk.Scrollbar(self,
                                  command=self._canvas.yview,
                                  style=f"{self._style}Vertical.TScrollbar")
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self._canvas.config(yscrollcommand=scrollbar.set)
        self.mainframe.bind("<Configure>", self.update_scrollbar)
        logger.debug("Added Config Scrollbar")

    def update_scrollbar(self, event):  # pylint:disable=unused-argument
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
        chk_frame = ttk.Frame(frame, name="chkbuttons", style=f"{self._style}Group.TFrame")
        holder = AutoFillContainer(chk_frame,
                                   self.option_columns,
                                   self.option_columns,
                                   style=f"{self._style}Group.")
        logger.debug("Added Options CheckButtons Frame")
        return holder

    def _get_subgroup_frame(self, parent, subgroup):
        if subgroup is None:
            return subgroup
        if subgroup not in self._sub_group_frames:
            sub_frame = ttk.Frame(parent, style=f"{self._style}Group.TFrame")
            self._sub_group_frames[subgroup] = AutoFillContainer(sub_frame,
                                                                 self.option_columns,
                                                                 self.option_columns,
                                                                 style=f"{self._style}Group.")
            sub_frame.pack(anchor=tk.W, expand=True, fill=tk.X)
            logger.debug("Added Subgroup Frame: %s", subgroup)
        return self._sub_group_frames[subgroup]


class AutoFillContainer():
    """ A container object that auto-fills columns.

    Parameters
    ----------
    parent: :class:`ttk.Frame`
        The parent widget that holds this container
    initial_columns: int
        The initial number of columns that this container should display
    max_columns: int
        The maximum number of column that this container is permitted to display
    style: str, optional
        The name of the style to use for the control panel. Styles are configured when TkInter
        initializes. The style name is the common prefix prior to the widget name. Default:
        empty string (use the OS style)
    """
    def __init__(self, parent, initial_columns, max_columns, style=""):
        logger.debug("Initializing: %s: (parent: %s, initial_columns: %s, max_columns: %s)",
                     self.__class__.__name__, parent, initial_columns, max_columns)
        self.max_columns = max_columns
        self.columns = initial_columns
        self.parent = parent
        self._style = style
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
            name = f"af_subframe_{idx}"
            subframe = ttk.Frame(self.parent, name=name, style=f"{self._style}TFrame")
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
        """ Compile all children recursively in correct order if not already compiled and add
        to :attr:`_widget_config` """
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
                                "custom_kwargs": self._custom_kwargs(child)}
                               for idx, child in enumerate(children)]
        logger.debug("Compiled AutoFillContainer children: %s", self._widget_config)

    @classmethod
    def _custom_kwargs(cls, widget):
        """ For custom widgets some custom arguments need to be passed from the old widget to the
        newly created widget.

        Parameters
        ----------
        widget: tkinter widget
            The widget to be checked for custom keyword arguments

        Returns
        -------
        dict
            The custom keyword arguments required for recreating the given widget
        """
        retval = {}
        if widget.__class__.__name__ == "MultiOption":
            retval = {"value": widget._value,  # pylint:disable=protected-access
                      "variable": widget._master_variable}  # pylint:disable=protected-access
        elif widget.__class__.__name__ == "ToggledFrame":
            # Toggled Frames need to have their variable tracked
            retval = {"text": widget._text,  # pylint:disable=protected-access
                      "toggle_var": widget._toggle_var}  # pylint:disable=protected-access
        return retval

    def get_all_children_config(self, widget, child_list):
        """ Return all children, recursively, of given widget.

        Parameters
        ----------
        widget: tkinter widget
            The widget to recursively obtain the configurations of each child
        child_list: list
            The list of child configurations already collected

        Returns
        -------
        list
            The list of configurations for all recursive children of the given widget
         """
        unpack = set()
        for child in widget.winfo_children():
            # Hidden Toggle Frame boxes need to be mapped
            if child.winfo_ismapped() or "toggledframe_subframe" in str(child):
                not_mapped = not child.winfo_ismapped()
                # ToggleFrame is a custom widget that creates it's own children and handles
                # bindings on the headers, to auto-hide the contents. To ensure that all child
                # information (specifically pack information) can be collected, we need to pack
                # any hidden sub-frames. These are then hidden again once collected.
                if not_mapped and (child.winfo_name() == "toggledframe_subframe" or
                                   child.winfo_name() == "chkbuttons"):
                    child.pack(fill=tk.X, expand=True)
                    child.update_idletasks()  # Updates the packing info of children
                    unpack.add(child)

                if child.winfo_name().startswith("toggledframe_header"):
                    # Headers should be entirely handled by parent widget
                    continue

                child_list.append({
                    "class": child.__class__,
                    "id": str(child),
                    "tooltip": _RECREATE_OBJECTS["tooltips"].get(str(child), None),
                    "rc_menu": _RECREATE_OBJECTS["contextmenus"].get(str(child), None),
                    "pack_info": self.pack_config_cleaner(child),
                    "name": child.winfo_name(),
                    "config": self.config_cleaner(child),
                    "parent": child.winfo_parent(),
                    "custom_kwargs": self._custom_kwargs(child)})
            self.get_all_children_config(child, child_list)

        # Re-hide any toggle frames that were expanded
        for hide in unpack:
            hide.pack_forget()
            hide.update_idletasks()
        return child_list

    @staticmethod
    def config_cleaner(widget):
        """ Some options don't like to be copied, so this returns a cleaned
            configuration from a widget
            We use config() instead of configure() because some items (ttk Scale) do
            not populate configure()"""
        new_config = {}
        for key in widget.config():
            if key == "class":
                continue
            val = widget.cget(key)
            # Some keys default to "" but tkinter doesn't like to set config to this value
            # so skip them to use default value.
            if key in ("anchor", "justify", "compound") and val == "":
                continue
            # Following keys cannot be defined after widget is created:
            if key in ("colormap", "container", "visual"):
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
        """ Recursively pass through the list of widgets creating clones and packing all
        children.

        Widgets cannot be given a new parent so we need to clone them and then pack the
        new widgets.

        Parameters
        ----------
        widget_dicts: list
            List of dictionaries, in appearance order, of widget information for cloning widgets
        old_childen: list, optional
            Used for recursion. Leave at ``None``
        new_childen: list, optional
            Used for recursion. Leave at ``None``
        """
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

            # Handle ToggledFrame sub-frames. If the parent is not set to expanded, then we need to
            # hide the sub-frame
            if clone.winfo_name() == "toggledframe_subframe":
                toggle_frame = clone.nametowidget(clone.winfo_parent())
                if not toggle_frame.is_expanded:
                    logger.debug("Hiding minimized toggle box: %s", clone)
                    clone.pack_forget()

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
    style: str
        The name of the style to use for the control panel. Styles are configured when TkInter
        initializes. The style name is the common prefix prior to the widget name. Provide an empty
        string to use the OS style
    blank_nones: bool
        Sets selected values to an empty string rather than None if this is true.
    """
    def __init__(self, parent, option, option_columns,  # pylint:disable=too-many-arguments
                 label_width, checkbuttons_frame, style, blank_nones):
        logger.debug("Initializing %s: (parent: %s, option: %s, option_columns: %s, "
                     "label_width: %s, checkbuttons_frame: %s, style: %s, blank_nones: %s)",
                     self.__class__.__name__, parent, option, option_columns, label_width,
                     checkbuttons_frame, style, blank_nones)

        self.option = option
        self.option_columns = option_columns
        self.helpset = False
        self.label_width = label_width
        self.filebrowser = None
        # Default to Control Panel Style
        self._style = style = style if style else "CPanel."
        self._theme = get_config().user_theme["group_panel"]
        if self._style.startswith("SPanel"):
            self._theme = {**self._theme, **get_config().user_theme["group_settings"]}

        self.frame = self.control_frame(parent)
        self.chkbtns = checkbuttons_frame

        self.set_tk_var(blank_nones)
        self.build_control()
        logger.debug("Initialized: %s", self.__class__.__name__)

    # Frame, control type and variable
    def control_frame(self, parent):
        """ Frame to hold control and it's label """
        logger.debug("Build control frame")
        frame = ttk.Frame(parent,
                          name=f"fr_{self.option.name}",
                          style=f"{self._style}Group.TFrame")
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
        lbl = ttk.Label(self.frame,
                        text=self.option.title,
                        width=self.label_width,
                        anchor=tk.W,
                        style=f"{self._style}Group.TLabel")
        lbl.pack(padx=5, pady=5, side=tk.LEFT, anchor=tk.N)
        if self.option.helptext is not None:
            _get_tooltip(lbl, text=self.option.helptext)
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
                tooltip_kwargs = {"text": self.option.helptext}
                if self.option.sysbrowser is not None:
                    tooltip_kwargs["text_variable"] = self.option.tk_var
                _get_tooltip(ctl, **tooltip_kwargs)

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
                             name=f"{option_type}_labelframe",
                             style=f"{self._style}Group.TLabelframe")
        holder = AutoFillContainer(ctl,
                                   self.option_columns,
                                   self.option_columns,
                                   style=f"{self._style}Group.")
        for choice in self.option.choices:
            if option_type == "radio":
                ctl = ttk.Radiobutton
                style = f"{self._style}Group.TRadiobutton"
            else:
                ctl = MultiOption
                style = f"{self._style}Group.TCheckbutton"

            ctl = ctl(holder.subframe,
                      text=choice.replace("_", " ").title(),
                      value=choice,
                      variable=self.option.tk_var,
                      style=style)
            if choice.lower() in help_items:
                self.helpset = True
                helptext = help_items[choice.lower()]
                helptext = f"{helptext}\n\n - {help_intro}"
                _get_tooltip(ctl, text=helptext)
            ctl.pack(anchor=tk.W, fill=tk.X)
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
        retval = (intro,
                  {re.sub(r"[^\w\-\_]+", "",
                          line.split()[1].lower()): " ".join(line.replace("_", " ").split()[1:])
                   for line in all_help if line.startswith(" - ")})
        logger.debug("help items: %s", retval)
        return retval

    def slider_control(self):
        """ A slider control with corresponding Entry box """
        logger.debug("Add slider control to Options Frame: (widget: '%s', dtype: %s, "
                     "rounding: %s, min_max: %s)", self.option.name, self.option.dtype,
                     self.option.rounding, self.option.min_max)
        validate = self.slider_check_int if self.option.dtype == int else self.slider_check_float
        vcmd = self.frame.register(validate)
        tbox = tk.Entry(self.frame,
                        width=8,
                        textvariable=self.option.tk_var,
                        justify=tk.RIGHT,
                        font=get_config().default_font,
                        validate="all",
                        validatecommand=(vcmd, "%P"),
                        bg=self._theme["input_color"],
                        fg=self._theme["input_font"],
                        highlightbackground=self._theme["input_font"],
                        highlightthickness=1,
                        bd=0)
        tbox.pack(padx=(0, 5), side=tk.RIGHT)
        cmd = partial(set_slider_rounding,
                      var=self.option.tk_var,
                      d_type=self.option.dtype,
                      round_to=self.option.rounding,
                      min_max=self.option.min_max)
        ctl = ttk.Scale(self.frame,
                        variable=self.option.tk_var,
                        command=cmd,
                        style=f"{self._style}Horizontal.TScale")
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
                                           self.option.sysbrowser,
                                           self._style)

        if self.option.control == tk.Entry:
            ctl = self.option.control(self.frame,
                                      textvariable=self.option.tk_var,
                                      font=get_config().default_font,
                                      bg=self._theme["input_color"],
                                      fg=self._theme["input_font"],
                                      highlightbackground=self._theme["input_font"],
                                      highlightthickness=1,
                                      bd=0)
        else:  # Combobox
            ctl = self.option.control(self.frame,
                                      textvariable=self.option.tk_var,
                                      font=get_config().default_font,
                                      state="readonly",
                                      style=f"{self._style}TCombobox")

            # Style for combo list boxes needs to be set directly on widget as no style parameter
            cmd = f"[ttk::combobox::PopdownWindow {ctl}].f.l configure -"
            ctl.tk.eval(f"{cmd}foreground {self._theme['input_font']}")
            ctl.tk.eval(f"{cmd}background {self._theme['input_color']}")
            ctl.tk.eval(f"{cmd}selectforeground {self._theme['control_active']}")
            ctl.tk.eval(f"{cmd}selectbackground {self._theme['control_disabled']}")

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
        frame = ttk.Frame(self.frame, style=f"{self._style}Group.TFrame")
        lbl = ttk.Label(frame,
                        text=self.option.title,
                        width=self.label_width,
                        anchor=tk.W,
                        style=f"{self._style}Group.TLabel")
        ctl = tk.Frame(frame,
                       bg=self.option.tk_var.get(),
                       bd=2,
                       cursor="hand2",
                       relief=tk.SUNKEN,
                       width=round(int(20 * get_config().scaling_factor)),
                       height=round(int(14 * get_config().scaling_factor)))
        ctl.bind("<Button-1>", lambda *e, c=ctl, t=self.option.title: self._ask_color(c, t))
        lbl.pack(side=tk.LEFT, anchor=tk.N)
        ctl.pack(side=tk.RIGHT, anchor=tk.W)
        frame.pack(padx=5, side=tk.LEFT, anchor=tk.W)
        if self.option.helptext is not None:
            _get_tooltip(frame, text=self.option.helptext)
        # Callback to set the color chooser background on an update (e.g. reset)
        self.option.tk_var.trace("w", lambda *e: ctl.config(bg=self.option.tk_var.get()))
        logger.debug("Added control to Options Frame: %s", self.option.name)
        return ctl

    def _ask_color(self, frame, title):
        """ Pop ask color dialog set to variable and change frame color """
        color = self.option.tk_var.get()
        chosen = colorchooser.askcolor(parent=frame, color=color, title=f"{title} Color")[1]
        if chosen is None:
            return
        self.option.tk_var.set(chosen)

    def control_to_checkframe(self):
        """ Add check-buttons to the check-button frame """
        logger.debug("Add control checkframe: '%s'", self.option.name)
        chkframe = self.chkbtns.subframe
        ctl = self.option.control(chkframe,
                                  variable=self.option.tk_var,
                                  text=self.option.title,
                                  name=self.option.name,
                                  style=f"{self._style}Group.TCheckbutton")
        _get_tooltip(ctl, text=self.option.helptext)
        ctl.pack(side=tk.TOP, anchor=tk.W, fill=tk.X)
        logger.debug("Added control checkframe: '%s'", self.option.name)
        return ctl


class FileBrowser():
    """ Add FileBrowser buttons to control and handle routing """
    def __init__(self, opt_name, tk_var, control_frame, sysbrowser_dict, style):
        logger.debug("Initializing: %s: (tk_var: %s, control_frame: %s, sysbrowser_dict: %s, "
                     "style: %s)", self.__class__.__name__, tk_var, control_frame,
                     sysbrowser_dict, style)
        self._opt_name = opt_name
        self.tk_var = tk_var
        self.frame = control_frame
        self._style = style
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
        retval = {"folder": _("Select a folder..."),
                  "load": _("Select a file..."),
                  "load2": _("Select a file..."),
                  "picture": _("Select a folder of images..."),
                  "video": _("Select a video..."),
                  "model": _("Select a model folder..."),
                  "multi_load": _("Select one or more files..."),
                  "context": _("Select a file or folder..."),
                  "save_as": _("Select a save location...")}
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
        frame = ttk.Frame(self.frame, style=f"{self._style}Group.TFrame")
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
            fileopn = tk.Button(frame,
                                image=img,
                                command=cmd,
                                relief=tk.SOLID,
                                bd=1,
                                bg=get_config().user_theme["group_panel"]["button_background"],
                                cursor="hand2")
            _add_command(fileopn.cget("command"), cmd)
            fileopn.pack(padx=1, side=tk.RIGHT)
            _get_tooltip(fileopn, text=self.helptext[lbl])
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
        dirname = FileHandler("dir", filetypes).return_file
        if dirname:
            logger.debug(dirname)
            filepath.set(dirname)

    @staticmethod
    def ask_load(filepath, filetypes):
        """ Pop-up to get path to a file """
        filename = FileHandler("filename", filetypes).return_file
        if filename:
            logger.debug(filename)
            filepath.set(filename)

    @staticmethod
    def ask_multi_load(filepath, filetypes):
        """ Pop-up to get path to a file """
        filenames = FileHandler("filename_multi", filetypes).return_file
        if filenames:
            final_names = " ".join(f"\"{fname}\"" for fname in filenames)
            logger.debug(final_names)
            filepath.set(final_names)

    @staticmethod
    def ask_save(filepath, filetypes=None):
        """ Pop-up to get path to save a new file """
        filename = FileHandler("save_filename", filetypes).return_file
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
                               variable=selected_variable).return_file
        if filename:
            logger.debug(filename)
            filepath.set(filename)
