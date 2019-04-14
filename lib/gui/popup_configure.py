#!/usr/bin python3
""" Configure Plugins popup of the Faceswap GUI """

from configparser import ConfigParser
import logging
import tkinter as tk

from tkinter import ttk

from .tooltip import Tooltip
from .utils import get_config, get_images, ContextMenu, set_slider_rounding

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
POPUP = dict()


def popup_config(config, root):
    """ Close any open popup and open requested popup """
    if POPUP:
        p_key = list(POPUP.keys())[0]
        logger.debug("Closing open popup: '%s'", p_key)
        POPUP[p_key].destroy()
        del POPUP[p_key]
    window = ConfigurePlugins(config, root)
    POPUP[config[0]] = window


class ConfigurePlugins(tk.Toplevel):
    """ Pop up for detailed graph/stats for selected session """
    def __init__(self, config, root):
        logger.debug("Initializing %s", self.__class__.__name__)
        super().__init__()
        name, self.config = config
        self.title("{} Plugins".format(name.title()))
        self.tk.call('wm', 'iconphoto', self._w, get_images().icons["favicon"])

        self.set_geometry(root)

        self.page_frame = ttk.Frame(self)
        self.page_frame.pack(fill=tk.BOTH, expand=True)

        self.plugin_info = dict()
        self.config_dict_gui = self.get_config()
        self.build()
        self.update()
        logger.debug("Initialized %s", self.__class__.__name__)

    def set_geometry(self, root):
        """ Set pop-up geometry """
        scaling_factor = get_config().scaling_factor
        pos_x = root.winfo_x() + 80
        pos_y = root.winfo_y() + 80
        width = int(720 * scaling_factor)
        height = int(400 * scaling_factor)
        logger.debug("Pop up Geometry: %sx%s, %s+%s", width, height, pos_x, pos_y)
        self.geometry("{}x{}+{}+{}".format(width, height, pos_x, pos_y))

    def get_config(self):
        """ Format config into useful format for GUI and pull default value if a value has not
            been supplied """
        logger.debug("Formatting Config for GUI")
        conf = dict()
        for section in self.config.config.sections():
            self.config.section = section
            category = section.split(".")[0]
            options = self.config.defaults[section]
            conf.setdefault(category, dict())[section] = options
            for key in options.keys():
                if key == "helptext":
                    self.plugin_info[section] = options[key]
                    continue
                options[key]["value"] = self.config.config_dict.get(key, options[key]["default"])
        logger.debug("Formatted Config for GUI: %s", conf)
        return conf

    def build(self):
        """ Build the config popup """
        logger.debug("Building plugin config popup")
        container = ttk.Notebook(self.page_frame)
        container.pack(fill=tk.BOTH, expand=True)
        categories = sorted(list(key for key in self.config_dict_gui.keys()))
        if "global" in categories:  # Move global to first item
            categories.insert(0, categories.pop(categories.index("global")))
        for category in categories:
            page = self.build_page(container, category)
            container.add(page, text=category.title())

        self.add_frame_separator()
        self.add_actions()
        logger.debug("Built plugin config popup")

    def build_page(self, container, category):
        """ Build a plugin config page """
        logger.debug("Building plugin config page: '%s'", category)
        plugins = sorted(list(key for key in self.config_dict_gui[category].keys()))
        if any(plugin != category for plugin in plugins):
            page = ttk.Notebook(container)
            page.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            for plugin in plugins:
                frame = ConfigFrame(page,
                                    self.config_dict_gui[category][plugin],
                                    self.plugin_info[plugin])
                title = plugin[plugin.rfind(".") + 1:]
                title = title.replace("_", " ").title()
                page.add(frame, text=title)
        else:
            page = ConfigFrame(container,
                               self.config_dict_gui[category][plugins[0]],
                               self.plugin_info[plugins[0]])

        logger.debug("Built plugin config page: '%s'", category)

        return page

    def add_frame_separator(self):
        """ Add a separator between top and bottom frames """
        logger.debug("Add frame seperator")
        sep = ttk.Frame(self.page_frame, height=2, relief=tk.RIDGE)
        sep.pack(fill=tk.X, pady=(5, 0), side=tk.BOTTOM)
        logger.debug("Added frame seperator")

    def add_actions(self):
        """ Add Action buttons """
        logger.debug("Add action buttons")
        frame = ttk.Frame(self.page_frame)
        frame.pack(fill=tk.BOTH, padx=5, pady=5, side=tk.BOTTOM)
        btn_cls = ttk.Button(frame, text="Cancel", width=10, command=self.destroy)
        btn_cls.pack(padx=2, side=tk.RIGHT)
        Tooltip(btn_cls, text="Close without saving", wraplength=720)
        btn_ok = ttk.Button(frame, text="OK", width=10, command=self.save_config)
        btn_ok.pack(padx=2, side=tk.RIGHT)
        Tooltip(btn_ok, text="Close and save config", wraplength=720)
        btn_rst = ttk.Button(frame, text="Reset", width=10, command=self.reset)
        btn_rst.pack(padx=2, side=tk.RIGHT)
        Tooltip(btn_rst, text="Reset all plugins to default values", wraplength=720)
        logger.debug("Added action buttons")

    def reset(self):
        """ Reset all config options to default """
        logger.debug("Resetting config")
        for section, items in self.config.defaults.items():
            logger.debug("Resetting section: '%s'", section)
            lookup = [section.split(".")[0], section] if "." in section else [section, section]
            for item, def_opt in items.items():
                if item == "helptext":
                    continue
                default = def_opt["default"]
                tk_var = self.config_dict_gui[lookup[0]][lookup[1]][item]["selected"]
                logger.debug("Resetting: '%s' to '%s'", item, default)
                tk_var.set(default)

    def save_config(self):
        """ Save the config file """
        logger.debug("Saving config")
        options = {sect: opts
                   for value in self.config_dict_gui.values()
                   for sect, opts in value.items()}

        new_config = ConfigParser(allow_no_value=True)
        for section, items in self.config.defaults.items():
            logger.debug("Adding section: '%s')", section)
            self.config.insert_config_section(section, items["helptext"], config=new_config)
            for item, def_opt in items.items():
                if item == "helptext":
                    continue
                new_opt = options[section][item]
                logger.debug("Adding option: (item: '%s', default: '%s' new: '%s'",
                             item, def_opt, new_opt)
                helptext = def_opt["helptext"]
                helptext = self.config.format_help(helptext, is_section=False)
                new_config.set(section, helptext)
                new_config.set(section, item, str(new_opt["selected"].get()))
        self.config.config = new_config
        self.config.save_config()
        print("Saved config: '{}'".format(self.config.configfile))
        self.destroy()
        logger.debug("Saved config")


class ConfigFrame(ttk.Frame):  # pylint: disable=too-many-ancestors
    """ Config Frame - Holds the Options for config """

    def __init__(self, parent, options, plugin_info):
        logger.debug("Initializing %s", self.__class__.__name__)
        ttk.Frame.__init__(self, parent)
        self.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.options = options
        self.plugin_info = plugin_info

        self.canvas = tk.Canvas(self, bd=0, highlightthickness=0)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.optsframe = ttk.Frame(self.canvas)
        self.optscanvas = self.canvas.create_window((0, 0), window=self.optsframe, anchor=tk.NW)

        self.build_frame()
        logger.debug("Initialized %s", self.__class__.__name__)

    def build_frame(self):
        """ Build the options frame for this command """
        logger.debug("Add Config Frame")
        self.add_scrollbar()
        self.canvas.bind("<Configure>", self.resize_frame)

        self.add_info()
        for key, val in self.options.items():
            if key == "helptext":
                continue
            OptionControl(key, val, self.optsframe)
        logger.debug("Added Config Frame")

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
        info = ttk.Label(info_frame, text=self.plugin_info)
        info.pack(padx=5, pady=5, fill=tk.X, expand=True)


class OptionControl():
    """ Build the correct control for the option parsed and place it on the
    frame """

    def __init__(self, title, values, option_frame):
        logger.debug("Initializing %s", self.__class__.__name__)
        self.title = title
        self.values = values
        self.option_frame = option_frame

        self.control = self.set_control()
        self.control_frame = self.set_control_frame()
        self.tk_var = self.set_tk_var()

        self.build_full_control()
        logger.debug("Initialized %s", self.__class__.__name__)

    @property
    def helptext(self):
        """ Format the help text for tooltips """
        logger.debug("Format control help: '%s'", self.title)
        helptext = self.values.get("helptext", "")
        helptext = helptext.replace("\n\t", "\n  - ").replace("%%", "%")
        helptext = self.title + " - " + helptext
        logger.debug("Formatted control help: (title: '%s', help: '%s'", self.title, helptext)
        return helptext

    def set_control(self):
        """ Set the correct control type for this option """
        dtype = self.values["type"]
        choices = self.values["choices"]
        if choices:
            control = ttk.Combobox
        elif dtype == bool:
            control = ttk.Checkbutton
        elif dtype in (int, float):
            control = ttk.Scale
        else:
            control = ttk.Entry
        logger.debug("Setting control '%s' to %s", self.title, control)
        return control

    def set_control_frame(self):
        """ Frame to hold control and it's label """
        logger.debug("Build config control frame")
        frame = ttk.Frame(self.option_frame)
        frame.pack(fill=tk.X, expand=True)
        logger.debug("Built confog control frame")
        return frame

    def set_tk_var(self):
        """ Correct variable type for control """
        logger.debug("Setting config variable type: '%s'", self.title)
        var = tk.BooleanVar if self.control == ttk.Checkbutton else tk.StringVar
        var = var(self.control_frame)
        logger.debug("Set config variable type: ('%s': %s", self.title, type(var))
        return var

    def build_full_control(self):
        """ Build the correct control type for the option passed through """
        logger.debug("Build confog option control")
        self.build_control_label()
        self.build_one_control()
        self.values["selected"] = self.tk_var
        logger.debug("Built option control")

    def build_control_label(self):
        """ Label for control """
        logger.debug("Build config control label: '%s'", self.title)
        title = self.title.replace("_", " ").title()
        lbl = ttk.Label(self.control_frame, text=title, width=20, anchor=tk.W)
        lbl.pack(padx=5, pady=5, side=tk.LEFT, anchor=tk.N)
        logger.debug("Built config control label: '%s'", self.title)

    def build_one_control(self):
        """ Build and place the option controls """
        logger.debug("Build control: (title: '%s', values: %s)", self.title, self.values)
        self.tk_var.set(self.values["value"])

        if self.control == ttk.Scale:
            self.slider_control()
        else:
            self.control_to_optionsframe()
        logger.debug("Built control: '%s'", self.title)

    def slider_control(self):
        """ A slider control with corresponding Entry box """
        logger.debug("Add slider control to Config Options Frame: %s", self.control)
        d_type = self.values["type"]
        rnd = self.values["rounding"]
        min_max = self.values["min_max"]

        tbox = ttk.Entry(self.control_frame, width=8, textvariable=self.tk_var, justify=tk.RIGHT)
        tbox.pack(padx=(0, 5), side=tk.RIGHT)
        ctl = self.control(
            self.control_frame,
            variable=self.tk_var,
            command=lambda val, var=self.tk_var, dt=d_type, rn=rnd, mm=min_max:
            set_slider_rounding(val, var, dt, rn, mm))
        ctl.pack(padx=5, pady=5, fill=tk.X, expand=True)
        rc_menu = ContextMenu(ctl)
        rc_menu.cm_bind()
        ctl["from_"] = min_max[0]
        ctl["to"] = min_max[1]

        Tooltip(ctl, text=self.helptext, wraplength=720)
        Tooltip(tbox, text=self.helptext, wraplength=720)
        logger.debug("Added slider control to Options Frame: %s", self.control)

    def control_to_optionsframe(self):
        """ Standard non-check buttons sit in the main options frame """
        logger.debug("Add control to Options Frame: %s", self.control)
        choices = self.values["choices"]
        if self.control == ttk.Checkbutton:
            ctl = self.control(self.control_frame, variable=self.tk_var, text=None)
        else:
            ctl = self.control(self.control_frame, textvariable=self.tk_var)
        ctl.pack(padx=5, pady=5, fill=tk.X, expand=True)
        rc_menu = ContextMenu(ctl)
        rc_menu.cm_bind()
        if choices:
            logger.debug("Adding combo choices: %s", choices)
            ctl["values"] = [choice for choice in choices]
        Tooltip(ctl, text=self.helptext, wraplength=720)
        logger.debug("Added control to Options Frame: %s", self.control)
