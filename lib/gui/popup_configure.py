#!/usr/bin python3
""" Configure Plugins popup of the Faceswap GUI """

from collections import OrderedDict
from configparser import ConfigParser
import logging
import tkinter as tk

from tkinter import ttk

from .control_helper import ControlPanel, ControlPanelOption
from .custom_widgets import Tooltip
from .utils import get_config, get_images

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
        self._name, self.config = config
        self.title("{} Plugins".format(self._name.title()))
        self.tk.call('wm', 'iconphoto', self._w, get_images().icons["favicon"])

        self._root = root
        self.set_geometry()

        self.page_frame = ttk.Frame(self)
        self.page_frame.pack(fill=tk.BOTH, expand=True)

        self.plugin_info = dict()
        self.config_cpanel_dict = self.get_config()
        self.build()
        self.update()
        logger.debug("Initialized %s", self.__class__.__name__)

    def set_geometry(self):
        """ Set pop-up geometry """
        scaling_factor = get_config().scaling_factor
        pos_x = self._root.winfo_x() + 80
        pos_y = self._root.winfo_y() + 80
        width = int(600 * scaling_factor)
        height = int(400 * scaling_factor)
        logger.debug("Pop up Geometry: %sx%s, %s+%s", width, height, pos_x, pos_y)
        self.geometry("{}x{}+{}+{}".format(width, height, pos_x, pos_y))

    def get_config(self):
        """ Format config into a dict of ControlPanelOptions """
        logger.debug("Formatting Config for GUI")
        conf = dict()
        for section in self.config.config.sections():
            self.config.section = section
            category = section.split(".")[0]
            options = self.config.defaults[section]
            section = section.split(".")[-1]
            conf.setdefault(category, dict())[section] = OrderedDict()
            for key, val in options.items():
                if key == "helptext":
                    self.plugin_info[section] = val
                    continue
                conf[category][section][key] = ControlPanelOption(
                    title=key,
                    dtype=val["type"],
                    group=val["group"],
                    default=val["default"],
                    initial_value=self.config.config_dict.get(key, val["default"]),
                    choices=val["choices"],
                    is_radio=val["gui_radio"],
                    rounding=val["rounding"],
                    min_max=val["min_max"],
                    helptext=val["helptext"])
        logger.debug("Formatted Config for GUI: %s", conf)
        return conf

    def build(self):
        """ Build the config popup """
        logger.debug("Building plugin config popup")
        container = ttk.Notebook(self.page_frame)
        container.pack(fill=tk.BOTH, expand=True)
        categories = sorted(list(self.config_cpanel_dict.keys()))
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
        plugins = sorted(list(key for key in self.config_cpanel_dict[category].keys()))
        panel_kwargs = dict(columns=2, max_columns=2, option_columns=2, blank_nones=False)
        if any(plugin != category for plugin in plugins):
            page = ttk.Notebook(container)
            page.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            for plugin in plugins:
                cp_options = list(self.config_cpanel_dict[category][plugin].values())
                frame = ControlPanel(page,
                                     cp_options,
                                     header_text=self.plugin_info[plugin],
                                     **panel_kwargs)
                title = plugin[plugin.rfind(".") + 1:]
                title = title.replace("_", " ").title()
                page.add(frame, text=title)
        else:
            cp_options = list(self.config_cpanel_dict[category][plugins[0]].values())
            page = ControlPanel(container,
                                cp_options,
                                header_text=self.plugin_info[plugins[0]],
                                **panel_kwargs)

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
            lookup = [section.split(".")[0], section.split(".")[-1]]
            for item, def_opt in items.items():
                if item == "helptext":
                    continue
                default = def_opt["default"]
                logger.debug("Resetting: '%s' to '%s'", item, default)
                self.config_cpanel_dict[lookup[0]][lookup[1]][item].set(default)

    def save_config(self):
        """ Save the config file """
        logger.debug("Saving config")
        options = {".".join((key, sect)) if sect != key else key: opts
                   for key, value in self.config_cpanel_dict.items()
                   for sect, opts in value.items()}
        new_config = ConfigParser(allow_no_value=True)
        for section, items in self.config.defaults.items():
            logger.debug("Adding section: '%s')", section)
            self.config.insert_config_section(section, items["helptext"], config=new_config)
            for item, def_opt in items.items():
                if item == "helptext":
                    continue
                new_opt = options[section][item].get()
                logger.debug("Adding option: (item: '%s', default: %s new: '%s'",
                             item, def_opt, new_opt)
                helptext = def_opt["helptext"]
                helptext = self.config.format_help(helptext, is_section=False)
                new_config.set(section, helptext)
                new_config.set(section, item, str(new_opt))
        self.config.config = new_config
        self.config.save_config()
        logger.info("Saved config: '%s'", self.config.configfile)
        self.destroy()

        running_task = get_config().tk_vars["runningtask"].get()
        if self._name.lower() == "gui" and not running_task:
            self._root.rebuild()
        elif self._name.lower() == "gui" and running_task:
            logger.info("Can't redraw GUI whilst a task is running. GUI Settings will be applied "
                        "at the next restart.")
        logger.debug("Saved config")
