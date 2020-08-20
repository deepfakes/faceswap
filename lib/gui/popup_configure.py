#!/usr/bin python3
""" The pop-up window of the Faceswap GUI for the setting of configuration options. """

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


def popup_config(name, configuration):
    """ Open the settings for the requested configuration file and close any already active
    pop-ups.

    Parameters
    ----------
    name: str
        The name of the configuration file. Used for the pop-up title bar.
    configuration: :class:`~lib.config.FaceswapConfig`
        The configuration options for the requested pop-up window
    """
    logger.debug("name: %s, configuration: %s", name, configuration)
    if POPUP:
        p_key = list(POPUP.keys())[0]
        logger.debug("Closing open popup: '%s'", p_key)
        POPUP[p_key].destroy()
        del POPUP[p_key]
    window = _ConfigurePlugins(name, configuration)
    POPUP[name] = window
    logger.debug("Current pop-up: %s", POPUP)


class _ConfigurePlugins(tk.Toplevel):
    """ Pop-up window for the setting of Faceswap Configuration Options.

    Parameters
    ----------
    name: str
        The name of the configuration file. Used for the pop-up title bar.
    configuration: :class:`~lib.config.FaceswapConfig`
        The configuration options for the requested pop-up window
    """
    def __init__(self, name, configuration):
        logger.debug("Initializing %s: (name: %s, configuration: %s)",
                     self.__class__.__name__, name, configuration)
        super().__init__()
        self._name = name
        self._config = configuration
        self._root = get_config().root

        self._set_geometry()

        self._page_frame = ttk.Frame(self)
        self._plugin_info = dict()

        self._config_cpanel_dict = self._get_config()
        self._build()
        self.update()

        self._page_frame.pack(fill=tk.BOTH, expand=True)
        self.title("{} Plugins".format(self._name.title()))
        self.tk.call('wm', 'iconphoto', self._w, get_images().icons["favicon"])
        logger.debug("Initialized %s", self.__class__.__name__)

    def _set_geometry(self):
        """ Set the geometry of the pop-up window """
        scaling_factor = get_config().scaling_factor
        pos_x = self._root.winfo_x() + 80
        pos_y = self._root.winfo_y() + 80
        width = int(600 * scaling_factor)
        height = int(400 * scaling_factor)
        logger.debug("Pop up Geometry: %sx%s, %s+%s", width, height, pos_x, pos_y)
        self.geometry("{}x{}+{}+{}".format(width, height, pos_x, pos_y))

    def _get_config(self):
        """ Format the configuration options stored in :attr:`_config` into a dict of
        :class:`~lib.gui.control_helper.ControlPanelOption's for placement into option frames.

        Returns
        -------
        dict
            A dictionary of section names to :class:`~lib.gui.control_helper.ControlPanelOption`
            objects
        """
        logger.debug("Formatting Config for GUI")
        conf = dict()
        for section in self._config.config.sections():
            self._config.section = section
            category = section.split(".")[0]
            options = self._config.defaults[section]
            section = section.split(".")[-1]
            conf.setdefault(category, dict())[section] = OrderedDict()
            for key, val in options.items():
                if key == "helptext":
                    self._plugin_info[section] = val
                    continue
                initial_value = self._config.config_dict[key]
                initial_value = "none" if initial_value is None else initial_value
                conf[category][section][key] = ControlPanelOption(
                    title=key,
                    dtype=val["type"],
                    group=val["group"],
                    default=val["default"],
                    initial_value=initial_value,
                    choices=val["choices"],
                    is_radio=val["gui_radio"],
                    rounding=val["rounding"],
                    min_max=val["min_max"],
                    helptext=val["helptext"])
        logger.debug("Formatted Config for GUI: %s", conf)
        return conf

    def _build(self):
        """ Build the configuration pop-up window"""
        logger.debug("Building plugin config popup")
        container = ttk.Notebook(self._page_frame)
        categories = sorted(list(self._config_cpanel_dict.keys()))
        if "global" in categories:  # Move global to first item
            categories.insert(0, categories.pop(categories.index("global")))
        for category in categories:
            page = self._build_page(container, category)
            container.add(page, text=category.title())

        self._add_frame_separator()
        self._add_actions()

        container.pack(fill=tk.BOTH, expand=True)
        logger.debug("Built plugin config popup")

    def _build_page(self, container, category):
        """ Build a single tab within the plugin's configuration pop-up.

        Parameters
        ----------
        container: :class:`ttk.Notebook`
            The notebook to place the category options into
        category: str
            The name of the categories to build options for

        Returns
        -------
        :class:'~lib.gui.control_helper.ControlPanel` or :class:`ttk.Notebook`
            The control panel options in a Control Panel frame (for single plugin configurations)
            or a Notebook containing tabs with Control Panel frames (for multi-plugin
            configurations)
        """
        logger.debug("Building plugin config page: '%s'", category)
        plugins = sorted(list(key for key in self._config_cpanel_dict[category].keys()))
        panel_kwargs = dict(columns=2, max_columns=2, option_columns=2, blank_nones=False)
        if any(plugin != category for plugin in plugins):
            page = ttk.Notebook(container)
            for plugin in plugins:
                cp_options = list(self._config_cpanel_dict[category][plugin].values())
                frame = ControlPanel(page,
                                     cp_options,
                                     header_text=self._plugin_info[plugin],
                                     **panel_kwargs)
                title = plugin[plugin.rfind(".") + 1:]
                title = title.replace("_", " ").title()
                page.add(frame, text=title)
            page.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        else:
            cp_options = list(self._config_cpanel_dict[category][plugins[0]].values())
            page = ControlPanel(container,
                                cp_options,
                                header_text=self._plugin_info[plugins[0]],
                                **panel_kwargs)

        logger.debug("Built plugin config page: '%s'", category)

        return page

    def _add_frame_separator(self):
        """ Add a separator between the configuration options and the action buttons. """
        logger.debug("Add frame seperator")
        sep = ttk.Frame(self._page_frame, height=2, relief=tk.RIDGE)
        sep.pack(fill=tk.X, pady=(5, 0), side=tk.BOTTOM)
        logger.debug("Added frame seperator")

    def _add_actions(self):
        """ Add Action buttons to the bottom of the pop-up window. """
        logger.debug("Add action buttons")
        frame = ttk.Frame(self._page_frame)
        btn_cls = ttk.Button(frame, text="Cancel", width=10, command=self.destroy)
        btn_ok = ttk.Button(frame, text="OK", width=10, command=self._save)
        btn_rst = ttk.Button(frame, text="Reset", width=10, command=self._reset)

        Tooltip(btn_cls, text="Close without saving", wraplength=720)
        Tooltip(btn_ok, text="Close and save config", wraplength=720)
        Tooltip(btn_rst, text="Reset all plugins to default values", wraplength=720)

        frame.pack(fill=tk.BOTH, padx=5, pady=5, side=tk.BOTTOM)
        btn_cls.pack(padx=2, side=tk.RIGHT)
        btn_ok.pack(padx=2, side=tk.RIGHT)
        btn_rst.pack(padx=2, side=tk.RIGHT)

        logger.debug("Added action buttons")

    def _reset(self):
        """ Reset all configuration options to their default values. """
        logger.debug("Resetting config")
        for section, items in self._config.defaults.items():
            logger.debug("Resetting section: '%s'", section)
            lookup = [section.split(".")[0], section.split(".")[-1]]
            for item, def_opt in items.items():
                if item == "helptext":
                    continue
                default = def_opt["default"]
                logger.debug("Resetting: '%s' to '%s'", item, default)
                self._config_cpanel_dict[lookup[0]][lookup[1]][item].set(default)

    def _save(self):
        """ Save the configuration file to disk. """
        logger.debug("Saving config")
        options = {".".join((key, sect)) if sect != key else key: opts
                   for key, value in self._config_cpanel_dict.items()
                   for sect, opts in value.items()}
        new_config = ConfigParser(allow_no_value=True)
        for section, items in self._config.defaults.items():
            logger.debug("Adding section: '%s')", section)
            self._config.insert_config_section(section, items["helptext"], config=new_config)
            for item, def_opt in items.items():
                if item == "helptext":
                    continue
                new_opt = options[section][item].get()
                logger.debug("Adding option: (item: '%s', default: %s new: '%s'",
                             item, def_opt, new_opt)
                helptext = def_opt["helptext"]
                helptext = self._config.format_help(helptext, is_section=False)
                new_config.set(section, helptext)
                new_config.set(section, item, str(new_opt))
        self._config.config = new_config
        self._config.save_config()
        logger.info("Saved config: '%s'", self._config.configfile)
        self.destroy()

        running_task = get_config().tk_vars["runningtask"].get()
        if self._name.lower() == "gui" and not running_task:
            self._root.rebuild()
        elif self._name.lower() == "gui" and running_task:
            logger.info("Can't redraw GUI whilst a task is running. GUI Settings will be applied "
                        "at the next restart.")
        logger.debug("Saved config")
