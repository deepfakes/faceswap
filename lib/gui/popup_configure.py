#!/usr/bin python3
""" The pop-up window of the Faceswap GUI for the setting of configuration options. """

from collections import OrderedDict
from configparser import ConfigParser
import logging
import os
import sys
import tkinter as tk
from tkinter import ttk
from importlib import import_module

from .control_helper import ControlPanel, ControlPanelOption
from .custom_widgets import Tooltip
from .utils import get_config, get_images

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
_POPUP = []
_CONFIG_FILES = []
_CONFIGS = dict()


class _State():
    """ Holds the existing config files and the current state of the popup window. """
    def __init__(self):
        self._popup = None
        # The GUI Config cannot be scanned until GUI is launched, so this is populated
        # on the first call to load the settings
        self._configs = dict()

    def open_popup(self, name=None):
        """ Launch the popup, ensuring only one instance is ever open

        Parameters
        ----------
        name: str, Optional
            The name of the configuration file. Used for selecting the correct section if required.
            Set to ``None`` if no initial section should be selected. Default: ``None``
        """
        if not self._configs:
            self._scan_for_configs()
        logger.debug("name: %s", name)
        if self._popup is not None:
            logger.info("Popup already open. Returning: %s", _POPUP)
            return
        self._popup = _ConfigurePlugins(name, self._configs)

    def close_popup(self):
        """ Destroy the open popup and remove it from tracking. """
        if self._popup is None:
            logger.info("No popup to close. Returning")
            return
        self._popup.destroy()
        del self._popup
        self._popup = None

    def _scan_for_configs(self):
        """ Scan the plugin folders for configuration settings. Add in the GUI configuration also.

        Populates the attribute :attr:`_configs`.
        """
        root_path = os.path.abspath(os.path.dirname(sys.argv[0]))
        plugins_path = os.path.join(root_path, "plugins")
        logger.debug("Scanning path: '%s'", plugins_path)
        for dirpath, _, filenames in os.walk(plugins_path):
            if "_config.py" in filenames:
                plugin_type = os.path.split(dirpath)[-1]
                config = self._load_config(plugin_type)
                self._configs[plugin_type] = config
        self._configs["gui"] = get_config().user_config
        logger.debug("Configs loaded: %s", sorted(list(self._configs.keys())))

    @classmethod
    def _load_config(cls, plugin_type):
        """ Load the config from disk. If the file doesn't exist, then it will be generated.

        Parameters
        ----------
        plugin_type: str
            The plugin type (i.e. extract, train convert) that the config should be loaded for

        Returns
        -------
        :class:`lib.config.FaceswapConfig`
            The Configuration for the selected plugin
        """
        # Load config to generate default if doesn't exist
        mod = ".".join(("plugins", plugin_type, "_config"))
        module = import_module(mod)
        config = module.Config(None)
        logger.debug("Found '%s' config at '%s'", plugin_type, config.configfile)
        return config


_STATE = _State()
open_popup = _STATE.open_popup


class _ConfigurePlugins(tk.Toplevel):
    """ Pop-up window for the setting of Faceswap Configuration Options.

    Parameters
    ----------
    name: str
        The name of the section that is being navigated to. Used for opening on the correct
        page in the Tree View.
    configurations: dict
        Dictionary containing the :class:`~lib.config.FaceswapConfig` object for each
        configuration section for the requested pop-up window
    """
    def __init__(self, name, configurations):
        logger.debug("Initializing %s: (name: %s, configurations: %s)",
                     self.__class__.__name__, name, configurations)
        super().__init__()
        self._root = get_config().root
        self._set_geometry()
        self._tk_vars = dict(header=tk.StringVar())

        header_frame = self._build_header()
        content_frame = ttk.Frame(self)

        self._tree = _Tree(content_frame, configurations, name).tree
        self._tree.bind("<ButtonRelease-1>", self._select_item)

        self._opts_frame = DisplayArea(content_frame, configurations, self._tree)
        self._opts_frame.pack(fill=tk.BOTH, expand=True, side=tk.RIGHT)
        footer_frame = self._build_footer()

        header_frame.pack(fill=tk.X, padx=5, pady=5, side=tk.TOP)
        content_frame.pack(fill=tk.BOTH, padx=5, pady=(0, 5), expand=True, side=tk.TOP)
        footer_frame.pack(fill=tk.X, padx=5, pady=(0, 5), side=tk.BOTTOM)

        select = name if name else self._tree.get_children()[0]
        self._tree.selection_set(select)
        self._tree.focus(select)
        self._select_item(0)

        self.title("Configure Settings")
        self.tk.call('wm', 'iconphoto', self._w, get_images().icons["favicon"])
        self.protocol("WM_DELETE_WINDOW", _STATE.close_popup)

        logger.debug("Initialized %s", self.__class__.__name__)

    def _set_geometry(self):
        """ Set the geometry of the pop-up window """
        scaling_factor = get_config().scaling_factor
        pos_x = self._root.winfo_x() + 80
        pos_y = self._root.winfo_y() + 80
        width = int(600 * scaling_factor)
        height = int(536 * scaling_factor)
        logger.debug("Pop up Geometry: %sx%s, %s+%s", width, height, pos_x, pos_y)
        self.geometry("{}x{}+{}+{}".format(width, height, pos_x, pos_y))

    def _build_header(self):
        """ Build the main header text and separator. """
        header_frame = ttk.Frame(self)
        lbl_frame = ttk.Frame(header_frame)

        self._tk_vars["header"].set("Settings")
        lbl_header = ttk.Label(lbl_frame,
                               textvariable=self._tk_vars["header"],
                               anchor=tk.W,
                               style="H1.TLabel")
        lbl_header.pack(fill=tk.X, expand=True, side=tk.LEFT)

        sep = ttk.Frame(header_frame, height=2, relief=tk.RIDGE)

        lbl_frame.pack(fill=tk.X, expand=True, side=tk.TOP)
        sep.pack(fill=tk.X, pady=(1, 0), side=tk.BOTTOM)
        return header_frame

    def _build_footer(self):
        """ Build the main footer buttons and separator. """
        logger.debug("Adding action buttons")
        frame = ttk.Frame(self)
        left_frame = ttk.Frame(frame)
        right_frame = ttk.Frame(frame)

        btn_saveall = ttk.Button(left_frame,
                                 text="Save All",
                                 width=10,
                                 command=self._opts_frame.save)
        btn_rstall = ttk.Button(left_frame,
                                text="Reset All",
                                width=10,
                                command=self._opts_frame.reset)

        btn_cls = ttk.Button(right_frame, text="Cancel", width=10, command=_STATE.close_popup)
        btn_save = ttk.Button(right_frame,
                              text="Save",
                              width=10,
                              command=lambda: self._opts_frame.save(page_only=True))
        btn_rst = ttk.Button(right_frame,
                             text="Reset",
                             width=10,
                             command=lambda: self._opts_frame.reset(page_only=True))

        Tooltip(btn_cls, text="Close without saving", wraplength=720)
        Tooltip(btn_save, text="Save this page's config", wraplength=720)
        Tooltip(btn_rst, text="Reset this page's config to default values", wraplength=720)
        Tooltip(btn_saveall,
                text="Save all settings for the currently selected config",
                wraplength=720)
        Tooltip(btn_rstall,
                text="Reset all settings for the currently selected config to default values",
                wraplength=720)

        btn_cls.pack(padx=2, side=tk.RIGHT)
        btn_save.pack(padx=2, side=tk.RIGHT)
        btn_rst.pack(padx=2, side=tk.RIGHT)
        btn_saveall.pack(padx=2, side=tk.RIGHT)
        btn_rstall.pack(padx=2, side=tk.RIGHT)

        left_frame.pack(side=tk.LEFT)
        right_frame.pack(side=tk.RIGHT)
        logger.debug("Added action buttons")
        return frame

    def _select_item(self, event):  # pylint:disable=unused-argument
        """ Update the session summary info with the selected item or launch graph.

        If the mouse is clicked on the graph icon, then the session summary pop-up graph is
        launched. Otherwise the selected ID is stored.

        Parameters
        ----------
        event: :class:`tkinter.Event`
            The tkinter mouse button release event. Unused.
        """
        selection = self._tree.focus()
        section = selection.split("|")[0]
        subsections = selection.split("|")[1:] if "|" in selection else []
        self._tk_vars["header"].set("{} Settings".format(section.title()))
        self._opts_frame.select_options(section, subsections)


class _Tree(ttk.Frame):  # pylint:disable=too-many-ancestors
    """ Frame that holds the Tree View Navigator and scroll bar for the configuration pop-up.

    Parameters
    ----------
    parent: :class:`tkinter.ttk.Frame`
        The parent frame to the Tree View area
    configurations: dict
        Dictionary containing the :class:`~lib.config.FaceswapConfig` object for each
        configuration section for the requested pop-up window
    name: str
        The name of the section that is being navigated to. Used for opening on the correct
        page in the Tree View. ``None`` if no specific area is being navigated to
    """
    def __init__(self, parent, configurations, name):
        super().__init__(parent)
        self._fix_styles()

        frame = ttk.Frame(self, relief=tk.SOLID, borderwidth=1)
        self._tree = self._build_tree(frame, configurations, name)
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=self._tree.yview)

        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self._tree.pack(fill=tk.Y, expand=True)
        self._tree.configure(yscrollcommand=scrollbar.set)
        frame.pack(expand=True, fill=tk.Y)
        self.pack(side=tk.LEFT, fill=tk.Y)

    @property
    def tree(self):
        """ :class:`tkinter.ttk.TreeView` The Tree View held within the frame """
        return self._tree

    @classmethod
    def _fix_styles(cls):
        """ Tkinter has a bug when setting the background style on certain OSes. This fixes the
        issue so we can set different colored backgrounds.

        We also set some default styles for our tree view.
        """
        style = ttk.Style()
        fix_map = lambda o: [elm for elm in style.map("Treeview", query_opt=o)  # noqa
                             if elm[:2] != ("!disabled", "!selected")]
        style.map("Treeview", foreground=fix_map("foreground"), background=fix_map("background"))
        # Remove the Borders
        style.configure("ConfigNav.Treeview", bd=0)
        style.layout("ConfigNav.Treeview", [('ConfigNav.Treeview.treearea', {'sticky': 'nswe'})])

    def _build_tree(self, parent, configurations, name):
        """ Build the configuration pop-up window.

        Parameters
        ----------
        configurations: dict
            Dictionary containing the :class:`~lib.config.FaceswapConfig` object for each
            configuration section for the requested pop-up window
        name: str
            The name of the section that is being navigated to. Used for opening on the correct
            page in the Tree View. ``None`` if no specific area is being navigated to

        Returns
        -------
        :class:`tkinter.ttk.TreeView`
            The populated tree view
        """
        logger.debug("Building Tree View Navigator")
        tree = ttk.Treeview(parent, show="tree", style="ConfigNav.Treeview")
        data = {category: [sect.split(".") for sect in sorted(conf.config.sections())]
                for category, conf in configurations.items()}
        ordered = sorted(list(data.keys()))
        categories = ["extract", "train", "convert"]
        categories += [x for x in ordered if x not in categories]

        for cat in categories:
            img = get_images().icons.get("settings_{}".format(cat), "")
            text = cat.replace("_", " ").title()
            text = " " + text if img else text
            is_open = tk.TRUE if name is None or name == cat else tk.FALSE
            tree.insert("", "end", cat, text=text, image=img, open=is_open, tags="category")
            self._process_sections(tree, data[cat], cat, name == cat)

        tree.tag_configure('category', background='#DFDFDF')
        tree.tag_configure('section', background='#E8E8E8')
        tree.tag_configure('option', background='#F0F0F0')
        logger.debug("Tree View Navigator")
        return tree

    @classmethod
    def _process_sections(cls, tree, sections, category, is_open):
        """ Process the sections of a category's configuration.

        Creates a category's sections, then the sub options for that category

        Parameters
        ----------
        tree: :class:`tkinter.ttk.TreeView`
            The tree view to insert sections into
        sections: list
            The sections to insert into the Tree View
        category: str
            The category node that these sections sit in
        is_open: bool
            ``True`` if the node should be created in "open" mode. ``False`` if it should be
            closed.
        """
        seen = set()
        for section in sections:
            if section[-1] == "global":  # Global categories get escalated to parent
                continue
            sect = section[0]
            section_id = "{}|{}".format(category, sect)
            if sect not in seen:
                seen.add(sect)
                text = sect.replace("_", " ").title()
                tree.insert(category, "end", section_id, text=text, open=is_open, tags="section")
            if len(section) == 2:
                opt = section[-1]
                opt_id = "{}|{}".format(section_id, opt)
                opt_text = opt.replace("_", " ").title()
                tree.insert(section_id, "end", opt_id, text=opt_text, open=is_open, tags="option")


class DisplayArea(ttk.Frame):  # pylint:disable=too-many-ancestors
    """ The option configuration area of the pop up options.

    Parameters
    ----------
    parent: :class:`tkinter.ttk.Frame`
        The parent frame that holds the Display Area of the pop up configuration window
    tree: :class:`tkinter.ttk.TreeView`
        The Tree View navigator for the pop up configuration window
    configurations: dict
        Dictionary containing the :class:`~lib.config.FaceswapConfig` object for each
        configuration section for the requested pop-up window
    """
    def __init__(self, parent, configurations, tree):
        super().__init__(parent)
        self._configs = configurations
        self._tree = tree
        self._vars = dict()
        self._cache = dict()
        self._config_cpanel_dict = self._get_config()
        self._build_header()
        self._displayed_frame = None

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
        retval = dict()
        for plugin, conf in self._configs.items():
            for section in conf.config.sections():
                conf.section = section
                category = section.split(".")[0]
                sect = section.split(".")[-1]
                # Elevate global to root
                key = plugin if sect == "global" else "{}|{}|{}".format(plugin, category, sect)
                retval[key] = dict(helptext=None, options=OrderedDict())

                for option, params in conf.defaults[section].items():
                    if option == "helptext":
                        retval[key]["helptext"] = params
                        continue
                    initial_value = conf.config_dict[option]
                    initial_value = "none" if initial_value is None else initial_value
                    retval[key]["options"][option] = ControlPanelOption(
                        title=option,
                        dtype=params["type"],
                        group=params["group"],
                        default=params["default"],
                        initial_value=initial_value,
                        choices=params["choices"],
                        is_radio=params["gui_radio"],
                        rounding=params["rounding"],
                        min_max=params["min_max"],
                        helptext=params["helptext"])
        logger.debug("Formatted Config for GUI: %s", retval)
        return retval

    def _build_header(self):
        """ Build the dynamic header text. """
        header_frame = ttk.Frame(self)
        var = tk.StringVar()
        lbl = ttk.Label(header_frame, textvariable=var, anchor=tk.W, style="H2.TLabel")
        lbl.pack(fill=tk.X, expand=True, side=tk.TOP)
        header_frame.pack(fill=tk.X, padx=5, pady=(5, 0), side=tk.TOP)
        self._vars["header"] = var

    def select_options(self, section, subsections):
        """ Display the page for the given section and subsections.

        Parameters
        ----------
        section: str
            The main section to be navigated to (or root node)
        subsections: list
            The full list of subsections ending on the required node
        """
        labels = ["global"] if not subsections else subsections
        self._vars["header"].set(" - ".join(sect.replace("_", " ").title() for sect in labels))
        self._set_display(section, subsections)

    def _set_display(self, section, subsections):
        """ Set the correct display page for the given section and subsections.

        Parameters
        ----------
        section: str
            The main section to be navigated to (or root node)
        subsections: list
            The full list of subsections ending on the required node
        """
        key = "|".join([section] + subsections)
        if self._displayed_frame is not None:
            self._displayed_frame.pack_forget()

        if key not in self._cache:
            self._cache_page(key)

        self._displayed_frame = self._cache[key]
        self._displayed_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

    def _cache_page(self, key):
        """ Create the control panel options for the requested configuration and cache.

        Parameters
        ----------
        key: str
            The lookup key to the settings cache
        """
        panel_kwargs = dict(columns=1, max_columns=1, option_columns=4, blank_nones=False)
        info = self._config_cpanel_dict.get(key, None)
        if info is None:
            logger.debug("key '%s' does not exist in options. Creating links page.", key)
            self._cache[key] = self._create_links_page(key)
        else:
            self._cache[key] = ControlPanel(self,
                                            list(info["options"].values()),
                                            header_text=info["helptext"],
                                            **panel_kwargs)

    def _create_links_page(self, key):
        """ For headings which don't have settings, build a links page to the subsections.

        Parameters
        ----------
        key: str
            The lookup key to set the links page for
        """
        frame = ttk.Frame(self)
        links = {item.replace(key, "")[1:].split("|")[0]
                 for item in self._config_cpanel_dict
                 if item.startswith(key)}

        if not links:
            return frame

        header_lbl = ttk.Label(frame, text="Select a plugin to configure:")
        header_lbl.pack(side=tk.TOP, fill=tk.X, padx=5, pady=(5, 10))
        for link in sorted(links):
            lbl = ttk.Label(frame,
                            text=link.replace("_", " ").title(),
                            anchor=tk.W,
                            foreground="blue",
                            cursor="hand2")
            lbl.pack(side=tk.TOP, fill=tk.X, padx=10, pady=(0, 5))
            bind = "{}|{}".format(key, link)
            lbl.bind("<Button-1>", lambda e, l=bind: self._link_callback(l))

        return frame

    def _link_callback(self, identifier):
        """ Set the tree view to the selected item and display the requested page on a link click.

        Parameters
        ----------
        identifier: str
            The identifier from the tree view for the page to display
        """
        parent = "|".join(identifier.split("|")[:-1])
        self._tree.item(parent, open=True)
        self._tree.selection_set(identifier)
        self._tree.focus(identifier)
        split = identifier.split("|")
        section = split[0]
        subsections = split[1:] if len(split) > 1 else []
        self.select_options(section, subsections)

    def reset(self, page_only=False):
        """ Reset all configuration options to their default values.

        Parameters
        ----------
        page_only: bool, optional
            ``True`` resets just the currently selected page's options to default, ``False`` resets
            all plugins within the currently selected config to default. Default: ``False``
        """
        logger.debug("Resetting config, page_only: %s", page_only)
        selection = self._tree.focus()
        if page_only:
            if selection not in self._config_cpanel_dict:
                logger.info("No configuration options to reset for current page: %s", selection)
                return
            items = list(self._config_cpanel_dict[selection]["options"].values())
        else:
            items = [opt
                     for key, val in self._config_cpanel_dict.items()
                     for opt in val["options"].values()
                     if key.startswith(selection.split("|")[0])]
        for item in items:
            logger.debug("Resetting item '%s' from '%s' to default '%s'",
                         item.title, item.get(), item.default)
            item.set(item.default)
        logger.debug("Reset config")

    def save(self, page_only=False):
        """ Save the configuration file to disk.

        Parameters
        ----------
        page_only: bool, optional
            ``True`` saves just the currently selected page's options, ``False`` saves all the
            plugins options within the currently selected config. Default: ``False``
        """
        logger.debug("Saving config")
        selection = self._tree.focus()
        category = selection.split("|")[0]
        config = self._configs[category]
        # Create a new config to pull through any defaults change
        new_config = ConfigParser(allow_no_value=True)

        if "|" in selection:
            lookup = ".".join(selection.split("|")[1:])
        else:  # Expand global out from root node
            lookup = "global"

        if page_only and lookup not in config.config.sections():
            logger.info("No settings to save for the current page")
            return

        for section, items in config.defaults.items():
            logger.debug("Adding section: '%s')", section)
            config.insert_config_section(section, items["helptext"], config=new_config)
            for item, options in items.items():
                if item == "helptext":
                    continue
                if page_only and section != lookup:
                    # Keep existing values for pages we are not updating
                    new_opt = config.get(section, item)
                    logger.debug("Retain existing value '%s' for %s",
                                 new_opt, ".".join([section, item]))
                else:
                    # Get currently selected value
                    key = category
                    if section != "global":
                        key += "|{}".format(section.replace(".", "|"))
                    new_opt = self._config_cpanel_dict[key]["options"][item].get()
                    logger.debug("Updating value to '%s' for %s",
                                 new_opt, ".".join([section, item]))
                helptext = config.format_help(options["helptext"], is_section=False)
                new_config.set(section, helptext)
                new_config.set(section, item, str(new_opt))
        config.config = new_config
        config.save_config()
        logger.info("Saved config: '%s'", config.configfile)

        if category == "gui":
            if not get_config().tk_vars["runningtask"].get():
                get_config().root.rebuild()
            else:
                logger.info("Can't redraw GUI whilst a task is running. GUI Settings will be "
                            "applied at the next restart.")
        logger.debug("Saved config")
