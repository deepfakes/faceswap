#!/usr/bin/env python3
""" Default configurations for faceswap. Handles parsing and validating of Faceswap Configs and
interfacing with :class:`configparser.ConfigParser` """
from __future__ import annotations

import inspect
import logging
import os
import sys

from importlib import import_module

from lib.utils import full_path_split, get_module_objects, PROJECT_ROOT

from .ini import ConfigFile
from .objects import ConfigItem, ConfigSection, GlobalSection


logger = logging.getLogger(__name__)

_CONFIGS: dict[str, FaceswapConfig] = {}
""" dict[str, FaceswapConfig] : plugin group to FaceswapConfig mapping for all loaded configs """


class FaceswapConfig():
    """ Config Items """
    def __init__(self, configfile: str | None = None) -> None:
        """ Init Configuration

        Parameters
        ----------
        configfile : str, optional
            Optional path to a config file. ``None`` for default location. Default: ``None``
        """
        logger.debug("Initializing: %s", self.__class__.__name__)

        self._plugin_group = self._get_plugin_group()

        self._ini = ConfigFile(self._plugin_group, ini_path=configfile)
        self.sections: dict[str, ConfigSection] = {}
        """ dict[str, :class:`ConfigSection`] : The Faceswap config sections and options """

        self._set_defaults()
        self._ini.on_load(self.sections)
        _CONFIGS[self._plugin_group] = self

        logger.debug("Initialized: %s", self.__class__.__name__)

    def _get_plugin_group(self) -> str:
        """ Obtain the name of the plugin group based on the child module's folder path

        Returns
        -------
        str
            The plugin group for this Config object
        """
        mod_split = self.__module__.split(".")
        mod_name = mod_split[-1]
        retval = mod_name.rsplit("_", maxsplit=1)[0]
        logger.debug("Got plugin group '%s' from module '%s'",
                     retval, self.__module__)
        # Sanity check in case of defaults config file name/location changes
        parent = mod_split[-2]
        assert mod_name == f"{parent}_config"
        return retval

    def add_section(self, title: str, info: str) -> None:
        """ Add a default section to config file

        Parameters
        ----------
        title : str
            The title for the section
        info : str
            The helptext for the section
        """
        logger.debug("Add section: (title: '%s', info: '%s')", title, info)
        self.sections[title] = ConfigSection(helptext=info, options={})

    def add_item(self, section: str, title: str, config_item: ConfigItem) -> None:
        """ Add a default item to a config section

        Parameters
        ----------
        section : str
            The section of the config to add the item to
        title : str
            The name of the config item
        config_item : :class:`~lib.config.objects.ConfigItem`
            The default config item object to add to the config
        """
        logger.debug("Add item: (section: '%s', item: %s", section, config_item)
        self.sections[section].options[title] = config_item

    def _import_defaults_from_module(self,
                                     filename: str,
                                     module_path: str,
                                     plugin_type: str) -> None:
        """ Load the plugin's defaults module, extract defaults and add to default configuration.

        Parameters
        ----------
        filename : str
            The filename to load the defaults from
        module_path : str
            The path to load the module from
        plugin_type : str
            The type of plugin that the defaults are being loaded for
        """
        logger.debug("Adding defaults: (filename: %s, module_path: %s, plugin_type: %s",
                     filename, module_path, plugin_type)
        module = os.path.splitext(filename)[0]
        section = ".".join((plugin_type, module.replace("_defaults", "")))
        logger.debug("Importing defaults module: %s.%s", module_path, module)
        mod = import_module(f"{module_path}.{module}")
        self.add_section(section, mod.HELPTEXT)  # type:ignore[attr-defined]
        for key, val in vars(mod).items():
            if isinstance(val, ConfigItem):
                self.add_item(section=section, title=key, config_item=val)
        logger.debug("Added defaults: %s", section)

    def _defaults_from_plugin(self, plugin_folder: str) -> None:
        """ Scan the given plugins folder for config defaults.py files and update the
        default configuration.

        Parameters
        ----------
        plugin_folder : str
            The folder to scan for plugins
        """
        for dirpath, _, filenames in os.walk(plugin_folder):
            default_files = [fname for fname in filenames if fname.endswith("_defaults.py")]
            if not default_files:
                continue
            base_path = os.path.dirname(os.path.realpath(sys.argv[0]))
            # Can't use replace as there is a bug on some Windows installs that lowers some paths
            import_path = ".".join(full_path_split(dirpath[len(base_path):])[1:])
            plugin_type = import_path.rsplit(".", maxsplit=1)[-1]
            for filename in default_files:
                self._import_defaults_from_module(filename, import_path, plugin_type)

    def set_defaults(self, helptext: str = "") -> None:
        """ Override for plugin specific config defaults.

        This method should always be overriden to add the help text for the global plugin group.
        If `helptext` is not provided, then it is assumed that there is no global section for this
        plugin group.

        The default action will parse the child class' module for
        :class:`~lib.config.objects.ConfigItem` objects and add them to this plugin group's
        "global" section of :attr:`sections`.

        The name of each config option will be the variable name found in the module.

        It will then parse the child class' module for subclasses of
        :class:`~lib.config.objects.GlobalSection` objects and add each of these sections to this
        plugin group's :attr:`sections`, adding any :class:`~lib.config.objects.ConfigItem` within
        the GlobalSection to that sub-section.

        The section name will be the name of the GlobalSection subclass, lowercased

        Parameters
        ----------
        helptext : str
            The help text to display for the plugin group

        Raises
        ------
        ValueError
            If the plugin group's help text has not been provided
        """
        section = "global"
        logger.debug("[%s:%s] Adding defaults", self._plugin_group, section)

        if not helptext:
            logger.debug("No help text provided for '%s'. Not creating global section",
                         self.__module__)
            return

        self.add_section(section, helptext)

        for key, val in vars(sys.modules[self.__module__]).items():
            if isinstance(val, ConfigItem):
                self.add_item(section=section, title=key, config_item=val)
        logger.debug("[%s:%s] Added defaults", self._plugin_group, section)

        # Add global sub-sections
        for key, val in vars(sys.modules[self.__module__]).items():
            if inspect.isclass(val) and issubclass(val, GlobalSection) and val != GlobalSection:
                section_name = f"{section}.{key.lower()}"
                self.add_section(section_name, val.helptext)
                for opt_name, opt in val.__dict__.items():
                    if isinstance(opt, ConfigItem):
                        self.add_item(section=section_name, title=opt_name, config_item=opt)

    def _set_defaults(self) -> None:
        """Load the plugin's default values, set the object names and order the sections, global
        first then alphabetically."""
        self.set_defaults()
        for section_name, section in self.sections.items():
            for opt_name, opt in section.options.items():
                opt.set_name(f"{self._plugin_group}.{section_name}.{opt_name}")

        global_keys = sorted(s for s in self.sections if s.startswith("global"))
        remaining_keys = sorted(s for s in self.sections if not s.startswith("global"))
        ordered = {k: self.sections[k] for k in global_keys + remaining_keys}

        self.sections = ordered

    def save_config(self) -> None:
        """Update the ini file with the currently stored app values and save the config file."""
        self._ini.update_from_app(self.sections)


def get_configs() -> dict[str, FaceswapConfig]:
    """ Get all of the FaceswapConfig options. Loads any configs that have not been loaded and
    return a dictionary of all configs.

    Returns
    -------
    dict[str, :class:`FaceswapConfig`]
        All of the loaded faceswap config objects
    """
    generate_configs(force=True)
    return _CONFIGS


def generate_configs(force: bool = False) -> None:
    """ Generate config files if they don't exist.

    This script is run prior to anything being set up, so don't use logging
    Generates the default config files for plugins in the faceswap config folder

    Logic:
        - Scan the plugins path for files named <parent_folder>_config.py>
        - Import the discovered module and look for instances of FaceswapConfig
        - If exists initialize the class

    Parameters
    ----------
    force : bool
        Force the loading of all plugin configs even if their .ini files pre-exist
    """
    configs_path = os.path.join(PROJECT_ROOT, "config")
    plugins_path = os.path.join(PROJECT_ROOT, "plugins")
    for dirpath, _, filenames in os.walk(plugins_path):
        relative_path = dirpath.replace(PROJECT_ROOT, "")[1:]
        if len(full_path_split(relative_path)) > 2:  # don't dig further than 1 folder deep
            continue
        plugin_group = os.path.basename(dirpath)
        filename = f"{plugin_group}_config.py"
        if filename not in filenames:
            continue

        if plugin_group in _CONFIGS:
            continue

        config_file = os.path.join(configs_path, f"{plugin_group}.ini")
        if not os.path.exists(config_file) or force:
            modname = os.path.splitext(filename)[0]
            modpath = os.path.join(dirpath.replace(PROJECT_ROOT, ""),
                                   modname)[1:].replace(os.sep, ".")
            mod = import_module(modpath)
            for obj in vars(mod).values():
                if (inspect.isclass(obj)
                        and issubclass(obj, FaceswapConfig)
                        and obj != FaceswapConfig):
                    obj()


__all__ = get_module_objects(__name__)
