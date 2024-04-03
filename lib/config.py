#!/usr/bin/env python3
""" Default configurations for faceswap.
    Extends out :class:`configparser.ConfigParser` functionality by checking for default
    configuration updates and returning data in it's correct format """

import gettext
import logging
import os
import sys
import textwrap

from collections import OrderedDict
from configparser import ConfigParser
from dataclasses import dataclass
from importlib import import_module

from lib.utils import full_path_split

# LOCALES
_LANG = gettext.translation("lib.config", localedir="locales", fallback=True)
_ = _LANG.gettext

OrderedDictSectionType = OrderedDict[str, "ConfigSection"]
OrderedDictItemType = OrderedDict[str, "ConfigItem"]

logger = logging.getLogger(__name__)
ConfigValueType = bool | int | float | list[str] | str | None


@dataclass
class ConfigItem:
    """ Dataclass for holding information about configuration items

    Parameters
    ----------
    default: any
        The default value for the configuration item
    helptext: str
        The helptext to be displayed for the configuration item
    datatype: type
        The type of the configuration item
    rounding: int
        The decimal places for floats or the step interval for ints for slider updates
    min_max: tuple
        The minumum and maximum value for the GUI slider for the configuration item
    gui_radio: bool
        ``True`` to display the configuration item in a Radio Box
    fixed: bool
        ``True`` if the item cannot be changed for existing models (training only)
    group: str
        The group that this configuration item belongs to in the GUI
    """
    default: ConfigValueType
    helptext: str
    datatype: type
    rounding: int
    min_max: tuple[int, int] | tuple[float, float] | None
    choices: str | list[str]
    gui_radio: bool
    fixed: bool
    group: str | None


@dataclass
class ConfigSection:
    """ Dataclass for holding information about configuration sections

    Parameters
    ----------
    helptext: str
        The helptext to be displayed for the configuration section
    items: :class:`collections.OrderedDict`
        Dictionary of configuration items for the section
    """
    helptext: str
    items: OrderedDictItemType


class FaceswapConfig():
    """ Config Items """
    def __init__(self, section: str | None, configfile: str | None = None) -> None:
        """ Init Configuration

        Parameters
        ----------
        section: str or ``None``
            The configuration section. ``None`` for all sections
        configfile: str, optional
            Optional path to a config file. ``None`` for default location. Default: ``None``
        """
        logger.debug("Initializing: %s", self.__class__.__name__)
        self.configfile = self._get_config_file(configfile)
        self.config = ConfigParser(allow_no_value=True)
        self.defaults: OrderedDictSectionType = OrderedDict()
        self.config.optionxform = str  # type:ignore
        self.section = section

        self.set_defaults()
        self._handle_config()
        logger.debug("Initialized: %s", self.__class__.__name__)

    @property
    def changeable_items(self) -> dict[str, ConfigValueType]:
        """ Training only.
            Return a dict of config items with their set values for items
            that can be altered after the model has been created """
        retval: dict[str, ConfigValueType] = {}
        sections = [sect for sect in self.config.sections() if sect.startswith("global")]
        all_sections = sections if self.section is None else sections + [self.section]
        for sect in all_sections:
            if sect not in self.defaults:
                continue
            for key, val in self.defaults[sect].items.items():
                if val.fixed:
                    continue
                retval[key] = self.get(sect, key)
        logger.debug("Alterable for existing models: %s", retval)
        return retval

    def set_defaults(self) -> None:
        """ Override for plugin specific config defaults

            Should be a series of self.add_section() and self.add_item() calls

            e.g:

            section = "sect_1"
            self.add_section(section,
                             "Section 1 Information")

            self.add_item(section=section,
                          title="option_1",
                          datatype=bool,
                          default=False,
                          info="sect_1 option_1 information")
        """
        raise NotImplementedError

    def _defaults_from_plugin(self, plugin_folder: str) -> None:
        """ Scan the given plugins folder for config defaults.py files and update the
        default configuration.

        Parameters
        ----------
        plugin_folder: str
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
                self._load_defaults_from_module(filename, import_path, plugin_type)

    def _load_defaults_from_module(self,
                                   filename: str,
                                   module_path: str,
                                   plugin_type: str) -> None:
        """ Load the plugin's defaults module, extract defaults and add to default configuration.

        Parameters
        ----------
        filename: str
            The filename to load the defaults from
        module_path: str
            The path to load the module from
        plugin_type: str
            The type of plugin that the defaults are being loaded for
        """
        logger.debug("Adding defaults: (filename: %s, module_path: %s, plugin_type: %s",
                     filename, module_path, plugin_type)
        module = os.path.splitext(filename)[0]
        section = ".".join((plugin_type, module.replace("_defaults", "")))
        logger.debug("Importing defaults module: %s.%s", module_path, module)
        mod = import_module(f"{module_path}.{module}")
        self.add_section(section, mod._HELPTEXT)  # type:ignore[attr-defined]  # pylint:disable=protected-access  # noqa:E501
        for key, val in mod._DEFAULTS.items():  # type:ignore[attr-defined]  # pylint:disable=protected-access  # noqa:E501
            self.add_item(section=section, title=key, **val)
        logger.debug("Added defaults: %s", section)

    @property
    def config_dict(self) -> dict[str, ConfigValueType]:
        """ dict: Collate global options and requested section into a dictionary with the correct
        data types """
        conf: dict[str, ConfigValueType] = {}
        sections = [sect for sect in self.config.sections() if sect.startswith("global")]
        if self.section is not None:
            sections.append(self.section)
        for sect in sections:
            if sect not in self.config.sections():
                continue
            for key in self.config[sect]:
                if key.startswith(("#", "\n")):  # Skip comments
                    continue
                conf[key] = self.get(sect, key)
        return conf

    def get(self, section: str, option: str) -> ConfigValueType:
        """ Return a config item in it's correct format.

        Parameters
        ----------
        section: str
            The configuration section currently being processed
        option: str
            The configuration option currently being processed

        Returns
        -------
        varies
            The selected configuration option in the correct data format
        """
        logger.debug("Getting config item: (section: '%s', option: '%s')", section, option)
        datatype = self.defaults[section].items[option].datatype

        retval: ConfigValueType
        if datatype == bool:
            retval = self.config.getboolean(section, option)
        elif datatype == int:
            retval = self.config.getint(section, option)
        elif datatype == float:
            retval = self.config.getfloat(section, option)
        elif datatype == list:
            retval = self._parse_list(section, option)
        else:
            retval = self.config.get(section, option)

        if isinstance(retval, str) and retval.lower() == "none":
            retval = None
        logger.debug("Returning item: (type: %s, value: %s)", datatype, retval)
        return retval

    def _parse_list(self, section: str, option: str) -> list[str]:
        """ Parse options that are stored as lists in the config file. These can be space or
        comma-separated items in the config file. They will be returned as a list of strings,
        regardless of what the final data type should be, so conversion from strings to other
        formats should be done explicitly within the retrieving code.

        Parameters
        ----------
        section: str
            The configuration section currently being processed
        option: str
            The configuration option currently being processed

        Returns
        -------
        list
            List of `str` selected items for the config choice.
        """
        raw_option = self.config.get(section, option)
        if not raw_option:
            logger.debug("No options selected, returning empty list")
            return []
        delimiter = "," if "," in raw_option else None
        retval = [opt.strip().lower() for opt in raw_option.split(delimiter)]
        logger.debug("Processed raw option '%s' to list %s for section '%s', option '%s'",
                     raw_option, retval, section, option)
        return retval

    def _get_config_file(self, configfile: str | None) -> str:
        """ Return the config file from the calling folder or the provided file

        Parameters
        ----------
        configfile: str or ``None``
            Path to a config file. ``None`` for default location.

        Returns
        -------
        str
            The full path to the configuration file
        """
        if configfile is not None:
            if not os.path.isfile(configfile):
                err = f"Config file does not exist at: {configfile}"
                logger.error(err)
                raise ValueError(err)
            return configfile
        filepath = sys.modules[self.__module__].__file__
        assert filepath is not None
        dirname = os.path.dirname(filepath)
        folder, fname = os.path.split(dirname)
        retval = os.path.join(os.path.dirname(folder), "config", f"{fname}.ini")
        logger.debug("Config File location: '%s'", retval)
        return retval

    def add_section(self, title: str, info: str) -> None:
        """ Add a default section to config file

        Parameters
        ----------
        title: str
            The title for the section
        info: str
            The helptext for the section
        """
        logger.debug("Add section: (title: '%s', info: '%s')", title, info)
        self.defaults[title] = ConfigSection(helptext=info, items=OrderedDict())

    def add_item(self,
                 section: str | None = None,
                 title: str | None = None,
                 datatype: type = str,
                 default: ConfigValueType = None,
                 info: str | None = None,
                 rounding: int | None = None,
                 min_max: tuple[int, int] | tuple[float, float] | None = None,
                 choices: str | list[str] | None = None,
                 gui_radio: bool = False,
                 fixed: bool = True,
                 group: str | None = None) -> None:
        """ Add a default item to a config section

            For int or float values, rounding and min_max must be set
            This is for the slider in the GUI. The min/max values are not enforced:
            rounding:   sets the decimal places for floats or the step interval for ints.
            min_max:    tuple of min and max accepted values

            For str values choices can be set to validate input and create a combo box
            in the GUI

            For list values, choices must be provided, and a multi-option select box will
            be created

            is_radio is to indicate to the GUI that it should display Radio Buttons rather than
            combo boxes for multiple choice options.

            The 'fixed' parameter is only for training configurations. Training configurations
            are set when the model is created, and then reloaded from the state file.
            Marking an item as fixed=False indicates that this value can be changed for
            existing models, and will override the value saved in the state file with the
            updated value in config.

            The 'Group' parameter allows you to assign the config item to a group in the GUI

        """
        logger.debug("Add item: (section: '%s', title: '%s', datatype: '%s', default: '%s', "
                     "info: '%s', rounding: '%s', min_max: %s, choices: %s, gui_radio: %s, "
                     "fixed: %s, group: %s)", section, title, datatype, default, info, rounding,
                     min_max, choices, gui_radio, fixed, group)

        choices = [] if not choices else choices

        assert (section is not None and
                title is not None and
                default is not None and
                info is not None), ("Default config items must have a section, title, defult and "
                                    "information text")
        if not self.defaults.get(section, None):
            raise ValueError(f"Section does not exist: {section}")
        assert datatype in (str, bool, float, int, list), (
            f"'datatype' must be one of str, bool, float or int: {section} - {title}")
        if datatype in (float, int) and (rounding is None or min_max is None):
            raise ValueError("'rounding' and 'min_max' must be set for numerical options")
        if isinstance(datatype, list) and not choices:
            raise ValueError("'choices' must be defined for list based configuration items")
        if choices != "colorchooser" and not isinstance(choices, (list, tuple)):
            raise ValueError("'choices' must be a list or tuple or 'colorchooser")

        info = self._expand_helptext(info, choices, default, datatype, min_max, fixed)
        self.defaults[section].items[title] = ConfigItem(default=default,
                                                         helptext=info,
                                                         datatype=datatype,
                                                         rounding=rounding or 0,
                                                         min_max=min_max,
                                                         choices=choices,
                                                         gui_radio=gui_radio,
                                                         fixed=fixed,
                                                         group=group)

    @classmethod
    def _expand_helptext(cls,
                         helptext: str,
                         choices: str | list[str],
                         default: ConfigValueType,
                         datatype: type,
                         min_max: tuple[int, int] | tuple[float, float] | None,
                         fixed: bool) -> str:
        """ Add extra helptext info from parameters """
        helptext += "\n"
        if not fixed:
            helptext += _("\nThis option can be updated for existing models.\n")
        if datatype == list:
            helptext += _("\nIf selecting multiple options then each option should be separated "
                          "by a space or a comma (e.g. item1, item2, item3)\n")
        if choices and choices != "colorchooser":
            helptext += _("\nChoose from: {}").format(choices)
        elif datatype == bool:
            helptext += _("\nChoose from: True, False")
        elif datatype == int:
            assert min_max is not None
            cmin, cmax = min_max
            helptext += _("\nSelect an integer between {} and {}").format(cmin, cmax)
        elif datatype == float:
            assert min_max is not None
            cmin, cmax = min_max
            helptext += _("\nSelect a decimal number between {} and {}").format(cmin, cmax)
        helptext += _("\n[Default: {}]").format(default)
        return helptext

    def _check_exists(self) -> bool:
        """ Check that a config file exists

        Returns
        -------
        bool
            ``True`` if the given configuration file exists
        """
        if not os.path.isfile(self.configfile):
            logger.debug("Config file does not exist: '%s'", self.configfile)
            return False
        logger.debug("Config file exists: '%s'", self.configfile)
        return True

    def _create_default(self) -> None:
        """ Generate a default config if it does not exist """
        logger.debug("Creating default Config")
        for name, section in self.defaults.items():
            logger.debug("Adding section: '%s')", name)
            self.insert_config_section(name, section.helptext)
            for item, opt in section.items.items():
                logger.debug("Adding option: (item: '%s', opt: '%s')", item, opt)
                self._insert_config_item(name, item, opt.default, opt)
        self.save_config()

    def insert_config_section(self,
                              section: str,
                              helptext: str,
                              config: ConfigParser | None = None) -> None:
        """ Insert a section into the config

        Parameters
        ----------
        section: str
            The section title to insert
        helptext: str
            The help text for the config section
        config: :class:`configparser.ConfigParser`, optional
            The config parser object to insert the section into. ``None`` to insert it into the
            default config. Default: ``None``
        """
        logger.debug("Inserting section: (section: '%s', helptext: '%s', config: '%s')",
                     section, helptext, config)
        config = self.config if config is None else config
        config.optionxform = str  # type:ignore
        helptext = self.format_help(helptext, is_section=True)
        config.add_section(section)
        config.set(section, helptext)
        logger.debug("Inserted section: '%s'", section)

    def _insert_config_item(self,
                            section: str,
                            item: str,
                            default: ConfigValueType,
                            option: ConfigItem,
                            config: ConfigParser | None = None) -> None:
        """ Insert an item into a config section

        Parameters
        ----------
        section: str
            The section to insert the item into
        item: str
            The name of the item to insert
        default: ConfigValueType
            The default value for the item
        option: :class:`ConfigItem`
            The configuration option to insert
        config: :class:`configparser.ConfigParser`, optional
            The config parser object to insert the section into. ``None`` to insert it into the
            default config. Default: ``None``
        """
        logger.debug("Inserting item: (section: '%s', item: '%s', default: '%s', helptext: '%s', "
                     "config: '%s')", section, item, default, option.helptext, config)
        config = self.config if config is None else config
        config.optionxform = str  # type:ignore
        helptext = option.helptext
        helptext = self.format_help(helptext, is_section=False)
        config.set(section, helptext)
        config.set(section, item, str(default))
        logger.debug("Inserted item: '%s'", item)

    @classmethod
    def format_help(cls, helptext: str, is_section: bool = False) -> str:
        """ Format comments for default ini file

        Parameters
        ----------
        helptext: str
            The help text to be formatted
        is_section: bool, optional
            ``True`` if the help text pertains to a section. ``False`` if it pertains to an item.
            Default: ``True``

        Returns
        -------
        str
            The formatted help text
        """
        logger.debug("Formatting help: (helptext: '%s', is_section: '%s')", helptext, is_section)
        formatted = ""
        for hlp in helptext.split("\n"):
            subsequent_indent = "\t\t" if hlp.startswith("\t") else ""
            hlp = f"\t- {hlp[1:].strip()}" if hlp.startswith("\t") else hlp
            formatted += textwrap.fill(hlp,
                                       100,
                                       tabsize=4,
                                       subsequent_indent=subsequent_indent) + "\n"
        helptext = '# {}'.format(formatted[:-1].replace("\n", "\n# "))  # Strip last newline
        if is_section:
            helptext = helptext.upper()
        else:
            helptext = f"\n{helptext}"
        logger.debug("formatted help: '%s'", helptext)
        return helptext

    def _load_config(self) -> None:
        """ Load values from config """
        logger.verbose("Loading config: '%s'", self.configfile)  # type:ignore[attr-defined]
        self.config.read(self.configfile, encoding="utf-8")

    def save_config(self) -> None:
        """ Save a config file """
        logger.info("Updating config at: '%s'", self.configfile)
        with open(self.configfile, "w", encoding="utf-8", errors="replace") as f_cfgfile:
            self.config.write(f_cfgfile)
        logger.debug("Updated config at: '%s'", self.configfile)

    def _validate_config(self) -> None:
        """ Check for options in default config against saved config
            and add/remove as appropriate """
        logger.debug("Validating config")
        if self._check_config_change():
            self._add_new_config_items()
        self._check_config_choices()
        logger.debug("Validated config")

    def _add_new_config_items(self) -> None:
        """ Add new items to the config file """
        logger.debug("Updating config")
        new_config = ConfigParser(allow_no_value=True)
        for section_name, section in self.defaults.items():
            self.insert_config_section(section_name, section.helptext, new_config)
            for item, opt in section.items.items():
                if section_name not in self.config.sections():
                    logger.debug("Adding new config section: '%s'", section_name)
                    opt_value = opt.default
                else:
                    opt_value = self.config[section_name].get(item, str(opt.default))
                self._insert_config_item(section_name,
                                         item,
                                         opt_value,
                                         opt,
                                         new_config)
        self.config = new_config
        self.config.optionxform = str  # type:ignore
        self.save_config()
        logger.debug("Updated config")

    def _check_config_choices(self) -> None:
        """ Check that config items are valid choices """
        logger.debug("Checking config choices")
        for section_name, section in self.defaults.items():
            for item, opt in section.items.items():
                if not opt.choices:
                    continue
                if opt.datatype == list:  # Multi-select items
                    opt_values = self._parse_list(section_name, item)
                    if not opt_values:  # No option selected
                        continue
                    if not all(val in opt.choices for val in opt_values):
                        invalid = [val for val in opt_values if val not in opt.choices]
                        valid = ", ".join(val for val in opt_values if val in opt.choices)
                        logger.warning("The option(s) %s are not valid selections for '%s': '%s'. "
                                       "setting to: '%s'", invalid, section_name, item, valid)
                        self.config.set(section_name, item, valid)
                else:  # Single-select items
                    if opt.choices == "colorchooser":
                        continue
                    opt_value = self.config.get(section_name, item)
                    if opt_value.lower() == "none" and any(choice.lower() == "none"
                                                           for choice in opt.choices):
                        continue
                    if opt_value not in opt.choices:
                        default = str(opt.default)
                        logger.warning("'%s' is not a valid config choice for '%s': '%s'. "
                                       "Defaulting to: '%s'",
                                       opt_value, section_name, item, default)
                        self.config.set(section_name, item, default)
        logger.debug("Checked config choices")

    def _check_config_change(self) -> bool:
        """ Check whether new default items have been added or removed from the config file
        compared to saved version

        Returns
        -------
        bool
            ``True`` if a config option has been added or removed
        """
        if set(self.config.sections()) != set(self.defaults.keys()):
            logger.debug("Default config has new section(s)")
            return True

        for section_name, section in self.defaults.items():
            opts = list(section.items)
            exists = [opt for opt in self.config[section_name].keys()
                      if not opt.startswith(("# ", "\n# "))]
            if set(exists) != set(opts):
                logger.debug("Default config has new item(s)")
                return True
        logger.debug("Default config has not changed")
        return False

    def _handle_config(self) -> None:
        """ Handle the config.

        Checks whether a config file exists for this section. If not then a default is created.

        Configuration choices are then loaded and validated
        """
        logger.debug("Handling config: (section: %s, configfile: '%s')",
                     self.section, self.configfile)
        if not self._check_exists():
            self._create_default()
        self._load_config()
        self._validate_config()
        logger.debug("Handled config")


def generate_configs() -> None:
    """ Generate config files if they don't exist.

    This script is run prior to anything being set up, so don't use logging
    Generates the default config files for plugins in the faceswap config folder
    """
    base_path = os.path.realpath(os.path.dirname(sys.argv[0]))
    plugins_path = os.path.join(base_path, "plugins")
    configs_path = os.path.join(base_path, "config")
    for dirpath, _, filenames in os.walk(plugins_path):
        if "_config.py" in filenames:
            section = os.path.split(dirpath)[-1]
            config_file = os.path.join(configs_path, f"{section}.ini")
            if not os.path.exists(config_file):
                mod = import_module(f"plugins.{section}._config")
                mod.Config(None)  # type:ignore[attr-defined]
