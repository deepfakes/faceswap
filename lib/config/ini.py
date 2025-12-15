#! /usr/env/bin/python3
""" Handles interfacing between Faceswap Configs and ConfigParser .ini files """
from __future__ import annotations

import logging
import os
import textwrap
import typing as T

from configparser import ConfigParser

from lib.logger import parse_class_init
from lib.utils import get_module_objects, PROJECT_ROOT

if T.TYPE_CHECKING:
    from .objects import ConfigSection, ConfigValueType

logger = logging.getLogger(__name__)


class ConfigFile():
    """ Handles the interfacing between saved faceswap .ini configs and internal Config objects

    Parameters
    ----------
    plugin_group : str
        The plugin group that is requesting a config file
    ini_path : str | None, optional
        Optional path to a .ini config file. ``None`` for default location. Default: ``None``
    """
    def __init__(self, plugin_group: str, ini_path: str | None = None) -> None:
        parse_class_init(locals())
        self._plugin_group = plugin_group
        self._file_path = self._get_config_path(ini_path)
        self._parser = self._get_new_configparser()
        if self._exists:  # Load or create new
            self.load()

    @property
    def _exists(self) -> bool:
        """ bool : ``True`` if the config.ini file exists """
        return os.path.isfile(self._file_path)

    def _get_config_path(self, ini_path: str | None) -> str:
        """ Return the path to the config file from the calling folder or the provided file

        Parameters
        ----------
        ini_path : str | None
            Path to a config ini file. ``None`` for default location.

        Returns
        -------
        str
            The full path to the configuration file
        """
        if ini_path is not None:
            if not os.path.isfile(ini_path):
                err = f"Config file does not exist at: {ini_path}"
                logger.error(err)
                raise ValueError(err)
            return ini_path

        retval = os.path.join(PROJECT_ROOT, "config", f"{self._plugin_group}.ini")
        logger.debug("[%s] Config File location: '%s'", os.path.basename(retval), retval)
        return retval

    def _get_new_configparser(self) -> ConfigParser:
        """ Obtain a fresh ConfigParser object and set it to case-sensitive

        Returns
        -------
        :class:`configparser.ConfigParser`
            A new ConfigParser object set to case-sensitive
        """
        retval = ConfigParser(allow_no_value=True)
        retval.optionxform = str  # type:ignore[assignment,method-assign]
        return retval

    # I/O
    def load(self) -> None:
        """ Load values from the saved config ini file into our Config object """
        logger.verbose("[%s] Loading config: '%s'",  # type:ignore[attr-defined]
                       self._plugin_group, self._file_path)
        self._parser.read(self._file_path, encoding="utf-8")

    def save(self) -> None:
        """ Save a config file """
        logger.debug("[%s] %s config: '%s'",
                     self._plugin_group, "Updating" if self._exists else "Saving", self._file_path)
        # TODO in python >= 3.14 this will error when there are delimiters in the comments
        with open(self._file_path, "w", encoding="utf-8", errors="replace") as f_cfgfile:
            self._parser.write(f_cfgfile)
        logger.info("[%s] Saved config: '%s'", self._plugin_group, self._file_path)

    # .ini vs Faceswap Config checking
    def _sections_synced(self, app_config: dict[str, ConfigSection]) -> bool:
        """ Validate that all of the sections within the application config match with all of the
        sections in the ini file

        Parameters
        ----------
        app_config : dict[str, :class:`ConfigSection`]
            The latest configuration settings from the application. Section name is key

        Returns
        -------
        bool
            ``True`` if application sections and saved ini sections match
        """
        given_sections = set(app_config)
        loaded_sections = set(self._parser.sections())
        retval = given_sections == loaded_sections
        if not retval:
            logger.debug("[%s] Config sections are not synced: (app: %s, ini: %s)",
                         self._plugin_group, sorted(given_sections), sorted(loaded_sections))
        return retval

    def _options_synced(self, app_config: dict[str, ConfigSection]) -> bool:
        """ Validate that all of the option names within the application config match with all of
        the option names in the ini file

        Note
        ----
        As we need to write a new config anyway, we return on the first change found

        Parameters
        ----------
        app_config : dict[str, :class:`ConfigSection`]
            The latest configuration settings from the application. Section name is key

        Returns
        -------
        bool
            ``True`` if application option names match with saved ini option names
        """
        for name, section in app_config.items():
            given_opts = set(opt for opt in section.options)
            loaded_opts = set(self._parser[name].keys())
            if given_opts != loaded_opts:
                logger.debug("[%s:%s] Config options are not synced: (app: %s, ini: %s)",
                             self._plugin_group, name, sorted(given_opts), sorted(loaded_opts))
                return False
        return True

    def _values_synced(self, app_section: ConfigSection, section: str) -> bool:
        """ Validate that all of the option values within the application config match with all of
        the option values in the ini file

        Parameters
        ----------
        app_section : :class:`ConfigSection`
            The latest configuration settings from the application for the given section
        section : str
            The section name to check the option values for

        Returns
        -------
        bool
            ``True`` if application option values match with saved ini option values
        """
        # Need to also pull in keys as False is omitted from the set with just values which can
        # cause edge-case false negatives
        given_vals = set((k, v.ini_value) for k, v in app_section.options.items())
        loaded_vals = set((k, v) for k, v in self._parser[section].items())
        retval = given_vals == loaded_vals
        if not retval:
            logger.debug("[%s:%s] Config values are not synced: (app: %s, ini: %s)",
                         self._plugin_group, section, sorted(given_vals), sorted(loaded_vals))
        return retval

    def _is_synced_structure(self, app_config: dict[str, ConfigSection]) -> bool:
        """ Validate that all the given sections and option names within the application config
        match with their corresponding items in the save .ini file

        Parameters
        ----------
        app_config: dict[str, :class:`ConfigSection`]
            The latest configuration settings from the application. Section name is key

        Returns
        -------
        bool
            ``True`` if the app config and saved ini config structure match
        """
        if not self._sections_synced(app_config):
            return False
        if not self._options_synced(app_config):
            return False

        logger.debug("[%s] Configs are synced", self._plugin_group)
        return True

    # .ini file insertion
    def format_help(self, helptext: str, is_section: bool = False) -> str:
        """ Format comments for insertion into a config ini file

        Parameters
        ----------
        helptext : str
            The help text to be formatted
        is_section : bool, optional
            ``True`` if the help text pertains to a section. ``False`` if it pertains to an option.
            Default: ``True``

        Returns
        -------
        str
            The formatted help text
        """
        logger.debug("[%s] Formatting help: (helptext: '%s', is_section: '%s')",
                     self._plugin_group, helptext, is_section)
        formatted = ""
        for hlp in helptext.split("\n"):
            subsequent_indent = "\t\t" if hlp.startswith("\t") else ""
            hlp = f"\t- {hlp[1:].strip()}" if hlp.startswith("\t") else hlp
            formatted += textwrap.fill(hlp,
                                       100,
                                       tabsize=4,
                                       subsequent_indent=subsequent_indent) + "\n"
        helptext = '# {}'.format(formatted[:-1].replace("\n", "\n# "))  # Strip last newline
        helptext = helptext.upper() if is_section else f"\n{helptext}"
        return helptext

    def _insert_section(self, section: str, helptext: str, config: ConfigParser) -> None:
        """ Insert a section into the config

        Parameters
        ----------
        section : str
            The section title to insert
        helptext : str
            The help text for the config section
        config : :class:`configparser.ConfigParser`
            The config parser object to insert the section into.
        """
        logger.debug("[%s:%s] Inserting section: (helptext: '%s', config: '%s')",
                     self._plugin_group, section, helptext, config)
        helptext = self.format_help(helptext, is_section=True)
        config.add_section(section)
        config.set(section, helptext)

    def _insert_option(self,
                       section: str,
                       name: str,
                       helptext: str,
                       value: str,
                       config: ConfigParser) -> None:
        """ Insert an option into a config section

        Parameters
        ----------
        section : str
            The section to insert the option into
        name : str
            The name of the option to insert
        helptext : str
            The help text for the option
        value : str
            The value for the option
        config : :class:`configparser.ConfigParser`
            The config parser object to insert the option into
        """
        logger.debug(
            "[%s:%s] Inserting option: (name: '%s', helptext: %s, value: '%s', config: '%s')",
            self._plugin_group, section, name, helptext, value, config)
        helptext = self.format_help(helptext, is_section=False)
        config.set(section, helptext)
        config.set(section, name, value)

    def _sync_from_app(self, app_config: dict[str, ConfigSection]) -> None:
        """ Update the saved config.ini file from the values stored in the application config

        Existing options keep their saved values as per the .ini files. New options are added with
        their application defined default value. Options in the .ini file not in application
        provided config are removed.

        Note
        ----
        A new configuration object is created as comments are stripped from the loaded ini files.

        Parameters
        ----------
        app_config: dict[str, :class:`ConfigSection`]
            The latest configuration settings from the application. Section name is key
        """
        logger.debug("[%s] Syncing from app", self._plugin_group)
        parser = self._get_new_configparser() if self._exists else self._parser
        for section_name, section in app_config.items():
            self._insert_section(section_name, section.helptext, parser)
            for name, opt in section.options.items():

                value = self._parser.get(section_name, name, fallback=None)
                if value is None:
                    value = opt.ini_value
                    logger.debug(
                        "[%s:%s] Setting default value for non-existent config option '%s': '%s'",
                        self._plugin_group, section_name, name, value)

                self._insert_option(section_name, name, opt.helptext, value, parser)

        if parser != self._parser:
            self._parser = parser

        self.save()

    # .ini extraction
    def _get_converted_value(self, section: str, option: str, datatype: type) -> ConfigValueType:
        """ Return a config item from the .ini file in it's correct type.

        Parameters
        ----------
        section : str
            The configuration section to obtain the config option for
        option : str
            The configuration option to obtain the converted value for
        datatype : type
            The type to return the value as

        Returns
        -------
        bool | int | float | list[str] | str
            The selected configuration option in the correct data format
        """
        logger.debug("[%s:%s] Getting config item: (option: '%s', datatype: %s)",
                     self._plugin_group, section, option, datatype)

        assert datatype in (bool, int, float, str, list), (
            f"Expected (bool, int, float, str, list). Got {datatype}")

        retval: ConfigValueType
        if datatype == bool:
            retval = self._parser.getboolean(section, option)
        elif datatype == int:
            retval = self._parser.getint(section, option)
        elif datatype == float:
            retval = self._parser.getfloat(section, option)
        else:
            retval = self._parser.get(section, option)

        logger.debug("[%s:%s] Got config item: (value: %s, type: %s)",
                     self._plugin_group, section, retval, type(retval))
        return retval

    def _sync_to_app(self, app_config: dict[str, ConfigSection]) -> None:
        """ Update the values in the application config to those loaded from the saved config.ini.

        Parameters
        ----------
        app_config: dict[str, :class:`ConfigSection`]
            The latest configuration settings from the application. Section name is key
        """
        logger.debug("[%s] Syncing to app", self._plugin_group)
        for section_name, section in app_config.items():
            if self._values_synced(section, section_name):
                continue
            for opt_name, opt in section.options.items():
                if section_name not in self._parser or opt_name not in self._parser[section_name]:
                    logger.debug("[%s:%s] Skipping new option: '%s'",
                                 self._plugin_group, section_name, opt_name)
                    continue

                ini_opt = self._parser[section_name][opt_name]
                if opt.ini_value != ini_opt:
                    logger.debug("[%s:%s] Updating '%s' from '%s' to '%s'",
                                 self._plugin_group, section_name,
                                 opt_name, ini_opt, opt.ini_value)
                    opt.set(self._get_converted_value(section_name, opt_name, opt.datatype))

    # .ini insertion and extraction
    def on_load(self, app_config: dict[str, ConfigSection]) -> None:
        """ Check whether there has been any change between the current application config and
        the loaded ini config. If so, update the relevant object(s) appropriately. This check will
        also create new config.ini files if they do not pre-exist

        Parameters
        ----------
        app_config : dict[str, :class:`ConfigSection`]
            The latest configuration settings from the application. Section name is key
        """
        if not self._exists:
            logger.debug("[%s] Creating new ini file", self._plugin_group)
            self._sync_from_app(app_config)

        if not self._is_synced_structure(app_config):
            self._sync_from_app(app_config)

        self._sync_to_app(app_config)

    def update_from_app(self, app_config: dict[str, ConfigSection]) -> None:
        """ Update the config.ini file to those values that are currently in Faceswap's app
        config

        Parameters
        ----------
        app_config : dict[str, :class:`ConfigSection`]
            The latest configuration settings from the application. Section name is key
        """
        logger.debug("[%s] Updating saved config", self._plugin_group)
        parser = self._get_new_configparser() if self._exists else self._parser
        for section_name, section in app_config.items():
            self._insert_section(section_name, section.helptext, parser)
            for name, opt in section.options.items():
                self._insert_option(section_name, name, opt.helptext, opt.ini_value, parser)
        if parser != self._parser:
            self._parser = parser
        self.save()


__all__ = get_module_objects(__name__)
