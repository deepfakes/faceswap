#!/usr/bin/env python3
""" Default configurations for faceswap.
    Extends out :class:`configparser.ConfigParser` functionality by checking for default
    configuration updates and returning data in it's correct format """

import logging
import os
import sys
from collections import OrderedDict
from configparser import ConfigParser
from importlib import import_module

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class FaceswapConfig():
    """ Config Items """
    def __init__(self, section, configfile=None):
        """ Init Configuration  """
        logger.debug("Initializing: %s", self.__class__.__name__)
        self.configfile = self.get_config_file(configfile)
        self.config = ConfigParser(allow_no_value=True)
        self.defaults = OrderedDict()
        self.config.optionxform = str
        self.section = section

        self.set_defaults()
        self.handle_config()
        logger.debug("Initialized: %s", self.__class__.__name__)

    @property
    def changeable_items(self):
        """ Training only.
            Return a dict of config items with their set values for items
            that can be altered after the model has been created """
        retval = dict()
        for sect in ("global", self.section):
            if sect not in self.defaults:
                continue
            for key, val in self.defaults[sect].items():
                if key == "helptext" or val["fixed"]:
                    continue
                retval[key] = self.get(sect, key)
        logger.debug("Alterable for existing models: %s", retval)
        return retval

    def set_defaults(self):
        """ Override for plugin specific config defaults

            Should be a series of self.add_section() and self.add_item() calls

            e.g:

            section = "sect_1"
            self.add_section(title=section,
                         info="Section 1 Information")

            self.add_item(section=section,
                          title="option_1",
                          datatype=bool,
                          default=False,
                          info="sect_1 option_1 information")
        """
        raise NotImplementedError

    @property
    def config_dict(self):
        """ Collate global options and requested section into a dictionary with the correct
        data types """
        conf = dict()
        sections = [sect for sect in self.config.sections() if sect.startswith("global")]
        sections.append(self.section)
        for sect in sections:
            if sect not in self.config.sections():
                continue
            for key in self.config[sect]:
                if key.startswith(("#", "\n")):  # Skip comments
                    continue
                conf[key] = self.get(sect, key)
        return conf

    def get(self, section, option):
        """ Return a config item in it's correct format """
        logger.debug("Getting config item: (section: '%s', option: '%s')", section, option)
        datatype = self.defaults[section][option]["type"]
        if datatype == bool:
            func = self.config.getboolean
        elif datatype == int:
            func = self.config.getint
        elif datatype == float:
            func = self.config.getfloat
        else:
            func = self.config.get
        retval = func(section, option)
        if isinstance(retval, str) and retval.lower() == "none":
            retval = None
        logger.debug("Returning item: (type: %s, value: %s)", datatype, retval)
        return retval

    def get_config_file(self, configfile):
        """ Return the config file from the calling folder or the provided file """
        if configfile is not None:
            if not os.path.isfile(configfile):
                err = "Config file does not exist at: {}".format(configfile)
                logger.error(err)
                raise ValueError(err)
            return configfile
        dirname = os.path.dirname(sys.modules[self.__module__].__file__)
        folder, fname = os.path.split(dirname)
        retval = os.path.join(os.path.dirname(folder), "config", "{}.ini".format(fname))
        logger.debug("Config File location: '%s'", retval)
        return retval

    def add_section(self, title=None, info=None):
        """ Add a default section to config file """
        logger.debug("Add section: (title: '%s', info: '%s')", title, info)
        if None in (title, info):
            raise ValueError("Default config sections must have a title and "
                             "information text")
        self.defaults[title] = OrderedDict()
        self.defaults[title]["helptext"] = info

    def add_item(self, section=None, title=None, datatype=str, default=None, info=None,
                 rounding=None, min_max=None, choices=None, gui_radio=False, fixed=True,
                 group=None):
        """ Add a default item to a config section

            For int or float values, rounding and min_max must be set
            This is for the slider in the GUI. The min/max values are not enforced:
            rounding:   sets the decimal places for floats or the step interval for ints.
            min_max:    tuple of min and max accepted values

            For str values choices can be set to validate input and create a combo box
            in the GUI

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

        choices = list() if not choices else choices

        if None in (section, title, default, info):
            raise ValueError("Default config items must have a section, "
                             "title, defult and  "
                             "information text")
        if not self.defaults.get(section, None):
            raise ValueError("Section does not exist: {}".format(section))
        if datatype not in (str, bool, float, int):
            raise ValueError("'datatype' must be one of str, bool, float or "
                             "int: {} - {}".format(section, title))
        if datatype in (float, int) and (rounding is None or min_max is None):
            raise ValueError("'rounding' and 'min_max' must be set for numerical options")
        if not isinstance(choices, (list, tuple)):
            raise ValueError("'choices' must be a list or tuple")

        info = self.expand_helptext(info, choices, default, datatype, min_max, fixed)
        self.defaults[section][title] = {"default": default,
                                         "helptext": info,
                                         "type": datatype,
                                         "rounding": rounding,
                                         "min_max": min_max,
                                         "choices": choices,
                                         "gui_radio": gui_radio,
                                         "fixed": fixed,
                                         "group": group}

    @staticmethod
    def expand_helptext(helptext, choices, default, datatype, min_max, fixed):
        """ Add extra helptext info from parameters """
        helptext += "\n"
        if not fixed:
            helptext += "\nThis option can be updated for existing models."
        if choices:
            helptext += "\nChoose from: {}".format(choices)
        elif datatype == bool:
            helptext += "\nChoose from: True, False"
        elif datatype == int:
            cmin, cmax = min_max
            helptext += "\nSelect an integer between {} and {}".format(cmin, cmax)
        elif datatype == float:
            cmin, cmax = min_max
            helptext += "\nSelect a decimal number between {} and {}".format(cmin, cmax)
        helptext += "\n[Default: {}]".format(default)
        return helptext

    def check_exists(self):
        """ Check that a config file exists """
        if not os.path.isfile(self.configfile):
            logger.debug("Config file does not exist: '%s'", self.configfile)
            return False
        logger.debug("Config file exists: '%s'", self.configfile)
        return True

    def create_default(self):
        """ Generate a default config if it does not exist """
        logger.debug("Creating default Config")
        for section, items in self.defaults.items():
            logger.debug("Adding section: '%s')", section)
            self.insert_config_section(section, items["helptext"])
            for item, opt in items.items():
                logger.debug("Adding option: (item: '%s', opt: '%s'", item, opt)
                if item == "helptext":
                    continue
                self.insert_config_item(section,
                                        item,
                                        opt["default"],
                                        opt)
        self.save_config()

    def insert_config_section(self, section, helptext, config=None):
        """ Insert a section into the config """
        logger.debug("Inserting section: (section: '%s', helptext: '%s', config: '%s')",
                     section, helptext, config)
        config = self.config if config is None else config
        helptext = self.format_help(helptext, is_section=True)
        config.add_section(section)
        config.set(section, helptext)
        logger.debug("Inserted section: '%s'", section)

    def insert_config_item(self, section, item, default, option,
                           config=None):
        """ Insert an item into a config section """
        logger.debug("Inserting item: (section: '%s', item: '%s', default: '%s', helptext: '%s', "
                     "config: '%s')", section, item, default, option["helptext"], config)
        config = self.config if config is None else config
        helptext = option["helptext"]
        helptext = self.format_help(helptext, is_section=False)
        config.set(section, helptext)
        config.set(section, item, str(default))
        logger.debug("Inserted item: '%s'", item)

    @staticmethod
    def format_help(helptext, is_section=False):
        """ Format comments for default ini file """
        logger.debug("Formatting help: (helptext: '%s', is_section: '%s')", helptext, is_section)
        helptext = '# {}'.format(helptext.replace("\n", "\n# "))
        if is_section:
            helptext = helptext.upper()
        else:
            helptext = "\n{}".format(helptext)
        logger.debug("formatted help: '%s'", helptext)
        return helptext

    def load_config(self):
        """ Load values from config """
        logger.verbose("Loading config: '%s'", self.configfile)
        self.config.read(self.configfile)

    def save_config(self):
        """ Save a config file """
        logger.info("Updating config at: '%s'", self.configfile)
        f_cfgfile = open(self.configfile, "w")
        self.config.write(f_cfgfile)
        f_cfgfile.close()
        logger.debug("Updated config at: '%s'", self.configfile)

    def validate_config(self):
        """ Check for options in default config against saved config
            and add/remove as appropriate """
        logger.debug("Validating config")
        if self.check_config_change():
            self.add_new_config_items()
        self.check_config_choices()
        logger.debug("Validated config")

    def add_new_config_items(self):
        """ Add new items to the config file """
        logger.debug("Updating config")
        new_config = ConfigParser(allow_no_value=True)
        for section, items in self.defaults.items():
            self.insert_config_section(section, items["helptext"], new_config)
            for item, opt in items.items():
                if item == "helptext":
                    continue
                if section not in self.config.sections():
                    logger.debug("Adding new config section: '%s'", section)
                    opt_value = opt["default"]
                else:
                    opt_value = self.config[section].get(item, opt["default"])
                self.insert_config_item(section,
                                        item,
                                        opt_value,
                                        opt,
                                        new_config)
        self.config = new_config
        self.config.optionxform = str
        self.save_config()
        logger.debug("Updated config")

    def check_config_choices(self):
        """ Check that config items are valid choices """
        logger.debug("Checking config choices")
        for section, items in self.defaults.items():
            for item, opt in items.items():
                if item == "helptext" or not opt["choices"]:
                    continue
                opt_value = self.config.get(section, item)
                if opt_value.lower() == "none" and any(choice.lower() == "none"
                                                       for choice in opt["choices"]):
                    continue
                if opt_value not in opt["choices"]:
                    default = str(opt["default"])
                    logger.warning("'%s' is not a valid config choice for '%s': '%s'. Defaulting "
                                   "to: '%s'", opt_value, section, item, default)
                    self.config.set(section, item, default)
        logger.debug("Checked config choices")

    def check_config_change(self):
        """ Check whether new default items have been added or removed
            from the config file compared to saved version """
        if set(self.config.sections()) != set(self.defaults.keys()):
            logger.debug("Default config has new section(s)")
            return True

        for section, items in self.defaults.items():
            opts = [opt for opt in items.keys() if opt != "helptext"]
            exists = [opt for opt in self.config[section].keys()
                      if not opt.startswith(("# ", "\n# "))]
            if set(exists) != set(opts):
                logger.debug("Default config has new item(s)")
                return True
        logger.debug("Default config has not changed")
        return False

    def handle_config(self):
        """ Handle the config """
        logger.debug("Handling config")
        if not self.check_exists():
            self.create_default()
        self.load_config()
        self.validate_config()
        logger.debug("Handled config")


def generate_configs():
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
            config_file = os.path.join(configs_path, "{}.ini".format(section))
            if not os.path.exists(config_file):
                mod = import_module("plugins.{}.{}".format(section, "_config"))
                mod.Config(None)
