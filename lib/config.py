#!/usr/bin/env python3
""" Default configurations for faceswap
    Extends out configparser funcionality
    by checking for default config updates
    and returning data in it's correct format """

import logging
import os
import sys
from collections import OrderedDict
from configparser import ConfigParser

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class FaceswapConfig():
    """ Config Items """
    def __init__(self, section):
        """ Init Configuration  """
        logger.debug("Initializing: %s", self.__class__.__name__)
        self.configfile = self.get_config_file()
        self.config = ConfigParser(allow_no_value=True)
        self.defaults = OrderedDict()
        self.config.optionxform = str
        self.section = section

        self.set_defaults()
        self.handle_config()
        logger.debug("Initialized: %s", self.__class__.__name__)

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
        """ Collate global options and requested section into a dictionary
            with the correct datatypes """
        conf = dict()
        for sect in ("global", self.section):
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
        logger.debug("Returning item: (type: %s, value: %s)", datatype, retval)
        return retval

    def get_config_file(self):
        """ Return the config file from the calling folder """
        dirname = os.path.dirname(sys.modules[self.__module__].__file__)
        retval = os.path.join(dirname, "config.ini")
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

    def add_item(self, section=None, title=None, datatype=str,
                 default=None, info=None, rounding=None, min_max=None, choices=None):
        """ Add a default item to a config section

            For int or float values, rounding and min_max must be set
            This is for the slider in the GUI. The min/max values are not enforced:
            rounding:   sets the decimal places for floats or the step interval for ints.
            min_max:    tuple of min and max accepted values

            For str values choices can be set to validate input and create a combo box
            in the GUI

        """
        logger.debug("Add item: (section: '%s', title: '%s', datatype: '%s', default: '%s', "
                     "info: '%s', rounding: '%s', min_max: %s, choices: %s)",
                     section, title, datatype, default, info, rounding, min_max, choices)

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
        self.defaults[section][title] = {"default": default,
                                         "helptext": info,
                                         "type": datatype,
                                         "rounding": rounding,
                                         "min_max": min_max,
                                         "choices": choices}

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
        helptext += self.set_helptext_choices(option)
        helptext += "\n[Default: {}]".format(default)
        helptext = self.format_help(helptext, is_section=False)
        config.set(section, helptext)
        config.set(section, item, str(default))
        logger.debug("Inserted item: '%s'", item)

    @staticmethod
    def set_helptext_choices(option):
        """ Set the helptext choices """
        choices = ""
        if option["choices"]:
            choices = "\nChoose from: {}".format(option["choices"])
        elif option["type"] == bool:
            choices = "\nChoose from: True, False"
        elif option["type"] == int:
            cmin, cmax = option["min_max"]
            choices = "\nSelect an integer between {} and {}".format(cmin, cmax)
        elif option["type"] == float:
            cmin, cmax = option["min_max"]
            choices = "\nSelect a decimal number between {} and {}".format(cmin, cmax)
        return choices

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
        logger.info("Loading config: '%s'", self.configfile)
        self.config.read(self.configfile)

    def save_config(self):
        """ Save a config file """
        logger.info("Updating config at: '%s'", self.configfile)
        f_cfgfile = open(self.configfile, "w")
        self.config.write(f_cfgfile)
        f_cfgfile.close()

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
