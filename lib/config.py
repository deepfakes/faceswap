#!/usr/bin/env python3
""" Default configurations for faceswap
    Extends out configparser funcionality
    by checking for default config updates
    and returning data in it's correct format """

import os
import sys
from collections import OrderedDict
from configparser import ConfigParser


class FaceswapConfig():
    """ Config Items """
    def __init__(self):
        """ Init Configuration  """
        self.configfile = self.get_config_file()
        self.config = ConfigParser(allow_no_value=True)
        self.defaults = OrderedDict()
        self.config.optionxform = str

        self.set_defaults()
        self.handle_config()

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

    def get(self, section, option):
        """ Return a config item in it's correct format """
        datatype = self.defaults[section][option]["type"]
        if datatype == bool:
            func = self.config.getboolean
        elif datatype == int:
            func = self.config.getint
        elif datatype == float:
            func = self.config.getfloat
        else:
            func = self.config.get
        return func(section, option)

    def get_config_file(self):
        """ Return the config file from the calling folder """
        dirname = os.path.dirname(sys.modules[self.__module__].__file__)
        return os.path.join(dirname, "config.ini")

    def add_section(self, title=None, info=None):
        """ Add a default section to config file """
        if None in (title, info):
            raise ValueError("Default config sections must have a title and "
                             "information text")
        self.defaults[title] = OrderedDict()
        self.defaults[title]["helptext"] = info

    def add_item(self, section=None, title=None, datatype=str,
                 default=None, info=None):
        """ Add a default item to a config section """
        if None in (section, title, default, info):
            raise ValueError("Default config items must have a section, "
                             "title, defult and  "
                             "information text")
        if not self.defaults.get(section, None):
            raise ValueError("Section does not exist: {}".format(section))
        if datatype not in (str, bool, float, int):
            raise ValueError("Datatype must be one of str, bool, float or "
                             "int: {} - {}".format(section, title))
        self.defaults[section][title] = {"default": default,
                                         "helptext": info,
                                         "type": datatype}

    def check_exists(self):
        """ Check that a config file exists """
        if not os.path.isfile(self.configfile):
            return False
        return True

    def create_default(self):
        """ Generate a default config if it does not exist """
        for section, items in self.defaults.items():
            self.insert_config_section(section, items["helptext"])
            for item, opt in items.items():
                if item == "helptext":
                    continue
                self.insert_config_item(section,
                                        item,
                                        opt["default"],
                                        opt["helptext"])
        self.save_config()

    def insert_config_section(self, section, helptext, config=None):
        """ Insert a section into the config """
        config = self.config if config is None else config
        helptext = self.format_help(helptext, is_section=True)
        config.add_section(section)
        config.set(section, helptext)

    def insert_config_item(self, section, item, default, helptext,
                           config=None):
        """ Insert an item into a config section """
        config = self.config if config is None else config
        helptext = self.format_help(helptext, is_section=False)
        config.set(section, helptext)
        config.set(section, item, str(default))

    @staticmethod
    def format_help(helptext, is_section=False):
        """ Format comments for default ini file """
        helptext = '# {}'.format(helptext.replace("\n", "\n# "))
        if is_section:
            helptext = helptext.upper()
        else:
            helptext = "\n{}".format(helptext)
        return helptext

    def load_config(self):
        """ Load values from config """
        self.config.read(self.configfile)

    def save_config(self):
        """ Save a config file """
        print("Saving config at: {}".format(self.configfile))
        f_cfgfile = open(self.configfile, "w")
        self.config.write(f_cfgfile)
        f_cfgfile.close()

    def validate_config(self):
        """ Check for options in default config against saved config
            and add/remove as appropriate """
        if not self.check_config_change():
            return
        new_config = ConfigParser(allow_no_value=True)
        for section, items in self.defaults.items():
            self.insert_config_section(section, items["helptext"], new_config)
            for item, opt in items.items():
                if item == "helptext":
                    continue
                if section not in self.config.sections():
                    opt_value = opt["default"]
                else:
                    opt_value = self.config[section].get(item, opt["default"])
                self.insert_config_item(section,
                                        item,
                                        opt_value,
                                        opt["helptext"],
                                        new_config)
        self.config = new_config
        self.save_config()

    def check_config_change(self):
        """ Check whether new default items have been added or removed
            from the config file compared to saved version """
        if set(self.config.sections()) != set(self.defaults.keys()):
            return True

        for section, items in self.defaults.items():
            opts = [opt for opt in items.keys() if opt != "helptext"]
            if set(self.config.options(section)) != set(opts):
                return True
        return False

    def handle_config(self):
        """ Handle the config """
        if not self.check_exists():
            self.create_default()
        self.load_config()
        self.validate_config()
