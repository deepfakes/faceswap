#!/usr/bin/env python3
""" Default configurations for models """

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
        """ Override for plugin specific config defaults """
        raise NotImplementedError

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

    def add_item(self, section=None, title=None, default=None, info=None):
        """ Add a default item to a config section """
        if None in (section, title, default, info):
            raise ValueError("Default config items must have a section, "
                             "title, defult and  "
                             "information text")
        if not self.defaults.get(section, None):
            raise ValueError("Section does not exist: {}".format(section))
        self.defaults[section][title] = {"default": default,
                                         "helptext": info}

    def check_exists(self):
        """ Check that a config file exists """
        if not os.path.isfile(self.configfile):
            return False
        return True

    def create_default(self):
        """ Generate a default config if it does not exist """
        print("Generating default config at: {}".format(self.configfile))
        f_cfgfile = open(self.configfile, "w")
        for section, items in self.defaults.items():
            helptext = self.format_help(items["helptext"], is_section=True)
            self.config.add_section(section)
            self.config.set(section, helptext)
            for item, value in items.items():
                if item == "helptext":
                    continue
                helptext = self.format_help(value["helptext"])
                self.config.set(section, helptext)
                self.config.set(section, item, str(value["default"]))
        self.config.write(f_cfgfile)
        f_cfgfile.close()

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

    def handle_config(self):
        """ Handle the config """
        if not self.check_exists():
            self.create_default()
        self.load_config()
