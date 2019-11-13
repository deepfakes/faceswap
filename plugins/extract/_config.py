#!/usr/bin/env python3
""" Default configurations for extract """

import logging
import os
import sys

from importlib import import_module
from lib.config import FaceswapConfig
from lib.utils import full_path_split

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class Config(FaceswapConfig):
    """ Config File for Extraction """

    def set_defaults(self):
        """ Set the default values for config """
        logger.debug("Setting defaults")
        self.set_globals()
        current_dir = os.path.dirname(__file__)
        for dirpath, _, filenames in os.walk(current_dir):
            default_files = [fname for fname in filenames if fname.endswith("_defaults.py")]
            if not default_files:
                continue
            base_path = os.path.dirname(os.path.realpath(sys.argv[0]))
            import_path = ".".join(full_path_split(dirpath.replace(base_path, ""))[1:])
            plugin_type = import_path.split(".")[-1]
            for filename in default_files:
                self.load_module(filename, import_path, plugin_type)

    def load_module(self, filename, module_path, plugin_type):
        """ Load the defaults module and add defaults """
        logger.debug("Adding defaults: (filename: %s, module_path: %s, plugin_type: %s",
                     filename, module_path, plugin_type)
        module = os.path.splitext(filename)[0]
        section = ".".join((plugin_type, module.replace("_defaults", "")))
        logger.debug("Importing defaults module: %s.%s", module_path, module)
        mod = import_module("{}.{}".format(module_path, module))
        self.add_section(title=section, info=mod._HELPTEXT)  # pylint:disable=protected-access
        for key, val in mod._DEFAULTS.items():  # pylint:disable=protected-access
            self.add_item(section=section, title=key, **val)
        logger.debug("Added defaults: %s", section)

    def set_globals(self):
        """
        Set the global options for extract
        """
        logger.debug("Setting global config")
        section = "global"
        self.add_section(title=section, info="Options that apply to all extraction plugins")
        self.add_item(
            section=section, title="allow_growth", datatype=bool, default=False,
            info="[Nvidia Only]. Enable the Tensorflow GPU `allow_growth` configuration option. "
                 "This option prevents Tensorflow from allocating all of the GPU VRAM at launch "
                 "but can lead to higher VRAM fragmentation and slower performance. Should only "
                 "be enabled if you are having problems running extraction.")
