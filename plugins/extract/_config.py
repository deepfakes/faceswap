#!/usr/bin/env python3
""" Default configurations for extract """

import logging
import os

from lib.config import FaceswapConfig

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class Config(FaceswapConfig):
    """ Config File for Extraction """

    def set_defaults(self):
        """ Set the default values for config """
        logger.debug("Setting defaults")
        self.set_globals()
        self._defaults_from_plugin(os.path.dirname(__file__))

    def set_globals(self):
        """
        Set the global options for extract
        """
        logger.debug("Setting global config")
        section = "global"
        self.add_section(title=section, info="Options that apply to all extraction plugins")
        self.add_item(
            section=section, title="allow_growth", datatype=bool, default=False, group="settings",
            info="[Nvidia Only]. Enable the Tensorflow GPU `allow_growth` configuration option. "
                 "This option prevents Tensorflow from allocating all of the GPU VRAM at launch "
                 "but can lead to higher VRAM fragmentation and slower performance. Should only "
                 "be enabled if you are having problems running extraction.")
