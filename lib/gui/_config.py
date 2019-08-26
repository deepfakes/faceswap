#!/usr/bin/env python3
""" Default configurations for models """

import logging

from lib.config import FaceswapConfig

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class Config(FaceswapConfig):
    """ Config File for GUI """
    # pylint: disable=too-many-statements
    def set_defaults(self):
        """ Set the default values for config """
        logger.debug("Setting defaults")
        self.set_globals()

    def set_globals(self):
        """
        Set the global options for GUI
        """
        logger.debug("Setting global config")
        section = "global"
        self.add_section(title=section,
                         info="Faceswap GUI Options.\nNB: Faceswap will need to be restarted for "
                              "any changes to take effect.")
        self.add_item(
            section=section, title="fullscreen", datatype=bool, default=False, group="startup",
            info="Start Faceswap maximized.")
        self.add_item(
            section=section, title="options_panel_width", datatype=int, default=30,
            min_max=(10, 90), rounding=1, group="layout",
            info="How wide the lefthand option panel is as a percentage of GUI width at startup.")
        self.add_item(
            section=section, title="console_panel_height", datatype=int, default=20,
            min_max=(10, 90), rounding=1, group="layout",
            info="How tall the bottom console panel is as a percentage of GUI height at startup.")
