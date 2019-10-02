#!/usr/bin/env python3
""" Default configurations for models """

import logging
import sys
import os
from tkinter import font

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
            section=section, title="tab", datatype=str, default="extract", group="startup",
            choices=get_commands(),
            info="Start Faceswap in this tab.")
        self.add_item(
            section=section, title="options_panel_width", datatype=int, default=30,
            min_max=(10, 90), rounding=1, group="layout",
            info="How wide the lefthand option panel is as a percentage of GUI width at startup.")
        self.add_item(
            section=section, title="console_panel_height", datatype=int, default=20,
            min_max=(10, 90), rounding=1, group="layout",
            info="How tall the bottom console panel is as a percentage of GUI height at startup.")
        self.add_item(
            section=section, title="font", datatype=str,
            choices=get_clean_fonts(),
            default="default", group="font", info="Global font")
        self.add_item(
            section=section, title="font_size", datatype=int, default=9,
            min_max=(6, 12), rounding=1, group="font",
            info="Global font size.")


def get_commands():
    """ Return commands formatted for GUI """
    root_path = os.path.abspath(os.path.dirname(sys.argv[0]))
    command_path = os.path.join(root_path, "scripts")
    tools_path = os.path.join(root_path, "tools")
    commands = [os.path.splitext(item)[0] for item in os.listdir(command_path)
                if os.path.splitext(item)[1] == ".py"
                and os.path.splitext(item)[0] not in ("gui", "fsmedia")
                and not os.path.splitext(item)[0].startswith("_")]
    tools = [os.path.splitext(item)[0] for item in os.listdir(tools_path)
             if os.path.splitext(item)[1] == ".py"
             and os.path.splitext(item)[0] not in ("gui", "cli")
             and not os.path.splitext(item)[0].startswith("_")]
    return commands + tools


def get_clean_fonts():
    """ Return the font list with any @prefixed or non-unicode characters stripped
        and default prefixed """
    cleaned_fonts = sorted([fnt for fnt in font.families()
                            if not fnt.startswith("@") and not any([ord(c) > 127 for c in fnt])])
    return ["default"] + cleaned_fonts
