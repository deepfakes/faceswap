#!/usr/bin/env python3
""" Default configurations for models """

import logging
import sys
import os
from tkinter import font as tk_font
from matplotlib import font_manager

from lib.config import FaceswapConfig

logger = logging.getLogger(__name__)


class Config(FaceswapConfig):
    """ Config File for GUI """
    # pylint:disable=too-many-statements
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
        self.add_section(section,
                         "Faceswap GUI Options.\nConfigure the appearance and behaviour of "
                         "the GUI")
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
            section=section, title="icon_size", datatype=int, default=14,
            min_max=(10, 20), rounding=1, group="layout",
            info="Pixel size for icons. NB: Size is scaled by DPI.")
        self.add_item(
            section=section, title="font", datatype=str,
            choices=get_clean_fonts(),
            default="default", group="font", info="Global font")
        self.add_item(
            section=section, title="font_size", datatype=int, default=9,
            min_max=(6, 12), rounding=1, group="font",
            info="Global font size.")
        self.add_item(
            section=section, title="autosave_last_session", datatype=str, default="prompt",
            choices=["never", "prompt", "always"], group="startup", gui_radio=True,
            info="Automatically save the current settings on close and reload on startup"
                 "\n\tnever - Don't autosave session"
                 "\n\tprompt - Prompt to reload last session on launch"
                 "\n\talways - Always load last session on launch")
        self.add_item(
            section=section, title="timeout", datatype=int, default=120,
            min_max=(10, 600), rounding=10, group="behaviour",
            info="Training can take some time to save and shutdown. Set the timeout in seconds "
                 "before giving up and force quitting.")
        self.add_item(
            section=section, title="auto_load_model_stats", datatype=bool, default=True,
            group="behaviour",
            info="Auto load model statistics into the Analysis tab when selecting a model "
                 "in Train or Convert tabs.")


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
    """ Return a sane list of fonts for the system that has both regular and bold variants.

    Pre-pend "default" to the beginning of the list.

    Returns
    -------
    list:
        A list of valid fonts for the system
    """
    fmanager = font_manager.FontManager()
    fonts = {}
    for font in fmanager.ttflist:
        if str(font.weight) in ("400", "normal", "regular"):
            fonts.setdefault(font.name, {})["regular"] = True
        if str(font.weight) in ("700", "bold"):
            fonts.setdefault(font.name, {})["bold"] = True
    valid_fonts = {key for key, val in fonts.items() if len(val) == 2}
    retval = sorted(list(valid_fonts.intersection(tk_font.families())))
    if not retval:
        # Return the font list with any @prefixed or non-Unicode characters stripped and default
        # prefixed
        logger.debug("No bold/regular fonts found. Running simple filter")
        retval = sorted([fnt for fnt in tk_font.families()
                         if not fnt.startswith("@") and not any(ord(c) > 127 for c in fnt)])
    return ["default"] + retval
