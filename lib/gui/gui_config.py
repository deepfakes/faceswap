#!/usr/bin/env python3
""" Default configurations for the GUI """

import logging
import os

from tkinter import font as tk_font
from matplotlib import font_manager

from lib.config import FaceswapConfig
from lib.config import ConfigItem
from lib.utils import get_module_objects, PROJECT_ROOT

logger = logging.getLogger(__name__)


class _Config(FaceswapConfig):
    """ Config File for GUI """
    def set_defaults(self, helptext="") -> None:
        """ Set the default values for config """
        logger.debug("Setting defaults")
        super().set_defaults(
            helptext="Faceswap GUI Options.\nConfigure the appearance and behaviour of the GUI")
        # Font choices cannot be added until tkinter has been launched
        logger.debug("Adding font list from tkinter")
        self.sections["global"].options["font"].choices = get_clean_fonts()


def get_commands() -> list[str]:
    """ Return commands formatted for GUI

    Returns
    -------
    list[str]
        A list of faceswap and tools commands that can be displayed in Faceswap's GUI
    """
    command_path = os.path.join(PROJECT_ROOT, "scripts")
    tools_path = os.path.join(PROJECT_ROOT, "tools")
    commands = [os.path.splitext(item)[0] for item in os.listdir(command_path)
                if os.path.splitext(item)[1] == ".py"
                and os.path.splitext(item)[0] not in ("gui", "fsmedia")
                and not os.path.splitext(item)[0].startswith("_")]
    tools = [os.path.splitext(item)[0] for item in os.listdir(tools_path)
             if os.path.splitext(item)[1] == ".py"
             and os.path.splitext(item)[0] not in ("gui", "cli")
             and not os.path.splitext(item)[0].startswith("_")]
    return commands + tools


def get_clean_fonts() -> list[str]:
    """ Return a sane list of fonts for the system that has both regular and bold variants.

    Pre-pend "default" to the beginning of the list.

    Returns
    -------
    list[str]:
        A list of valid fonts for the system
    """
    fmanager = font_manager.FontManager()
    fonts: dict[str, dict[str, bool]] = {}
    for fnt in fmanager.ttflist:
        if str(fnt.weight) in ("400", "normal", "regular"):
            fonts.setdefault(fnt.name, {})["regular"] = True
        if str(fnt.weight) in ("700", "bold"):
            fonts.setdefault(fnt.name, {})["bold"] = True
    valid_fonts = {key for key, val in fonts.items() if len(val) == 2}
    retval = sorted(list(valid_fonts.intersection(tk_font.families())))
    if not retval:
        # Return the font list with any @prefixed or non-Unicode characters stripped and default
        # prefixed
        logger.debug("No bold/regular fonts found. Running simple filter")
        retval = sorted([fnt for fnt in tk_font.families()
                         if not fnt.startswith("@") and not any(ord(c) > 127 for c in fnt)])
    return ["default"] + retval


fullscreen = ConfigItem(
    datatype=bool,
    default=False,
    group="startup",
    info="Start Faceswap maximized.")


tab = ConfigItem(
    datatype=str,
    default="extract",
    group="startup",
    info="Start Faceswap in this tab.",
    choices=get_commands())


options_panel_width = ConfigItem(
    datatype=int,
    default=30,
    group="layout",
    info="How wide the lefthand option panel is as a percentage of GUI width at "
         "startup.",
    min_max=(10, 90),
    rounding=1)


console_panel_height = ConfigItem(
    datatype=int,
    default=20,
    group="layout",
    info="How tall the bottom console panel is as a percentage of GUI height at "
         "startup.",
    min_max=(10, 90),
    rounding=1)


icon_size = ConfigItem(
    datatype=int,
    default=14,
    group="layout",
    info="Pixel size for icons. NB: Size is scaled by DPI.",
    min_max=(10, 20),
    rounding=1)


font = ConfigItem(
    datatype=str,
    default="default",
    group="font",
    info="Global font",
    choices=["default"])  # Cannot get tk fonts until tk is loaded, so real value populated later


font_size = ConfigItem(
    datatype=int,
    default=9,
    group="font",
    info="Global font size.",
    min_max=(6, 12),
    rounding=1)


autosave_last_session = ConfigItem(
    datatype=str,
    default="prompt",
    group="startup",
    info="Automatically save the current settings on close and reload on startup"
         "\n\tnever - Don't autosave session"
         "\n\tprompt - Prompt to reload last session on launch"
         "\n\talways - Always load last session on launch",
    choices=["never", "prompt", "always"],
    gui_radio=True)


timeout = ConfigItem(
    datatype=int,
    default=120,
    group="behaviour",
    info="Training can take some time to save and shutdown. Set the timeout "
         "in seconds before giving up and force quitting.",
    min_max=(10, 600),
    rounding=10)


auto_load_model_stats = ConfigItem(
    datatype=bool,
    default=True,
    group="behaviour",
    info="Auto load model statistics into the Analysis tab when selecting a model "
         "in Train or Convert tabs.")


def load_config(config_file: str | None = None) -> None:
    """ Load the GUI configuration .ini file

    Parameters
    ----------
    config_file : str | None, optional
        Path to a custom .ini configuration file to load. Default: ``None`` (use default
        configuration file)
    """
    _Config(configfile=config_file)


__all__ = get_module_objects(__name__)
