#!/usr/bin/env python3
""" The default options for the faceswap Import Alignments plugin.

Defaults files should be named `<plugin_name>_defaults.py`

Any qualifying items placed into this file will automatically get added to the relevant config
.ini files within the faceswap/config folder and added to the relevant GUI settings page.

The following variable should be defined:

    Parameters
    ----------
    HELPTEXT: str
        A string describing what this plugin does

Further plugin configuration options are assigned using:
>>> <config_item> = ConfigItem(...)

where <config_item> is the name of the configuration option to be added (lower-case, alpha-numeric
+ underscore only) and ConfigItem(...) is the [`~lib.config.objects.ConfigItem`] data for the
option.

See the docstring/ReadtheDocs documentation required parameters for the ConfigItem object.
Items will be grouped together as per their `group` parameter, but otherwise will be processed in
the order that they are added to this module.
from lib.config import ConfigItem
"""
# pylint:disable=duplicate-code
from lib.config import ConfigItem


HELPTEXT = (
    "Import Detector options.\n"
    "Imports a detected face bounding box from an external .json file.\n"
    )


file_name = ConfigItem(
    datatype=str,
    default="import.json",
    group="settings",
    info="The import file should be stored in the same folder as the video (if extracting "
         "from a video file) or inside the folder of images (if importing from a folder of "
         "images)")

origin = ConfigItem(
    datatype=str,
    default="top-left",
    group="output",
    info="The origin (0, 0) location of the co-ordinates system used. "
         "\n\t top-left: The origin (0, 0) of the canvas is at the top left "
         "corner."
         "\n\t bottom-left: The origin (0, 0) of the canvas is at the bottom "
         "left corner."
         "\n\t top-right: The origin (0, 0) of the canvas is at the top right "
         "corner."
         "\n\t bottom-right: The origin (0, 0) of the canvas is at the bottom "
         "right corner.",
    choices=["top-left", "bottom-left", "top-right", "bottom-right"],
    gui_radio=True)
