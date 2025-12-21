#!/usr/bin/env python3
""" The default options for the external faceswap Import Alignments plugin.

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
    "Import Aligner options.\n"
    "Imports either 68 point 2D landmarks or an aligned bounding box from an external .json file."
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
    group="input",
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

four_point_centering = ConfigItem(
    datatype=str,
    default="head",
    group="input",
    info="4 point ROI landmarks only. The approximate centering for the location of the "
         "corner points to be imported. Default faceswap extracts are generated at 'head' "
         "centering, but it is possible to pass in ROI points at a tighter centering. "
         "Refer to https://github.com/deepfakes/faceswap/pull/1095 for a visual guide"
         "\n\t head: The ROI points represent a loose crop enclosing the whole head."
         "\n\t face: The ROI points represent a medium crop enclosing the face."
         "\n\t legacy: The ROI points represent a tight crop enclosing the central face "
         "area."
         "\n\t none: Only required if importing 4 point ROI landmarks back into faceswap "
         "having generated them from the 'alignments' tool 'export' job.",
    choices=["head", "face", "legacy", "none"],
    gui_radio=True)
