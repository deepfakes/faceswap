#!/usr/bin/env python3
""" The default options for the faceswap Gif Writer plugin.

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
from lib.config import ConfigItem


HELPTEXT = "Options for outputting converted frames to an animated gif."


fps = ConfigItem(
    datatype=int,
    default=25,
    group="settings",
    info="Frames per Second.",
    rounding=1,
    min_max=(1, 60))

loop = ConfigItem(
    datatype=int,
    default=0,
    group="settings",
    info="The number of iterations. Set to 0 to loop indefinitely.",
    rounding=1,
    min_max=(0, 100))

palettesize = ConfigItem(
    datatype=str,
    default="256",
    group="settings",
    info="The number of colors to quantize the image to. Is rounded to the nearest power of "
         "two.",
    choices=["2", "4", "8", "16", "32", "64", "128", "256"])

subrectangles = ConfigItem(
    datatype=bool,
    default=False,
    group="settings",
    info="If True, will try and optimize the GIF by storing only the rectangular parts of "
         "each frame that change with respect to the previous.")
