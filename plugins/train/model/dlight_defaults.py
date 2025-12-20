#!/usr/bin/env python3
""" The default options for the faceswap Dfaker Model plugin.

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


HELPTEXT = ("A lightweight, high resolution Dfaker variant "
            "(Adapted from https://github.com/dfaker/df)")


features = ConfigItem(
    datatype=str,
    default="best",
    group="settings",
    info="Higher settings will allow learning more features such as tatoos, piercing and "
         "wrinkles.\nStrongly affects VRAM usage.",
    choices=["lowmem", "fair", "best"],
    gui_radio=True,
    fixed=True)

details = ConfigItem(
    datatype=str,
    default="good",
    group="settings",
    info="Defines detail fidelity. Lower setting can appear 'rugged' while 'good' might take "
         "a longer time to train.\nAffects VRAM usage.",
    choices=["fast", "good"],
    gui_radio=True,
    fixed=True)

output_size = ConfigItem(
    datatype=int,
    default=256,
    group="settings",
    info="Output image resolution (in pixels).\nBe aware that larger resolution will increase "
         "VRAM requirements.\nNB: Must be either 128, 256, or 384.",
    rounding=128,
    min_max=(128, 384),
    fixed=True)
