#!/usr/bin/env python3
""" The default options for the faceswap BiSeNet Face Parsing plugin.

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
    "Custom (dummy) Mask options..\n"
    "The custom mask just fills a face patch with all 0's (masked out) or all 1's (masked in) for "
    "later manual editing. It does not use the GPU for creation."
    )


batch_size = ConfigItem(
    datatype=int,
    default=8,
    group="settings",
    info="The batch size to use. To a point, higher batch sizes equal better performance, "
         "but setting it too high can harm performance.",
    rounding=1,
    min_max=(1, 64))

centering = ConfigItem(
    datatype=str,
    group="settings",
    default="face",
    info="Whether to create a dummy mask with face or head centering.",
    choices=["face", "head"],
    gui_radio=True)

fill = ConfigItem(
    datatype=bool,
    default=False,
    group="settings",
    info="Whether the mask should be filled (True) in which case the custom mask will be "
         "created with the whole area masked in (i.e. you would need to manually edit out "
         "the background) or unfilled (False) in which case you would need to manually "
         "edit in the face.")
