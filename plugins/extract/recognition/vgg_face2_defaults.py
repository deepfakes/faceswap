#!/usr/bin/env python3
""" The default options for the faceswap VGG Face2 recognition plugin.


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


HELPTEXT = (
    "VGG Face 2 identity recognition.\n"
    "A Keras port of the model trained for VGGFace2: A dataset for recognising faces across pose "
    "and age. (https://arxiv.org/abs/1710.08092)"
    )


batch_size = ConfigItem(
    datatype=int,
    default=16,
    group="settings",
    info="The batch size to use. To a point, higher batch sizes equal better performance, "
          "but setting it too high can harm performance.\n"
          "\n\tNvidia users: If the batchsize is set higher than the your GPU can "
          "accomodate then this will automatically be lowered.",
    rounding=1,
    min_max=(1, 64))

cpu = ConfigItem(
    datatype=bool,
    default=False,
    group="settings",
    info="VGG Face2 still runs fairly quickly on CPU on some setups. Enable "
         "CPU mode here to use the CPU for this plugin to save some VRAM at a speed cost.")
