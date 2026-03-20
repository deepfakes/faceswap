#!/usr/bin/env python3
""" The default options for the faceswap HRNet Alignments plugin.

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
    "HRNet Aligner options.\n"
    "Trained on 128k heavily augmented faces with full 360 degree rotation. Fast on GPU, slow on "
    "CPU."
    )


batch_size = ConfigItem(
    datatype=int,
    default=16,
    group="settings",
    info="The batch size to use. To a point, higher batch sizes equal better performance, "
         "but setting it too high can harm performance.",
    rounding=1,
    min_max=(1, 256))

dark_decoder = ConfigItem(
    datatype=bool,
    default=True,
    group="settings",
    info=("Use DARK decoder. A more refined method for obtaining landmarks from generated "
          "heatmaps. (Ref: https://arxiv.org/abs/1910.06278)."))
