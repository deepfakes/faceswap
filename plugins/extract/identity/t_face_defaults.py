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
# pylint:disable=duplicate-code
from lib.config import ConfigItem


HELPTEXT = (
    "Tencent TFace identity recognition.\n"
    "(https://github.com/Tencent/TFace)"
    )


batch_size = ConfigItem(
    datatype=int,
    default=16,
    group="settings",
    info="The batch size to use. To a point, higher batch sizes equal better performance, "
         "but setting it too high can harm performance.",
    rounding=1,
    min_max=(1, 256))

cpu = ConfigItem(
    datatype=bool,
    default=False,
    group="settings",
    info="The IR-50 backbone still runs fairly quickly on CPU on some setups. Enable "
         "CPU mode here to use the CPU for this plugin to save some VRAM at a speed cost.")

backbone = ConfigItem(
    datatype=str,
    default="ir-101",
    group="settings",
    info="The model backbone to use."
         "\n\tir-50 - InsightFace ResNet-50 (50 layers). Can run at a reasonable speed "
         r"on CPU. Reports 95%-96% accuracy."
         "\n\tir-101 - InsightFace ResNet-101 (100 layers). "
         r"Reports ~97% accuracy",
    choices=["ir-50", "ir-101"],
    gui_radio=True)
