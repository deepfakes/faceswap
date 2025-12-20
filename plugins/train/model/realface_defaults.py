#!/usr/bin/env python3
""" The default options for the faceswap Realface Model plugin.

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
    "An extra detailed variant of Original model.\n"
    "Incorporates ideas from Bryanlyon and inspiration from the Villain model.\n"
    "Requires about 6GB-8GB of VRAM (batchsize 8-16).\n"
)


input_size = ConfigItem(
    datatype=int,
    default=64,
    group="size",
    info="Resolution (in pixels) of the input image to train on.\n"
         "BE AWARE Larger resolution will dramatically increase VRAM requirements.\n"
         "Higher resolutions may increase prediction accuracy, but does not effect the "
         "resulting output size.\nMust be between 64 and 128 and be divisible by 16.",
    rounding=16,
    min_max=(64, 128),
    fixed=True)

output_size = ConfigItem(
    datatype=int,
    default=128,
    group="size",
    info="Output image resolution (in pixels).\nBe aware that larger resolution will "
         "increase VRAM requirements.\nNB: Must be between 64 and 256 and be divisible "
         "by 16.",
    rounding=16,
    min_max=(64, 256),
    fixed=True)

dense_nodes = ConfigItem(
    datatype=int,
    default=1536,
    group="network",
    info="Number of nodes for decoder. Might affect your model's ability to learn in "
         "general.\nNote that: Lower values will affect the ability to predict "
         "details.",
    rounding=64,
    min_max=(768, 2048),
    fixed=True)

complexity_encoder = ConfigItem(
    datatype=int,
    default=128,
    group="network",
    info="Encoder Convolution Layer Complexity. sensible ranges: 128 to 150.",
    rounding=4,
    min_max=(96, 160),
    fixed=True)

complexity_decoder = ConfigItem(
    datatype=int,
    default=512,
    group="network",
    info="Decoder Complexity.",
    rounding=4,
    min_max=(512, 544),
    fixed=True)
