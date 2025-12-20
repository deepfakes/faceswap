#!/usr/bin/env python3
""" The default options for the faceswap Unbalanced Model plugin.

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
    "An unbalanced model with adjustable input size options.\n"
    "This is an unbalanced model so b>a swaps may not work well\n"
)


input_size = ConfigItem(
    datatype=int,
    default=128,
    group="size",
    info="Resolution (in pixels) of the image to train on.\n"
         "BE AWARE Larger resolution will dramatically increaseVRAM requirements.\n"
         "Make sure your resolution is divisible by 64 (e.g. 64, 128, 256 etc.).\n"
         "NB: Your faceset must be at least 1.6x larger than your required input "
         "size.\n(e.g. 160 is the maximum input size for a 256x256 faceset).",
    rounding=64,
    min_max=(64, 512),
    fixed=True)

lowmem = ConfigItem(
    datatype=bool,
    default=False,
    group="settings",
    info="Lower memory mode. Set to 'True' if having issues with VRAM useage.\n"
         "NB: Models with a changed lowmem mode are not compatible with each other.\n"
         "NB: lowmem will override cutom nodes and complexity settings.",
    fixed=True)

nodes = ConfigItem(
    datatype=int,
    default=1024,
    group="network",
    info="Number of nodes for decoder. Don't change this unless you know what you are doing!",
    rounding=64,
    min_max=(512, 4096),
    fixed=True)

complexity_encoder = ConfigItem(
    datatype=int,
    default=128,
    group="network",
    info="Encoder Convolution Layer Complexity. sensible ranges: 128 to 160.",
    rounding=16,
    min_max=(64, 1024),
    fixed=True)

complexity_decoder_a = ConfigItem(
    datatype=int,
    default=384,
    group="network",
    info="Decoder A Complexity.",
    rounding=16,
    min_max=(64, 1024),
    fixed=True)

complexity_decoder_b = ConfigItem(
    datatype=int,
    default=512,
    group="network",
    info="Decoder B Complexity.",
    rounding=16,
    min_max=(64, 1024),
    fixed=True)
