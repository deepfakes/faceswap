#!/usr/bin/env python3
""" The default options for the faceswap Dfl_SAE Model plugin.

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


HELPTEXT = "DFL SAE Model (Adapted from https://github.com/iperov/DeepFaceLab)"


input_size = ConfigItem(
    datatype=int,
    default=128,
    group="size",
    info="Resolution (in pixels) of the input image to train on.\n"
         "BE AWARE Larger resolution will dramatically increase VRAM requirements.\n"
         "\nMust be divisible by 16.",
    rounding=16,
    min_max=(64, 256),
    fixed=True)

architecture = ConfigItem(
    datatype=str,
    default="df",
    group="network",
    info="Model architecture:"
         "\n\t'df': Keeps the faces more natural."
         "\n\t'liae': Can help fix overly different face shapes.",
    choices=["df", "liae"],
    gui_radio=True,
    fixed=True)

autoencoder_dims = ConfigItem(
    datatype=int,
    default=0,
    group="network",
    info="Face information is stored in AutoEncoder dimensions. If there are not enough "
         "dimensions then certain facial features may not be recognized."
         "\nHigher number of dimensions are better, but require more VRAM."
         "\nSet to 0 to use the architecture defaults (256 for liae, 512 for df).",
    rounding=32,
    min_max=(0, 1024),
    fixed=True)

encoder_dims = ConfigItem(
    datatype=int,
    default=42,
    group="network",
    info="Encoder dimensions per channel. Higher number of encoder dimensions will help "
         "the model to recognize more facial features, but will require more VRAM.",
    rounding=1,
    min_max=(21, 85),
    fixed=True)

decoder_dims = ConfigItem(
    datatype=int,
    default=21,
    group="network",
    info="Decoder dimensions per channel. Higher number of decoder dimensions will help "
         "the model to improve details, but will require more VRAM.",
    rounding=1,
    min_max=(10, 85),
    fixed=True)

multiscale_decoder = ConfigItem(
    datatype=bool,
    default=False,
    group="network",
    info="Multiscale decoder can help to obtain better details.",
    fixed=True)
