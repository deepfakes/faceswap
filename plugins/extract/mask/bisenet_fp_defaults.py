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
    "BiSeNet Face Parsing options.\n"
    "Mask ported from https://github.com/zllrunning/face-parsing.PyTorch."
    )


batch_size = ConfigItem(
    datatype=int,
    default=8,
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
    info="BiseNet mask still runs fairly quickly on CPU on some setups. Enable "
         "CPU mode here to use the CPU for this masker to save some VRAM at a speed cost.")

weights = ConfigItem(
    datatype=str,
    default="faceswap",
    group="settings",
    info="The trained weights to use.\n"
         "\n\tfaceswap - Weights trained on wildly varied Faceswap extracted data to "
         "better handle varying conditions, obstructions, glasses and multiple targets "
         "within a single extracted image."
         "\n\toriginal - The original weights trained on the CelebAMask-HQ dataset.",
    choices=["faceswap", "original"],
    gui_radio=True)

include_ears = ConfigItem(
    datatype=bool,
    default=False,
    group="settings",
    info="Whether to include ears within the face mask.")

include_hair = ConfigItem(
    datatype=bool,
    default=False,
    group="settings",
    info="Whether to include hair within the face mask.")

include_glasses = ConfigItem(
    datatype=bool,
    default=True,
    group="settings",
    info="Whether to include glasses within the face mask.\n\tFor 'original' weights "
         "excluding glasses will mask out the lenses as well as the frames.\n\tFor "
         "'faceswap' weights, the model has been trained to mask out lenses if eyes cannot "
         "be seen (i.e. dark sunglasses) or just the frames if the eyes can be seen.")
