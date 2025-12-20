#!/usr/bin/env python3
""" The default options for the faceswap Mtcnn Detect plugin.

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
    "MTCNN Detector options.\n"
    "Fast on GPU, slow on CPU. Uses fewer resources than other GPU detectors but can often return "
    "more false positives."
)


minsize = ConfigItem(
    datatype=int,
    default=20,
    group="settings",
    info="The minimum size of a face (in pixels) to be accepted as a positive match."
         "\nLower values use significantly more VRAM and will detect more false positives.",
    rounding=10,
    min_max=(20, 1000))

scalefactor = ConfigItem(
    datatype=float,
    default=0.709,
    group="settings",
    info="The scale factor for the image pyramid.",
    rounding=3,
    min_max=(0.1, 0.9))

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
    default=True,
    group="settings",
    info="MTCNN detector still runs fairly quickly on CPU on some setups. "
         "Enable CPU mode here to use the CPU for this detector to save some VRAM at a "
         "speed cost.")

threshold_1 = ConfigItem(
    datatype=float,
    default=0.6,
    group="threshold",
    info="First stage threshold for face detection. This stage obtains face candidates.",
    rounding=2,
    min_max=(0.1, 0.9))

threshold_2 = ConfigItem(
    datatype=float,
    default=0.7,
    group="threshold",
    info="Second stage threshold for face detection. This stage refines face candidates.",
    rounding=2,
    min_max=(0.1, 0.9))

threshold_3 = ConfigItem(
    datatype=float,
    default=0.7,
    group="threshold",
    info="Third stage threshold for face detection. This stage further refines face "
         "candidates.",
    rounding=2,
    min_max=(0.1, 0.9))
