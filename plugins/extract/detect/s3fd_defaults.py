#!/usr/bin/env python3
""" The default options for the faceswap S3Fd Detect plugin.

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
    "S3FD Detector options.\n"
    "Fast on GPU, slow on CPU. Can detect more faces and fewer false positives than other GPU "
    "detectors, but is a lot more resource intensive."
    )


confidence = ConfigItem(
    datatype=int,
    default=70,
    group="settings",
    info="The confidence level at which the detector has succesfully found a face.\n"
         "Higher levels will be more discriminating, lower levels will have more false "
         "positives.",
    rounding=5,
    min_max=(25, 100))

batch_size = ConfigItem(
    datatype=int,
    default=4,
    group="settings",
    info="The batch size to use. To a point, higher batch sizes equal better performance, "
         "but setting it too high can harm performance.\n"
         "\n\tNvidia users: If the batchsize is set higher than the your GPU can "
         "accomodate then this will automatically be lowered."
         "\n\tAMD users: A batchsize of 8 requires about 2 GB vram.",
    rounding=1,
    min_max=(1, 64))
