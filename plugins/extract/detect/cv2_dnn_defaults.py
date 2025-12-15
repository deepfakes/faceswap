#!/usr/bin/env python3
""" The default options for the faceswap Cv2_Dnn Detect plugin.

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
    "CV2 DNN Detector options.\n"
    "A CPU only extractor, is the least reliable, but uses least resources and runs fast on CPU. "
    "Use this if not using a GPU and time is important"
)


confidence = ConfigItem(
    datatype=int,
    default=50,
    group="settings",
    info="The confidence level at which the detector has succesfully found a face.\nHigher "
         "levels will be more discriminating, lower levels will have more false positives.",
    rounding=5,
    min_max=(25, 100))
