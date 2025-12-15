#!/usr/bin/env python3
""" The default options for the faceswap Color_Transfer Color plugin.

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
"""
from lib.config import ConfigItem


HELPTEXT = (
    "Options for transfering the color distribution from the source to the target image using the "
    "mean and standard deviations of the L*a*b* color space.\nThis implementation is (loosely) "
    "based on the 'Color Transfer between Images' paper by Reinhard et al., 2001. matching the "
    "histograms between the source and destination faces.")


clip = ConfigItem(
    datatype=bool,
    default=True,
    group="method",
    info="Should components of L*a*b* image be scaled by numpy.clip before converting back to "
         "BGR color space?\nIf False then components will be min-max scaled appropriately.\n"
         "Clipping will keep target image brightness truer to the input.\nScaling will adjust "
         "image brightness to avoid washed out portions in the resulting color transfer that "
         "can be caused by clipping.")

preserve_paper = ConfigItem(
    datatype=bool,
    group="method",
    default=True,
    info="Should color transfer strictly follow methodology layed out in original paper?\nThe "
         "method does not always produce aesthetically pleasing results.\nIf False then "
         "L*a*b* components will be scaled using the reciprocal of the scaling factor "
         "proposed in the paper. This method seems to produce more consistently aesthetically "
         "pleasing results.")
