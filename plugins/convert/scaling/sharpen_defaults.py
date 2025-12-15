#!/usr/bin/env python3
""" The default options for the faceswap Sharpen Scaling plugin.

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


HELPTEXT = "Options for sharpening the face after placement"


method = ConfigItem(
    datatype=str,
    default="none",
    group="sharpen type",
    info="The type of sharpening to use:"
         "\n\t none: Don't perform any sharpening."
         "\n\t box: Fastest, but weakest method. Uses a box filter to assess edges."
         "\n\t gaussian: Slower, but better than box. Uses a gaussian filter to assess edges."
         "\n\t unsharp-mask: Slowest, but most tweakable. Uses the unsharp-mask method to "
         "assess edges.",
    choices=["none", "box", "gaussian", "unsharp_mask"],
    gui_radio=True)

amount = ConfigItem(
    datatype=int,
    default=150,
    group="settings",
    info="Percentage that controls the magnitude of each overshoot (how much darker and how "
         "much lighter the edge borders become).\nThis can also be thought of as how much "
         "contrast is added at the edges. It does not affect the width of the edge rims.",
    rounding=1,
    min_max=(100, 500))

radius = ConfigItem(
    datatype=float,
    default=0.3,
    group="settings",
    info="Affects the size of the edges to be enhanced or how wide the edge rims become, so a "
         "smaller radius enhances smaller-scale detail.\nRadius is set as a percentage of the "
         "final frame width and rounded to the nearest pixel. E.g for a 1280 width frame, a "
         "0.6 percenatage will give a radius of 8px.\nHigher radius values can cause halos at "
         "the edges, a detectable faint light rim around objects. Fine detail needs a smaller "
         "radius. \nRadius and amount interact; reducing one allows more of the other.",
    rounding=1,
    min_max=(0.1, 5.0))

threshold = ConfigItem(
    datatype=float,
    default=5.0,
    group="settings",
    info="[unsharp_mask only] Controls the minimal brightness change that will be sharpened "
         "or how far apart adjacent tonal values have to be before the filter does anything.\n"
         "This lack of action is important to prevent smooth areas from becoming speckled. "
         "The threshold setting can be used to sharpen more pronounced edges, while leaving "
         "subtler edges untouched. \nLow values should sharpen more because fewer areas are "
         "excluded. \nHigher threshold values exclude areas of lower contrast.",
    rounding=1,
    min_max=(1.0, 10.0))
