#!/usr/bin/env python3
""" The default options for the faceswap Manual_Balance Color plugin.

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


HELPTEXT = "Options for manually altering the balance of colors of the swapped face"


colorspace = ConfigItem(
    datatype=str,
    default="HSV",
    group="color balance",
    info="The colorspace to use for adjustment: The three adjustment sliders will "
         "effect the image differently depending on which colorspace is selected:"
         "\n\t RGB: Red, Green, Blue. An additive colorspace where colors are obtained "
         "by a linear combination of Red, Green, and Blue values. The three channels "
         "are correlated by the amount of light hitting the surface. In RGB color "
         "space the color information is separated into three channels but the same "
         "three channels also encode brightness information."
         "\n\t HSV: Hue, Saturation, Value. Hue - Dominant wavelength. Saturation - "
         "Purity / shades of color. Value - Intensity. Best thing is that it uses only "
         "one channel to describe color (H), making it very intuitive to specify color."
         "\n\t LAB: Lightness, A, B. Lightness - Intensity. A - Color range from green "
         "to magenta. B - Color range from blue to yellow. The L channel is "
         "independent of color information and encodes brightness only. The other two "
         "channels encode color."
         "\n\t YCrCb: Y - Luminance or Luma component obtained from RGB after gamma "
         "correction. Cr - how far is the red component from Luma. Cb - how far is the "
         "blue component from Luma. Separates the luminance and chrominance components "
         "into different channels.",
    choices=["RGB", "HSV", "LAB", "YCrCb"],
    gui_radio=True)

balance_1 = ConfigItem(
    datatype=float,
    default=0.0,
    group="color balance",
    info="Balance of channel 1:"
         "\n\tRGB: Red"
         "\n\tHSV: Hue"
         "\n\tLAB: Lightness"
         "\n\tYCrCb: Luma",
    rounding=1,
    min_max=(-100.0, 100.0))

balance_2 = ConfigItem(
    datatype=float,
    default=0.0,
    group="color balance",
    info="Balance of channel 2:"
         "\n\tRGB: Green"
         "\n\tHSV: Saturation"
         "\n\tLAB: Green > Magenta"
         "\n\tYCrCb: Distance of red from Luma",
    rounding=1,
    min_max=(-100.0, 100.0))

balance_3 = ConfigItem(
    datatype=float,
    default=0.0,
    group="color balance",
    info="Balance of channel 3:"
         "\n\tRGB: Blue"
         "\n\tHSV: Intensity"
         "\n\tLAB: Blue > Yellow"
         "\n\tYCrCb: Distance of blue from Luma",
    rounding=1,
    min_max=(-100.0, 100.0))

contrast = ConfigItem(
    datatype=float,
    default=0.0,
    group="brightness contrast",
    info="Amount of contrast applied.",
    rounding=1,
    min_max=(-100.0, 100.0))

brightness = ConfigItem(
    datatype=float,
    default=0.0,
    group="brightness contrast",
    info="Amount of brighness applied.",
    rounding=1,
    min_max=(-100.0, 100.0))
