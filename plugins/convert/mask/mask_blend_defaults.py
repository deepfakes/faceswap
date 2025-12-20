#!/usr/bin/env python3
""" The default options for the faceswap Mask_Blend Mask plugin.

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


HELPTEXT = "Options for blending the edges between the mask and the background image"


type = ConfigItem(  # pylint:disable=redefined-builtin
    datatype=str,
    default="normalized",
    group="Blending type",
    info="The type of blending to use:"
         "\n\t gaussian: Blend with Gaussian filter. Slower, but often better than Normalized"
         "\n\t normalized: Blend with Normalized box filter. Faster than Gaussian"
         "\n\t none: Don't perform blending",
    choices=["gaussian", "normalized", "none"])

kernel_size = ConfigItem(
    datatype=int,
    default=3,
    group="settings",
    info="The kernel size dictates how much blending should occur.\n"
         "The size is the diameter of the kernel in pixels (calculated from a 128px mask). "
         "This value should be odd, if an even number is passed in then it will be rounded to "
         "the next odd number. Higher sizes means more blending.",
    rounding=1,
    min_max=(1, 9))

passes = ConfigItem(
    default=4,
    datatype=int,
    group="settings",
    info="The number of passes to perform. Additional passes of the blending algorithm can "
         "improve smoothing at a time cost. This is more useful for 'box' type blending.\n"
         "Additional passes have exponentially less effect so it's not worth setting this too "
         "high.",
    rounding=1,
    min_max=(1, 8))

threshold = ConfigItem(
    default=4,
    datatype=int,
    group="settings",
    info="Sets pixels that are near white to white and near black to black. Set to 0 for off.",
    rounding=1,
    min_max=(0, 50))

erosion = ConfigItem(
    datatype=float,
    default=0.0,
    group="settings",
    info="Apply erosion to the whole of the face mask.\n"
         "Erosion kernel size as a percentage of the mask radius area.\n"
         "Positive values apply erosion which reduces the size of the swapped area.\n"
         "Negative values apply dilation which increases the swapped area.",
    rounding=1,
    min_max=(-100.0, 100.0))

erosion_top = ConfigItem(
    datatype=float,
    default=0.0,
    group="settings",
    info="Apply erosion to the top part of the mask only.\n"
         "Positive values apply erosion which pulls the mask into the center.\n"
         "Negative values apply dilation which pushes the mask away from the center.",
    rounding=1,
    min_max=(-100.0, 100.0))

erosion_bottom = ConfigItem(
    datatype=float,
    default=0.0,
    group="settings",
    info="Apply erosion to the bottom part of the mask only.\n"
         "Positive values apply erosion which pulls the mask into the center.\n"
         "Negative values apply dilation which pushes the mask away from the center.",
    rounding=1,
    min_max=(-100.0, 100.0))

erosion_left = ConfigItem(
    default=0.0,
    datatype=float,
    group="settings",
    info="Apply erosion to the left part of the mask only.\n"
         "Positive values apply erosion which pulls the mask into the center.\n"
         "Negative values apply dilation which pushes the mask away from the center.",
    rounding=1,
    min_max=(-100.0, 100.0))

erosion_right = ConfigItem(
    datatype=float,
    default=0.0,
    group="settings",
    info="Apply erosion to the right part of the mask only.\n"
         "Positive values apply erosion which pulls the mask into the center.\n"
         "Negative values apply dilation which pushes the mask away from the center.",
    rounding=1,
    min_max=(-100.0, 100.0))
