#!/usr/bin/env python3
"""
    The default options for the faceswap Manual_Balance Color plugin.

    Defaults files should be named <plugin_name>_defaults.py
    Any items placed into this file will automatically get added to the relevant config .ini files
    within the faceswap/config folder.

    The following variables should be defined:
        _HELPTEXT: A string describing what this plugin does
        _DEFAULTS: A dictionary containing the options, defaults and meta information. The
                   dictionary should be defined as:
                       {<option_name>: {<metadata>}}

                   <option_name> should always be lower text.
                   <metadata> dictionary requirements are listed below.

    The following keys are expected for the _DEFAULTS <metadata> dict:
        datatype:  [required] A python type class. This limits the type of data that can be
                   provided in the .ini file and ensures that the value is returned in the
                   correct type to faceswap. Valid datatypes are: <class 'int'>, <class 'float'>,
                   <class 'str'>, <class 'bool'>.
        default:   [required] The default value for this option.
        info:      [required] A string describing what this option does.
        choices:   [optional] If this option's datatype is of <class 'str'> then valid
                   selections can be defined here. This validates the option and also enables
                   a combobox / radio option in the GUI.
        gui_radio: [optional] If <choices> are defined, this indicates that the GUI should use
                   radio buttons rather than a combobox to display this option.
        min_max:   [partial] For <class 'int'> and <class 'float'> datatypes this is required
                   otherwise it is ignored. Should be a tuple of min and max accepted values.
                   This is used for controlling the GUI slider range. Values are not enforced.
        rounding:  [partial] For <class 'int'> and <class 'float'> datatypes this is
                   required otherwise it is ignored. Used for the GUI slider. For floats, this
                   is the number of decimal places to display. For ints this is the step size.
        fixed:     [optional] [train only]. Training configurations are fixed when the model is
                   created, and then reloaded from the state file. Marking an item as fixed=False
                   indicates that this value can be changed for existing models, and will override
                   the value saved in the state file with the updated value in config. If not
                   provided this will default to True.
"""


_HELPTEXT = "Options for manually altering the balance of colors of the swapped face"


_DEFAULTS = {
    "colorspace": {
        "default": "HSV",
        "info": "The colorspace to use for adjustment: The three adjustment sliders will "
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
        "datatype": str,
        "rounding": None,
        "min_max": None,
        "group": "color balance",
        "choices": ["RGB", "HSV", "LAB", "YCrCb"],
        "gui_radio": True,
        "fixed": True,
    },
    "balance_1": {
        "default": 0.0,
        "info": "Balance of channel 1:"
                "\n\tRGB: Red"
                "\n\tHSV: Hue"
                "\n\tLAB: Lightness"
                "\n\tYCrCb: Luma",
        "datatype": float,
        "rounding": 1,
        "min_max": (-100.0, 100.0),
        "choices": [],
        "group": "color balance",
        "gui_radio": False,
        "fixed": True,
    },
    "balance_2": {
        "default": 0.0,
        "info": "Balance of channel 2:"
                "\n\tRGB: Green"
                "\n\tHSV: Saturation"
                "\n\tLAB: Green > Magenta"
                "\n\tYCrCb: Distance of red from Luma",
        "datatype": float,
        "rounding": 1,
        "min_max": (-100.0, 100.0),
        "choices": [],
        "gui_radio": False,
        "group": "color balance",
        "fixed": True,
    },
    "balance_3": {
        "default": 0.0,
        "info": "Balance of channel 3:"
                "\n\tRGB: Blue"
                "\n\tHSV: Intensity"
                "\n\tLAB: Blue > Yellow"
                "\n\tYCrCb: Distance of blue from Luma",
        "datatype": float,
        "rounding": 1,
        "min_max": (-100.0, 100.0),
        "choices": [],
        "gui_radio": False,
        "group": "color balance",
        "fixed": True,
    },
    "contrast": {
        "default": 0.0,
        "info": "Amount of contrast applied.",
        "datatype": float,
        "rounding": 1,
        "min_max": (-100.0, 100.0),
        "choices": [],
        "gui_radio": False,
        "group": "brightness contrast",
        "fixed": True,
    },
    "brightness": {
        "default": 0.0,
        "info": "Amount of brighness applied.",
        "datatype": float,
        "rounding": 1,
        "min_max": (-100.0, 100.0),
        "choices": [],
        "gui_radio": False,
        "group": "brightness contrast",
        "fixed": True,
    },
}
