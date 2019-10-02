#!/usr/bin/env python3
"""
    The default options for the faceswap Box_Blend Mask plugin.

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


_HELPTEXT = "Options for blending the edges of the swapped box with the background image"


_DEFAULTS = {
    "type": {
        "default": "gaussian",
        "info": "The type of blending to use:"
                "\n\t gaussian: Blend with Gaussian filter. Slower, but often better than "
                "Normalized"
                "\n\t normalized: Blend with Normalized box filter. Faster than Gaussian"
                "\n\t none: Don't perform blending",
        "datatype": str,
        "rounding": None,
        "min_max": None,
        "choices": ["gaussian", "normalized", "none"],
        "gui_radio": True,
        "fixed": True,
    },
    "distance": {
        "default": 11.0,
        "info": "The distance from the edges of the swap box to start blending.\nThe distance "
                "is set as percentage of the swap box size to give the number of pixels from "
                "the edge of the box. Eg: For a swap area of 256px and a percentage of 4%, "
                "blending would commence 10 pixels from the edge.\nHigher percentages start "
                "the blending from closer to the center of the face, so will reveal more of "
                "the source face.",
        "datatype": float,
        "rounding": 1,
        "group": "settings",
        "min_max": (0.1, 25.0),
        "choices": [],
        "gui_radio": False,
        "fixed": True,
    },
    "radius": {
        "default": 5.0,
        "info": "Radius dictates how much blending should occur, or more specifically, how "
                "far the blending will spread away from the 'distance' parameter.\nThis "
                "figure is set as a percentage of the swap box size to give the radius in "
                "pixels. Eg: For a swap area of 256px and a percentage of 5%, the radius "
                "would be 13 pixels\nNB: Higher percentage means more blending, but too high "
                "may reveal more of the source face, or lead to hard lines at the border.",
        "datatype": float,
        "rounding": 1,
        "min_max": (0.1, 25.0),
        "choices": [],
        "gui_radio": False,
        "group": "settings",
        "fixed": True,
    },
    "passes": {
        "default": 1,
        "info": "The number of passes to perform. Additional passes of the blending algorithm "
                "can improve smoothing at a time cost. This is more useful for 'box' type "
                "blending.\nAdditional passes have exponentially less effect so it's not "
                "worth setting this too high.",
        "datatype": int,
        "rounding": 1,
        "min_max": (1, 8),
        "choices": [],
        "gui_radio": False,
        "group": "settings",
        "fixed": True,
    },
}
