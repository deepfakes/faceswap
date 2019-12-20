#!/usr/bin/env python3
"""
    The default options for the faceswap Mask_Blend Mask plugin.

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


_HELPTEXT = "Options for blending the edges between the mask and the background image"


_DEFAULTS = {
    "type": {
        "default": "normalized",
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
    "kernel_size": {
        "default": 3,
        "info": "The kernel size dictates how much blending should occur.\n"
                "The size is the diameter of the kernel in pixels (calculated from a 128px mask). "
                " This value should be odd, if an even number is passed in then it will be "
                "rounded to the next odd number. Higher sizes means more blending.",
        "datatype": int,
        "rounding": 1,
        "min_max": (1, 9),
        "choices": [],
        "gui_radio": False,
        "group": "settings",
        "fixed": True,
    },
    "passes": {
        "default": 4,
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
    "threshold": {
        "default": 4,
        "info": "Sets pixels that are near white to white and near black to black. Set to 0 for "
                "off.",
        "datatype": int,
        "rounding": 1,
        "min_max": (0, 50),
        "choices": [],
        "gui_radio": False,
        "group": "settings",
        "fixed": True,
    },
    "erosion": {
        "default": 0.0,
        "info": "Erosion kernel size as a percentage of the mask radius area.\nPositive "
                "values apply erosion which reduces the size of the swapped area.\nNegative "
                "values apply dilation which increases the swapped area.",
        "datatype": float,
        "rounding": 1,
        "min_max": (-100.0, 100.0),
        "choices": [],
        "gui_radio": False,
        "group": "settings",
        "fixed": True,
    },
}
