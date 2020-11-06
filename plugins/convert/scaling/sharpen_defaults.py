#!/usr/bin/env python3
"""
    The default options for the faceswap Sharpen Scaling plugin.

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


_HELPTEXT = "Options for sharpening the face after placement"


_DEFAULTS = {
    "method": {
        "default": "none",
        "info": "The type of sharpening to use:"
                "\n\t none: Don't perform any sharpening."
                "\n\t box: Fastest, but weakest method. Uses a box filter to assess edges."
                "\n\t gaussian: Slower, but better than box. Uses a gaussian filter to assess "
                "edges."
                "\n\t unsharp-mask: Slowest, but most tweakable. Uses the unsharp-mask method "
                "to assess edges.",
        "datatype": str,
        "rounding": None,
        "min_max": None,
        "choices": ["none", "box", "gaussian", "unsharp_mask"],
        "gui_radio": True,
        "fixed": True,
    },
    "amount": {
        "default": 150,
        "info": "Percentage that controls the magnitude of each overshoot (how much darker "
                "and how much lighter the edge borders become).\nThis can also be thought of "
                "as how much contrast is added at the edges. It does not affect the width of "
                "the edge rims.",
        "datatype": int,
        "rounding": 1,
        "min_max": (100, 500),
        "choices": [],
        "gui_radio": False,
        "group": "settings",
        "fixed": True,
    },
    "radius": {
        "default": 0.3,
        "info": "Affects the size of the edges to be enhanced or how wide the edge rims "
                "become, so a smaller radius enhances smaller-scale detail.\nRadius is set as "
                "a percentage of the final frame width and rounded to the nearest pixel. E.g "
                "for a 1280 width frame, a 0.6 percenatage will give a radius of 8px.\nHigher "
                "radius values can cause halos at the edges, a detectable faint light rim "
                "around objects. Fine detail needs a smaller radius. \nRadius and amount "
                "interact; reducing one allows more of the other.",
        "datatype": float,
        "rounding": 1,
        "min_max": (0.1, 5.0),
        "choices": [],
        "gui_radio": False,
        "group": "settings",
        "fixed": True,
    },
    "threshold": {
        "default": 5.0,
        "info": "[unsharp_mask only] Controls the minimal brightness change that will be "
                "sharpened or how far apart adjacent tonal values have to be before the "
                "filter does anything.\nThis lack of action is important to prevent smooth "
                "areas from becoming speckled. The threshold setting can be used to sharpen "
                "more pronounced edges, while leaving subtler edges untouched. \nLow values "
                "should sharpen more because fewer areas are excluded. \nHigher threshold "
                "values exclude areas of lower contrast.",
        "datatype": float,
        "rounding": 1,
        "min_max": (1.0, 10.0),
        "choices": [],
        "gui_radio": False,
        "group": "settings",
        "fixed": True,
    },
}
