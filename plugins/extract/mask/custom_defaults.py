#!/usr/bin/env python3
"""
    The default options for the faceswap BiSeNet Face Parsing plugin.

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
                   correct type to faceswap. Valid data types are: <class 'int'>, <class 'float'>,
                   <class 'str'>, <class 'bool'>.
        default:   [required] The default value for this option.
        info:      [required] A string describing what this option does.
        group:     [optional]. A group for grouping options together in the GUI. If not
                   provided this will not group this option with any others.
        choices:   [optional] If this option's datatype is of <class 'str'> then valid
                   selections can be defined here. This validates the option and also enables
                   a combobox / radio option in the GUI.
        gui_radio: [optional] If <choices> are defined, this indicates that the GUI should use
                   radio buttons rather than a combobox to display this option.
        min_max:   [partial] For <class 'int'> and <class 'float'> data types this is required
                   otherwise it is ignored. Should be a tuple of min and max accepted values.
                   This is used for controlling the GUI slider range. Values are not enforced.
        rounding:  [partial] For <class 'int'> and <class 'float'> data types this is
                   required otherwise it is ignored. Used for the GUI slider. For floats, this
                   is the number of decimal places to display. For ints this is the step size.
        fixed:     [optional] [train only]. Training configurations are fixed when the model is
                   created, and then reloaded from the state file. Marking an item as fixed=False
                   indicates that this value can be changed for existing models, and will override
                   the value saved in the state file with the updated value in config. If not
                   provided this will default to True.
"""


_HELPTEXT = (
    "Custom (dummy) Mask options..\n"
    "The custom mask just fills a face patch with all 0's (masked out) or all 1's (masked in) for "
    "later manual editing. It does not use the GPU for creation."
    )


_DEFAULTS = {
    "batch-size": {
        "default": 8,
        "info": "The batch size to use. To a point, higher batch sizes equal better performance, "
                "but setting it too high can harm performance.",
        "datatype": int,
        "rounding": 1,
        "min_max": (1, 64),
        "group": "settings"
    },
    "centering": {
        "default": "face",
        "info": "Whether to create a dummy mask with face or head centering.",
        "choices": ["face", "head"],
        "datatype": str,
        "group": "settings",
        "gui_radio": True
    },
    "fill": {
        "default": False,
        "info": "Whether the mask should be filled (True) in which case the custom mask will be "
                "created with the whole area masked in (i.e. you would need to manually edit out "
                "the background) or unfilled (False) in which case you would need to manually "
                "edit in the face.",
        "datatype": bool,
        "group": "settings",
        "gui_radio": True,
    },
}
