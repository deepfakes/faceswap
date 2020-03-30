#!/usr/bin/env python3
"""
    The default options for the faceswap Mtcnn Detect plugin.

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
    "MTCNN Detector options.\n"
    "Fast on GPU, slow on CPU. Uses fewer resources than other GPU detectors but can often return "
    "more false positives."
)


_DEFAULTS = {
    "minsize": {
        "default": 20,
        "info": "The minimum size of a face (in pixels) to be accepted as a positive match.\n"
                "Lower values use significantly more VRAM and will detect more false "
                "positives.",
        "datatype": int,
        "rounding": 10,
        "min_max": (20, 1000),
        "choices": [],
        "gui_radio": False,
        "fixed": True,
    },
    "threshold_1": {
        "default": 0.6,
        "info": "First stage threshold for face detection. This stage obtains face "
                "candidates.",
        "datatype": float,
        "rounding": 2,
        "min_max": (0.1, 0.9),
        "choices": [],
        "gui_radio": False,
        "fixed": True,
    },
    "threshold_2": {
        "default": 0.7,
        "info": "Second stage threshold for face detection. This stage refines face "
                "candidates.",
        "datatype": float,
        "rounding": 2,
        "min_max": (0.1, 0.9),
        "choices": [],
        "gui_radio": False,
        "fixed": True,
    },
    "threshold_3": {
        "default": 0.7,
        "info": "Third stage threshold for face detection. This stage further refines face "
                "candidates.",
        "datatype": float,
        "rounding": 2,
        "min_max": (0.1, 0.9),
        "choices": [],
        "gui_radio": False,
        "fixed": True,
    },
    "scalefactor": {
        "default": 0.709,
        "info": "The scale factor for the image pyramid.",
        "datatype": float,
        "rounding": 3,
        "min_max": (0.1, 0.9),
        "choices": [],
        "gui_radio": False,
        "fixed": True,
    },
    "batch-size": {
        "default": 8,
        "info": "The batch size to use. To a point, higher batch sizes equal better performance, "
                "but setting it too high can harm performance.\n"
                "\n\tNvidia users: If the batchsize is set higher than the your GPU can "
                "accomodate then this will automatically be lowered.",
        "datatype": int,
        "rounding": 1,
        "min_max": (1, 64),
        "choices": [],
        "gui_radio": False,
        "fixed": True,
    }
}
