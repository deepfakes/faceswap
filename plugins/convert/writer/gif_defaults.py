#!/usr/bin/env python3
"""
    The default options for the faceswap Gif Writer plugin.

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


_HELPTEXT = "Options for outputting converted frames to an animated gif."


_DEFAULTS = dict(
    fps=dict(
        default=25,
        info="Frames per Second.",
        datatype=int,
        rounding=1,
        min_max=(1, 60),
        choices=[],
        group="settings",
        gui_radio=False,
        fixed=True,
    ),
    loop=dict(
        default=0,
        info="The number of iterations. Set to 0 to loop indefinitely.",
        datatype=int,
        rounding=1,
        min_max=(0, 100),
        choices=[],
        group="settings",
        gui_radio=False,
        fixed=True,
    ),
    palettesize=dict(
        default="256",
        info="The number of colors to quantize the image to. Is rounded to the nearest power of "
             "two.",
        datatype=str,
        rounding=None,
        min_max=None,
        choices=["2", "4", "8", "16", "32", "64", "128", "256"],
        group="settings",
        gui_radio=False,
        fixed=True,
    ),
    subrectangles=dict(
        default=False,
        info="If True, will try and optimize the GIF by storing only the rectangular parts of "
             "each frame that change with respect to the previous.",
        datatype=bool,
        rounding=None,
        min_max=None,
        choices=[],
        group="settings",
        gui_radio=False,
        fixed=True,
    ),
)
