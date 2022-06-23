#!/usr/bin/env python3
"""
    The default options for the faceswap Unbalanced Model plugin.

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


_HELPTEXT = (
    "An unbalanced model with adjustable input size options.\n"
    "This is an unbalanced model so b>a swaps may not work well\n"
)


_DEFAULTS = dict(
    input_size=dict(
        default=128,
        info="Resolution (in pixels) of the image to train on.\n"
             "BE AWARE Larger resolution will dramatically increaseVRAM requirements.\n"
             "Make sure your resolution is divisible by 64 (e.g. 64, 128, 256 etc.).\n"
             "NB: Your faceset must be at least 1.6x larger than your required input "
             "size.\n(e.g. 160 is the maximum input size for a 256x256 faceset).",
        datatype=int,
        rounding=64,
        min_max=(64, 512),
        choices=[],
        gui_radio=False,
        group="size",
        fixed=True),
    lowmem=dict(
        default=False,
        info="Lower memory mode. Set to 'True' if having issues with VRAM useage.\n"
             "NB: Models with a changed lowmem mode are not compatible with each other.\n"
             "NB: lowmem will override cutom nodes and complexity settings.",
        datatype=bool,
        rounding=None,
        min_max=None,
        choices=[],
        gui_radio=False,
        group="settings",
        fixed=True),
    nodes=dict(
        default=1024,
        info="Number of nodes for decoder. Don't change this unless you know what you are doing!",
        datatype=int,
        rounding=64,
        min_max=(512, 4096),
        choices=[],
        gui_radio=False,
        fixed=True,
        group="network"),
    complexity_encoder=dict(
        default=128,
        info="Encoder Convolution Layer Complexity. sensible ranges: 128 to 160.",
        datatype=int,
        rounding=16,
        min_max=(64, 1024),
        choices=[],
        gui_radio=False,
        fixed=True,
        group="network"),
    complexity_decoder_a=dict(
        default=384,
        info="Decoder A Complexity.",
        datatype=int,
        rounding=16,
        min_max=(64, 1024),
        choices=[],
        gui_radio=False,
        fixed=True,
        group="network"),
    complexity_decoder_b=dict(
        default=512,
        info="Decoder B Complexity.",
        datatype=int,
        rounding=16,
        min_max=(64, 1024),
        choices=[],
        gui_radio=False,
        fixed=True,
        group="network"))
