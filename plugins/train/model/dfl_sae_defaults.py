#!/usr/bin/env python3
"""
    The default options for the faceswap Dfl_SAE Model plugin.

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


_HELPTEXT = "DFL SAE Model (Adapted from https://github.com/iperov/DeepFaceLab)"


_DEFAULTS = dict(
    input_size=dict(
        default=128,
        info="Resolution (in pixels) of the input image to train on.\n"
             "BE AWARE Larger resolution will dramatically increase VRAM requirements.\n"
             "\nMust be divisible by 16.",
        datatype=int,
        rounding=16,
        min_max=(64, 256),
        group="size",
        fixed=True),
    architecture=dict(
        default="df",
        info="Model architecture:"
             "\n\t'df': Keeps the faces more natural."
             "\n\t'liae': Can help fix overly different face shapes.",
        datatype=str,
        choices=["df", "liae"],
        gui_radio=True,
        fixed=True,
        group="network"),
    autoencoder_dims=dict(
        default=0,
        info="Face information is stored in AutoEncoder dimensions. If there are not enough "
             "dimensions then certain facial features may not be recognized."
             "\nHigher number of dimensions are better, but require more VRAM."
             "\nSet to 0 to use the architecture defaults (256 for liae, 512 for df).",
        datatype=int,
        rounding=32,
        min_max=(0, 1024),
        fixed=True,
        group="network"),
    encoder_dims=dict(
        default=42,
        info="Encoder dimensions per channel. Higher number of encoder dimensions will help "
             "the model to recognize more facial features, but will require more VRAM.",
        datatype=int,
        rounding=1,
        min_max=(21, 85),
        fixed=True,
        group="network"),
    decoder_dims=dict(
        default=21,
        info="Decoder dimensions per channel. Higher number of decoder dimensions will help "
             "the model to improve details, but will require more VRAM.",
        datatype=int,
        rounding=1,
        min_max=(10, 85),
        fixed=True,
        group="network"),
    multiscale_decoder=dict(
        default=False,
        info="Multiscale decoder can help to obtain better details.",
        datatype=bool,
        fixed=True,
        group="network"))
