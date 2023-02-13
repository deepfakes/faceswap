#!/usr/bin/env python3
"""
    The default options for the faceswap Original Model plugin.

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


_HELPTEXT = ("Original Trainer Options.\n"
             "WARNING: The defaults for augmentation will be fine for 99.9% of use cases. "
             "Only change them if you absolutely know what you are doing!")


_DEFAULTS = dict(
    preview_images=dict(
        default=14,
        info="Number of sample faces to display for each side in the preview when training.",
        datatype=int,
        rounding=2,
        min_max=(2, 16),
        group="evaluation"),
    mask_opacity=dict(
        default=30,
        info="The opacity of the mask overlay in the training preview. Lower values are more "
             "transparent.",
        datatype=int,
        rounding=2,
        min_max=(0, 100),
        group="evaluation"),
    mask_color=dict(
        default="#ff0000",
        choices="colorchooser",
        info="The RGB hex color to use for the mask overlay in the training preview.",
        datatype=str,
        group="evaluation"),
    zoom_amount=dict(
        default=5,
        info="Percentage amount to randomly zoom each training image in and out.",
        datatype=int,
        rounding=1,
        min_max=(0, 25),
        group="image augmentation"),
    rotation_range=dict(
        default=10,
        info="Percentage amount to randomly rotate each training image.",
        datatype=int,
        rounding=1,
        min_max=(0, 25),
        group="image augmentation"),
    shift_range=dict(
        default=5,
        info="Percentage amount to randomly shift each training image horizontally and "
             "vertically.",
        datatype=int,
        rounding=1,
        min_max=(0, 25),
        group="image augmentation"),
    flip_chance=dict(
        default=50,
        info="Percentage chance to randomly flip each training image horizontally.\n"
             "NB: This is ignored if the 'no-flip' option is enabled",
        datatype=int,
        rounding=1,
        min_max=(0, 75),
        group="image augmentation"),

    color_lightness=dict(
        default=30,
        info="Percentage amount to randomly alter the lightness of each training image.\n"
             "NB: This is ignored if the 'no-augment-color' option is enabled",
        datatype=int,
        rounding=1,
        min_max=(0, 75),
        group="color augmentation"),
    color_ab=dict(
        default=8,
        info="Percentage amount to randomly alter the 'a' and 'b' colors of the L*a*b* color "
             "space of each training image.\nNB: This is ignored if the 'no-augment-color' option"
             "is enabled",
        datatype=int,
        rounding=1,
        min_max=(0, 50),
        group="color augmentation"),
    color_clahe_chance=dict(
        default=50,
        info="Percentage chance to perform Contrast Limited Adaptive Histogram Equalization on "
             "each training image.\nNB: This is ignored if the 'no-augment-color' option is "
             "enabled",
        datatype=int,
        rounding=1,
        min_max=(0, 75),
        fixed=False,
        group="color augmentation"),
    color_clahe_max_size=dict(
        default=4,
        info="The grid size dictates how much Contrast Limited Adaptive Histogram Equalization is "
             "performed on any training image selected for clahe. Contrast will be applied "
             "randomly with a gridsize of 0 up to the maximum. This value is a multiplier "
             "calculated from the training image size.\nNB: This is ignored if the "
             "'no-augment-color' option is enabled",
        datatype=int,
        rounding=1,
        min_max=(1, 8),
        group="color augmentation"),
)
