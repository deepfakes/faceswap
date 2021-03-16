#!/usr/bin/env python3
"""
    The default options for the faceswap Color_Transfer Color plugin.

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
    "Options for transfering the color distribution from the source to the target image using the "
    "mean and standard deviations of the L*a*b* color space.\nThis implementation is (loosely) "
    "based on the 'Color Transfer between Images' paper by Reinhard et al., 2001. matching the "
    "histograms between the source and destination faces."
)


_DEFAULTS = dict(
    clip=dict(
        default=True,
        info="Should components of L*a*b* image be scaled by np.clip before converting back to "
             "BGR color space?\nIf False then components will be min-max scaled appropriately.\n"
             "Clipping will keep target image brightness truer to the input.\nScaling will adjust "
             "image brightness to avoid washed out portions in the resulting color transfer that "
             "can be caused by clipping.",
        datatype=bool,
        group="method",
        rounding=None,
        min_max=None,
        choices=[],
        gui_radio=False,
        fixed=True,
    ),
    preserve_paper=dict(
        default=True,
        info="Should color transfer strictly follow methodology layed out in original paper?\nThe "
             "method does not always produce aesthetically pleasing results.\nIf False then "
             "L*a*b* components will be scaled using the reciprocal of the scaling factor "
             "proposed in the paper. This method seems to produce more consistently aesthetically "
             "pleasing results.",
        datatype=bool,
        group="method",
        rounding=None,
        min_max=None,
        choices=[],
        gui_radio=False,
        fixed=True,
    ),
)
