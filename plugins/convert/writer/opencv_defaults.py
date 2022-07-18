#!/usr/bin/env python3
"""
    The default options for the faceswap Opencv Writer plugin.

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
    "Options for outputting converted frames to a series of images using OpenCV\n"
    "OpenCV can be faster than other image writers, but lacks some configuration "
    "options and formats."
)


_DEFAULTS = dict(
    format=dict(
        default="png",
        info="Image format to use:"
             "\n\t bmp: Windows bitmap"
             "\n\t jpg: JPEG format"
             "\n\t jp2: JPEG 2000 format"
             "\n\t png: Portable Network Graphics"
             "\n\t ppm: Portable Pixmap Format",
        datatype=str,
        rounding=None,
        min_max=None,
        choices=["bmp", "jpg", "jp2", "png", "ppm"],
        group="format",
        gui_radio=True,
        fixed=True,
    ),
    draw_transparent=dict(
        default=False,
        info="Place the swapped face on a transparent layer rather than the original frame.\nNB: "
             "This is only compatible with images saved in png format. If an incompatible format "
             "is selected then the image will be saved as a png.",
        datatype=bool,
        rounding=None,
        min_max=None,
        choices=[],
        group="format",
        gui_radio=False,
        fixed=True,
    ),
    separate_mask=dict(
        default=False,
        info="Seperate the mask into its own single channel image. This only applies when "
             "'draw-transparent' is selected. If enabled, the RGB image will be saved into the "
             "selected output folder whilst the masks will be saved into a sub-folder named "
             "`masks`. If not enabled then the mask will be included in the alpha-channel of the "
             "RGBA output.",
        datatype=bool,
        rounding=None,
        min_max=None,
        choices=[],
        group="format",
        gui_radio=False,
        fixed=True,
    ),
    jpg_quality=dict(
        default=75,
        info="[jpg only] Set the jpg quality. 1 is worst 95 is best. Higher quality leads to "
             "larger file sizes.",
        datatype=int,
        rounding=1,
        min_max=(1, 95),
        choices=[],
        group="compression",
        gui_radio=False,
        fixed=True,
    ),
    png_compress_level=dict(
        default=3,
        info="[png only] ZLIB compression level, 1 gives best speed, 9 gives best compression, 0 "
             "gives no compression at all.",
        datatype=int,
        rounding=1,
        min_max=(0, 9),
        choices=[],
        group="compression",
        gui_radio=False,
        fixed=True,
    ),
)
