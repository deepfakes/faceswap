#!/usr/bin/env python3
"""
    The default options for the faceswap patch Writer plugin.

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
    "Options for outputting the raw converted face patches from faceswap\n"
    "The raw face patches are output along with the transformation matrix, per face, to "
    "transform the face back into the original frame in external tools"
)

_DEFAULTS = {
    "start_index": {
        "default": "0",
        "info": "The starting frame number for the first output frame.",
        "datatype": str,
        "choices": ["0", "1"],
        "group": "file_naming",
        "gui_radio": True,
    },
    "index_offset": {
        "default": 0,
        "info": "How much to offset the frame numbering by.",
        "datatype": int,
        "rounding": 1,
        "min_max": (0, 1000),
        "group": "file_naming",
    },
    "number_padding": {
        "default": 6,
        "info": "Length to pad the frame numbers by.",
        "datatype": int,
        "rounding": 6,
        "min_max": (0, 10),
        "group": "file_naming",
    },
    "include_filename": {
        "default": True,
        "info": "Prefix the filename of the original frame to each face patch's output filename.",
        "datatype": bool,
        "group": "file_naming",
    },
    "face_index_location": {
        "default": "before",
        "info": "For frames that contain multiple faces, where the face index should appear in "
                "the filename:"
                "\n\t before: places the face index before the frame number."
                "\n\t after: places the face index after the frame number.",
        "datatype": str,
        "choices": ["before", "after"],
        "group": "file_naming",
        "gui_radio": True,
    },
    "origin": {
        "default": "bottom-left",
        "info": "The origin (0, 0) location of the software that patches will be imported into. "
                "This impacts the transformation matrix that is supplied with the image patch. "
                "Setting the correct origin here will make importing into the external tool "
                "simpler."
                "\n\t top-left: The origin (0, 0) of the external canvas is at the top left "
                "corner."
                "\n\t bottom-left: The origin (0, 0) of the external canvas is at the bottom "
                "left corner."
                "\n\t top-right: The origin (0, 0) of the external canvas is at the top right "
                "corner."
                "\n\t bottom-right: The origin (0, 0) of the external canvas is at the bottom "
                "right corner.",
        "datatype": str,
        "choices": ["top-left", "bottom-left", "top-right", "bottom-right"],
        "group": "output",
        "gui_radio": True
    },
    "empty_frames": {
        "default": "blank",
        "info": "How to handle the output of frames without faces:"
                "\n\t skip: skips any frames that do not have a face within it. This will lead to "
                "gaps within the final image sequence."
                "\n\t blank: outputs a blank (empty) face patch for any frames without faces. "
                "There will be no gaps within the final image sequence, as those gaps will be "
                "padded with empty face patches",
        "datatype": str,
        "choices": ["skip", "blank"],
        "group": "output",
        "gui_radio": True,
    },
    "json_output": {
        "default": False,
        "info": "The transformation matrix, and other associated metadata, is output within the "
                "face images EXIF fields. Some external tools can read this data, others cannot."
                "enable this option to output a json file which contains this same metadata "
                "mapped to each output face patch's filename.",
        "datatype": bool,
        "group": "output"
    },
    "separate_mask": {
        "default": False,
        "info": "Seperate the mask into its own single channel patch. If enabled, the RGB image "
                "will be saved into the selected output folder whilst the masks will be saved "
                "into a sub-folder named `masks`. If not enabled then the mask will be included "
                "in the alpha-channel of the RGBA output.",
        "datatype": bool,
        "group": "output",
    },
    "bit_depth": {
        "default": "16",
        "info": "The bit-depth for the output images:"
                "\n\t 8: 8-bit unsigned - Supported by all formats."
                "\n\t 16: 16-bit unsigned - Supported by all formats."
                "\n\t 32: 32-bit float - Supported by Tiff only.",
        "datatype": str,
        "choices": ["8", "16", "32"],
        "group": "format",
        "gui_radio": True,
    },
    "format": {
        "default": "png",
        "info": "File format to save as."
                "\n\t png: PNG file format. Transformation matrix is written to the custom iTxt "
                "header field 'faceswap'"
                "\n\t tiff: TIFF file format. Transformation matrix is written to the "
                "'image_description' header field",
        "datatype": str,
        "choices": ["png", "tiff"],
        "group": "format",
        "gui_radio": True
    },
    "png_compress_level": {
        "default": 3,
        "info": "ZLIB compression level, 1 gives best speed, 9 gives best compression, 0 gives no "
                "compression at all.",
        "datatype": int,
        "rounding": 1,
        "min_max": (0, 9),
        "group": "format",
    },
    "tiff_compression_method": {
        "default": "lzw",
        "info": "The compression method to use for Tiff files. Note: For 32bit output, SGILOG "
                "compression will always be used regardless of what is selected here.",
        "datatype": str,
        "choices": ["none", "lzw", "deflate"],
        "group": "format",
        "gui_radio": True
    },
}
