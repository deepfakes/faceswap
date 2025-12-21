#!/usr/bin/env python3
""" The default options for the faceswap patch Writer plugin.

Defaults files should be named `<plugin_name>_defaults.py`

Any qualifying items placed into this file will automatically get added to the relevant config
.ini files within the faceswap/config folder and added to the relevant GUI settings page.

The following variable should be defined:

    Parameters
    ----------
    HELPTEXT: str
        A string describing what this plugin does

Further plugin configuration options are assigned using:
>>> <config_item> = ConfigItem(...)

where <config_item> is the name of the configuration option to be added (lower-case, alpha-numeric
+ underscore only) and ConfigItem(...) is the [`~lib.config.objects.ConfigItem`] data for the
option.

See the docstring/ReadtheDocs documentation required parameters for the ConfigItem object.
Items will be grouped together as per their `group` parameter, but otherwise will be processed in
the order that they are added to this module.
from lib.config import ConfigItem
"""
from lib.config import ConfigItem


HELPTEXT = (
    "Options for outputting the raw converted face patches from faceswap\n"
    "The raw face patches are output along with the transformation matrix, per face, to "
    "transform the face back into the original frame in external tools"
)

start_index = ConfigItem(
    default="0",
    info="The starting frame number for the first output frame.",
    datatype=str,
    choices=["0", "1"],
    group="file_naming",
    gui_radio=True)

index_offset = ConfigItem(
    default=0,
    datatype=int,
    group="file_naming",
    info="How much to offset the frame numbering by.",
    rounding=1,
    min_max=(0, 1000))

number_padding = ConfigItem(
    datatype=int,
    default=6,
    group="file_naming",
    info="Length to pad the frame numbers by.",
    rounding=6,
    min_max=(0, 10))

include_filename = ConfigItem(
    datatype=bool,
    default=True,
    group="file_naming",
    info="Prefix the filename of the original frame to each face patch's output filename.")

face_index_location = ConfigItem(
    datatype=str,
    default="before",
    group="file_naming",
    info="For frames that contain multiple faces, where the face index should appear in "
         "the filename:"
         "\n\t before: places the face index before the frame number."
         "\n\t after: places the face index after the frame number.",
    choices=["before", "after"],
    gui_radio=True)

origin = ConfigItem(
    datatype=str,
    default="bottom-left",
    group="output",
    info="The origin (0, 0) location of the software that patches will be imported into. "
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
    choices=["top-left", "bottom-left", "top-right", "bottom-right"],
    gui_radio=True)

empty_frames = ConfigItem(
    datatype=str,
    group="output",
    default="blank",
    info="How to handle the output of frames without faces:"
         "\n\t skip: skips any frames that do not have a face within it. This will lead to "
         "gaps within the final image sequence."
         "\n\t blank: outputs a blank (empty) face patch for any frames without faces. "
         "There will be no gaps within the final image sequence, as those gaps will be "
         "padded with empty face patches",
    choices=["skip", "blank"],
    gui_radio=True)

json_output = ConfigItem(
    datatype=bool,
    default=False,
    group="output",
    info="The transformation matrix, and other associated metadata, is output within the "
         "face images EXIF fields. Some external tools can read this data, others cannot."
         "enable this option to output a json file which contains this same metadata "
         "mapped to each output face patch's filename.")

separate_mask = ConfigItem(
    datatype=bool,
    default=False,
    group="output",
    info="Seperate the mask into its own single channel patch. If enabled, the RGB image "
         "will be saved into the selected output folder whilst the masks will be saved "
         "into a sub-folder named `masks`. If not enabled then the mask will be included "
         "in the alpha-channel of the RGBA output.")

bit_depth = ConfigItem(
    datatype=str,
    default="16",
    group="format",
    info="The bit-depth for the output images:"
         "\n\t 8: 8-bit unsigned - Supported by all formats."
         "\n\t 16: 16-bit unsigned - Supported by all formats."
         "\n\t 32: 32-bit float - Supported by Tiff only.",
    choices=["8", "16", "32"],
    gui_radio=True)

format = ConfigItem(  # pylint:disable=redefined-builtin
    datatype=str,
    default="png",
    group="format",
    info="File format to save as."
         "\n\t png: PNG file format. Transformation matrix is written to the custom iTxt "
         "header field 'faceswap'"
         "\n\t tiff: TIFF file format. Transformation matrix is written to the "
         "'image_description' header field",
    choices=["png", "tiff"],
    gui_radio=True)

png_compress_level = ConfigItem(
    datatype=int,
    default=3,
    group="format",
    info="ZLIB compression level, 1 gives best speed, 9 gives best compression, 0 gives no "
         "compression at all.",
    rounding=1,
    min_max=(0, 9))

tiff_compression_method = ConfigItem(
    datatype=str,
    default="lzw",
    group="format",
    info="The compression method to use for Tiff files. Note: For 32bit output, SGILOG "
         "compression will always be used regardless of what is selected here.",
    choices=["none", "lzw", "deflate"],
    gui_radio=True)
