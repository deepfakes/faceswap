#!/usr/bin/env python3
""" The default options for the faceswap Pillow Writer plugin.

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
# pylint:disable=duplicate-code
from lib.config import ConfigItem


HELPTEXT = (
    "Options for outputting converted frames to a series of images using Pillow\n"
    "Pillow is more feature rich than OpenCV but can be slower."
)


format = ConfigItem(  # pylint:disable=redefined-builtin
    group="format",
    datatype=str,
    default="png",
    info="Image format to use:"
         "\n\t bmp: Windows bitmap"
         "\n\t gif: Graphics Interchange Format (NB: Not animated)"
         "\n\t jpg: JPEG format"
         "\n\t jp2: JPEG 2000 format"
         "\n\t png: Portable Network Graphics"
         "\n\t ppm: Portable Pixmap Format"
         "\n\t tif: Tag Image File Format",
    choices=["bmp", "gif", "jpg", "jp2", "png", "ppm", "tif"],
    gui_radio=True)

draw_transparent = ConfigItem(
    datatype=bool,
    default=False,
    group="format",
    info="Place the swapped face on a transparent layer rather than the original frame.\nNB: "
         "This is only compatible with images saved in png or tif format. If an incompatible "
         "format is selected then the image will be saved as a png.")

separate_mask = ConfigItem(
    datatype=bool,
    default=False,
    group="format",
    info="Seperate the mask into its own single channel image. This only applies when "
         "'draw-transparent' is selected. If enabled, the RGB image will be saved into the "
         "selected output folder whilst the masks will be saved into a sub-folder named "
         "`masks`. If not enabled then the mask will be included in the alpha-channel of the "
         "RGBA output.")

optimize = ConfigItem(
    datatype=bool,
    default=False,
    group="settings",
    info="[gif, jpg and png only] If enabled, indicates that the encoder should make an extra "
         "pass over the image in order to select optimal encoder settings.")

gif_interlace = ConfigItem(
    datatype=bool,
    default=True,
    group="settings",
    info="[gif only] Set whether to save the gif as interlaced or not.")

jpg_quality = ConfigItem(
    datatype=int,
    default=75,
    group="compression",
    info="[jpg only] Set the jpg quality. 1 is worst 95 is best. Higher quality leads to "
         "larger file sizes.",
    rounding=1,
    min_max=(1, 95))

png_compress_level = ConfigItem(
    datatype=int,
    default=3,
    group="compression",
    info="[png only] ZLIB compression level, 1 gives best speed, 9 gives best compression, 0 "
         "gives no compression at all. When optimize option is set to True this has no effect "
         "(it is set to 9 regardless of a value passed).",
    rounding=1,
    min_max=(0, 9))

tif_compression = ConfigItem(
    datatype=str,
    default="tiff_deflate",
    group="compression",
    info="[tif only] The desired compression method for the file.",
    choices=["none", "tiff_ccitt", "group3", "group4", "tiff_jpeg", "tiff_adobe_deflate",
             "tiff_thunderscan", "tiff_deflate", "tiff_sgilog", "tiff_sgilog24",
             "tiff_raw_16"])
