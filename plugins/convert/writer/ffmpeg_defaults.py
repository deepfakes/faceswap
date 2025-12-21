#!/usr/bin/env python3
""" The default options for the faceswap Ffmpeg Writer plugin.

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


HELPTEXT = "Options for encoding converted frames to video."


container = ConfigItem(
    datatype=str,
    default="mp4",
    group="codec",
    info="Video container to use.",
    choices=["avi", "flv", "mkv", "mov", "mp4", "mpeg", "webm"],
    gui_radio=True)

codec = ConfigItem(
    datatype=str,
    default="libx264",
    group="codec",
    info="Video codec to use:"
         "\n\t libx264: H.264. A widely supported and commonly used codec."
         "\n\t libx265: H.265 / HEVC video encoder application library.",
    choices=["libx264", "libx265"],
    gui_radio=True)

crf = ConfigItem(
    datatype=int,
    default=23,
    group="quality",
    info="Constant Rate Factor:  0 is lossless and 51 is worst quality possible. A "
         "lower value generally leads to higher quality, and a subjectively sane range "
         "is 17-28. Consider 17 or 18 to be visually lossless or nearly so; it should "
         "look the same or nearly the same as the input but it isn't technically "
         "lossless.\nThe range is exponential, so increasing the CRF value +6 results "
         "in roughly half the bitrate / file size, while -6 leads to roughly twice the "
         "bitrate.",
    rounding=1,
    min_max=(0, 51))

preset = ConfigItem(
    datatype=str,
    default="medium",
    group="quality",
    info="A preset is a collection of options that will provide a certain encoding "
         "speed to compression ratio.\nA slower preset will provide better compression "
         "(compression is quality per filesize).\nUse the slowest preset that you have "
         "patience for.",
    choices=["ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow",
             "slower", "veryslow"],
    gui_radio=True)

tune = ConfigItem(
    datatype=str,
    default="none",
    group="settings",
    info="Change settings based upon the specifics of your input:"
         "\n\t none: Don't perform any additional tuning."
         "\n\t film: [H.264 only] Use for high quality movie content; lowers deblocking."
         "\n\t animation: [H.264 only] Good for cartoons; uses higher deblocking and more "
         "reference frames."
         "\n\t grain: Preserves the grain structure in old, grainy film material."
         "\n\t stillimage: [H.264 only] Good for slideshow-like content."
         "\n\t fastdecode: Allows faster decoding by disabling certain filters."
         "\n\t zerolatency: Good for fast encoding and low-latency streaming.",
    choices=["none", "film", "animation", "grain", "stillimage", "fastdecode", "zerolatency"])

profile = ConfigItem(
    datatype=str,
    default="auto",
    group="settings",
    info="[H.264 Only] Limit the output to a specific H.264 profile. Don't change this "
         "unless your target device only supports a certain profile.",
    choices=["auto", "baseline", "main", "high", "high10", "high422", "high444"])

level = ConfigItem(
    datatype=str,
    default="auto",
    group="settings",
    info="[H.264 Only] Set the encoder level, Don't change this unless your target "
         "device only supports a certain level.",
    choices=["auto", "1", "1b", "1.1", "1.2", "1.3", "2", "2.1", "2.2", "3", "3.1", "3.2", "4",
             "4.1", "4.2", "5", "5.1", "5.2", "6", "6.1", "6.2"])

skip_mux = ConfigItem(
    datatype=bool,
    default=False,
    group="settings",
    info="Skip muxing audio to the final video output. This will result in a video without an "
         "audio track.")
