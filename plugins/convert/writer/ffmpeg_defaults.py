#!/usr/bin/env python3
"""
    The default options for the faceswap Ffmpeg Writer plugin.

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


_HELPTEXT = "Options for encoding converted frames to video."


_DEFAULTS = dict(
    container=dict(
        default="mp4",
        info="Video container to use.",
        datatype=str,
        rounding=None,
        min_max=None,
        choices=["avi", "flv", "mkv", "mov", "mp4", "mpeg", "webm"],
        gui_radio=True,
    ),
    codec=dict(
        default="libx264",
        info="Video codec to use:"
             "\n\t libx264: H.264. A widely supported and commonly used codec."
             "\n\t libx265: H.265 / HEVC video encoder application library.",
        datatype=str,
        rounding=None,
        min_max=None,
        choices=["libx264", "libx265"],
        gui_radio=True,
    ),
    crf=dict(
        default=23,
        info="Constant Rate Factor:  0 is lossless and 51 is worst quality possible. A "
             "lower value generally leads to higher quality, and a subjectively sane range "
             "is 17-28. Consider 17 or 18 to be visually lossless or nearly so; it should "
             "look the same or nearly the same as the input but it isn't technically "
             "lossless.\nThe range is exponential, so increasing the CRF value +6 results "
             "in roughly half the bitrate / file size, while -6 leads to roughly twice the "
             "bitrate.",
        datatype=int,
        rounding=1,
        min_max=(0, 51),
        choices=[],
        gui_radio=False,
        group="quality",
    ),
    preset=dict(
        default="medium",
        info="A preset is a collection of options that will provide a certain encoding "
             "speed to compression ratio.\nA slower preset will provide better compression "
             "(compression is quality per filesize).\nUse the slowest preset that you have "
             "patience for.",
        datatype=str,
        rounding=None,
        min_max=None,
        choices=["ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow",
                 "slower", "veryslow"],
        gui_radio=True,
        group="quality",
    ),
    tune=dict(
        default="none",
        info="Change settings based upon the specifics of your input:"
             "\n\t none: Don't perform any additional tuning."
             "\n\t film: [H.264 only] Use for high quality movie content; lowers deblocking."
             "\n\t animation: [H.264 only] Good for cartoons; uses higher deblocking and more "
             "reference frames."
             "\n\t grain: Preserves the grain structure in old, grainy film material."
             "\n\t stillimage: [H.264 only] Good for slideshow-like content."
             "\n\t fastdecode: Allows faster decoding by disabling certain filters."
             "\n\t zerolatency: Good for fast encoding and low-latency streaming.",
        datatype=str,
        rounding=None,
        min_max=None,
        choices=["none", "film", "animation", "grain", "stillimage", "fastdecode", "zerolatency"],
        gui_radio=False,
        group="settings",
    ),
    profile=dict(
        default="auto",
        info="[H.264 Only] Limit the output to a specific H.264 profile. Don't change this "
             "unless your target device only supports a certain profile.",
        datatype=str,
        rounding=None,
        min_max=None,
        choices=["auto", "baseline", "main", "high", "high10", "high422", "high444"],
        gui_radio=False,
        group="settings",
    ),
    level=dict(
        default="auto",
        info="[H.264 Only] Set the encoder level, Don't change this unless your target "
             "device only supports a certain level.",
        datatype=str,
        rounding=None,
        min_max=None,
        choices=["auto", "1", "1b", "1.1", "1.2", "1.3", "2", "2.1", "2.2", "3", "3.1", "3.2", "4",
                 "4.1", "4.2", "5", "5.1", "5.2", "6", "6.1", "6.2"],
        gui_radio=False,
        group="settings",
    ),
    skip_mux=dict(
        default=False,
        info="Skip muxing audio to the final video output. This will result in a video without an "
             "audio track.",
        datatype=bool,
        group="settings",
    ),
)
