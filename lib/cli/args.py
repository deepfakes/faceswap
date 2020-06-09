#!/usr/bin/env python3
""" The Command Line Argument options for faceswap.py """
import argparse
import logging
import re
import sys
import textwrap

from lib.utils import get_backend
from plugins.plugin_loader import PluginLoader

from .actions import (DirFullPaths, DirOrFileFullPaths, FileFullPaths, FilesFullPaths, MultiOption,
                      Radio, SaveFileFullPaths, Slider)
from .launcher import ScriptExecutor

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class FullHelpArgumentParser(argparse.ArgumentParser):
    """ Extends :class:`argparse.ArgumentParser` to output full help on bad arguments. """
    def error(self, message):
        self.print_help(sys.stderr)
        self.exit(2, "{}: error: {}\n".format(self.prog, message))


class SmartFormatter(argparse.HelpFormatter):
    """ Extends the class :class:`argparse.HelpFormatter` to allow custom formatting in help text.

    Adapted from: https://stackoverflow.com/questions/3853722

    Notes
    -----
    Prefix help text with "R|" to override default formatting and use explicitly defined formatting
    within the help text.
    Prefixing a new line within the help text with "L|" will turn that line into a list item in
    both the cli help text and the GUI.
    """
    def __init__(self, prog, indent_increment=2, max_help_position=24, width=None):
        super().__init__(prog, indent_increment, max_help_position, width)
        self._whitespace_matcher_limited = re.compile(r'[ \r\f\v]+', re.ASCII)

    def _split_lines(self, text, width):
        """ Split the given text by the given display width.

        If the text is not prefixed with "R|" then the standard
        :func:`argparse.HelpFormatter._split_lines` function is used, otherwise raw
        formatting is processed,

        Parameters
        ----------
        text: str
            The help text that is to be formatted for display
        width: int
            The display width, in characters, for the help text
        """
        if text.startswith("R|"):
            text = self._whitespace_matcher_limited.sub(' ', text).strip()[2:]
            output = list()
            for txt in text.splitlines():
                indent = ""
                if txt.startswith("L|"):
                    indent = "    "
                    txt = "  - {}".format(txt[2:])
                output.extend(textwrap.wrap(txt, width, subsequent_indent=indent))
            return output
        return argparse.HelpFormatter._split_lines(self, text, width)


class FaceSwapArgs():
    """ Faceswap argument parser functions that are universal to all commands.

    This is the parent class to all subsequent argparsers which holds global arguments that pertain
    to all commands.

    Process the incoming command line arguments, validates then launches the relevant faceswap
    script with the given arguments.

    Parameters
    ----------
    subparser: :class:`argparse._SubParsersAction`
        The subparser for the given command
    command: str
        The faceswap command that is to be executed
    description: str, optional
        The description for the given command. Default: "default"
    """
    def __init__(self, subparser, command, description="default"):
        self.global_arguments = self._get_global_arguments()
        self.info = self.get_info()
        self.argument_list = self.get_argument_list()
        self.optional_arguments = self.get_optional_arguments()
        self._process_suppressions()
        if not subparser:
            return
        self.parser = self._create_parser(subparser, command, description)
        self._add_arguments()
        script = ScriptExecutor(command)
        self.parser.set_defaults(func=script.execute_script)

    @staticmethod
    def get_info():
        """ Returns the information text for the current command.

        This function should be overridden with the actual command help text for each
        commands' parser.

        Returns
        -------
        str
            The information text for this command.
        """
        return None

    @staticmethod
    def get_argument_list():
        """ Returns the argument list for the current command.

        The argument list should be a list of dictionaries pertaining to each option for a command.
        This function should be overridden with the actual argument list for each command's
        argument list.

        See existing parsers for examples.

        Returns
        -------
        list
            The list of command line options for the given command
        """
        argument_list = []
        return argument_list

    @staticmethod
    def get_optional_arguments():
        """ Returns the optional argument list for the current command.

        The optional arguments list is not always required, but is used when there are shared
        options between multiple commands (e.g. convert and extract). Only override if required.

        Returns
        -------
        list
            The list of optional command line options for the given command
        """
        argument_list = []
        return argument_list

    @staticmethod
    def _get_global_arguments():
        """ Returns the global Arguments list that are required for ALL commands in Faceswap.

        This method should NOT be overridden.

        Returns
        -------
        list
            The list of global command line options for all Faceswap commands.
        """
        global_args = list()
        global_args.append(dict(
            opts=("-C", "--configfile"),
            action=FileFullPaths,
            filetypes="ini",
            type=str,
            group="Global Options",
            help="Optionally overide the saved config with the path to a custom config file."))
        global_args.append(dict(
            opts=("-L", "--loglevel"),
            type=str.upper,
            dest="loglevel",
            default="INFO",
            choices=("INFO", "VERBOSE", "DEBUG", "TRACE"),
            group="Global Options",
            help="Log level. Stick with INFO or VERBOSE unless you need to file an error report. "
                 "Be careful with TRACE as it will generate a lot of data"))
        global_args.append(dict(
            opts=("-LF", "--logfile"),
            action=SaveFileFullPaths,
            filetypes='log',
            type=str,
            dest="logfile",
            default=None,
            group="Global Options",
            help="Path to store the logfile. Leave blank to store in the faceswap folder"))
        # These are hidden arguments to indicate that the GUI/Colab is being used
        global_args.append(dict(
            opts=("-gui", "--gui"),
            action="store_true",
            dest="redirect_gui",
            default=False,
            help=argparse.SUPPRESS))
        global_args.append(dict(
            opts=("-colab", "--colab"),
            action="store_true",
            dest="colab",
            default=False,
            help=argparse.SUPPRESS))
        return global_args

    @staticmethod
    def _create_parser(subparser, command, description):
        """ Create the parser for the selected command.

        Parameters
        ----------
        command: str
            The faceswap command that is to be executed
        description: str
            The description for the given command

        Returns
        -------
        :class:`~lib.cli.args.FullHelpArgumentParser`
            The parser for the given command
        """
        parser = subparser.add_parser(command,
                                      help=description,
                                      description=description,
                                      epilog="Questions and feedback: https://faceswap.dev/forum",
                                      formatter_class=SmartFormatter)
        return parser

    def _add_arguments(self):
        """ Parse the list of dictionaries containing the command line arguments and convert to
        argparse parser arguments. """
        options = self.global_arguments + self.argument_list + self.optional_arguments
        for option in options:
            args = option["opts"]
            kwargs = {key: option[key] for key in option.keys() if key not in ("opts", "group")}
            self.parser.add_argument(*args, **kwargs)

    def _process_suppressions(self):
        """ Certain options are only available for certain backends.

        Suppresses command line options that are not available for the running backend.
        """
        fs_backend = get_backend()
        for opt_list in [self.global_arguments, self.argument_list, self.optional_arguments]:
            for opts in opt_list:
                if opts.get("backend", None) is None:
                    continue
                opt_backend = opts.pop("backend")
                if isinstance(opt_backend, (list, tuple)):
                    opt_backend = [backend.lower() for backend in opt_backend]
                else:
                    opt_backend = [opt_backend.lower()]
                if fs_backend not in opt_backend:
                    opts["help"] = argparse.SUPPRESS


class ExtractConvertArgs(FaceSwapArgs):
    """ Parent class to capture arguments that will be used in both extract and convert processes.

    Extract and Convert share a fair amount of arguments, so arguments that can be used in both of
    these processes should be placed here.

    No further processing is done in this class (this is handled by the children), this just
    captures the shared arguments.
    """

    @staticmethod
    def get_argument_list():
        """ Returns the argument list for shared Extract and Convert arguments.

        Returns
        -------
        list
            The list of command line options for the given Extract and Convert
        """
        argument_list = list()
        argument_list.append(dict(
            opts=("-i", "--input-dir"),
            action=DirOrFileFullPaths,
            filetypes="video",
            dest="input_dir",
            required=True,
            group="Data",
            help="Input directory or video. Either a directory containing the image files you "
                 "wish to process or path to a video file. NB: This should be the source video/"
                 "frames NOT the source faces."))
        argument_list.append(dict(
            opts=("-o", "--output-dir"),
            action=DirFullPaths,
            dest="output_dir",
            required=True,
            group="Data",
            help="Output directory. This is where the converted files will be saved."))
        argument_list.append(dict(
            opts=("-al", "--alignments"),
            action=FileFullPaths,
            filetypes="alignments",
            type=str,
            dest="alignments_path",
            group="Data",
            help="Optional path to an alignments file. Leave blank if the alignments file is "
                 "at the default location."))
        return argument_list


class ExtractArgs(ExtractConvertArgs):
    """ Creates the command line arguments for extraction.

    This class inherits base options from :class:`ExtractConvertArgs` where arguments that are used
    for both Extract and Convert should be placed.

    Commands explicit to Extract should be added in :func:`get_optional_arguments`
    """

    @staticmethod
    def get_info():
        """ The information text for the Extract command.

        Returns
        -------
        str
            The information text for the Extract command.
        """
        return ("Extract faces from image or video sources.\n"
                "Extraction plugins can be configured in the 'Settings' Menu")

    @staticmethod
    def get_optional_arguments():
        """ Returns the argument list unique to the Extract command.

        Returns
        -------
        list
            The list of optional command line options for the Extract command
        """
        if get_backend() == "cpu":
            default_detector = default_aligner = "cv2-dnn"
        else:
            default_detector = "s3fd"
            default_aligner = "fan"

        argument_list = []
        argument_list.append(dict(
            opts=("-D", "--detector"),
            action=Radio,
            type=str.lower,
            default=default_detector,
            choices=PluginLoader.get_available_extractors("detect"),
            group="Plugins",
            help="R|Detector to use. Some of these have configurable settings in "
                 "'/config/extract.ini' or 'Settings > Configure Extract 'Plugins':"
                 "\nL|cv2-dnn: A CPU only extractor which is the least reliable and least "
                 "resource intensive. Use this if not using a GPU and time is important."
                 "\nL|mtcnn: Good detector. Fast on CPU, faster on GPU. Uses fewer resources "
                 "than other GPU detectors but can often return more false positives."
                 "\nL|s3fd: Best detector. Fast on GPU, slow on CPU. Can detect more faces and "
                 "fewer false positives than other GPU detectors, but is a lot more resource "
                 "intensive."))
        argument_list.append(dict(
            opts=("-A", "--aligner"),
            action=Radio,
            type=str.lower,
            default=default_aligner,
            choices=PluginLoader.get_available_extractors("align"),
            group="Plugins",
            help="R|Aligner to use."
                 "\nL|cv2-dnn: A CPU only landmark detector. Faster, less resource intensive, "
                 "but less accurate. Only use this if not using a GPU and time is important."
                 "\nL|fan: Best aligner. Fast on GPU, slow on CPU."))
        argument_list.append(dict(
            opts=("-M", "--masker"),
            action=MultiOption,
            type=str.lower,
            nargs="+",
            choices=[mask for mask in PluginLoader.get_available_extractors("mask")
                     if mask not in ("components", "extended")],
            group="Plugins",
            help="R|Additional Masker(s) to use. The masks generated here will all take up GPU "
                 "RAM. You can select none, one or multiple masks, but the extraction may take "
                 "longer the more you select. NB: The Extended and Components (landmark based) "
                 "masks are automatically generated on extraction."
                 "\nL|vgg-clear: Mask designed to provide smart segmentation of mostly frontal "
                 "faces clear of obstructions. Profile faces and obstructions may result in "
                 "sub-par performance."
                 "\nL|vgg-obstructed: Mask designed to provide smart segmentation of mostly "
                 "frontal faces. The mask model has been specifically trained to recognize "
                 "some facial obstructions (hands and eyeglasses). Profile faces may result in "
                 "sub-par performance."
                 "\nL|unet-dfl: Mask designed to provide smart segmentation of mostly frontal "
                 "faces. The mask model has been trained by community members and will need "
                 "testing for further description. Profile faces may result in sub-par "
                 "performance."
                 "\nThe auto generated masks are as follows:"
                 "\nL|components: Mask designed to provide facial segmentation based on the "
                 "positioning of landmark locations. A convex hull is constructed around the "
                 "exterior of the landmarks to create a mask."
                 "\nL|extended: Mask designed to provide facial segmentation based on the "
                 "positioning of landmark locations. A convex hull is constructed around the "
                 "exterior of the landmarks and the mask is extended upwards onto the "
                 "forehead."
                 "\n(eg: `-M unet-dfl vgg-clear`, `--masker vgg-obstructed`)"))
        argument_list.append(dict(
            opts=("-nm", "--normalization"),
            action=Radio,
            type=str.lower,
            dest="normalization",
            default="none",
            choices=["none", "clahe", "hist", "mean"],
            group="plugins",
            help="R|Performing normalization can help the aligner better align faces with "
                 "difficult lighting conditions at an extraction speed cost. Different methods "
                 "will yield different results on different sets. NB: This does not impact the "
                 "output face, just the input to the aligner."
                 "\nL|none: Don't perform normalization on the face."
                 "\nL|clahe: Perform Contrast Limited Adaptive Histogram Equalization on the "
                 "face."
                 "\nL|hist: Equalize the histograms on the RGB channels."
                 "\nL|mean: Normalize the face colors to the mean."))
        argument_list.append(dict(
            opts=("-r", "--rotate-images"),
            type=str,
            dest="rotate_images",
            default=None,
            group="plugins",
            help="If a face isn't found, rotate the images to try to find a face. Can find "
                 "more faces at the cost of extraction speed. Pass in a single number to use "
                 "increments of that size up to 360, or pass in a list of numbers to enumerate "
                 "exactly what angles to check."))
        argument_list.append(dict(
            opts=("-min", "--min-size"),
            action=Slider,
            min_max=(0, 1080),
            rounding=20,
            type=int,
            dest="min_size",
            default=0,
            group="Face Processing",
            help="Filters out faces detected below this size. Length, in pixels across the "
                 "diagonal of the bounding box. Set to 0 for off"))
        argument_list.append(dict(
            opts=("-n", "--nfilter"),
            action=FilesFullPaths,
            filetypes="image",
            dest="nfilter",
            default=None,
            nargs="+",
            group="Face Processing",
            help="Optionally filter out people who you do not wish to process by passing in an "
                 "image of that person. Should be a front portrait with a single person in the "
                 "image. Multiple images can be added space separated. NB: Using face filter "
                 "will significantly decrease extraction speed and its accuracy cannot be "
                 "guaranteed."))
        argument_list.append(dict(
            opts=("-f", "--filter"),
            action=FilesFullPaths,
            filetypes="image",
            dest="filter",
            default=None,
            nargs="+",
            group="Face Processing",
            help="Optionally select people you wish to process by passing in an image of that "
                 "person. Should be a front portrait with a single person in the image. "
                 "Multiple images can be added space separated. NB: Using face filter will "
                 "significantly decrease extraction speed and its accuracy cannot be "
                 "guaranteed."))
        argument_list.append(dict(
            opts=("-l", "--ref_threshold"),
            action=Slider,
            min_max=(0.01, 0.99),
            rounding=2,
            type=float,
            dest="ref_threshold",
            default=0.4,
            group="Face Processing",
            help="For use with the optional nfilter/filter files. Threshold for positive face "
                 "recognition. Lower values are stricter. NB: Using face filter will "
                 "significantly decrease extraction speed and its accuracy cannot be "
                 "guaranteed."))
        argument_list.append(dict(
            opts=("-een", "--extract-every-n"),
            action=Slider,
            min_max=(1, 100),
            rounding=1,
            type=int,
            dest="extract_every_n",
            default=1,
            group="output",
            help="Extract every 'nth' frame. This option will skip frames when extracting "
                 "faces. For example a value of 1 will extract faces from every frame, a value "
                 "of 10 will extract faces from every 10th frame."))
        argument_list.append(dict(
            opts=("-sz", "--size"),
            action=Slider,
            min_max=(128, 512),
            rounding=64,
            type=int,
            default=256,
            group="output",
            help="The output size of extracted faces. Make sure that the model you intend to "
                 "train supports your required size. This will only need to be changed for "
                 "hi-res models."))
        argument_list.append(dict(
            opts=("-si", "--save-interval"),
            action=Slider,
            min_max=(0, 1000),
            rounding=10,
            type=int,
            dest="save_interval",
            default=0,
            group="output",
            help="Automatically save the alignments file after a set amount of frames. By "
                 "default the alignments file is only saved at the end of the extraction "
                 "process. NB: If extracting in 2 passes then the alignments file will only "
                 "start to be saved out during the second pass. WARNING: Don't interrupt the "
                 "script when writing the file because it might get corrupted. Set to 0 to "
                 "turn off"))
        argument_list.append(dict(
            opts=("-dl", "--debug-landmarks"),
            action="store_true",
            dest="debug_landmarks",
            default=False,
            group="output",
            help="Draw landmarks on the ouput faces for debugging purposes."))
        argument_list.append(dict(
            opts=("-sp", "--singleprocess"),
            action="store_true",
            default=False,
            backend="nvidia",
            group="settings",
            help="Don't run extraction in parallel. Will run each part of the extraction "
                 "process separately (one after the other) rather than all at the smae time. "
                 "Useful if VRAM is at a premium."))
        argument_list.append(dict(
            opts=("-s", "--skip-existing"),
            action="store_true",
            dest="skip_existing",
            default=False,
            group="settings",
            help="Skips frames that have already been extracted and exist in the alignments "
                 "file"))
        argument_list.append(dict(
            opts=("-sf", "--skip-existing-faces"),
            action="store_true",
            dest="skip_faces",
            default=False,
            group="settings",
            help="Skip frames that already have detected faces in the alignments file"))
        argument_list.append(dict(
            opts=("-ssf", "--skip-saving-faces"),
            action="store_true",
            dest="skip_saving_faces",
            default=False,
            group="settings",
            help="Skip saving out the face images"))
        return argument_list


class ConvertArgs(ExtractConvertArgs):
    """ Creates the command line arguments for conversion.

    This class inherits base options from :class:`ExtractConvertArgs` where arguments that are used
    for both Extract and Convert should be placed.

    Commands explicit to Convert should be added in :func:`get_optional_arguments`
    """

    @staticmethod
    def get_info():
        """ The information text for the Convert command.

        Returns
        -------
        str
            The information text for the Convert command.
        """
        return ("Swap the original faces in a source video/images to your final faces.\n"
                "Conversion plugins can be configured in the 'Settings' Menu")

    @staticmethod
    def get_optional_arguments():
        """ Returns the argument list unique to the Convert command.

        Returns
        -------
        list
            The list of optional command line options for the Convert command
        """

        argument_list = []
        argument_list.append(dict(
            opts=("-ref", "--reference-video"),
            action=FileFullPaths,
            filetypes="video",
            type=str,
            dest="reference_video",
            group="data",
            help="Only required if converting from images to video. Provide The original video "
                 "that the source frames were extracted from (for extracting the fps and "
                 "audio)."))
        argument_list.append(dict(
            opts=("-m", "--model-dir"),
            action=DirFullPaths,
            dest="model_dir",
            required=True,
            group="data",
            help="Model directory. The directory containing the trained model you wish to use "
                 "for conversion."))
        argument_list.append(dict(
            opts=("-c", "--color-adjustment"),
            action=Radio,
            type=str.lower,
            dest="color_adjustment",
            default="avg-color",
            choices=PluginLoader.get_available_convert_plugins("color", True),
            group="plugins",
            help="R|Performs color adjustment to the swapped face. Some of these options have "
                 "configurable settings in '/config/convert.ini' or 'Settings > Configure "
                 "Convert Plugins':"
                 "\nL|avg-color: Adjust the mean of each color channel in the swapped "
                 "reconstruction to equal the mean of the masked area in the original image."
                 "\nL|color-transfer: Transfers the color distribution from the source to the "
                 "target image using the mean and standard deviations of the L*a*b* "
                 "color space."
                 "\nL|manual-balance: Manually adjust the balance of the image in a variety of "
                 "color spaces. Best used with the Preview tool to set correct values."
                 "\nL|match-hist: Adjust the histogram of each color channel in the swapped "
                 "reconstruction to equal the histogram of the masked area in the original "
                 "image."
                 "\nL|seamless-clone: Use cv2's seamless clone function to remove extreme "
                 "gradients at the mask seam by smoothing colors. Generally does not give "
                 "very satisfactory results."
                 "\nL|none: Don't perform color adjustment."))
        argument_list.append(dict(
            opts=("-M", "--mask-type"),
            action=Radio,
            type=str.lower,
            dest="mask_type",
            default="extended",
            choices=PluginLoader.get_available_extractors("mask", add_none=True) + ["predicted"],
            group="Plugins",
            help="R|Masker to use. NB: The mask you require must exist within the alignments "
                 "file. You can add additional masks with the Mask Tool."
                 "\nL|none: Don't use a mask."
                 "\nL|components: Mask designed to provide facial segmentation based on the "
                 "positioning of landmark locations. A convex hull is constructed around the "
                 "exterior of the landmarks to create a mask."
                 "\nL|extended: Mask designed to provide facial segmentation based on the "
                 "positioning of landmark locations. A convex hull is constructed around the "
                 "exterior of the landmarks and the mask is extended upwards onto the forehead."
                 "\nL|vgg-clear: Mask designed to provide smart segmentation of mostly frontal "
                 "faces clear of obstructions. Profile faces and obstructions may result in "
                 "sub-par performance."
                 "\nL|vgg-obstructed: Mask designed to provide smart segmentation of mostly "
                 "frontal faces. The mask model has been specifically trained to recognize "
                 "some facial obstructions (hands and eyeglasses). Profile faces may result in "
                 "sub-par performance."
                 "\nL|unet-dfl: Mask designed to provide smart segmentation of mostly frontal "
                 "faces. The mask model has been trained by community members and will need "
                 "testing for further description. Profile faces may result in sub-par "
                 "performance."))
        argument_list.append(dict(
            opts=("-sc", "--scaling"),
            action=Radio,
            type=str.lower,
            default="none",
            choices=PluginLoader.get_available_convert_plugins("scaling", True),
            group="plugins",
            help="R|Performs a scaling process to attempt to get better definition on the "
                 "final swap. Some of these options have configurable settings in "
                 "'/config/convert.ini' or 'Settings > Configure Convert Plugins':"
                 "\nL|sharpen: Perform sharpening on the final face."
                 "\nL|none: Don't perform any scaling operations."))
        argument_list.append(dict(
            opts=("-w", "--writer"),
            action=Radio,
            type=str,
            default="opencv",
            choices=PluginLoader.get_available_convert_plugins("writer", False),
            group="plugins",
            help="R|The plugin to use to output the converted images. The writers are "
                 "configurable in '/config/convert.ini' or 'Settings > Configure Convert "
                 "Plugins:'"
                 "\nL|ffmpeg: [video] Writes out the convert straight to video. When the input "
                 "is a series of images then the '-ref' (--reference-video) parameter must be "
                 "set."
                 "\nL|gif: [animated image] Create an animated gif."
                 "\nL|opencv: [images] The fastest image writer, but less options and formats "
                 "than other plugins."
                 "\nL|pillow: [images] Slower than opencv, but has more options and supports "
                 "more formats."))
        argument_list.append(dict(
            opts=("-osc", "--output-scale"),
            action=Slider,
            min_max=(25, 400),
            rounding=1,
            type=int,
            dest="output_scale",
            default=100,
            group="Frame Processing",
            help="Scale the final output frames by this amount. 100%% will output the frames "
                 "at source dimensions. 50%% at half size 200%% at double size"))
        argument_list.append(dict(
            opts=("-fr", "--frame-ranges"),
            type=str,
            nargs="+",
            group="Frame Processing",
            help="Frame ranges to apply transfer to e.g. For frames 10 to 50 and 90 to 100 use "
                 "--frame-ranges 10-50 90-100. Frames falling outside of the selected range "
                 "will be discarded unless '-k' (--keep-unchanged) is selected. NB: If you are "
                 "converting from images, then the filenames must end with the frame-number!"))
        argument_list.append(dict(
            opts=("-a", "--input-aligned-dir"),
            action=DirFullPaths,
            dest="input_aligned_dir",
            default=None,
            group="Face Processing",
            help="If you have not cleansed your alignments file, then you can filter out faces "
                 "by defining a folder here that contains the faces extracted from your input "
                 "files/video. If this folder is defined, then only faces that exist within "
                 "your alignments file and also exist within the specified folder will be "
                 "converted. Leaving this blank will convert all faces that exist within the "
                 "alignments file."))
        argument_list.append(dict(
            opts=("-n", "--nfilter"),
            action=FilesFullPaths,
            filetypes="image",
            dest="nfilter",
            default=None,
            nargs="+",
            group="Face Processing",
            help="Optionally filter out people who you do not wish to process by passing in an "
                 "image of that person. Should be a front portrait with a single person in the "
                 "image. Multiple images can be added space separated. NB: Using face filter "
                 "will significantly decrease extraction speed and its accuracy cannot be "
                 "guaranteed."))
        argument_list.append(dict(
            opts=("-f", "--filter"),
            action=FilesFullPaths,
            filetypes="image",
            dest="filter",
            default=None,
            nargs="+",
            group="Face Processing",
            help="Optionally select people you wish to process by passing in an image of that "
                 "person. Should be a front portrait with a single person in the image. "
                 "Multiple images can be added space separated. NB: Using face filter will "
                 "significantly decrease extraction speed and its accuracy cannot be "
                 "guaranteed."))
        argument_list.append(dict(
            opts=("-l", "--ref_threshold"),
            action=Slider,
            min_max=(0.01, 0.99),
            rounding=2,
            type=float,
            dest="ref_threshold",
            default=0.4,
            group="Face Processing",
            help="For use with the optional nfilter/filter files. Threshold for positive face "
                 "recognition. Lower values are stricter. NB: Using face filter will "
                 "significantly decrease extraction speed and its accuracy cannot be "
                 "guaranteed."))
        argument_list.append(dict(
            opts=("-j", "--jobs"),
            action=Slider,
            min_max=(0, 40),
            rounding=1,
            type=int,
            dest="jobs",
            default=0,
            group="settings",
            help="The maximum number of parallel processes for performing conversion. "
                 "Converting images is system RAM heavy so it is possible to run out of memory "
                 "if you have a lot of processes and not enough RAM to accommodate them all. "
                 "Setting this to 0 will use the maximum available. No matter what you set "
                 "this to, it will never attempt to use more processes than are available on "
                 "your system. If singleprocess is enabled this setting will be ignored."))
        argument_list.append(dict(
            opts=("-g", "--gpus"),
            action=Slider,
            min_max=(1, 10),
            rounding=1,
            type=int,
            default=1,
            backend="nvidia",
            group="settings",
            help="Number of GPUs to use for conversion"))
        argument_list.append(dict(
            opts=("-t", "--trainer"),
            type=str.lower,
            choices=PluginLoader.get_available_models(),
            group="settings",
            help="[LEGACY] This only needs to be selected if a legacy model is being loaded or "
                 "if there are multiple models in the model folder"))
        argument_list.append(dict(
            opts=("-ag", "--allow-growth"),
            action="store_true",
            dest="allow_growth",
            default=False,
            backend="nvidia",
            group="settings",
            help="Sets allow_growth option of Tensorflow to spare memory on some "
                 "configurations."))
        argument_list.append(dict(
            opts=("-otf", "--on-the-fly"),
            action="store_true",
            dest="on_the_fly",
            default=False,
            group="settings",
            help="Enable On-The-Fly Conversion. NOT recommended. You should generate a clean "
                 "alignments file for your destination video. However, if you wish you can "
                 "generate the alignments on-the-fly by enabling this option. This will use "
                 "an inferior extraction pipeline and will lead to substandard results. If an "
                 "alignments file is found, this option will be ignored."))
        argument_list.append(dict(
            opts=("-k", "--keep-unchanged"),
            action="store_true",
            dest="keep_unchanged",
            default=False,
            group="Frame Processing",
            help="When used with --frame-ranges outputs the unchanged frames that are not "
                 "processed instead of discarding them."))
        argument_list.append(dict(
            opts=("-s", "--swap-model"),
            action="store_true",
            dest="swap_model",
            default=False,
            group="settings",
            help="Swap the model. Instead converting from of A -> B, converts B -> A"))
        argument_list.append(dict(
            opts=("-sp", "--singleprocess"),
            action="store_true",
            default=False,
            group="settings",
            help="Disable multiprocessing. Slower but less resource intensive."))
        return argument_list


class TrainArgs(FaceSwapArgs):
    """ Creates the command line arguments for training. """

    @staticmethod
    def get_info():
        """ The information text for the Train command.

        Returns
        -------
        str
            The information text for the Train command.
        """
        return ("Train a model on extracted original (A) and swap (B) faces.\n"
                "Training models can take a long time. Anything from 24hrs to over a week\n"
                "Model plugins can be configured in the 'Settings' Menu")

    @staticmethod
    def get_argument_list():
        """ Returns the argument list for Train arguments.

        Returns
        -------
        list
            The list of command line options for training
        """
        argument_list = list()
        argument_list.append(dict(
            opts=("-A", "--input-A"),
            action=DirFullPaths,
            dest="input_a",
            required=True,
            group="faces",
            help="Input directory. A directory containing training images for face A. This is "
                 "the original face, i.e. the face that you want to remove and replace with "
                 "face B."))
        argument_list.append(dict(
            opts=("-ala", "--alignments-A"),
            action=FileFullPaths,
            filetypes='alignments',
            type=str,
            dest="alignments_path_a",
            default=None,
            group="faces",
            help="Path to alignments file for training set A. Only required if you are using a "
                 "masked model or warp-to-landmarks is enabled. Defaults to "
                 "<input-A>/alignments.json if not provided."))
        argument_list.append(dict(
            opts=("-B", "--input-B"),
            action=DirFullPaths,
            dest="input_b",
            required=True,
            group="faces",
            help="Input directory. A directory containing training images for face B. This is "
                 "the swap face, i.e. the face that you want to place onto the head of person "
                 "A."))
        argument_list.append(dict(
            opts=("-alb", "--alignments-B"),
            action=FileFullPaths,
            filetypes='alignments',
            type=str,
            dest="alignments_path_b",
            default=None,
            group="faces",
            help="Path to alignments file for training set B. Only required if you are using a "
                 "masked model or warp-to-landmarks is enabled. Defaults to "
                 "<input-B>/alignments.json if not provided."))
        argument_list.append(dict(
            opts=("-m", "--model-dir"),
            action=DirFullPaths,
            dest="model_dir",
            required=True,
            group="model",
            help="Model directory. This is where the training data will be stored. You should "
                 "always specify a new folder for new models. If starting a new model, select "
                 "either an empty folder, or a folder which does not exist (which will be "
                 "created). If continuing to train an existing model, specify the location of "
                 "the existing model."))
        argument_list.append(dict(
            opts=("-t", "--trainer"),
            action=Radio,
            type=str.lower,
            default=PluginLoader.get_default_model(),
            choices=PluginLoader.get_available_models(),
            group="model",
            help="R|Select which trainer to use. Trainers can be configured from the Settings "
                 "menu or the config folder."
                 "\nL|original: The original model created by /u/deepfakes."
                 "\nL|dfaker: 64px in/128px out model from dfaker. Enable 'warp-to-landmarks' "
                 "for full dfaker method."
                 "\nL|dfl-h128. 128px in/out model from deepfacelab"
                 "\nL|dfl-sae. Adaptable model from deepfacelab"
                 "\nL|dlight. A lightweight, high resolution DFaker variant."
                 "\nL|iae: A model that uses intermediate layers to try to get better details"
                 "\nL|lightweight: A lightweight model for low-end cards. Don't expect great "
                 "results. Can train as low as 1.6GB with batch size 8."
                 "\nL|realface: A high detail, dual density model based on DFaker, with "
                 "customizable in/out resolution. The autoencoders are unbalanced so B>A swaps "
                 "won't work so well. By andenixa et al. Very configurable."
                 "\nL|unbalanced: 128px in/out model from andenixa. The autoencoders are "
                 "unbalanced so B>A swaps won't work so well. Very configurable."
                 "\nL|villain: 128px in/out model from villainguy. Very resource hungry (11GB "
                 "for batchsize 16). Good for details, but more susceptible to color "
                 "differences."))
        argument_list.append(dict(
            opts=("-bs", "--batch-size"),
            action=Slider,
            min_max=(2, 256),
            rounding=2,
            type=int,
            dest="batch_size",
            default=64,
            group="training",
            help="Batch size. This is the number of images processed through the model for "
                 "each iteration. Larger batches require more GPU RAM."))
        argument_list.append(dict(
            opts=("-it", "--iterations"),
            action=Slider,
            min_max=(0, 5000000),
            rounding=20000,
            type=int,
            default=1000000,
            group="training",
            help="Length of training in iterations. This is only really used for automation. "
                 "There is no 'correct' number of iterations a model should be trained for. "
                 "You should stop training when you are happy with the previews. However, if "
                 "you want the model to stop automatically at a set number of iterations, you "
                 "can set that value here."))
        argument_list.append(dict(
            opts=("-g", "--gpus"),
            action=Slider,
            min_max=(1, 10),
            rounding=1,
            type=int,
            default=1,
            backend="nvidia",
            group="training",
            help="Number of GPUs to use for training"))
        argument_list.append(dict(
            opts=("-msg", "--memory-saving-gradients"),
            action="store_true",
            dest="memory_saving_gradients",
            default=False,
            backend="nvidia",
            group="VRAM Savings",
            help="Trades off VRAM usage against computation time. Can fit larger models into "
                 "memory at a cost of slower training speed. 50%%-150%% batch size increase "
                 "for 20%%-50%% longer training time. NB: Launch time will be significantly "
                 "delayed. Switching sides using ping-pong training will take longer."))
        argument_list.append(dict(
            opts=("-o", "--optimizer-savings"),
            action="store_true",
            dest="optimizer_savings",
            default=False,
            backend="nvidia",
            group="VRAM Savings",
            help="To save VRAM some optimizer gradient calculations can be performed on the "
                 "CPU rather than the GPU. This allows you to increase batchsize at a training "
                 "speed/system RAM cost."))
        argument_list.append(dict(
            opts=("-pp", "--ping-pong"),
            action="store_true",
            dest="pingpong",
            default=False,
            backend="nvidia",
            group="VRAM Savings",
            help="Enable ping pong training. Trains one side at a time, switching sides at "
                 "each save iteration. Training will take 2 to 4 times longer, with about a "
                 "30%%-50%% reduction in VRAM useage. NB: Preview won't show until both sides "
                 "have been trained once."))
        argument_list.append(dict(
            opts=("-s", "--save-interval"),
            action=Slider,
            min_max=(10, 1000),
            rounding=10,
            type=int,
            dest="save_interval",
            default=100,
            group="Saving",
            help="Sets the number of iterations between each model save."))
        argument_list.append(dict(
            opts=("-ss", "--snapshot-interval"),
            action=Slider,
            min_max=(0, 100000),
            rounding=5000,
            type=int,
            dest="snapshot_interval",
            default=25000,
            group="Saving",
            help="Sets the number of iterations before saving a backup snapshot of the model "
                 "in it's current state. Set to 0 for off."))
        argument_list.append(dict(
            opts=("-tia", "--timelapse-input-A"),
            action=DirFullPaths,
            dest="timelapse_input_a",
            default=None,
            group="timelapse",
            help="Optional for creating a timelapse. Timelapse will save an image of your "
                 "selected faces into the timelapse-output folder at every save iteration. "
                 "This should be the input folder of 'A' faces that you would like to use for "
                 "creating the timelapse. You must also supply a --timelapse-output and a "
                 "--timelapse-input-B parameter."))
        argument_list.append(dict(
            opts=("-tib", "--timelapse-input-B"),
            action=DirFullPaths,
            dest="timelapse_input_b",
            default=None,
            group="timelapse",
            help="Optional for creating a timelapse. Timelapse will save an image of your "
                 "selected faces into the timelapse-output folder at every save iteration. "
                 "This should be the input folder of 'B' faces that you would like to use for "
                 "creating the timelapse. You must also supply a --timelapse-output and a "
                 "--timelapse-input-A parameter."))
        argument_list.append(dict(
            opts=("-to", "--timelapse-output"),
            action=DirFullPaths,
            dest="timelapse_output",
            default=None,
            group="timelapse",
            help="Optional for creating a timelapse. Timelapse will save an image of your "
                 "selected faces into the timelapse-output folder at every save iteration. If "
                 "the input folders are supplied but no output folder, it will default to your "
                 "model folder /timelapse/"))
        argument_list.append(dict(
            opts=("-ps", "--preview-scale"),
            action=Slider,
            min_max=(25, 200),
            rounding=25,
            type=int,
            dest="preview_scale",
            default=50,
            group="preview",
            help="Percentage amount to scale the preview by."))
        argument_list.append(dict(
            opts=("-p", "--preview"),
            action="store_true",
            dest="preview",
            default=False,
            group="preview",
            help="Show training preview output. in a separate window."))
        argument_list.append(dict(
            opts=("-w", "--write-image"),
            action="store_true",
            dest="write_image",
            default=False,
            group="preview",
            help="Writes the training result to a file. The image will be stored in the root "
                 "of your FaceSwap folder."))
        argument_list.append(dict(
            opts=("-ag", "--allow-growth"),
            action="store_true",
            dest="allow_growth",
            default=False,
            backend="nvidia",
            group="model",
            help="Sets allow_growth option of Tensorflow to spare memory on some "
                 "configurations."))
        argument_list.append(dict(
            opts=("-nl", "--no-logs"),
            action="store_true",
            dest="no_logs",
            default=False,
            group="training",
            help="Disables TensorBoard logging. NB: Disabling logs means that you will not be "
                 "able to use the graph or analysis for this session in the GUI."))
        argument_list.append(dict(
            opts=("-wl", "--warp-to-landmarks"),
            action="store_true",
            dest="warp_to_landmarks",
            default=False,
            group="training",
            help="Warps training faces to closely matched Landmarks from the opposite face-set "
                 "rather than randomly warping the face. This is the 'dfaker' way of doing "
                 "warping. Alignments files for both sets of faces must be provided if using "
                 "this option."))
        argument_list.append(dict(
            opts=("-nf", "--no-flip"),
            action="store_true",
            dest="no_flip",
            default=False,
            group="training",
            help="To effectively learn, a random set of images are flipped horizontally. "
                 "Sometimes it is desirable for this not to occur. Generally this should be "
                 "left off except for during 'fit training'."))
        argument_list.append(dict(
            opts=("-nac", "--no-augment-color"),
            action="store_true",
            dest="no_augment_color",
            default=False,
            group="training",
            help="Color augmentation helps make the model less susceptible to color "
                 "differences between the A and B sets, at an increased training time cost. "
                 "Enable this option to disable color augmentation."))
        return argument_list


class GuiArgs(FaceSwapArgs):
    """ Creates the command line arguments for the GUI. """

    @staticmethod
    def get_argument_list():
        """ Returns the argument list for GUI arguments.

        Returns
        -------
        list
            The list of command line options for the GUI
        """
        argument_list = []
        argument_list.append(dict(
            opts=("-d", "--debug"),
            action="store_true",
            dest="debug",
            default=False,
            help="Output to Shell console instead of GUI console"))
        return argument_list
