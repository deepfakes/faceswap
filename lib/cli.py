#!/usr/bin/env python3
""" Command Line Arguments """

# pylint: disable=too-many-lines

import argparse
import logging
import os
import platform
import re
import sys
import textwrap

from importlib import import_module

from lib.logger import crash_log, log_setup
from lib.model.masks import get_available_masks, get_default_mask
from lib.utils import FaceswapError, get_backend, safe_shutdown
from plugins.plugin_loader import PluginLoader

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class ScriptExecutor():
    """ Loads the relevant script modules and executes the script.
        This class is initialised in each of the argparsers for the relevant
        command, then execute script is called within their set_default
        function. """

    def __init__(self, command, subparsers=None):
        self.command = command.lower()
        self.subparsers = subparsers

    def import_script(self):
        """ Only import a script's modules when running that script."""
        self.test_for_tf_version()
        self.test_for_gui()
        cmd = os.path.basename(sys.argv[0])
        src = "tools" if cmd == "tools.py" else "scripts"
        mod = ".".join((src, self.command.lower()))
        module = import_module(mod)
        script = getattr(module, self.command.title())
        return script

    @staticmethod
    def test_for_tf_version():
        """ Check that the minimum required Tensorflow version is installed """
        min_ver = 1.12
        max_ver = 1.14
        try:
            # Ensure tensorflow doesn't pin all threads to one core when using tf-mkl
            os.environ["KMP_AFFINITY"] = "disabled"
            import tensorflow as tf
        except ImportError as err:
            raise FaceswapError("There was an error importing Tensorflow. This is most likely "
                                "because you do not have TensorFlow installed, or you are trying "
                                "to run tensorflow-gpu on a system without an Nvidia graphics "
                                "card. Original import error: {}".format(str(err)))
        tf_ver = float(".".join(tf.__version__.split(".")[:2]))
        if tf_ver < min_ver:
            raise FaceswapError("The minimum supported Tensorflow is version {} but you have "
                                "version {} installed. Please upgrade Tensorflow.".format(
                                    min_ver, tf_ver))
        if tf_ver > max_ver:
            raise FaceswapError("The maximumum supported Tensorflow is version {} but you have "
                                "version {} installed. Please downgrade Tensorflow.".format(
                                    max_ver, tf_ver))
        logger.debug("Installed Tensorflow Version: %s", tf_ver)

    def test_for_gui(self):
        """ If running the gui, check the prerequisites """
        if self.command != "gui":
            return
        self.test_tkinter()
        self.check_display()

    @staticmethod
    def test_tkinter():
        """ If the user is running the GUI, test whether the
            tkinter app is available on their machine. If not
            exit gracefully.

            This avoids having to import every tk function
            within the GUI in a wrapper and potentially spamming
            traceback errors to console """

        try:
            # pylint: disable=unused-variable
            import tkinter  # noqa pylint: disable=unused-import
        except ImportError:
            logger.error(
                "It looks like TkInter isn't installed for your OS, so "
                "the GUI has been disabled. To enable the GUI please "
                "install the TkInter application. You can try:")
            logger.info("Anaconda: conda install tk")
            logger.info("Windows/macOS: Install ActiveTcl Community Edition from "
                        "http://www.activestate.com")
            logger.info("Ubuntu/Mint/Debian: sudo apt install python3-tk")
            logger.info("Arch: sudo pacman -S tk")
            logger.info("CentOS/Redhat: sudo yum install tkinter")
            logger.info("Fedora: sudo dnf install python3-tkinter")
            raise FaceswapError("TkInter not found")

    @staticmethod
    def check_display():
        """ Check whether there is a display to output the GUI. If running on
            Windows then assume not running in headless mode """
        if not os.environ.get("DISPLAY", None) and os.name != "nt":
            if platform.system() == "Darwin":
                logger.info("macOS users need to install XQuartz. "
                            "See https://support.apple.com/en-gb/HT201341")
            raise FaceswapError("No display detected. GUI mode has been disabled.")

    def execute_script(self, arguments):
        """ Run the script for called command """
        is_gui = hasattr(arguments, "redirect_gui") and arguments.redirect_gui
        log_setup(arguments.loglevel, arguments.logfile, self.command, is_gui)
        logger.debug("Executing: %s. PID: %s", self.command, os.getpid())
        success = False
        if get_backend() == "amd":
            plaidml_found = self.setup_amd(arguments.loglevel)
            if not plaidml_found:
                safe_shutdown(got_error=True)
                return
        try:
            script = self.import_script()
            process = script(arguments)
            process.process()
            success = True
        except FaceswapError as err:
            for line in str(err).splitlines():
                logger.error(line)
            crash_file = crash_log()
            logger.info("To get more information on this error see the crash report written to "
                        "'%s'", crash_file)
        except KeyboardInterrupt:  # pylint: disable=try-except-raise
            raise
        except SystemExit:
            pass
        except Exception:  # pylint: disable=broad-except
            crash_file = crash_log()
            logger.exception("Got Exception on main handler:")
            logger.critical("An unexpected crash has occurred. Crash report written to '%s'. "
                            "You MUST provide this file if seeking assistance. Please verify you "
                            "are running the latest version of faceswap before reporting",
                            crash_file)

        finally:
            safe_shutdown(got_error=not success)

    @staticmethod
    def setup_amd(loglevel):
        """ Test for plaidml and setup for AMD """
        logger.debug("Setting up for AMD")
        try:
            import plaidml  # noqa pylint:disable=unused-import
        except ImportError:
            logger.error("PlaidML not found. Run `pip install plaidml-keras` for AMD support")
            return False
        from lib.plaidml_tools import setup_plaidml
        setup_plaidml(loglevel)
        logger.debug("setup up for PlaidML")
        return True


class Radio(argparse.Action):  # pylint: disable=too-few-public-methods
    """ Adds support for the GUI Radio buttons

        Just a wrapper class to tell the gui to use radio buttons instead of combo boxes
        """
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        if nargs is not None:
            raise ValueError("nargs not allowed")
        super().__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)


class Slider(argparse.Action):  # pylint: disable=too-few-public-methods
    """ Adds support for the GUI slider

        An additional option 'min_max' must be provided containing tuple of min and max accepted
        values.

        'rounding' sets the decimal places for floats or the step interval for ints.
        """
    def __init__(self, option_strings, dest, nargs=None, min_max=None, rounding=None, **kwargs):
        if nargs is not None:
            raise ValueError("nargs not allowed")
        super().__init__(option_strings, dest, **kwargs)
        self.min_max = min_max
        self.rounding = rounding

    def _get_kwargs(self):
        names = ["option_strings",
                 "dest",
                 "nargs",
                 "const",
                 "default",
                 "type",
                 "choices",
                 "help",
                 "metavar",
                 "min_max",  # Tuple containing min and max values of scale
                 "rounding"]  # Decimal places to round floats to or step interval for ints
        return [(name, getattr(self, name)) for name in names]

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)


class FullPaths(argparse.Action):  # pylint: disable=too-few-public-methods
    """ Expand user- and relative-paths """
    def __call__(self, parser, namespace, values, option_string=None):
        if isinstance(values, (list, tuple)):
            vals = [os.path.abspath(os.path.expanduser(val)) for val in values]
        else:
            vals = os.path.abspath(os.path.expanduser(values))
        setattr(namespace, self.dest, vals)


class DirFullPaths(FullPaths):
    """ Class that gui uses to determine if you need to open a directory """
    # pylint: disable=too-few-public-methods,unnecessary-pass
    pass


class FileFullPaths(FullPaths):
    """
    Class that gui uses to determine if you need to open a file.

    see lib/gui/utils.py FileHandler for current GUI filetypes
    """
    # pylint: disable=too-few-public-methods
    def __init__(self, option_strings, dest, nargs=None, filetypes=None, **kwargs):
        super().__init__(option_strings, dest, nargs, **kwargs)
        self.filetypes = filetypes

    def _get_kwargs(self):
        names = ["option_strings",
                 "dest",
                 "nargs",
                 "const",
                 "default",
                 "type",
                 "choices",
                 "help",
                 "metavar",
                 "filetypes"]
        return [(name, getattr(self, name)) for name in names]


class FilesFullPaths(FileFullPaths):  # pylint: disable=too-few-public-methods
    """ Class that the gui uses to determine that the input can take multiple files as an input.
        Inherits functionality from FileFullPaths
        Has the effect of giving the user 2 Open Dialogue buttons in the gui """
    pass


class DirOrFileFullPaths(FileFullPaths):  # pylint: disable=too-few-public-methods
    """ Class that the gui uses to determine that the input can take a folder or a filename.
        Inherits functionality from FileFullPaths
        Has the effect of giving the user 2 Open Dialogue buttons in the gui """
    pass


class SaveFileFullPaths(FileFullPaths):
    """
    Class that gui uses to determine if you need to save a file.

    see lib/gui/utils.py FileHandler for current GUI filetypes
    """
    # pylint: disable=too-few-public-methods,unnecessary-pass
    pass


class ContextFullPaths(FileFullPaths):
    """
    Class that gui uses to determine if you need to open a file or a
    directory based on which action you are choosing

    To use ContextFullPaths the action_option item should indicate which
    cli option dictates the context of the filesystem dialogue

    Bespoke actions are then set in lib/gui/utils.py FileHandler
    """
    # pylint: disable=too-few-public-methods, too-many-arguments
    def __init__(self, option_strings, dest, nargs=None, filetypes=None,
                 action_option=None, **kwargs):
        if nargs is not None:
            raise ValueError("nargs not allowed")
        super(ContextFullPaths, self).__init__(option_strings, dest,
                                               filetypes=None, **kwargs)
        self.action_option = action_option
        self.filetypes = filetypes

    def _get_kwargs(self):
        names = ["option_strings",
                 "dest",
                 "nargs",
                 "const",
                 "default",
                 "type",
                 "choices",
                 "help",
                 "metavar",
                 "filetypes",
                 "action_option"]
        return [(name, getattr(self, name)) for name in names]


class FullHelpArgumentParser(argparse.ArgumentParser):
    """ Identical to the built-in argument parser, but on error it
        prints full help message instead of just usage information """
    def error(self, message):
        self.print_help(sys.stderr)
        args = {"prog": self.prog, "message": message}
        self.exit(2, "%(prog)s: error: %(message)s\n" % args)


class SmartFormatter(argparse.HelpFormatter):
    """ Smart formatter for allowing raw formatting in help
        text and lists in the helptext

        To use: prefix the help item with "R|" to overide
        default formatting. List items can be marked with "L|"
        at the start of a newline

        adapted from: https://stackoverflow.com/questions/3853722 """

    def __init__(self,
                 prog,
                 indent_increment=2,
                 max_help_position=24,
                 width=None):

        super().__init__(prog, indent_increment, max_help_position, width)
        self._whitespace_matcher_limited = re.compile(r'[ \r\f\v]+', re.ASCII)

    def _split_lines(self, text, width):
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
    """ Faceswap argument parser functions that are universal
        to all commands. Should be the parent function of all
        subsequent argparsers """
    def __init__(self, subparser, command,
                 description="default", subparsers=None):

        self.global_arguments = self.get_global_arguments()
        self.info = self.get_info()
        self.argument_list = self.get_argument_list()
        self.optional_arguments = self.get_optional_arguments()
        self.process_suppressions()
        if not subparser:
            return

        self.parser = self.create_parser(subparser, command, description)

        self.add_arguments()

        script = ScriptExecutor(command, subparsers)
        self.parser.set_defaults(func=script.execute_script)

    @staticmethod
    def get_info():
        """ Return command information for display in the GUI.
            Override for command specific info """
        return None

    @staticmethod
    def get_argument_list():
        """ Put the arguments in a list so that they are accessible from both
            argparse and gui override for command specific arguments """
        argument_list = []
        return argument_list

    @staticmethod
    def get_optional_arguments():
        """ Put the arguments in a list so that they are accessible from both
            argparse and gui. This is used for when there are sub-children
            (e.g. convert and extract) Override this for custom arguments """
        argument_list = []
        return argument_list

    @staticmethod
    def get_global_arguments():
        """ Arguments that are used in ALL parts of Faceswap
            DO NOT override this """
        global_args = list()
        global_args.append({
            "opts": ("-C", "--configfile"),
            "action": FileFullPaths,
            "filetypes": "ini",
            "type": str,
            "group": "Global Options",
            "help": "Optionally overide the saved config with the path to a custom config file."})
        global_args.append({
            "opts": ("-L", "--loglevel"),
            "type": str.upper,
            "dest": "loglevel",
            "default": "INFO",
            "choices": ("INFO", "VERBOSE", "DEBUG", "TRACE"),
            "group": "Global Options",
            "help": "Log level. Stick with INFO or VERBOSE unless you need to file an error "
                    "report. Be careful with TRACE as it will generate a lot of data"})
        global_args.append({
            "opts": ("-LF", "--logfile"),
            "action": SaveFileFullPaths,
            "filetypes": 'log',
            "type": str,
            "dest": "logfile",
            "group": "Global Options",
            "help": "Path to store the logfile. Leave blank to store in the faceswap folder",
            "default": None})
        # This is a hidden argument to indicate that the GUI is being used,
        # so the preview window should be redirected Accordingly
        global_args.append({
            "opts": ("-gui", "--gui"),
            "action": "store_true",
            "dest": "redirect_gui",
            "default": False,
            "help": argparse.SUPPRESS})
        return global_args

    @staticmethod
    def create_parser(subparser, command, description):
        """ Create the parser for the selected command """
        parser = subparser.add_parser(
            command,
            help=description,
            description=description,
            epilog="Questions and feedback: https://faceswap.dev/forum",
            formatter_class=SmartFormatter)
        return parser

    def add_arguments(self):
        """ Parse the arguments passed in from argparse """
        options = self.global_arguments + self.argument_list + self.optional_arguments
        for option in options:
            args = option["opts"]
            kwargs = {key: option[key]
                      for key in option.keys() if key not in ("opts", "group")}
            self.parser.add_argument(*args, **kwargs)

    def process_suppressions(self):
        """ Suppress option if it is not available for running backend """
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
    """ This class is used as a parent class to capture arguments that
        will be used in both the extract and convert process.

        Arguments that can be used in both of these processes should be
        placed here, but no further processing should be done. This class
        just captures arguments """

    @staticmethod
    def get_argument_list():
        """ Put the arguments in a list so that they are accessible from both
        argparse and gui """
        argument_list = list()
        argument_list.append({
            "opts": ("-i", "--input-dir"),
            "action": DirOrFileFullPaths,
            "filetypes": "video",
            "dest": "input_dir",
            "required": True,
            "group": "Data",
            "help": "Input directory or video. Either a directory containing the image files you "
                    "wish to process or path to a video file. NB: This should be the source video/"
                    "frames NOT the source faces."})
        argument_list.append({
            "opts": ("-o", "--output-dir"),
            "action": DirFullPaths,
            "dest": "output_dir",
            "required": True,
            "group": "Data",
            "help": "Output directory. This is where the converted files will be saved."})
        argument_list.append({
            "opts": ("-al", "--alignments"),
            "action": FileFullPaths,
            "filetypes": "alignments",
            "type": str,
            "dest": "alignments_path",
            "group": "Data",
            "help": "Optional path to an alignments file. Leave blank if the alignments file is "
                    "at the default location."})
        return argument_list


class ExtractArgs(ExtractConvertArgs):
    """ Class to parse the command line arguments for extraction.
        Inherits base options from ExtractConvertArgs where arguments
        that are used for both extract and convert should be placed """

    @staticmethod
    def get_info():
        """ Return command information """
        return ("Extract faces from image or video sources.\n"
                "Extraction plugins can be configured in the 'Settings' Menu")

    @staticmethod
    def get_optional_arguments():
        """ Put the arguments in a list so that they are accessible from both
        argparse and gui """
        if get_backend() == "cpu":
            default_detector = default_aligner = "cv2-dnn"
        else:
            default_detector = "s3fd"
            default_aligner = "fan"

        argument_list = []
        argument_list.append({
            "opts": ("-D", "--detector"),
            "action": Radio,
            "type": str.lower,
            "choices":  PluginLoader.get_available_extractors("detect"),
            "default": default_detector,
            "group": "Plugins",
            "help": "R|Detector to use. Some of these have configurable settings in "
                    "'/config/extract.ini' or 'Settings > Configure Extract 'Plugins':"
                    "\nL|cv2-dnn: A CPU only extractor which is the least reliable and least "
                    "resource intensive. Use this if not using a GPU and time is important."
                    "\nL|mtcnn: Good detector. Fast on CPU, faster on GPU. Uses fewer resources "
                    "than other GPU detectors but can often return more false positives."
                    "\nL|s3fd: Best detector. Fast on GPU, slow on CPU. Can detect more faces and "
                    "fewer false positives than other GPU detectors, but is a lot more resource "
                    "intensive."})
        argument_list.append({
            "opts": ("-A", "--aligner"),
            "action": Radio,
            "type": str.lower,
            "choices": PluginLoader.get_available_extractors("align"),
            "default": default_aligner,
            "group": "Plugins",
            "help": "R|Aligner to use."
                    "\nL|cv2-dnn: A CPU only landmark detector. Faster, less resource intensive, "
                    "but less accurate. Only use this if not using a GPU and time is important."
                    "\nL|fan: Best aligner. Fast on GPU, slow on CPU."})
        argument_list.append({
            "opts": ("-M", "--masker"),
            "action": Radio,
            "type": str.lower,
            "choices": PluginLoader.get_available_extractors("mask"),
            "default": "extended",
            "group": "Plugins",
            "help": "R|Masker to use. NB: Masker is not currently used by the rest of the process "
                    "but this will store a mask in the alignments file for use when it has been "
                    "implemented."
                    "\nL|none: An array of all ones is created to provide a 4th channel that will "
                    "not mask any portion of the image."
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
                    "performance."})
        argument_list.append({
            "opts": ("-nm", "--normalization"),
            "action": Radio,
            "type": str.lower,
            "dest": "normalization",
            "choices": ["none", "clahe", "hist", "mean"],
            "default": "none",
            "group": "plugins",
            "help": "R|Performing normalization can help the aligner better align faces with "
                    "difficult lighting conditions at an extraction speed cost. Different methods "
                    "will yield different results on different sets. NB: This does not impact the "
                    "output face, just the input to the aligner."
                    "\nL|none: Don't perform normalization on the face."
                    "\nL|clahe: Perform Contrast Limited Adaptive Histogram Equalization on the "
                    "face."
                    "\nL|hist: Equalize the histograms on the RGB channels."
                    "\nL|mean: Normalize the face colors to the mean."})
        argument_list.append({
            "opts": ("-r", "--rotate-images"),
            "type": str,
            "dest": "rotate_images",
            "default": None,
            "group": "plugins",
            "help": "If a face isn't found, rotate the images to try to find a face. Can find "
                    "more faces at the cost of extraction speed. Pass in a single number to use "
                    "increments of that size up to 360, or pass in a list of numbers to enumerate "
                    "exactly what angles to check."})
        argument_list.append({
            "opts": ("-min", "--min-size"),
            "type": int,
            "action": Slider,
            "dest": "min_size",
            "min_max": (0, 1080),
            "default": 0,
            "rounding": 20,
            "group": "Face Processing",
            "help": "Filters out faces detected below this size. Length, in pixels across the "
                    "diagonal of the bounding box. Set to 0 for off"})
        argument_list.append({
            "opts": ("-n", "--nfilter"),
            "action": FilesFullPaths,
            "filetypes": "image",
            "dest": "nfilter",
            "nargs": "+",
            "default": None,
            "group": "Face Processing",
            "help": "Optionally filter out people who you do not wish to process by passing in an "
                    "image of that person. Should be a front portrait with a single person in the "
                    "image. Multiple images can be added space separated. NB: Using face filter "
                    "will significantly decrease extraction speed and its accuracy cannot be "
                    "guaranteed."})
        argument_list.append({
            "opts": ("-f", "--filter"),
            "action": FilesFullPaths,
            "filetypes": "image",
            "dest": "filter",
            "nargs": "+",
            "default": None,
            "group": "Face Processing",
            "help": "Optionally select people you wish to process by passing in an image of that "
                    "person. Should be a front portrait with a single person in the image. "
                    "Multiple images can be added space separated. NB: Using face filter will "
                    "significantly decrease extraction speed and its accuracy cannot be "
                    "guaranteed."})
        argument_list.append({
            "opts": ("-l", "--ref_threshold"),
            "action": Slider,
            "min_max": (0.01, 0.99),
            "rounding": 2,
            "type": float,
            "dest": "ref_threshold",
            "default": 0.4,
            "group": "Face Processing",
            "help": "For use with the optional nfilter/filter files. Threshold for positive face "
                    "recognition. Lower values are stricter. NB: Using face filter will "
                    "significantly decrease extraction speed and its accuracy cannot be "
                    "guaranteed."})
        argument_list.append({
            "opts": ("-bt", "--blur-threshold"),
            "type": float,
            "action": Slider,
            "min_max": (0.0, 100.0),
            "rounding": 1,
            "dest": "blur_thresh",
            "default": 0.0,
            "group": "Face Processing",
            "help": "Automatically discard images blurrier than the specified threshold. "
                    "Discarded images are moved into a \"blurry\" sub-folder. Lower values allow "
                    "more blur. Set to 0.0 to turn off."})
        argument_list.append({
            "opts": ("-een", "--extract-every-n"),
            "type": int,
            "action": Slider,
            "dest": "extract_every_n",
            "min_max": (1, 100),
            "default": 1,
            "rounding": 1,
            "group": "output",
            "help": "Extract every 'nth' frame. This option will skip frames when extracting "
                    "faces. For example a value of 1 will extract faces from every frame, a value "
                    "of 10 will extract faces from every 10th frame."})
        argument_list.append({
            "opts": ("-sz", "--size"),
            "type": int,
            "action": Slider,
            "min_max": (128, 512),
            "default": 256,
            "rounding": 64,
            "group": "output",
            "help": "The output size of extracted faces. Make sure that the model you intend to "
                    "train supports your required size. This will only need to be changed for "
                    "hi-res models."})
        argument_list.append({
            "opts": ("-si", "--save-interval"),
            "dest": "save_interval",
            "type": int,
            "action": Slider,
            "min_max": (0, 1000),
            "rounding": 10,
            "default": 0,
            "group": "output",
            "help": "Automatically save the alignments file after a set amount of frames. By "
                    "default the alignments file is only saved at the end of the extraction "
                    "process. NB: If extracting in 2 passes then the alignments file will only "
                    "start to be saved out during the second pass. WARNING: Don't interrupt the "
                    "script when writing the file because it might get corrupted. Set to 0 to "
                    "turn off"})
        argument_list.append({
            "opts": ("-dl", "--debug-landmarks"),
            "action": "store_true",
            "dest": "debug_landmarks",
            "group": "output",
            "default": False,
            "help": "Draw landmarks on the ouput faces for debugging purposes."})
        argument_list.append({
            "opts": ("-sp", "--singleprocess"),
            "action": "store_true",
            "default": False,
            "backend": "nvidia",
            "group": "settings",
            "help": "Don't run extraction in parallel. Will run each part of the extraction "
                    "process separately (one after the other) rather than all at the smae time. "
                    "Useful if VRAM is at a premium."})
        argument_list.append({
            "opts": ("-s", "--skip-existing"),
            "action": "store_true",
            "dest": "skip_existing",
            "group": "settings",
            "default": False,
            "help": "Skips frames that have already been extracted and exist in the alignments "
                    "file"})
        argument_list.append({
            "opts": ("-sf", "--skip-existing-faces"),
            "action": "store_true",
            "dest": "skip_faces",
            "group": "settings",
            "default": False,
            "help": "Skip frames that already have detected faces in the alignments file"})
        return argument_list


class ConvertArgs(ExtractConvertArgs):
    """ Class to parse the command line arguments for conversion.
        Inherits base options from ExtractConvertArgs where arguments
        that are used for both extract and convert should be placed """

    @staticmethod
    def get_info():
        """ Return command information """
        return ("Swap the original faces in a source video/images to your final faces.\n"
                "Conversion plugins can be configured in the 'Settings' Menu")

    @staticmethod
    def get_optional_arguments():
        """ Put the arguments in a list so that they are accessible from both
        argparse and gui """
        argument_list = []
        argument_list.append({
            "opts": ("-ref", "--reference-video"),
            "action": FileFullPaths,
            "dest": "reference_video",
            "filetypes": "video",
            "type": str,
            "group": "data",
            "help": "Only required if converting from images to video. Provide The original video "
                    "that the source frames were extracted from (for extracting the fps and "
                    "audio)."})
        argument_list.append({
            "opts": ("-m", "--model-dir"),
            "action": DirFullPaths,
            "dest": "model_dir",
            "required": True,
            "group": "data",
            "help": "Model directory. The directory containing the trained model you wish to use "
                    "for conversion."})
        argument_list.append({
            "opts": ("-c", "--color-adjustment"),
            "action": Radio,
            "type": str.lower,
            "dest": "color_adjustment",
            "choices": PluginLoader.get_available_convert_plugins("color", True),
            "default": "avg-color",
            "group": "plugins",
            "help": "R|Performs color adjustment to the swapped face. Some of these options have "
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
                    "\nL|none: Don't perform color adjustment."})
        argument_list.append({
            "opts": ("-M", "--mask-type"),
            "action": Radio,
            "type": str.lower,
            "dest": "mask_type",
            "choices": get_available_masks() + ["predicted"],
            "group": "plugins",
            "default": "predicted",
            "help": "R|Mask to use to replace faces. Blending of the masks can be adjusted in "
                    "'/config/convert.ini' or 'Settings > Configure Convert Plugins':"
                    "\nL|components: An improved face hull mask using a facehull of 8 facial "
                    "parts."
                    "\nL|dfl_full: An improved face hull mask using a facehull of 3 facial parts."
                    "\nL|extended: Based on components mask. Extends the eyebrow points to "
                    "further up the forehead. May perform badly on difficult angles."
                    "\nL|facehull: Face cutout based on landmarks."
                    "\nL|predicted: The predicted mask generated from the model. If the model was "
                    "not trained with a mask then this will fallback to "
                    "'{}'".format(get_default_mask()) +
                    "\nL|none: Don't use a mask."})
        argument_list.append({
            "opts": ("-sc", "--scaling"),
            "action": Radio,
            "type": str.lower,
            "choices": PluginLoader.get_available_convert_plugins("scaling", True),
            "group": "plugins",
            "default": "none",
            "help": "R|Performs a scaling process to attempt to get better definition on the "
                    "final swap. Some of these options have configurable settings in "
                    "'/config/convert.ini' or 'Settings > Configure Convert Plugins':"
                    "\nL|sharpen: Perform sharpening on the final face."
                    "\nL|none: Don't perform any scaling operations."})
        argument_list.append({
            "opts": ("-w", "--writer"),
            "action": Radio,
            "type": str,
            "choices": PluginLoader.get_available_convert_plugins("writer", False),
            "group": "plugins",
            "default": "opencv",
            "help": "R|The plugin to use to output the converted images. The writers are "
                    "configurable in '/config/convert.ini' or 'Settings > Configure Convert "
                    "Plugins:'"
                    "\nL|ffmpeg: [video] Writes out the convert straight to video. When the input "
                    "is a series of images then the '-ref' (--reference-video) parameter must be "
                    "set."
                    "\nL|gif: [animated image] Create an animated gif."
                    "\nL|opencv: [images] The fastest image writer, but less options and formats "
                    "than other plugins."
                    "\nL|pillow: [images] Slower than opencv, but has more options and supports "
                    "more formats."})
        argument_list.append({
            "opts": ("-osc", "--output-scale"),
            "dest": "output_scale",
            "action": Slider,
            "type": int,
            "default": 100,
            "min_max": (25, 400),
            "rounding": 1,
            "group": "Frame Processing",
            "help": "Scale the final output frames by this amount. 100%% will output the frames "
                    "at source dimensions. 50%% at half size 200%% at double size"})
        argument_list.append({
            "opts": ("-fr", "--frame-ranges"),
            "nargs": "+",
            "type": str,
            "group": "Frame Processing",
            "help": "Frame ranges to apply transfer to e.g. For frames 10 to 50 and 90 to 100 use "
                    "--frame-ranges 10-50 90-100. Frames falling outside of the selected range "
                    "will be discarded unless '-k' (--keep-unchanged) is selected. NB: If you are "
                    "converting from images, then the filenames must end with the frame-number!"})
        argument_list.append({
            "opts": ("-a", "--input-aligned-dir"),
            "action": DirFullPaths,
            "dest": "input_aligned_dir",
            "group": "Face Processing",
            "default": None,
            "help": "If you have not cleansed your alignments file, then you can filter out faces "
                    "by defining a folder here that contains the faces extracted from your input "
                    "files/video. If this folder is defined, then only faces that exist within "
                    "your alignments file and also exist within the specified folder will be "
                    "converted. Leaving this blank will convert all faces that exist within the "
                    "alignments file."})
        argument_list.append({
            "opts": ("-n", "--nfilter"),
            "action": FilesFullPaths,
            "filetypes": "image",
            "dest": "nfilter",
            "nargs": "+",
            "default": None,
            "group": "Face Processing",
            "help": "Optionally filter out people who you do not wish to process by passing in an "
                    "image of that person. Should be a front portrait with a single person in the "
                    "image. Multiple images can be added space separated. NB: Using face filter "
                    "will significantly decrease extraction speed and its accuracy cannot be "
                    "guaranteed."})
        argument_list.append({
            "opts": ("-f", "--filter"),
            "action": FilesFullPaths,
            "filetypes": "image",
            "dest": "filter",
            "nargs": "+",
            "default": None,
            "group": "Face Processing",
            "help": "Optionally select people you wish to process by passing in an image of that "
                    "person. Should be a front portrait with a single person in the image. "
                    "Multiple images can be added space separated. NB: Using face filter will "
                    "significantly decrease extraction speed and its accuracy cannot be "
                    "guaranteed."})
        argument_list.append({
            "opts": ("-l", "--ref_threshold"),
            "action": Slider,
            "min_max": (0.01, 0.99),
            "rounding": 2,
            "type": float,
            "dest": "ref_threshold",
            "default": 0.4,
            "group": "Face Processing",
            "help": "For use with the optional nfilter/filter files. Threshold for positive face "
                    "recognition. Lower values are stricter. NB: Using face filter will "
                    "significantly decrease extraction speed and its accuracy cannot be "
                    "guaranteed."})
        argument_list.append({
            "opts": ("-j", "--jobs"),
            "dest": "jobs",
            "action": Slider,
            "group": "settings",
            "type": int,
            "default": 0,
            "min_max": (0, 40),
            "rounding": 1,
            "help": "The maximum number of parallel processes for performing conversion. "
                    "Converting images is system RAM heavy so it is possible to run out of memory "
                    "if you have a lot of processes and not enough RAM to accommodate them all. "
                    "Setting this to 0 will use the maximum available. No matter what you set "
                    "this to, it will never attempt to use more processes than are available on "
                    "your system. If singleprocess is enabled this setting will be ignored."})
        argument_list.append({
            "opts": ("-g", "--gpus"),
            "type": int,
            "backend": "nvidia",
            "action": Slider,
            "min_max": (1, 10),
            "rounding": 1,
            "group": "settings",
            "default": 1,
            "help": "Number of GPUs to use for conversion"})
        argument_list.append({
            "opts": ("-t", "--trainer"),
            "type": str.lower,
            "choices": PluginLoader.get_available_models(),
            "group": "settings",
            "help": "[LEGACY] This only needs to be selected if a legacy model is being loaded or "
                    "if there are multiple models in the model folder"})
        argument_list.append({
            "opts": ("-k", "--keep-unchanged"),
            "action": "store_true",
            "dest": "keep_unchanged",
            "group": "Frame Processing",
            "default": False,
            "help": "When used with --frame-ranges outputs the unchanged frames that are not "
                    "processed instead of discarding them."})
        argument_list.append({
            "opts": ("-s", "--swap-model"),
            "action": "store_true",
            "dest": "swap_model",
            "group": "settings",
            "default": False,
            "help": "Swap the model. Instead converting from of A -> B, converts B -> A"})
        argument_list.append({
            "opts": ("-sp", "--singleprocess"),
            "action": "store_true",
            "group": "settings",
            "default": False,
            "help": "Disable multiprocessing. Slower but less resource intensive."})
        return argument_list


class TrainArgs(FaceSwapArgs):
    """ Class to parse the command line arguments for training """

    @staticmethod
    def get_info():
        """ Return command information """
        return ("Train a model on extracted original (A) and swap (B) faces.\n"
                "Training models can take a long time. Anything from 24hrs to over a week\n"
                "Model plugins can be configured in the 'Settings' Menu")

    @staticmethod
    def get_argument_list():
        """ Put the arguments in a list so that they are accessible from both
        argparse and gui """
        argument_list = list()
        argument_list.append({"opts": ("-A", "--input-A"),
                              "action": DirFullPaths,
                              "dest": "input_a",
                              "required": True,
                              "group": "faces",
                              "help": "Input directory. A directory containing training images "
                                      "for face A. This is the original face, i.e. the face that "
                                      "you want to remove and replace with face B."})
        argument_list.append({"opts": ("-ala", "--alignments-A"),
                              "action": FileFullPaths,
                              "filetypes": 'alignments',
                              "type": str,
                              "dest": "alignments_path_a",
                              "default": None,
                              "group": "faces",
                              "help": "Path to alignments file for training set A. Only required "
                                      "if you are using a masked model or warp-to-landmarks is "
                                      "enabled. Defaults to <input-A>/alignments.json if not "
                                      "provided."})
        argument_list.append({"opts": ("-B", "--input-B"),
                              "action": DirFullPaths,
                              "dest": "input_b",
                              "required": True,
                              "group": "faces",
                              "help": "Input directory. A directory containing training images "
                                      "for face B. This is the swap face, i.e. the face that "
                                      "you want to place onto the head of person A."})
        argument_list.append({"opts": ("-alb", "--alignments-B"),
                              "action": FileFullPaths,
                              "filetypes": 'alignments',
                              "type": str,
                              "dest": "alignments_path_b",
                              "default": None,
                              "group": "faces",
                              "help": "Path to alignments file for training set B. Only required "
                                      "if you are using a masked model or warp-to-landmarks is "
                                      "enabled. Defaults to <input-B>/alignments.json if not "
                                      "provided."})
        argument_list.append({"opts": ("-m", "--model-dir"),
                              "action": DirFullPaths,
                              "dest": "model_dir",
                              "required": True,
                              "group": "model",
                              "help": "Model directory. This is where the training data will be "
                                      "stored. You should always specify a new folder for new "
                                      "models. If starting a new model, select either an empty "
                                      "folder, or a folder which does not exist (which will be "
                                      "created). If continuing to train an existing model, "
                                      "specify the location of the existing model."})
        argument_list.append({"opts": ("-t", "--trainer"),
                              "action": Radio,
                              "type": str.lower,
                              "choices": PluginLoader.get_available_models(),
                              "default": PluginLoader.get_default_model(),
                              "group": "model",
                              "help": "R|Select which trainer to use. Trainers can be"
                                      "configured from the Settings menu or the config folder."
                                      "\nL|original: The original model created by /u/deepfakes."
                                      "\nL|dfaker: 64px in/128px out model from dfaker. "
                                      "Enable 'warp-to-landmarks' for full dfaker method."
                                      "\nL|dfl-h128. 128px in/out model from deepfacelab"
                                      "\nL|dfl-sae. Adaptable model from deepfacelab"
                                      "\nL|iae: A model that uses intermediate layers to try to "
                                      "get better details"
                                      "\nL|lightweight: A lightweight model for low-end cards. "
                                      "Don't expect great results. Can train as low as 1.6GB "
                                      "with batch size 8."
                                      "\nL|realface: A high detail, dual density model based on "
                                      "DFaker, with customizable in/out resolution. The "
                                      "autoencoders are unbalanced so B>A swaps won't work "
                                      "so well. By andenixa et al. Very configurable."
                                      "\nL|unbalanced: 128px in/out model from andenixa. The "
                                      "autoencoders are unbalanced so B>A swaps won't work so "
                                      "well. Very configurable."
                                      "\nL|villain: 128px in/out model from villainguy. Very "
                                      "resource hungry (11GB for batchsize 16). Good for "
                                      "details, but more susceptible to color differences."})
        argument_list.append({"opts": ("-bs", "--batch-size"),
                              "type": int,
                              "action": Slider,
                              "min_max": (2, 256),
                              "rounding": 2,
                              "dest": "batch_size",
                              "default": 64,
                              "group": "training",
                              "help": "Batch size. This is the number of images processed through "
                                      "the model for each iteration. Larger batches require more "
                                      "GPU RAM."})
        argument_list.append({"opts": ("-it", "--iterations"),
                              "type": int,
                              "action": Slider,
                              "min_max": (0, 5000000),
                              "rounding": 20000,
                              "default": 1000000,
                              "group": "training",
                              "help": "Length of training in iterations. This is only really used "
                                      "for automation. There is no 'correct' number of iterations "
                                      "a model should be trained for. You should stop training "
                                      "when you are happy with the previews. However, if you want "
                                      "the model to stop automatically at a set number of "
                                      "iterations, you can set that value here."})
        argument_list.append({"opts": ("-g", "--gpus"),
                              "type": int,
                              "backend": "nvidia",
                              "action": Slider,
                              "min_max": (1, 10),
                              "rounding": 1,
                              "group": "training",
                              "default": 1,
                              "help": "Number of GPUs to use for training"})
        argument_list.append({"opts": ("-msg", "--memory-saving-gradients"),
                              "action": "store_true",
                              "dest": "memory_saving_gradients",
                              "group": "VRAM Savings",
                              "default": False,
                              "backend": "nvidia",
                              "help": "Trades off VRAM usage against computation time. Can fit "
                                      "larger models into memory at a cost of slower training "
                                      "speed. 50%%-150%% batch size increase for 20%%-50%% longer "
                                      "training time. NB: Launch time will be significantly "
                                      "delayed. Switching sides using ping-pong training will "
                                      "take longer."})
        argument_list.append({"opts": ("-o", "--optimizer-savings"),
                              "dest": "optimizer_savings",
                              "action": "store_true",
                              "default": False,
                              "group": "VRAM Savings",
                              "backend": "nvidia",
                              "help": "To save VRAM some optimizer gradient calculations can be "
                                      "performed on the CPU rather than the GPU. This allows you "
                                      "to increase batchsize at a training speed/system RAM "
                                      "cost."})
        argument_list.append({"opts": ("-pp", "--ping-pong"),
                              "action": "store_true",
                              "dest": "pingpong",
                              "group": "VRAM Savings",
                              "default": False,
                              "backend": "nvidia",
                              "help": "Enable ping pong training. Trains one side at a time, "
                                      "switching sides at each save iteration. Training will "
                                      "take 2 to 4 times longer, with about a 30%%-50%% reduction "
                                      "in VRAM useage. NB: Preview won't show until both sides "
                                      "have been trained once."})
        argument_list.append({"opts": ("-s", "--save-interval"),
                              "type": int,
                              "action": Slider,
                              "min_max": (10, 1000),
                              "rounding": 10,
                              "dest": "save_interval",
                              "group": "Saving",
                              "default": 100,
                              "help": "Sets the number of iterations between each model save."})
        argument_list.append({"opts": ("-ss", "--snapshot-interval"),
                              "type": int,
                              "action": Slider,
                              "min_max": (0, 100000),
                              "rounding": 5000,
                              "dest": "snapshot_interval",
                              "group": "Saving",
                              "default": 25000,
                              "help": "Sets the number of iterations before saving a backup "
                                      "snapshot of the model in it's current state. Set to 0 for "
                                      "off."})
        argument_list.append({"opts": ("-tia", "--timelapse-input-A"),
                              "action": DirFullPaths,
                              "dest": "timelapse_input_a",
                              "default": None,
                              "group": "timelapse",
                              "help": "Optional for creating a timelapse. Timelapse will save an "
                                      "image of your selected faces into the timelapse-output "
                                      "folder at every save iteration. This should be the "
                                      "input folder of 'A' faces that you would like to use for "
                                      "creating the timelapse. You must also supply a "
                                      "--timelapse-output and a --timelapse-input-B parameter."})
        argument_list.append({"opts": ("-tib", "--timelapse-input-B"),
                              "action": DirFullPaths,
                              "dest": "timelapse_input_b",
                              "default": None,
                              "group": "timelapse",
                              "help": "Optional for creating a timelapse. Timelapse will save an "
                                      "image of your selected faces into the timelapse-output "
                                      "folder at every save iteration. This should be the "
                                      "input folder of 'B' faces that you would like to use for "
                                      "creating the timelapse. You must also supply a "
                                      "--timelapse-output and a --timelapse-input-A parameter."})
        argument_list.append({"opts": ("-to", "--timelapse-output"),
                              "action": DirFullPaths,
                              "dest": "timelapse_output",
                              "default": None,
                              "group": "timelapse",
                              "help": "Optional for creating a timelapse. Timelapse will save an "
                                      "image of your selected faces into the timelapse-output "
                                      "folder at every save iteration. If the input folders are "
                                      "supplied but no output folder, it will default to your "
                                      "model folder /timelapse/"})
        argument_list.append({"opts": ("-ps", "--preview-scale"),
                              "type": int,
                              "action": Slider,
                              "dest": "preview_scale",
                              "min_max": (25, 200),
                              "group": "preview",
                              "rounding": 25,
                              "default": 50,
                              "help": "Percentage amount to scale the preview by."})
        argument_list.append({"opts": ("-p", "--preview"),
                              "action": "store_true",
                              "dest": "preview",
                              "group": "preview",
                              "default": False,
                              "help": "Show training preview output. in a separate window."})
        argument_list.append({"opts": ("-w", "--write-image"),
                              "action": "store_true",
                              "dest": "write_image",
                              "group": "preview",
                              "default": False,
                              "help": "Writes the training result to a file. The image will be "
                                      "stored in the root of your FaceSwap folder."})
        argument_list.append({"opts": ("-ag", "--allow-growth"),
                              "action": "store_true",
                              "dest": "allow_growth",
                              "group": "model",
                              "default": False,
                              "backend": "nvidia",
                              "help": "Sets allow_growth option of Tensorflow to spare memory "
                                      "on some configurations."})
        argument_list.append({"opts": ("-nl", "--no-logs"),
                              "action": "store_true",
                              "dest": "no_logs",
                              "group": "training",
                              "default": False,
                              "help": "Disables TensorBoard logging. NB: Disabling logs means "
                                      "that you will not be able to use the graph or analysis "
                                      "for this session in the GUI."})
        argument_list.append({"opts": ("-wl", "--warp-to-landmarks"),
                              "action": "store_true",
                              "dest": "warp_to_landmarks",
                              "group": "training",
                              "default": False,
                              "help": "Warps training faces to closely matched Landmarks from the "
                                      "opposite face-set rather than randomly warping the face. "
                                      "This is the 'dfaker' way of doing warping. Alignments "
                                      "files for both sets of faces must be provided if using "
                                      "this option."})
        argument_list.append({"opts": ("-nf", "--no-flip"),
                              "action": "store_true",
                              "dest": "no_flip",
                              "group": "training",
                              "default": False,
                              "help": "To effectively learn, a random set of images are flipped "
                                      "horizontally. Sometimes it is desirable for this not to "
                                      "occur. Generally this should be left off except for "
                                      "during 'fit training'."})
        argument_list.append({"opts": ("-nac", "--no-augment-color"),
                              "action": "store_true",
                              "dest": "no_augment_color",
                              "group": "training",
                              "default": False,
                              "help": "Color augmentation helps make the model less susceptible "
                                      "to color differences between the A and B sets, at an "
                                      "increased training time cost. Enable this option to "
                                      "disable color augmentation."})
        return argument_list


class GuiArgs(FaceSwapArgs):
    """ Class to parse the command line arguments for training """

    @staticmethod
    def get_argument_list():
        """ Put the arguments in a list so that they are accessible from both
        argparse and gui """
        argument_list = []
        argument_list.append({
            "opts": ("-d", "--debug"),
            "action": "store_true",
            "dest": "debug",
            "default": False,
            "help": "Output to Shell console instead of GUI console"})
        return argument_list
