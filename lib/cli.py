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
from lib.utils import FaceswapError, safe_shutdown
from lib.model.masks import get_available_masks, get_default_mask
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
        max_ver = 1.13
        try:
            import tensorflow as tf
        except ImportError:
            raise FaceswapError("Tensorflow is a requirement but is not installed on your system.")
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
        if hasattr(arguments, "amd") and arguments.amd:
            plaidml_found = self.setup_amd(arguments.loglevel)
            if not plaidml_found:
                safe_shutdown()
                exit(1)
        try:
            script = self.import_script()
            process = script(arguments)
            process.process()
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
                            "Please verify you are running the latest version of faceswap "
                            "before reporting", crash_file)

        finally:
            safe_shutdown()

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
        self.argument_list = self.get_argument_list()
        self.optional_arguments = self.get_optional_arguments()
        if not subparser:
            return

        self.parser = self.create_parser(subparser, command, description)

        self.add_arguments()

        script = ScriptExecutor(command, subparsers)
        self.parser.set_defaults(func=script.execute_script)

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
        global_args.append({"opts": ("-amd", "--amd"),
                            "action": "store_true",
                            "dest": "amd",
                            "default": False,
                            "help": "AMD GPU users must enable this option for PlaidML support"})
        global_args.append({"opts": ("-C", "--configfile"),
                            "action": FileFullPaths,
                            "filetypes": "ini",
                            "type": str,
                            "help": "Optionally overide the saved config with the path to a "
                                    "custom config file."})
        global_args.append({"opts": ("-L", "--loglevel"),
                            "type": str.upper,
                            "dest": "loglevel",
                            "default": "INFO",
                            "choices": ("INFO", "VERBOSE", "DEBUG", "TRACE"),
                            "help": "Log level. Stick with INFO or VERBOSE unless you need to "
                                    "file an error report. Be careful with TRACE as it will "
                                    "generate a lot of data"})
        global_args.append({"opts": ("-LF", "--logfile"),
                            "action": SaveFileFullPaths,
                            "filetypes": 'log',
                            "type": str,
                            "dest": "logfile",
                            "help": "Path to store the logfile. Leave blank to store in the "
                                    "faceswap folder",
                            "default": None})
        # This is a hidden argument to indicate that the GUI is being used,
        # so the preview window should be redirected Accordingly
        global_args.append({"opts": ("-gui", "--gui"),
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
            epilog="Questions and feedback: \
            https://github.com/deepfakes/faceswap-playground",
            formatter_class=SmartFormatter)
        return parser

    def add_arguments(self):
        """ Parse the arguments passed in from argparse """
        options = self.global_arguments + self.argument_list + self.optional_arguments
        for option in options:
            args = option["opts"]
            kwargs = {key: option[key]
                      for key in option.keys() if key != "opts"}
            self.parser.add_argument(*args, **kwargs)


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
        argument_list.append({"opts": ("-i", "--input-dir"),
                              "action": DirOrFileFullPaths,
                              "filetypes": "video",
                              "dest": "input_dir",
                              "required": True,
                              "help": "Input directory or video. Either a directory containing "
                                      "the image files you wish to process or path to a video "
                                      "file. NB: This should be the source video/frames NOT the "
                                      "source faces."})
        argument_list.append({"opts": ("-o", "--output-dir"),
                              "action": DirFullPaths,
                              "dest": "output_dir",
                              "required": True,
                              "help": "Output directory. This is where the converted files will "
                                      "be saved."})
        argument_list.append({"opts": ("-al", "--alignments"),
                              "action": FileFullPaths,
                              "filetypes": "alignments",
                              "type": str,
                              "dest": "alignments_path",
                              "help": "Optional path to an alignments file."})
        argument_list.append({"opts": ("-l", "--ref_threshold"),
                              "action": Slider,
                              "min_max": (0.01, 0.99),
                              "rounding": 2,
                              "type": float,
                              "dest": "ref_threshold",
                              "default": 0.4,
                              "help": "Threshold for positive face recognition. For use with "
                                      "nfilter or filter. Lower values are stricter. NB: Using "
                                      "face filter will significantly decrease extraction speed."})
        argument_list.append({"opts": ("-n", "--nfilter"),
                              "action": FilesFullPaths,
                              "filetypes": "image",
                              "dest": "nfilter",
                              "nargs": "+",
                              "default": None,
                              "help": "Reference image for the persons you do not want to "
                                      "process. Should be a front portrait with a single person "
                                      "in the image. Multiple images can be added space "
                                      "separated. NB: Using face filter will significantly "
                                      "decrease extraction speed."})
        argument_list.append({"opts": ("-f", "--filter"),
                              "action": FilesFullPaths,
                              "filetypes": "image",
                              "dest": "filter",
                              "nargs": "+",
                              "default": None,
                              "help": "Reference images for the person you want to process. "
                                      "Should be a front portrait with a single person in the "
                                      "image. Multiple images can be added space separated. NB: "
                                      "Using face filter will significantly decrease extraction "
                                      "speed."})
        return argument_list


class ExtractArgs(ExtractConvertArgs):
    """ Class to parse the command line arguments for extraction.
        Inherits base options from ExtractConvertArgs where arguments
        that are used for both extract and convert should be placed """

    @staticmethod
    def get_optional_arguments():
        """ Put the arguments in a list so that they are accessible from both
        argparse and gui """
        argument_list = []
        argument_list.append({"opts": ("--serializer", ),
                              "type": str.lower,
                              "dest": "serializer",
                              "default": "json",
                              "choices": ("json", "pickle", "yaml"),
                              "help": "Serializer for alignments file. If "
                                      "yaml is chosen and not available, then "
                                      "json will be used as the default "
                                      "fallback."})
        argument_list.append({
            "opts": ("-D", "--detector"),
            "action": Radio,
            "type": str.lower,
            "choices":  PluginLoader.get_available_extractors("detect"),
            "default": "mtcnn",
            "help": "R|Detector to use. NB: Unless stated, all aligners will run on CPU for AMD "
                    "GPUs. Some of these have configurable settings in "
                    "'/config/extract.ini' or 'Edit > Configure Extract Plugins':"
                    "\nL|'cv2-dnn': A CPU only extractor, is the least reliable, but uses least "
                    "resources and runs fast on CPU. Use this if not using a GPU and time is "
                    "important."
                    "\nL|'mtcnn': Fast on GPU, slow on CPU. Uses fewer resources than other GPU "
                    "detectors but can often return more false positives."
                    "\nL|'s3fd': Fast on GPU, slow on CPU. Can detect more faces and fewer false "
                    "positives than other GPU detectors, but is a lot more resource intensive."})
        argument_list.append({
            "opts": ("-A", "--aligner"),
            "action": Radio,
            "type": str.lower,
            "choices": PluginLoader.get_available_extractors("align"),
            "default": "fan",
            "help": "R|Aligner to use. NB: Unless stated, all aligners will run on CPU for AMD "
                    "GPUs."
                    "\nL|'cv2-dnn': A cpu only CNN based landmark detector. Faster, less "
                    "resource intensive, but less accurate. Only use this if not using a gpu "
                    " and time is important."
                    "\nL|'fan': Face Alignment Network. Best aligner. GPU heavy, slow when not "
                    "running on GPU"
                    "\nL|'fan-amd': Face Alignment Network. Uses Keras backend to support AMD "
                    "Cards. Best aligner. GPU heavy, slow when not running on GPU"})
        argument_list.append({"opts": ("-nm", "--normalization"),
                              "action": Radio,
                              "type": str.lower,
                              "dest": "normalization",
                              "choices": ["none", "clahe", "hist", "mean"],
                              "default": "none",
                              "help": "R|Performing normalization can help the aligner better "
                                      "align faces with difficult lighting conditions at an "
                                      "extraction speed cost. Different methods will yield "
                                      "different results on different sets."
                                      "\nL|'none': Don't perform normalization on the face."
                                      "\nL|'clahe': Perform Contrast Limited Adaptive Histogram "
                                      "Equalization on the face."
                                      "\nL|'hist': Equalize the histograms on the RGB channels."
                                      "\nL|'mean': Normalize the face colors to the mean."})
        argument_list.append({"opts": ("-r", "--rotate-images"),
                              "type": str,
                              "dest": "rotate_images",
                              "default": None,
                              "help": "If a face isn't found, rotate the "
                                      "images to try to find a face. Can find "
                                      "more faces at the cost of extraction "
                                      "speed. Pass in a single number to use "
                                      "increments of that size up to 360, or "
                                      "pass in a list of numbers to enumerate "
                                      "exactly what angles to check"})
        argument_list.append({"opts": ("-bt", "--blur-threshold"),
                              "type": float,
                              "action": Slider,
                              "min_max": (0.0, 100.0),
                              "rounding": 1,
                              "dest": "blur_thresh",
                              "default": 0.0,
                              "help": "Automatically discard images blurrier than the specified "
                                      "threshold. Discarded images are moved into a \"blurry\" "
                                      "sub-folder. Lower values allow more blur. Set to 0.0 to "
                                      "turn off."})
        argument_list.append({"opts": ("-sp", "--singleprocess"),
                              "action": "store_true",
                              "default": False,
                              "help": "Don't run extraction in parallel. Will run detection first "
                                      "then alignment (2 passes). Useful if VRAM is at a premium. "
                                      "Only has an effect if both the aligner and detector use "
                                      "the GPU, otherwise this is automatically off. NB: AMD "
                                      "cards do not support parallel processing, so if both "
                                      "aligner and detector use an AMD GPU this will "
                                      "automatically be enabled."})
        argument_list.append({"opts": ("-sz", "--size"),
                              "type": int,
                              "action": Slider,
                              "min_max": (128, 512),
                              "default": 256,
                              "rounding": 64,
                              "help": "The output size of extracted faces. Make sure that the "
                                      "model you intend to train supports your required size. "
                                      "This will only need to be changed for hi-res models."})
        argument_list.append({"opts": ("-min", "--min-size"),
                              "type": int,
                              "action": Slider,
                              "dest": "min_size",
                              "min_max": (0, 1080),
                              "default": 0,
                              "rounding": 20,
                              "help": "Filters out faces detected below this size. Length, in "
                                      "pixels across the diagonal of the bounding box. Set to 0 "
                                      "for off"})
        argument_list.append({"opts": ("-een", "--extract-every-n"),
                              "type": int,
                              "action": Slider,
                              "dest": "extract_every_n",
                              "min_max": (1, 100),
                              "default": 1,
                              "rounding": 1,
                              "help": "Extract every 'nth' frame. This option will skip frames "
                                      "when extracting faces. For example a value of 1 will "
                                      "extract faces from every frame, a value of 10 will extract "
                                      "faces from every 10th frame."})
        argument_list.append({"opts": ("-s", "--skip-existing"),
                              "action": "store_true",
                              "dest": "skip_existing",
                              "default": False,
                              "help": "Skips frames that have already been "
                                      "extracted and exist in the alignments "
                                      "file"})
        argument_list.append({"opts": ("-sf", "--skip-existing-faces"),
                              "action": "store_true",
                              "dest": "skip_faces",
                              "default": False,
                              "help": "Skip frames that already have "
                                      "detected faces in the alignments "
                                      "file"})
        argument_list.append({"opts": ("-dl", "--debug-landmarks"),
                              "action": "store_true",
                              "dest": "debug_landmarks",
                              "default": False,
                              "help": "Draw landmarks on the ouput faces for "
                                      "debug"})
        argument_list.append({"opts": ("-ae", "--align-eyes"),
                              "action": "store_true",
                              "dest": "align_eyes",
                              "default": False,
                              "help": "Perform extra alignment to ensure "
                                      "left/right eyes are  at the same "
                                      "height"})
        argument_list.append({"opts": ("-si", "--save-interval"),
                              "dest": "save_interval",
                              "type": int,
                              "action": Slider,
                              "min_max": (0, 1000),
                              "rounding": 10,
                              "default": 0,
                              "help": "Automatically save the alignments file after a set amount "
                                      "of frames. Will only save at the end of extracting by "
                                      "default. WARNING: Don't interrupt the script when writing "
                                      "the file because it might get corrupted. Set to 0 to turn "
                                      "off"})
        return argument_list


class ConvertArgs(ExtractConvertArgs):
    """ Class to parse the command line arguments for conversion.
        Inherits base options from ExtractConvertArgs where arguments
        that are used for both extract and convert should be placed """

    @staticmethod
    def get_optional_arguments():
        """ Put the arguments in a list so that they are accessible from both
        argparse and gui """
        argument_list = []
        argument_list.append({"opts": ("-m", "--model-dir"),
                              "action": DirFullPaths,
                              "dest": "model_dir",
                              "required": True,
                              "help": "Model directory. A directory containing the trained model "
                                      "you wish to process."})
        argument_list.append({"opts": ("-a", "--input-aligned-dir"),
                              "action": DirFullPaths,
                              "dest": "input_aligned_dir",
                              "default": None,
                              "help": "Input \"aligned directory\". A "
                                      "directory that should contain the "
                                      "aligned faces extracted from the input "
                                      "files. If you delete faces from this "
                                      "folder, they'll be skipped during "
                                      "conversion. If no aligned dir is "
                                      "specified, all faces will be "
                                      "converted"})
        argument_list.append({"opts": ("-ref", "--reference-video"),
                              "action": FileFullPaths,
                              "dest": "reference_video",
                              "filetypes": "video",
                              "type": str,
                              "help": "Only required if converting from images to video. Provide "
                                      "The original video that the source frames were extracted "
                                      "from (for extracting the fps and audio)."})
        argument_list.append({
            "opts": ("-c", "--color-adjustment"),
            "action": Radio,
            "type": str.lower,
            "dest": "color_adjustment",
            "choices": PluginLoader.get_available_convert_plugins("color", True),
            "default": "avg-color",
            "help": "R|Performs color adjustment to the swapped face. Some of these options have "
                    "configurable settings in '/config/convert.ini' or 'Edit > Configure "
                    "Convert Plugins':"
                    "\nL|avg-color: Adjust the mean of each color channel in the swapped "
                    "reconstruction to equal the mean of the masked area in the orginal image."
                    "\nL|color-transfer: Transfers the color distribution from the source to the "
                    "target image using the mean and standard deviations of the L*a*b* "
                    "color space."
                    "\nL|manual-balance: Manually adjust the balance of the image in a variety of "
                    "color spaces. Best used with the Preview tool to set correct values."
                    "\nL|match-hist: Adjust the histogram of each color channel in the swapped "
                    "reconstruction to equal the histogram of the masked area in the orginal "
                    "image."
                    "\nL|seamless-clone: Use cv2's seamless clone function to remove extreme "
                    "gradients at the mask seam by smoothing colors. Generally does not give "
                    "very satisfactory results."
                    "\nL|none: Don't perform color adjustment."})
        argument_list.append({
            "opts": ("-sc", "--scaling"),
            "action": Radio,
            "type": str.lower,
            "choices": PluginLoader.get_available_convert_plugins("scaling", True),
            "default": "none",
            "help": "R|Performs a scaling process to attempt to get better definition on the "
                    "final swap. Some of these options have configurable settings in "
                    "'/config/convert.ini' or 'Edit > Configure Convert Plugins':"
                    "\nL|sharpen: Perform sharpening on the final face."
                    "\nL|none: Don't perform any scaling operations."})
        argument_list.append({
            "opts": ("-M", "--mask-type"),
            "action": Radio,
            "type": str.lower,
            "dest": "mask_type",
            "choices": get_available_masks() + ["predicted"],
            "default": "predicted",
            "help": "R|Mask to use to replace faces. Blending of the masks can be adjusted in "
                    "'/config/convert.ini' or 'Edit > Configure Convert Plugins':"
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
        argument_list.append({"opts": ("-w", "--writer"),
                              "action": Radio,
                              "type": str,
                              "choices": PluginLoader.get_available_convert_plugins("writer",
                                                                                    False),
                              "default": "opencv",
                              "help": "R|The plugin to use to output the converted images. The "
                                      "writers are configurable in '/config/convert.ini' or 'Edit "
                                      "> Configure Convert Plugins:'"
                                      "\nL|ffmpeg: [video] Writes out the convert straight to "
                                      "video. When the input is a series of images then the "
                                      "'-ref' (--reference-video) parameter must be set."
                                      "\nL|gif: [animated image] Create an animated gif."
                                      "\nL|opencv: [images] The fastest image writer, but less "
                                      "options and formats than other plugins."
                                      "\nL|pillow: [images] Slower than opencv, but has more "
                                      "options and supports more formats."})
        argument_list.append({"opts": ("-osc", "--output-scale"),
                              "dest": "output_scale",
                              "action": Slider,
                              "type": int,
                              "default": 100,
                              "min_max": (25, 400),
                              "rounding": 1,
                              "help": "Scale the final output frames by this amount. 100%% will "
                                      "output the frames at source dimensions. 50%% at half size "
                                      "200%% at double size"})
        argument_list.append({"opts": ("-j", "--jobs"),
                              "dest": "jobs",
                              "action": Slider,
                              "type": int,
                              "default": 0,
                              "min_max": (0, 40),
                              "rounding": 1,
                              "help": "The maximum number of parallel processes for performing "
                                      "conversion. Converting images is system RAM heavy so it is "
                                      "possible to run out of memory if you have a lot of "
                                      "processes and not enough RAM to accomodate them all. "
                                      "Setting this to 0 will use the maximum available. No "
                                      "matter what you set this to, it will never attempt to use "
                                      "more processes than are available on your system. If "
                                      "singleprocess is enabled this setting will be ignored."})
        argument_list.append({"opts": ("-g", "--gpus"),
                              "type": int,
                              "action": Slider,
                              "min_max": (1, 10),
                              "rounding": 1,
                              "default": 1,
                              "help": "Number of GPUs to use for conversion"})
        argument_list.append({"opts": ("-fr", "--frame-ranges"),
                              "nargs": "+",
                              "type": str,
                              "help": "frame ranges to apply transfer to e.g. For frames 10 to 50 "
                                      "and 90 to 100 use --frame-ranges 10-50 90-100. Files "
                                      "must have the frame-number as the last number in the name! "
                                      "Frames falling outside of the selected range will be "
                                      "discarded unless '-k' (--keep-unchanged) is selected."})
        argument_list.append({"opts": ("-k", "--keep-unchanged"),
                              "action": "store_true",
                              "dest": "keep_unchanged",
                              "default": False,
                              "help": "When used with --frame-ranges outputs the unchanged frames "
                                      "that are not processed instead of discarding them."})
        argument_list.append({"opts": ("-s", "--swap-model"),
                              "action": "store_true",
                              "dest": "swap_model",
                              "default": False,
                              "help": "Swap the model. Instead of A -> B, "
                                      "swap B -> A"})
        argument_list.append({"opts": ("-sp", "--singleprocess"),
                              "action": "store_true",
                              "default": False,
                              "help": "Disable multiprocessing. Slower but less resource "
                                      "intensive."})
        argument_list.append({"opts": ("-t", "--trainer"),
                              "type": str.lower,
                              "choices": PluginLoader.get_available_models(),
                              "help": "[LEGACY] This only needs to be selected if a legacy "
                                      "model is being loaded or if there are multiple models in "
                                      "the  model folder"})

        return argument_list


class TrainArgs(FaceSwapArgs):
    """ Class to parse the command line arguments for training """

    @staticmethod
    def get_argument_list():
        """ Put the arguments in a list so that they are accessible from both
        argparse and gui """
        argument_list = list()
        argument_list.append({"opts": ("-A", "--input-A"),
                              "action": DirFullPaths,
                              "dest": "input_a",
                              "required": True,
                              "help": "Input directory. A directory containing training images "
                                      "for face A."})
        argument_list.append({"opts": ("-B", "--input-B"),
                              "action": DirFullPaths,
                              "dest": "input_b",
                              "required": True,
                              "help": "Input directory. A directory containing training images "
                                      "for face B."})
        argument_list.append({"opts": ("-ala", "--alignments-A"),
                              "action": FileFullPaths,
                              "filetypes": 'alignments',
                              "type": str,
                              "dest": "alignments_path_a",
                              "default": None,
                              "help": "Path to alignments file for training set A. Only required "
                                      "if you are using a masked model or warp-to-landmarks is "
                                      "enabled. Defaults to <input-A>/alignments.json if not "
                                      "provided."})
        argument_list.append({"opts": ("-alb", "--alignments-B"),
                              "action": FileFullPaths,
                              "filetypes": 'alignments',
                              "type": str,
                              "dest": "alignments_path_b",
                              "default": None,
                              "help": "Path to alignments file for training set B. Only required "
                                      "if you are using a masked model or warp-to-landmarks is "
                                      "enabled. Defaults to <input-B>/alignments.json if not "
                                      "provided."})
        argument_list.append({"opts": ("-m", "--model-dir"),
                              "action": DirFullPaths,
                              "dest": "model_dir",
                              "required": True,
                              "help": "Model directory. This is where the training data will be "
                                      "stored."})
        argument_list.append({"opts": ("-t", "--trainer"),
                              "action": Radio,
                              "type": str.lower,
                              "choices": PluginLoader.get_available_models(),
                              "default": PluginLoader.get_default_model(),
                              "help": "R|Select which trainer to use. Trainers can be"
                                      "configured from the edit menu or the config folder."
                                      "\nL|original: The original model created by /u/deepfakes."
                                      "\nL|dfaker: 64px in/128px out model from dfaker. "
                                      "Enable 'warp-to-landmarks' for full dfaker method."
                                      "\nL|dfl-h128. 128px in/out model from deepfacelab"
                                      "\nL|iae: A model that uses intermediate layers to try to "
                                      "get better details"
                                      "\nL|lightweight: A lightweight model for low-end cards. "
                                      "Don't expect great results. Can train as low as 1.6GB "
                                      "with batch size 8."
                                      "\nL|realface: Customizable in/out resolution model "
                                      "from andenixa. The autoencoders are unbalanced so B>A "
                                      "swaps won't work so well. Very configurable."
                                      "\nL|unbalanced: 128px in/out model from andenixa. The "
                                      "autoencoders are unbalanced so B>A swaps won't work so "
                                      "well. Very configurable."
                                      "\nL|villain: 128px in/out model from villainguy. Very "
                                      "resource hungry (11GB for batchsize 16). Good for "
                                      "details, but more susceptible to color differences."})
        argument_list.append({"opts": ("-s", "--save-interval"),
                              "type": int,
                              "action": Slider,
                              "min_max": (10, 1000),
                              "rounding": 10,
                              "dest": "save_interval",
                              "default": 100,
                              "help": "Sets the number of iterations before saving the model"})
        argument_list.append({"opts": ("-ss", "--snapshot-interval"),
                              "type": int,
                              "action": Slider,
                              "min_max": (0, 100000),
                              "rounding": 5000,
                              "dest": "snapshot_interval",
                              "default": 25000,
                              "help": "Sets the number of iterations before saving a backup "
                                      "snapshot of the model in it's current state. Set to 0 for "
                                      "off."})
        argument_list.append({"opts": ("-bs", "--batch-size"),
                              "type": int,
                              "action": Slider,
                              "min_max": (2, 256),
                              "rounding": 2,
                              "dest": "batch_size",
                              "default": 64,
                              "help": "Batch size, as a power of 2 (64, 128, 256, etc)"})
        argument_list.append({"opts": ("-it", "--iterations"),
                              "type": int,
                              "action": Slider,
                              "min_max": (0, 5000000),
                              "rounding": 20000,
                              "default": 1000000,
                              "help": "Length of training in iterations."})
        argument_list.append({"opts": ("-g", "--gpus"),
                              "type": int,
                              "action": Slider,
                              "min_max": (1, 10),
                              "rounding": 1,
                              "default": 1,
                              "help": "Number of GPUs to use for training"})
        argument_list.append({"opts": ("-ps", "--preview-scale"),
                              "type": int,
                              "action": Slider,
                              "dest": "preview_scale",
                              "min_max": (25, 200),
                              "rounding": 25,
                              "default": 50,
                              "help": "Percentage amount to scale the preview by."})
        argument_list.append({"opts": ("-p", "--preview"),
                              "action": "store_true",
                              "dest": "preview",
                              "default": False,
                              "help": "Show training preview output. in a separate window."})
        argument_list.append({"opts": ("-w", "--write-image"),
                              "action": "store_true",
                              "dest": "write_image",
                              "default": False,
                              "help": "Writes the training result to a file. The image will be "
                                      "stored in the root of your FaceSwap folder."})
        argument_list.append({"opts": ("-ag", "--allow-growth"),
                              "action": "store_true",
                              "dest": "allow_growth",
                              "default": False,
                              "help": "Sets allow_growth option of Tensorflow "
                                      "to spare memory on some configs"})
        argument_list.append({"opts": ("-nl", "--no-logs"),
                              "action": "store_true",
                              "dest": "no_logs",
                              "default": False,
                              "help": "Disables TensorBoard logging. NB: Disabling logs means "
                                      "that you will not be able to use the graph or analysis "
                                      "for this session in the GUI."})
        argument_list.append({"opts": ("-pp", "--ping-pong"),
                              "action": "store_true",
                              "dest": "pingpong",
                              "default": False,
                              "help": "Enable ping pong training. Trains one side at a time, "
                                      "switching sides at each save iteration. Training will take "
                                      "2 to 4 times longer, with about a 30%%-50%% reduction in "
                                      "VRAM useage. NB: Preview won't show until both sides have "
                                      "been trained once."})
        argument_list.append({"opts": ("-msg", "--memory-saving-gradients"),
                              "action": "store_true",
                              "dest": "memory_saving_gradients",
                              "default": False,
                              "help": "Trades off VRAM useage against computation time. Can fit "
                                      "larger models into memory at a cost of slower training "
                                      "speed. 50%%-150%% batch size increase for 20%%-50%% longer "
                                      "training time. NB: Launch time will be significantly "
                                      "delayed. Switching sides using ping-pong training will "
                                      "take longer."})
        argument_list.append({"opts": ("-wl", "--warp-to-landmarks"),
                              "action": "store_true",
                              "dest": "warp_to_landmarks",
                              "default": False,
                              "help": "Warps training faces to closely matched Landmarks from the "
                                      "opposite face-set rather than randomly warping the face. "
                                      "This is the 'dfaker' way of doing warping. Alignments "
                                      "files for both sets of faces must be provided if using "
                                      "this option."})
        argument_list.append({"opts": ("-nf", "--no-flip"),
                              "action": "store_true",
                              "dest": "no_flip",
                              "default": False,
                              "help": "To effectively learn, a random set of images are flipped "
                                      "horizontally. Sometimes it is desirable for this not to "
                                      "occur. Generally this should be left off except for "
                                      "during 'fit training'."})
        argument_list.append({"opts": ("-nac", "--no-augment-color"),
                              "action": "store_true",
                              "dest": "no_augment_color",
                              "default": False,
                              "help": "Color augmentation helps make the model less susceptible "
                                      "to color differences between the A and B sets, at an "
                                      "increased training time cost. Enable this option to "
                                      "disable color augmentation."})
        argument_list.append({"opts": ("-tia", "--timelapse-input-A"),
                              "action": DirFullPaths,
                              "dest": "timelapse_input_a",
                              "default": None,
                              "help": "For if you want a timelapse: "
                                      "The input folder for the timelapse. "
                                      "This folder should contain faces of A "
                                      "which will be converted for the "
                                      "timelapse. You must supply a "
                                      "--timelapse-output and a "
                                      "--timelapse-input-B parameter."})
        argument_list.append({"opts": ("-tib", "--timelapse-input-B"),
                              "action": DirFullPaths,
                              "dest": "timelapse_input_b",
                              "default": None,
                              "help": "For if you want a timelapse: "
                                      "The input folder for the timelapse. "
                                      "This folder should contain faces of B "
                                      "which will be converted for the "
                                      "timelapse. You must supply a "
                                      "--timelapse-output and a "
                                      "--timelapse-input-A parameter."})
        argument_list.append({"opts": ("-to", "--timelapse-output"),
                              "action": DirFullPaths,
                              "dest": "timelapse_output",
                              "default": None,
                              "help": "The output folder for the timelapse. "
                                      "If the input folders are supplied but "
                                      "no output folder, it will default to "
                                      "your model folder /timelapse/"})
        return argument_list


class GuiArgs(FaceSwapArgs):
    """ Class to parse the command line arguments for training """

    @staticmethod
    def get_argument_list():
        """ Put the arguments in a list so that they are accessible from both
        argparse and gui """
        argument_list = []
        argument_list.append({"opts": ("-d", "--debug"),
                              "action": "store_true",
                              "dest": "debug",
                              "default": False,
                              "help": "Output to Shell console instead of "
                                      "GUI console"})
        return argument_list
