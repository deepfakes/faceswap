#!/usr/bin/env python3
""" Command Line Arguments """
import argparse
import logging
import os
import platform
import sys

from importlib import import_module

from lib.logger import crash_log, log_setup
from lib.utils import safe_shutdown
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
        self.test_for_gui()
        cmd = os.path.basename(sys.argv[0])
        src = "tools" if cmd == "tools.py" else "scripts"
        mod = ".".join((src, self.command.lower()))
        module = import_module(mod)
        script = getattr(module, self.command.title())
        return script

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
            logger.warning(
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
            exit(1)

    @staticmethod
    def check_display():
        """ Check whether there is a display to output the GUI. If running on
            Windows then assume not running in headless mode """
        if not os.environ.get("DISPLAY", None) and os.name != "nt":
            logger.warning("No display detected. GUI mode has been disabled.")
            if platform.system() == "Darwin":
                logger.info("macOS users need to install XQuartz. "
                            "See https://support.apple.com/en-gb/HT201341")
            exit(1)

    def execute_script(self, arguments):
        """ Run the script for called command """
        log_setup(arguments.loglevel, arguments.logfile, self.command)
        logger.debug("Executing: %s. PID: %s", self.command, os.getpid())
        try:
            script = self.import_script()
            process = script(arguments)
            process.process()
        except KeyboardInterrupt:  # pylint: disable=try-except-raise
            raise
        except SystemExit:
            pass
        except Exception:  # pylint: disable=broad-except
            crash_file = crash_log()
            logger.exception("Got Exception on main handler:")
            logger.critical("An unexpected crash has occurred. Crash report written to %s. "
                            "Please verify you are running the latest version of faceswap "
                            "before reporting", crash_file)

        finally:
            safe_shutdown()


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
        setattr(namespace, self.dest, os.path.abspath(
            os.path.expanduser(values)))


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
        super(FileFullPaths, self).__init__(option_strings, dest, **kwargs)
        if nargs is not None:
            raise ValueError("nargs not allowed")
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
        text.

        To use prefix the help item with "R|" to overide
        default formatting

        from: https://stackoverflow.com/questions/3853722 """

    def _split_lines(self, text, width):
        if text.startswith("R|"):
            return text[2:].splitlines()
        # this is the RawTextHelpFormatter._split_lines
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
        global_args.append({"opts": ("-L", "--loglevel"),
                            "type": str.upper,
                            "dest": "loglevel",
                            "default": "INFO",
                            "choices": ("INFO", "VERBOSE", "DEBUG", "TRACE"),
                            "help": "Log level. Stick with INFO or VERBOSE unless you need to "
                                    "file an error report. Be careful with TRACE as it will "
                                    "generate a lot of data"})
        global_args.append({"opts": ("-LF", "--logfile"),
                            "action": FileFullPaths,
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
                              "default": "input",
                              "help": "Input directory or video. Either a "
                                      "directory containing the image files "
                                      "you wish to process or path to a "
                                      "video file. Defaults to 'input'"})
        argument_list.append({"opts": ("-o", "--output-dir"),
                              "action": DirFullPaths,
                              "dest": "output_dir",
                              "default": "output",
                              "help": "Output directory. This is where the "
                                      "converted files will be stored. "
                                      "Defaults to 'output'"})
        argument_list.append({"opts": ("-al", "--alignments"),
                              "action": FileFullPaths,
                              "filetypes": 'alignments',
                              "type": str,
                              "dest": "alignments_path",
                              "help": "Optional path to an alignments file."})
        argument_list.append({"opts": ("-l", "--ref_threshold"),
                              "action": Slider,
                              "min_max": (0.01, 0.99),
                              "rounding": 2,
                              "type": float,
                              "dest": "ref_threshold",
                              "default": 0.6,
                              "help": "Threshold for positive face recognition. For use with "
                                      "nfilter or filter. Lower values are stricter."})
        argument_list.append({"opts": ("-n", "--nfilter"),
                              "type": str,
                              "dest": "nfilter",
                              "nargs": "+",
                              "default": None,
                              "help": "Reference image for the persons you do "
                                      "not want to process. Should be a front "
                                      "portrait. Multiple images can be added "
                                      "space separated"})
        argument_list.append({"opts": ("-f", "--filter"),
                              "type": str,
                              "dest": "filter",
                              "nargs": "+",
                              "default": None,
                              "help": "Reference images for the person you "
                                      "want to process. Should be a front "
                                      "portrait. Multiple images can be added "
                                      "space separated"})
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
            "type": str.lower,
            "choices":  PluginLoader.get_available_extractors(
                "detect"),
            "default": "mtcnn",
            "help": "R|Detector to use."
                    "\n'dlib-hog': uses least resources, but is the"
                    "\n\tleast reliable."
                    "\n'dlib-cnn': faster than mtcnn but detects"
                    "\n\tfewer faces and fewer false positives."
                    "\n'mtcnn': slower than dlib, but uses fewer"
                    "\n\tresources whilst detecting more faces and"
                    "\n\tmore false positives. Has superior"
                    "\n\talignment to dlib"})
        argument_list.append({
            "opts": ("-A", "--aligner"),
            "type": str.lower,
            "choices": PluginLoader.get_available_extractors(
                "align"),
            "default": "fan",
            "help": "R|Aligner to use."
                    "\n'dlib': Dlib Pose Predictor. Faster, less "
                    "\n\tresource intensive, but less accurate."
                    "\n'fan': Face Alignment Network. Best aligner."
                    "\n\tGPU heavy."})
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
        argument_list.append({"opts": ("-mp", "--multiprocess"),
                              "action": "store_true",
                              "default": False,
                              "help": "Run extraction in parallel. Offers "
                                      "speed up for some extractor/detector "
                                      "combinations, less so for others. "
                                      "Only has an effect if both the "
                                      "aligner and detector use the GPU, "
                                      "otherwise this is automatic."})
        argument_list.append({"opts": ("-sz", "--size"),
                              "type": int,
                              "action": Slider,
                              "min_max": (128, 512),
                              "default": 256,
                              "rounding": 64,
                              "help": "The output size of extracted faces. Make sure that the "
                                      "model you intend to train supports your required size. "
                                      "This will only need to be changed for hi-res models."})
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
                              "default": "models",
                              "help": "Model directory. A directory "
                                      "containing the trained model you wish "
                                      "to process. Defaults to 'models'"})
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
        argument_list.append({"opts": ("-t", "--trainer"),
                              "type": str.lower,
                              "choices": PluginLoader.get_available_models(),
                              "default": PluginLoader.get_default_model(),
                              "help": "Select the trainer that was used to "
                                      "create the model"})
        argument_list.append({"opts": ("-c", "--converter"),
                              "type": str.lower,
                              "choices": PluginLoader.get_available_converters(),
                              "default": "masked",
                              "help": "Converter to use"})
        argument_list.append({
            "opts": ("-M", "--mask-type"),
            "type": str.lower,
            "dest": "mask_type",
            "choices": ["rect",
                        "ellipse",
                        "smoothed",
                        "facehull",
                        "facehull_rect",
                        # "dfaker",
                        "dfl",
                        "cnn"],
            "default": "facehull_rect",
            "help": "R|Mask to use to replace faces."
                    "\nrect: Rectangle around face."
                    "\nellipse: Oval around face."
                    "\nsmoothed: Rectangle around face with smoothing."
                    "\nfacehull: Face cutout based on landmarks."
                    "\nfacehull_rect: Rectangle around faces with facehull"
                    "\n\tbetween the edges of the face and the background."
                    "\ndfl: A Face Hull mask from DeepFaceLabs."
                    "\ncnn: Not yet implemented"})
        argument_list.append({"opts": ("-cov", "--coverage"),
                              "type": float,
                              "dest": "coverage",
                              "action": Slider,
                              "min_max": (0.5, 1.0),
                              "rounding": 0.0625,
                              "default": .625,
                              "help": "Input images to the model are cropped to "
                                      "a central square that spans from eyebrow "
                                      "to chin cleft vertically and eyebrow to "
                                      "eyebrow horizontally at the default scale. "
                                      "0.625 spans from eyebrow to eyebrow, "
                                      "0.750 spans from temple to temple, "
                                      "0.875 spans from ear to ear, "
                                      "1.000 is a mugshot -- WARNING: Best left "
                                      "at default value of 0.625"})
        argument_list.append({"opts": ("-b", "--blur-size"),
                              "type": float,
                              "action": Slider,
                              "min_max": (0.0, 256.0),
                              "rounding": 0.05,
                              "default": 0.1,
                              "help": "Blur kernel size for smoothing the "
                                      "transition between the swapped face and "
                                      "the background image. Integer values "
                                      "will blur x pixels, fractions will blur "
                                      "that %% of the face area radius"})
        argument_list.append({"opts": ("-e", "--erosion-size"),
                              "dest": "erosion_size",
                              "type": float,
                              "action": Slider,
                              "min_max": (-100.0, 100.0),
                              "rounding": 0.05,
                              "default": 0.0,
                              "help": "Erosion kernel size. Positive values "
                                      "apply erosion which reduces the size "
                                      "of the swapped area. Negative values "
                                      "apply dilation which increases the "
                                      "swapped area. Abs values >1 use pixels "
                                      ". Fractions will erode/dilate that %% "
                                      " of the mask area radius"})
        argument_list.append({"opts": ("-g", "--gpus"),
                              "type": int,
                              "action": Slider,
                              "min_max": (1, 10),
                              "rounding": 1,
                              "default": 1,
                              "help": "Number of GPUs to use for conversion"})
        argument_list.append({"opts": ("-sh", "--sharpen"),
                              "type": str.lower,
                              "dest": "sharpen_image",
                              "choices": ["box_filter", "gaussian_filter"],
                              "default": None,
                              "help": "Sharpen the masked facial region of "
                                      "the converted images. Choice of filter "
                                      "to use in sharpening process -- box"
                                      "filter or gaussian filter."})
        argument_list.append({"opts": ("-fr", "--frame-ranges"),
                              "nargs": "+",
                              "type": str,
                              "help": "frame ranges to apply transfer to e.g. "
                                      "For frames 10 to 50 and 90 to 100 use "
                                      "--frame-ranges 10-50 90-100. Files "
                                      "must have the frame-number as the last "
                                      "number in the name!"})
        argument_list.append({"opts": ("-d", "--discard-frames"),
                              "action": "store_true",
                              "dest": "discard_frames",
                              "default": False,
                              "help": "When used with --frame-ranges discards "
                                      "frames that are not processed instead "
                                      "of writing them out unchanged"})
        argument_list.append({"opts": ("-s", "--swap-model"),
                              "action": "store_true",
                              "dest": "swap_model",
                              "default": False,
                              "help": "Swap the model. Instead of A -> B, "
                                      "swap B -> A"})
        argument_list.append({"opts": ("-S", "--seamless"),
                              "action": "store_true",
                              "dest": "seamless_clone",
                              "default": False,
                              "help": "Use cv2's seamless clone function to "
                                      "remove extreme gradients at the mask "
                                      "seam by smoothing colors."})
        argument_list.append({"opts": ("-mh", "--match-histogram"),
                              "action": "store_true",
                              "dest": "match_histogram",
                              "default": False,
                              "help": "Adjust the histogram of each color "
                                      "channel in the swapped reconstruction "
                                      "to equal the histogram of the masked "
                                      "area in the orginal image"})
        argument_list.append({"opts": ("-aca", "--avg-color-adjust"),
                              "action": "store_true",
                              "dest": "avg_color_adjust",
                              "default": False,
                              "help": "Adjust the mean of each color channel "
                                      " in the swapped reconstruction to "
                                      "equal the mean of the masked area in "
                                      "the orginal image"})
        argument_list.append({"opts": ("-dt", "--draw-transparent"),
                              "action": "store_true",
                              "dest": "draw_transparent",
                              "default": False,
                              "help": "Place the swapped face on a "
                                      "transparent layer rather than the "
                                      "original frame."})
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
                              "default": "input_a",
                              "help": "Input directory. A directory "
                                      "containing training images for face A. "
                                      "Defaults to 'input'"})
        argument_list.append({"opts": ("-B", "--input-B"),
                              "action": DirFullPaths,
                              "dest": "input_b",
                              "default": "input_b",
                              "help": "Input directory. A directory "
                                      "containing training images for face B. "
                                      "Defaults to 'input'"})
        argument_list.append({"opts": ("-m", "--model-dir"),
                              "action": DirFullPaths,
                              "dest": "model_dir",
                              "default": "models",
                              "help": "Model directory. This is where the "
                                      "training data will be stored. "
                                      "Defaults to 'model'"})
        argument_list.append({"opts": ("-s", "--save-interval"),
                              "type": int,
                              "action": Slider,
                              "min_max": (10, 1000),
                              "rounding": 10,
                              "dest": "save_interval",
                              "default": 100,
                              "help": "Sets the number of iterations before saving the model"})
        argument_list.append({"opts": ("-t", "--trainer"),
                              "type": str.lower,
                              "choices": PluginLoader.get_available_models(),
                              "default": PluginLoader.get_default_model(),
                              "help": "Select which trainer to use, Use "
                                      "LowMem for cards with less than 2GB of "
                                      "VRAM"})
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
        argument_list.append({"opts": ("-p", "--preview"),
                              "action": "store_true",
                              "dest": "preview",
                              "default": False,
                              "help": "Show preview output. If not specified, "
                                      "write progress to file"})
        argument_list.append({"opts": ("-w", "--write-image"),
                              "action": "store_true",
                              "dest": "write_image",
                              "default": False,
                              "help": "Writes the training result to a file "
                                      "even on preview mode"})
        argument_list.append({"opts": ("-ag", "--allow-growth"),
                              "action": "store_true",
                              "dest": "allow_growth",
                              "default": False,
                              "help": "Sets allow_growth option of Tensorflow "
                                      "to spare memory on some configs"})
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
