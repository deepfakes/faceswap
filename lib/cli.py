#!/usr/bin python3
""" Command Line Arguments """

import argparse
import os
import sys

from plugins.PluginLoader import PluginLoader

class ScriptExecutor(object):
    """ Loads the relevant script modules and executes the script.
        This class is initialised in each of the argparsers for the relevant command,
        then execute script is called within their set_default function """

    def __init__(self, command, subparsers=None):
        self.command = command.lower()
        self.subparsers = subparsers

    def import_script(self):
        """ Only import a script's modules when running that script."""
        if self.command == 'extract':
            from scripts.extract import Extract as script
        elif self.command == 'train':
            from scripts.train import Train as script
        elif self.command == 'convert':
            from scripts.convert import Convert as script
        elif self.command == 'gui':
            from scripts.gui import Gui as script
        else:
            script = None
        return script

    def execute_script(self, arguments):
        """ Run the script for called command """
        script = self.import_script()
        args = (arguments, ) if self.command != 'gui' else (arguments, self.subparsers)
        process = script(*args)
        process.process()

class FullPaths(argparse.Action):
    """Expand user- and relative-paths"""
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, os.path.abspath(
            os.path.expanduser(values)))

class FullHelpArgumentParser(argparse.ArgumentParser):
    """ Identical to the built-in argument parser, but on error it
        prints full help message instead of just usage information """
    def error(self, message):
        self.print_help(sys.stderr)
        args = {"prog": self.prog, "message": message}
        self.exit(2, "%(prog)s: error: %(message)s\n" % args)

class FaceSwapArgs(object):
    """ Faceswap argument parser functions that are universal
        to all commands. Should be the parent function of all
        subsequent argparsers """
    def __init__(self, subparser, command, description="default", subparsers=None):
        self.argument_list = self.get_argument_list()
        self.optional_arguments = self.get_optional_arguments()
        self.parser = self.create_parser(subparser, command, description)

        self.add_arguments()

        script = ScriptExecutor(command, subparsers)
        self.parser.set_defaults(func=script.execute_script)

    @staticmethod
    def get_argument_list():
        """ Put the arguments in a list so that they are accessible from both argparse and gui
            overide for command specific arguments """
        argument_list = []
        return argument_list

    @staticmethod
    def get_optional_arguments():
        """ Put the arguments in a list so that they are accessible from both argparse and gui
            This is used for when there are sub-children (e.g. convert and extract)
            Override this for custom arguments """
        argument_list = []
        return argument_list

    @staticmethod
    def create_parser(subparser, command, description):
        """ Create the parser for the selected command """
        parser = subparser.add_parser(
            command,
            help=description,
            description=description,
            epilog="Questions and feedback: \
            https://github.com/deepfakes/faceswap-playground")
        return parser

    def add_arguments(self):
        """ Parse the arguments passed in from argparse """
        for option in self.argument_list + self.optional_arguments:
            args = option["opts"]
            kwargs = {key: option[key] for key in option.keys() if key != "opts"}
            self.parser.add_argument(*args, **kwargs)


class ExtractConvertArgs(FaceSwapArgs):
    """ This class is used as a parent class to capture arguments that
        will be used in both the extract and convert process.

        Arguments that can be used in both of these processes should be
        placed here, but no further processing should be done. This class
        just captures arguments """

    @staticmethod
    def get_argument_list():
        """ Put the arguments in a list so that they are accessible from both argparse and gui """
        argument_list = []
        argument_list.append({"opts": ("-i", "--input-dir"),
                              "action": FullPaths,
                              "dest": "input_dir",
                              "default": "input",
                              "help": "Input directory. A directory containing the files "
                                      "you wish to process. Defaults to 'input'"})
        argument_list.append({"opts": ("-o", "--output-dir"),
                              "action": FullPaths,
                              "dest": "output_dir",
                              "default": "output",
                              "help": "Output directory. This is where the converted files will "
                                      "be stored. Defaults to 'output'"})
        argument_list.append({"opts": ("--alignments", ),
                              "type": str,
                              "dest": "alignments_path",
                              "help": "optional path to alignments file"})
        argument_list.append({"opts": ("--serializer", ),
                              "type": str.lower,
                              "dest": "serializer",
                              "choices": ("yaml", "json", "pickle"),
                              "help": "serializer for alignments file"})
        argument_list.append({"opts": ("-D", "--detector"),
                              "type": str,
                              # case sensitive because this is used to load a plugin.
                              "choices": ("hog", "cnn", "all"),
                              "default": "hog",
                              "help": "Detector to use. 'cnn' detects many more angles but "
                                      "will be much more resource intensive and may fail "
                                      "on large files"})
        argument_list.append({"opts": ("-l", "--ref_threshold"),
                              "type": float,
                              "dest": "ref_threshold",
                              "default": 0.6,
                              "help": "Threshold for positive face recognition"})
        argument_list.append({"opts": ("-n", "--nfilter"),
                              "type": str,
                              "dest": "nfilter",
                              "nargs": "+",
                              "default": None,
                              "help": "Reference image for the persons you do not want to "
                                      "process. Should be a front portrait. Multiple images"
                                      "can be added space separated"})
        argument_list.append({"opts": ("-f", "--filter"),
                              "type": str,
                              "dest": "filter",
                              "nargs": "+",
                              "default": None,
                              "help": "Reference images for the person you want to process. "
                                      "Should be a front portrait. Multiple images"
                                      "can be added space separated"})
        argument_list.append({"opts": ("-v", "--verbose"),
                              "action": "store_true",
                              "dest": "verbose",
                              "default": False,
                              "help": "Show verbose output"})
        return argument_list

class ExtractArgs(ExtractConvertArgs):
    """ Class to parse the command line arguments for extraction.
        Inherits base options from ExtractConvertArgs where arguments
        that are used for both extract and convert should be placed """

    @staticmethod
    def get_optional_arguments():
        """ Put the arguments in a list so that they are accessible from both argparse and gui """
        argument_list = []
        argument_list.append({"opts": ("-r", "--rotate-images"),
                              "type": str,
                              "dest": "rotate_images",
                              "default": None,
                              "help": "If a face isn't found, rotate the images to try to "
                                      "find a face. Can find more faces at the cost of extraction "
                                      "speed. Pass in a single number to use increments of that "
                                      "size up to 360, or pass in a list of numbers to enumerate "
                                      "exactly what angles to check"})
        argument_list.append({"opts": ("-bt", "--blur-threshold"),
                              "type": int,
                              "dest": "blur_thresh",
                              "default": None,
                              "help": "Automatically discard images blurrier than the specified "
                                      "threshold. Discarded images are moved into a \"blurry\" "
                                      "sub-folder. Lower values allow more blur"})
        argument_list.append({"opts": ("-j", "--processes"),
                              "type": int,
                              "default": 1,
                              "help": "Number of CPU processes to use. WARNING: ONLY USE THIS "
                                      " IF YOU ARE NOT EXTRACTING ON A GPU. Anything above 1 "
                                      " process on a GPU will run out of memory and will crash"})
        argument_list.append({"opts": ("-s", "--skip-existing"),
                              "action": "store_true",
                              "dest": "skip_existing",
                              "default": False,
                              "help": "Skips frames that have already been extracted"})
        argument_list.append({"opts": ("-dl", "--debug-landmarks"),
                              "action": "store_true",
                              "dest": "debug_landmarks",
                              "default": False,
                              "help": "Draw landmarks on the ouput faces for debug"})
        argument_list.append({"opts": ("-ae", "--align-eyes"),
                              "action": "store_true",
                              "dest": "align_eyes",
                              "default": False,
                              "help": "Perform extra alignment to ensure left/right eyes "
                                      "are  at the same height"})
        return argument_list

class ConvertArgs(ExtractConvertArgs):
    """ Class to parse the command line arguments for conversion.
        Inherits base options from ExtractConvertArgs where arguments
        that are used for both extract and convert should be placed """

    @staticmethod
    def get_optional_arguments():
        """ Put the arguments in a list so that they are accessible from both argparse and gui """
        argument_list = []
        argument_list.append({"opts": ("-m", "--model-dir"),
                              "action": FullPaths,
                              "dest": "model_dir",
                              "default": "models",
                              "help": "Model directory. A directory containing the trained model "
                                      "you wish to process. Defaults to 'models'"})
        argument_list.append({"opts": ("-a", "--input-aligned-dir"),
                              "action": FullPaths,
                              "dest": "input_aligned_dir",
                              "default": None,
                              "help": "Input \"aligned directory\". A directory that should "
                                      "contain the aligned faces extracted from the input files. "
                                      "If you delete faces from this folder, they'll be skipped "
                                      "during conversion. If no aligned dir is specified, all "
                                      "faces will be converted"})
        argument_list.append({"opts": ("-t", "--trainer"),
                              "type": str,
                              # case sensitive because this is used to load a plug-in.
                              "choices": PluginLoader.get_available_models(),
                              "default": PluginLoader.get_default_model(),
                              "help": "Select the trainer that was used to create the model"})
        argument_list.append({"opts": ("-c", "--converter"),
                              "type": str,
                              # case sensitive because this is used to load a plugin.
                              "choices": ("Masked", "Adjust"),
                              "default": "Masked",
                              "help": "Converter to use"})
        argument_list.append({"opts": ("-b", "--blur-size"),
                              "type": int,
                              "default": 2,
                              "help": "Blur size. (Masked converter only)"})
        argument_list.append({"opts": ("-e", "--erosion-kernel-size"),
                              "dest": "erosion_kernel_size",
                              "type": int,
                              "default": None,
                              "help": "Erosion kernel size. Positive values apply erosion "
                                      "which reduces the edge of the swapped face. Negative "
                                      "values apply dilation which allows the swapped face "
                                      "to cover more space. (Masked converter only)"})
        argument_list.append({"opts": ("-M", "--mask-type"),
                              #lowercase this, because its just a string later on.
                              "type": str.lower,
                              "dest": "mask_type",
                              "choices": ["rect", "facehull", "facehullandrect"],
                              "default": "facehullandrect",
                              "help": "Mask to use to replace faces. (Masked converter only)"})
        argument_list.append({"opts": ("-sh", "--sharpen"),
                              "type": str.lower,
                              "dest": "sharpen_image",
                              "choices": ["bsharpen", "gsharpen"],
                              "default": None,
                              "help": "Use Sharpen Image.bsharpen for Box Blur, gsharpen for "
                                      "Gaussian Blur (Masked converter only)"})
        argument_list.append({"opts": ("-g", "--gpus"),
                              "type": int,
                              "default": 1,
                              "help": "Number of GPUs to use for conversion"})
        argument_list.append({"opts": ("-fr", "--frame-ranges"),
                              "nargs": "+",
                              "type": str,
                              "help": "frame ranges to apply transfer to e.g. For frames 10 to "
                                      "50 and 90 to 100 use --frame-ranges 10-50 90-100. Files "
                                      "must have the frame-number as the last number in the "
                                      "name!"})
        argument_list.append({"opts": ("-d", "--discard-frames"),
                              "action": "store_true",
                              "dest": "discard_frames",
                              "default": False,
                              "help": "When used with --frame-ranges discards frames that are "
                                      "not processed instead of writing them out unchanged"})
        argument_list.append({"opts": ("-s", "--swap-model"),
                              "action": "store_true",
                              "dest": "swap_model",
                              "default": False,
                              "help": "Swap the model. Instead of A -> B, swap B -> A"})
        argument_list.append({"opts": ("-S", "--seamless"),
                              "action": "store_true",
                              "dest": "seamless_clone",
                              "default": False,
                              "help": "Use cv2's seamless clone. (Masked converter only)"})
        argument_list.append({"opts": ("-mh", "--match-histgoram"),
                              "action": "store_true",
                              "dest": "match_histogram",
                              "default": False,
                              "help": "Use histogram matching. (Masked converter only)"})
        argument_list.append({"opts": ("-sm", "--smooth-mask"),
                              "action": "store_true",
                              "dest": "smooth_mask",
                              "default": True,
                              "help": "Smooth mask (Adjust converter only)"})
        argument_list.append({"opts": ("-aca", "--avg-color-adjust"),
                              "action": "store_true",
                              "dest": "avg_color_adjust",
                              "default": True,
                              "help": "Average color adjust. (Adjust converter only)"})
        return argument_list

class TrainArgs(FaceSwapArgs):
    """ Class to parse the command line arguments for training """

    @staticmethod
    def get_argument_list():
        """ Put the arguments in a list so that they are accessible from both argparse and gui """
        argument_list = []
        argument_list.append({"opts": ("-A", "--input-A"),
                              "action": FullPaths,
                              "dest": "input_A",
                              "default": "input_A",
                              "help": "Input directory. A directory containing training images "
                                      "for face A. Defaults to 'input'"})
        argument_list.append({"opts": ("-B", "--input-B"),
                              "action": FullPaths,
                              "dest": "input_B",
                              "default": "input_B",
                              "help": "Input directory. A directory containing training images "
                                      "for face B Defaults to 'input'"})
        argument_list.append({"opts": ("-m", "--model-dir"),
                              "action": FullPaths,
                              "dest": "model_dir",
                              "default": "models",
                              "help": "Model directory. This is where the training data will "
                                      "be stored. Defaults to 'model'"})
        argument_list.append({"opts": ("-s", "--save-interval"),
                              "type": int,
                              "dest": "save_interval",
                              "default": 100,
                              "help": "Sets the number of iterations before saving the model"})
        argument_list.append({"opts": ("-t", "--trainer"),
                              "type": str,
                              "choices": PluginLoader.get_available_models(),
                              "default": PluginLoader.get_default_model(),
                              "help": "Select which trainer to use, Use LowMem for cards with "
                                      " less than 2GB of VRAM"})
        argument_list.append({"opts": ("-bs", "--batch-size"),
                              "type": int,
                              "default": 64,
                              "help": "Batch size, as a power of 2 (64, 128, 256, etc)"})
        argument_list.append({"opts": ("-ep", "--epochs"),
                              "type": int,
                              "default": 1000000,
                              "help": "Length of training in epochs"})
        argument_list.append({"opts": ("-g", "--gpus"),
                              "type": int,
                              "default": 1,
                              "help": "Number of GPUs to use for training"})
        argument_list.append({"opts": ("-p", "--preview"),
                              "action": "store_true",
                              "dest": "preview",
                              "default": False,
                              "help": "Show preview output. If not specified, write progress "
                                      "to file"})
        argument_list.append({"opts": ("-w", "--write-image"),
                              "action": "store_true",
                              "dest": "write_image",
                              "default": False,
                              "help": "Writes the training result to a file even on "
                                      "preview mode"})
        argument_list.append({"opts": ("-pl", "--use-perceptual-loss"),
                              "action": "store_true",
                              "dest": "perceptual_loss",
                              "default": False,
                              "help": "Use perceptual loss while training"})
        argument_list.append({"opts": ("-ag", "--allow-growth"),
                              "action": "store_true",
                              "dest": "allow_growth",
                              "default": False,
                              "help": "Sets allow_growth option of Tensorflow to spare memory "
                                      "on some configs"})
        argument_list.append({"opts": ("-v", "--verbose"),
                              "action": "store_true",
                              "dest": "verbose",
                              "default": False,
                              "help": "Show verbose output"})
        # This is a hidden argument to indicate that the GUI is being used,
        # so the preview window should be redirected Accordingly
        argument_list.append({"opts": ("-gui", "--gui"),
                              "action": "store_true",
                              "dest": "redirect_gui",
                              "default": False,
                              "help": argparse.SUPPRESS})
        return argument_list

class GuiArgs(FaceSwapArgs):
    """ Class to parse the command line arguments for training """

    @staticmethod
    def get_argument_list():
        """ Put the arguments in a list so that they are accessible from both argparse and gui """
        argument_list = []
        argument_list.append({"opts": ("-d", "--debug"),
                              "action": "store_true",
                              "dest": "debug",
                              "default": False,
                              "help": "Output to Shell console instead of GUI console"})
        return argument_list
