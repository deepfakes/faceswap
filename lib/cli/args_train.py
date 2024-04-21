#!/usr/bin/env python3
""" The Command Line Argument options for training with faceswap.py """
import argparse
import gettext
import typing as T

from plugins.plugin_loader import PluginLoader

from .actions import DirFullPaths, FileFullPaths, Radio, Slider
from .args import FaceSwapArgs


# LOCALES
_LANG = gettext.translation("lib.cli.args_train", localedir="locales", fallback=True)
_ = _LANG.gettext


class TrainArgs(FaceSwapArgs):
    """ Creates the command line arguments for training. """

    @staticmethod
    def get_info() -> str:
        """ The information text for the Train command.

        Returns
        -------
        str
            The information text for the Train command.
        """
        return _("Train a model on extracted original (A) and swap (B) faces.\n"
                 "Training models can take a long time. Anything from 24hrs to over a week\n"
                 "Model plugins can be configured in the 'Settings' Menu")

    @staticmethod
    def get_argument_list() -> list[dict[str, T.Any]]:
        """ Returns the argument list for Train arguments.

        Returns
        -------
        list
            The list of command line options for training
        """
        argument_list: list[dict[str, T.Any]] = []
        argument_list.append({
            "opts": ("-A", "--input-A"),
            "action": DirFullPaths,
            "dest": "input_a",
            "required": True,
            "group": _("faces"),
            "help": _(
                "Input directory. A directory containing training images for face A. This is the "
                "original face, i.e. the face that you want to remove and replace with face B.")})
        argument_list.append({
            "opts": ("-B", "--input-B"),
            "action": DirFullPaths,
            "dest": "input_b",
            "required": True,
            "group": _("faces"),
            "help": _(
                "Input directory. A directory containing training images for face B. This is the "
                "swap face, i.e. the face that you want to place onto the head of person A.")})
        argument_list.append({
            "opts": ("-m", "--model-dir"),
            "action": DirFullPaths,
            "dest": "model_dir",
            "required": True,
            "group": _("model"),
            "help": _(
                "Model directory. This is where the training data will be stored. You should "
                "always specify a new folder for new models. If starting a new model, select "
                "either an empty folder, or a folder which does not exist (which will be "
                "created). If continuing to train an existing model, specify the location of the "
                "existing model.")})
        argument_list.append({
            "opts": ("-l", "--load-weights"),
            "action": FileFullPaths,
            "filetypes": "model",
            "dest": "load_weights",
            "required": False,
            "group": _("model"),
            "help": _(
                "R|Load the weights from a pre-existing model into a newly created model. For "
                "most models this will load weights from the Encoder of the given model into the "
                "encoder of the newly created model. Some plugins may have specific configuration "
                "options allowing you to load weights from other layers. Weights will only be "
                "loaded when creating a new model. This option will be ignored if you are "
                "resuming an existing model. Generally you will also want to 'freeze-weights' "
                "whilst the rest of your model catches up with your Encoder.\n"
                "NB: Weights can only be loaded from models of the same plugin as you intend to "
                "train.")})
        argument_list.append({
            "opts": ("-t", "--trainer"),
            "action": Radio,
            "type": str.lower,
            "default": PluginLoader.get_default_model(),
            "choices": PluginLoader.get_available_models(),
            "group": _("model"),
            "help": _(
                "R|Select which trainer to use. Trainers can be configured from the Settings menu "
                "or the config folder."
                "\nL|original: The original model created by /u/deepfakes."
                "\nL|dfaker: 64px in/128px out model from dfaker. Enable 'warp-to-landmarks' for "
                "full dfaker method."
                "\nL|dfl-h128: 128px in/out model from deepfacelab"
                "\nL|dfl-sae: Adaptable model from deepfacelab"
                "\nL|dlight: A lightweight, high resolution DFaker variant."
                "\nL|iae: A model that uses intermediate layers to try to get better details"
                "\nL|lightweight: A lightweight model for low-end cards. Don't expect great "
                "results. Can train as low as 1.6GB with batch size 8."
                "\nL|realface: A high detail, dual density model based on DFaker, with "
                "customizable in/out resolution. The autoencoders are unbalanced so B>A swaps "
                "won't work so well. By andenixa et al. Very configurable."
                "\nL|unbalanced: 128px in/out model from andenixa. The autoencoders are "
                "unbalanced so B>A swaps won't work so well. Very configurable."
                "\nL|villain: 128px in/out model from villainguy. Very resource hungry (You will "
                "require a GPU with a fair amount of VRAM). Good for details, but more "
                "susceptible to color differences.")})
        argument_list.append({
            "opts": ("-u", "--summary"),
            "action": "store_true",
            "dest": "summary",
            "default": False,
            "group": _("model"),
            "help": _(
                "Output a summary of the model and exit. If a model folder is provided then a "
                "summary of the saved model is displayed. Otherwise a summary of the model that "
                "would be created by the chosen plugin and configuration settings is displayed.")})
        argument_list.append({
            "opts": ("-f", "--freeze-weights"),
            "action": "store_true",
            "dest": "freeze_weights",
            "default": False,
            "group": _("model"),
            "help": _(
                "Freeze the weights of the model. Freezing weights means that some of the "
                "parameters in the model will no longer continue to learn, but those that are not "
                "frozen will continue to learn. For most models, this will freeze the encoder, "
                "but some models may have configuration options for freezing other layers.")})
        argument_list.append({
            "opts": ("-b", "--batch-size"),
            "action": Slider,
            "min_max": (1, 256),
            "rounding": 1,
            "type": int,
            "dest": "batch_size",
            "default": 16,
            "group": _("training"),
            "help": _(
                "Batch size. This is the number of images processed through the model for each "
                "side per iteration. NB: As the model is fed 2 sides at a time, the actual number "
                "of images within the model at any one time is double the number that you set "
                "here. Larger batches require more GPU RAM.")})
        argument_list.append({
            "opts": ("-i", "--iterations"),
            "action": Slider,
            "min_max": (0, 5000000),
            "rounding": 20000,
            "type": int,
            "default": 1000000,
            "group": _("training"),
            "help": _(
                "Length of training in iterations. This is only really used for automation. There "
                "is no 'correct' number of iterations a model should be trained for. You should "
                "stop training when you are happy with the previews. However, if you want the "
                "model to stop automatically at a set number of iterations, you can set that "
                "value here.")})
        argument_list.append({
            "opts": ("-D", "--distribution-strategy"),
            "dest": "distribution_strategy",
            "action": Radio,
            "type": str.lower,
            "choices": ["default", "central-storage", "mirrored"],
            "default": "default",
            "backend": ("nvidia", "directml", "rocm", "apple_silicon"),
            "group": _("training"),
            "help": _(
                "R|Select the distribution stategy to use."
                "\nL|default: Use Tensorflow's default distribution strategy."
                "\nL|central-storage: Centralizes variables on the CPU whilst operations are "
                "performed on 1 or more local GPUs. This can help save some VRAM at the cost of "
                "some speed by not storing variables on the GPU. Note: Mixed-Precision is not "
                "supported on multi-GPU setups."
                "\nL|mirrored: Supports synchronous distributed training across multiple local "
                "GPUs. A copy of the model and all variables are loaded onto each GPU with "
                "batches distributed to each GPU at each iteration.")})
        argument_list.append({
            "opts": ("-n", "--no-logs"),
            "action": "store_true",
            "dest": "no_logs",
            "default": False,
            "group": _("training"),
            "help": _(
                "Disables TensorBoard logging. NB: Disabling logs means that you will not be able "
                "to use the graph or analysis for this session in the GUI.")})
        argument_list.append({
            "opts": ("-r", "--use-lr-finder"),
            "action": "store_true",
            "dest": "use_lr_finder",
            "default": False,
            "group": _("training"),
            "help": _(
                "Use the Learning Rate Finder to discover the optimal learning rate for training. "
                "For new models, this will calculate the optimal learning rate for the model. For "
                "existing models this will use the optimal learning rate that was discovered when "
                "initializing the model. Setting this option will ignore the manually configured "
                "learning rate (configurable in train settings).")})
        argument_list.append({
            "opts": ("-s", "--save-interval"),
            "action": Slider,
            "min_max": (10, 1000),
            "rounding": 10,
            "type": int,
            "dest": "save_interval",
            "default": 250,
            "group": _("Saving"),
            "help": _("Sets the number of iterations between each model save.")})
        argument_list.append({
            "opts": ("-I", "--snapshot-interval"),
            "action": Slider,
            "min_max": (0, 100000),
            "rounding": 5000,
            "type": int,
            "dest": "snapshot_interval",
            "default": 25000,
            "group": _("Saving"),
            "help": _(
                "Sets the number of iterations before saving a backup snapshot of the model in "
                "it's current state. Set to 0 for off.")})
        argument_list.append({
            "opts": ("-x", "--timelapse-input-A"),
            "action": DirFullPaths,
            "dest": "timelapse_input_a",
            "default": None,
            "group": _("timelapse"),
            "help": _(
                "Optional for creating a timelapse. Timelapse will save an image of your selected "
                "faces into the timelapse-output folder at every save iteration. This should be "
                "the input folder of 'A' faces that you would like to use for creating the "
                "timelapse. You must also supply a --timelapse-output and a --timelapse-input-B "
                "parameter.")})
        argument_list.append({
            "opts": ("-y", "--timelapse-input-B"),
            "action": DirFullPaths,
            "dest": "timelapse_input_b",
            "default": None,
            "group": _("timelapse"),
            "help": _(
                "Optional for creating a timelapse. Timelapse will save an image of your selected "
                "faces into the timelapse-output folder at every save iteration. This should be "
                "the input folder of 'B' faces that you would like to use for creating the "
                "timelapse. You must also supply a --timelapse-output and a --timelapse-input-A "
                "parameter.")})
        argument_list.append({
            "opts": ("-z", "--timelapse-output"),
            "action": DirFullPaths,
            "dest": "timelapse_output",
            "default": None,
            "group": _("timelapse"),
            "help": _(
                "Optional for creating a timelapse. Timelapse will save an image of your selected "
                "faces into the timelapse-output folder at every save iteration. If the input "
                "folders are supplied but no output folder, it will default to your model folder/"
                "timelapse/")})
        argument_list.append({
            "opts": ("-p", "--preview"),
            "action": "store_true",
            "dest": "preview",
            "default": False,
            "group": _("preview"),
            "help": _("Show training preview output. in a separate window.")})
        argument_list.append({
            "opts": ("-w", "--write-image"),
            "action": "store_true",
            "dest": "write_image",
            "default": False,
            "group": _("preview"),
            "help": _(
                "Writes the training result to a file. The image will be stored in the root of "
                "your FaceSwap folder.")})
        argument_list.append({
            "opts": ("-M", "--warp-to-landmarks"),
            "action": "store_true",
            "dest": "warp_to_landmarks",
            "default": False,
            "group": _("augmentation"),
            "help": _(
                "Warps training faces to closely matched Landmarks from the opposite face-set "
                "rather than randomly warping the face. This is the 'dfaker' way of doing "
                "warping.")})
        argument_list.append({
            "opts": ("-P", "--no-flip"),
            "action": "store_true",
            "dest": "no_flip",
            "default": False,
            "group": _("augmentation"),
            "help": _(
                "To effectively learn, a random set of images are flipped horizontally. Sometimes "
                "it is desirable for this not to occur. Generally this should be left off except "
                "for during 'fit training'.")})
        argument_list.append({
            "opts": ("-c", "--no-augment-color"),
            "action": "store_true",
            "dest": "no_augment_color",
            "default": False,
            "group": _("augmentation"),
            "help": _(
                "Color augmentation helps make the model less susceptible to color differences "
                "between the A and B sets, at an increased training time cost. Enable this option "
                "to disable color augmentation.")})
        argument_list.append({
            "opts": ("-W", "--no-warp"),
            "action": "store_true",
            "dest": "no_warp",
            "default": False,
            "group": _("augmentation"),
            "help": _(
                "Warping is integral to training the Neural Network. This option should only be "
                "enabled towards the very end of training to try to bring out more detail. Think "
                "of it as 'fine-tuning'. Enabling this option from the beginning is likely to "
                "kill a model and lead to terrible results.")})
        # Deprecated multi-character switches
        argument_list.append({
            "opts": ("-su", ),
            "action": "store_true",
            "dest": "depr_summary_su_u",
            "help": argparse.SUPPRESS})
        argument_list.append({
            "opts": ("-bs", ),
            "type": int,
            "dest": "depr_batch-size_bs_b",
            "help": argparse.SUPPRESS})
        argument_list.append({
            "opts": ("-it", ),
            "type": int,
            "dest": "depr_iterations_it_i",
            "help": argparse.SUPPRESS})
        argument_list.append({
            "opts": ("-nl", ),
            "action": "store_true",
            "dest": "depr_no-logs_nl_n",
            "help": argparse.SUPPRESS})
        argument_list.append({
            "opts": ("-ss", ),
            "type": int,
            "dest": "depr_snapshot-interval_ss_I",
            "help": argparse.SUPPRESS})
        argument_list.append({
            "opts": ("-tia", ),
            "type": str,
            "dest": "depr_timelapse-input-A_tia_x",
            "help": argparse.SUPPRESS})
        argument_list.append({
            "opts": ("-tib", ),
            "type": str,
            "dest": "depr_timelapse-input-B_tib_y",
            "help": argparse.SUPPRESS})
        argument_list.append({
            "opts": ("-to", ),
            "type": str,
            "dest": "depr_timelapse-output_to_z",
            "help": argparse.SUPPRESS})
        argument_list.append({
            "opts": ("-wl", ),
            "action": "store_true",
            "dest": "depr_warp-to-landmarks_wl_M",
            "help": argparse.SUPPRESS})
        argument_list.append({
            "opts": ("-nf", ),
            "action": "store_true",
            "dest": "depr_no-flip_nf_P",
            "help": argparse.SUPPRESS})
        argument_list.append({
            "opts": ("-nac", ),
            "action": "store_true",
            "dest": "depr_no-augment-color_nac_c",
            "help": argparse.SUPPRESS})
        argument_list.append({
            "opts": ("-nw", ),
            "action": "store_true",
            "dest": "depr_no-warp_nw_W",
            "help": argparse.SUPPRESS})
        return argument_list
