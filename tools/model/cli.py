#!/usr/bin/env python3
""" Command Line Arguments for tools """
import gettext
import typing as T

from lib.cli.args import FaceSwapArgs
from lib.cli.actions import DirFullPaths, Radio

# LOCALES
_LANG = gettext.translation("tools.restore.cli", localedir="locales", fallback=True)
_ = _LANG.gettext

_HELPTEXT = _("This tool lets you perform actions on saved Faceswap models.")


class ModelArgs(FaceSwapArgs):
    """ Class to perform actions on  model files """

    @staticmethod
    def get_info() -> str:
        """ Return command information """
        return _("A tool for performing actions on Faceswap trained model files")

    @staticmethod
    def get_argument_list() -> list[dict[str, T.Any]]:
        """ Put the arguments in a list so that they are accessible from both argparse and gui """
        argument_list = []
        argument_list.append({
            "opts": ("-m", "--model-dir"),
            "action": DirFullPaths,
            "dest": "model_dir",
            "required": True,
            "help": _(
                "Model directory. A directory containing the model you wish to perform an action "
                "on.")})
        argument_list.append({
            "opts": ("-j", "--job"),
            "action": Radio,
            "type": str,
            "choices": ("inference", "nan-scan", "restore"),
            "required": True,
            "help": _(
                "R|Choose which action you want to perform."
                "\nL|'inference' - Create an inference only copy of the model. Strips any layers "
                "from the model which are only required for training. NB: This is for exporting "
                "the model for use in external applications. Inference generated models cannot be "
                "used within Faceswap. See the 'format' option for specifying the model output "
                "format."
                "\nL|'nan-scan' - Scan the model file for NaNs or Infs (invalid data)."
                "\nL|'restore' - Restore a model from backup.")})
        argument_list.append({
            "opts": ("-f", "--format"),
            "action": Radio,
            "type": str,
            "choices": ("h5", "saved-model"),
            "default": "h5",
            "group": _("inference"),
            "help": _(
                "R|The format to save the model as. Note: Only used for 'inference' job."
                "\nL|'h5' - Standard Keras H5 format. Does not store any custom layer "
                "information. Layers will need to be loaded from Faceswap to use."
                "\nL|'saved-model' - Tensorflow's Saved Model format. Contains all information "
                "required to load the model outside of Faceswap.")})
        argument_list.append({
            "opts": ("-s", "--swap-model"),
            "action": "store_true",
            "dest": "swap_model",
            "default": False,
            "group": _("inference"),
            "help": _(
                "Only used for 'inference' job. Generate the inference model for B -> A  instead "
                "of A -> B.")})
        return argument_list
