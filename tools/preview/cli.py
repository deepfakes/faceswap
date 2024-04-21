#!/usr/bin/env python3
""" Command Line Arguments for tools """
import argparse
import gettext
import typing as T

from lib.cli.args import FaceSwapArgs
from lib.cli.actions import DirOrFileFullPaths, DirFullPaths, FileFullPaths

# LOCALES
_LANG = gettext.translation("tools.preview", localedir="locales", fallback=True)
_ = _LANG.gettext


_HELPTEXT = _("This command allows you to preview swaps to tweak convert settings.")


class PreviewArgs(FaceSwapArgs):
    """ Class to parse the command line arguments for Preview (Convert Settings) tool """

    @staticmethod
    def get_info() -> str:
        """ Return command information

        Returns
        -------
        str
            Top line information about the Preview tool
        """
        return _("Preview tool\nAllows you to configure your convert settings with a live preview")

    @staticmethod
    def get_argument_list() -> list[dict[str, T.Any]]:
        """ Put the arguments in a list so that they are accessible from both argparse and gui

        Returns
        -------
        list[dict[str, Any]]
            Top command line options for the preview tool
        """
        argument_list = []
        argument_list.append({
            "opts": ("-i", "--input-dir"),
            "action": DirOrFileFullPaths,
            "filetypes": "video",
            "dest": "input_dir",
            "group": _("data"),
            "required": True,
            "help": _(
                "Input directory or video. Either a directory containing the image files you wish "
                "to process or path to a video file.")})
        argument_list.append({
            "opts": ("-a", "--alignments"),
            "action": FileFullPaths,
            "filetypes": "alignments",
            "type": str,
            "group": _("data"),
            "dest": "alignments_path",
            "help": _(
                "Path to the alignments file for the input, if not at the default location")})
        argument_list.append({
            "opts": ("-m", "--model-dir"),
            "action": DirFullPaths,
            "dest": "model_dir",
            "group": _("data"),
            "required": True,
            "help": _(
                "Model directory. A directory containing the trained model you wish to process.")})
        argument_list.append({
            "opts": ("-s", "--swap-model"),
            "action": "store_true",
            "dest": "swap_model",
            "default": False,
            "help": _("Swap the model. Instead of A -> B, swap B -> A")})
        # Deprecated multi-character switches
        argument_list.append({
            "opts": ("-al", ),
            "type": str,
            "dest": "depr_alignments_al_a",
            "help": argparse.SUPPRESS})
        return argument_list
