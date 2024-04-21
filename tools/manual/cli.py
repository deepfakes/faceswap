#!/usr/bin/env python3
""" The Command Line Arguments for the Manual Editor tool. """
import argparse
import gettext

from lib.cli.args import FaceSwapArgs
from lib.cli.actions import DirOrFileFullPaths, FileFullPaths

# LOCALES
_LANG = gettext.translation("tools.manual", localedir="locales", fallback=True)
_ = _LANG.gettext

_HELPTEXT = _("This command lets you perform various actions on frames, "
              "faces and alignments files using visual tools.")


class ManualArgs(FaceSwapArgs):
    """ Generate the command line options for the Manual Editor Tool."""

    @staticmethod
    def get_info():
        """ Obtain the information about what the Manual Tool does. """
        return _("A tool to perform various actions on frames, faces and alignments files using "
                 "visual tools")

    @staticmethod
    def get_argument_list():
        """ Generate the command line argument list for the Manual Tool. """
        argument_list = []
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
            "opts": ("-f", "--frames"),
            "action": DirOrFileFullPaths,
            "filetypes": "video",
            "required": True,
            "group": _("data"),
            "help": _(
                "Video file or directory containing source frames that faces were extracted "
                "from.")})
        argument_list.append({
            "opts": ("-t", "--thumb-regen"),
            "action": "store_true",
            "dest": "thumb_regen",
            "default": False,
            "group": _("options"),
            "help": _(
                "Force regeneration of the low resolution jpg thumbnails in the alignments "
                "file.")})
        argument_list.append({
            "opts": ("-s", "--single-process"),
            "action": "store_true",
            "dest": "single_process",
            "default": False,
            "group": _("options"),
            "help": _(
                "The process attempts to speed up generation of thumbnails by extracting from the "
                "video in parallel threads. For some videos, this causes the caching process to "
                "hang. If this happens, then set this option to generate the thumbnails in a "
                "slower, but more stable single thread.")})
        # Deprecated multi-character switches
        argument_list.append({
            "opts": ("-al", ),
            "type": str,
            "dest": "depr_alignments_al_a",
            "help": argparse.SUPPRESS})
        argument_list.append({
            "opts": ("-fr", ),
            "type": str,
            "dest": "depr_frames_fr_f",
            "help": argparse.SUPPRESS})
        return argument_list
