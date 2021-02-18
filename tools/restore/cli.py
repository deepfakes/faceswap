#!/usr/bin/env python3
""" Command Line Arguments for tools """
import gettext

from lib.cli.args import FaceSwapArgs
from lib.cli.actions import DirFullPaths


# LOCALES
_LANG = gettext.translation("tools.restore.cli", localedir="locales", fallback=True)
_ = _LANG.gettext

_HELPTEXT = _("This command lets you restore models from backup.")


class RestoreArgs(FaceSwapArgs):
    """ Class to restore model files from backup """

    @staticmethod
    def get_info():
        """ Return command information """
        return _("A tool for restoring models from backup (.bk) files")

    @staticmethod
    def get_argument_list():
        """ Put the arguments in a list so that they are accessible from both argparse and gui """
        argument_list = list()
        argument_list.append(dict(
            opts=("-m", "--model-dir"),
            action=DirFullPaths,
            dest="model_dir",
            required=True,
            help=_("Model directory. A directory containing the model you wish to restore from "
                   "backup.")))
        return argument_list
