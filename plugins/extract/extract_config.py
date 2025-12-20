#!/usr/bin/env python3
""" Default configurations for extract """

import gettext
import logging
import os

from lib.config import FaceswapConfig
from lib.config import ConfigItem

# LOCALES
_LANG = gettext.translation("plugins.extract.extract_config", localedir="locales", fallback=True)
_ = _LANG.gettext

logger = logging.getLogger(__name__)


class _Config(FaceswapConfig):
    """ Config File for Extraction """

    def set_defaults(self, helptext="") -> None:
        """ Set the default values for config """
        super().set_defaults(helptext=_("Options that apply to all extraction plugins"))
        self._defaults_from_plugin(os.path.dirname(__file__))


aligner_min_scale = ConfigItem(
        datatype=float,
        default=0.03,
        group=_("filters"),
        info=_(
            "Filters out faces below this size. This is a multiplier of the minimum dimension of "
            "the frame (i.e. 1280x720 = 720). If the original face extract box is smaller than "
            "the minimum dimension times this multiplier, it is considered a false positive and "
            "discarded. Faces which are found to be unusually smaller than the frame tend to be "
            "misaligned images, except in extreme long-shots. These can be usually be safely "
            "discarded."),
        min_max=(0.0, 1.0),
        rounding=2)


aligner_max_scale = ConfigItem(
        datatype=float,
        default=4.00,
        group=_("filters"),
        info=_(
            "Filters out faces above this size. This is a multiplier of the minimum dimension of "
            "the frame (i.e. 1280x720 = 720). If the original face extract box is larger than the "
            "minimum dimension times this multiplier, it is considered a false positive and "
            "discarded. Faces which are found to be unusually larger than the frame tend to be "
            "misaligned images except in extreme close-ups. These can be usually be safely "
            "discarded."),
        min_max=(0.0, 10.0),
        rounding=2)


aligner_distance = ConfigItem(
        datatype=float,
        default=40.0,
        group=_("filters"),
        info=_(
            "Filters out faces who's landmarks are above this distance from an 'average' face. "
            "Values above 15 tend to be fairly safe. Values above 10 will remove more false "
            "positives, but may also filter out some faces at extreme angles."),
        min_max=(0.0, 45.0),
        rounding=1)


aligner_roll = ConfigItem(
        datatype=float,
        default=0.0,
        group=_("filters"),
        info=_(
            "Filters out faces who's calculated roll is greater than zero +/- this value in "
            "degrees. Aligned faces should have a roll value close to zero. Values that are a "
            "significant distance from 0 degrees tend to be misaligned images. These can usually "
            "be safely disgarded."),
        min_max=(0.0, 90.0),
        rounding=1)


aligner_features = ConfigItem(
        datatype=bool,
        default=True,
        group=_("filters"),
        info=_(
            "Filters out faces where the lowest point of the aligned face's eye or eyebrow is "
            "lower than the highest point of the aligned face's mouth. Any faces where this "
            "occurs are misaligned and can be safely disgarded."))


filter_refeed = ConfigItem(
        datatype=bool,
        default=True,
        group=_("filters"),
        info=_(
            "If enabled, and 're-feed' has been selected for extraction, then interim alignments "
            "will be filtered prior to averaging the final landmarks. This can help improve the "
            "final alignments by removing any obvious misaligns from the interim results, and may "
            "also help pick up difficult alignments. If disabled, then all re-feed results will "
            "be averaged."))


save_filtered = ConfigItem(
        datatype=bool,
        default=False,
        group=_("filters"),
        info=_(
            "If enabled, saves any filtered out images into a sub-folder during the extraction "
            "process. If disabled, filtered faces are deleted. Note: The faces will always be "
            "filtered out of the alignments file, regardless of whether you keep the faces or "
            "not."))


realign_refeeds = ConfigItem(
        datatype=bool,
        default=True,
        group=_("re-align"),
        info=_(
            "If enabled, and 're-align' has been selected for extraction, then all re-feed "
            "iterations are re-aligned. If disabled, then only the final averaged output from re-"
            "feed will be re-aligned."))


filter_realign = ConfigItem(
        datatype=bool,
        default=True,
        group=_("re-align"),
        info=_(
            "If enabled, and 're-align' has been selected for extraction, then any alignments "
            "which would be filtered out will not be re-aligned."))


# pylint:disable=duplicate-code
_IS_LOADED: bool = False


def load_config(config_file: str | None = None) -> None:
    """ Load the Extraction configuration .ini file

    Parameters
    ----------
    config_file : str | None, optional
        Path to a custom .ini configuration file to load. Default: ``None`` (use default
        configuration file)
    """
    global _IS_LOADED  # pylint:disable=global-statement
    if not _IS_LOADED:
        _Config(configfile=config_file)
    _IS_LOADED = True
