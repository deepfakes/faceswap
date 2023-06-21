#!/usr/bin/env python3
""" Default configurations for extract """

import gettext
import logging
import os

from lib.config import FaceswapConfig

# LOCALES
_LANG = gettext.translation("plugins.extract._config", localedir="locales", fallback=True)
_ = _LANG.gettext

logger = logging.getLogger(__name__)


class Config(FaceswapConfig):
    """ Config File for Extraction """

    def set_defaults(self) -> None:
        """ Set the default values for config """
        logger.debug("Setting defaults")
        self.set_globals()
        self._defaults_from_plugin(os.path.dirname(__file__))

    def set_globals(self) -> None:
        """
        Set the global options for extract
        """
        logger.debug("Setting global config")
        section = "global"
        self.add_section(section, _("Options that apply to all extraction plugins"))
        self.add_item(
            section=section,
            title="allow_growth",
            datatype=bool,
            default=False,
            group=_("settings"),
            info=_("Enable the Tensorflow GPU `allow_growth` configuration option. "
                   "This option prevents Tensorflow from allocating all of the GPU VRAM at launch "
                   "but can lead to higher VRAM fragmentation and slower performance. Should only "
                   "be enabled if you are having problems running extraction."))
        self.add_item(
            section=section,
            title="aligner_min_scale",
            datatype=float,
            min_max=(0.0, 1.0),
            rounding=2,
            default=0.07,
            group=_("filters"),
            info=_("Filters out faces below this size. This is a multiplier of the minimum "
                   "dimension of the frame (i.e. 1280x720 = 720). If the original face extract "
                   "box is smaller than the minimum dimension times this multiplier, it is "
                   "considered a false positive and discarded. Faces which are found to be "
                   "unusually smaller than the frame tend to be misaligned images, except in "
                   "extreme long-shots. These can be usually be safely discarded."))
        self.add_item(
            section=section,
            title="aligner_max_scale",
            datatype=float,
            min_max=(0.0, 10.0),
            rounding=2,
            default=2.00,
            group=_("filters"),
            info=_("Filters out faces above this size. This is a multiplier of the minimum "
                   "dimension of the frame (i.e. 1280x720 = 720). If the original face extract "
                   "box is larger than the minimum dimension times this multiplier, it is "
                   "considered a false positive and discarded. Faces which are found to be "
                   "unusually larger than the frame tend to be misaligned images except in "
                   "extreme close-ups. These can be usually be safely discarded."))
        self.add_item(
            section=section,
            title="aligner_distance",
            datatype=float,
            min_max=(0.0, 45.0),
            rounding=1,
            default=22.5,
            group=_("filters"),
            info=_("Filters out faces who's landmarks are above this distance from an 'average' "
                   "face. Values above 15 tend to be fairly safe. Values above 10 will remove "
                   "more false positives, but may also filter out some faces at extreme angles."))
        self.add_item(
            section=section,
            title="aligner_roll",
            datatype=float,
            min_max=(0.0, 90.0),
            rounding=1,
            default=45.0,
            group=_("filters"),
            info=_("Filters out faces who's calculated roll is greater than zero +/- this value "
                   "in degrees. Aligned faces should have a roll value close to zero. Values that "
                   "are a significant distance from 0 degrees tend to be misaligned images. These "
                   "can usually be safely disgarded."))
        self.add_item(
            section=section,
            title="aligner_features",
            datatype=bool,
            default=True,
            group=_("filters"),
            info=_("Filters out faces where the lowest point of the aligned face's eye or eyebrow "
                   "is lower than the highest point of the aligned face's mouth. Any faces where "
                   "this occurs are misaligned and can be safely disgarded."))
        self.add_item(
            section=section,
            title="filter_refeed",
            datatype=bool,
            default=True,
            group=_("filters"),
            info=_("If enabled, and 're-feed' has been selected for extraction, then interim "
                   "alignments will be filtered prior to averaging the final landmarks. This can "
                   "help improve the final alignments by removing any obvious misaligns from the "
                   "interim results, and may also help pick up difficult alignments. If disabled, "
                   "then all re-feed results will be averaged."))
        self.add_item(
            section=section,
            title="save_filtered",
            datatype=bool,
            default=False,
            group=_("filters"),
            info=_("If enabled, saves any filtered out images into a sub-folder during the "
                   "extraction process. If disabled, filtered faces are deleted. Note: The faces "
                   "will always be filtered out of the alignments file, regardless of whether you "
                   "keep the faces or not."))
        self.add_item(
            section=section,
            title="realign_refeeds",
            datatype=bool,
            default=True,
            group=_("re-align"),
            info=_("If enabled, and 're-align' has been selected for extraction, then all re-feed "
                   "iterations are re-aligned. If disabled, then only the final averaged output "
                   "from re-feed will be re-aligned."))
        self.add_item(
            section=section,
            title="filter_realign",
            datatype=bool,
            default=True,
            group=_("re-align"),
            info=_("If enabled, and 're-align' has been selected for extraction, then any "
                   "alignments which would be filtered out will not be re-aligned."))
