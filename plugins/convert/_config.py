#!/usr/bin/env python3
""" Default configurations for convert """

import logging

from lib.config import FaceswapConfig

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

BLUR_TYPES = ["gaussian", "normalized", "none"]
BLUR_INFO = ("The type of blending to use:"
             "\n\t gaussian: Blend with Gaussian filter. Slower, but often better than Normalized"
             "\n\t normalized: Blend with Normalized box filter. Faster than Gaussian"
             "\n\t none: Don't perform blending")


class Config(FaceswapConfig):
    """ Config File for Convert """

    def set_defaults(self):
        """ Set the default values for config """
        logger.debug("Setting defaults")

        # << GLOBAL OPTIONS >> #
#        section = "global"
#        self.add_section(title=section,
#                         info="Options that apply to all models")

        # << BOX OPTIONS >> #
        section = "box.blend"
        self.add_section(title=section,
                         info="Options for blending the edges of the swapped box with the "
                              "background image")
        self.add_item(
            section=section, title="type", datatype=str, choices=BLUR_TYPES, default="gaussian",
            info=BLUR_INFO)
        self.add_item(
            section=section, title="blending_box", datatype=float, default=93.3, rounding=1,
            min_max=(0.1, 99.9),
            info="A box is created within the swap box from where the blending commences."
                 "\nSet the size of this box as a percentage of the swap box size."
                 "\nLower percentages starts the blending from closer to the center of the face")
        self.add_item(
            section=section, title="kernel_size", datatype=float, default=75.0, rounding=1,
            min_max=(0.1, 99.9),
            info="Kernel size dictates how much blending should occur."
                 "\nThis figure is set as a percentage of the blending box size and "
                 "should not exceed 100%."
                 "\nHigher percentage means more blending")
        self.add_item(
            section=section, title="passes", datatype=int, default=1, rounding=1,
            min_max=(1, 8),
            info="The number of passes to perform. Additional passes of the blending "
                 "algorithm can improve smoothing at a time cost."
                 "\nAdditional passes have exponentially less effect so it's not worth setting "
                 "this too high")

        section = "box.crop"
        self.add_section(title=section,
                         info="Options for cropping the swap box.\nUseful for removing unwanted "
                              "artefacts from the edge of the swap area")
        self.add_item(
            section=section, title="pixels", datatype=int, default=0, rounding=1,
            min_max=(1, 10),
            info="The number of pixels to remove from each edge of the swap box.")

        # << MASK OPTIONS >> #
        section = "mask.blend"
        self.add_section(title=section,
                         info="Options for blending the edges between the mask and the "
                              "background image")
        self.add_item(
            section=section, title="type", datatype=str, choices=BLUR_TYPES, default="normalized",
            info=BLUR_INFO)
        self.add_item(section=section, title="internal_only", datatype=bool, default=True,
                      info="Only blend sections of the mask that directly interact with the face."
                           "\nIE: If True, the forehead area will be blended, but the jawline "
                           "will not. If False, all edges of the mask will be blended")
        self.add_item(
            section=section, title="kernel_size", datatype=float, default=10.0, rounding=1,
            min_max=(0.1, 99.9),
            info="Kernel size dictates how much blending should occur."
                 "\nThis figure is set as a percentage of the mask size and "
                 "should not exceed 100%."
                 "\nHigher percentage means more blending")
        self.add_item(
            section=section, title="passes", datatype=int, default=4, rounding=1,
            min_max=(1, 8),
            info="The number of passes to perform. Additional passes of the blending "
                 "algorithm can improve smoothing at a time cost."
                 "\nAdditional passes have exponentially less effect so it's not worth setting "
                 "this too high")

        # << FACE OPTIONS >> #
        section = "face.match_histogram"
        self.add_section(title=section,
                         info="Options for matching the histograms between the source and "
                              "destination faces")
        self.add_item(
            section=section, title="threshold", datatype=int, default=98, rounding=1,
            min_max=(75, 100),
            info="Adjust the threshold for histogram matching. Can reduce extreme colors leaking "
                 "in")
