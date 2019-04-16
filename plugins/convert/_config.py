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
            section=section, title="distance", datatype=float, default=11.0, rounding=1,
            min_max=(0.1, 25.0),
            info="The distance from the edges of the swap box to start blending. "
                 "\nThe distance is set as percentage of the swap box size to give the number of "
                 "pixels from the edge of the box. Eg: For a swap area of 256px and a percentage "
                 "of 4%, blending would commence 10 pixels from the edge."
                 "\nHigher percentages start the blending from closer to the center of the face, "
                 "so will reveal more of the source face.")
        self.add_item(
            section=section, title="radius", datatype=float, default=5.0, rounding=1,
            min_max=(0.1, 25.0),
            info="Radius dictates how much blending should occur, or more specifically, how far "
                 "the blending will spread away from the 'distance' parameter."
                 "\nThis figure is set as a percentage of the swap box size to give the radius in "
                 "pixels. Eg: For a swap area of 256px and a percentage of 5%, the radius would "
                 "be 13 pixels"
                 "\nNB: Higher percentage means more blending, but too high may reveal more of "
                 "the source face, or lead to hard lines at the border.")
        self.add_item(
            section=section, title="passes", datatype=int, default=1, rounding=1,
            min_max=(1, 8),
            info="The number of passes to perform. Additional passes of the blending "
                 "algorithm can improve smoothing at a time cost. This is more useful for 'box' "
                 "type blending."
                 "\nAdditional passes have exponentially less effect so it's not worth setting "
                 "this too high")

        # << MASK OPTIONS >> #
        section = "mask.blend"
        self.add_section(title=section,
                         info="Options for blending the edges between the mask and the "
                              "background image")
        self.add_item(
            section=section, title="type", datatype=str, choices=BLUR_TYPES, default="normalized",
            info=BLUR_INFO)
        self.add_item(
            section=section, title="radius", datatype=float, default=3.0, rounding=1,
            min_max=(0.1, 25.0),
            info="Radius dictates how much blending should occur."
                 "\nThis figure is set as a percentage of the mask diameter to give the radius in "
                 "pixels. Eg: for a mask with diameter 200px, a percentage of 6% would give a "
                 "final radius of 3px."
                 "\nHigher percentage means more blending")
        self.add_item(
            section=section, title="passes", datatype=int, default=4, rounding=1,
            min_max=(1, 8),
            info="The number of passes to perform. Additional passes of the blending "
                 "algorithm can improve smoothing at a time cost. This is more useful for 'box' "
                 "type blending."
                 "\nAdditional passes have exponentially less effect so it's not worth setting "
                 "this too high")

        # << PRE WARP OPTIONS >> #
        section = "face.match_histogram"
        self.add_section(title=section,
                         info="Options for matching the histograms between the source and "
                              "destination faces")
        self.add_item(
            section=section, title="threshold", datatype=float, default=99.0, rounding=1,
            min_max=(90.0, 100.0),
            info="Adjust the threshold for histogram matching. Can reduce extreme colors leaking "
                 "in by filtering out colors at the extreme ends of the histogram spectrum")

        # << POST WARP OPTIONS >> #
        section = "scaling.sharpen_image"
        self.add_section(title=section,
                         info="Options for sharpening the face after placement")
        self.add_item(
            section=section, title="method", datatype=str,
            choices=["none", "box", "gaussian", "unsharp_mask"], default="none",
            info="The type of sharpening to use: "
                 "\n\t box: Fastest, but weakest method. Uses a box filter to assess edges."
                 "\n\t gaussian: Slower, but better than box. Uses a gaussian filter to assess "
                 "edges."
                 "\n\t unsharp-mask: Slowest, but most tweakable. Uses the unsharp-mask method "
                 "to assess edges.")
        self.add_item(
            section=section, title="amount", datatype=int, default=150, rounding=1,
            min_max=(100, 500),
            info="Percentage that controls the magnitude of each overshoot "
                 "(how much darker and how much lighter the edge borders become)."
                 "\nThis can also be thought of as how much contrast is added at the edges. It "
                 "does not affect the width of the edge rims.")
        self.add_item(
            section=section, title="radius", datatype=float, default=0.3, rounding=1,
            min_max=(0.1, 5.0),
            info="Affects the size of the edges to be enhanced or how wide the edge rims become, "
                 "so a smaller radius enhances smaller-scale detail."
                 "\nRadius is set as a percentage of the final frame width and rounded to the "
                 "nearest pixel. E.g for a 1280 width frame, a 0.6 percenatage will give a radius "
                 "of 8px."
                 "\nHigher radius values can cause halos at the edges, a detectable faint light "
                 "rim around objects. Fine detail needs a smaller radius. "
                 "\nRadius and amount interact; reducing one allows more of the other.")
        self.add_item(
            section=section, title="threshold", datatype=float, default=5.0, rounding=1,
            min_max=(1.0, 10.0),
            info="[unsharp_mask only] Controls the minimal brightness change that will be "
                 "sharpened or how far apart adjacent tonal values have to be before the filter "
                 "does anything."
                 "\nThis lack of action is important to prevent smooth areas from becoming "
                 "speckled. The threshold setting can be used to sharpen more pronounced edges, "
                 "while leaving subtler edges untouched. "
                 "\nLow values should sharpen more because fewer areas are excluded. "
                 "\nHigher threshold values exclude areas of lower contrast.")
