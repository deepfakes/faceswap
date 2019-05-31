#!/usr/bin/env python3
""" Default configurations for convert """

import logging

from lib.config import FaceswapConfig
from lib.utils import _video_extensions

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

        # << MASK OPTIONS >> #
        section = "mask.box_blend"
        self.add_section(title=section,
                         info="Options for blending the edges of the swapped box with the "
                              "background image")
        self.add_item(
            section=section, title="type", datatype=str, choices=BLUR_TYPES, default="gaussian",
            info=BLUR_INFO, gui_radio=True)
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

        section = "mask.mask_blend"
        self.add_section(title=section,
                         info="Options for blending the edges between the mask and the "
                              "background image")
        self.add_item(
            section=section, title="type", datatype=str, choices=BLUR_TYPES, default="normalized",
            info=BLUR_INFO, gui_radio=True)
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
        self.add_item(
            section=section, title="erosion", datatype=float, default=0.0, rounding=1,
            min_max=(-100.0, 100.0),
            info="Erosion kernel size as a percentage of the mask radius area.\n"
                 "Positive values apply erosion which reduces the size of the swapped area.\n"
                 "Negative values apply dilation which increases the swapped area")

        # <<<<<< COLOUR  OPTIONS >>>>>> #
        section = "color.color_transfer"
        self.add_section(title=section,
                         info="Options for transfering the color distribution from the source to "
                              "the target image using the mean and standard deviations of the "
                              "L*a*b* color space.\n"
                              "This implementation is (loosely) based on to the 'Color Transfer "
                              "between Images' paper by Reinhard et al., 2001. matching the "
                              "histograms between the source and destination faces.")
        self.add_item(
            section=section, title="clip", datatype=bool, default=True,
            info="Should components of L*a*b* image be scaled by np.clip before converting back "
                 "to BGR color space?\n"
                 "If False then components will be min-max scaled appropriately.\n"
                 "Clipping will keep target image brightness truer to the input.\n"
                 "Scaling will adjust image brightness to avoid washed out portions in the "
                 "resulting color transfer that can be caused by clipping.")
        self.add_item(
            section=section, title="preserve_paper", datatype=bool, default=True,
            info="Should color transfer strictly follow methodology layed out in original paper?\n"
                 "The method does not always produce aesthetically pleasing results.\n"
                 "If False then L*a*b* components will be scaled using the reciprocal of the "
                 "scaling factor proposed in the paper. This method seems to produce more "
                 "consistently aesthetically pleasing results")

        section = "color.manual_balance"
        self.add_section(title=section,
                         info="Options for manually altering the balance of colors of the swapped "
                              "face")
        self.add_item(
            section=section, title="colorspace", datatype=str, default="HSV", gui_radio=True,
            choices=["RGB", "HSV", "LAB", "YCrCb"],
            info="The colorspace to use for adjustment: The three adjustment sliders will effect "
                 "the image differently depending on which colorspace is selected:"
                 "\n\t RGB: Red, Green, Blue. An additive colorspace where colors are obtained by "
                 "a linear combination of Red, Green, and Blue values. The three channels are "
                 "correlated by the amount of light hitting the surface. In RGB color space the "
                 "color information is separated into three channels but the same three channels "
                 "also encode brightness information."
                 "\n\t HSV: Hue, Saturation, Value. Hue - Dominant wavelength. Saturation - "
                 "Purity / shades of color. Value - Intensity. Best thing is that it uses only "
                 "one channel to describe color (H), making it very intuitive to specify color."
                 "\n\t LAB: Lightness, A, B. Lightness - Intensity. A - Color range from green to "
                 "magenta. B - Color range from blue to yellow. The L channel is independent of "
                 "color information and encodes brightness only. The other two channels encode "
                 "color."
                 "\n\t YCrCb: Y – Luminance or Luma component obtained from RGB after gamma "
                 "correction. Cr - how far is the red component from Luma. Cb - how far is the "
                 "blue component from Luma. Separates the luminance and chrominance components "
                 "into different channels.")
        self.add_item(
            section=section, title="balance_1", datatype=float, default=0.0, rounding=1,
            min_max=(-100.0, 100.0),
            info="Balance of channel 1: "
                 "\n\tRGB: Red "
                 "\n\tHSV: Hue "
                 "\n\tLAB: Lightness "
                 "\n\tYCrCb: Luma ")
        self.add_item(
            section=section, title="balance_2", datatype=float, default=0.0, rounding=1,
            min_max=(-100.0, 100.0),
            info="Balance of channel 2: "
                 "\n\tRGB: Green "
                 "\n\tHSV: Saturation "
                 "\n\tLAB: Green > Magenta "
                 "\n\tYCrCb: Distance of red from Luma")
        self.add_item(
            section=section, title="balance_3", datatype=float, default=0.0, rounding=1,
            min_max=(-100.0, 100.0),
            info="Balance of channel 3: "
                 "\n\tRGB: Blue "
                 "\n\tHSV: Intensity "
                 "\n\tLAB: Blue > Yellow "
                 "\n\tYCrCb: Distance of blue from Luma")
        section = "color.match_hist"
        self.add_section(title=section,
                         info="Options for matching the histograms between the source and "
                              "destination faces")
        self.add_item(
            section=section, title="threshold", datatype=float, default=99.0, rounding=1,
            min_max=(90.0, 100.0),
            info="Adjust the threshold for histogram matching. Can reduce extreme colors leaking "
                 "in by filtering out colors at the extreme ends of the histogram spectrum")

        # <<<<<< SCALING  OPTIONS >>>>>> #
        section = "scaling.sharpen"
        self.add_section(title=section,
                         info="Options for sharpening the face after placement")
        self.add_item(
            section=section, title="method", datatype=str,
            choices=["box", "gaussian", "unsharp_mask"], default="unsharp_mask",
            gui_radio=True,
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

        # <<<<<< OUTPUT  OPTIONS >>>>>> #
        section = "writer.gif"
        self.add_section(title=section,
                         info="Options for outputting converted frames to an animated gif.")
        self.add_item(
            section=section, title="fps", datatype=int, min_max=(1, 60),
            rounding=1, default=25,
            info="Frames per Second.")
        self.add_item(
            section=section, title="loop", datatype=int, min_max=(0, 100),
            rounding=1, default=0,
            info="The number of iterations. Set to 0 to loop indefinitely.")
        self.add_item(
            section=section, title="palettesize", datatype=str, default="256",
            choices=["2", "4", "8", "16", "32", "64", "128", "256"],
            info="The number of colors to quantize the image to. Is rounded to the nearest power "
                 "of two.")
        self.add_item(
            section=section, title="subrectangles", datatype=bool, default=False,
            info="If True, will try and optimize the GIF by storing only the rectangular parts of "
                 "each frame that change with respect to the previous.")

        section = "writer.opencv"
        self.add_section(title=section,
                         info="Options for outputting converted frames to a series of images "
                              "using OpenCV\n"
                              "OpenCV can be faster than other image writers, but lacks some of "
                              " configuration options and formats.")
        self.add_item(
            section=section, title="format", datatype=str, default="png",
            choices=["bmp", "jpg", "jp2", "png", "ppm"],
            gui_radio=True,
            info="Image format to use:"
                 "\n\t bmp: Windows bitmap"
                 "\n\t jpg: JPEG format"
                 "\n\t jp2: JPEG 2000 format"
                 "\n\t png: Portable Network Graphics"
                 "\n\t ppm: Portable Pixmap Format")
        self.add_item(
            section=section, title="draw_transparent", datatype=bool, default=False,
            info="Place the swapped face on a transparent layer rather than the original frame.\n"
                 "NB: This is only compatible with images saved in png format. If an "
                 "incompatible format is selected then the image will be saved as a png.")
        self.add_item(
            section=section, title="jpg_quality", datatype=int, min_max=(1, 95),
            rounding=1, default=75,
            info="[jpg only] Set the jpg quality. 1 is worst 95 is best. Higher quality leads to "
                 "larger file sizes.")
        self.add_item(
            section=section, title="png_compress_level", datatype=int, min_max=(0, 9),
            rounding=1, default=3,
            info="[png only] ZLIB compression level, 1 gives best speed, 9 gives best "
                 "compression, 0 gives no compression at all.")

        section = "writer.pillow"
        self.add_section(title=section,
                         info="Options for outputting converted frames to a series of images "
                              "using Pillow\n"
                              "Pillow is more feature rich than OpenCV but can be slower.")
        self.add_item(
            section=section, title="format", datatype=str, default="png",
            choices=["bmp", "gif", "jpg", "jp2", "png", "ppm", "tif"],
            gui_radio=True,
            info="Image format to use:"
                 "\n\t bmp: Windows bitmap"
                 "\n\t gif: Graphics Interchange Format (NB: Not animated)"
                 "\n\t jpg: JPEG format"
                 "\n\t jp2: JPEG 2000 format"
                 "\n\t png: Portable Network Graphics"
                 "\n\t ppm: Portable Pixmap Format"
                 "\n\t tif: Tag Image File Format")
        self.add_item(
            section=section, title="draw_transparent", datatype=bool, default=False,
            info="Place the swapped face on a transparent layer rather than the original frame.\n"
                 "NB: This is only compatible with images saved in png or tif format. If an "
                 "incompatible format is selected then the image will be saved as a png.")
        self.add_item(
            section=section, title="optimize", datatype=bool, default=False,
            info="[gif, jpg and png only] If enabled, indicates that the encoder should make an "
                 "extra pass over the image in order to select optimal encoder settings.")
        self.add_item(
            section=section, title="gif_interlace", datatype=bool, default=True,
            info="[gif only] Set whether to save the gif as interlaced or not.")
        self.add_item(
            section=section, title="jpg_quality", datatype=int, min_max=(1, 95),
            rounding=1, default=75,
            info="[jpg only] Set the jpg quality. 1 is worst 95 is best. Higher quality leads to "
                 "larger file sizes.")
        self.add_item(
            section=section, title="png_compress_level", datatype=int, min_max=(0, 9),
            rounding=1, default=3,
            info="[png only] ZLIB compression level, 1 gives best speed, 9 gives best "
                 "compression, 0 gives no compression at all. When optimize option is set to True "
                 "this has no effect (it is set to 9 regardless of a value passed).")
        self.add_item(
            section=section, title="tif_compression", datatype=str, default="tiff_deflate",
            choices=["none", "tiff_ccitt", "group3", "group4", "tiff_jpeg", "tiff_adobe_deflate",
                     "tiff_thunderscan", "tiff_deflate", "tiff_sgilog", "tiff_sgilog24",
                     "tiff_raw_16"],
            info="[tif only] The desired compression method for the file.")

        section = "writer.ffmpeg"
        self.add_section(title=section,
                         info="Options for encoding converted frames to video.")
        self.add_item(
            section=section, title="container", datatype=str, default="mp4",
            choices=[ext.replace(".", "") for ext in _video_extensions],
            gui_radio=True,
            info="Video container to use.")
        self.add_item(
            section=section, title="codec", datatype=str,
            choices=["libx264", "libx265"], default="libx264",
            gui_radio=True,
            info="Video codec to use:"
                 "\n\t libx264: H.264. A widely supported and commonly used codec."
                 "\n\t libx265: H.265 / HEVC video encoder application library.")
        self.add_item(
            section=section, title="crf", datatype=int, min_max=(0, 51), rounding=1, default=23,
            info="Constant Rate Factor:  0 is lossless and 51 is worst quality possible. A lower "
                 "value generally leads to higher quality, and a subjectively sane range is "
                 "17-28. Consider 17 or 18 to be visually lossless or nearly so; it should look "
                 "the same or nearly the same as the input but it isn't technically lossless.\n"
                 "The range is exponential, so increasing the CRF value +6 results in roughly "
                 "half the bitrate / file size, while -6 leads to roughly twice the bitrate.\n"
                 "Choose the highest CRF value that still provides an acceptable quality. If the "
                 "output looks good, then try a higher value. If it looks bad, choose a lower "
                 "value.")
        self.add_item(
            section=section, title="preset", datatype=str, default="medium",
            choices=["ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow",
                     "slower", "veryslow"],
            gui_radio=True,
            info="A preset is a collection of options that will provide a certain encoding speed "
                 "to compression ratio.\nA slower preset will provide better compression "
                 "(compression is quality per filesize).\nUse the slowest preset that you have "
                 "patience for")
        self.add_item(
            section=section, title="tune", datatype=str, default="none",
            choices=["none", "film", "animation", "grain", "stillimage", "fastdecode",
                     "zerolatency"],
            info="Change settings based upon the specifics of your input:"
                 "\n\t none: Don't perform any additional tuning."
                 "\n\t film: [H.264 only] Use for high quality movie content; lowers deblocking."
                 "\n\t animation: [H.264 only] Good for cartoons; uses higher deblocking and more "
                 "reference frames."
                 "\n\t grain: Preserves the grain structure in old, grainy film material."
                 "\n\t stillimage: [H.264 only] Good for slideshow-like content."
                 "\n\t fastdecode: Allows faster decoding by disabling certain filters."
                 "\n\t zerolatency: Good for fast encoding and low-latency streaming.")
        self.add_item(
            section=section, title="profile", datatype=str, default="auto",
            choices=["auto", "baseline", "main", "high", "high10", "high422", "high444"],
            info="[H.264 Only] Limit the output to a specific H.264 profile. Don't change this "
                 "unless your target device only supports a certain profile.")
        self.add_item(
            section=section, title="level", datatype=str, default="auto",
            choices=["auto", "1", "1b", "1.1", "1.2", "1.3", "2", "2.1", "2.2", "3", "3.1", "3.2",
                     "4", "4.1", "4.2", "5", "5.1", "5.2", "6", "6.1", "6.2"],
            info="[H.264 Only] Set the encoder level, Don't change this unless your target device "
                 "only supports a certain level.")
