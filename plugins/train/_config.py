#!/usr/bin/env python3
""" Default configurations for models """

import logging

from lib.config import FaceswapConfig

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

MASK_TYPES = ["none", "dfaker", "dfl_full", "components"]
MASK_INFO = ("The mask to be used for training:"
             "\n\tnone: Doesn't use any mask."
             "\n\tdfaker: A basic face hull mask using a facehull of all 68 landmarks."
             "\n\tdfl_full: An improved face hull mask using a facehull of 3 facial parts"
             "\n\tcomponents: An improved face hull mask using a facehull of 8 facial parts")
COVERAGE_INFO = ("How much of the extracted image to train on. Generally the model is optimized\n"
                 "to the default value. Sensible values to use are:"
                 "\n\t62.5%% spans from eyebrow to eyebrow."
                 "\n\t75.0%% spans from temple to temple."
                 "\n\t87.5%% spans from ear to ear."
                 "\n\t100.0%% is a mugshot.")
ADDITIONAL_INFO = "\nNB: Values changed here will only take effect when creating a new model."


class Config(FaceswapConfig):
    """ Config File for Models """

    def set_defaults(self):
        """ Set the default values for config """
        logger.debug("Setting defaults")
        # << GLOBAL OPTIONS >> #
        section = "global"
        self.add_section(title=section,
                         info="Options that apply to all models" + ADDITIONAL_INFO)
        self.add_item(
            section=section, title="icnr_init", datatype=bool, default=False,
            info="\nUse ICNR to tile the default initializer in a repeating pattern. \n"
                 "This strategy is designed for  sub-pixel / pixel shuffler upscaling \n"
                 "and should only be used on upscaling layers. This can help reduce the \n"
                 "'checkerboard effect' when upscaling the image in the decoder.\n"
                 "https://arxiv.org/ftp/arxiv/papers/1707/1707.02937.pdf")
        self.add_item(
            section=section, title="subpixel_upscaling", datatype=bool, default=False,
            info="\nUse subpixel upscaling rather than pixel shuffler. These techniques \n"
                 "are both designed to produce better resolving upscaling than other \n"
                 "methods. Each perform the same operations, but using different TF opts.\n"
                 "https://arxiv.org/pdf/1609.05158.pdf")
        self.add_item(
            section=section, title="reflect_padding", datatype=bool, default=False,
            info="\nUse reflection padding rather than zero padding when either \n"
                 "downscaling or using simple convolutions. Each convolution must \n"
                 "pad the image/feature boundaries to maintain the proper sizing. \n"
                 "More complex padding schemes can reduce artifacts at the border \n"
                 "of the image.\n"
                 "http://www-cs.engr.ccny.cuny.edu/~wolberg/cs470/hw/hw2_pad.txt")
        self.add_item(
            section=section, title="image_loss_function", datatype=str,
            default="Mean_Absolute_Error",
            choices=["Mean_Absolute_Error", "Mean_Squared_Error", "LogCosh",
                     "SSIM", "GMSD", "Total_Variation", "Smooth_L1", "L_inf_norm"],
            info="\nDSSIM ---\n Use Structural Dissimilarity Index as a loss function \n"
                 "for training the neural net's image reconstruction in lieu of \n"
                 "Mean Absolute Error. Potentially better textural, second-order \n"
                 "statistics, and translation invariance than MAE.\n"
                 "http://www.cns.nyu.edu/pub/eero/wang03-reprint.pdf\n")
        self.add_item(
            section=section, title="mask_loss_function", datatype=str,
            default="Mean_Squared_Error",
            choices=["Mean_Absolute_Error", "Mean_Squared_Error", "LogCosh",
                     "SSIM", "GMSD", "Total_Variation", "Smooth_L1", "L_inf_norm"],
            info="\nDSSIM ---\n Use Structural Dissimilarity Index as a loss function \n"
                 "for training the neural net's image reconstruction in lieu of \n"
                 "Mean Absolute Error. Potentially better textural, second-order \n"
                 "statistics, and translation invariance than MAE.\n"
                 "http://www.cns.nyu.edu/pub/eero/wang03-reprint.pdf\n")
        self.add_item(
            section=section, title="mask-penalized_loss", datatype=bool, default=True,
            info="\nImage loss function is weighted by mask presence. For areas of \n"
                 "the image without the facial mask, reconstuction errors will be \n"
                 "ignored. May increase overall quality by focusing attention on \n"
                 "the core face area.")
        self.add_item(
            section=section, title="perform_augmentation", datatype=bool, default=True,
            info="\nImage augmentation is a technique that is used to artificially expand \n"
                 "image datasets. This is helpful when we are using a data-set with very \n"
                 "few images. In typical cases of Deep Learning, this situation is bad as \n"
                 "the model tends to over-fit when we train it on limited number of data \n"
                 "samples. Image augmentation parameters that are generally used to \n"
                 "increase data diversity are zoom, rotation, translation, flip, and so on.")
        '''
        self.add_item(
            section=section, title="augmentation_flipping", datatype=bool, default=True,
            info="\nTo effectively learn, a random set of images are flipped horizontally. \n"
                 "Sometimes it is desirable for this not to occur. Generally this should "
                 "be applied during all 'fit training'.")
        '''

        # << DFAKER OPTIONS >> #
        section = "model.dfaker"
        self.add_section(title=section,
                         info="Dfaker Model (Adapted from https://github.com/dfaker/df)" +
                         ADDITIONAL_INFO)
        self.add_item(
            section=section, title="mask_type", datatype=str, default="dfaker",
            choices=MASK_TYPES, info=MASK_INFO)
        self.add_item(
            section=section, title="coverage", datatype=float, default=100.0, rounding=1,
            min_max=(62.5, 100.0), info=COVERAGE_INFO)

        # << DFL MODEL OPTIONS >> #
        section = "model.dfl_h128"
        self.add_section(title=section,
                         info="DFL H128 Model (Adapted from "
                              "https://github.com/iperov/DeepFaceLab)" + ADDITIONAL_INFO)
        self.add_item(
            section=section, title="lowmem", datatype=bool, default=False,
            info="Lower memory mode. Set to 'True' if having issues with VRAM useage.\nNB: Models "
                 "with a changed lowmem mode are not compatible with each other.")
        self.add_item(
            section=section, title="mask_type", datatype=str, default="dfl_full",
            choices=MASK_TYPES, info=MASK_INFO)
        self.add_item(
            section=section, title="coverage", datatype=float, default=62.5, rounding=1,
            min_max=(62.5, 100.0), info=COVERAGE_INFO)

        # << IAE MODEL OPTIONS >> #
        section = "model.iae"
        self.add_section(title=section,
                         info="Intermediate Auto Encoder. Based on Original Model, uses "
                              "intermediate layers to try to better get details" + ADDITIONAL_INFO)
        self.add_item(
            section=section, title="mask_type", datatype=str, default="none",
            choices=MASK_TYPES, info=MASK_INFO)
        self.add_item(
            section=section, title="coverage", datatype=float, default=62.5, rounding=1,
            min_max=(62.5, 100.0), info=COVERAGE_INFO)

        # << LIGHTWEIGHT MODEL OPTIONS >> #
        section = "model.lightweight"
        self.add_section(title=section,
                         info="A lightweight version of the Original Faceswap Model, designed to "
                              "run on lower end GPUs (~2GB).\nDon't expect great results, but it "
                              "allows users with lower end cards to play with the "
                              "software." + ADDITIONAL_INFO)
        self.add_item(
            section=section, title="mask_type", datatype=str, default="none",
            choices=MASK_TYPES, info=MASK_INFO)
        self.add_item(
            section=section, title="coverage", datatype=float, default=62.5, rounding=1,
            min_max=(62.5, 100.0), info=COVERAGE_INFO)

        # << ORIGINAL MODEL OPTIONS >> #
        section = "model.original"
        self.add_section(title=section,
                         info="Original Faceswap Model" + ADDITIONAL_INFO)
        self.add_item(
            section=section, title="lowmem", datatype=bool, default=False,
            info="Lower memory mode. Set to 'True' if having issues with VRAM useage.\nNB: Models "
                 "with a changed lowmem mode are not compatible with each other.")
        self.add_item(
            section=section, title="mask_type", datatype=str, default="none",
            choices=MASK_TYPES, info=MASK_INFO)
        self.add_item(
            section=section, title="coverage", datatype=float, default=62.5, rounding=1,
            min_max=(62.5, 100.0), info=COVERAGE_INFO)

        # << UNBALANCED MODEL OPTIONS >> #
        section = "model.unbalanced"
        self.add_section(title=section,
                         info="An unbalanced model with adjustable input size options.\nThis is "
                              "an unbalanced model so b>a swaps may not work "
                              "well" + ADDITIONAL_INFO)
        self.add_item(
            section=section, title="lowmem", datatype=bool, default=False,
            info="Lower memory mode. Set to 'True' if having issues with VRAM useage.\nNB: Models "
                 "with a changed lowmem mode are not compatible with each other. NB: lowmem will "
                 "override cutom nodes and complexity settings.")
        self.add_item(
            section=section, title="clipnorm", datatype=bool, default=False,
            info="Controls gradient clipping of the optimizer. Can prevent model corruption at "
                 "the expense of VRAM")
        self.add_item(
            section=section, title="mask_type", datatype=str, default="none",
            choices=MASK_TYPES, info=MASK_INFO)
        self.add_item(
            section=section, title="nodes", datatype=int, default=1024, rounding=64,
            min_max=(512, 4096),
            info="Number of nodes for decoder. Don't change this unless you "
                 "know what you are doing!")
        self.add_item(
            section=section, title="complexity_encoder", datatype=int, default=128,
            rounding=16, min_max=(64, 1024),
            info="Encoder Convolution Layer Complexity. sensible ranges: "
                 "128 to 160")
        self.add_item(
            section=section, title="complexity_decoder_a", datatype=int, default=384,
            rounding=16, min_max=(64, 1024),
            info="Decoder A Complexity.")
        self.add_item(
            section=section, title="complexity_decoder_b", datatype=int, default=512,
            rounding=16, min_max=(64, 1024),
            info="Decoder B Complexity.")
        self.add_item(
            section=section, title="input_size", datatype=int, default=128,
            rounding=64, min_max=(64, 512),
            info="Resolution (in pixels) of the image to train on.\n"
                 "BE AWARE Larger resolution will dramatically increase"
                 "VRAM requirements.\n"
                 "Make sure your resolution is divisible by 64 (e.g. 64, 128, 256 etc.).\n"
                 "NB: Your faceset must be at least 1.6x larger than your required input size.\n"
                 "    (e.g. 160 is the maximum input size for a 256x256 faceset)")
        self.add_item(
            section=section, title="coverage", datatype=float, default=62.5, rounding=1,
            min_max=(62.5, 100.0), info=COVERAGE_INFO)

        # << VILLAIN MODEL OPTIONS >> #
        section = "model.villain"
        self.add_section(title=section,
                         info="A Higher resolution version of the Original "
                              "Model by VillainGuy.\nExtremely VRAM heavy. Full model requires "
                              "9GB+ for batchsize 16" + ADDITIONAL_INFO)
        self.add_item(
            section=section, title="lowmem", datatype=bool, default=False,
            info="Lower memory mode. Set to 'True' if having issues with VRAM useage.\nNB: Models "
                 "with a changed lowmem mode are not compatible with each other.")
        self.add_item(
            section=section, title="mask_type", datatype=str, default="none",
            choices=["none", "dfaker", "dfl_full"],
            info="The mask to be used for training. Select none to not use a mask")
        self.add_item(
            section=section, title="coverage", datatype=float, default=62.5, rounding=1,
            min_max=(62.5, 100.0), info=COVERAGE_INFO)
