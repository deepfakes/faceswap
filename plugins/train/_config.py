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
ADDITIONAL_INFO = ("\nNB: Unless specifically stated, values changed here will only take effect "
                   "when creating a new model.")


class Config(FaceswapConfig):
    """ Config File for Models """
    # pylint: disable=too-many-statements
    def set_defaults(self):
        """ Set the default values for config """
        logger.debug("Setting defaults")
        # << GLOBAL OPTIONS >> #
        section = "global"
        self.add_section(title=section,
                         info="Options that apply to all models" + ADDITIONAL_INFO)
        self.add_item(
            section=section, title="icnr_init", datatype=bool, default=False,
            info="Use ICNR Kernel Initializer for upscaling.\nThis can help reduce the "
                 "'checkerboard effect' when upscaling the image.")
        self.add_item(
            section=section, title="subpixel_upscaling", datatype=bool, default=False,
            info="Use subpixel upscaling rather than pixel shuffler.\n"
                 "Might increase speed at cost of VRAM")
        self.add_item(
            section=section, title="reflect_padding", datatype=bool, default=False,
            info="Use reflect padding rather than zero padding. Only enable this option if the "
                 "model you are training has a distinct line appearing around the edge of the "
                 "swap area.")
        self.add_item(
            section=section, title="dssim_loss", datatype=bool, default=True,
            info="Use DSSIM for Loss rather than Mean Absolute Error\n"
                 "May increase overall quality.")
        self.add_item(
            section=section, title="penalized_mask_loss", datatype=bool, default=True,
            info="If using a mask, This penalizes the loss for the masked area, to give higher "
                 "priority to the face area. \nShould increase overall quality and speed up "
                 "training. This should probably be left at True")

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
            section=section, title="clipnorm", datatype=bool, default=True,
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

        # << PEGASUS MODEL OPTIONS >> #
        section = "model.realface"
        self.add_section(title=section,
                         info="An extra detailed variant of Original model.\n"
                              "Incorporates ideas from Bryanlyon and inspiration from the Villain "
                              "model.\n"
                              "Requires about 6GB-8GB of VRAM (batchsize 8-16)." + ADDITIONAL_INFO)
        self.add_item(
            section=section, title="dssim_loss", datatype=bool, default=True,
            info="Use DSSIM for Loss rather than Mean Absolute Error\n"
                 "May increase overall quality.")
        self.add_item(
            section=section, title="mask_type", datatype=str, default="components",
            choices=MASK_TYPES, info=MASK_INFO)
        self.add_item(
            section=section, title="coverage", datatype=float, default=62.5, rounding=1,
            min_max=(62.5, 100.0),
            info="{}\nThe model is essentially created for 60-80% coverage as it follows "
                 "Original paradigm.\nYou may try higher values but good results are not "
                 "guaranteed.".format(COVERAGE_INFO))
        self.add_item(
            section=section, title="input_size", datatype=int, default=64,
            rounding=16, min_max=(64, 128),
            info="Resolution (in pixels) of the input image to train on.\n"
                 "BE AWARE Larger resolution will dramatically increase"
                 "VRAM requirements.\n"
                 "Higher resolutions may increase prediction accuracy, but does not effect the "
                 "resulting output size.\n"
                 "Must be between 64 and 128 and be divisible by 16.")
        self.add_item(
            section=section, title="output_size", datatype=int, default=128,
            rounding=16, min_max=(64, 256),
            info="Output image resolution (in pixels).\n"
                 "Be aware that larger resolution will increase VRAM requirements.\n"
                 "NB: Must be between 64 and 256 and be divisible by 16.")
        self.add_item(
            section=section, title="dense_nodes", datatype=int, default=1536, rounding=64,
            min_max=(768, 2048),
            info="Number of nodes for decoder. Might affect your model's ability to learn in "
                 "general.\n"
                 "Note that: Lower values will affect the ability to predict details.")
        self.add_item(
            section=section, title="complexity_encoder", datatype=int, default=128,
            min_max=(96, 160), rounding=4,
            info="Encoder Convolution Layer Complexity. sensible ranges: "
                 "128 to 150")
        self.add_item(
            section=section, title="complexity_decoder", datatype=int, default=512,
            rounding=4, min_max=(512, 544),
            info="Decoder Complexity.")
        self.add_item(
            section=section, title="learning_rate", datatype=float, default=5e-5,
            min_max=(5e-6, 1e-4), rounding=6, fixed=False,
            info="Learning rate - how fast your network will learn.\n"
                 "Note that: Higher values might result in RSoD failure.")

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
            choices=MASK_TYPES, info=MASK_INFO)
        self.add_item(
            section=section, title="coverage", datatype=float, default=62.5, rounding=1,
            min_max=(62.5, 100.0), info=COVERAGE_INFO)
