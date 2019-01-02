#!/usr/bin/env python3
""" Default configurations for models """

import logging

from lib.config import FaceswapConfig

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class Config(FaceswapConfig):
    """ Config File for Models """

    def set_defaults(self):
        """ Set the default values for config """
        # << GLOBAL OPTIONS >> #
        logger.debug("Setting defaults")
        section = "global"
        self.add_section(title=section,
                         info="Options that apply to all models")

        self.add_item(
            section=section, title="dssim_loss", datatype=bool, default=False,
            info="Use DSSIM for Loss rather than Mean Absolute Error\n"
                 "May increase overall quality.")

        # << ORIGINAL MODEL OPTIONS >> #
        section = "original"
        self.add_section(title=section,
                         info="Original Faceswap Model")
        self.add_item(
            section=section, title="lowmem", datatype=bool, default=False,
            info="Lower memory mode. Set to 'True' if having issues with VRAM useage.\nNB: Models "
                 "with a changed lowmem mode are not compatible with each other."
                 "\nChoose from: True, False")

        # << ORIGINAL HIRES MODEL OPTIONS >> #
        section = "original_hires"
        self.add_section(title=section,
                         info="Higher resolution version of the Original "
                              "Model")

        self.add_item(
            section=section, title="encoder_type", datatype=str, default="ORIGINAL",
            info="Encoder type to use. Choose from:\n"
                 "ORIGINAL: Basic encoder for this model type\n"
                 "STANDARD: New, balanced encoder. More memory consuming\n"
                 "HIGHRES: High resolution tensors optimized encoder: 176x+")
        self.add_item(
            section=section, title="nodes", datatype=int, default=1024,
            info="Number of nodes for decoder. Don't change this unless you "
                 "know what you are doing!")
        self.add_item(
            section=section, title="image_size", datatype=int, default=128,
            info="Number of pixels for face width and height. Don't change "
                 "this unless you know what you are doing!")
        self.add_item(
            section=section, title="complexity_encoder", datatype=int, default=128,
            info="Encoder Convolution Layer Complexity. sensible ranges: "
                 "128 to 160")
        self.add_item(
            section=section, title="complexity_decoder_a", datatype=int, default=384,
            info="Decoder A Complexity. Only applicable for STANDARD and "
                 "ORIGINAL encoders")
        self.add_item(
            section=section, title="complexity_decoder_b", datatype=int, default=512,
            info="Decoder B Complexity. Only applicable for STANDARD and "
                 "ORIGINAL encoders")
        self.add_item(
            section=section, title="subpixel_upscaling", datatype=bool, default=False,
            info="Might increase upscaling quality at cost of VRAM")

        # << DFAKER MODEL OPTIONS >> #
        section = "dfaker"
        self.add_section(title=section,
                         info="Dfaker Model (Adapted from https://github.com/dfaker/df)")

        self.add_item(
            section=section, title="alignments_format", datatype=str, default="json",
            info="Dfaker model requires the alignments for your training "
                 "images to be avalaible within the FACES folder.\nIt should "
                 "be named 'alignments.<file extension>' (eg. "
                 "alignments.json)."
                 "\nChoose from: 'json', 'pickle' or 'yaml'")

        # << DFL MODEL OPTIONS >> #
        section = "dfl_h128"
        self.add_section(title=section,
                         info="DFL H128 Model (Adapted from https://github.com/iperov/DeepFaceLab")
        self.add_item(
            section=section, title="lowmem", datatype=bool, default=False,
            info="Lower memory mode. Set to 'True' if having issues with VRAM useage.\nNB: Models "
                 "with a changed lowmem mode are not compatible with each other."
                 "\nChoose from: True, False")

        # << GAN MODEL OPTIONS >> #
        section = "gan_v2_2"
        self.add_section(title=section,
                         info="GAN v2.2. Model (Adapted from "
                              "https://github.com/shaoanlu/faceswap-GAN)")

        # Main Options
        self.add_item(
            section=section, title="resolution", datatype=int, default=64,
            info="Resolution (in pixels) of the image to train on.\n"
                 "BE AWARE Larger resolution will dramatically increase"
                 "VRAM requirements.\nMake sure your resolution is divisible by 64 "
                 "(e.g. 64, 128, 256 etc.)")
        self.add_item(
            section=section, title="use_self_attention", datatype=bool, default=True,
            info="Use a self-attention mechanism as proposed in SAGAN("
                 "https://arxiv.org/abs/1805.08318)\n NB: There is still no official code release "
                 "for SAGAN, this implementation may be wrong."
                 "\nChoose from: 'True', 'False'")
        self.add_item(
            section=section, title="normalization", datatype=str, default="instancenorm",
            info="Normalization method.\n"
                 "\nChoose from: 'instancenorm', 'batchnorm', 'layernorm', 'groupnorm', 'none'")
        self.add_item(
            section=section, title="model_capacity", datatype=str, default="standard",
            info="Capacity of the model.\nChoose from: 'standard', 'light'")
        # Loss Config
        self.add_item(
            section=section, title="gan_training", datatype=str, default="mixup_LSGAN",
            info="GAN Training method.\nChoose from: 'mixup_LSGAN' or 'relativistic_avg_LSGAN'")
        self.add_item(
            section=section, title="use_pl", datatype=bool, default=False,
            info="Use Perceptual Loss.\nChoose from: 'True', 'False'")
        self.add_item(
            section=section, title="pl_before_activ", datatype=bool, default=False,
            info="Perceptual Loss before activation.\nChoose from: 'True', 'False'")
        self.add_item(
            section=section, title="use_mask_hinge_loss", datatype=bool, default=False,
            info="Use Mask Hinge Loss.\nChoose from: 'True', 'False'")
        self.add_item(
            section=section, title="m_mask", datatype=float, default=0.0,
            info="M Mask\nSpecify a number")
        self.add_item(
            section=section, title="lr_factor", datatype=float, default=1.0,
            info="Learning Rate Factor for Optimizer\nSpecify a number")
        self.add_item(
            section=section, title="use_cyclic_loss", datatype=bool, default=False,
            info="Use Cycle Consistency Loss.\nChoose from: 'True', 'False'")
        # Loss function weights configuration
        self.add_item(
            section=section, title="w_D", datatype=float, default=0.1,
            info="Discriminator Loss weights.\nSpecify a number")
        self.add_item(
            section=section, title="w_recon", datatype=float, default=1.0,
            info="L1 Reconstuction Loss weights.\nSpecify a number")
        self.add_item(
            section=section, title="w_edge", datatype=float, default=1.0,
            info="Edge Loss weights.\nSpecify a number")
        self.add_item(
            section=section, title="w_eyes", datatype=float, default=30.0,
            info="Reconstruction and Edge loss on eyes area.\nSpecify a number")
        self.add_item(
            section=section, title="w_pl", datatype=str, default="(0.01, 0.1, 0.3, 0.1)",
            info=("Perceptual Loss weights.\n"
                  "Specify a list of 4 numbers in brackets. eg: (0.003, 0.03, 0.3, 0.3)"))

        logger.debug("Set defaults: %s", self.defaults)
