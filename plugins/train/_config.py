#!/usr/bin/env python3
""" Default configurations for models """

from lib.config import FaceswapConfig


class Config(FaceswapConfig):
    """ Config File for Models """

    def set_defaults(self):
        """ Set the default values for config """
        # << GLOBAL OPTIONS >> #
        section = "Global"
        self.add_section(title=section,
                         info="Options that apply to all models")

        self.add_item(
            section=section, title="dssim_loss", datatype=bool, default=False,
            info="Use DSSIM for Loss rather than Mean Absolute Error\n"
                 "May increase overall quality.")

        # << ORIGINAL HIRES MODEL OPTIONS >> #
        section = "Original_HiRes"
        self.add_section(title=section,
                         info="Higher resolution version of the Original "
                              "Model")

        self.add_item(
            section=section, title="encoder_type", datatype=str,
            default="ORIGINAL",
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
            section=section, title="complexity_encoder", datatype=int,
            default=128,
            info="Encoder Convolution Layer Complexity. sensible ranges: "
                 "128 to 160")
        self.add_item(
            section=section, title="complexity_decoder_a", datatype=int,
            default=384,
            info="Decoder A Complexity. Only applicable for STANDARD and "
                 "ORIGINAL encoders")
        self.add_item(
            section=section, title="complexity_decoder_b", datatype=int,
            default=512,
            info="Decoder B Complexity. Only applicable for STANDARD and "
                 "ORIGINAL encoders")
        self.add_item(
            section=section, title="subpixel_upscaling", datatype=bool,
            default=False,
            info="Might increase upscaling quality at cost of VRAM")
