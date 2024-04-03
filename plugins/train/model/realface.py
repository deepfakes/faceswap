#!/usr/bin/env python3
""" RealFaceRC1, codenamed 'Pegasus'
    Based on the original https://www.reddit.com/r/deepfakes/
    code sample + contributions
    Major thanks goes to BryanLyon as it vastly powered by his ideas and insights.
    Without him it would not be possible to come up with the model.
    Additional thanks: Birb - source of inspiration, great Encoder ideas
                       Kvrooman - additional counseling on auto-encoders and practical advice
    """
import logging
import sys

# Ignore linting errors from Tensorflow's thoroughly broken import system
from tensorflow.keras.initializers import RandomNormal  # pylint:disable=import-error
from tensorflow.keras.layers import Dense, Flatten, Input, LeakyReLU, Reshape  # noqa:E501  # pylint:disable=import-error
from tensorflow.keras.models import Model as KModel  # pylint:disable=import-error

from lib.model.nn_blocks import Conv2DOutput, Conv2DBlock, ResidualBlock, UpscaleBlock
from ._base import ModelBase

logger = logging.getLogger(__name__)


class Model(ModelBase):
    """ RealFace(tm) Faceswap Model """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_shape = (self.config["input_size"], self.config["input_size"], 3)
        self.check_input_output()
        self.dense_width, self.upscalers_no = self.get_dense_width_upscalers_numbers()
        self.kernel_initializer = RandomNormal(0, 0.02)

    @property
    def downscalers_no(self):
        """ Number of downscale blocks. Don't change! """
        return 4

    @property
    def _downscale_ratio(self):
        """ Downscale Ratio """
        return 2**self.downscalers_no

    @property
    def dense_filters(self):
        """ Dense Filters. Don't change! """
        return (int(1024 - (self.dense_width - 4) * 64) // 16) * 16

    def check_input_output(self):
        """ Confirm valid input and output sized have been provided """
        if not 64 <= self.config["input_size"] <= 128 or self.config["input_size"] % 16 != 0:
            logger.error("Config error: input_size must be between 64 and 128 and be divisible by "
                         "16.")
            sys.exit(1)
        if not 64 <= self.config["output_size"] <= 256 or self.config["output_size"] % 32 != 0:
            logger.error("Config error: output_size must be between 64 and 256 and be divisible "
                         "by 32.")
            sys.exit(1)
        logger.debug("Input and output sizes are valid")

    def get_dense_width_upscalers_numbers(self):
        """ Return the dense width and number of upscale blocks """
        output_size = self.config["output_size"]
        sides = [(output_size // 2**n, n) for n in [4, 5] if (output_size // 2**n) < 10]
        closest = min([x * self._downscale_ratio for x, _ in sides],
                      key=lambda x: abs(x - self.config["input_size"]))
        dense_width, upscalers_no = [(s, n) for s, n in sides
                                     if s * self._downscale_ratio == closest][0]
        logger.debug("dense_width: %s, upscalers_no: %s", dense_width, upscalers_no)
        return dense_width, upscalers_no

    def build_model(self, inputs):
        """ Build the RealFace model. """
        encoder = self.encoder()
        encoder_a = encoder(inputs[0])
        encoder_b = encoder(inputs[1])

        outputs = [self.decoder_a()(encoder_a), self.decoder_b()(encoder_b)]

        autoencoder = KModel(inputs, outputs, name=self.model_name)
        return autoencoder

    def encoder(self):
        """ RealFace Encoder Network """
        input_ = Input(shape=self.input_shape)
        var_x = input_

        encoder_complexity = self.config["complexity_encoder"]

        for idx in range(self.downscalers_no - 1):
            var_x = Conv2DBlock(encoder_complexity * 2**idx, activation=None)(var_x)
            var_x = LeakyReLU(alpha=0.2)(var_x)
            var_x = ResidualBlock(encoder_complexity * 2**idx, use_bias=True)(var_x)
            var_x = ResidualBlock(encoder_complexity * 2**idx, use_bias=True)(var_x)

        var_x = Conv2DBlock(encoder_complexity * 2**(idx + 1), activation="leakyrelu")(var_x)

        return KModel(input_, var_x, name="encoder")

    def decoder_b(self):
        """ RealFace Decoder Network """
        input_filters = self.config["complexity_encoder"] * 2**(self.downscalers_no-1)
        input_width = self.config["input_size"] // self._downscale_ratio
        input_ = Input(shape=(input_width, input_width, input_filters))

        var_xy = input_

        var_xy = Dense(self.config["dense_nodes"])(Flatten()(var_xy))
        var_xy = Dense(self.dense_width * self.dense_width * self.dense_filters)(var_xy)
        var_xy = Reshape((self.dense_width, self.dense_width, self.dense_filters))(var_xy)
        var_xy = UpscaleBlock(self.dense_filters, activation=None)(var_xy)

        var_x = var_xy
        var_x = LeakyReLU(alpha=0.2)(var_x)
        var_x = ResidualBlock(self.dense_filters, use_bias=False)(var_x)

        decoder_b_complexity = self.config["complexity_decoder"]
        for idx in range(self.upscalers_no - 2):
            var_x = UpscaleBlock(decoder_b_complexity // 2**idx, activation=None)(var_x)
            var_x = LeakyReLU(alpha=0.2)(var_x)
            var_x = ResidualBlock(decoder_b_complexity // 2**idx, use_bias=False)(var_x)
            var_x = ResidualBlock(decoder_b_complexity // 2**idx, use_bias=True)(var_x)
        var_x = UpscaleBlock(decoder_b_complexity // 2**(idx + 1), activation="leakyrelu")(var_x)

        var_x = Conv2DOutput(3, 5, name="face_out_b")(var_x)

        outputs = [var_x]

        if self.config.get("learn_mask", False):
            var_y = var_xy
            var_y = LeakyReLU(alpha=0.1)(var_y)

            mask_b_complexity = 384
            for idx in range(self.upscalers_no-2):
                var_y = UpscaleBlock(mask_b_complexity // 2**idx, activation="leakyrelu")(var_y)
            var_y = UpscaleBlock(mask_b_complexity // 2**(idx + 1), activation="leakyrelu")(var_y)

            var_y = Conv2DOutput(1, 5, name="mask_out_b")(var_y)

            outputs += [var_y]

        return KModel(input_, outputs=outputs, name="decoder_b")

    def decoder_a(self):
        """ RealFace Decoder (A) Network """
        input_filters = self.config["complexity_encoder"] * 2**(self.downscalers_no-1)
        input_width = self.config["input_size"] // self._downscale_ratio
        input_ = Input(shape=(input_width, input_width, input_filters))

        var_xy = input_

        dense_nodes = int(self.config["dense_nodes"]/1.5)
        dense_filters = int(self.dense_filters/1.5)

        var_xy = Dense(dense_nodes)(Flatten()(var_xy))
        var_xy = Dense(self.dense_width * self.dense_width * dense_filters)(var_xy)
        var_xy = Reshape((self.dense_width, self.dense_width, dense_filters))(var_xy)

        var_xy = UpscaleBlock(dense_filters, activation=None)(var_xy)

        var_x = var_xy
        var_x = LeakyReLU(alpha=0.2)(var_x)
        var_x = ResidualBlock(dense_filters, use_bias=False)(var_x)

        decoder_a_complexity = int(self.config["complexity_decoder"] / 1.5)
        for idx in range(self.upscalers_no-2):
            var_x = UpscaleBlock(decoder_a_complexity // 2**idx, activation="leakyrelu")(var_x)
        var_x = UpscaleBlock(decoder_a_complexity // 2**(idx + 1), activation="leakyrelu")(var_x)

        var_x = Conv2DOutput(3, 5, name="face_out_a")(var_x)

        outputs = [var_x]

        if self.config.get("learn_mask", False):
            var_y = var_xy
            var_y = LeakyReLU(alpha=0.1)(var_y)

            mask_a_complexity = 384
            for idx in range(self.upscalers_no-2):
                var_y = UpscaleBlock(mask_a_complexity // 2**idx, activation="leakyrelu")(var_y)
            var_y = UpscaleBlock(mask_a_complexity // 2**(idx + 1), activation="leakyrelu")(var_y)

            var_y = Conv2DOutput(1, 5, name="mask_out_a")(var_y)

            outputs += [var_y]

        return KModel(input_, outputs=outputs, name="decoder_a")

    def _legacy_mapping(self):
        """ The mapping of legacy separate model names to single model names """
        return {f"{self.name}_encoder.h5": "encoder",
                f"{self.name}_decoder_A.h5": "decoder_a",
                f"{self.name}_decoder_B.h5": "decoder_b"}
