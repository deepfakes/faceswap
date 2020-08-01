#!/usr/bin/env python3
""" A lightweight variant of DFaker Model
    By AnDenix, 2018-2019
    Based on the dfaker model: https://github.com/dfaker

    Acknowledgments:
    kvrooman for numerous insights and invaluable aid
    DeepHomage for lots of testing
    """

from keras.layers import (AveragePooling2D, BatchNormalization, Concatenate, Dense, Dropout,
                          Flatten, Input, Reshape, LeakyReLU, UpSampling2D)

from lib.model.nn_blocks import (Conv2DOutput, Conv2DBlock, ResidualBlock, UpscaleBlock,
                                 Upscale2xBlock)
from lib.utils import FaceswapError

from ._base import ModelBase, KerasModel, logger


class Model(ModelBase):
    """ DLight Autoencoder Model """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_shape = (128, 128, 3)

        self.features = dict(lowmem=0, fair=1, best=2)[self.config["features"]]
        self.encoder_filters = 64 if self.features > 0 else 48

        bonum_fortunam = 128
        self.encoder_dim = {0: 512 + bonum_fortunam,
                            1: 1024 + bonum_fortunam,
                            2: 1536 + bonum_fortunam}[self.features]
        self.details = dict(fast=0, good=1)[self.config["details"]]
        try:
            self.upscale_ratio = {128: 2,
                                  256: 4,
                                  384: 6}[self.config["output_size"]]
        except KeyError:
            logger.error("Config error: output_size must be one of: 128, 256, or 384.")
            raise FaceswapError("Config error: output_size must be one of: 128, 256, or 384.")

        logger.debug("output_size: %s, features: %s, encoder_filters: %s, encoder_dim: %s, "
                     " details: %s, upscale_ratio: %s", self.config["output_size"], self.features,
                     self.encoder_filters, self.encoder_dim, self.details, self.upscale_ratio)

    def build_model(self, inputs):
        """ Build the Dlight Model. """
        encoder = self.encoder()
        encoder_a = encoder(inputs[0])
        encoder_b = encoder(inputs[1])

        decoder_b = self.decoder_b if self.details > 0 else self.decoder_b_fast

        outputs = [self.decoder_a()(encoder_a), decoder_b()(encoder_b)]

        autoencoder = KerasModel(inputs, outputs, name=self.name)
        return autoencoder

    def encoder(self):
        """ DeLight Encoder Network """
        input_ = Input(shape=self.input_shape)
        var_x = input_

        var_x1 = Conv2DBlock(self.encoder_filters // 2)(var_x)
        var_x2 = AveragePooling2D()(var_x)
        var_x2 = LeakyReLU(0.1)(var_x2)
        var_x = Concatenate()([var_x1, var_x2])

        var_x1 = Conv2DBlock(self.encoder_filters)(var_x)
        var_x2 = AveragePooling2D()(var_x)
        var_x2 = LeakyReLU(0.1)(var_x2)
        var_x = Concatenate()([var_x1, var_x2])

        var_x1 = Conv2DBlock(self.encoder_filters * 2)(var_x)
        var_x2 = AveragePooling2D()(var_x)
        var_x2 = LeakyReLU(0.1)(var_x2)
        var_x = Concatenate()([var_x1, var_x2])

        var_x1 = Conv2DBlock(self.encoder_filters * 4)(var_x)
        var_x2 = AveragePooling2D()(var_x)
        var_x2 = LeakyReLU(0.1)(var_x2)
        var_x = Concatenate()([var_x1, var_x2])

        var_x1 = Conv2DBlock(self.encoder_filters * 8)(var_x)
        var_x2 = AveragePooling2D()(var_x)
        var_x2 = LeakyReLU(0.1)(var_x2)
        var_x = Concatenate()([var_x1, var_x2])

        var_x = Dense(self.encoder_dim)(Flatten()(var_x))
        var_x = Dropout(0.05)(var_x)
        var_x = Dense(4 * 4 * 1024)(var_x)
        var_x = Dropout(0.05)(var_x)
        var_x = Reshape((4, 4, 1024))(var_x)

        return KerasModel(input_, var_x, name="encoder")

    def decoder_a(self):
        """ DeLight Decoder A(old face) Network """
        input_ = Input(shape=(4, 4, 1024))
        decoder_a_complexity = 256
        mask_complexity = 128

        var_xy = input_
        var_xy = UpSampling2D(self.upscale_ratio, interpolation='bilinear')(var_xy)

        var_x = var_xy
        var_x = Upscale2xBlock(decoder_a_complexity, fast=False)(var_x)
        var_x = Upscale2xBlock(decoder_a_complexity // 2, fast=False)(var_x)
        var_x = Upscale2xBlock(decoder_a_complexity // 4, fast=False)(var_x)
        var_x = Upscale2xBlock(decoder_a_complexity // 8, fast=False)(var_x)

        var_x = Conv2DOutput(3, 5, name="face_out")(var_x)

        outputs = [var_x]

        if self.config.get("learn_mask", False):
            var_y = var_xy  # mask decoder
            var_y = Upscale2xBlock(mask_complexity, fast=False)(var_y)
            var_y = Upscale2xBlock(mask_complexity // 2, fast=False)(var_y)
            var_y = Upscale2xBlock(mask_complexity // 4, fast=False)(var_y)
            var_y = Upscale2xBlock(mask_complexity // 8, fast=False)(var_y)

            var_y = Conv2DOutput(1, 5, name="mask_out")(var_y)

            outputs.append(var_y)

        return KerasModel([input_], outputs=outputs, name="decoder_a")

    def decoder_b_fast(self):
        """ DeLight Fast Decoder B(new face) Network  """
        input_ = Input(shape=(4, 4, 1024))

        decoder_b_complexity = 512
        mask_complexity = 128

        var_xy = input_

        var_xy = UpscaleBlock(512, scale_factor=self.upscale_ratio)(var_xy)
        var_x = var_xy

        var_x = Upscale2xBlock(decoder_b_complexity, fast=True)(var_x)
        var_x = Upscale2xBlock(decoder_b_complexity // 2, fast=True)(var_x)
        var_x = Upscale2xBlock(decoder_b_complexity // 4, fast=True)(var_x)
        var_x = Upscale2xBlock(decoder_b_complexity // 8, fast=True)(var_x)

        var_x = Conv2DOutput(3, 5, name="face_out")(var_x)

        outputs = [var_x]

        if self.config.get("learn_mask", False):
            var_y = var_xy  # mask decoder

            var_y = Upscale2xBlock(mask_complexity, fast=False)(var_y)
            var_y = Upscale2xBlock(mask_complexity // 2, fast=False)(var_y)
            var_y = Upscale2xBlock(mask_complexity // 4, fast=False)(var_y)
            var_y = Upscale2xBlock(mask_complexity // 8, fast=False)(var_y)

            var_y = Conv2DOutput(1, 5, name="mask_out")(var_y)

            outputs.append(var_y)

        return KerasModel([input_], outputs=outputs, name="decoder_b_fast")

    def decoder_b(self):
        """ DeLight Decoder B(new face) Network  """
        input_ = Input(shape=(4, 4, 1024))

        decoder_b_complexity = 512
        mask_complexity = 128

        var_xy = input_

        var_xy = Upscale2xBlock(512, scale_factor=self.upscale_ratio, fast=False)(var_xy)

        var_x = var_xy

        var_x = ResidualBlock(512, use_bias=True)(var_x)
        var_x = ResidualBlock(512, use_bias=False)(var_x)
        var_x = ResidualBlock(512, use_bias=False)(var_x)
        var_x = Upscale2xBlock(decoder_b_complexity, fast=False)(var_x)
        var_x = ResidualBlock(decoder_b_complexity, use_bias=True)(var_x)
        var_x = ResidualBlock(decoder_b_complexity, use_bias=False)(var_x)
        var_x = BatchNormalization()(var_x)
        var_x = Upscale2xBlock(decoder_b_complexity // 2, fast=False)(var_x)
        var_x = ResidualBlock(decoder_b_complexity // 2, use_bias=True)(var_x)
        var_x = Upscale2xBlock(decoder_b_complexity // 4, fast=False)(var_x)
        var_x = ResidualBlock(decoder_b_complexity // 4, use_bias=False)(var_x)
        var_x = BatchNormalization()(var_x)
        var_x = Upscale2xBlock(decoder_b_complexity // 8, fast=False)(var_x)

        var_x = Conv2DOutput(3, 5, name="face_out")(var_x)

        outputs = [var_x]

        if self.config.get("learn_mask", False):
            var_y = var_xy  # mask decoder

            var_y = Upscale2xBlock(mask_complexity, fast=False)(var_y)
            var_y = Upscale2xBlock(mask_complexity // 2, fast=False)(var_y)
            var_y = Upscale2xBlock(mask_complexity // 4, fast=False)(var_y)
            var_y = Upscale2xBlock(mask_complexity // 8, fast=False)(var_y)

            var_y = Conv2DOutput(1, 5, name="mask_out")(var_y)

            outputs.append(var_y)

        return KerasModel([input_], outputs=outputs, name="decoder_b")

    def _legacy_mapping(self):
        """ The mapping of legacy separate model names to single model names """
        decoder_b = "decoder_b" if self.details > 0 else "decoder_b_fast"
        return {"{}_encoder.h5".format(self.name): "encoder",
                "{}_decoder_A.h5".format(self.name): "decoder_a",
                "{}_decoder_B.h5".format(self.name): decoder_b}
