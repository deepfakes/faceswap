#!/usr/bin/env python3
""" A lightweight variant of DFaker Model
    By AnDenix, 2018-2019
    Based on the dfaker model: https://github.com/dfaker

    Acknowledgments:
    kvrooman for numerous insights and invaluable aid
    DeepHomage for lots of testing
    """
import logging

from keras import layers, Input, Model as KModel

from lib.model.nn_blocks import (Conv2DOutput, Conv2DBlock, ResidualBlock, UpscaleBlock,
                                 Upscale2xBlock)
from lib.utils import FaceswapError
from plugins.train.train_config import Loss as cfg_loss

from ._base import ModelBase
from . import dlight_defaults as cfg


logger = logging.getLogger(__name__)


class Model(ModelBase):
    """ DLight Autoencoder Model """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_shape = (128, 128, 3)

        self.features = {"lowmem": 0, "fair": 1, "best": 2}[cfg.features()]
        self.encoder_filters = 64 if self.features > 0 else 48

        bonum_fortunam = 128
        self.encoder_dim = {0: 512 + bonum_fortunam,
                            1: 1024 + bonum_fortunam,
                            2: 1536 + bonum_fortunam}[self.features]
        self.details = {"fast": 0, "good": 1}[cfg.details()]
        try:
            self.upscale_ratio = {128: 2,
                                  256: 4,
                                  384: 6}[cfg.output_size()]
        except KeyError as err:
            logger.error("Config error: output_size must be one of: 128, 256, or 384.")
            raise FaceswapError("Config error: output_size must be one of: "
                                "128, 256, or 384.") from err

        logger.debug("output_size: %s, features: %s, encoder_filters: %s, encoder_dim: %s, "
                     " details: %s, upscale_ratio: %s", cfg.output_size(), self.features,
                     self.encoder_filters, self.encoder_dim, self.details, self.upscale_ratio)

    def build_model(self, inputs):
        """ Build the Dlight Model. """
        encoder = self.encoder()
        encoder_a = encoder(inputs[0])
        encoder_b = encoder(inputs[1])

        decoder_b = self.decoder_b if self.details > 0 else self.decoder_b_fast

        outputs = self.decoder_a()(encoder_a) + decoder_b()(encoder_b)

        autoencoder = KModel(inputs, outputs, name=self.model_name)
        return autoencoder

    def encoder(self):
        """ DeLight Encoder Network """
        input_ = Input(shape=self.input_shape)
        var_x = input_

        var_x1 = Conv2DBlock(self.encoder_filters // 2, activation="leakyrelu")(var_x)
        var_x2 = layers.AveragePooling2D(pool_size=(2, 2))(var_x)
        var_x2 = layers.LeakyReLU(0.1)(var_x2)
        var_x = layers.Concatenate()([var_x1, var_x2])

        var_x1 = Conv2DBlock(self.encoder_filters, activation="leakyrelu")(var_x)
        var_x2 = layers.AveragePooling2D(pool_size=(2, 2))(var_x)
        var_x2 = layers.LeakyReLU(0.1)(var_x2)
        var_x = layers.Concatenate()([var_x1, var_x2])

        var_x1 = Conv2DBlock(self.encoder_filters * 2, activation="leakyrelu")(var_x)
        var_x2 = layers.AveragePooling2D(pool_size=(2, 2))(var_x)
        var_x2 = layers.LeakyReLU(0.1)(var_x2)
        var_x = layers.Concatenate()([var_x1, var_x2])

        var_x1 = Conv2DBlock(self.encoder_filters * 4, activation="leakyrelu")(var_x)
        var_x2 = layers.AveragePooling2D(pool_size=(2, 2))(var_x)
        var_x2 = layers.LeakyReLU(0.1)(var_x2)
        var_x = layers.Concatenate()([var_x1, var_x2])

        var_x1 = Conv2DBlock(self.encoder_filters * 8, activation="leakyrelu")(var_x)
        var_x2 = layers.AveragePooling2D(pool_size=(2, 2))(var_x)
        var_x2 = layers.LeakyReLU(0.1)(var_x2)
        var_x = layers.Concatenate()([var_x1, var_x2])

        var_x = layers.Dense(self.encoder_dim)(layers.Flatten()(var_x))
        var_x = layers.Dropout(0.05)(var_x)
        var_x = layers.Dense(4 * 4 * 1024)(var_x)
        var_x = layers.Dropout(0.05)(var_x)
        var_x = layers.Reshape((4, 4, 1024))(var_x)

        return KModel(input_, var_x, name="encoder")

    def decoder_a(self):
        """ DeLight Decoder A(old face) Network """
        input_ = Input(shape=(4, 4, 1024))
        dec_a_complexity = 256
        mask_complexity = 128

        var_xy = input_
        var_xy = layers.UpSampling2D(self.upscale_ratio, interpolation='bilinear')(var_xy)

        var_x = var_xy
        var_x = Upscale2xBlock(dec_a_complexity, activation="leakyrelu", fast=False)(var_x)
        var_x = Upscale2xBlock(dec_a_complexity // 2, activation="leakyrelu", fast=False)(var_x)
        var_x = Upscale2xBlock(dec_a_complexity // 4, activation="leakyrelu", fast=False)(var_x)
        var_x = Upscale2xBlock(dec_a_complexity // 8, activation="leakyrelu", fast=False)(var_x)

        var_x = Conv2DOutput(3, 5, name="face_out")(var_x)

        outputs = [var_x]

        if cfg_loss.learn_mask():
            var_y = var_xy  # mask decoder
            var_y = Upscale2xBlock(mask_complexity, activation="leakyrelu", fast=False)(var_y)
            var_y = Upscale2xBlock(mask_complexity // 2, activation="leakyrelu", fast=False)(var_y)
            var_y = Upscale2xBlock(mask_complexity // 4, activation="leakyrelu", fast=False)(var_y)
            var_y = Upscale2xBlock(mask_complexity // 8, activation="leakyrelu", fast=False)(var_y)

            var_y = Conv2DOutput(1, 5, name="mask_out")(var_y)

            outputs.append(var_y)

        return KModel([input_], outputs=outputs, name="decoder_a")

    def decoder_b_fast(self):
        """ DeLight Fast Decoder B(new face) Network  """
        input_ = Input(shape=(4, 4, 1024))

        dec_b_complexity = 512
        mask_complexity = 128

        var_xy = input_

        var_xy = UpscaleBlock(512, scale_factor=self.upscale_ratio, activation="leakyrelu")(var_xy)
        var_x = var_xy

        var_x = Upscale2xBlock(dec_b_complexity, activation="leakyrelu", fast=True)(var_x)
        var_x = Upscale2xBlock(dec_b_complexity // 2, activation="leakyrelu", fast=True)(var_x)
        var_x = Upscale2xBlock(dec_b_complexity // 4, activation="leakyrelu", fast=True)(var_x)
        var_x = Upscale2xBlock(dec_b_complexity // 8, activation="leakyrelu", fast=True)(var_x)

        var_x = Conv2DOutput(3, 5, name="face_out")(var_x)

        outputs = [var_x]

        if cfg_loss.learn_mask():
            var_y = var_xy  # mask decoder

            var_y = Upscale2xBlock(mask_complexity, activation="leakyrelu", fast=False)(var_y)
            var_y = Upscale2xBlock(mask_complexity // 2, activation="leakyrelu", fast=False)(var_y)
            var_y = Upscale2xBlock(mask_complexity // 4, activation="leakyrelu", fast=False)(var_y)
            var_y = Upscale2xBlock(mask_complexity // 8, activation="leakyrelu", fast=False)(var_y)

            var_y = Conv2DOutput(1, 5, name="mask_out")(var_y)

            outputs.append(var_y)

        return KModel([input_], outputs=outputs, name="decoder_b_fast")

    def decoder_b(self):
        """ DeLight Decoder B(new face) Network  """
        input_ = Input(shape=(4, 4, 1024))

        dec_b_complexity = 512
        mask_complexity = 128

        var_xy = input_

        var_xy = Upscale2xBlock(512,
                                scale_factor=self.upscale_ratio,
                                activation=None,
                                fast=False)(var_xy)
        var_x = var_xy

        var_x = layers.LeakyReLU(negative_slope=0.2)(var_x)
        var_x = ResidualBlock(512, use_bias=True)(var_x)
        var_x = ResidualBlock(512, use_bias=False)(var_x)
        var_x = ResidualBlock(512, use_bias=False)(var_x)
        var_x = Upscale2xBlock(dec_b_complexity, activation=None, fast=False)(var_x)
        var_x = layers.LeakyReLU(negative_slope=0.2)(var_x)
        var_x = ResidualBlock(dec_b_complexity, use_bias=True)(var_x)
        var_x = ResidualBlock(dec_b_complexity, use_bias=False)(var_x)
        var_x = layers.BatchNormalization()(var_x)
        var_x = Upscale2xBlock(dec_b_complexity // 2, activation=None, fast=False)(var_x)
        var_x = layers.LeakyReLU(negative_slope=0.2)(var_x)
        var_x = ResidualBlock(dec_b_complexity // 2, use_bias=True)(var_x)
        var_x = Upscale2xBlock(dec_b_complexity // 4, activation=None, fast=False)(var_x)
        var_x = layers.LeakyReLU(negative_slope=0.2)(var_x)
        var_x = ResidualBlock(dec_b_complexity // 4, use_bias=False)(var_x)
        var_x = layers.BatchNormalization()(var_x)
        var_x = Upscale2xBlock(dec_b_complexity // 8, activation="leakyrelu", fast=False)(var_x)

        var_x = Conv2DOutput(3, 5, name="face_out")(var_x)

        outputs = [var_x]

        if cfg_loss.learn_mask():
            var_y = var_xy  # mask decoder
            var_y = layers.LeakyReLU(negative_slope=0.1)(var_y)

            var_y = Upscale2xBlock(mask_complexity, activation="leakyrelu", fast=False)(var_y)
            var_y = Upscale2xBlock(mask_complexity // 2, activation="leakyrelu", fast=False)(var_y)
            var_y = Upscale2xBlock(mask_complexity // 4, activation="leakyrelu", fast=False)(var_y)
            var_y = Upscale2xBlock(mask_complexity // 8, activation="leakyrelu", fast=False)(var_y)

            var_y = Conv2DOutput(1, 5, name="mask_out")(var_y)

            outputs.append(var_y)

        return KModel([input_], outputs=outputs, name="decoder_b")
