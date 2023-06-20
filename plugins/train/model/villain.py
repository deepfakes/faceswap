#!/usr/bin/env python3
""" Original - VillainGuy model
    Based on the original https://www.reddit.com/r/deepfakes/ code sample + contributions
    Adapted from a model by VillainGuy (https://github.com/VillainGuy) """

# Ignore linting errors from Tensorflow's thoroughly broken import system
from tensorflow.keras.initializers import RandomNormal  # pylint:disable=import-error
from tensorflow.keras.layers import add, Dense, Flatten, Input, LeakyReLU, Reshape  # noqa:E501  # pylint:disable=import-error
from tensorflow.keras.models import Model as KModel  # pylint:disable=import-error

from lib.model.layers import PixelShuffler
from lib.model.nn_blocks import (Conv2DOutput, Conv2DBlock, ResidualBlock, SeparableConv2DBlock,
                                 UpscaleBlock)

from .original import Model as OriginalModel


class Model(OriginalModel):
    """ Villain Faceswap Model """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_shape = (128, 128, 3)
        self.encoder_dim = 512 if self.low_mem else 1024
        self.kernel_initializer = RandomNormal(0, 0.02)

    def encoder(self):
        """ Encoder Network """
        kwargs = {"kernel_initializer": self.kernel_initializer}
        input_ = Input(shape=self.input_shape)
        in_conv_filters = self.input_shape[0]
        if self.input_shape[0] > 128:
            in_conv_filters = 128 + (self.input_shape[0] - 128) // 4
        dense_shape = self.input_shape[0] // 16

        var_x = Conv2DBlock(in_conv_filters, activation=None, **kwargs)(input_)
        tmp_x = var_x

        var_x = LeakyReLU(alpha=0.2)(var_x)
        res_cycles = 8 if self.config.get("lowmem", False) else 16
        for _ in range(res_cycles):
            nn_x = ResidualBlock(in_conv_filters, **kwargs)(var_x)
            var_x = nn_x
        # consider adding scale before this layer to scale the residual chain
        tmp_x = LeakyReLU(alpha=0.1)(tmp_x)
        var_x = add([var_x, tmp_x])
        var_x = Conv2DBlock(128, activation="leakyrelu", **kwargs)(var_x)
        var_x = PixelShuffler()(var_x)
        var_x = Conv2DBlock(128, activation="leakyrelu", **kwargs)(var_x)
        var_x = PixelShuffler()(var_x)
        var_x = Conv2DBlock(128, activation="leakyrelu", **kwargs)(var_x)
        var_x = SeparableConv2DBlock(256, **kwargs)(var_x)
        var_x = Conv2DBlock(512, activation="leakyrelu", **kwargs)(var_x)
        if not self.config.get("lowmem", False):
            var_x = SeparableConv2DBlock(1024, **kwargs)(var_x)

        var_x = Dense(self.encoder_dim, **kwargs)(Flatten()(var_x))
        var_x = Dense(dense_shape * dense_shape * 1024, **kwargs)(var_x)
        var_x = Reshape((dense_shape, dense_shape, 1024))(var_x)
        var_x = UpscaleBlock(512, activation="leakyrelu", **kwargs)(var_x)
        return KModel(input_, var_x, name="encoder")

    def decoder(self, side):
        """ Decoder Network """
        kwargs = {"kernel_initializer": self.kernel_initializer}
        decoder_shape = self.input_shape[0] // 8
        input_ = Input(shape=(decoder_shape, decoder_shape, 512))

        var_x = input_
        var_x = UpscaleBlock(512, activation=None, **kwargs)(var_x)
        var_x = LeakyReLU(alpha=0.2)(var_x)
        var_x = ResidualBlock(512, **kwargs)(var_x)
        var_x = UpscaleBlock(256, activation=None, **kwargs)(var_x)
        var_x = LeakyReLU(alpha=0.2)(var_x)
        var_x = ResidualBlock(256, **kwargs)(var_x)
        var_x = UpscaleBlock(self.input_shape[0], activation=None, **kwargs)(var_x)
        var_x = LeakyReLU(alpha=0.2)(var_x)
        var_x = ResidualBlock(self.input_shape[0], **kwargs)(var_x)
        var_x = Conv2DOutput(3, 5, name=f"face_out_{side}")(var_x)
        outputs = [var_x]

        if self.config.get("learn_mask", False):
            var_y = input_
            var_y = UpscaleBlock(512, activation="leakyrelu")(var_y)
            var_y = UpscaleBlock(256, activation="leakyrelu")(var_y)
            var_y = UpscaleBlock(self.input_shape[0], activation="leakyrelu")(var_y)
            var_y = Conv2DOutput(1, 5, name=f"mask_out_{side}")(var_y)
            outputs.append(var_y)
        return KModel(input_, outputs=outputs, name=f"decoder_{side}")
