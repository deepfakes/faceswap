#!/usr/bin/env python3
""" Improved autoencoder for faceswap """

from keras.layers import Concatenate, Dense, Flatten, Input, Reshape
from keras.models import Model as KerasModel

from lib.model.nn_blocks import Conv2D, Conv2DBlock, UpscaleBlock
from ._base import ModelBase


class Model(ModelBase):
    """ Improved Autoencoder Model """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_shape = (64, 64, 3)
        self.output_shape = (64, 64, 3)
        self.encoder_dim = 1024

    def build_model(self, inputs):
        """ Build the IAE Model """
        input_a = inputs[0]
        input_b = inputs[1]

        encoder = self.encoder()
        decoder = self.decoder()
        inter_a = self.intermediate("a")
        inter_b = self.intermediate("b")
        inter_both = self.intermediate("both")

        encoder_a = encoder(input_a)
        encoder_b = encoder(input_b)

        outputs = [decoder(Concatenate()([inter_a(encoder_a), inter_both(encoder_a)])),
                   decoder(Concatenate()([inter_b(encoder_b), inter_both(encoder_b)]))]

        autoencoder = KerasModel(inputs, outputs, name=self.name)
        return autoencoder

    def encoder(self):
        """ Encoder Network """
        input_ = Input(shape=self.input_shape)
        var_x = input_
        var_x = Conv2DBlock(128)(var_x)
        var_x = Conv2DBlock(256)(var_x)
        var_x = Conv2DBlock(512)(var_x)
        var_x = Conv2DBlock(1024)(var_x)
        var_x = Flatten()(var_x)
        return KerasModel(input_, var_x, name="encoder")

    def intermediate(self, side):
        """ Intermediate Network """
        input_ = Input(shape=(4 * 4 * 1024))
        var_x = Dense(self.encoder_dim)(input_)
        var_x = Dense(4 * 4 * int(self.encoder_dim/2))(var_x)
        var_x = Reshape((4, 4, int(self.encoder_dim/2)))(var_x)
        return KerasModel(input_, var_x, name="inter_{}".format(side))

    def decoder(self):
        """ Decoder Network """
        input_ = Input(shape=(4, 4, self.encoder_dim))
        var_x = input_
        var_x = UpscaleBlock(512)(var_x)
        var_x = UpscaleBlock(256)(var_x)
        var_x = UpscaleBlock(128)(var_x)
        var_x = UpscaleBlock(64)(var_x)
        var_x = Conv2D(3,
                       kernel_size=5,
                       padding="same",
                       activation="sigmoid",
                       name="face_out")(var_x)
        outputs = [var_x]

        if self.config.get("learn_mask", False):
            var_y = input_
            var_y = UpscaleBlock(512)(var_y)
            var_y = UpscaleBlock(256)(var_y)
            var_y = UpscaleBlock(128)(var_y)
            var_y = UpscaleBlock(64)(var_y)
            var_y = Conv2D(var_y, 1,
                           kernel_size=5,
                           padding="same",
                           activation="sigmoid",
                           name="mask_out")(var_y)
            outputs.append(var_y)
        return KerasModel(input_, outputs=outputs, name="decoder")
