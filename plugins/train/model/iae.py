#!/usr/bin/env python3
""" Improved autoencoder for faceswap """

from keras import Input, layers, Model as KModel

from lib.model.nn_blocks import Conv2DOutput, Conv2DBlock, UpscaleBlock
from plugins.train.train_config import Loss as cfg_loss

from ._base import ModelBase
# pylint:disable=duplicate-code


class Model(ModelBase):
    """ Improved Autoencoder Model """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_shape = (64, 64, 3)
        self.encoder_dim = 1024

    def build_model(self, inputs):
        """ Build the IAE Model """
        encoder = self.encoder()
        decoder = self.decoder()
        inter_a = self.intermediate("a")
        inter_b = self.intermediate("b")
        inter_both = self.intermediate("both")

        encoder_a = encoder(inputs[0])
        encoder_b = encoder(inputs[1])

        outputs = (decoder(layers.Concatenate()([inter_a(encoder_a), inter_both(encoder_a)])) +
                   decoder(layers.Concatenate()([inter_b(encoder_b), inter_both(encoder_b)])))

        autoencoder = KModel(inputs, outputs, name=self.model_name)
        return autoencoder

    def encoder(self):
        """ Encoder Network """
        input_ = Input(shape=self.input_shape)
        var_x = input_
        var_x = Conv2DBlock(128, activation="leakyrelu")(var_x)
        var_x = Conv2DBlock(256, activation="leakyrelu")(var_x)
        var_x = Conv2DBlock(512, activation="leakyrelu")(var_x)
        var_x = Conv2DBlock(1024, activation="leakyrelu")(var_x)
        var_x = layers.Flatten()(var_x)
        return KModel(input_, var_x, name="encoder")

    def intermediate(self, side):
        """ Intermediate Network """
        input_ = Input(shape=(4 * 4 * 1024, ))
        var_x = layers.Dense(self.encoder_dim)(input_)
        var_x = layers.Dense(4 * 4 * int(self.encoder_dim/2))(var_x)
        var_x = layers.Reshape((4, 4, int(self.encoder_dim/2)))(var_x)
        return KModel(input_, var_x, name=f"inter_{side}")

    def decoder(self):
        """ Decoder Network """
        input_ = Input(shape=(4, 4, self.encoder_dim))
        var_x = input_
        var_x = UpscaleBlock(512, activation="leakyrelu")(var_x)
        var_x = UpscaleBlock(256, activation="leakyrelu")(var_x)
        var_x = UpscaleBlock(128, activation="leakyrelu")(var_x)
        var_x = UpscaleBlock(64, activation="leakyrelu")(var_x)
        var_x = Conv2DOutput(3, 5, name="face_out")(var_x)
        outputs = [var_x]

        if cfg_loss.learn_mask():
            var_y = input_
            var_y = UpscaleBlock(512, activation="leakyrelu")(var_y)
            var_y = UpscaleBlock(256, activation="leakyrelu")(var_y)
            var_y = UpscaleBlock(128, activation="leakyrelu")(var_y)
            var_y = UpscaleBlock(64, activation="leakyrelu")(var_y)
            var_y = Conv2DOutput(1, 5, name="mask_out")(var_y)
            outputs.append(var_y)
        return KModel(input_, outputs=outputs, name="decoder")
