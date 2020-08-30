#!/usr/bin/env python3
""" DeepFaceLab H128 Model
    Based on https://github.com/iperov/DeepFaceLab
"""

from keras.layers import Dense, Flatten, Input, Reshape

from lib.model.nn_blocks import Conv2DOutput, Conv2DBlock, UpscaleBlock
from .original import Model as OriginalModel, KerasModel


class Model(OriginalModel):
    """ H128 Model from DFL """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_shape = (128, 128, 3)
        self.encoder_dim = 256 if self.config["lowmem"] else 512

    def encoder(self):
        """ DFL H128 Encoder """
        input_ = Input(shape=self.input_shape)
        var_x = Conv2DBlock(128)(input_)
        var_x = Conv2DBlock(256)(var_x)
        var_x = Conv2DBlock(512)(var_x)
        var_x = Conv2DBlock(1024)(var_x)
        var_x = Dense(self.encoder_dim)(Flatten()(var_x))
        var_x = Dense(8 * 8 * self.encoder_dim)(var_x)
        var_x = Reshape((8, 8, self.encoder_dim))(var_x)
        var_x = UpscaleBlock(self.encoder_dim)(var_x)
        return KerasModel(input_, var_x, name="encoder")

    def decoder(self, side):
        """ DFL H128 Decoder """
        input_ = Input(shape=(16, 16, self.encoder_dim))
        var_x = input_
        var_x = UpscaleBlock(self.encoder_dim)(var_x)
        var_x = UpscaleBlock(self.encoder_dim // 2)(var_x)
        var_x = UpscaleBlock(self.encoder_dim // 4)(var_x)
        var_x = Conv2DOutput(3, 5, name="face_out_{}".format(side))(var_x)
        outputs = [var_x]

        if self.config.get("learn_mask", False):
            var_y = input_
            var_y = UpscaleBlock(self.encoder_dim)(var_y)
            var_y = UpscaleBlock(self.encoder_dim // 2)(var_y)
            var_y = UpscaleBlock(self.encoder_dim // 4)(var_y)
            var_y = Conv2DOutput(1, 5, name="mask_out_{}".format(side))(var_y)
            outputs.append(var_y)
        return KerasModel(input_, outputs=outputs, name="decoder_{}".format(side))
