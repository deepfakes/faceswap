#!/usr/bin/env python3
""" DFaker Model
    Based on the dfaker model: https://github.com/dfaker """


from keras.initializers import RandomNormal
from keras.layers import Input

from lib.model.nn_blocks import Conv2DOutput, UpscaleBlock, ResidualBlock
from .original import Model as OriginalModel, KerasModel


class Model(OriginalModel):
    """ Dfaker Model """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_shape = (64, 64, 3)
        self.encoder_dim = 1024
        self.kernel_initializer = RandomNormal(0, 0.02)

    def decoder(self, side):
        """ Decoder Network """
        input_ = Input(shape=(8, 8, 512))
        var_x = input_

        var_x = UpscaleBlock(512, res_block_follows=True)(var_x)
        var_x = ResidualBlock(512, kernel_initializer=self.kernel_initializer)(var_x)
        var_x = UpscaleBlock(256, res_block_follows=True)(var_x)
        var_x = ResidualBlock(256, kernel_initializer=self.kernel_initializer)(var_x)
        var_x = UpscaleBlock(128, res_block_follows=True)(var_x)
        var_x = ResidualBlock(128, kernel_initializer=self.kernel_initializer)(var_x)
        var_x = UpscaleBlock(64)(var_x)
        var_x = Conv2DOutput(3, 5, name="face_out_{}".format(side))(var_x)
        outputs = [var_x]

        if self.config.get("learn_mask", False):
            var_y = input_
            var_y = UpscaleBlock(512)(var_y)
            var_y = UpscaleBlock(256)(var_y)
            var_y = UpscaleBlock(128)(var_y)
            var_y = UpscaleBlock(64)(var_y)
            var_y = Conv2DOutput(1, 5, name="mask_out_{}".format(side))(var_y)
            outputs.append(var_y)
        return KerasModel([input_], outputs=outputs, name="decoder_{}".format(side))
