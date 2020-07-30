#!/usr/bin/env python3
""" Lightweight Model by torzdf
    An extremely limited model for training on low-end graphics cards
    Based on the original https://www.reddit.com/r/deepfakes/
    code sample + contributions """

from keras.layers import Dense, Flatten, Input, Reshape

from lib.model.nn_blocks import Conv2DOutput, Conv2DBlock, UpscaleBlock
from .original import Model as OriginalModel, KerasModel


class Model(OriginalModel):
    """ Lightweight Model for ~2GB Graphics Cards """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder_dim = 512

    def encoder(self):
        """ Encoder Network """
        input_ = Input(shape=self.input_shape)
        var_x = input_
        var_x = Conv2DBlock(128)(var_x)
        var_x = Conv2DBlock(256)(var_x)
        var_x = Conv2DBlock(512)(var_x)
        var_x = Dense(self.encoder_dim)(Flatten()(var_x))
        var_x = Dense(4 * 4 * 512)(var_x)
        var_x = Reshape((4, 4, 512))(var_x)
        var_x = UpscaleBlock(256)(var_x)
        return KerasModel(input_, var_x, name="encoder")

    def decoder(self, side):
        """ Decoder Network """
        input_ = Input(shape=(8, 8, 256))
        var_x = input_
        var_x = UpscaleBlock(512)(var_x)
        var_x = UpscaleBlock(256)(var_x)
        var_x = UpscaleBlock(128)(var_x)
        var_x = Conv2DOutput(3, 5, activation="sigmoid", name="face_out_{}".format(side))(var_x)
        outputs = [var_x]

        if self.config.get("learn_mask", False):
            var_y = input_
            var_y = UpscaleBlock(512)(var_y)
            var_y = UpscaleBlock(256)(var_y)
            var_y = UpscaleBlock(128)(var_y)
            var_y = Conv2DOutput(1, 5,
                                 activation="sigmoid",
                                 name="mask_out_{}".format(side))(var_y)
            outputs.append(var_y)
        return KerasModel(input_, outputs=outputs, name="decoder_{}".format(side))
