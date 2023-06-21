#!/usr/bin/env python3
""" Lightweight Model by torzdf
    An extremely limited model for training on low-end graphics cards
    Based on the original https://www.reddit.com/r/deepfakes/
    code sample + contributions """

from tensorflow.keras.models import Model as KModel  # pylint:disable=import-error

from lib.model.nn_blocks import Conv2DOutput, Conv2DBlock, UpscaleBlock
from .original import Model as OriginalModel, Dense, Flatten, Input, Reshape


class Model(OriginalModel):
    """ Lightweight Model for ~2GB Graphics Cards """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder_dim = 512

    def encoder(self):
        """ Encoder Network """
        input_ = Input(shape=self.input_shape)
        var_x = input_
        var_x = Conv2DBlock(128, activation="leakyrelu")(var_x)
        var_x = Conv2DBlock(256, activation="leakyrelu")(var_x)
        var_x = Conv2DBlock(512, activation="leakyrelu")(var_x)
        var_x = Dense(self.encoder_dim)(Flatten()(var_x))
        var_x = Dense(4 * 4 * 512)(var_x)
        var_x = Reshape((4, 4, 512))(var_x)
        var_x = UpscaleBlock(256, activation="leakyrelu")(var_x)
        return KModel(input_, var_x, name="encoder")

    def decoder(self, side):
        """ Decoder Network """
        input_ = Input(shape=(8, 8, 256))
        var_x = input_
        var_x = UpscaleBlock(512, activation="leakyrelu")(var_x)
        var_x = UpscaleBlock(256, activation="leakyrelu")(var_x)
        var_x = UpscaleBlock(128, activation="leakyrelu")(var_x)
        var_x = Conv2DOutput(3, 5, activation="sigmoid", name=f"face_out_{side}")(var_x)
        outputs = [var_x]

        if self.config.get("learn_mask", False):
            var_y = input_
            var_y = UpscaleBlock(512, activation="leakyrelu")(var_y)
            var_y = UpscaleBlock(256, activation="leakyrelu")(var_y)
            var_y = UpscaleBlock(128, activation="leakyrelu")(var_y)
            var_y = Conv2DOutput(1, 5,
                                 activation="sigmoid",
                                 name=f"mask_out_{side}")(var_y)
            outputs.append(var_y)
        return KModel(input_, outputs=outputs, name=f"decoder_{side}")
