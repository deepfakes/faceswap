#!/usr/bin/env python3
""" Original Model
    Based on the original https://www.reddit.com/r/deepfakes/
    code sample + contribs """

from keras.layers import Dense, Flatten, Input, Reshape
from keras.models import Model as KerasModel

from .original import logger, Model as OriginalModel


class Model(OriginalModel):
    """ Lightweight Model for ~2GB Graphics Cards """
    def __init__(self, *args, **kwargs):
        logger.debug("Initializing %s: (args: %s, kwargs: %s",
                     self.__class__.__name__, args, kwargs)

        kwargs["input_shape"] = (64, 64, 3)
        kwargs["encoder_dim"] = 512
        super().__init__(*args, **kwargs)
        logger.debug("Initialized %s", self.__class__.__name__)

    def encoder(self):
        """ Encoder Network """
        input_ = Input(shape=self.input_shape)
        var_x = input_
        var_x = self.blocks.conv(var_x, 128)
        var_x = self.blocks.conv(var_x, 256)
        var_x = self.blocks.conv(var_x, 512)
        var_x = Dense(self.encoder_dim)(Flatten()(var_x))
        var_x = Dense(4 * 4 * 512)(var_x)
        var_x = Reshape((4, 4, 512))(var_x)
        var_x = self.blocks.upscale(var_x, 256)
        return KerasModel(input_, var_x)

    def decoder(self):
        """ Decoder Network """
        input_ = Input(shape=(8, 8, 256))
        var_x = input_
        var_x = self.blocks.upscale(var_x, 512)
        var_x = self.blocks.upscale(var_x, 256)
        var_x = self.blocks.upscale(var_x, 128)
        var_x = self.blocks.conv2d(var_x, 3,
                                   kernel_size=5,
                                   padding="same",
                                   activation="sigmoid",
                                   name="face_out")
        outputs = [var_x]

        if self.config.get("learn_mask", False):
            var_y = input_
            var_y = self.blocks.upscale(var_y, 512)
            var_y = self.blocks.upscale(var_y, 256)
            var_y = self.blocks.upscale(var_y, 128)
            var_y = self.blocks.conv2d(var_y, 1,
                                       kernel_size=5,
                                       padding="same",
                                       activation="sigmoid",
                                       name="mask_out")
            outputs.append(var_y)
        return KerasModel(input_, outputs=outputs)
