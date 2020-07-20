#!/usr/bin/env python3
""" DFaker Model
    Based on the dfaker model: https://github.com/dfaker """


from keras.initializers import RandomNormal
from keras.layers import Input
from keras.models import Model as KerasModel

from lib.model.nn_blocks import Conv2D, UpscaleBlock, ResidualBlock
from .original import logger, Model as OriginalModel


class Model(OriginalModel):
    """ Dfaker Model """
    def __init__(self, *args, **kwargs):
        logger.debug("Initializing %s: (args: %s, kwargs: %s",
                     self.__class__.__name__, args, kwargs)
        kwargs["input_shape"] = (64, 64, 3)
        kwargs["encoder_dim"] = 1024
        self.kernel_initializer = RandomNormal(0, 0.02)
        super().__init__(*args, **kwargs)
        logger.debug("Initialized %s", self.__class__.__name__)

    def decoder(self):
        """ Decoder Network """
        input_ = Input(shape=(8, 8, 512))
        var_x = input_

        var_x = UpscaleBlock(512, res_block_follows=True)(var_x)
        var_x = ResidualBlock(512, kernel_initializer=self.kernel_initializer)(var_x)
        var_x = UpscaleBlock(var_x, 256, res_block_follows=True)(var_x)
        var_x = ResidualBlock(256, kernel_initializer=self.kernel_initializer)(var_x)
        var_x = UpscaleBlock(128, res_block_follows=True)(var_x)
        var_x = ResidualBlock(128, kernel_initializer=self.kernel_initializer)(var_x)
        var_x = UpscaleBlock(64)(var_x)
        var_x = Conv2D(3, 5, activation="sigmoid", name="face_out")(var_x)
        outputs = [var_x]

        if self.config.get("learn_mask", False):
            var_y = input_
            var_y = UpscaleBlock(512)(var_y)
            var_y = UpscaleBlock(256)(var_y)
            var_y = UpscaleBlock(128)(var_y)
            var_y = UpscaleBlock(64)(var_y)
            var_y = Conv2D(1, 5, activation="sigmoid", name="mask_out")(var_y)
            outputs.append(var_y)
        return KerasModel([input_], outputs=outputs)
