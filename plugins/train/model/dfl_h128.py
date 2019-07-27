#!/usr/bin/env python3
""" DeepFakesLab H128 Model
    Based on https://github.com/iperov/DeepFaceLab
"""

from keras.layers import Conv2D, Dense, Flatten, Input, Reshape, Concatenate
from keras.models import Model as KerasModel

from .original import logger, Model as OriginalModel


class Model(OriginalModel):
    """ Low Memory version of Original Faceswap Model """
    def __init__(self, *args, **kwargs):
        logger.debug("Initializing %s: (args: %s, kwargs: %s",
                     self.__class__.__name__, args, kwargs)

        self.configfile = kwargs.get("configfile", None)
        kwargs["input_shape"] = (128, 128, 3)
        kwargs["encoder_dim"] = 256 if self.config["lowmem"] else 512

        super().__init__(*args, **kwargs)
        logger.debug("Initialized %s", self.__class__.__name__)

    def encoder(self):
        """ DFL H128 Encoder """
        face_ = Input(shape=self.input_shape)
        mask_ = Input(shape=self.mask_shape)
        var_x = Concatenate(axis=-1)([face_, mask_])
        var_x = self.blocks.conv(var_x, 128)
        var_x = self.blocks.conv(var_x, 256)
        var_x = self.blocks.conv(var_x, 512)
        var_x = self.blocks.conv(var_x, 1024)
        var_x = Flatten()(var_x)
        var_x = Dense(self.encoder_dim)(var_x)
        var_x = Dense(8 * 8 * self.encoder_dim)(var_x)
        var_x = Reshape((8, 8, self.encoder_dim))(var_x)
        var_x = self.blocks.upscale(var_x, self.encoder_dim)
        return KerasModel([face_, mask_], var_x)

    def decoder(self):
        """ DFL H128 Decoder """
        input_ = Input(shape=(16, 16, self.encoder_dim))
        # Face
        var_x = input_
        var_x = self.blocks.upscale(var_x, self.encoder_dim)
        var_x = self.blocks.upscale(var_x, self.encoder_dim // 2)
        var_x = self.blocks.upscale(var_x, self.encoder_dim // 4)
        var_x = self.blocks.conv2d(var_x, 3,
                                   kernel_size=5,
                                   padding="same",
                                   activation="sigmoid",
                                   name="face_out")
        outputs = [var_x]
        # Mask
        if self.config.get("mask_type", None):
            var_y = input_
            var_y = self.blocks.upscale(var_y, self.encoder_dim)
            var_y = self.blocks.upscale(var_y, self.encoder_dim // 2)
            var_y = self.blocks.upscale(var_y, self.encoder_dim // 4)
            var_y = self.blocks.conv2d(var_y, 1,
                                       kernel_size=5,
                                       padding="same",
                                       activation="sigmoid",
                                       name="mask_out")
            outputs.append(var_y)
        return KerasModel(input_, outputs=outputs)
