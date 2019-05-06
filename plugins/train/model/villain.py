#!/usr/bin/env python3
""" Original - VillainGuy model
    Based on the original https://www.reddit.com/r/deepfakes/ code sample + contribs
    Adapted from a model by VillainGuy (https://github.com/VillainGuy) """

from keras.initializers import RandomNormal
from keras.layers import Add, Conv2D, Dense, Flatten, Input, Reshape, Concatenate
from keras.models import Model as KerasModel

from lib.model.layers import PixelShuffler
from .original import logger, Model as OriginalModel


class Model(OriginalModel):
    """ Villain Faceswap Model """
    def __init__(self, *args, **kwargs):
        logger.debug("Initializing %s: (args: %s, kwargs: %s",
                     self.__class__.__name__, args, kwargs)
        kwargs["input_shape"] = (128, 128, 3)
        kwargs["encoder_dim"] = 512 if self.config["lowmem"] else 1024
        self.kernel_initializer = RandomNormal(0, 0.02)

        super().__init__(*args, **kwargs)
        logger.debug("Initialized %s", self.__class__.__name__)

    def encoder(self):
        """ Encoder Network """
        kwargs = dict(kernel_initializer=self.kernel_initializer)
        extra = max((self.input_shape[0] - 128) // 4, 0)
        in_conv_filters = self.input_shape[0] + extra
        dense_shape = self.input_shape[0] // 16
        res_cycles = 8 if self.config.get("lowmem", False) else 16

        face_ = Input(shape=self.input_shape)
        mask_ = Input(shape=self.mask_shape)
        var_x = Concatenate(axis=-1)([face_, mask_])
        var_x = self.blocks.conv(var_x, in_conv_filters, res_block_follows=True, **kwargs)
        initial_conv = var_x
        for _ in range(res_cycles):
            var_x = self.blocks.res_block(var_x, 128, **kwargs)
        var_x = Add()([var_x, initial_conv])
        var_x = self.blocks.conv(var_x, 128, **kwargs)
        var_x = PixelShuffler()(var_x)
        var_x = self.blocks.conv(var_x, 128, **kwargs)
        var_x = PixelShuffler()(var_x)
        var_x = self.blocks.conv(var_x, 128, **kwargs)
        var_x = self.blocks.conv_sep(var_x, 256, **kwargs)
        var_x = self.blocks.conv(var_x, 512, **kwargs)
        if not self.config.get("lowmem", False):
            var_x = self.blocks.conv_sep(var_x, 1024, **kwargs)
        var_x = Flatten()(var_x)
        var_x = Dense(self.encoder_dim, **kwargs)(var_x)
        var_x = Dense(dense_shape * dense_shape * 1024, **kwargs)(var_x)
        var_x = Reshape((dense_shape, dense_shape, 1024))(var_x)
        var_x = self.blocks.upscale(var_x, 512, **kwargs)
        return KerasModel([face_, mask_], var_x)

    def decoder(self):
        """ Decoder Network """
        kwargs = dict(kernel_initializer=self.kernel_initializer)
        decoder_shape = self.input_shape[0] // 8
        input_ = Input(shape=(decoder_shape, decoder_shape, 512))

        var_x = input_
        var_x = self.blocks.upscale(var_x, 512, res_block_follows=True, **kwargs)
        var_x = self.blocks.res_block(var_x, 512, **kwargs)
        var_x = self.blocks.upscale(var_x, 256, res_block_follows=True, **kwargs)
        var_x = self.blocks.res_block(var_x, 256, **kwargs)
        var_x = self.blocks.upscale(var_x, self.input_shape[0], res_block_follows=True, **kwargs)
        var_x = self.blocks.res_block(var_x, self.input_shape[0], **kwargs)
        var_x = Conv2D(3, kernel_size=5, padding='same', activation='sigmoid')(var_x)
        outputs = [var_x]

        if self.config.get("mask_type", None):
            var_y = input_
            var_y = self.blocks.upscale(var_y, 512)
            var_y = self.blocks.upscale(var_y, 256)
            var_y = self.blocks.upscale(var_y, self.input_shape[0])
            var_y = Conv2D(1, kernel_size=5, padding='same', activation='sigmoid')(var_y)
            outputs.append(var_y)
        return KerasModel(input_, outputs=outputs)
