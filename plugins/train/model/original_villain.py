#!/usr/bin/env python3
""" Original - VillainGuy model
    Based on the original https://www.reddit.com/r/deepfakes/ code sample + contribs
    Adapted from a model by VillainGuy (https://github.com/VillainGuy) """

from keras.initializers import RandomNormal
from keras.layers import add, Conv2D, Dense, Flatten, Input, Reshape
from keras.models import Model as KerasModel

from lib.model.nn_blocks import conv, conv_sep, PixelShuffler, res_block, upscale
from .original import logger, Model as OriginalModel


class Model(OriginalModel):
    """ Original HiRes Faceswap Model """
    def __init__(self, *args, **kwargs):
        logger.debug("Initializing %s: (args: %s, kwargs: %s",
                     self.__class__.__name__, args, kwargs)

        kwargs["input_shape"] = (128, 128, 3)
        kwargs["encoder_dim"] = 512 if self.config["lowmem"] else 1024
        self.kernel_initializer = RandomNormal(0, 0.02)

        super().__init__(*args, **kwargs)
        logger.debug("Initialized %s", self.__class__.__name__)

    def set_training_data(self):
        """ Set the dictionary for training """
        logger.debug("Setting training data")
        training_opts = dict()
        training_opts["preview_images"] = 10
        logger.debug("Set training data: %s", training_opts)
        return training_opts

    def encoder(self):
        """ Encoder Network """
        kwargs = {"kernel_initializer": self.kernel_initializer}
        input_ = Input(shape=self.input_shape)
        in_conv_filters = self.input_shape[0]
        if self.input_shape[0] > 128:
            in_conv_filters = 128 + (self.input_shape[0] - 128) // 4
        dense_shape = self.input_shape[0] // 16

        var_x = conv(in_conv_filters, **kwargs)(input_)
        tmp_x = var_x
        res_cycles = 8 if self.config.get("lowmem", False) else 16
        for _ in range(res_cycles):
            nn_x = res_block(var_x, 128, **kwargs)
            var_x = nn_x
        var_x = add([var_x, tmp_x])
        var_x = conv(128, **kwargs)(var_x)
        var_x = PixelShuffler()(var_x)
        var_x = conv(128, **kwargs)(var_x)
        var_x = PixelShuffler()(var_x)
        var_x = conv(128, **kwargs)(var_x)
        var_x = conv_sep(256, **kwargs)(var_x)
        var_x = conv(512, **kwargs)(var_x)
        if not self.config.get("lowmem", False):
            var_x = conv_sep(1024, **kwargs)(var_x)

        var_x = Dense(self.encoder_dim, **kwargs)(Flatten()(var_x))
        var_x = Dense(dense_shape * dense_shape * 1024, **kwargs)(var_x)
        var_x = Reshape((dense_shape, dense_shape, 1024))(var_x)
        var_x = upscale(512, **kwargs)(var_x)
        return KerasModel(input_, var_x)

    def decoder(self):
        """ Decoder Network """
        decoder_shape = self.input_shape[0] // 8
        kwargs = {"kernel_initializer": self.kernel_initializer}
        input_ = Input(shape=(decoder_shape, decoder_shape, 512))

        var_x = upscale(512, **kwargs)(input_)
        var_x = res_block(var_x, 512, **kwargs)
        var_x = upscale(256, **kwargs)(var_x)
        var_x = res_block(var_x, 256, **kwargs)
        var_x = upscale(self.input_shape[0], **kwargs)(var_x)
        var_x = res_block(var_x, self.input_shape[0], **kwargs)
        var_x = Conv2D(3, kernel_size=5, padding='same', activation='sigmoid')(var_x)
        return KerasModel(input_, var_x)
