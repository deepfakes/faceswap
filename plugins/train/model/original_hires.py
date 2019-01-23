#!/usr/bin/env python3
""" Original - HiRes Model
    Based on the original https://www.reddit.com/r/deepfakes/
        code sample + contribs """

from keras.initializers import RandomNormal
from keras.layers import Conv2D, Dense, Flatten, Input, Reshape, SpatialDropout2D
from keras.models import Model as KerasModel

from .original import logger, Model as OriginalModel


class Model(OriginalModel):
    """ Original HiRes Faceswap Model """
    def __init__(self, *args, **kwargs):
        logger.debug("Initializing %s: (args: %s, kwargs: %s",
                     self.__class__.__name__, args, kwargs)

        kwargs["input_shape"] = (self.config["input_size"], self.config["input_size"], 3)
        kwargs["encoder_dim"] = self.config["nodes"]
        self.lowmem = self.config.get("lowmem", False)
        self.kernel_initializer = RandomNormal(0, 0.02)

        super().__init__(*args, **kwargs)
        logger.debug("Initialized %s", self.__class__.__name__)

    def add_networks(self):
        """ Add the original model weights """
        logger.debug("Adding networks")
        if not self.lowmem:
            self.add_network("decoder", "a", self.decoder_a())
            self.add_network("decoder", "b", self.decoder_b())
            self.add_network("encoder", None, self.encoder())
        else:
            self.add_network("decoder", "a", self.decoder_a_lowmem())
            self.add_network("decoder", "b", self.decoder_b_lowmem())
            self.add_network("encoder", None, self.encoder_lowmem())

        logger.debug("Added networks")

    def encoder(self):
        """ Original HiRes Encoder """
        kwargs = dict(kernel_size=5, kernel_initializer=self.kernel_initializer)
        dense_shape = self.input_shape[0] // 16
        input_ = Input(shape=self.input_shape)

        encoder_complexity = self.config["complexity_encoder"]

        var_x = input_

        var_x = self.blocks.conv(var_x, encoder_complexity, res_block_follows=True, **kwargs)
        var_x = self.blocks.res_block(var_x, encoder_complexity)
        var_x = self.blocks.conv(var_x, encoder_complexity * 2, res_block_follows=True, **kwargs)
        var_x = self.blocks.res_block(var_x, encoder_complexity*2)
        var_x = self.blocks.conv(var_x, encoder_complexity * 4, res_block_follows=True, **kwargs)
        var_x = self.blocks.res_block(var_x, encoder_complexity*4)
        var_x = self.blocks.conv(var_x, encoder_complexity * 6, **kwargs)
        var_x = self.blocks.conv(var_x, encoder_complexity * 8, **kwargs)

        var_x = Dense(self.encoder_dim,
                      kernel_initializer=self.kernel_initializer)(Flatten()(var_x))
        var_x = Dense(dense_shape * dense_shape * 512,
                      kernel_initializer=self.kernel_initializer)(var_x)
        var_x = Reshape((dense_shape, dense_shape, 512))(var_x)
        return KerasModel(input_, var_x)

    def decoder_a(self):
        """ Decoder for side A """
        kwargs = dict(kernel_size=5, kernel_initializer=self.kernel_initializer)
        decoder_shape = self.input_shape[0] // 16

        # decoder_complexity_a = self.config["complexity_decoder_a"]
        decoder_complexity_a = 256

        input_ = Input(shape=(decoder_shape, decoder_shape, 512))

        var_x = input_
        var_x = self.blocks.upscale(var_x, 384, **kwargs)
        var_x = SpatialDropout2D(0.25)(var_x)

        var_x = self.blocks.upscale(var_x, decoder_complexity_a, **kwargs)
        var_x = SpatialDropout2D(0.15)(var_x)

        var_x = self.blocks.upscale(var_x, decoder_complexity_a // 2, **kwargs)

        var_x = self.blocks.upscale(var_x, decoder_complexity_a // 4, **kwargs)

        var_x = Conv2D(3, kernel_size=5, padding='same', activation='sigmoid')(var_x)
        return KerasModel(input_, var_x)

    def decoder_b(self):
        """ Decoder for side B """
        kwargs = dict(kernel_size=5, kernel_initializer=self.kernel_initializer)
        decoder_shape = self.input_shape[0] // 16

        # decoder_complexity_b = self.config["complexity_decoder_b"]
        decoder_complexity_b = 384

        input_ = Input(shape=(decoder_shape, decoder_shape, 512))

        var_x = input_

        var_x = self.blocks.upscale(var_x, 512, res_block_follows=True, **kwargs)
        var_x = self.blocks.res_block(var_x, 512, kernel_initializer=self.kernel_initializer)

        var_x = self.blocks.upscale(var_x, decoder_complexity_b, res_block_follows=True, **kwargs)
        var_x = self.blocks.res_block(var_x,
                                      decoder_complexity_b,
                                      kernel_initializer=self.kernel_initializer)

        var_x = self.blocks.upscale(var_x, decoder_complexity_b // 2, res_block_follows=True, **kwargs)
        var_x = self.blocks.res_block(var_x,
                                      decoder_complexity_b // 2,
                                      kernel_initializer=self.kernel_initializer)

        var_x = self.blocks.upscale(var_x, decoder_complexity_b // 4, **kwargs)

        var_x = Conv2D(3, kernel_size=5, padding='same', activation='sigmoid')(var_x)
        return KerasModel(input_, var_x)

    def encoder_lowmem(self):
        """ Original HiRes Encoder """
        kwargs = dict(kernel_size=5, kernel_initializer=self.kernel_initializer)
        dense_shape = self.input_shape[0] // 16
        input_ = Input(shape=self.input_shape)

        encoder_complexity = 128

        var_x = input_

        var_x = self.blocks.conv(var_x, encoder_complexity, **kwargs)
        var_x = self.blocks.conv(var_x, encoder_complexity * 2, **kwargs)
        var_x = self.blocks.conv(var_x, encoder_complexity * 4, **kwargs)
        var_x = self.blocks.conv(var_x, encoder_complexity * 6, **kwargs)
        var_x = self.blocks.conv(var_x, encoder_complexity * 8, **kwargs)

        var_x = Dense(self.encoder_dim,
                      kernel_initializer=self.kernel_initializer)(Flatten()(var_x))
        var_x = Dense(dense_shape * dense_shape * 384,
                      kernel_initializer=self.kernel_initializer)(var_x)
        var_x = Reshape((dense_shape, dense_shape, 384))(var_x)

        return KerasModel(input_, var_x)

    def decoder_a_lowmem(self):
        """ Decoder for side A """
        kwargs = dict(kernel_size=5, kernel_initializer=self.kernel_initializer)
        decoder_shape = self.input_shape[0] // 16

        decoder_complexity = 320

        input_ = Input(shape=(decoder_shape, decoder_shape, 384))

        var_x = input_

        var_x = self.blocks.upscale(var_x, decoder_complexity, **kwargs)
        var_x = SpatialDropout2D(0.25)(var_x)

        var_x = self.blocks.upscale(var_x, decoder_complexity // 2, **kwargs)
        var_x = SpatialDropout2D(0.15)(var_x)

        var_x = self.blocks.upscale(var_x, decoder_complexity // 4, **kwargs)

        var_x = self.blocks.upscale(var_x, decoder_complexity // 8, **kwargs)

        var_x = Conv2D(3, kernel_size=5, padding='same', activation='sigmoid')(var_x)
        return KerasModel(input_, var_x)

    def decoder_b_lowmem(self):
        """ Decoder for side B """
        kwargs = dict(kernel_size=5, kernel_initializer=self.kernel_initializer)
        decoder_shape = self.input_shape[0] // 16

        input_ = Input(shape=(decoder_shape, decoder_shape, 384))

        decoder_b_complexity = 384

        var_x = input_
        var_x = self.blocks.upscale(var_x, decoder_b_complexity, **kwargs)

        var_x = self.blocks.upscale(var_x, decoder_b_complexity // 2, **kwargs)

        var_x = self.blocks.upscale(var_x, decoder_b_complexity // 4, **kwargs)

        var_x = self.blocks.upscale(var_x, decoder_b_complexity // 8, **kwargs)

        var_x = Conv2D(3, kernel_size=5, padding='same', activation='sigmoid')(var_x)
        return KerasModel(input_, var_x)
