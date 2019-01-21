#!/usr/bin/env python3
""" Original - HiRes Model
    Based on the original https://www.reddit.com/r/deepfakes/
        code sample + contribs """

from keras.initializers import RandomNormal
from keras.layers import Dense, Flatten, Input, Reshape, SpatialDropout2D
from keras.layers.convolutional import Conv2D
from keras.models import Model as KerasModel

from lib.model.nn_blocks import conv, res_block, upscale
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
        self.add_network("decoder", "a", self.decoder_a())
        self.add_network("decoder", "b", self.decoder_b())
        self.add_network("encoder", None, self.encoder())
        logger.debug("Added networks")

    def encoder(self):
        """ Original HiRes Encoder """
        kwargs = dict(kernel_size=5, kernel_initializer=self.kernel_initializer)
        dense_shape = self.input_shape[0] // 16
        input_ = Input(shape=self.input_shape)

        if self.lowmem:
            encoder_complexity = 128
            dense_dim = 384
        else:
            encoder_complexity = self.config["complexity_encoder"]
            dense_dim = 512

        var_x = input_
        if self.lowmem:
            var_x = conv(var_x, encoder_complexity, **kwargs)
            var_x = conv(var_x, encoder_complexity * 2, **kwargs)
        else:
            var_x = conv(var_x, encoder_complexity, use_instance_norm=True, **kwargs)
            var_x = conv(var_x, encoder_complexity * 2, use_instance_norm=True, **kwargs)
        var_x = conv(var_x, encoder_complexity * 4, **kwargs)
        var_x = conv(var_x, encoder_complexity * 6, **kwargs)
        var_x = conv(var_x, encoder_complexity * 8, **kwargs)

        var_x = Dense(self.encoder_dim,
                      kernel_initializer=self.kernel_initializer)(Flatten()(var_x))
        var_x = Dense(dense_shape * dense_shape * dense_dim,
                      kernel_initializer=self.kernel_initializer)(var_x)
        var_x = Reshape((dense_shape, dense_shape, dense_dim))(var_x)
        return KerasModel(input_, var_x)

    def decoder_a(self):
        """ Decoder for side A """
        kwargs = dict(kernel_size=5, kernel_initializer=self.kernel_initializer)
        use_subpixel = self.config["subpixel_upscaling"]
        decoder_shape = self.input_shape[0] // 16

        if self.lowmem:
            decoder_complexity = 256
            inp_dim = 384
        else:
            decoder_complexity = self.config["complexity_decoder_a"]
            inp_dim = 512
        input_ = Input(shape=(decoder_shape, decoder_shape, inp_dim))

        var_x = input_
        if self.lowmem:
            var_x = upscale(var_x, inp_dim, use_subpixel=use_subpixel, **kwargs)
        else:
            var_x = upscale(var_x, decoder_complexity, use_subpixel=use_subpixel, **kwargs)
        var_x = SpatialDropout2D(0.25)(var_x)
        var_x = upscale(var_x, decoder_complexity, use_subpixel=use_subpixel, **kwargs)
        if self.lowmem:
            var_x = SpatialDropout2D(0.15)(var_x)
        else:
            var_x = SpatialDropout2D(0.25)(var_x)
        var_x = upscale(var_x, decoder_complexity // 2, use_subpixel=use_subpixel, **kwargs)
        var_x = upscale(var_x, decoder_complexity // 4, use_subpixel=use_subpixel, **kwargs)

        var_x = Conv2D(3, kernel_size=5, padding='same', activation='sigmoid')(var_x)
        return KerasModel(input_, var_x)

    def decoder_b(self):
        """ Decoder for side B """
        kwargs = dict(kernel_size=5, kernel_initializer=self.kernel_initializer)
        decoder_shape = self.input_shape[0] // 16
        use_subpixel = self.config["subpixel_upscaling"]

        if self.lowmem:
            inp_dim = 384
            dec_dim = 384
        else:
            inp_dim = self.config["complexity_decoder_b"]
            dec_dim = 512
        input_ = Input(shape=(decoder_shape, decoder_shape, inp_dim))

        var_x = input_
        var_x = upscale(var_x, dec_dim, use_subpixel=use_subpixel, **kwargs)
        if not self.lowmem:
            var_x = res_block(var_x, dec_dim, kernel_initializer=self.kernel_initializer)
        var_x = upscale(var_x, dec_dim, use_subpixel=use_subpixel, **kwargs)
        if not self.lowmem:
            var_x = res_block(var_x, dec_dim, kernel_initializer=self.kernel_initializer)
        var_x = upscale(var_x, dec_dim // 2, use_subpixel=use_subpixel, **kwargs)
        if not self.lowmem:
            var_x = res_block(var_x, dec_dim // 2, kernel_initializer=self.kernel_initializer)
        var_x = upscale(var_x, dec_dim // 4, use_subpixel=use_subpixel, **kwargs)

        var_x = Conv2D(3, kernel_size=5, padding='same', activation='sigmoid')(var_x)
        return KerasModel(input_, var_x)
