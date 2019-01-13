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
        encoder_complexity = self.config["complexity_encoder"]
        dense_shape = self.input_shape[0] // 16
        kwargs = {"kernel_size": 5,
                  "strides": 2,
                  "kernel_initializer": self.kernel_initializer}
        input_ = Input(shape=self.input_shape)

        var_x = input_
        var_x = conv(encoder_complexity, use_instance_norm=True, **kwargs)(var_x)
        var_x = conv(encoder_complexity * 2, use_instance_norm=True, **kwargs)(var_x)
        var_x = conv(encoder_complexity * 4, **kwargs)(var_x)
        var_x = conv(encoder_complexity * 6, **kwargs)(var_x)
        var_x = conv(encoder_complexity * 8, **kwargs)(var_x)
        var_x = Dense(self.encoder_dim,
                      kernel_initializer=self.kernel_initializer)(Flatten()(var_x))
        var_x = Dense(dense_shape * dense_shape * 512,
                      kernel_initializer=self.kernel_initializer)(var_x)
        var_x = Reshape((dense_shape, dense_shape, 512))(var_x)
        return KerasModel(input_, var_x)

    def decoder_a(self):
        """ Decoder for side A """
        decoder_complexity = self.config["complexity_decoder_a"]
        decoder_shape = self.input_shape[0] // 16
        kwargs = {"kernel_size": 5,
                  "kernel_initializer": self.kernel_initializer}
        use_subpixel = self.config["subpixel_upscaling"]
        input_ = Input(shape=(decoder_shape, decoder_shape, 512))

        var_x = input_
        var_x = upscale(decoder_complexity, use_subpixel=use_subpixel, **kwargs)(var_x)
        var_x = SpatialDropout2D(0.25)(var_x)
        var_x = upscale(decoder_complexity, use_subpixel=use_subpixel, **kwargs)(var_x)
        var_x = SpatialDropout2D(0.25)(var_x)
        var_x = upscale(decoder_complexity // 2, use_subpixel=use_subpixel, **kwargs)(var_x)
        var_x = upscale(decoder_complexity // 4, use_subpixel=use_subpixel, **kwargs)(var_x)
        var_x = Conv2D(3, kernel_size=5, padding='same', activation='sigmoid')(var_x)
        return KerasModel(input_, var_x)

    def decoder_b(self):
        """ Decoder for side B """
        decoder_complexity = self.config["complexity_decoder_b"]
        decoder_shape = self.input_shape[0] // 16
        kwargs = {"kernel_size": 5,
                  "kernel_initializer": self.kernel_initializer}
        use_subpixel = self.config["subpixel_upscaling"]
        input_ = Input(shape=(decoder_shape, decoder_shape, decoder_complexity))

        var_x = input_
        var_x = upscale(512, use_subpixel=use_subpixel, **kwargs)(var_x)
        var_x = res_block(var_x, 512, kernel_initializer=self.kernel_initializer)
        var_x = upscale(512, use_subpixel=use_subpixel, **kwargs)(var_x)
        var_x = res_block(var_x, 512, kernel_initializer=self.kernel_initializer)
        var_x = upscale(256, use_subpixel=use_subpixel, **kwargs)(var_x)
        var_x = res_block(var_x, 256, kernel_initializer=self.kernel_initializer)
        var_x = upscale(128, use_subpixel=use_subpixel, **kwargs)(var_x)
        var_x = Conv2D(3, kernel_size=5, padding='same', activation='sigmoid')(var_x)
        return KerasModel(input_, var_x)
