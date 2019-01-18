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

        if not self.config.get("lowmem", False):
            encoder, decoder_a, decoder_b = self.build_standard()
        else:
            encoder, decoder_a, decoder_b = self.build_lowmem()

        self.add_network("decoder", "a", decoder_a())
        self.add_network("decoder", "b", decoder_b())
        self.add_network("encoder", None, encoder())

        logger.debug("Added networks")

    def build_lowmem(self):
        """ Build a low memory version """
        kwargs = dict(kernel_size=5, kernel_initializer=self.kernel_initializer)

        def encoder(self):
            dense_shape = self.input_shape[0] // 16
            input_ = Input(shape=self.input_shape)

            var_x = input_
            var_x = conv(var_x, 128, **kwargs)
            var_x = conv(var_x, 256, **kwargs)
            var_x = conv(var_x, 512, **kwargs)
            var_x = conv(var_x, 768, **kwargs)
            var_x = conv(var_x, 1024, **kwargs)
            var_x = Dense(self.encoder_dim,
                          kernel_initializer=self.kernel_initializer)(Flatten()(var_x))
            var_x = Dense(dense_shape * dense_shape * 384,
                          kernel_initializer=self.kernel_initializer)(var_x)
            var_x = Reshape((dense_shape, dense_shape, 384))(var_x)
            return KerasModel(input_, var_x)

        def decoder_a(self):
            """ Decoder for side A """
            decoder_shape = self.input_shape[0] // 16
            input_ = Input(shape=(decoder_shape, decoder_shape, 384))

            use_subpixel = self.config["subpixel_upscaling"]

            var_x = input_
            var_x = upscale(var_x, 384, use_subpixel=use_subpixel, **kwargs)
            var_x = SpatialDropout2D(0.25)(var_x)
            var_x = upscale(var_x, 256, use_subpixel=use_subpixel, **kwargs)
            var_x = SpatialDropout2D(0.15)(var_x)
            var_x = upscale(var_x, 256 // 2, use_subpixel=use_subpixel, **kwargs)
            var_x = upscale(var_x, 256 // 4, use_subpixel=use_subpixel, **kwargs)

            var_x = Conv2D(3, kernel_size=5, padding='same', activation='sigmoid')(var_x)
            return KerasModel(input_, var_x)

        def decoder_b(self):
            """ Decoder for side B """
            decoder_shape = self.input_shape[0] // 16
            input_ = Input(shape=(decoder_shape, decoder_shape, 384))

            use_subpixel = self.config["subpixel_upscaling"]

            var_x = input_
            var_x = upscale(var_x, 384, use_subpixel=use_subpixel, **kwargs)
            var_x = upscale(var_x, 384, use_subpixel=use_subpixel, **kwargs)
            var_x = upscale(var_x, 384 // 2, use_subpixel=use_subpixel, **kwargs)
            var_x = upscale(var_x, 384 // 4, use_subpixel=use_subpixel, **kwargs)

            var_x = Conv2D(3, kernel_size=5, padding='same', activation='sigmoid')(var_x)
            return KerasModel(input_, var_x)

        return encoder, decoder_a, decoder_b

    def build_standard(self):
        """ build a standard version """
        def encoder(self):
            kwargs = dict(kernel_size=5, kernel_initializer=self.kernel_initializer)
            encoder_complexity = self.config["complexity_encoder"]
            dense_shape = self.input_shape[0] // 16
            input_ = Input(shape=self.input_shape)

            var_x = input_
            var_x = conv(var_x, encoder_complexity, use_instance_norm=True, **kwargs)
            var_x = conv(var_x, encoder_complexity * 2, use_instance_norm=True, **kwargs)
            var_x = conv(var_x, encoder_complexity * 4, **kwargs)
            var_x = conv(var_x, encoder_complexity * 6, **kwargs)
            var_x = conv(var_x, encoder_complexity * 8, **kwargs)
            var_x = Dense(self.encoder_dim,
                          kernel_initializer=self.kernel_initializer)(Flatten()(var_x))
            var_x = Dense(dense_shape * dense_shape * 512,
                          kernel_initializer=self.kernel_initializer)(var_x)
            var_x = Reshape((dense_shape, dense_shape, 512))(var_x)
            return KerasModel(input_, var_x)

        def decoder_a(self):
            """ Decoder for side A """
            kwargs = dict(kernel_size=5, kernel_initializer=self.kernel_initializer)
            use_subpixel = self.config["subpixel_upscaling"]

            decoder_complexity = self.config["complexity_decoder_a"]
            decoder_shape = self.input_shape[0] // 16
            input_ = Input(shape=(decoder_shape, decoder_shape, 512))

            var_x = input_
            var_x = upscale(var_x, decoder_complexity, use_subpixel=use_subpixel, **kwargs)
            var_x = SpatialDropout2D(0.25)(var_x)
            var_x = upscale(var_x, decoder_complexity, use_subpixel=use_subpixel, **kwargs)
            var_x = SpatialDropout2D(0.25)(var_x)
            var_x = upscale(var_x, decoder_complexity // 2, use_subpixel=use_subpixel, **kwargs)
            var_x = upscale(var_x, decoder_complexity // 4, use_subpixel=use_subpixel, **kwargs)

            var_x = Conv2D(3, kernel_size=5, padding='same', activation='sigmoid')(var_x)
            return KerasModel(input_, var_x)

        def decoder_b(self):
            """ Decoder for side B """
            kwargs = dict(kernel_size=5, kernel_initializer=self.kernel_initializer)
            use_subpixel = self.config["subpixel_upscaling"]

            decoder_complexity = self.config["complexity_decoder_b"]
            decoder_shape = self.input_shape[0] // 16

            input_ = Input(shape=(decoder_shape, decoder_shape, decoder_complexity))

            var_x = input_
            var_x = upscale(var_x, 512, use_subpixel=use_subpixel, **kwargs)
            var_x = res_block(var_x, 512, kernel_initializer=self.kernel_initializer)
            var_x = upscale(var_x, 512, use_subpixel=use_subpixel, **kwargs)
            var_x = res_block(var_x, 512, kernel_initializer=self.kernel_initializer)
            var_x = upscale(var_x, 256, use_subpixel=use_subpixel, **kwargs)
            var_x = res_block(var_x, 256, kernel_initializer=self.kernel_initializer)
            var_x = upscale(var_x, 128, use_subpixel=use_subpixel, **kwargs)

            var_x = Conv2D(3, kernel_size=5, padding='same', activation='sigmoid')(var_x)
            return KerasModel(input_, var_x)

        return encoder, decoder_a, decoder_b
