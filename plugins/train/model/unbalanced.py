#!/usr/bin/env python3
""" Unbalanced Model
    Based on the original https://www.reddit.com/r/deepfakes/
        code sample + contribs """

from keras.initializers import RandomNormal
from keras.layers import Dense, Flatten, Input, Reshape, SpatialDropout2D
from keras.models import Model as KerasModel

from .original import logger, Model as OriginalModel


class Model(OriginalModel):
    """ Unbalanced Faceswap Model """
    def __init__(self, *args, **kwargs):
        logger.debug("Initializing %s: (args: %s, kwargs: %s",
                     self.__class__.__name__, args, kwargs)

        self.configfile = kwargs.get("configfile", None)
        self.lowmem = self.config.get("lowmem", False)
        kwargs["input_shape"] = (self.config["input_size"], self.config["input_size"], 3)
        kwargs["encoder_dim"] = 512 if self.lowmem else self.config["nodes"]
        self.kernel_initializer = RandomNormal(0, 0.02)

        super().__init__(*args, **kwargs)
        logger.debug("Initialized %s", self.__class__.__name__)

    def add_networks(self):
        """ Add the original model weights """
        logger.debug("Adding networks")
        self.add_network("decoder", "a", self.decoder_a(), is_output=True)
        self.add_network("decoder", "b", self.decoder_b(), is_output=True)
        self.add_network("encoder", None, self.encoder())
        logger.debug("Added networks")

    def encoder(self):
        """ Unbalanced Encoder """
        kwargs = dict(kernel_initializer=self.kernel_initializer)
        encoder_complexity = 128 if self.lowmem else self.config["complexity_encoder"]
        dense_dim = 384 if self.lowmem else 512
        dense_shape = self.input_shape[0] // 16
        input_ = Input(shape=self.input_shape)

        var_x = input_
        var_x = self.blocks.conv(var_x, encoder_complexity, use_instance_norm=True, **kwargs)
        var_x = self.blocks.conv(var_x, encoder_complexity * 2, use_instance_norm=True, **kwargs)
        var_x = self.blocks.conv(var_x, encoder_complexity * 4, **kwargs)
        var_x = self.blocks.conv(var_x, encoder_complexity * 6, **kwargs)
        var_x = self.blocks.conv(var_x, encoder_complexity * 8, **kwargs)
        var_x = Dense(self.encoder_dim,
                      kernel_initializer=self.kernel_initializer)(Flatten()(var_x))
        var_x = Dense(dense_shape * dense_shape * dense_dim,
                      kernel_initializer=self.kernel_initializer)(var_x)
        var_x = Reshape((dense_shape, dense_shape, dense_dim))(var_x)
        return KerasModel(input_, var_x)

    def decoder_a(self):
        """ Decoder for side A """
        kwargs = dict(kernel_size=5, kernel_initializer=self.kernel_initializer)
        decoder_complexity = 320 if self.lowmem else self.config["complexity_decoder_a"]
        dense_dim = 384 if self.lowmem else 512
        decoder_shape = self.input_shape[0] // 16
        input_ = Input(shape=(decoder_shape, decoder_shape, dense_dim))

        var_x = input_

        var_x = self.blocks.upscale(var_x, decoder_complexity, **kwargs)
        var_x = SpatialDropout2D(0.25)(var_x)
        var_x = self.blocks.upscale(var_x, decoder_complexity, **kwargs)
        if self.lowmem:
            var_x = SpatialDropout2D(0.15)(var_x)
        else:
            var_x = SpatialDropout2D(0.25)(var_x)
        var_x = self.blocks.upscale(var_x, decoder_complexity // 2, **kwargs)
        var_x = self.blocks.upscale(var_x, decoder_complexity // 4, **kwargs)
        var_x = self.blocks.conv2d(var_x, 3,
                                   kernel_size=5,
                                   padding="same",
                                   activation="sigmoid",
                                   name="face_out")
        outputs = [var_x]

        if self.config.get("learn_mask", False):
            var_y = input_
            var_y = self.blocks.upscale(var_y, decoder_complexity)
            var_y = self.blocks.upscale(var_y, decoder_complexity)
            var_y = self.blocks.upscale(var_y, decoder_complexity // 2)
            var_y = self.blocks.upscale(var_y, decoder_complexity // 4)
            var_y = self.blocks.conv2d(var_y, 1,
                                       kernel_size=5,
                                       padding="same",
                                       activation="sigmoid",
                                       name="mask_out")
            outputs.append(var_y)
        return KerasModel(input_, outputs=outputs)

    def decoder_b(self):
        """ Decoder for side B """
        kwargs = dict(kernel_size=5, kernel_initializer=self.kernel_initializer)
        dense_dim = 384 if self.lowmem else self.config["complexity_decoder_b"]
        decoder_complexity = 384 if self.lowmem else 512
        decoder_shape = self.input_shape[0] // 16
        input_ = Input(shape=(decoder_shape, decoder_shape, dense_dim))

        var_x = input_
        if self.lowmem:
            var_x = self.blocks.upscale(var_x, decoder_complexity, **kwargs)
            var_x = self.blocks.upscale(var_x, decoder_complexity // 2, **kwargs)
            var_x = self.blocks.upscale(var_x, decoder_complexity // 4, **kwargs)
            var_x = self.blocks.upscale(var_x, decoder_complexity // 8, **kwargs)
        else:
            var_x = self.blocks.upscale(var_x, decoder_complexity,
                                        res_block_follows=True, **kwargs)
            var_x = self.blocks.res_block(var_x, decoder_complexity,
                                          kernel_initializer=self.kernel_initializer)
            var_x = self.blocks.upscale(var_x, decoder_complexity,
                                        res_block_follows=True, **kwargs)
            var_x = self.blocks.res_block(var_x, decoder_complexity,
                                          kernel_initializer=self.kernel_initializer)
            var_x = self.blocks.upscale(var_x, decoder_complexity // 2,
                                        res_block_follows=True, **kwargs)
            var_x = self.blocks.res_block(var_x, decoder_complexity // 2,
                                          kernel_initializer=self.kernel_initializer)
            var_x = self.blocks.upscale(var_x, decoder_complexity // 4, **kwargs)
        var_x = self.blocks.conv2d(var_x, 3,
                                   kernel_size=5,
                                   padding="same",
                                   activation="sigmoid",
                                   name="face_out")
        outputs = [var_x]

        if self.config.get("learn_mask", False):
            var_y = input_
            var_y = self.blocks.upscale(var_y, decoder_complexity)
            if not self.lowmem:
                var_y = self.blocks.upscale(var_y, decoder_complexity)
            var_y = self.blocks.upscale(var_y, decoder_complexity // 2)
            var_y = self.blocks.upscale(var_y, decoder_complexity // 4)
            if self.lowmem:
                var_y = self.blocks.upscale(var_y, decoder_complexity // 8)
            var_y = self.blocks.conv2d(var_y, 1,
                                       kernel_size=5,
                                       padding="same",
                                       activation="sigmoid",
                                       name="mask_out")
            outputs.append(var_y)
        return KerasModel(input_, outputs=outputs)
