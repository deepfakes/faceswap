#!/usr/bin/env python3
""" Unbalanced Model
    Based on the original https://www.reddit.com/r/deepfakes/
        code sample + contributions """

from keras.initializers import RandomNormal
from keras.layers import Dense, Flatten, Input, Reshape, SpatialDropout2D

from lib.model.nn_blocks import Conv2DOutput, Conv2DBlock, ResidualBlock, UpscaleBlock
from ._base import ModelBase, KerasModel


class Model(ModelBase):
    """ Unbalanced Faceswap Model """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_shape = (self.config["input_size"], self.config["input_size"], 3)
        self.low_mem = self.config.get("lowmem", False)
        self.encoder_dim = 512 if self.low_mem else self.config["nodes"]
        self.kernel_initializer = RandomNormal(0, 0.02)

    def build_model(self, inputs):
        """ build the Unbalanced Model. """
        encoder = self.encoder()
        encoder_a = encoder(inputs[0])
        encoder_b = encoder(inputs[1])

        outputs = [self.decoder_a()(encoder_a), self.decoder_b()(encoder_b)]

        autoencoder = KerasModel(inputs, outputs, name=self.name)
        return autoencoder

    def encoder(self):
        """ Unbalanced Encoder """
        kwargs = dict(kernel_initializer=self.kernel_initializer)
        encoder_complexity = 128 if self.low_mem else self.config["complexity_encoder"]
        dense_dim = 384 if self.low_mem else 512
        dense_shape = self.input_shape[0] // 16
        input_ = Input(shape=self.input_shape)

        var_x = input_
        var_x = Conv2DBlock(encoder_complexity, use_instance_norm=True, **kwargs)(var_x)
        var_x = Conv2DBlock(encoder_complexity * 2, use_instance_norm=True, **kwargs)(var_x)
        var_x = Conv2DBlock(encoder_complexity * 4, **kwargs)(var_x)
        var_x = Conv2DBlock(encoder_complexity * 6, **kwargs)(var_x)
        var_x = Conv2DBlock(encoder_complexity * 8, **kwargs)(var_x)
        var_x = Dense(self.encoder_dim,
                      kernel_initializer=self.kernel_initializer)(Flatten()(var_x))
        var_x = Dense(dense_shape * dense_shape * dense_dim,
                      kernel_initializer=self.kernel_initializer)(var_x)
        var_x = Reshape((dense_shape, dense_shape, dense_dim))(var_x)
        return KerasModel(input_, var_x, name="encoder")

    def decoder_a(self):
        """ Decoder for side A """
        kwargs = dict(kernel_size=5, kernel_initializer=self.kernel_initializer)
        decoder_complexity = 320 if self.low_mem else self.config["complexity_decoder_a"]
        dense_dim = 384 if self.low_mem else 512
        decoder_shape = self.input_shape[0] // 16
        input_ = Input(shape=(decoder_shape, decoder_shape, dense_dim))

        var_x = input_

        var_x = UpscaleBlock(decoder_complexity, **kwargs)(var_x)
        var_x = SpatialDropout2D(0.25)(var_x)
        var_x = UpscaleBlock(decoder_complexity, **kwargs)(var_x)
        if self.low_mem:
            var_x = SpatialDropout2D(0.15)(var_x)
        else:
            var_x = SpatialDropout2D(0.25)(var_x)
        var_x = UpscaleBlock(decoder_complexity // 2, **kwargs)(var_x)
        var_x = UpscaleBlock(decoder_complexity // 4, **kwargs)(var_x)
        var_x = Conv2DOutput(3, 5, name="face_out_a")(var_x)
        outputs = [var_x]

        if self.config.get("learn_mask", False):
            var_y = input_
            var_y = UpscaleBlock(decoder_complexity)(var_y)
            var_y = UpscaleBlock(decoder_complexity)(var_y)
            var_y = UpscaleBlock(decoder_complexity // 2)(var_y)
            var_y = UpscaleBlock(decoder_complexity // 4)(var_y)
            var_y = Conv2DOutput(1, 5, name="mask_out_a")(var_y)
            outputs.append(var_y)
        return KerasModel(input_, outputs=outputs, name="decoder_a")

    def decoder_b(self):
        """ Decoder for side B """
        kwargs = dict(kernel_size=5, kernel_initializer=self.kernel_initializer)
        dense_dim = 384 if self.low_mem else self.config["complexity_decoder_b"]
        decoder_complexity = 384 if self.low_mem else 512
        decoder_shape = self.input_shape[0] // 16
        input_ = Input(shape=(decoder_shape, decoder_shape, dense_dim))

        var_x = input_
        if self.low_mem:
            var_x = UpscaleBlock(decoder_complexity, **kwargs)(var_x)
            var_x = UpscaleBlock(decoder_complexity // 2, **kwargs)(var_x)
            var_x = UpscaleBlock(decoder_complexity // 4, **kwargs)(var_x)
            var_x = UpscaleBlock(decoder_complexity // 8, **kwargs)(var_x)
        else:
            var_x = UpscaleBlock(decoder_complexity, res_block_follows=True, **kwargs)(var_x)
            var_x = ResidualBlock(decoder_complexity,
                                  kernel_initializer=self.kernel_initializer)(var_x)
            var_x = UpscaleBlock(decoder_complexity, res_block_follows=True, **kwargs)(var_x)
            var_x = ResidualBlock(decoder_complexity,
                                  kernel_initializer=self.kernel_initializer)(var_x)
            var_x = UpscaleBlock(decoder_complexity // 2, res_block_follows=True, **kwargs)(var_x)
            var_x = ResidualBlock(decoder_complexity // 2,
                                  kernel_initializer=self.kernel_initializer)(var_x)
            var_x = UpscaleBlock(decoder_complexity // 4, **kwargs)(var_x)
        var_x = Conv2DOutput(3, 5, name="face_out_b")(var_x)
        outputs = [var_x]

        if self.config.get("learn_mask", False):
            var_y = input_
            var_y = UpscaleBlock(decoder_complexity)(var_y)
            if not self.low_mem:
                var_y = UpscaleBlock(decoder_complexity)(var_y)
            var_y = UpscaleBlock(decoder_complexity // 2)(var_y)
            var_y = UpscaleBlock(decoder_complexity // 4)(var_y)
            if self.low_mem:
                var_y = UpscaleBlock(decoder_complexity // 8)(var_y)
            var_y = Conv2DOutput(1, 5, name="mask_out_b")(var_y)
            outputs.append(var_y)
        return KerasModel(input_, outputs=outputs, name="decoder_b")

    def _legacy_mapping(self):
        """ The mapping of legacy separate model names to single model names """
        return {"{}_encoder.h5".format(self.name): "encoder",
                "{}_decoder_A.h5".format(self.name): "decoder_a",
                "{}_decoder_B.h5".format(self.name): "decoder_b"}
