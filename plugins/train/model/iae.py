#!/usr/bin/env python3
""" Improved autoencoder for faceswap """

<<<<<<< HEAD
from keras.layers import Concatenate, Dense, Flatten, Input, Reshape
from keras.layers.convolutional import Conv2D
from keras.models import Model as KerasModel

from lib.model.nn_blocks import conv, upscale
=======
from keras.layers import Concatenate, Conv2D, Dense, Flatten, Input, Reshape
from keras.models import Model as KerasModel

>>>>>>> 60e0099c4d88a551b33592bf5126ab96bd5dc5ae
from ._base import ModelBase, logger


class Model(ModelBase):
    """ Improved Autoeencoder Model """
    def __init__(self, *args, **kwargs):
        logger.debug("Initializing %s: (args: %s, kwargs: %s",
                     self.__class__.__name__, args, kwargs)
        kwargs["input_shape"] = (64, 64, 3)
        kwargs["encoder_dim"] = 1024
        super().__init__(*args, **kwargs)
        logger.debug("Initialized %s", self.__class__.__name__)

    def add_networks(self):
        """ Add the IAE model weights """
        logger.debug("Adding networks")
        self.add_network("encoder", None, self.encoder())
        self.add_network("decoder", None, self.decoder())
<<<<<<< HEAD
        self.add_network("inter", "A", self.intermediate())
        self.add_network("inter", "B", self.intermediate())
        self.add_network("inter", None, self.intermediate())
        logger.debug("Added networks")

    def initialize(self):
        """ Initialize IAE model """
        logger.debug("Initializing model")
        inp = Input(shape=self.input_shape)
        decoder = self.networks["decoder"].network
        encoder = self.networks["encoder"].network
        inter_a = self.networks["inter_a"].network
        inter_b = self.networks["inter_b"].network
        inter_both = self.networks["inter"].network

        ae_a = decoder(Concatenate()([inter_a(encoder(inp)),
                                      inter_both(encoder(inp))]))
        ae_b = decoder(Concatenate()([inter_b(encoder(inp)),
                                      inter_both(encoder(inp))]))
        self.add_predictors(KerasModel(inp, ae_a), KerasModel(inp, ae_b))

=======
        self.add_network("intermediate", "a", self.intermediate())
        self.add_network("intermediate", "b", self.intermediate())
        self.add_network("inter", None, self.intermediate())
        logger.debug("Added networks")

    def build_autoencoders(self):
        """ Initialize IAE model """
        logger.debug("Initializing model")
        inputs = [Input(shape=self.input_shape, name="face")]
        if self.config.get("mask_type", None):
            mask_shape = (self.input_shape[:2] + (1, ))
            inputs.append(Input(shape=mask_shape, name="mask"))

        decoder = self.networks["decoder"].network
        encoder = self.networks["encoder"].network
        inter_both = self.networks["inter"].network
        for side in ("a", "b"):
            inter_side = self.networks["intermediate_{}".format(side)].network
            output = decoder(Concatenate()([inter_side(encoder(inputs[0])),
                                            inter_both(encoder(inputs[0]))]))

            autoencoder = KerasModel(inputs, output)
            self.add_predictor(side, autoencoder)
>>>>>>> 60e0099c4d88a551b33592bf5126ab96bd5dc5ae
        logger.debug("Initialized model")

    def encoder(self):
        """ Encoder Network """
        input_ = Input(shape=self.input_shape)
        var_x = input_
<<<<<<< HEAD
        var_x = conv(128)(var_x)
        var_x = conv(256)(var_x)
        var_x = conv(512)(var_x)
        var_x = conv(1024)(var_x)
=======
        var_x = self.blocks.conv(var_x, 128)
        var_x = self.blocks.conv(var_x, 256)
        var_x = self.blocks.conv(var_x, 512)
        var_x = self.blocks.conv(var_x, 1024)
>>>>>>> 60e0099c4d88a551b33592bf5126ab96bd5dc5ae
        var_x = Flatten()(var_x)
        return KerasModel(input_, var_x)

    def intermediate(self):
        """ Intermediate Network """
        input_ = Input(shape=(None, 4 * 4 * 1024))
        var_x = input_
        var_x = Dense(self.encoder_dim)(var_x)
        var_x = Dense(4 * 4 * int(self.encoder_dim/2))(var_x)
        var_x = Reshape((4, 4, int(self.encoder_dim/2)))(var_x)
        return KerasModel(input_, var_x)

    def decoder(self):
        """ Decoder Network """
        input_ = Input(shape=(4, 4, self.encoder_dim))
        var_x = input_
<<<<<<< HEAD
        var_x = upscale(512)(var_x)
        var_x = upscale(256)(var_x)
        var_x = upscale(128)(var_x)
        var_x = upscale(64)(var_x)
        var_x = Conv2D(3, kernel_size=5, padding="same", activation="sigmoid")(var_x)
        return KerasModel(input_, var_x)
=======
        var_x = self.blocks.upscale(var_x, 512)
        var_x = self.blocks.upscale(var_x, 256)
        var_x = self.blocks.upscale(var_x, 128)
        var_x = self.blocks.upscale(var_x, 64)
        var_x = Conv2D(3, kernel_size=5, padding="same", activation="sigmoid")(var_x)
        outputs = [var_x]

        if self.config.get("mask_type", None):
            var_y = Conv2D(1, kernel_size=5, padding="same", activation="sigmoid")(var_x)
            outputs.append(var_y)
        return KerasModel(input_, outputs=outputs)
>>>>>>> 60e0099c4d88a551b33592bf5126ab96bd5dc5ae
