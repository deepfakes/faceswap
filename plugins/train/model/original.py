#!/usr/bin/env python3
""" Original Model
    Based on the original https://www.reddit.com/r/deepfakes/
    code sample + contribs """

from keras.layers import Dense, Flatten, Input, Reshape

from keras.models import Model as KerasModel

from lib.train.nn_blocks import conv, Conv2D, upscale
from ._base import ModelBase, logger


class Model(ModelBase):
    """ Original Faceswap Model """
    def __init__(self, *args, **kwargs):
        logger.debug("Initializing %s: (args: %s, kwargs: %s",
                     self.__class__.__name__, args, kwargs)
        if "image_shape" not in kwargs:
            kwargs["image_shape"] = (64, 64, 3)
        if "encoder_dim" not in kwargs:
            kwargs["encoder_dim"] = 1024

        super().__init__(*args, **kwargs)
        logger.debug("Initialized %s", self.__class__.__name__)

    def add_networks(self):
        """ Add the original model weights """
        logger.debug("Adding networks")
        self.add_network("decoder", "A", self.decoder())
        self.add_network("decoder", "B", self.decoder())
        self.add_network("encoder", None, self.encoder())
        logger.debug("Added networks")

    def initialize(self):
        """ Initialize original model """
        logger.debug("Initializing model")
        inp = Input(shape=self.image_shape)
        for network in self.networks:
            if network.type == "encoder":
                encoder = network.network
            elif network.type == "decoder" and network.side == "A":
                decoder_a = network.network
            elif network.type == "decoder" and network.side == "B":
                decoder_b = network.network

        self.autoencoders["a"] = KerasModel(inp, decoder_a(encoder(inp)))
        self.autoencoders["b"] = KerasModel(inp, decoder_b(encoder(inp)))

        self.log_summary("encoder", encoder)
        self.log_summary("decoder", decoder_a)

        self.compile_autoencoders()
        logger.debug("Initialized model")

    def encoder(self):
        """ Encoder Network """
        input_ = Input(shape=self.image_shape)
        inp = input_
        inp = conv(128)(inp)
        inp = conv(256)(inp)
        inp = conv(512)(inp)
        inp = conv(1024)(inp)
        inp = Dense(self.encoder_dim)(Flatten()(inp))
        inp = Dense(4 * 4 * 1024)(inp)
        inp = Reshape((4, 4, 1024))(inp)
        inp = upscale(512)(inp)
        return KerasModel(input_, inp)

    @staticmethod
    def decoder():
        """ Decoder Network """
        input_ = Input(shape=(8, 8, 512))
        inp = input_
        inp = upscale(256)(inp)
        inp = upscale(128)(inp)
        inp = upscale(64)(inp)
        inp = Conv2D(3,
                     kernel_size=5,
                     padding='same',
                     activation='sigmoid')(inp)
        return KerasModel(input_, inp)
