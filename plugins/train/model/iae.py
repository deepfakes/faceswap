#!/usr/bin/env python3
""" Improved autoencoder for faceswap """

from keras.layers import Concatenate, Dense, Flatten, Input, Reshape
from keras.layers.convolutional import Conv2D
from keras.models import Model as KerasModel

from .original import Model as OriginalModel


class Model(OriginalModel):
    """ Improved Autoeencoder Model """
    def __init__(self, *args, **kwargs):
        kwargs["image_shape"] = (64, 64, 3)
        kwargs["encoder_dim"] = 1024
        super().__init__(*args, **kwargs)

    def add_networks(self):
        """ Add the IAE model weights """
        self.add_network("encoder", None, self.encoder())
        self.add_network("decoder", None, self.decoder())
        self.add_network("inter", "A", self.intermediate())
        self.add_network("inter", "B", self.intermediate())
        self.add_network("inter", None, self.intermediate())

    def initialize(self):
        """ Initialize IAE model """
        inp = Input(shape=self.image_shape)
        for network in self.networks:
            if network.type == "encoder":
                encoder = network.network
            elif network.type == "decoder":
                decoder = network.network
            elif network.type == "inter" and network.side == "A":
                inter_a = network.network
            elif network.type == "inter" and network.side == "B":
                inter_b = network.network
            elif network.type == "inter" and not network.side:
                inter_both = network.network

        output_a = decoder(Concatenate()([inter_a(encoder(inp)),
                                          inter_both(encoder(inp))]))
        output_b = decoder(Concatenate()([inter_b(encoder(inp)),
                                          inter_both(encoder(inp))]))
        self.autoencoders["a"] = KerasModel(inp, output_a)
        self.autoencoders["b"] = KerasModel(inp, output_b)
        self.compile_autoencoders()

    def encoder(self):
        """ Encoder Network """
        input_ = Input(shape=self.image_shape)
        inp = input_
        inp = self.conv(128)(inp)
        inp = self.conv(256)(inp)
        inp = self.conv(512)(inp)
        inp = self.conv(1024)(inp)
        inp = Flatten()(inp)
        return KerasModel(input_, inp)

    def intermediate(self):
        """ Intermediate Network """
        input_ = Input(shape=(None, 4 * 4 * 1024))
        inp = input_
        inp = Dense(self.encoder_dim)(inp)
        inp = Dense(4 * 4 * int(self.encoder_dim/2))(inp)
        inp = Reshape((4, 4, int(self.encoder_dim/2)))(inp)
        return KerasModel(input_, inp)

    def decoder(self):
        """ Decoder Network """
        input_ = Input(shape=(4, 4, self.encoder_dim))
        inp = input_
        inp = self.upscale(512)(inp)
        inp = self.upscale(256)(inp)
        inp = self.upscale(128)(inp)
        inp = self.upscale(64)(inp)
        inp = Conv2D(3,
                     kernel_size=5,
                     padding='same',
                     activation='sigmoid')(inp)
        return KerasModel(input_, inp)
