#!/usr/bin/env python3
""" Original Model
    Based on the original https://www.reddit.com/r/deepfakes/
    code sample + contribs """

from keras.layers import Dense, Flatten, Input, Reshape
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D
from keras.models import Model as KerasModel
from keras.optimizers import Adam
from keras.utils import multi_gpu_model

from lib.PixelShuffler import PixelShuffler

from ._base import ModelBase


class Model(ModelBase):
    """ Original Faceswap Model """
    def __init__(self, *args, **kwargs):
        self.autoencoder_a = None
        self.autoencoder_b = None

        if "image_shape" not in kwargs:
            kwargs["image_shape"] = (64, 64, 3)
        if "encoder_dim" not in kwargs:
            kwargs["encoder_dim"] = 1024

        super().__init__(*args, **kwargs)

    def add_networks(self):
        """ Add the original model weights """
        self.add_network("decoder_A.h5", "decoder", "A", self.decoder())
        self.add_network("decoder_B.h5", "decoder", "B", self.decoder())
        self.add_network("encoder.h5", "encoder", None, self.encoder())

    def initialize(self):
        """ Initialize original model """
        inp = Input(shape=self.image_shape)
        for network in self.networks:
            if network.type == "encoder":
                encoder = network.network
            elif network.type == "decoder" and network.side == "A":
                decoder_a = network.network
            elif network.type == "decoder" and network.side == "B":
                decoder_b = network.network

        self.autoencoder_a = KerasModel(inp, decoder_a(encoder(inp)))
        self.autoencoder_b = KerasModel(inp, decoder_b(encoder(inp)))
        self.compile_autoencoders()

    def compile_autoencoders(self):
        """ Compile the autoencoders """
        optimizer = Adam(lr=5e-5, beta_1=0.5, beta_2=0.999)
        if self.gpus > 1:
            self.autoencoder_a = multi_gpu_model(self.autoencoder_a, self.gpus)
            self.autoencoder_b = multi_gpu_model(self.autoencoder_b, self.gpus)

        self.autoencoder_a.compile(optimizer=optimizer,
                                   loss='mean_absolute_error')
        self.autoencoder_b.compile(optimizer=optimizer,
                                   loss='mean_absolute_error')

    def converter(self, swap):
        """ Load converter straight or swapped """
        autoencoder = self.autoencoder_a if not swap else self.autoencoder_b
        return lambda img: autoencoder.predict(img)

    @staticmethod
    def conv(filters):
        """ Convolution Layer"""
        def block(inp):
            inp = Conv2D(filters,
                         kernel_size=5,
                         strides=2,
                         padding='same')(inp)
            inp = LeakyReLU(0.1)(inp)
            return inp
        return block

    @staticmethod
    def upscale(filters):
        """ Updacale Layer """
        def block(inp):
            inp = Conv2D(filters * 4, kernel_size=3, padding='same')(inp)
            inp = LeakyReLU(0.1)(inp)
            inp = PixelShuffler()(inp)
            return inp
        return block

    def encoder(self):
        """ Encoder Network """
        input_ = Input(shape=self.image_shape)
        inp = input_
        inp = self.conv(128)(inp)
        inp = self.conv(256)(inp)
        inp = self.conv(512)(inp)
        inp = self.conv(1024)(inp)
        inp = Dense(self.encoder_dim)(Flatten()(inp))
        inp = Dense(4 * 4 * 1024)(inp)
        inp = Reshape((4, 4, 1024))(inp)
        inp = self.upscale(512)(inp)
        return KerasModel(input_, inp)

    def decoder(self):
        """ Decoder Network """
        input_ = Input(shape=(8, 8, 512))
        inp = input_
        inp = self.upscale(256)(inp)
        inp = self.upscale(128)(inp)
        inp = self.upscale(64)(inp)
        inp = Conv2D(3,
                     kernel_size=5,
                     padding='same',
                     activation='sigmoid')(inp)
        return KerasModel(input_, inp)
