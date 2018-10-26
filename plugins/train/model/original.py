#!/usr/bin/env python3
""" Based on the original https://www.reddit.com/r/deepfakes/
    code sample + contribs """

from keras.models import Model as KerasModel
from keras.layers import Input, Dense, Flatten, Reshape
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D
from keras.optimizers import Adam
from keras.utils import multi_gpu_model

from lib.PixelShuffler import PixelShuffler

from ._base import ModelBase


class Model(ModelBase):
    """ Original Faceswap Model """
    def __init__(self, *args, **kwargs):
        image_shape = (64, 64, 3)
        encoder_dim = 1024
        self.autoencoder_a = None
        self.autoencoder_b = None
        super().__init__(*args, **kwargs,
                         image_shape=image_shape,
                         encoder_dim=encoder_dim)

    def add_networks(self):
        """ Add the original model weights """
        self.add_network("decoder_A.h5", "decoder", "A", self.decoder())
        self.add_network("decoder_B.h5", "decoder", "B", self.decoder())
        self.add_network("encoder.h5", "encoder", None, self.encoder())

    def initialize(self):
        """ Initialize original model """
        optimizer = Adam(lr=5e-5, beta_1=0.5, beta_2=0.999)
        inp = Input(shape=self.image_shape)

        for network in self.networks:
            if network.type == "encoder":
                m_encoder = network.network
            elif network.type == "decoder" and network.side == "A":
                m_decoder_a = network.network
            elif network.type == "decoder" and network.side == "B":
                m_decoder_b = network.network

        self.autoencoder_a = KerasModel(inp, m_decoder_a(m_encoder(inp)))
        self.autoencoder_b = KerasModel(inp, m_decoder_b(m_encoder(inp)))

        if self.gpus > 1:
            self.autoencoder_a = multi_gpu_model(self.autoencoder_a, self.gpus)
            self.autoencoder_b = multi_gpu_model(self.autoencoder_b, self.gpus)

        self.autoencoder_a.compile(optimizer=optimizer,
                                   loss='mean_absolute_error')
        self.autoencoder_b.compile(optimizer=optimizer,
                                   loss='mean_absolute_error')

    def converter(self, swap):
        """ Return converter or swapped converter """
        autoencoder = self.autoencoder_b if not swap else self.autoencoder_a
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
