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

from plugins.model._base import ModelBase



class Model(ModelBase):
    """ Original Faceswap Model """
    def __init__(self, *args, **kwargs):
        model_data = [self.get_model("decoder_A.h5", "decoder",
                                     "A", self.decoder()),
                      self.get_model("decoder_B.h5", "decoder",
                                     "B", self.decoder()),
                      self.get_model("encoder.h5", "encoder",
                                     None, self.encoder())]
        self.image_shape = (64, 64, 3)
        self.encoder_dim = 1024
        self.autoencoder_a = None
        self.autoencoder_b = None
        super().__init__(*args, **kwargs, model_data=model_data)

    def initModel(self):
        optimizer = Adam(lr=5e-5, beta_1=0.5, beta_2=0.999)
        inp = Input(shape=self.image_shape)

        for model in self.models:
            if model.type == "encoder":
                m_encoder = model.model
            elif model.type == "decoder" and model.side == "A":
                m_decoder_a = model.model
            elif model.type == "decoder" and model.side == "B":
                m_decoder_b = model.model

        self.autoencoder_a = KerasModel(inp, m_decoder_a(m_encoder(inp)))
        self.autoencoder_a = KerasModel(inp, m_decoder_b(m_encoder(inp)))

        if self.gpus > 1:
            self.autoencoder_a = multi_gpu_model(self.autoencoder_a, self.gpus)
            self.autoencoder_b = multi_gpu_model(self.autoencoder_b, self.gpus)

        self.autoencoder_a.compile(optimizer=optimizer,
                                   loss='mean_absolute_error')
        self.autoencoder_a.compile(optimizer=optimizer,
                                   loss='mean_absolute_error')

    def converter(self, swap):
        """ Return converter or swapped converter """
        autoencoder = self.autoencoder_b if not swap else self.autoencoder_a
        return lambda img: autoencoder.predict(img)

    @staticmethod
    def conv(filters):
        """ Convolution """
        def block(inp):
            inp = Conv2D(filters, kernel_size=5, strides=2, padding='same')(inp)
            inp = LeakyReLU(0.1)(inp)
            return inp
        return block

    @staticmethod
    def upscale(filters):
        """ Updacaler """
        def block(inp):
            inp = Conv2D(filters * 4, kernel_size=3, padding='same')(inp)
            inp = LeakyReLU(0.1)(inp)
            inp = PixelShuffler()(inp)
            return inp
        return block

    def encoder(self):
        """ Encoder """
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
        """ Decoder """
        input_ = Input(shape=(8, 8, 512))
        inp = input_
        inp = self.upscale(256)(inp)
        inp = self.upscale(128)(inp)
        inp = self.upscale(64)(inp)
        inp = Conv2D(3, kernel_size=5, padding='same', activation='sigmoid')(inp)
        return KerasModel(input_, inp)
