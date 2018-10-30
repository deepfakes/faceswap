#!/usr/bin/env python3
""" Original - HiRes Model
    Based on the original https://www.reddit.com/r/deepfakes/
        code sample + contribs """


from keras.layers import Dense, Flatten, Input, Reshape, SeparableConv2D
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation
from keras.models import Model as KerasModel

from .original import Model as OriginalModel


class Model(OriginalModel):
    """ Original HiRes Faceswap Model """
    def __init__(self, *args, **kwargs):
        kwargs["image_shape"] = (128, 128, 3)
        super().__init__(*args, **kwargs)

    def add_networks(self):
        """ Add the original model weights """
        self.add_network("decoder", "A", self.decoder())
        self.add_network("decoder", "B", self.decoder())
        self.add_network("encoder", None, self.encoder())

    @staticmethod
    def conv_sep(filters):
        """ Seperable Convolution Layer """
        def block(inp):
            inp = SeparableConv2D(filters,
                                  kernel_size=5,
                                  strides=2,
                                  padding='same')(inp)
            inp = Activation("relu")(inp)
            return inp
        return block

    def encoder(self):
        """ Original HiRes Encoder """
        input_ = Input(shape=self.image_shape)
        inp = input_
        inp = self.conv(128)(inp)
        inp = self.conv_sep(256)(inp)
        inp = self.conv(512)(inp)
        inp = self.conv_sep(1024)(inp)
        inp = Dense(self.encoder_dim)(Flatten()(inp))
        inp = Dense(8 * 8 * 512)(inp)
        inp = Reshape((8, 8, 512))(inp)
        inp = self.upscale(512)(inp)
        return KerasModel(input_, inp)

    def decoder(self):
        """ Original HiRes Encoder """
        input_ = Input(shape=(16, 16, 512))
        inp = input_
        inp = self.upscale(384)(inp)
        inp = self.upscale(256-32)(inp)
        inp = self.upscale(128)(inp)
        inp = Conv2D(3,
                     kernel_size=5,
                     padding='same',
                     activation='sigmoid')(inp)
        return KerasModel(input_, inp)
