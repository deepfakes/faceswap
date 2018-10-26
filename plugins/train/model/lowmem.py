#!/usr/bin/env python3
""" Low Memory Original Model
    Based on the original https://www.reddit.com/r/deepfakes/
    code sample + contribs """

from keras.layers import Dense, Flatten, Input, Reshape
from keras.models import Model as KerasModel

from .original import Model as OriginalModel


class Model(OriginalModel):
    """ Low Memory version of Original Faceswap Model """
    def __init__(self, *args, **kwargs):
        kwargs["encoder_dim"] = 512
        super().__init__(*args, **kwargs)

    def add_networks(self):
        """ Add the original lowmem model weights """
        self.add_network("lowmem_decoder_A.h5", "decoder", "A",
                         self.decoder())
        self.add_network("lowmem_decoder_B.h5", "decoder", "B",
                         self.decoder())
        self.add_network("lowmem_encoder.h5", "encoder", None, self.encoder())

    def encoder(self):
        """ Encoder Network
            1 layer fewer for lowmem """
        input_ = Input(shape=self.image_shape)
        inp = input_
        inp = self.conv(128)(inp)
        inp = self.conv(256)(inp)
        inp = self.conv(512)(inp)
        inp = Dense(self.encoder_dim)(Flatten()(inp))
        inp = Dense(4 * 4 * 1024)(inp)
        inp = Reshape((4, 4, 1024))(inp)
        inp = self.upscale(512)(inp)
        return KerasModel(input_, inp)
