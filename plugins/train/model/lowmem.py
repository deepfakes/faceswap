#!/usr/bin/env python3
""" Low Memory Original Model
    Based on the original https://www.reddit.com/r/deepfakes/
    code sample + contribs """

from keras.layers import Dense, Flatten, Input, Reshape
from keras.models import Model as KerasModel

from lib.train.nn_blocks import conv, upscale
from .original import logger, Model as OriginalModel


class Model(OriginalModel):
    """ Low Memory version of Original Faceswap Model """
    def __init__(self, *args, **kwargs):
        logger.debug("Initializing %s: (args: %s, kwargs: %s",
                     self.__class__.__name__, args, kwargs)
        kwargs["encoder_dim"] = 512
        super().__init__(*args, **kwargs)
        logger.debug("Initialized %s", self.__class__.__name__)

    def add_networks(self):
        """ Add the original lowmem model weights """
        logger.debug("Adding networks")
        self.add_network("decoder", "A", self.decoder())
        self.add_network("decoder", "B", self.decoder())
        self.add_network("encoder", None, self.encoder())
        logger.debug("Added networks")

    def encoder(self):
        """ Encoder Network
            1 layer fewer for lowmem """
        input_ = Input(shape=self.image_shape)
        var_x = input_
        var_x = conv(128)(var_x)
        var_x = conv(256)(var_x)
        var_x = conv(512)(var_x)
        var_x = Dense(self.encoder_dim)(Flatten()(var_x))
        var_x = Dense(4 * 4 * 1024)(var_x)
        var_x = Reshape((4, 4, 1024))(var_x)
        var_x = upscale(512)(var_x)
        return KerasModel(input_, var_x)
