#!/usr/bin/env python3
""" Original Model
    Based on the original https://www.reddit.com/r/deepfakes/
    code sample + contribs """

from keras.layers import Dense, Flatten, Input, Reshape

from keras.models import Model as KerasModel

from lib.model.nn_blocks import conv, Conv2D, upscale
from ._base import get_config, ModelBase, logger


class Model(ModelBase):
    """ Original Faceswap Model """
    def __init__(self, *args, **kwargs):
        logger.debug("Initializing %s: (args: %s, kwargs: %s",
                     self.__class__.__name__, args, kwargs)

        if "input_shape" not in kwargs:
            kwargs["input_shape"] = (64, 64, 3)
        if "encoder_dim" not in kwargs:
            config = get_config(".".join(self.__module__.split(".")[-2:]))
            kwargs["encoder_dim"] = 512 if config["lowmem"] else 1024

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
        inp = Input(shape=self.input_shape)

        ae_a = KerasModel(
            inp,
            self.networks["decoder_a"].network(self.networks["encoder"].network(inp)))
        ae_b = KerasModel(
            inp,
            self.networks["decoder_b"].network(self.networks["encoder"].network(inp)))
        self.add_predictors(ae_a, ae_b)
        logger.debug("Initialized model")

    def encoder(self):
        """ Encoder Network """
        input_ = Input(shape=self.input_shape)
        var_x = input_
        var_x = conv(128)(var_x)
        var_x = conv(256)(var_x)
        var_x = conv(512)(var_x)
        if not self.config.get("lowmem", False):
            var_x = conv(1024)(var_x)
        var_x = Dense(self.encoder_dim)(Flatten()(var_x))
        var_x = Dense(4 * 4 * 1024)(var_x)
        var_x = Reshape((4, 4, 1024))(var_x)
        var_x = upscale(512)(var_x)
        return KerasModel(input_, var_x)

    @staticmethod
    def decoder():
        """ Decoder Network """
        input_ = Input(shape=(8, 8, 512))
        var_x = input_
        var_x = upscale(256)(var_x)
        var_x = upscale(128)(var_x)
        var_x = upscale(64)(var_x)
        var_x = Conv2D(3, kernel_size=5, padding="same", activation="sigmoid")(var_x)
        return KerasModel(input_, var_x)
