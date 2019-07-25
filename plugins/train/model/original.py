#!/usr/bin/env python3
""" Original Model
    Based on the original https://www.reddit.com/r/deepfakes/
    code sample + contribs """

from keras.layers import Conv2D, Dense, Flatten, Input, Reshape, Concatenate

from keras.models import Model as KerasModel

from ._base import ModelBase, logger


class Model(ModelBase):
    """ Original Faceswap Model """
    def __init__(self, *args, **kwargs):
        logger.debug("Initializing %s: (args: %s, kwargs: %s",
                     self.__class__.__name__, args, kwargs)

        if "input_shape" not in kwargs:
            kwargs["input_shape"] = (64, 64, 3)
        if "encoder_dim" not in kwargs:
            kwargs["encoder_dim"] = 512 if self.config["lowmem"] else 1024

        super().__init__(*args, **kwargs)
        logger.debug("Initialized %s", self.__class__.__name__)

    def add_networks(self):
        """ Add the original model weights """
        logger.debug("Adding networks")
        self.add_network("decoder", "a", self.decoder())
        self.add_network("decoder", "b", self.decoder())
        self.add_network("encoder", None, self.encoder())
        logger.debug("Added networks")

    def build_autoencoders(self):
        """ Initialize original model """
        logger.debug("Initializing model")
        face = Input(shape=self.input_shape, name="face")
        mask = Input(shape=self.mask_shape, name="mask")
        inputs = [face, mask]

        for side in ("a", "b"):
            logger.debug("Adding Autoencoder. Side: %s", side)
            decoder = self.networks["decoder_{}".format(side)].network
            output = decoder(self.networks["encoder"].network(inputs))
            autoencoder = KerasModel(inputs, output)
            self.add_predictor(side, autoencoder)
        logger.debug("Initialized model")

    def encoder(self):
        """ Encoder Network """
        face_ = Input(shape=self.input_shape)
        mask_ = Input(shape=self.mask_shape)
        var_x = Concatenate(axis=-1)([face_, mask_])
        var_x = self.blocks.conv(var_x, 128)
        var_x = self.blocks.conv(var_x, 256)
        var_x = self.blocks.conv(var_x, 512)
        if not self.config.get("lowmem", False):
            var_x = self.blocks.conv(var_x, 1024)
        var_x = Flatten()(var_x)
        var_x = Dense(self.encoder_dim)(var_x)
        var_x = Dense(4 * 4 * 1024)(var_x)
        var_x = Reshape((4, 4, 1024))(var_x)
        var_x = self.blocks.upscale(var_x, 512)
        return KerasModel([face_, mask_], var_x)

    def decoder(self):
        """ Decoder Network """
        input_ = Input(shape=(8, 8, 512))
        var_x = input_
        var_x = self.blocks.upscale(var_x, 256)
        var_x = self.blocks.upscale(var_x, 128)
        var_x = self.blocks.upscale(var_x, 64)
        var_x = Conv2D(3, kernel_size=5, padding="same", activation="sigmoid")(var_x)
        outputs = [var_x]

        if self.config.get("mask_type", None):
            var_y = input_
            var_y = self.blocks.upscale(var_y, 256)
            var_y = self.blocks.upscale(var_y, 128)
            var_y = self.blocks.upscale(var_y, 64)
            var_y = Conv2D(1, kernel_size=5, padding='same', activation='sigmoid')(var_y)
            outputs.append(var_y)
        return KerasModel(input_, outputs=outputs)
