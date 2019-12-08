#!/usr/bin/env python3
""" DeepFakesLab SAE Model
    Based on https://github.com/iperov/DeepFaceLab
"""

import numpy as np

from keras.layers import Concatenate, Dense, Flatten, Input, Reshape
from keras.models import Model as KerasModel

from ._base import ModelBase, logger


class Model(ModelBase):
    """ Low Memory version of Original Faceswap Model """
    def __init__(self, *args, **kwargs):
        logger.debug("Initializing %s: (args: %s, kwargs: %s",
                     self.__class__.__name__, args, kwargs)

        self.configfile = kwargs.get("configfile", None)
        kwargs["input_shape"] = (self.config["input_size"], self.config["input_size"], 3)

        super().__init__(*args, **kwargs)
        logger.debug("Initialized %s", self.__class__.__name__)

    @property
    def architecture(self):
        """ Return the architecture used from config """
        return self.config["architecture"].lower()

    @property
    def use_mask(self):
        """ Return True if a mask has been set else false """
        return self.config.get("learn_mask", False)

    @property
    def ae_dims(self):
        """ Set the Autoencoder Dimensions or set to default """
        retval = self.config["autoencoder_dims"]
        if retval == 0:
            retval = 256 if self.architecture == "liae" else 512
        return retval

    @property
    def multiscale_count(self):
        """ Return 3 if multiscale decoder is set else 1 """
        retval = 3 if self.config["multiscale_decoder"] else 1
        return retval

    def add_networks(self):
        """ Add the DFL SAE Networks """
        logger.debug("Adding networks")
        # Encoder
        self.add_network("encoder", None, getattr(self, "encoder_{}".format(self.architecture))())

        # Intermediate
        if self.architecture == "liae":
            self.add_network("intermediate", "b", self.inter_liae())
            self.add_network("intermediate", None, self.inter_liae())

        # Decoder
        decoder_sides = [None] if self.architecture == "liae" else ["a", "b"]
        for side in decoder_sides:
            self.add_network("decoder", side, self.decoder(), is_output=True)
        logger.debug("Added networks")

    def build_autoencoders(self, inputs):
        """ Initialize DFL SAE model """
        logger.debug("Initializing model")
        getattr(self, "build_{}_autoencoder".format(self.architecture))(inputs)
        logger.debug("Initialized model")

    def build_liae_autoencoder(self, inputs):
        """ Build the LIAE Autoencoder """
        for side in ("a", "b"):
            encoder = self.networks["encoder"].network(inputs[0])
            if side == "a":
                intermediate = Concatenate()([self.networks["intermediate"].network(encoder),
                                              self.networks["intermediate"].network(encoder)])
            else:
                intermediate = Concatenate()([self.networks["intermediate_b"].network(encoder),
                                              self.networks["intermediate"].network(encoder)])
            output = self.networks["decoder"].network(intermediate)
            autoencoder = KerasModel(inputs, output)
            self.add_predictor(side, autoencoder)

    def build_df_autoencoder(self, inputs):
        """ Build the DF Autoencoder """
        for side in ("a", "b"):
            logger.debug("Adding Autoencoder. Side: %s", side)
            decoder = self.networks["decoder_{}".format(side)].network
            output = decoder(self.networks["encoder"].network(inputs[0]))
            autoencoder = KerasModel(inputs, output)
            self.add_predictor(side, autoencoder)

    def encoder_df(self):
        """ DFL SAE DF Encoder Network"""
        input_ = Input(shape=self.input_shape)
        dims = self.input_shape[-1] * self.config["encoder_dims"]
        lowest_dense_res = self.input_shape[0] // 16
        var_x = input_
        var_x = self.blocks.conv(var_x, dims)
        var_x = self.blocks.conv(var_x, dims * 2)
        var_x = self.blocks.conv(var_x, dims * 4)
        var_x = self.blocks.conv(var_x, dims * 8)
        var_x = Dense(self.ae_dims)(Flatten()(var_x))
        var_x = Dense(lowest_dense_res * lowest_dense_res * self.ae_dims)(var_x)
        var_x = Reshape((lowest_dense_res, lowest_dense_res, self.ae_dims))(var_x)
        var_x = self.blocks.upscale(var_x, self.ae_dims)
        return KerasModel(input_, var_x)

    def encoder_liae(self):
        """ DFL SAE LIAE Encoder Network """
        input_ = Input(shape=self.input_shape)
        dims = self.input_shape[-1] * self.config["encoder_dims"]
        var_x = input_
        var_x = self.blocks.conv(var_x, dims)
        var_x = self.blocks.conv(var_x, dims * 2)
        var_x = self.blocks.conv(var_x, dims * 4)
        var_x = self.blocks.conv(var_x, dims * 8)
        var_x = Flatten()(var_x)
        return KerasModel(input_, var_x)

    def inter_liae(self):
        """ DFL SAE LIAE Intermediate Network """
        input_ = Input(shape=self.networks["encoder"].output_shapes[0][1:])
        lowest_dense_res = self.input_shape[0] // 16
        var_x = input_
        var_x = Dense(self.ae_dims)(var_x)
        var_x = Dense(lowest_dense_res * lowest_dense_res * self.ae_dims * 2)(var_x)
        var_x = Reshape((lowest_dense_res, lowest_dense_res, self.ae_dims * 2))(var_x)
        var_x = self.blocks.upscale(var_x, self.ae_dims * 2)
        return KerasModel(input_, var_x)

    def decoder(self):
        """ DFL SAE Decoder Network"""
        if self.architecture == "liae":
            input_shape = np.array(self.networks["intermediate"].output_shapes[0][1:]) * (1, 1, 2)
        else:
            input_shape = self.networks["encoder"].output_shapes[0][1:]
        input_ = Input(shape=input_shape)
        outputs = list()

        dims = self.input_shape[-1] * self.config["decoder_dims"]
        var_x = input_

        var_x1 = self.blocks.upscale(var_x, dims * 8, res_block_follows=True)
        var_x1 = self.blocks.res_block(var_x1, dims * 8)
        var_x1 = self.blocks.res_block(var_x1, dims * 8)
        if self.multiscale_count >= 3:
            outputs.append(self.blocks.conv2d(var_x1, 3,
                                              kernel_size=5,
                                              padding="same",
                                              activation="sigmoid",
                                              name="face_out_32"))

        var_x2 = self.blocks.upscale(var_x1, dims * 4, res_block_follows=True)
        var_x2 = self.blocks.res_block(var_x2, dims * 4)
        var_x2 = self.blocks.res_block(var_x2, dims * 4)
        if self.multiscale_count >= 2:
            outputs.append(self.blocks.conv2d(var_x2, 3,
                                              kernel_size=5,
                                              padding="same",
                                              activation="sigmoid",
                                              name="face_out_64"))

        var_x3 = self.blocks.upscale(var_x2, dims * 2, res_block_follows=True)
        var_x3 = self.blocks.res_block(var_x3, dims * 2)
        var_x3 = self.blocks.res_block(var_x3, dims * 2)

        outputs.append(self.blocks.conv2d(var_x3, 3,
                                          kernel_size=5,
                                          padding="same",
                                          activation="sigmoid",
                                          name="face_out_128"))

        if self.use_mask:
            var_y = input_
            var_y = self.blocks.upscale(var_y, self.config["decoder_dims"] * 8)
            var_y = self.blocks.upscale(var_y, self.config["decoder_dims"] * 4)
            var_y = self.blocks.upscale(var_y, self.config["decoder_dims"] * 2)
            var_y = self.blocks.conv2d(var_y, 1,
                                       kernel_size=5,
                                       padding="same",
                                       activation="sigmoid",
                                       name="mask_out")
            outputs.append(var_y)
        return KerasModel(input_, outputs=outputs)
