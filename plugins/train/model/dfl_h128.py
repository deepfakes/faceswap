#!/usr/bin/env python3
""" DeepFakesLab H128 Model
    Based on https://github.com/iperov/DeepFaceLab
"""

from keras.layers import Dense, Flatten, Input, Reshape
from keras.layers.convolutional import Conv2D
from keras.models import Model as KerasModel

from lib.model.nn_blocks import conv, upscale
from .original import logger, Model as OriginalModel

# TODO Check whether using DFL Loss function rather than DFaker makes a difference


class Model(OriginalModel):
    """ Low Memory version of Original Faceswap Model """
    def __init__(self, *args, **kwargs):
        logger.debug("Initializing %s: (args: %s, kwargs: %s",
                     self.__class__.__name__, args, kwargs)
        kwargs["image_shape"] = (128, 128, 3)
        kwargs["encoder_dim"] = 256 if self.config["lowmem"] else 512

        super().__init__(*args, **kwargs)
        logger.debug("Initialized %s", self.__class__.__name__)

    def initialize(self):
        """ Initialize DFL H128 model """
        logger.debug("Initializing model")
        mask_shape = self.image_shape[:2] + (1, )
        inp_a = Input(shape=self.image_shape)
        mask_a = Input(shape=mask_shape)
        inp_b = Input(shape=self.image_shape)
        mask_b = Input(shape=mask_shape)

        ae_a = KerasModel(
            [inp_a, mask_a],
            self.networks["decoder_a"].network(self.networks["encoder"].network(inp_a)))

        ae_b = KerasModel(
            [inp_b, mask_b],
            self.networks["decoder_b"].network(self.networks["encoder"].network(inp_b)))

        self.add_predictors(ae_a, ae_b)
        self.masks = [mask_a, mask_b]
        logger.debug("Initialized model")

    def encoder(self):
        """ DFL H128 Encoder """
        input_ = Input(shape=self.image_shape)
        var_x = input_
        var_x = conv(128)(var_x)
        var_x = conv(256)(var_x)
        var_x = conv(512)(var_x)
        var_x = conv(1024)(var_x)
        var_x = Dense(self.encoder_dim)(Flatten()(var_x))
        var_x = Dense(8 * 8 * self.encoder_dim)(var_x)
        var_x = Reshape((8, 8, self.encoder_dim))(var_x)
        var_x = upscale(self.encoder_dim)(var_x)
        return KerasModel(input_, var_x)

    def decoder(self):
        """ DFL H128 Decoder """
        input_ = Input(shape=(16, 16, self.encoder_dim))
        var = input_
        var = upscale(self.encoder_dim)(var)
        var = upscale(self.encoder_dim // 2)(var)
        var = upscale(self.encoder_dim // 4)(var)

        # Face
        var_x = Conv2D(3, kernel_size=5, padding="same", activation="sigmoid")(var)
        # Mask
        var_y = Conv2D(1, kernel_size=5, padding="same", activation="sigmoid")(var)
        return KerasModel(input_, [var_x, var_y])
