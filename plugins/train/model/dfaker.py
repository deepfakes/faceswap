#!/usr/bin/env python3
""" DFaker Model
    Based on the dfaker model: https://github.com/dfaker """


from keras.initializers import RandomNormal
from keras.layers import Conv2D, Input
from keras.models import Model as KerasModel

from .original import logger, Model as OriginalModel


class Model(OriginalModel):
    """ Improved Autoeencoder Model """
    def __init__(self, *args, **kwargs):
        logger.debug("Initializing %s: (args: %s, kwargs: %s",
                     self.__class__.__name__, args, kwargs)
        kwargs["input_shape"] = (64, 64, 3)
        kwargs["encoder_dim"] = 1024
        self.kernel_initializer = RandomNormal(0, 0.02)
        super().__init__(*args, **kwargs)
        logger.debug("Initialized %s", self.__class__.__name__)

    def build_autoencoders(self):
        """ Initialize Dfaker model """
        logger.debug("Initializing model")
        inputs = [Input(shape=self.input_shape, name="face")]
        if self.config.get("mask_type", None):
            mask_shape = (self.input_shape[0] * 2, self.input_shape[1] * 2, 1)
            inputs.append(Input(shape=mask_shape, name="mask"))

        for side in ("a", "b"):
            decoder = self.networks["decoder_{}".format(side)].network
            output = decoder(self.networks["encoder"].network(inputs[0]))
            autoencoder = KerasModel(inputs, output)
            self.add_predictor(side, autoencoder)
        logger.debug("Initialized model")

    def decoder(self):
        """ Decoder Network """
        input_ = Input(shape=(8, 8, 512))
        var_x = input_

        var_x = self.blocks.upscale(var_x, 512, res_block_follows=True)
        var_x = self.blocks.res_block(var_x, 512, kernel_initializer=self.kernel_initializer)
        var_x = self.blocks.upscale(var_x, 256, res_block_follows=True)
        var_x = self.blocks.res_block(var_x, 256, kernel_initializer=self.kernel_initializer)
        var_x = self.blocks.upscale(var_x, 128, res_block_follows=True)
        var_x = self.blocks.res_block(var_x, 128, kernel_initializer=self.kernel_initializer)
        var_x = self.blocks.upscale(var_x, 64)
        var_x = Conv2D(3, kernel_size=5, padding='same', activation='sigmoid')(var_x)
        outputs = [var_x]

        if self.config.get("mask_type", None):
            var_y = input_
            var_y = self.blocks.upscale(var_y, 512)
            var_y = self.blocks.upscale(var_y, 256)
            var_y = self.blocks.upscale(var_y, 128)
            var_y = self.blocks.upscale(var_y, 64)
            var_y = Conv2D(1, kernel_size=5, padding='same', activation='sigmoid')(var_y)
            outputs.append(var_y)
        return KerasModel([input_], outputs=outputs)
