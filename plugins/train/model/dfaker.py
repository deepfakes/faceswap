#!/usr/bin/env python3
""" DFaker Model
    Based on the dfaker model: https://github.com/dfaker """


<<<<<<< HEAD
from keras.layers import Input
from keras.models import Model as KerasModel

from lib.model.nn_blocks import Conv2D, res_block, upscale

=======
from keras.initializers import RandomNormal
from keras.layers import Conv2D, Input
from keras.models import Model as KerasModel

>>>>>>> 60e0099c4d88a551b33592bf5126ab96bd5dc5ae
from .original import logger, Model as OriginalModel


class Model(OriginalModel):
    """ Improved Autoeencoder Model """
    def __init__(self, *args, **kwargs):
        logger.debug("Initializing %s: (args: %s, kwargs: %s",
                     self.__class__.__name__, args, kwargs)
        kwargs["input_shape"] = (64, 64, 3)
        kwargs["encoder_dim"] = 1024
<<<<<<< HEAD
        kwargs["trainer"] = "dfaker"
        super().__init__(*args, **kwargs)
        logger.debug("Initialized %s", self.__class__.__name__)

    def set_training_data(self):
        """ Set the dictionary for training """
        logger.debug("Setting training data")
        training_opts = dict()
        training_opts["serializer"] = self.config["alignments_format"]
        training_opts["mask_type"] = self.config["mask_type"]
        training_opts["full_face"] = True
        training_opts["preview_images"] = 10
        logger.debug("Set training data: %s", training_opts)
        return training_opts

    def initialize(self):
        """ Initialize Dfaker model """
        logger.debug("Initializing model")
        mask_shape = (self.input_shape[0] * 2, self.input_shape[1] * 2, 1)
        inp_a = Input(shape=self.input_shape)
        mask_a = Input(shape=mask_shape)
        inp_b = Input(shape=self.input_shape)
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

    @staticmethod
    def decoder():
        """ Decoder Network """
        input_ = Input(shape=(8, 8, 512))
        inp_x = input_
        inp_y = input_

        inp_x = upscale(512)(inp_x)
        inp_x = res_block(inp_x, 512)
        inp_x = upscale(256)(inp_x)
        inp_x = res_block(inp_x, 256)
        inp_x = upscale(128)(inp_x)
        inp_x = res_block(inp_x, 128)
        inp_x = upscale(64)(inp_x)
        inp_x = Conv2D(3,
                       kernel_size=5,
                       padding='same',
                       activation='sigmoid')(inp_x)

        inp_y = upscale(512)(inp_y)
        inp_y = upscale(256)(inp_y)
        inp_y = upscale(128)(inp_y)
        inp_y = upscale(64)(inp_y)
        inp_y = Conv2D(1,
                       kernel_size=5,
                       padding='same',
                       activation='sigmoid')(inp_y)

        return KerasModel([input_], outputs=[inp_x, inp_y])
=======
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
>>>>>>> 60e0099c4d88a551b33592bf5126ab96bd5dc5ae
