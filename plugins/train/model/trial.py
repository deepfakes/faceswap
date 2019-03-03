#!/usr/bin/env python3
""" Original Model
    Based on the original https://www.reddit.com/r/deepfakes/
    code sample + contribs """

from keras.layers import Conv2D, Dense, Flatten, Input, Reshape, UpSampling2D, Lambda, Add, ReflectionPadding2D
from keras.initializers import Constant
import tensorflow as tf

from keras.models import Model as KerasModel

from ._base import ModelBase, logger
from lib.model.layers import Scale, SubPixelUpscaling


class Model(ModelBase):
    """ Original Faceswap Model """
    def __init__(self, *args, **kwargs):
        logger.debug("Initializing %s: (args: %s, kwargs: %s",
                     self.__class__.__name__, args, kwargs)

        kwargs["input_shape"] = (160, 160, 3)
        kwargs["encoder_dim"] = 1024

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
        inputs = [Input(shape=self.input_shape, name="face")]
        if self.config.get("mask_type", None):
            mask_shape = (self.input_shape[:2] + (1, ))
            inputs.append(Input(shape=mask_shape, name="mask"))

        for side in ("a", "b"):
            logger.debug("Adding Autoencoder. Side: %s", side)
            decoder = self.networks["decoder_{}".format(side)].network
            output = decoder(self.networks["encoder"].network(inputs[0]))
            autoencoder = KerasModel(inputs, output)
            self.add_predictor(side, autoencoder)
        logger.debug("Initialized model")

    def encoder(self):
        """ Encoder Network """
        input_ = Input(shape=self.input_shape)
        var_x = input_ # 160
        skip = Lambda(lambda skip: tf.reduce_mean(skip, axis=[1,2], keepdims=True))(var_x)
        var_x = self.blocks.conv(var_x, 3,   kernel_size=2, strides=2,
                                 kernel_initializer=Constant(value=0.25),
                                 padding="valid")  # 80
        
        var_x = self.blocks.conv(var_x, 64,  kernel_size=5, strides=1)  # 80
        var_x = Lambda(lambda x: tf.space_to_depth(x, block_size=2))(var_x) # 40
        
        var_x = self.blocks.conv(var_x, 128, kernel_size=5, strides=1)  # 40
        var_x = Lambda(lambda x: tf.space_to_depth(x, block_size=2))(var_x) # 20
        
        var_x = self.blocks.conv(var_x, 256, kernel_size=5, strides=1)  # 20
        var_x = Lambda(lambda x: tf.space_to_depth(x, block_size=2))(var_x) # 10
        
        var_x = self.blocks.conv(var_x, 512, kernel_size=5, strides=1) # 10
        var_x = Lambda(lambda x: tf.space_to_depth(x, block_size=2))(var_x) # 5
        
        var_x = self.blocks.conv(var_x, 1024, kernel_size=5, padding="valid") # 1
        var_x = self.blocks.conv(var_x, 5*5*1024, kernel_size=1, padding="valid") # 1
        var_x = SubPixelUpscaling(scale_factor=5)(var_x) # 5
        var_x = self.blocks.upscale(var_x, 512, kernel_size=3, res_block_follows=True) # 10
        
        return KerasModel(input_, [skip, var_x])

    def decoder(self):
        """ Decoder Network """
        input_ = Input(shape=(10, 10, 512))         #10
        skip = Input(shape=(160, 160, 3))         #160
        
        var = input_
        var = self.blocks.res_block(var, 512) # 10
        
        var = self.blocks.upscale(var, 256, kernel_size=3, res_block_follows=True)     #20
        var = self.blocks.res_block(var, 256) # 20
        
        var = self.blocks.upscale(var, 128, kernel_size=3, res_block_follows=True)     #40
        var = self.blocks.res_block(var, 128) # 40
        
        var = self.blocks.upscale(var, 64, kernel_size=3, res_block_follows=True)      #80
        var = self.blocks.res_block(var, 64) # 80
        
        var = Conv2D(3, kernel_size=5, padding="same", activation="tanh")(var)
        var = UpSampling2D(size=2, interpolation='bilinear')(var)
        var = Scale(gamma_init=Constant(value=0.1))(var)
        var_x = Add()([skip, var])
        outputs = [var_x]

        if self.config.get("mask_type", None):
            var_y = Conv2D(1, kernel_size=5, padding="same", activation="sigmoid")(var)
            var_y = UpSampling2D(size=2, interpolation='bilinear')(var_y)
            outputs.append(var_y)
        return KerasModel([skip, input_], outputs=outputs)
