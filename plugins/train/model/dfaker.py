#!/usr/bin/env python3
""" DFaker Model
    Based on the dfaker model: https://github.com/dfaker """


from keras.layers import Input
from keras.models import Model as KerasModel
from keras.optimizers import Adam
from keras.utils import multi_gpu_model

from lib.train import DSSIMObjective, PenalizedLoss
from lib.train.nn_blocks import Conv2D, res_block, upscale

from .original import logger, Model as OriginalModel


class Model(OriginalModel):
    """ Improved Autoeencoder Model """
    def __init__(self, *args, **kwargs):
        logger.debug("Initializing %s: (args: %s, kwargs: %s",
                     self.__class__.__name__, args, kwargs)
        kwargs["image_shape"] = (64, 64, 3)
        kwargs["encoder_dim"] = 1024
        kwargs["trainer"] = "dfaker"
        super().__init__(*args, **kwargs)
        logger.debug("Initialized %s", self.__class__.__name__)

    def add_networks(self):
        """ Add the Dfaker model networks """
        logger.debug("Adding networks")
        self.add_network("encoder", None, self.encoder())
        self.add_network("decoder", "A", self.decoder())
        self.add_network("decoder", "B", self.decoder())
        logger.debug("Added networks")

    def initialize(self):
        """ Initialize Dfaker model """
        logger.debug("Initializing model")
        self.set_training_data()
        inp1 = Input(shape=self.image_shape)
        mask1 = Input(shape=(64*2, 64*2, 1))
        inp2 = Input(shape=self.image_shape)
        mask2 = Input(shape=(64*2, 64*2, 1))
        for network in self.networks:
            if network.type == "encoder":
                encoder = network.network
            elif network.type == "decoder" and network.side == "A":
                decoder_a = network.network
            elif network.type == "decoder" and network.side == "B":
                decoder_b = network.network

        self.log_summary("encoder", encoder)
        self.log_summary("decoder", decoder_a)

        self.autoencoders["a"] = KerasModel([inp1, mask1],
                                            decoder_a(encoder(inp1)))
        self.autoencoders["b"] = KerasModel([inp2, mask2],
                                            decoder_b(encoder(inp2)))
        self.compile_autoencoders(mask1=mask1, mask2=mask2)
        logger.debug("Initialized model")

    def set_training_data(self):
        """ Set the dictionary for training """
        logger.debug("Setting training data")
        serializer = self.config.get("DFaker", "alignments_format")
        self.training_opts["serializer"] = serializer
        self.training_opts["use_mask"] = True
        self.training_opts["use_alignments"] = True
        logger.debug("Set training data: %s", self.training_opts)

    def compile_autoencoders(self, *args, **kwargs):
        """ Compile the autoencoders """
        logger.debug("Compiling Autoencoders")
        optimizer = Adam(lr=5e-5, beta_1=0.5, beta_2=0.999)
        if self.gpus > 1:
            for acr in self.autoencoders.keys():
                autoencoder = multi_gpu_model(self.autoencoders[acr],
                                              self.gpus)
                self.autoencoders[acr] = autoencoder

        for key, autoencoder in self.autoencoders.items():
            mask = kwargs["mask1"] if key == "a" else kwargs["mask2"]
            autoencoder.compile(optimizer=optimizer,
                                loss=[PenalizedLoss(mask, DSSIMObjective()),
                                      'mse'])
        logger.debug("Compiled Autoencoders")

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
