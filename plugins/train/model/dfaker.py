#!/usr/bin/env python3
""" DFaker Model
    Based on the dfaker model: https://github.com/dfaker """

from keras.initializers import RandomNormal
from keras.layers import Add, Input
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D
from keras.models import Model as KerasModel
from keras.optimizers import Adam
from keras.utils import multi_gpu_model

from lib.train.dssim import DSSIMObjective
from lib.train.penalized_loss import PenalizedLoss

from .original import Model as OriginalModel


class Model(OriginalModel):
    """ Improved Autoeencoder Model """
    def __init__(self, *args, **kwargs):
        kwargs["image_shape"] = (64, 64, 3)
        kwargs["encoder_dim"] = 1024
        super().__init__(*args, **kwargs)

    def add_networks(self):
        """ Add the IAE model weights """
        self.add_network("encoder", None, self.encoder())
        self.add_network("decoder", "A", self.decoder())
        self.add_network("decoder", "B", self.decoder())

    def initialize(self):
        """ Initialize original model """
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

        print(encoder.summary())
        print(decoder_a.summary())

        self.autoencoders["a"] = KerasModel([inp1, mask1],
                                            decoder_a(encoder(inp1)))
        self.autoencoders["b"] = KerasModel([inp2, mask2],
                                            decoder_b(encoder(inp2)))
        self.compile_autoencoders(mask1=mask1, mask2=mask2)

    def set_training_data(self):
        """ Set the dictionary for training """
        serializer = self.config.get("DFaker", "alignments_format")
        self.training_opts["serializer"] = serializer
        self.training_opts["use_mask"] = True
        self.training_opts["use_alignments"] = True

    def compile_autoencoders(self, *args, **kwargs):
        """ Compile the autoencoders """
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
                                      'mean_standard_error'])

    def decoder(self):
        """ Decoder Network """
        input_ = Input(shape=(8, 8, 512))
        inp_x = input_
        inp_y = input_

        inp_x = self.upscale(512)(inp_x)
        inp_x = self.res_block(inp_x, 512)
        inp_x = self.upscale(256)(inp_x)
        inp_x = self.res_block(inp_x, 256)
        inp_x = self.upscale(128)(inp_x)
        inp_x = self.res_block(inp_x, 128)
        inp_x = self.upscale(64)(inp_x)
        inp_x = Conv2D(3,
                       kernel_size=5,
                       padding='same',
                       activation='sigmoid')(inp_x)

        inp_y = self.upscale(512)(inp_y)
        inp_y = self.upscale(256)(inp_y)
        inp_y = self.upscale(128)(inp_y)
        inp_y = self.upscale(64)(inp_y)
        inp_y = Conv2D(1,
                       kernel_size=5,
                       padding='same',
                       activation='sigmoid')(inp_y)

        return KerasModel([input_], outputs=[inp_x, inp_y])

    @staticmethod
    def res_block(input_tensor, filters):
        """ Residual block """
        conv_init = RandomNormal(0, 0.02)
        inp = input_tensor
        inp = Conv2D(filters,
                     kernel_size=3,
                     kernel_initializer=conv_init,
                     use_bias=False,
                     padding="same")(inp)
        inp = LeakyReLU(alpha=0.2)(inp)
        inp = Conv2D(filters,
                     kernel_size=3,
                     kernel_initializer=conv_init,
                     use_bias=False,
                     padding="same")(inp)
        inp = Add()([inp, input_tensor])
        inp = LeakyReLU(alpha=0.2)(inp)
        return inp
