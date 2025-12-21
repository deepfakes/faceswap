#!/usr/bin/env python3
""" DeepFaceLab SAE Model
    Based on https://github.com/iperov/DeepFaceLab
"""
import logging

import numpy as np

from keras import Input, layers, Model as KModel

from lib.model.nn_blocks import Conv2DOutput, Conv2DBlock, ResidualBlock, UpscaleBlock
from plugins.train.train_config import Loss as cfg_loss

from ._base import ModelBase
from . import dfl_sae_defaults as cfg

logger = logging.getLogger(__name__)


class Model(ModelBase):
    """ SAE Model from DFL """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_shape = (cfg.input_size(), cfg.input_size(), 3)
        self.architecture = cfg.architecture().lower()
        self.use_mask = cfg_loss.learn_mask()
        self.multiscale_count = 3 if cfg.multiscale_decoder() else 1
        self.encoder_dim = cfg.encoder_dims()
        self.decoder_dim = cfg.decoder_dims()

    @property
    def model_name(self):
        """ str: The name of the keras model. Varies depending on selected architecture. """
        return f"{self.name}_{self.architecture}"

    @property
    def ae_dims(self):
        """ Set the Autoencoder Dimensions or set to default """
        retval = cfg.autoencoder_dims()
        if retval == 0:
            retval = 256 if self.architecture == "liae" else 512
        return retval

    @property
    def freeze_layers(self) -> list[str]:
        """ list[str] : The layer name for freezing based on the configured architecture """
        return [f"encoder_{self.architecture}"]

    @property
    def load_layers(self) -> list[str]:
        """ list[str] : The layer name for loading based on the configured architecture """
        return [f"encoder_{self.architecture}"]

    def build_model(self, inputs):
        """ Build the DFL-SAE Model """
        encoder = getattr(self, f"encoder_{self.architecture}")()
        enc_output_shape = encoder.output_shape[1:]
        encoder_a = encoder(inputs[0])
        encoder_b = encoder(inputs[1])

        if self.architecture == "liae":
            inter_both = self.inter_liae("both", enc_output_shape)
            int_output_shape = (np.array(inter_both.output_shape[1:]) * (1, 1, 2)).tolist()

            inter_a = layers.Concatenate()([inter_both(encoder_a), inter_both(encoder_a)])
            inter_b = layers.Concatenate()([self.inter_liae("b", enc_output_shape)(encoder_b),
                                            inter_both(encoder_b)])

            decoder = self.decoder("both", int_output_shape)
            outputs = decoder(inter_a) + decoder(inter_b)
        else:
            outputs = (self.decoder("a", enc_output_shape)(encoder_a) +
                       self.decoder("b", enc_output_shape)(encoder_b))
        autoencoder = KModel(inputs, outputs, name=self.model_name)
        return autoencoder

    def encoder_df(self):
        """ DFL SAE DF Encoder Network"""
        input_ = Input(shape=self.input_shape)
        dims = self.input_shape[-1] * self.encoder_dim
        lowest_dense_res = self.input_shape[0] // 16
        var_x = Conv2DBlock(dims, activation="leakyrelu")(input_)
        var_x = Conv2DBlock(dims * 2, activation="leakyrelu")(var_x)
        var_x = Conv2DBlock(dims * 4, activation="leakyrelu")(var_x)
        var_x = Conv2DBlock(dims * 8, activation="leakyrelu")(var_x)
        var_x = layers.Dense(self.ae_dims)(layers.Flatten()(var_x))
        var_x = layers.Dense(lowest_dense_res * lowest_dense_res * self.ae_dims)(var_x)
        var_x = layers.Reshape((lowest_dense_res, lowest_dense_res, self.ae_dims))(var_x)
        var_x = UpscaleBlock(self.ae_dims, activation="leakyrelu")(var_x)
        return KModel(input_, var_x, name="encoder_df")

    def encoder_liae(self):
        """ DFL SAE LIAE Encoder Network """
        input_ = Input(shape=self.input_shape)
        dims = self.input_shape[-1] * self.encoder_dim
        var_x = Conv2DBlock(dims, activation="leakyrelu")(input_)
        var_x = Conv2DBlock(dims * 2, activation="leakyrelu")(var_x)
        var_x = Conv2DBlock(dims * 4, activation="leakyrelu")(var_x)
        var_x = Conv2DBlock(dims * 8, activation="leakyrelu")(var_x)
        var_x = layers.Flatten()(var_x)
        return KModel(input_, var_x, name="encoder_liae")

    def inter_liae(self, side, input_shape):
        """ DFL SAE LIAE Intermediate Network """
        input_ = Input(shape=input_shape)
        lowest_dense_res = self.input_shape[0] // 16
        var_x = input_
        var_x = layers.Dense(self.ae_dims)(var_x)
        var_x = layers.Dense(lowest_dense_res * lowest_dense_res * self.ae_dims * 2)(var_x)
        var_x = layers.Reshape((lowest_dense_res, lowest_dense_res, self.ae_dims * 2))(var_x)
        var_x = UpscaleBlock(self.ae_dims * 2, activation="leakyrelu")(var_x)
        return KModel(input_, var_x, name=f"intermediate_{side}")

    def decoder(self, side, input_shape):
        """ DFL SAE Decoder Network"""
        input_ = Input(shape=input_shape)
        outputs = []

        dims = self.input_shape[-1] * self.decoder_dim
        var_x = input_

        var_x1 = UpscaleBlock(dims * 8, activation=None)(var_x)
        var_x1 = layers.LeakyReLU(negative_slope=0.2)(var_x1)
        var_x1 = ResidualBlock(dims * 8)(var_x1)
        var_x1 = ResidualBlock(dims * 8)(var_x1)
        if self.multiscale_count >= 3:
            outputs.append(Conv2DOutput(3, 5, name=f"face_out_32_{side}")(var_x1))

        var_x2 = UpscaleBlock(dims * 4, activation=None)(var_x1)
        var_x2 = layers.LeakyReLU(negative_slope=0.2)(var_x2)
        var_x2 = ResidualBlock(dims * 4)(var_x2)
        var_x2 = ResidualBlock(dims * 4)(var_x2)
        if self.multiscale_count >= 2:
            outputs.append(Conv2DOutput(3, 5, name=f"face_out_64_{side}")(var_x2))

        var_x3 = UpscaleBlock(dims * 2, activation=None)(var_x2)
        var_x3 = layers.LeakyReLU(negative_slope=0.2)(var_x3)
        var_x3 = ResidualBlock(dims * 2)(var_x3)
        var_x3 = ResidualBlock(dims * 2)(var_x3)

        outputs.append(Conv2DOutput(3, 5, name=f"face_out_128_{side}")(var_x3))

        if self.use_mask:
            var_y = input_
            var_y = UpscaleBlock(self.decoder_dim * 8, activation="leakyrelu")(var_y)
            var_y = UpscaleBlock(self.decoder_dim * 4, activation="leakyrelu")(var_y)
            var_y = UpscaleBlock(self.decoder_dim * 2, activation="leakyrelu")(var_y)
            var_y = Conv2DOutput(1, 5, name=f"mask_out_{side}")(var_y)
            outputs.append(var_y)
        return KModel(input_, outputs=outputs, name=f"decoder_{side}")
