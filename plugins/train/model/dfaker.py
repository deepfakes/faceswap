#!/usr/bin/env python3
""" DFaker Model
    Based on the dfaker model: https://github.com/dfaker """
import logging
import sys

# Ignore linting errors from Tensorflow's thoroughly broken import system
from tensorflow.keras.initializers import RandomNormal  # pylint:disable=import-error
from tensorflow.keras.layers import Input, LeakyReLU  # pylint:disable=import-error
from tensorflow.keras.models import Model as KModel  # pylint:disable=import-error

from lib.model.nn_blocks import Conv2DOutput, UpscaleBlock, ResidualBlock
from .original import Model as OriginalModel

logger = logging.getLogger(__name__)


class Model(OriginalModel):
    """ Dfaker Model """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._output_size = self.config["output_size"]
        if self._output_size not in (128, 256):
            logger.error("Dfaker output shape should be 128 or 256 px")
            sys.exit(1)
        self.input_shape = (self._output_size // 2, self._output_size // 2, 3)
        self.encoder_dim = 1024
        self.kernel_initializer = RandomNormal(0, 0.02)

    def decoder(self, side):
        """ Decoder Network """
        input_ = Input(shape=(8, 8, 512))
        var_x = input_

        if self._output_size == 256:
            var_x = UpscaleBlock(1024, activation=None)(var_x)
            var_x = LeakyReLU(alpha=0.2)(var_x)
            var_x = ResidualBlock(1024, kernel_initializer=self.kernel_initializer)(var_x)
        var_x = UpscaleBlock(512, activation=None)(var_x)
        var_x = LeakyReLU(alpha=0.2)(var_x)
        var_x = ResidualBlock(512, kernel_initializer=self.kernel_initializer)(var_x)
        var_x = UpscaleBlock(256, activation=None)(var_x)
        var_x = LeakyReLU(alpha=0.2)(var_x)
        var_x = ResidualBlock(256, kernel_initializer=self.kernel_initializer)(var_x)
        var_x = UpscaleBlock(128, activation=None)(var_x)
        var_x = LeakyReLU(alpha=0.2)(var_x)
        var_x = ResidualBlock(128, kernel_initializer=self.kernel_initializer)(var_x)
        var_x = UpscaleBlock(64, activation="leakyrelu")(var_x)
        var_x = Conv2DOutput(3, 5, name=f"face_out_{side}")(var_x)
        outputs = [var_x]

        if self.config.get("learn_mask", False):
            var_y = input_
            if self._output_size == 256:
                var_y = UpscaleBlock(1024, activation="leakyrelu")(var_y)
            var_y = UpscaleBlock(512, activation="leakyrelu")(var_y)
            var_y = UpscaleBlock(256, activation="leakyrelu")(var_y)
            var_y = UpscaleBlock(128, activation="leakyrelu")(var_y)
            var_y = UpscaleBlock(64, activation="leakyrelu")(var_y)
            var_y = Conv2DOutput(1, 5, name=f"mask_out_{side}")(var_y)
            outputs.append(var_y)
        return KModel([input_], outputs=outputs, name=f"decoder_{side}")
