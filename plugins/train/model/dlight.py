#!/usr/bin/env python3
""" A lightweight variant of DFaker Model
    By AnDenix, 2018-2019
    Based on the dfaker model: https://github.com/dfaker

    Acknowledgments:
    kvrooman for numerous insights and invaluable aid
    DeepHomage for lots of testing
    """

from keras.layers import Dense, Flatten, Input, Reshape, AveragePooling2D, LeakyReLU
from keras.layers import UpSampling2D
from keras.layers.core import Dropout
from keras.layers.merge import Concatenate
from keras.layers.normalization import BatchNormalization
from keras.models import Model as KerasModel

from lib.utils import FaceswapError

from ._base import logger
from .original import Model as OriginalModel


class Model(OriginalModel):
    """ DeLight Autoencoder Model """

    def __init__(self, *args, **kwargs):
        logger.debug("Initializing %s: (args: %s, kwargs: %s",
                     self.__class__.__name__, args, kwargs)

        kwargs["input_shape"] = (128, 128, 3)
        kwargs["encoder_dim"] = -1
        self.dense_output = None
        self.detail_level = None
        self.features = None
        self.encoder_filters = None
        self.encoder_dim = None
        self.details = None
        self.upscale_ratio = None
        super().__init__(*args, **kwargs)

        logger.debug("Initialized %s", self.__class__.__name__)

    def _detail_level_setup(self):
        logger.debug('self.config[output_size]: %d', self.config["output_size"])
        self.features = dict(lowmem=0, fair=1, best=2)[self.config["features"]]
        logger.debug('self.features: %d', self.features)
        self.encoder_filters = 64 if self.features > 0 else 48
        logger.debug('self.encoder_filters: %d', self.encoder_filters)
        bonum_fortunam = 128
        self.encoder_dim = {0: 512 + bonum_fortunam,
                            1: 1024 + bonum_fortunam,
                            2: 1536 + bonum_fortunam}[self.features]
        logger.debug('self.encoder_dim: %d', self.encoder_dim)
        self.details = dict(fast=0, good=1)[self.config["details"]]
        logger.debug('self.details: %d', self.details)

        try:
            self.upscale_ratio = {128: 2,
                                  256: 4,
                                  384: 6}[self.config["output_size"]]
        except KeyError:
            logger.error("Config error: output_size must be one of: 128, 256, or 384.")
            raise FaceswapError("Config error: output_size must be one of: 128, 256, or 384.")
        logger.debug('output_size: %r', self.config["output_size"])
        logger.debug('self.upscale_ratio: %r', self.upscale_ratio)

    def build(self):
        self._detail_level_setup()
        super().build()

    def add_networks(self):
        """ Add the DeLight model weights """
        logger.debug("Adding networks")
        self.add_network("decoder", "a", self.decoder_a(), is_output=True)
        self.add_network("decoder", "b",
                         self.decoder_b() if self.details > 0 else self.decoder_b_fast(),
                         is_output=True)
        self.add_network("encoder", None, self.encoder())
        logger.debug("Added networks")

    def compile_predictors(self, **kwargs):  # pylint: disable=arguments-differ
        self.set_networks_trainable()
        super().compile_predictors(**kwargs)

    def set_networks_trainable(self):
        """ Set the network state to trainable """
        train_encoder = True
        train_decoder_a = True
        train_decoder_b = True

        encoder = self.networks['encoder'].network
        for layer in encoder.layers:
            layer.trainable = train_encoder

        decoder_a = self.networks['decoder_a'].network
        for layer in decoder_a.layers:
            layer.trainable = train_decoder_a

        decoder_b = self.networks['decoder_b'].network
        for layer in decoder_b.layers:
            layer.trainable = train_decoder_b

    def encoder(self):
        """ DeLight Encoder Network """
        input_ = Input(shape=self.input_shape)
        var_x = input_

        var_x1 = self.blocks.conv(var_x, self.encoder_filters // 2)
        var_x2 = AveragePooling2D()(var_x)
        var_x2 = LeakyReLU(0.1)(var_x2)
        var_x = Concatenate()([var_x1, var_x2])

        var_x1 = self.blocks.conv(var_x, self.encoder_filters)
        var_x2 = AveragePooling2D()(var_x)
        var_x2 = LeakyReLU(0.1)(var_x2)
        var_x = Concatenate()([var_x1, var_x2])

        var_x1 = self.blocks.conv(var_x, self.encoder_filters * 2)
        var_x2 = AveragePooling2D()(var_x)
        var_x2 = LeakyReLU(0.1)(var_x2)
        var_x = Concatenate()([var_x1, var_x2])

        var_x1 = self.blocks.conv(var_x, self.encoder_filters * 4)
        var_x2 = AveragePooling2D()(var_x)
        var_x2 = LeakyReLU(0.1)(var_x2)
        var_x = Concatenate()([var_x1, var_x2])

        var_x1 = self.blocks.conv(var_x, self.encoder_filters * 8)
        var_x2 = AveragePooling2D()(var_x)
        var_x2 = LeakyReLU(0.1)(var_x2)
        var_x = Concatenate()([var_x1, var_x2])

        var_x = Dense(self.encoder_dim)(Flatten()(var_x))
        var_x = Dropout(0.05)(var_x)
        var_x = Dense(4 * 4 * 1024)(var_x)
        var_x = Dropout(0.05)(var_x)
        var_x = Reshape((4, 4, 1024))(var_x)

        return KerasModel(input_, var_x)

    def decoder_a(self):
        """ DeLight Decoder A(old face) Network """
        input_ = Input(shape=(4, 4, 1024))
        decoder_a_complexity = 256
        mask_complexity = 128

        var_xy = input_
        var_xy = UpSampling2D(self.upscale_ratio, interpolation='bilinear')(var_xy)

        var_x = var_xy
        var_x = self.blocks.upscale2x(var_x, decoder_a_complexity, fast=False)
        var_x = self.blocks.upscale2x(var_x, decoder_a_complexity // 2, fast=False)
        var_x = self.blocks.upscale2x(var_x, decoder_a_complexity // 4, fast=False)
        var_x = self.blocks.upscale2x(var_x, decoder_a_complexity // 8, fast=False)

        var_x = self.blocks.conv2d(var_x, 3, kernel_size=5, padding="same",
                                   activation="sigmoid", name="face_out")

        outputs = [var_x]

        if self.config.get("learn_mask", False):
            var_y = var_xy  # mask decoder
            var_y = self.blocks.upscale2x(var_y, mask_complexity, fast=False)
            var_y = self.blocks.upscale2x(var_y, mask_complexity // 2, fast=False)
            var_y = self.blocks.upscale2x(var_y, mask_complexity // 4, fast=False)
            var_y = self.blocks.upscale2x(var_y, mask_complexity // 8, fast=False)

            var_y = self.blocks.conv2d(var_y, 1, kernel_size=5, padding="same",
                                       activation="sigmoid", name="mask_out")

            outputs.append(var_y)

        return KerasModel([input_], outputs=outputs)

    def decoder_b_fast(self):
        """ DeLight Fast Decoder B(new face) Network  """
        input_ = Input(shape=(4, 4, 1024))

        decoder_b_complexity = 512
        mask_complexity = 128

        var_xy = input_

        var_xy = self.blocks.upscale(var_xy, 512, scale_factor=self.upscale_ratio)
        var_x = var_xy

        var_x = self.blocks.upscale2x(var_x, decoder_b_complexity, fast=True)
        var_x = self.blocks.upscale2x(var_x, decoder_b_complexity // 2, fast=True)
        var_x = self.blocks.upscale2x(var_x, decoder_b_complexity // 4, fast=True)
        var_x = self.blocks.upscale2x(var_x, decoder_b_complexity // 8, fast=True)

        var_x = self.blocks.conv2d(var_x, 3, kernel_size=5, padding="same",
                                   activation="sigmoid", name="face_out")

        outputs = [var_x]

        if self.config.get("learn_mask", False):
            var_y = var_xy  # mask decoder

            var_y = self.blocks.upscale2x(var_y, mask_complexity, fast=False)
            var_y = self.blocks.upscale2x(var_y, mask_complexity // 2, fast=False)
            var_y = self.blocks.upscale2x(var_y, mask_complexity // 4, fast=False)
            var_y = self.blocks.upscale2x(var_y, mask_complexity // 8, fast=False)

            var_y = self.blocks.conv2d(var_y, 1, kernel_size=5, padding="same",
                                       activation="sigmoid", name="mask_out")

            outputs.append(var_y)

        return KerasModel([input_], outputs=outputs)

    def decoder_b(self):
        """ DeLight Decoder B(new face) Network  """
        input_ = Input(shape=(4, 4, 1024))

        decoder_b_complexity = 512
        mask_complexity = 128

        var_xy = input_

        var_xy = self.blocks.upscale2x(var_xy, 512, scale_factor=self.upscale_ratio, fast=False)

        var_x = var_xy

        var_x = self.blocks.res_block(var_x, 512, use_bias=True)
        var_x = self.blocks.res_block(var_x, 512, use_bias=False)
        var_x = self.blocks.res_block(var_x, 512, use_bias=False)
        var_x = self.blocks.upscale2x(var_x, decoder_b_complexity, fast=False)
        var_x = self.blocks.res_block(var_x, decoder_b_complexity, use_bias=True)
        var_x = self.blocks.res_block(var_x, decoder_b_complexity, use_bias=False)
        var_x = BatchNormalization()(var_x)
        var_x = self.blocks.upscale2x(var_x, decoder_b_complexity // 2, fast=False)
        var_x = self.blocks.res_block(var_x, decoder_b_complexity // 2, use_bias=True)
        var_x = self.blocks.upscale2x(var_x, decoder_b_complexity // 4, fast=False)
        var_x = self.blocks.res_block(var_x, decoder_b_complexity // 4, use_bias=False)
        var_x = BatchNormalization()(var_x)
        var_x = self.blocks.upscale2x(var_x, decoder_b_complexity // 8, fast=False)

        var_x = self.blocks.conv2d(var_x, 3, kernel_size=5, padding="same",
                                   activation="sigmoid", name="face_out")

        outputs = [var_x]

        if self.config.get("learn_mask", False):
            var_y = var_xy  # mask decoder

            var_y = self.blocks.upscale2x(var_y, mask_complexity, fast=False)
            var_y = self.blocks.upscale2x(var_y, mask_complexity // 2, fast=False)
            var_y = self.blocks.upscale2x(var_y, mask_complexity // 4, fast=False)
            var_y = self.blocks.upscale2x(var_y, mask_complexity // 8, fast=False)

            var_y = self.blocks.conv2d(var_y, 1, kernel_size=5, padding="same",
                                       activation="sigmoid", name="mask_out")

            outputs.append(var_y)

        return KerasModel([input_], outputs=outputs)
