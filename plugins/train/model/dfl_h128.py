#!/usr/bin/env python3
""" DeepFakesLab H128 Model
    Based on https://github.com/iperov/DeepFaceLab
"""

from keras.layers import Dense, Flatten, Input, Reshape
from keras.layers.convolutional import Conv2D
from keras.models import Model as KerasModel

from lib.model.nn_blocks import conv, upscale
from .original import logger, Model as OriginalModel


class Model(OriginalModel):
    """ Low Memory version of Original Faceswap Model """
    def __init__(self, *args, **kwargs):
        logger.debug("Initializing %s: (args: %s, kwargs: %s",
                     self.__class__.__name__, args, kwargs)

        kwargs["input_shape"] = (128, 128, 3)
        kwargs["encoder_dim"] = 256 if self.config["lowmem"] else 512

        super().__init__(*args, **kwargs)
        logger.debug("Initialized %s", self.__class__.__name__)

    def set_training_data(self):
        """ Set the dictionary for training """
        logger.debug("Setting training data")
        training_opts = dict()
        training_opts["mask_type"] = self.config["mask_type"]
        training_opts["preview_images"] = 10
        logger.debug("Set training data: %s", training_opts)
        return training_opts

    def build_autoencoders(self):
        """ Initialize DFL H128 model """
        logger.debug("Initializing model")
        mask_shape = self.input_shape[:2] + (1, )
        for side in ("a", "b"):
            inp = [Input(shape=self.input_shape, name="face"),
                   Input(shape=mask_shape, name="mask")]
            decoder = self.networks["decoder_{}".format(side)].network
            output = decoder(self.networks["encoder"].network(inp[0]))
            autoencoder = KerasModel(inp, output)
            self.add_predictor(side, autoencoder)
        logger.debug("Initialized model")

    def encoder(self):
        """ DFL H128 Encoder """
        input_ = Input(shape=self.input_shape)
        use_subpixel = self.config["subpixel_upscaling"]

        var_x = input_
        var_x = conv(128)(var_x)
        var_x = conv(256)(var_x)
        var_x = conv(512)(var_x)
        var_x = conv(1024)(var_x)
        var_x = Dense(self.encoder_dim)(Flatten()(var_x))
        var_x = Dense(8 * 8 * self.encoder_dim)(var_x)
        var_x = Reshape((8, 8, self.encoder_dim))(var_x)
        var_x = upscale(self.encoder_dim, use_subpixel=use_subpixel)(var_x)
        return KerasModel(input_, var_x)

    def decoder(self):
        """ DFL H128 Decoder """
        input_ = Input(shape=(16, 16, self.encoder_dim))
        use_subpixel = self.config["subpixel_upscaling"]

        var = input_
        var = upscale(self.encoder_dim, use_subpixel=use_subpixel)(var)
        var = upscale(self.encoder_dim // 2, use_subpixel=use_subpixel)(var)
        var = upscale(self.encoder_dim // 4, use_subpixel=use_subpixel)(var)

        # Face
        var_x = Conv2D(3, kernel_size=5, padding="same", activation="sigmoid")(var)
        # Mask
        var_y = Conv2D(1, kernel_size=5, padding="same", activation="sigmoid")(var)
        return KerasModel(input_, [var_x, var_y])
