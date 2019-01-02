#!/usr/bin/env python3
""" Original - HiRes Model
    Based on the original https://www.reddit.com/r/deepfakes/
        code sample + contribs """


from keras.layers import Dense, Flatten, Input, Reshape
from keras.layers.convolutional import Conv2D
from keras.models import Model as KerasModel

from lib.model.nn_blocks import conv, conv_sep, upscale
from .original import logger, Model as OriginalModel


class Model(OriginalModel):
    """ Original HiRes Faceswap Model """
    def __init__(self, *args, **kwargs):
        logger.debug("Initializing %s: (args: %s, kwargs: %s",
                     self.__class__.__name__, args, kwargs)
        kwargs["image_shape"] = (128, 128, 3)
        super().__init__(*args, **kwargs)
        logger.debug("Initialized %s", self.__class__.__name__)

    def encoder(self):
        """ Original HiRes Encoder """
        input_ = Input(shape=self.image_shape)
        var_x = input_
        var_x = conv(128)(var_x)
        var_x = conv_sep(256)(var_x)
        var_x = conv(512)(var_x)
        var_x = conv_sep(1024)(var_x)
        var_x = Dense(self.encoder_dim)(Flatten()(var_x))
        var_x = Dense(8 * 8 * 512)(var_x)
        var_x = Reshape((8, 8, 512))(var_x)
        var_x = upscale(512)(var_x)
        return KerasModel(input_, var_x)

    @staticmethod
    def decoder():
        """ Original HiRes Encoder """
        input_ = Input(shape=(16, 16, 512))
        var_x = input_
        var_x = upscale(384)(var_x)
        var_x = upscale(256-32)(var_x)
        var_x = upscale(128)(var_x)
        var_x = Conv2D(3, kernel_size=5, padding="same", activation="sigmoid")(var_x)
        return KerasModel(input_, var_x)
