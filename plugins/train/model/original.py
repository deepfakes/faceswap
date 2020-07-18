#!/usr/bin/env python3
""" Original Model
    Based on the original https://www.reddit.com/r/deepfakes/ code sample + contributions. """
import os
from keras.layers import Dense, Flatten, Reshape, Layer, Conv2D

from keras.models import Model as KerasModel
from lib.model.nn_blocks import FSConv2D, Upscale
from ._base import ModelBase, logger


class Model(ModelBase):
    """ Original Faceswap Model """
    def __init__(self, *args, **kwargs):
        logger.debug("Initializing %s: (args: %s, kwargs: %s",
                     self.__class__.__name__, args, kwargs)
        super().__init__(*args, **kwargs)
        self.input_shape = (64, 64, 3)
        self.output_shape = (64, 64, 3)
        logger.debug("Initialized %s", self.__class__.__name__)

    def build_model(self, inputs):
        """ Initialize original model """
        logger.debug("Initializing model")

        input_a = inputs[0]
        input_b = inputs[1]

        encoder = Encoder(self.config["lowmem"])
        encoder_a = encoder(input_a)
        encoder_b = encoder(input_b)

        decoder_a = Decoder("a", self.config["learn_mask"])(encoder_a)
        decoder_b = Decoder("b", self.config["learn_mask"])(encoder_b)

        autoencoder = KerasModel(inputs, [decoder_a, decoder_b],
                                 name=os.path.splitext(os.path.basename(__file__).lower())[0])
        return autoencoder


class Encoder(Layer):
    """Original Encoder Network

    Parameters
    ----------
    low_mem: bool
        ``True`` if the model should be run in low memory mode otherwise ``False``
    kwargs: dict
        Any additional standard Keras layer key word arguments
    """
    def __init__(self, low_mem, **kwargs):
        super().__init__(name="original_encoder", **kwargs)
        self.low_mem = low_mem
        encoder_dims = 512 if self.low_mem else 1024

        self._conv1 = FSConv2D(128)
        self._conv2 = FSConv2D(256)
        self._conv3 = FSConv2D(512)
        if not self.low_mem:
            self._conv4 = FSConv2D(1024)
        self._dense1 = Dense(encoder_dims)
        self._dense2 = Dense(4 * 4 * 1024)
        self._flatten = Flatten()
        self._reshape = Reshape((4, 4, 1024))
        self._upscale = Upscale(512)

    def call(self, inputs):
        """ Call the Original Encoder Layer.

        Parameters
        ----------
        inputs: Tensor
            The input to the layer

        Returns
        -------
        Tensor
            The output from the layer
        """
        var_x = self._conv1(inputs)
        var_x = self._conv2(var_x)
        var_x = self._conv3(var_x)
        if not self.low_mem:
            var_x = self._conv4(var_x)
        var_x = self._dense1(self._flatten(var_x))
        var_x = self._dense2(var_x)
        var_x = self._reshape(var_x)
        var_x = self._upscale(var_x)
        return var_x

    def get_config(self):
        """Returns the config of the layer.

        A layer config is a Python dictionary (serializable) containing the configuration of a
        layer. The same layer can be reinstated later (without its trained weights) from this
        configuration.

        The configuration of a layer does not include connectivity information, nor the layer
        class name. These are handled by `Network` (one layer of abstraction above).

        Returns
        --------
        dict
            A python dictionary containing the layer configuration
        """
        return dict(low_mem=self.low_mem)


class Decoder(Layer):
    """ Original Decoder Network

    Parameters
    ----------
    side: ['a', 'b']
        The side that this decoder resides on. Used for naming
    learn_mask: bool
        ``True`` if the model should learn the mask otherwise ``False``
    kwargs: dict
        Any additional standard Keras layer key word arguments
    """
    def __init__(self, side, learn_mask, **kwargs):
        super().__init__(name="original_decoder_{}".format(side), **kwargs)
        self.learn_mask = learn_mask
        self.side = side

        self._upscale1 = Upscale(256)
        self._upscale2 = Upscale(128)
        self._upscale3 = Upscale(64)
        # TODO Need to move this to NN Blocks so we use our initializers etc. See also mask conv
        # TODO Output name does not appear to cascade, so need a mechanism for naming
        self._conv = Conv2D(3,
                            kernel_size=5,
                            strides=(1, 1),
                            padding="same",
                            activation="sigmoid",
                            name="face_out_{}".format(self.side))

        if self.learn_mask:
            self._mask_upscale1 = Upscale(256)
            self._mask_upscale2 = Upscale(128)
            self._mask_upscale3 = Upscale(64)
            self._mask_conv = Conv2D(1,
                                     kernel_size=5,
                                     strides=(1, 1),
                                     padding="same",
                                     activation="sigmoid",
                                     name="mask_out_{}".format(self.side))

    def call(self, inputs):
        """ Call the Original Decoder Layer.

        Parameters
        ----------
        inputs: Tensor
            The input to the layer

        Returns
        -------
        Tensor
            The output from the layer
        """
        mask_input = None

        var_x = self._upscale1(inputs)
        var_x = self._upscale2(var_x)
        var_x = self._upscale3(var_x)
        var_x = self._conv(var_x)

        if self.learn_mask:
            var_y = self._mask_upscale1(mask_input)
            var_y = self._mask_upscale2(var_y)
            var_y = self._mask_upscale3(var_y)
            var_y = self._mask_conv(var_y)
            output = [var_x, var_y]
        else:
            output = var_x

        return output

    def get_config(self):
        """Returns the config of the layer.

        A layer config is a Python dictionary (serializable) containing the configuration of a
        layer. The same layer can be reinstated later (without its trained weights) from this
        configuration.

        The configuration of a layer does not include connectivity information, nor the layer
        class name. These are handled by `Network` (one layer of abstraction above).

        Returns
        --------
        dict
            A python dictionary containing the layer configuration
        """
        return dict(side=self.side,
                    learn_mask=self.learn_mask)
