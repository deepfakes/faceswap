#!/usr/bin/env python3
""" Original Model
    Based on the original https://www.reddit.com/r/deepfakes/ code sample + contributions. """
from keras.layers import Dense, Flatten, Reshape, Conv2D, Input

from lib.model.nn_blocks import FSConv2D, FSUpscale
from ._base import KerasModel, ModelBase, logger


class Model(ModelBase):
    """ Original Faceswap Model.

    This is the original faceswap model and acts as a template for plugin development.

    The model must call the :func:`__init__` method of it's parent class prior to defining any
    attribute overrides.

    All plugins must define the attribute overrides:

        * :attr:`input_shape` (`tuple`): a tuple of ints defining the shape of the faces that the
        model takes as input

        * :attr:`output_shape` (`tuple`) a tuple of ints defining the shape of the output from the
        model

    Any additional attributes used exclusively by this model should be defined here, but make sure
    that you are not accidentally overriding any existing :class:`plugins.train.model._base.Model`
    attributes.

    Parameters
    ----------
    args: varies
        The default command line arguments passed in from :class:`scripts.train.Train`
    kwargs: varies
        The default command line keyword arguments passed in from :class:`scripts.train.Train`

    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_shape = (64, 64, 3)
        self.output_shape = (64, 64, 3)
        self.low_mem = self.config["lowmem"]
        self.learn_mask = self.config["learn_mask"]
        self.encoder_dim = 512 if self.low_mem else 1024

    def build_model(self, inputs):
        """ Initialize the model.

        This function is called immediately after :func:`__init__` has been called if a new model
        is being created. It is ignored if an existing model is being loaded from disk.

        This is where the model structure is defined.

        For the original model, An encoder instance is defined, then the same instance is
        referenced twice, one for each input "A" and "B" so that the same model is used for
        both inputs.

        2 Decoders are then defined (one for each side) with the encoder instances passed in as
        input to the corresponding decoders.

        Parameters
        ----------
        inputs: list
            A list of input tensors for the model. At a minimum this will be a list of 2 tensors of
            shape :attr:`input_shape`, the first for side "a", the second for side "b". If the
            configuration option "learn_mask" has been enabled  then this will be a list of 2
            sub-lists, the fist for side "a", the second for side "b". Each sub-lists will contain
            2 input tensors, the first being of shape :attr:`input_shape` for the face input, and
            the second being of the same height and width dimensions of :attr:`input_shape` but
            with a single channel for the 3rd dimension for the mask.

        Returns
        -------
        :class:`keras.models.Model`
            The output of this function must be a keras model. See Keras documentation for the
            correct structure. You should include the keyword argument ``name`` assigned to the
            attribute :attr:`name` to automatically name the model based on the filename.
        """
        logger.debug("Initializing model")

        input_a = inputs[0]
        input_b = inputs[1]

        encoder = self.encoder()
        encoder_a = encoder(input_a)
        encoder_b = encoder(input_b)

        outputs = [self.decoder("a")(encoder_a), self.decoder("b")(encoder_b)]

        autoencoder = KerasModel(inputs, outputs, name=self.name)
        return autoencoder

    def encoder(self):
        """ The original Faceswap Encoder Network.

        Returns
        -------
        :class:`keras.models.Model`
            The Keras encoder model, for sharing between inputs from both sides.
        """
        input_ = Input(shape=self.input_shape)
        var_x = input_
        var_x = FSConv2D(128)(var_x)
        var_x = FSConv2D(256)(var_x)
        var_x = FSConv2D(512)(var_x)
        if not self.low_mem:
            var_x = FSConv2D(1024)(var_x)
        var_x = Dense(self.encoder_dim)(Flatten()(var_x))
        var_x = Dense(4 * 4 * 1024)(var_x)
        var_x = Reshape((4, 4, 1024))(var_x)
        var_x = FSUpscale(512)(var_x)
        return KerasModel(input_, var_x, name="encoder")

    def decoder(self, side):
        """ The original Faceswap Decoder Network.

        Parameters
        ----------
        side: str
            Either `"a` or `"b"`. This is used for naming the decoder model.

        Returns
        -------
        :class:`keras.models.Model`
            The Keras decoder model. This will be called twice, once for each side.
        """
        input_ = Input(shape=(8, 8, 512))
        var_x = input_
        var_x = FSUpscale(256)(var_x)
        var_x = FSUpscale(128)(var_x)
        var_x = FSUpscale(64)(var_x)
        # TODO
        var_x = Conv2D(3,
                       kernel_size=5,
                       strides=(1, 1),
                       padding="same",
                       activation="sigmoid",
                       name="face_out_{}".format(side))(var_x)
        outputs = [var_x]

        if self.learn_mask:
            var_y = input_
            var_x = FSUpscale(256)(var_y)
            var_x = FSUpscale(128)(var_y)
            var_x = FSUpscale(64)(var_y)
            # TODO
            var_y = Conv2D(1,
                           kernel_size=5,
                           padding="same",
                           activation="sigmoid",
                           name="mask_out_{}".format(side))(var_y)
            outputs.append(var_y)
        return KerasModel(input_, outputs=outputs, name="decoder_{}".format(side))
