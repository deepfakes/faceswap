#!/usr/bin/env python3
""" Normalization methods for faceswap.py specific to Tensorflow backend """
import inspect
import sys

import tensorflow as tf
import tensorflow.keras.backend as K  # pylint:disable=no-name-in-module,import-error
# tf.keras has a LayerNormaliztion implementation
# pylint:disable=unused-import
from tensorflow.keras.layers import (  # noqa pylint:disable=no-name-in-module,import-error
    Layer,
    LayerNormalization)

from lib.utils import get_keras_custom_objects as get_custom_objects


class RMSNormalization(Layer):
    """ Root Mean Square Layer Normalization (Biao Zhang, Rico Sennrich, 2019)

    RMSNorm is a simplification of the original layer normalization (LayerNorm). LayerNorm is a
    regularization technique that might handle the internal covariate shift issue so as to
    stabilize the layer activations and improve model convergence. It has been proved quite
    successful in NLP-based model. In some cases, LayerNorm has become an essential component
    to enable model optimization, such as in the SOTA NMT model Transformer.

    RMSNorm simplifies LayerNorm by removing the mean-centering operation, or normalizing layer
    activations with RMS statistic.

    Parameters
    ----------
    axis: int
        The axis to normalize across. Typically this is the features axis. The left-out axes are
        typically the batch axis/axes. This argument defaults to `-1`, the last dimension in the
        input.
    epsilon: float, optional
        Small float added to variance to avoid dividing by zero. Default: `1e-8`
    partial: float, optional
        Partial multiplier for calculating pRMSNorm. Valid values are between `0.0` and `1.0`.
        Setting to `0.0` or `1.0` disables. Default: `0.0`
    bias: bool, optional
        Whether to use a bias term for RMSNorm. Disabled by default because RMSNorm does not
        enforce re-centering invariance. Default ``False``
    kwargs: dict
        Standard keras layer kwargs

    References
    ----------
        - RMS Normalization - https://arxiv.org/abs/1910.07467
        - Official implementation - https://github.com/bzhangGo/rmsnorm
    """
    def __init__(self, axis=-1, epsilon=1e-8, partial=0.0, bias=False, **kwargs):
        self.scale = None
        self.offset = 0
        super().__init__(**kwargs)

        # Checks
        if not isinstance(axis, int):
            raise TypeError(f"Expected an int for the argument 'axis', but received: {axis}")

        if not 0.0 <= partial <= 1.0:
            raise ValueError(f"partial must be between 0.0 and 1.0, but received {partial}")

        self.axis = axis
        self.epsilon = epsilon
        self.partial = partial
        self.bias = bias
        self.offset = 0.

    def build(self, input_shape):
        """ Validate and populate :attr:`axis`

        Parameters
        ----------
        input_shape: tensor
            Keras tensor (future input to layer) or ``list``/``tuple`` of Keras tensors to
            reference for weight shape computations.
        """
        ndims = len(input_shape)
        if ndims is None:
            raise ValueError(f"Input shape {input_shape} has undefined rank.")

        # Resolve negative axis
        if self.axis < 0:
            self.axis += ndims

        # Validate axes
        if self.axis < 0 or self.axis >= ndims:
            raise ValueError(f"Invalid axis: {self.axis}")

        param_shape = [input_shape[self.axis]]
        self.scale = self.add_weight(
            name="scale",
            shape=param_shape,
            initializer="ones")
        if self.bias:
            self.offset = self.add_weight(
                name="offset",
                shape=param_shape,
                initializer="zeros")

        self.built = True  # pylint:disable=attribute-defined-outside-init

    def call(self, inputs, **kwargs):  # pylint:disable=unused-argument
        """ Call Root Mean Square Layer Normalization

        Parameters
        ----------
        inputs: tensor
            Input tensor, or list/tuple of input tensors

        Returns
        -------
        tensor
            A tensor or list/tuple of tensors
        """
        # Compute the axes along which to reduce the mean / variance
        input_shape = K.int_shape(inputs)
        layer_size = input_shape[self.axis]

        if self.partial in (0.0, 1.0):
            mean_square = K.mean(K.square(inputs), axis=self.axis, keepdims=True)
        else:
            partial_size = int(layer_size * self.partial)
            partial_x, _ = tf.split(  # pylint:disable=redundant-keyword-arg,no-value-for-parameter
                inputs,
                [partial_size, layer_size - partial_size],
                axis=self.axis)
            mean_square = K.mean(K.square(partial_x), axis=self.axis, keepdims=True)

        recip_square_root = tf.math.rsqrt(mean_square + self.epsilon)
        output = self.scale * inputs * recip_square_root + self.offset
        return output

    def compute_output_shape(self, input_shape):  # pylint:disable=no-self-use
        """ The output shape of the layer is the same as the input shape.

        Parameters
        ----------
        input_shape: tuple
            The input shape to the layer

        Returns
        -------
        tuple
            The output shape to the layer
        """
        return input_shape

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
        base_config = super().get_config()
        config = dict(axis=self.axis,
                      epsilon=self.epsilon,
                      partial=self.partial,
                      bias=self.bias)
        return dict(list(base_config.items()) + list(config.items()))


# Update normalization into Keras custom objects
for name, obj in inspect.getmembers(sys.modules[__name__]):
    if inspect.isclass(obj) and obj.__module__ == __name__:
        get_custom_objects().update({name: obj})
