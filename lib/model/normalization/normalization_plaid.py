#!/usr/bin/env python3
""" Normalization methods for faceswap.py. """

import sys
import inspect

from plaidml.op import slice_tensor
from keras.layers import Layer
from keras import initializers, regularizers, constraints
from keras import backend as K
from keras.utils import get_custom_objects


class LayerNormalization(Layer):
    """Instance normalization layer (Lei Ba et al, 2016). Implementation adapted from
    tensorflow.keras implementation and https://github.com/CyberZHG/keras-layer-normalization

    Normalize the activations of the previous layer for each given example in a batch
    independently, rather than across a batch like Batch Normalization. i.e. applies a
    transformation that maintains the mean activation within each example close to 0 and the
    activation standard deviation close to 1.

    Parameters
    ----------
    axis: int or list/tuple
        The axis or axes to normalize across. Typically this is the features axis/axes.
        The left-out axes are typically the batch axis/axes. This argument defaults to `-1`, the
        last dimension in the input.
    epsilon: float, optional
        Small float added to variance to avoid dividing by zero. Default: `1e-3`
    center: bool, optional
        If ``True``, add offset of `beta` to normalized tensor. If ``False``, `beta` is ignored.
        Default: ``True``
    scale: bool, optional
        If ``True``, multiply by `gamma`. If ``False``, `gamma` is not used. When the next layer
        is linear (also e.g. `relu`), this can be disabled since the scaling will be done by
        the next layer. Default: ``True``
    beta_initializer: str, optional
        Initializer for the beta weight. Default: `"zeros"`
    gamma_initializer: str, optional
        Initializer for the gamma weight. Default: `"ones"`
    beta_regularizer: str, optional
        Optional regularizer for the beta weight. Default: ``None``
    gamma_regularizer: str, optional
        Optional regularizer for the gamma weight. Default: ``None``
    beta_constraint: float, optional
        Optional constraint for the beta weight. Default: ``None``
    gamma_constraint: float, optional
        Optional constraint for the gamma weight. Default: ``None``
    kwargs: dict
        Standard keras layer kwargs

    References
    ----------
        - Layer Normalization - https://arxiv.org/abs/1607.06450
        - Keras implementation - https://github.com/CyberZHG/keras-layer-normalization
    """
    def __init__(self,
                 axis=-1,
                 epsilon=1e-3,
                 center=True,
                 scale=True,
                 beta_initializer="zeros",
                 gamma_initializer="ones",
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 **kwargs):

        self.gamma = None
        self.beta = None
        super().__init__(**kwargs)

        if isinstance(axis, (list, tuple)):
            self.axis = axis[:]
        elif isinstance(axis, int):
            self.axis = axis
        else:
            raise TypeError("Expected an int or a list/tuple of ints for the argument 'axis', "
                            f"but received: {axis}")

        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma_constraint = constraints.get(gamma_constraint)
        self.supports_masking = True

    def build(self, input_shape):
        """Creates the layer weights.

        Parameters
        ----------
        input_shape: tensor
            Keras tensor (future input to layer) or ``list``/``tuple`` of Keras tensors to
            reference for weight shape computations.
        """
        ndims = len(input_shape)
        if ndims is None:
            raise ValueError(f"Input shape {input_shape} has undefined rank.")

        # Convert axis to list and resolve negatives
        if isinstance(self.axis, int):
            self.axis = [self.axis]
        elif isinstance(self.axis, tuple):
            self.axis = list(self.axis)
        for idx, axs in enumerate(self.axis):
            if axs < 0:
                self.axis[idx] = ndims + axs

        # Validate axes
        for axs in self.axis:
            if axs < 0 or axs >= ndims:
                raise ValueError(f"Invalid axis: {axs}")
        if len(self.axis) != len(set(self.axis)):
            raise ValueError("Duplicate axis: {}".format(tuple(self.axis)))

        param_shape = [input_shape[dim] for dim in self.axis]
        if self.scale:
            self.gamma = self.add_weight(
                name="gamma",
                shape=param_shape,
                initializer=self.gamma_initializer,
                regularizer=self.gamma_regularizer,
                constraint=self.gamma_constraint)
        if self.center:
            self.beta = self.add_weight(
                name='beta',
                shape=param_shape,
                initializer=self.beta_initializer,
                regularizer=self.beta_regularizer,
                constraint=self.beta_constraint)

        self.built = True  # pylint:disable=attribute-defined-outside-init

    def call(self, inputs, **kwargs):  # pylint:disable=unused-argument
        """This is where the layer's logic lives.

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
        ndims = len(input_shape)

        # Broadcasting only necessary for norm when the axis is not just the last dimension
        broadcast_shape = [1] * ndims
        for dim in self.axis:
            broadcast_shape[dim] = input_shape[dim]

        def _broadcast(var):
            if (var is not None and len(var.shape) != ndims and self.axis != [ndims - 1]):
                return K.reshape(var, broadcast_shape)
            return var

        # Calculate the moments on the last axis (layer activations).
        mean = K.mean(inputs, self.axis, keepdims=True)
        variance = K.mean(K.square(inputs - mean), axis=self.axis, keepdims=True)
        std = K.sqrt(variance + self.epsilon)
        outputs = (inputs - mean) / std

        scale, offset = _broadcast(self.gamma), _broadcast(self.beta)
        if self.scale:
            outputs *= scale
        if self.center:
            outputs *= offset

        return outputs

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
                      center=self.center,
                      scale=self.scale,
                      beta_initializer=initializers.serialize(self.beta_initializer),
                      gamma_initializer=initializers.serialize(self.gamma_initializer),
                      beta_regularizer=regularizers.serialize(self.beta_regularizer),
                      gamma_regularizer=regularizers.serialize(self.gamma_regularizer),
                      beta_constraint=constraints.serialize(self.beta_constraint),
                      gamma_constraint=constraints.serialize(self.gamma_constraint))
        return dict(list(base_config.items()) + list(config.items()))


class RMSNormalization(Layer):
    """ Root Mean Square Layer Normalization (Biao Zhang, Rico Sennrich, 2019)

    RMSNorm is a simplification of the original layer normalization (LayerNorm). LayerNorm is a
    regularization technique that might handle the internal covariate shift issue so as to
    stabilize the layer activations and improve model convergence. It has been proved quite
    successful in NLP-based model. In some cases, LayerNorm has become an essential component
    to enable model optimization, such as in the SOTA NMT model Transformer.

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
    def __init__(self, axis=-1, epsilon=1e-8, partial=-1.0, bias=False, **kwargs):
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
            partial_x = slice_tensor(inputs,
                                     axes=[self.axis],
                                     starts=[0],
                                     ends=[partial_size])
            mean_square = K.mean(K.square(partial_x), axis=self.axis, keepdims=True)

        recip_square_root = 1. / K.sqrt(mean_square + self.epsilon)
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
