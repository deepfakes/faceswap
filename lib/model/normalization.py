#!/usr/bin/env python3
""" Normalization methods for faceswap.py. """

import sys
import inspect

from keras.engine import Layer, InputSpec
from keras import initializers, regularizers, constraints
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects


class InstanceNormalization(Layer):
    """Instance normalization layer (Lei Ba et al, 2016, Ulyanov et al., 2016).

    Normalize the activations of the previous layer at each step, i.e. applies a transformation
    that maintains the mean activation close to 0 and the activation standard deviation close to 1.

    Parameters
    ----------
    axis: int, optional
        The axis that should be normalized (typically the features axis). For instance, after a
        `Conv2D` layer with `data_format="channels_first"`, set `axis=1` in
        :class:`InstanceNormalization`. Setting `axis=None` will normalize all values in each
        instance of the batch. Axis 0 is the batch dimension. `axis` cannot be set to 0 to avoid
        errors. Default: ``None``
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

    References
    ----------
        - Layer Normalization - https://arxiv.org/abs/1607.06450

        - Instance Normalization: The Missing Ingredient for Fast Stylization -
    https://arxiv.org/abs/1607.08022

    """
    def __init__(self,
                 axis=None,
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
        self.beta = None
        self.gamma = None
        super().__init__(**kwargs)
        self.supports_masking = True
        self.axis = axis
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma_constraint = constraints.get(gamma_constraint)

    def build(self, input_shape):
        """Creates the layer weights.

        Must be implemented on all layers that have weights.

        Parameters
        ----------
        input_shape: tensor
            Keras tensor (future input to layer) or ``list``/``tuple`` of Keras tensors to
            reference for weight shape computations.
        """
        ndim = len(input_shape)
        if self.axis == 0:
            raise ValueError("Axis cannot be zero")

        if (self.axis is not None) and (ndim == 2):
            raise ValueError("Cannot specify axis for rank 1 tensor")

        self.input_spec = InputSpec(ndim=ndim)

        if self.axis is None:
            shape = (1,)
        else:
            shape = (input_shape[self.axis],)

        if self.scale:
            self.gamma = self.add_weight(shape=shape,
                                         name="gamma",
                                         initializer=self.gamma_initializer,
                                         regularizer=self.gamma_regularizer,
                                         constraint=self.gamma_constraint)
        else:
            self.gamma = None
        if self.center:
            self.beta = self.add_weight(shape=shape,
                                        name="beta",
                                        initializer=self.beta_initializer,
                                        regularizer=self.beta_regularizer,
                                        constraint=self.beta_constraint)
        else:
            self.beta = None
        self.built = True

    def call(self, inputs, training=None):  # pylint:disable=arguments-differ,unused-argument
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
        input_shape = K.int_shape(inputs)
        reduction_axes = list(range(0, len(input_shape)))

        if self.axis is not None:
            del reduction_axes[self.axis]

        del reduction_axes[0]

        mean = K.mean(inputs, reduction_axes, keepdims=True)
        stddev = K.std(inputs, reduction_axes, keepdims=True) + self.epsilon
        normed = (inputs - mean) / stddev

        broadcast_shape = [1] * len(input_shape)
        if self.axis is not None:
            broadcast_shape[self.axis] = input_shape[self.axis]

        if self.scale:
            broadcast_gamma = K.reshape(self.gamma, broadcast_shape)
            normed = normed * broadcast_gamma
        if self.center:
            broadcast_beta = K.reshape(self.beta, broadcast_shape)
            normed = normed + broadcast_beta
        return normed

    def get_config(self):
        config = {
            "axis": self.axis,
            "epsilon": self.epsilon,
            "center": self.center,
            "scale": self.scale,
            "beta_initializer": initializers.serialize(self.beta_initializer),
            "gamma_initializer": initializers.serialize(self.gamma_initializer),
            "beta_regularizer": regularizers.serialize(self.beta_regularizer),
            "gamma_regularizer": regularizers.serialize(self.gamma_regularizer),
            "beta_constraint": constraints.serialize(self.beta_constraint),
            "gamma_constraint": constraints.serialize(self.gamma_constraint)
        }
        base_config = super(InstanceNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


# Update normalization into Keras custom objects
for name, obj in inspect.getmembers(sys.modules[__name__]):
    if inspect.isclass(obj) and obj.__module__ == __name__:
        get_custom_objects().update({name: obj})
