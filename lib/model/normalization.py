#!/usr/bin/env python3
""" Normalization methods for faceswap.py. """

import sys
import inspect

from keras.layers import Layer, InputSpec
from keras import initializers, regularizers, constraints
from keras import backend as K
from keras.utils import get_custom_objects


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

        - Instance Normalization: The Missing Ingredient for Fast Stylization - \
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


class AdaInstanceNormalization(Layer):
    """ Adaptive Instance Normalization Layer for Keras.

    Parameters
    ----------
    axis: int, optional
        The axis that should be normalized (typically the features axis). For instance, after a
        `Conv2D` layer with `data_format="channels_first"`, set `axis=1` in
        :class:`InstanceNormalization`. Setting `axis=None` will normalize all values in each
        instance of the batch. Axis 0 is the batch dimension. `axis` cannot be set to 0 to avoid
        errors. Default: ``None``
    momentum: float, optional
        Momentum for the moving mean and the moving variance. Default: `0.99`
    epsilon: float, optional
        Small float added to variance to avoid dividing by zero. Default: `1e-3`
    center: bool, optional
        If ``True``, add offset of `beta` to normalized tensor. If ``False``, `beta` is ignored.
        Default: ``True``
    scale: bool, optional
        If ``True``, multiply by `gamma`. If ``False``, `gamma` is not used. When the next layer
        is linear (also e.g. `relu`), this can be disabled since the scaling will be done by
        the next layer. Default: ``True``

    References
    ----------
        Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization - \
        https://arxiv.org/abs/1703.06868
    """
    def __init__(self, axis=-1, momentum=0.99, epsilon=1e-3, center=True, scale=True, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis
        self.momentum = momentum
        self.epsilon = epsilon
        self.center = center
        self.scale = scale

    def build(self, input_shape):
        """Creates the layer weights.

        Parameters
        ----------
        input_shape: tensor
            Keras tensor (future input to layer) or ``list``/``tuple`` of Keras tensors to
            reference for weight shape computations.
        """
        dim = input_shape[0][self.axis]
        if dim is None:
            raise ValueError('Axis ' + str(self.axis) + ' of '
                             'input tensor should have a defined dimension '
                             'but the layer received an input with shape ' +
                             str(input_shape[0]) + '.')

        super(AdaInstanceNormalization, self).build(input_shape)

    def call(self, inputs, training=None):  # pylint:disable=unused-argument
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
        input_shape = K.int_shape(inputs[0])
        reduction_axes = list(range(0, len(input_shape)))

        beta = inputs[1]
        gamma = inputs[2]

        if self.axis is not None:
            del reduction_axes[self.axis]

        del reduction_axes[0]
        mean = K.mean(inputs[0], reduction_axes, keepdims=True)
        stddev = K.std(inputs[0], reduction_axes, keepdims=True) + self.epsilon
        normed = (inputs[0] - mean) / stddev

        return normed * gamma + beta

    def get_config(self):
        """Returns the config of the layer.

        The Keras configuration for the layer.

        Returns
        --------
        dict
            A python dictionary containing the layer configuration
        """
        config = {
            'axis': self.axis,
            'momentum': self.momentum,
            'epsilon': self.epsilon,
            'center': self.center,
            'scale': self.scale
        }
        base_config = super(AdaInstanceNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        """ Calculate the output shape from this layer.

        Parameters
        ----------
        input_shape: tuple
            The input shape to the layer

        Returns
        -------
        int
            The output shape to the layer
        """
        return input_shape[0]


class GroupNormalization(Layer):
    """ Group Normalization

    Parameters
    ----------
    axis: int, optional
        The axis that should be normalized (typically the features axis). For instance, after a
        `Conv2D` layer with `data_format="channels_first"`, set `axis=1` in
        :class:`InstanceNormalization`. Setting `axis=None` will normalize all values in each
        instance of the batch. Axis 0 is the batch dimension. `axis` cannot be set to 0 to avoid
        errors. Default: ``None``
    gamma_init: str, optional
        Initializer for the gamma weight. Default: `"one"`
    beta_init: str, optional
        Initializer for the beta weight. Default `"zero"`
    gamma_regularizer: varies, optional
        Optional regularizer for the gamma weight. Default: ``None``
    beta_regularizer:  varies, optional
        Optional regularizer for the beta weight. Default ``None``
    epsilon: float, optional
        Small float added to variance to avoid dividing by zero. Default: `1e-3`
    group: int, optional
        The group size. Default: `32`
    data_format: ["channels_first", "channels_last"], optional
        The required data format. Optional. Default: ``None``
    kwargs: dict
        Any additional standard Keras Layer key word arguments

    References
    ----------
    Shaoanlu GAN: https://github.com/shaoanlu/faceswap-GAN
    """
    def __init__(self, axis=-1, gamma_init='one', beta_init='zero', gamma_regularizer=None,
                 beta_regularizer=None, epsilon=1e-6, group=32, data_format=None, **kwargs):
        self.beta = None
        self.gamma = None
        super(GroupNormalization, self).__init__(**kwargs)
        self.axis = axis if isinstance(axis, (list, tuple)) else [axis]
        self.gamma_init = initializers.get(gamma_init)
        self.beta_init = initializers.get(beta_init)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.epsilon = epsilon
        self.group = group
        self.data_format = K.normalize_data_format(data_format)

        self.supports_masking = True

    def build(self, input_shape):
        """Creates the layer weights.

        Parameters
        ----------
        input_shape: tensor
            Keras tensor (future input to layer) or ``list``/``tuple`` of Keras tensors to
            reference for weight shape computations.
        """
        self.input_spec = [InputSpec(shape=input_shape)]
        shape = [1 for _ in input_shape]
        if self.data_format == 'channels_last':
            channel_axis = -1
            shape[channel_axis] = input_shape[channel_axis]
        elif self.data_format == 'channels_first':
            channel_axis = 1
            shape[channel_axis] = input_shape[channel_axis]
        # for i in self.axis:
        #    shape[i] = input_shape[i]
        self.gamma = self.add_weight(shape=shape,
                                     initializer=self.gamma_init,
                                     regularizer=self.gamma_regularizer,
                                     name='gamma')
        self.beta = self.add_weight(shape=shape,
                                    initializer=self.beta_init,
                                    regularizer=self.beta_regularizer,
                                    name='beta')
        self.built = True

    def call(self, inputs, mask=None):  # pylint: disable=unused-argument
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
        if len(input_shape) != 4 and len(input_shape) != 2:
            raise ValueError('Inputs should have rank ' +
                             str(4) + " or " + str(2) +
                             '; Received input shape:', str(input_shape))

        if len(input_shape) == 4:
            if self.data_format == 'channels_last':
                batch_size, height, width, channels = input_shape
                if batch_size is None:
                    batch_size = -1

                if channels < self.group:
                    raise ValueError('Input channels should be larger than group size' +
                                     '; Received input channels: ' + str(channels) +
                                     '; Group size: ' + str(self.group))

                var_x = K.reshape(inputs, (batch_size,
                                           height,
                                           width,
                                           self.group,
                                           channels // self.group))
                mean = K.mean(var_x, axis=[1, 2, 4], keepdims=True)
                std = K.sqrt(K.var(var_x, axis=[1, 2, 4], keepdims=True) + self.epsilon)
                var_x = (var_x - mean) / std

                var_x = K.reshape(var_x, (batch_size, height, width, channels))
                retval = self.gamma * var_x + self.beta
            elif self.data_format == 'channels_first':
                batch_size, channels, height, width = input_shape
                if batch_size is None:
                    batch_size = -1

                if channels < self.group:
                    raise ValueError('Input channels should be larger than group size' +
                                     '; Received input channels: ' + str(channels) +
                                     '; Group size: ' + str(self.group))

                var_x = K.reshape(inputs, (batch_size,
                                           self.group,
                                           channels // self.group,
                                           height,
                                           width))
                mean = K.mean(var_x, axis=[2, 3, 4], keepdims=True)
                std = K.sqrt(K.var(var_x, axis=[2, 3, 4], keepdims=True) + self.epsilon)
                var_x = (var_x - mean) / std

                var_x = K.reshape(var_x, (batch_size, channels, height, width))
                retval = self.gamma * var_x + self.beta

        elif len(input_shape) == 2:
            reduction_axes = list(range(0, len(input_shape)))
            del reduction_axes[0]
            batch_size, _ = input_shape
            if batch_size is None:
                batch_size = -1

            mean = K.mean(inputs, keepdims=True)
            std = K.sqrt(K.var(inputs, keepdims=True) + self.epsilon)
            var_x = (inputs - mean) / std

            retval = self.gamma * var_x + self.beta
        return retval

    def get_config(self):
        """Returns the config of the layer.

        The Keras configuration for the layer.

        Returns
        --------
        dict
            A python dictionary containing the layer configuration
        """
        config = {'epsilon': self.epsilon,
                  'axis': self.axis,
                  'gamma_init': initializers.serialize(self.gamma_init),
                  'beta_init': initializers.serialize(self.beta_init),
                  'gamma_regularizer': regularizers.serialize(self.gamma_regularizer),
                  'beta_regularizer': regularizers.serialize(self.gamma_regularizer),
                  'group': self.group}
        base_config = super(GroupNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


# Update normalization into Keras custom objects
for name, obj in inspect.getmembers(sys.modules[__name__]):
    if inspect.isclass(obj) and obj.__module__ == __name__:
        get_custom_objects().update({name: obj})
