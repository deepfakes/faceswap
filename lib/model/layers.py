#!/usr/bin/env python3
""" Custom Layers for faceswap.py. """

from __future__ import absolute_import

import sys
import inspect

import tensorflow as tf
import keras.backend as K

from keras.engine import InputSpec, Layer
from keras.utils import conv_utils
from keras.utils.generic_utils import get_custom_objects
from keras.layers.pooling import _GlobalPooling2D

from lib.utils import get_backend

if get_backend() == "amd":
    from lib.plaidml_utils import pad
else:
    from tensorflow import pad


class PixelShuffler(Layer):
    """ PixelShuffler layer for Keras.

    This layer requires a Convolution2D prior to it, having output filters computed according to
    the formula :math:`filters = k * (scale_factor * scale_factor)` where `k` is a user defined
    number of filters (generally larger than 32) and `scale_factor` is the up-scaling factor
    (generally 2).

    This layer performs the depth to space operation on the convolution filters, and returns a
    tensor with the size as defined below.

    Notes
    -----
    In practice, it is useful to have a second convolution layer after the
    :class:`PixelShuffler` layer to speed up the learning process. However, if you are stacking
    multiple :class:`PixelShuffler` blocks, it may increase the number of parameters greatly,
    so the Convolution layer after :class:`PixelShuffler` layer can be removed.

    Example
    -------
    >>> # A standard sub-pixel up-scaling block
    >>> x = Convolution2D(256, 3, 3, padding="same", activation="relu")(...)
    >>> u = PixelShuffler(size=(2, 2))(x)
    [Optional]
    >>> x = Convolution2D(256, 3, 3, padding="same", activation="relu")(u)

    Parameters
    ----------
    size: tuple, optional
        The (`h`, `w`) scaling factor for up-scaling. Default: `(2, 2)`
    data_format: ["channels_first", "channels_last", ``None``], optional
        The data format for the input. Default: ``None``
    kwargs: dict
        The standard Keras Layer keyword arguments (if any)

    References
    ----------
    https://gist.github.com/t-ae/6e1016cc188104d123676ccef3264981
    """
    def __init__(self, size=(2, 2), data_format=None, **kwargs):
        super().__init__(**kwargs)
        self.data_format = K.normalize_data_format(data_format)
        self.size = conv_utils.normalize_tuple(size, 2, "size")

    def call(self, inputs, **kwargs):
        """This is where the layer's logic lives.

        Parameters
        ----------
        inputs: tensor
            Input tensor, or list/tuple of input tensors
        kwargs: dict
            Additional keyword arguments

        Returns
        -------
        tensor
            A tensor or list/tuple of tensors
        """
        input_shape = K.int_shape(inputs)
        if len(input_shape) != 4:
            raise ValueError('Inputs should have rank ' +
                             str(4) +
                             '; Received input shape:', str(input_shape))

        if self.data_format == 'channels_first':
            batch_size, channels, height, width = input_shape
            if batch_size is None:
                batch_size = -1
            r_height, r_width = self.size
            o_height, o_width = height * r_height, width * r_width
            o_channels = channels // (r_height * r_width)

            out = K.reshape(inputs, (batch_size, r_height, r_width, o_channels, height, width))
            out = K.permute_dimensions(out, (0, 3, 4, 1, 5, 2))
            out = K.reshape(out, (batch_size, o_channels, o_height, o_width))
        elif self.data_format == 'channels_last':
            batch_size, height, width, channels = input_shape
            if batch_size is None:
                batch_size = -1
            r_height, r_width = self.size
            o_height, o_width = height * r_height, width * r_width
            o_channels = channels // (r_height * r_width)

            out = K.reshape(inputs, (batch_size, height, width, r_height, r_width, o_channels))
            out = K.permute_dimensions(out, (0, 1, 3, 2, 4, 5))
            out = K.reshape(out, (batch_size, o_height, o_width, o_channels))
        return out

    def compute_output_shape(self, input_shape):
        """Computes the output shape of the layer.

        Assumes that the layer will be built to match that input shape provided.

        Parameters
        ----------
        input_shape: tuple or list of tuples
            Shape tuple (tuple of integers) or list of shape tuples (one per output tensor of the
            layer).  Shape tuples can include None for free dimensions, instead of an integer.

        Returns
        -------
        tuple
            An input shape tuple
        """
        if len(input_shape) != 4:
            raise ValueError('Inputs should have rank ' +
                             str(4) +
                             '; Received input shape:', str(input_shape))

        if self.data_format == 'channels_first':
            height = None
            width = None
            if input_shape[2] is not None:
                height = input_shape[2] * self.size[0]
            if input_shape[3] is not None:
                width = input_shape[3] * self.size[1]
            channels = input_shape[1] // self.size[0] // self.size[1]

            if channels * self.size[0] * self.size[1] != input_shape[1]:
                raise ValueError('channels of input and size are incompatible')

            retval = (input_shape[0],
                      channels,
                      height,
                      width)
        elif self.data_format == 'channels_last':
            height = None
            width = None
            if input_shape[1] is not None:
                height = input_shape[1] * self.size[0]
            if input_shape[2] is not None:
                width = input_shape[2] * self.size[1]
            channels = input_shape[3] // self.size[0] // self.size[1]

            if channels * self.size[0] * self.size[1] != input_shape[3]:
                raise ValueError('channels of input and size are incompatible')

            retval = (input_shape[0],
                      height,
                      width,
                      channels)
        return retval

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
        config = {'size': self.size,
                  'data_format': self.data_format}
        base_config = super(PixelShuffler, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))


class SubPixelUpscaling(Layer):
    """ Sub-pixel convolutional up-scaling layer.

    This layer requires a Convolution2D prior to it, having output filters computed according to
    the formula :math:`filters = k * (scale_factor * scale_factor)` where `k` is a user defined
    number of filters (generally larger than 32) and `scale_factor` is the up-scaling factor
    (generally 2).

    This layer performs the depth to space operation on the convolution filters, and returns a
    tensor with the size as defined below.

    Notes
    -----
    This method is deprecated as it just performs the same as :class:`PixelShuffler`
    using explicit Tensorflow ops. The method is kept in the repository to support legacy
    models that have been created with this layer.

    In practice, it is useful to have a second convolution layer after the
    :class:`SubPixelUpscaling` layer to speed up the learning process. However, if you are stacking
    multiple :class:`SubPixelUpscaling` blocks, it may increase the number of parameters greatly,
    so the Convolution layer after :class:`SubPixelUpscaling` layer can be removed.

    Example
    -------
    >>> # A standard sub-pixel up-scaling block
    >>> x = Convolution2D(256, 3, 3, padding="same", activation="relu")(...)
    >>> u = SubPixelUpscaling(scale_factor=2)(x)
    [Optional]
    >>> x = Convolution2D(256, 3, 3, padding="same", activation="relu")(u)

    Parameters
    ----------
    size: int, optional
        The up-scaling factor. Default: `2`
    data_format: ["channels_first", "channels_last", ``None``], optional
        The data format for the input. Default: ``None``
    kwargs: dict
        The standard Keras Layer keyword arguments (if any)

    References
    ----------
    based on the paper "Real-Time Single Image and Video Super-Resolution Using an Efficient
    Sub-Pixel Convolutional Neural Network" (https://arxiv.org/abs/1609.05158).
    """

    def __init__(self, scale_factor=2, data_format=None, **kwargs):
        super(SubPixelUpscaling, self).__init__(**kwargs)

        self.scale_factor = scale_factor
        self.data_format = K.normalize_data_format(data_format)

    def build(self, input_shape):
        """Creates the layer weights.

        Must be implemented on all layers that have weights.

        Parameters
        ----------
        input_shape: tensor
            Keras tensor (future input to layer) or ``list``/``tuple`` of Keras tensors to
            reference for weight shape computations.
        """
        pass

    def call(self, input_tensor, mask=None):  # pylint:disable=unused-argument,arguments-differ
        """This is where the layer's logic lives.

        Parameters
        ----------
        inputs: tensor
            Input tensor, or list/tuple of input tensors
        kwargs: dict
            Additional keyword arguments

        Returns
        -------
        tensor
            A tensor or list/tuple of tensors
        """
        retval = self._depth_to_space(input_tensor, self.scale_factor, self.data_format)
        return retval

    def compute_output_shape(self, input_shape):
        """Computes the output shape of the layer.

        Assumes that the layer will be built to match that input shape provided.

        Parameters
        ----------
        input_shape: tuple or list of tuples
            Shape tuple (tuple of integers) or list of shape tuples (one per output tensor of the
            layer).  Shape tuples can include None for free dimensions, instead of an integer.

        Returns
        -------
        tuple
            An input shape tuple
        """
        if self.data_format == "channels_first":
            batch, channels, rows, columns = input_shape
            return (batch,
                    channels // (self.scale_factor ** 2),
                    rows * self.scale_factor,
                    columns * self.scale_factor)
        batch, rows, columns, channels = input_shape
        return (batch,
                rows * self.scale_factor,
                columns * self.scale_factor,
                channels // (self.scale_factor ** 2))

    @classmethod
    def _depth_to_space(cls, ipt, scale, data_format=None):
        """ Uses phase shift algorithm to convert channels/depth for spatial resolution """
        if data_format is None:
            data_format = K.image_data_format()
        data_format = data_format.lower()
        ipt = cls._preprocess_conv2d_input(ipt, data_format)
        out = tf.depth_to_space(ipt, scale)
        out = cls._postprocess_conv2d_output(out, data_format)
        return out

    @staticmethod
    def _postprocess_conv2d_output(input_tensor, data_format):
        """Transpose and cast the output from conv2d if needed.

        Parameters
        ----------
        input_tensor: tensor
            The input that requires transposing and casting
        data_format: str
            `"channels_last"` or `"channels_first"`

        Returns
        -------
        tensor
            The transposed and cast input tensor
        """

        if data_format == "channels_first":
            input_tensor = tf.transpose(input_tensor, (0, 3, 1, 2))

        if K.floatx() == "float64":
            input_tensor = tf.cast(input_tensor, "float64")
        return input_tensor

    @staticmethod
    def _preprocess_conv2d_input(input_tensor, data_format):
        """Transpose and cast the input before the conv2d.

        Parameters
        ----------
        input_tensor: tensor
            The input that requires transposing and casting
        data_format: str
            `"channels_last"` or `"channels_first"`

        Returns
        -------
        tensor
            The transposed and cast input tensor
        """
        if K.dtype(input_tensor) == "float64":
            input_tensor = tf.cast(input_tensor, "float32")
        if data_format == "channels_first":
            # Tensorflow uses the last dimension as channel dimension, instead of the 2nd one.
            # Theano input shape: (samples, input_depth, rows, cols)
            # Tensorflow input shape: (samples, rows, cols, input_depth)
            input_tensor = tf.transpose(input_tensor, (0, 2, 3, 1))
        return input_tensor

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
        config = {"scale_factor": self.scale_factor,
                  "data_format": self.data_format}
        base_config = super(SubPixelUpscaling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ReflectionPadding2D(Layer):
    """Reflection-padding layer for 2D input (e.g. picture).

    This layer can add rows and columns at the top, bottom, left and right side of an image tensor.

    Parameters
    ----------
    stride: int, optional
        The stride of the following convolution. Default: `2`
    kernel_size: int, optional
        The kernel size of the following convolution. Default: `5`
    kwargs: dict
        The standard Keras Layer keyword arguments (if any)
    """
    def __init__(self, stride=2, kernel_size=5, **kwargs):
        self.stride = stride
        self.kernel_size = kernel_size
        super().__init__(**kwargs)

    def build(self, input_shape):
        """Creates the layer weights.

        Must be implemented on all layers that have weights.

        Parameters
        ----------
        input_shape: tensor
            Keras tensor (future input to layer) or ``list``/``tuple`` of Keras tensors to
            reference for weight shape computations.
        """
        self.input_spec = [InputSpec(shape=input_shape)]
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        """Computes the output shape of the layer.

        Assumes that the layer will be built to match that input shape provided.

        Parameters
        ----------
        input_shape: tuple or list of tuples
            Shape tuple (tuple of integers) or list of shape tuples (one per output tensor of the
            layer).  Shape tuples can include None for free dimensions, instead of an integer.

        Returns
        -------
        tuple
            An input shape tuple
        """
        input_shape = self.input_spec[0].shape
        in_width, in_height = input_shape[2], input_shape[1]
        kernel_width, kernel_height = self.kernel_size, self.kernel_size

        if (in_height % self.stride) == 0:
            padding_height = max(kernel_height - self.stride, 0)
        else:
            padding_height = max(kernel_height - (in_height % self.stride), 0)
        if (in_width % self.stride) == 0:
            padding_width = max(kernel_width - self.stride, 0)
        else:
            padding_width = max(kernel_width - (in_width % self.stride), 0)

        return (input_shape[0],
                input_shape[1] + padding_height,
                input_shape[2] + padding_width,
                input_shape[3])

    def call(self, x, mask=None):  # pylint:disable=unused-argument,arguments-differ
        """This is where the layer's logic lives.

        Parameters
        ----------
        inputs: tensor
            Input tensor, or list/tuple of input tensors
        kwargs: dict
            Additional keyword arguments

        Returns
        -------
        tensor
            A tensor or list/tuple of tensors
        """
        input_shape = self.input_spec[0].shape
        in_width, in_height = input_shape[2], input_shape[1]
        kernel_width, kernel_height = self.kernel_size, self.kernel_size

        if (in_height % self.stride) == 0:
            padding_height = max(kernel_height - self.stride, 0)
        else:
            padding_height = max(kernel_height - (in_height % self.stride), 0)
        if (in_width % self.stride) == 0:
            padding_width = max(kernel_width - self.stride, 0)
        else:
            padding_width = max(kernel_width - (in_width % self.stride), 0)

        padding_top = padding_height // 2
        padding_bot = padding_height - padding_top
        padding_left = padding_width // 2
        padding_right = padding_width - padding_left

        return pad(x,
                   [[0, 0],
                    [padding_top, padding_bot],
                    [padding_left, padding_right],
                    [0, 0]],
                   'REFLECT')

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
        config = {'stride': self.stride,
                  'kernel_size': self.kernel_size}
        base_config = super(ReflectionPadding2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class GlobalMinPooling2D(_GlobalPooling2D):
    """Global minimum pooling operation for spatial data. """

    def call(self, inputs):
        """This is where the layer's logic lives.

        Parameters
        ----------
        inputs: tensor
            Input tensor, or list/tuple of input tensors
        kwargs: dict
            Additional keyword arguments

        Returns
        -------
        tensor
            A tensor or list/tuple of tensors
        """
        if self.data_format == 'channels_last':
            pooled = K.min(inputs, axis=[1, 2])
        else:
            pooled = K.min(inputs, axis=[2, 3])
        return pooled


class GlobalStdDevPooling2D(_GlobalPooling2D):
    """Global standard deviation pooling operation for spatial data. """

    def call(self, inputs):
        """This is where the layer's logic lives.

        Parameters
        ----------
        inputs: tensor
            Input tensor, or list/tuple of input tensors
        kwargs: dict
            Additional keyword arguments

        Returns
        -------
        tensor
            A tensor or list/tuple of tensors
        """
        if self.data_format == 'channels_last':
            pooled = K.std(inputs, axis=[1, 2])
        else:
            pooled = K.std(inputs, axis=[2, 3])
        return pooled


class L2_normalize(Layer):  # Pylint:disable=invalid-name
    """ Normalizes a tensor w.r.t. the L2 norm alongside the specified axis.

    Parameters
    ----------
    axis: int
        The axis to perform normalization across
    kwargs: dict
        The standard Keras Layer keyword arguments (if any)
    """
    def __init__(self, axis, **kwargs):
        self.axis = axis
        super(L2_normalize, self).__init__(**kwargs)

    def call(self, inputs):  # pylint:disable=arguments-differ
        """This is where the layer's logic lives.

        Parameters
        ----------
        inputs: tensor
            Input tensor, or list/tuple of input tensors
        kwargs: dict
            Additional keyword arguments

        Returns
        -------
        tensor
            A tensor or list/tuple of tensors
        """
        return K.l2_normalize(inputs, self.axis)

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
        config = super(L2_normalize, self).get_config()
        config["axis"] = self.axis
        return config


# Update layers into Keras custom objects
for name, obj in inspect.getmembers(sys.modules[__name__]):
    if inspect.isclass(obj) and obj.__module__ == __name__:
        get_custom_objects().update({name: obj})
