#!/usr/bin/env python3
""" Custom Layers for faceswap.py
    Layers from:
        the original https://www.reddit.com/r/deepfakes/ code sample + contribs
        shoanlu GAN: https://github.com/shaoanlu/faceswap-GAN"""

from __future__ import absolute_import

import sys
import inspect

import tensorflow as tf
import keras.backend as K

from keras.engine import InputSpec, Layer
from keras.utils import conv_utils
from keras.utils.generic_utils import get_custom_objects
from keras import initializers
from keras.layers import ZeroPadding2D

if K.backend() == "plaidml.keras.backend":
    from lib.plaidml_utils import pad
else:
    from tensorflow import pad 

class PixelShuffler(Layer):
    """ PixelShuffler layer for Keras
       by t-ae: https://gist.github.com/t-ae/6e1016cc188104d123676ccef3264981 """
    # pylint: disable=C0103
    def __init__(self, size=(2, 2), data_format=None, **kwargs):
        super(PixelShuffler, self).__init__(**kwargs)
        self.data_format = K.normalize_data_format(data_format)
        self.size = conv_utils.normalize_tuple(size, 2, 'size')

    def call(self, inputs, **kwargs):

        input_shape = K.int_shape(inputs)
        if len(input_shape) != 4:
            raise ValueError('Inputs should have rank ' +
                             str(4) +
                             '; Received input shape:', str(input_shape))

        if self.data_format == 'channels_first':
            batch_size, c, h, w = input_shape
            if batch_size is None:
                batch_size = -1
            rh, rw = self.size
            oh, ow = h * rh, w * rw
            oc = c // (rh * rw)

            out = K.reshape(inputs, (batch_size, rh, rw, oc, h, w))
            out = K.permute_dimensions(out, (0, 3, 4, 1, 5, 2))
            out = K.reshape(out, (batch_size, oc, oh, ow))
        elif self.data_format == 'channels_last':
            batch_size, h, w, c = input_shape
            if batch_size is None:
                batch_size = -1
            rh, rw = self.size
            oh, ow = h * rh, w * rw
            oc = c // (rh * rw)

            out = K.reshape(inputs, (batch_size, h, w, rh, rw, oc))
            out = K.permute_dimensions(out, (0, 1, 3, 2, 4, 5))
            out = K.reshape(out, (batch_size, oh, ow, oc))
        return out

    def compute_output_shape(self, input_shape):

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
        config = {'size': self.size,
                  'data_format': self.data_format}
        base_config = super(PixelShuffler, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))


class Scale(Layer):
    """
    GAN Custom Scal Layer
    Code borrows from https://github.com/flyyufelix/cnn_finetune
    """
    def __init__(self, weights=None, axis=-1, gamma_init='zero', **kwargs):
        self.axis = axis
        self.gamma_init = initializers.get(gamma_init)
        self.initial_weights = weights
        super(Scale, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]

        # Compatibility with TensorFlow >= 1.0.0
        self.gamma = K.variable(self.gamma_init((1,)), name='{}_gamma'.format(self.name))
        self.trainable_weights = [self.gamma]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def call(self, x, mask=None):
        return self.gamma * x

    def get_config(self):
        config = {"axis": self.axis}
        base_config = super(Scale, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class SubPixelUpscaling(Layer):
    # pylint: disable=C0103
    """ Sub-pixel convolutional upscaling layer based on the paper "Real-Time
    Single Image and Video Super-Resolution Using an Efficient Sub-Pixel
    Convolutional Neural Network" (https://arxiv.org/abs/1609.05158).
    This layer requires a Convolution2D prior to it, having output filters
    computed according to the formula :
        filters = k * (scale_factor * scale_factor)
        where k = a user defined number of filters (generally larger than 32)
              scale_factor = the upscaling factor (generally 2)
    This layer performs the depth to space operation on the convolution
    filters, and returns a tensor with the size as defined below.
    # Example :
    ```python
        # A standard subpixel upscaling block
        x = Convolution2D(256, 3, 3, padding="same", activation="relu")(...)
        u = SubPixelUpscaling(scale_factor=2)(x)
        [Optional]
        x = Convolution2D(256, 3, 3, padding="same", activation="relu")(u)
    ```
        In practice, it is useful to have a second convolution layer after the
        SubPixelUpscaling layer to speed up the learning process.
        However, if you are stacking multiple SubPixelUpscaling blocks,
        it may increase the number of parameters greatly, so the Convolution
        layer after SubPixelUpscaling layer can be removed.
    # Arguments
        scale_factor: Upscaling factor.
        data_format: Can be None, "channels_first" or "channels_last".
    # Input shape
        4D tensor with shape:
        `(samples, k * (scale_factor * scale_factor) channels, rows, cols)`
            if data_format="channels_first"
        or 4D tensor with shape:
        `(samples, rows, cols, k * (scale_factor * scale_factor) channels)`
            if data_format="channels_last".
    # Output shape
        4D tensor with shape:
        `(samples, k channels, rows * scale_factor, cols * scale_factor))`
            if data_format="channels_first"
        or 4D tensor with shape:
        `(samples, rows * scale_factor, cols * scale_factor, k channels)`
            if data_format="channels_last".
    """

    def __init__(self, scale_factor=2, data_format=None, **kwargs):
        super(SubPixelUpscaling, self).__init__(**kwargs)

        self.scale_factor = scale_factor
        self.data_format = K.normalize_data_format(data_format)

    def build(self, input_shape):
        pass

    def call(self, x, mask=None):
        y = self.depth_to_space(x, self.scale_factor, self.data_format)
        return y

    def compute_output_shape(self, input_shape):
        if self.data_format == "channels_first":
            b, k, r, c = input_shape
            return (b,
                    k // (self.scale_factor ** 2),
                    r * self.scale_factor,
                    c * self.scale_factor)
        b, r, c, k = input_shape
        return (b,
                r * self.scale_factor,
                c * self.scale_factor,
                k // (self.scale_factor ** 2))

    @classmethod
    def depth_to_space(cls, ipt, scale, data_format=None):
        """ Uses phase shift algorithm to convert channels/depth
            for spatial resolution """
        if data_format is None:
            data_format = K.image_data_format()
        data_format = data_format.lower()
        ipt = cls._preprocess_conv2d_input(ipt, data_format)
        out = tf.depth_to_space(ipt, scale)
        out = cls._postprocess_conv2d_output(out, data_format)
        return out

    @staticmethod
    def _postprocess_conv2d_output(x, data_format):
        """Transpose and cast the output from conv2d if needed.
        # Arguments
            x: A tensor.
            data_format: string, `"channels_last"` or `"channels_first"`.
        # Returns
            A tensor.
        """

        if data_format == "channels_first":
            x = tf.transpose(x, (0, 3, 1, 2))

        if K.floatx() == "float64":
            x = tf.cast(x, "float64")
        return x

    @staticmethod
    def _preprocess_conv2d_input(x, data_format):
        """Transpose and cast the input before the conv2d.
        # Arguments
            x: input tensor.
            data_format: string, `"channels_last"` or `"channels_first"`.
        # Returns
            A tensor.
        """
        if K.dtype(x) == "float64":
            x = tf.cast(x, "float32")
        if data_format == "channels_first":
            # TF uses the last dimension as channel dimension,
            # instead of the 2nd one.
            # TH input shape: (samples, input_depth, rows, cols)
            # TF input shape: (samples, rows, cols, input_depth)
            x = tf.transpose(x, (0, 2, 3, 1))
        return x

    def get_config(self):
        config = {"scale_factor": self.scale_factor,
                  "data_format": self.data_format}
        base_config = super(SubPixelUpscaling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ReflectionPadding2D(Layer):
    def __init__(self, stride=2, kernel_size=5, **kwargs):
        '''
        # Arguments
            stride: stride of following convolution (2)
            kernel_size: kernel size of following convolution (5,5)
        '''
        self.stride = stride
        self.kernel_size = kernel_size
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        super(ReflectionPadding2D, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        """ If you are using "channels_last" configuration"""
        input_shape = self.input_spec[0].shape
        in_width, in_height = input_shape[2], input_shape[1]
        kernel_width, kernel_height  = self.kernel_size, self.kernel_size

        if (in_height % self.stride == 0):
            padding_height = max(kernel_height - self.stride, 0)
        else:
            padding_height = max(kernel_height - (in_height % self.stride), 0)
        if (in_width % self.stride == 0):
            padding_width = max(kernel_width - self.stride, 0)
        else:
            padding_width = max(kernel_width- (in_width % self.stride), 0)

        return (input_shape[0],
                input_shape[1] + padding_height,
                input_shape[2] + padding_width,
                input_shape[3])

    def call(self, x, mask=None):
        input_shape = self.input_spec[0].shape
        in_width, in_height = input_shape[2], input_shape[1]
        kernel_width, kernel_height  = self.kernel_size, self.kernel_size

        if (in_height % self.stride == 0):
            padding_height = max(kernel_height - self.stride, 0)
        else:
            padding_height = max(kernel_height - (in_height % self.stride), 0)
        if (in_width % self.stride == 0):
            padding_width = max(kernel_width - self.stride, 0)
        else:
            padding_width = max(kernel_width- (in_width % self.stride), 0)

        padding_top = padding_height // 2
        padding_bot = padding_height - padding_top
        padding_left = padding_width // 2
        padding_right = padding_width - padding_left

        return pad(x, [[0,0],
                          [padding_top, padding_bot],
                          [padding_left, padding_right],
                          [0,0] ],
                          'REFLECT')

    def get_config(self):
        config = {'stride': self.stride,
                  'kernel_size': self.kernel_size}
        base_config = super(ReflectionPadding2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items())) 


# Update layers into Keras custom objects
for name, obj in inspect.getmembers(sys.modules[__name__]):
    if inspect.isclass(obj) and obj.__module__ == __name__:
        get_custom_objects().update({name: obj})
