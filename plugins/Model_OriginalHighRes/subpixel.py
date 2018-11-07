from __future__ import absolute_import

import keras.backend as K
import tensorflow as tf
from keras.engine import Layer
from keras.utils.generic_utils import get_custom_objects
try:
    from keras.utils.conv_utils import normalize_data_format
except ImportError:
    from keras.backend import normalize_data_format



class SubPixelUpscaling(Layer):
    """ Sub-pixel convolutional upscaling layer based on the paper "Real-Time Single Image
    and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network"
    (https://arxiv.org/abs/1609.05158).
    This layer requires a Convolution2D prior to it, having output filters computed according to
    the formula :
        filters = k * (scale_factor * scale_factor)
        where k = a user defined number of filters (generally larger than 32)
              scale_factor = the upscaling factor (generally 2)
    This layer performs the depth to space operation on the convolution filters, and returns a
    tensor with the size as defined below.
    # Example :
    ```python
        # A standard subpixel upscaling block
        x = Convolution2D(256, 3, 3, padding='same', activation='relu')(...)
        u = SubPixelUpscaling(scale_factor=2)(x)
        [Optional]
        x = Convolution2D(256, 3, 3, padding='same', activation='relu')(u)
    ```
        In practice, it is useful to have a second convolution layer after the
        SubPixelUpscaling layer to speed up the learning process.
        However, if you are stacking multiple SubPixelUpscaling blocks, it may increase
        the number of parameters greatly, so the Convolution layer after SubPixelUpscaling
        layer can be removed.
    # Arguments
        scale_factor: Upscaling factor.
        data_format: Can be None, 'channels_first' or 'channels_last'.
    # Input shape
        4D tensor with shape:
        `(samples, k * (scale_factor * scale_factor) channels, rows, cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(samples, rows, cols, k * (scale_factor * scale_factor) channels)` if data_format='channels_last'.
    # Output shape
        4D tensor with shape:
        `(samples, k channels, rows * scale_factor, cols * scale_factor))` if data_format='channels_first'
        or 4D tensor with shape:
        `(samples, rows * scale_factor, cols * scale_factor, k channels)` if data_format='channels_last'.
    """

    def __init__(self, scale_factor=2, data_format=None, **kwargs):
        super(SubPixelUpscaling, self).__init__(**kwargs)

        self.scale_factor = scale_factor
        self.data_format = normalize_data_format(data_format)

    def build(self, input_shape):
        pass

    def call(self, x, mask=None):
        y = self.depth_to_space(x, self.scale_factor, self.data_format)
        return y        

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_first':
            b, k, r, c = input_shape
            return (b, k // (self.scale_factor ** 2), r * self.scale_factor, c * self.scale_factor)
        else:
            b, r, c, k = input_shape
            return (b, r * self.scale_factor, c * self.scale_factor, k // (self.scale_factor ** 2))
        
    @classmethod
    def depth_to_space(cls, ipt, scale, data_format=None):
        ''' Uses phase shift algorithm to convert channels/depth for spatial resolution '''
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
    
        if data_format == 'channels_first':
            x = tf.transpose(x, (0, 3, 1, 2))
    
        if K.floatx() == 'float64':
            x = tf.cast(x, 'float64')
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
        if K.dtype(x) == 'float64':
            x = tf.cast(x, 'float32')
        if data_format == 'channels_first':
            # TF uses the last dimension as channel dimension,
            # instead of the 2nd one.
            # TH input shape: (samples, input_depth, rows, cols)
            # TF input shape: (samples, rows, cols, input_depth)
            x = tf.transpose(x, (0, 2, 3, 1))
        return x 

    def get_config(self):
        config = {'scale_factor': self.scale_factor,
                  'data_format': self.data_format}
        base_config = super(SubPixelUpscaling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


get_custom_objects().update({'SubPixelUpscaling': SubPixelUpscaling})