def PenalizedLossClass(tf):
    class PenalizedLoss(object):
        def __init__(self,mask,lossFunc):
            self.lossFunc = lossFunc
            self.mask = mask
            
        def __call__(self,y_true, y_pred):
            return self.lossFunc (y_true*self.mask,y_pred*self.mask)
    return PenalizedLoss

def PixelShufflerClass(keras):
    class PixelShuffler(keras.engine.topology.Layer):
        def __init__(self, size=(2, 2), data_format=None, **kwargs):
            super(PixelShuffler, self).__init__(**kwargs)
            self.data_format = keras.utils.conv_utils.normalize_data_format(data_format)
            self.size = keras.utils.conv_utils.normalize_tuple(size, 2, 'size')

        def call(self, inputs):

            input_shape = keras.backend.int_shape(inputs)
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

                out = keras.backend.reshape(inputs, (batch_size, rh, rw, oc, h, w))
                out = keras.backend.permute_dimensions(out, (0, 3, 4, 1, 5, 2))
                out = keras.backend.reshape(out, (batch_size, oc, oh, ow))
                return out

            elif self.data_format == 'channels_last':
                batch_size, h, w, c = input_shape
                if batch_size is None:
                    batch_size = -1
                rh, rw = self.size
                oh, ow = h * rh, w * rw
                oc = c // (rh * rw)

                out = keras.backend.reshape(inputs, (batch_size, h, w, rh, rw, oc))
                out = keras.backend.permute_dimensions(out, (0, 1, 3, 2, 4, 5))
                out = keras.backend.reshape(out, (batch_size, oh, ow, oc))
                return out

        def compute_output_shape(self, input_shape):

            if len(input_shape) != 4:
                raise ValueError('Inputs should have rank ' +
                                 str(4) +
                                 '; Received input shape:', str(input_shape))

            if self.data_format == 'channels_first':
                height = input_shape[2] * self.size[0] if input_shape[2] is not None else None
                width = input_shape[3] * self.size[1] if input_shape[3] is not None else None
                channels = input_shape[1] // self.size[0] // self.size[1]

                if channels * self.size[0] * self.size[1] != input_shape[1]:
                    raise ValueError('channels of input and size are incompatible')

                return (input_shape[0],
                        channels,
                        height,
                        width)

            elif self.data_format == 'channels_last':
                height = input_shape[1] * self.size[0] if input_shape[1] is not None else None
                width = input_shape[2] * self.size[1] if input_shape[2] is not None else None
                channels = input_shape[3] // self.size[0] // self.size[1]

                if channels * self.size[0] * self.size[1] != input_shape[3]:
                    raise ValueError('channels of input and size are incompatible')

                return (input_shape[0],
                        height,
                        width,
                        channels)

        def get_config(self):
            config = {'size': self.size,
                      'data_format': self.data_format}
            base_config = super(PixelShuffler, self).get_config()

            return dict(list(base_config.items()) + list(config.items()))
    return PixelShuffler

def conv(keras, input_tensor, filters):
    x = input_tensor
    x = keras.layers.convolutional.Conv2D(filters, kernel_size=5, strides=2, padding='same')(x)
    x = keras.layers.advanced_activations.LeakyReLU(0.1)(x)
    return x
    
def sepconv(keras, input_tensor, filters):
    x = input_tensor
    x = keras.layers.convolutional.SeparableConv2D(filters, kernel_size=5, strides=2, padding='same')(x)
    x = keras.layers.advanced_activations.LeakyReLU(0.1)(x)
    return x

def sepupscale(keras, input_tensor, filters):
    x = input_tensor
    x = keras.layers.convolutional.SeparableConv2D(filters * 4, kernel_size=3, padding='same')(x)
    x = keras.layers.advanced_activations.LeakyReLU(0.1)(x)
    x = PixelShufflerClass(keras)()(x)
    return x
    
def sepres(keras, input_tensor, filters):
    x = input_tensor
    x = keras.layers.convolutional.SeparableConv2D(filters, kernel_size=3, kernel_initializer=keras.initializers.RandomNormal(0, 0.02), use_bias=False, padding="same")(x)
    x = keras.layers.advanced_activations.LeakyReLU(alpha=0.2)(x)
    x = keras.layers.convolutional.SeparableConv2D(filters, kernel_size=3, kernel_initializer=keras.initializers.RandomNormal(0, 0.02), use_bias=False, padding="same")(x)
    x = keras.layers.Add()([x, input_tensor])
    x = keras.layers.advanced_activations.LeakyReLU(alpha=0.2)(x)
    return x
    
def upscale(keras, input_tensor, filters):
    x = input_tensor
    x = keras.layers.convolutional.Conv2D(filters * 4, kernel_size=3, padding='same')(x)
    x = keras.layers.advanced_activations.LeakyReLU(0.1)(x)
    x = PixelShufflerClass(keras)()(x)
    return x
    
def res(keras, input_tensor, filters):
    x = input_tensor
    x = keras.layers.convolutional.Conv2D(filters, kernel_size=3, kernel_initializer=keras.initializers.RandomNormal(0, 0.02), use_bias=False, padding="same")(x)
    x = keras.layers.advanced_activations.LeakyReLU(alpha=0.2)(x)
    x = keras.layers.convolutional.Conv2D(filters, kernel_size=3, kernel_initializer=keras.initializers.RandomNormal(0, 0.02), use_bias=False, padding="same")(x)
    x = keras.layers.Add()([x, input_tensor])
    x = keras.layers.advanced_activations.LeakyReLU(alpha=0.2)(x)
    return x