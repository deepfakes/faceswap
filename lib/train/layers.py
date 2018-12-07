#!/usr/bin/env python3
""" Custom Layers for faceswap.py
    Layers from:
        shoanlu GAN: https://github.com/shaoanlu/faceswap-GAN"""


from keras.layers.core import Layer
from keras.engine import InputSpec
from keras import backend as K
from keras import initializers


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
