#!/usr/bin python3
""" FAN model for face alignment
    Code adapted and modified from:
    https://github.com/1adrianb/face-alignment """

import os

import keras
from keras import backend as K
from tensorflow import ConfigProto, Graph, Session


class TorchBatchNorm2D(keras.engine.base_layer.Layer):
    """" Keras Model """
    def __init__(self, axis=-1, momentum=0.99, epsilon=1e-3, **kwargs):
        super(TorchBatchNorm2D, self).__init__(**kwargs)
        self.supports_masking = True
        self.axis = axis
        self.momentum = momentum
        self.epsilon = epsilon

        self.built = False
        self.gamma = None
        self.beta = None
        self.moving_mean = None
        self.moving_variance = None

    def build(self, input_shape):
        dim = input_shape[self.axis]
        if dim is None:
            raise ValueError("Axis {} of input tensor should have a "
                             "defined dimension but the layer received "
                             "an input with  shape {}."
                             .format(str(self.axis), str(input_shape)))
        shape = (dim,)
        self.gamma = self.add_weight(shape=shape,
                                     name='gamma',
                                     initializer='ones',
                                     regularizer=None,
                                     constraint=None)
        self.beta = self.add_weight(shape=shape,
                                    name='beta',
                                    initializer='zeros',
                                    regularizer=None,
                                    constraint=None)
        self.moving_mean = self.add_weight(shape=shape,
                                           name='moving_mean',
                                           initializer='zeros',
                                           trainable=False)
        self.moving_variance = self.add_weight(shape=shape,
                                               name='moving_variance',
                                               initializer='ones',
                                               trainable=False)
        self.built = True

    def call(self, inputs, **kwargs):
        input_shape = K.int_shape(inputs)

        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[self.axis] = input_shape[self.axis]

        broadcast_moving_mean = K.reshape(self.moving_mean, broadcast_shape)
        broadcast_moving_variance = K.reshape(self.moving_variance,
                                              broadcast_shape)
        broadcast_gamma = K.reshape(self.gamma, broadcast_shape)
        broadcast_beta = K.reshape(self.beta, broadcast_shape)
        invstd = (K.ones(shape=broadcast_shape,
                         dtype='float32')
                  / K.sqrt(broadcast_moving_variance
                           + K.constant(self.epsilon,
                                        dtype='float32')))

        return((inputs - broadcast_moving_mean)
               * invstd
               * broadcast_gamma
               + broadcast_beta)

    def get_config(self):
        config = {'axis': self.axis,
                  'momentum': self.momentum,
                  'epsilon': self.epsilon}
        base_config = super(TorchBatchNorm2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class KerasModel(object):
    "Load the Keras Model"
    def __init__(self):
        self.initialized = False
        self.verbose = False
        self.model_path = self.set_model_path()
        self.model = None
        self.session = None

    @staticmethod
    def set_model_path():
        """ Set the path to the Face Alignment Network Model """
        model_path = os.path.join(os.path.dirname(__file__),
                                  ".cache", "2DFAN-4.h5")
        if not os.path.exists(model_path):
            raise Exception("Error: Unable to find {}, "
                            "reinstall the lib!".format(model_path))
        return model_path

    def load_model(self, verbose, dummy, ratio):
        """ Load the Keras Model """
        self.verbose = verbose
        if self.verbose:
            print("Initializing keras model...")

        keras_graph = Graph()
        with keras_graph.as_default():
            config = ConfigProto()
            if ratio:
                config.gpu_options.per_process_gpu_memory_fraction = ratio
            self.session = Session(config=config)
            with self.session.as_default():
                self.model = keras.models.load_model(
                    self.model_path,
                    custom_objects={'TorchBatchNorm2D':
                                    TorchBatchNorm2D})
                self.model.predict(dummy)
        keras_graph.finalize()

        self.initialized = True
