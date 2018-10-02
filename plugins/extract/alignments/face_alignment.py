#!/usr/bin/env python3
""" Facial landmarks extractor for faceswap.py
    Code adapted and modified from:
    https://github.com/1adrianb/face-alignment
"""
import os

import cv2
import numpy as np
import tensorflow as tf

import keras
from keras import backend as K


class Alignments():
    """ Perform transformation to align and get landmarks """
    def __init__(self):
        self.verbose = None
        self.image = None
        self.detected_faces = None
        self.keras = None

        self.landmarks = self.process_landmarks()

    @staticmethod
    def transform(point, center, scale, resolution):
        """ Transform Image """
        pnt = np.array([point[0], point[1], 1.0])
        hscl = 200.0 * scale
        eye = np.eye(3)
        eye[0, 0] = resolution / hscl
        eye[1, 1] = resolution / hscl
        eye[0, 2] = resolution * (-center[0] / hscl + 0.5)
        eye[1, 2] = resolution * (-center[1] / hscl + 0.5)
        eye = np.linalg.inv(eye)
        return np.matmul(eye, pnt)[0:2]

    def crop(self, image, center, scale, resolution=256.0):
        """ Crop image around the center point """
        v_ul = self.transform([1, 1], center, scale, resolution).astype(np.int)
        v_br = self.transform([resolution, resolution],
                              center,
                              scale,
                              resolution).astype(np.int)
        if image.ndim > 2:
            new_dim = np.array([v_br[1] - v_ul[1],
                                v_br[0] - v_ul[0],
                                image.shape[2]],
                               dtype=np.int32)
            new_img = np.zeros(new_dim, dtype=np.uint8)
        else:
            new_dim = np.array([v_br[1] - v_ul[1],
                                v_br[0] - v_ul[0]],
                               dtype=np.int)
            new_img = np.zeros(new_dim, dtype=np.uint8)
        height = image.shape[0]
        width = image.shape[1]
        new_x = np.array([max(1, -v_ul[0] + 1), min(v_br[0], width) - v_ul[0]],
                         dtype=np.int32)
        new_y = np.array([max(1, -v_ul[1] + 1),
                          min(v_br[1], height) - v_ul[1]],
                         dtype=np.int32)
        old_x = np.array([max(1, v_ul[0] + 1), min(v_br[0], width)],
                         dtype=np.int32)
        old_y = np.array([max(1, v_ul[1] + 1), min(v_br[1], height)],
                         dtype=np.int32)
        new_img[new_y[0] - 1:new_y[1],
                new_x[0] - 1:new_x[1]] = image[old_y[0] - 1:old_y[1],
                                               old_x[0] - 1:old_x[1], :]
        new_img = cv2.resize(new_img,
                             dsize=(int(resolution), int(resolution)),
                             interpolation=cv2.INTER_LINEAR)
        return new_img

    def get_pts_from_predict(self, var_a, center, scale):
        """ Get points from predictor """
        var_b = var_a.reshape((var_a.shape[0],
                               var_a.shape[1] * var_a.shape[2]))
        var_c = var_b.argmax(1).reshape((var_a.shape[0],
                                         1)).repeat(2,
                                                    axis=1).astype(np.float)
        var_c[:, 0] %= var_a.shape[2]
        var_c[:, 1] = np.apply_along_axis(
            lambda x: np.floor(x / var_a.shape[2]),
            0,
            var_c[:, 1])

        for i in range(var_a.shape[0]):
            pt_x, pt_y = int(var_c[i, 0]), int(var_c[i, 1])
            if pt_x > 0 and pt_x < 63 and pt_y > 0 and pt_y < 63:
                diff = np.array([var_a[i, pt_y, pt_x+1]
                                 - var_a[i, pt_y, pt_x-1],
                                 var_a[i, pt_y+1, pt_x]
                                 - var_a[i, pt_y-1, pt_x]])

                var_c[i] += np.sign(diff)*0.25

        var_c += 0.5
        return [self.transform(var_c[i], center, scale, var_a.shape[2])
                for i in range(var_a.shape[0])]

    def process_landmarks(self):
        """ Align image and process landmarks """
        landmarks = list()
        if not self.detected_faces:
            if self.verbose:
                print("Warning: No faces were detected.")
            return landmarks

        for detected_face in self.detected_faces:

            center, scale = self.get_center_scale(detected_face)
            image = self.align_image(center, scale)

            landmarks_xy = self.predict_landmarks(image, center, scale)

            landmarks.append(((detected_face['left'],
                               detected_face['top'],
                               detected_face['right'],
                               detected_face['bottom']),
                              landmarks_xy))

        return landmarks

    @staticmethod
    def get_center_scale(detected_face):
        """ Get the center and set scale of bounding box """
        center = np.array([(detected_face['left']
                            + detected_face['right']) / 2.0,
                           (detected_face['top']
                            + detected_face['bottom']) / 2.0])

        center[1] -= (detected_face['bottom']
                      - detected_face['top']) * 0.12

        scale = (detected_face['right']
                 - detected_face['left']
                 + detected_face['bottom']
                 - detected_face['top']) / 195.0

        return center, scale

    def align_image(self, center, scale):
        """ Crop and align image around center """
        image = self.crop(
            self.image,
            center,
            scale).transpose((2, 0, 1)).astype(np.float32) / 255.0

        return np.expand_dims(image, 0)

    def predict_landmarks(self, image, center, scale):
        """ Predict the 68 point landmarks """
        with self.keras.session.as_default():
            pts_img = self.get_pts_from_predict(
                self.keras.model.predict(image)[-1][0],
                center,
                scale)

        return [(int(pt[0]), int(pt[1])) for pt in pts_img]


class TorchBatchNorm2D(keras.engine.base_layer.Layer):
    """ FAN model for face alignment
        Code adapted and modified from:
        https://github.com/1adrianb/face-alignment """
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

        keras_graph = tf.Graph()
        with keras_graph.as_default():
            config = tf.ConfigProto()
            if ratio:
                config.gpu_options.per_process_gpu_memory_fraction = ratio
            self.session = tf.Session(config=config)
            with self.session.as_default():
                self.model = keras.models.load_model(
                    self.model_path,
                    custom_objects={'TorchBatchNorm2D':
                                    TorchBatchNorm2D})
                self.model.predict(dummy)
        keras_graph.finalize()

        self.initialized = True
