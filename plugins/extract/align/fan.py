#!/usr/bin/env python3
""" Facial landmarks extractor for faceswap.py
    Code adapted and modified from:
    https://github.com/1adrianb/face-alignment
"""
import cv2
import numpy as np
import keras
from keras import backend as K

from lib.model.session import KSession
from ._base import Aligner, logger


class Align(Aligner):
    """ Perform transformation to align and get landmarks """
    def __init__(self, **kwargs):
        git_model_id = 9
        model_filename = "face-alignment-network_2d4_keras_v1.h5"
        super().__init__(git_model_id=git_model_id, model_filename=model_filename, **kwargs)
        self.name = "FAN"
        self.input_size = 256
        self.colorformat = "RGB"
        self.vram = 2240
        self.vram_warnings = 512  # Will run at this with warnings
        self.vram_per_batch = 64
        self.batchsize = self.config["batch-size"]
        self.reference_scale = 195

    def init_model(self):
        """ Initialize FAN model """
        model_kwargs = dict(custom_objects={'TorchBatchNorm2D': TorchBatchNorm2D})
        self.model = KSession(self.name,
                              self.model_path,
                              model_kwargs=model_kwargs,
                              allow_growth=self.config["allow_growth"])
        self.model.load_model()
        # Feed a placeholder so Aligner is primed for Manual tool
        placeholder = np.zeros((self.batchsize, 3, self.input_size, self.input_size),
                               dtype="float32")
        self.model.predict(placeholder)

    def process_input(self, batch):
        """ Compile the detected faces for prediction """
        # TODO Batching
        logger.trace("Aligning faces around center")
        batch["center_scale"] = self.get_center_scale(batch["detected_faces"])
        faces = self.crop(batch)
        logger.trace("Aligned image around center")
        faces = self._normalize_faces(faces)
        batch["feed"] = np.array(faces, dtype="float32").transpose((0, 3, 1, 2)) / 255.0
        return batch

    def get_center_scale(self, detected_faces):
        """ Get the center and set scale of bounding box """
        logger.trace("Calculating center and scale")
        l_center = []
        l_scale = []
        for face in detected_faces:
            center = np.array([(face.left + face.right) / 2.0, (face.top + face.bottom) / 2.0])
            center[1] -= face.h * 0.12
            l_center.append(center)
            l_scale.append((face.w + face.h) / self.reference_scale)
        logger.trace("Calculated center and scale: %s, %s", l_center, l_scale)
        return l_center, l_scale

    def crop(self, batch):  # pylint:disable=too-many-locals
        """ Crop image around the center point """
        logger.trace("Cropping images")
        new_images = []
        for face, center, scale in zip(batch["detected_faces"], *batch["center_scale"]):
            is_color = face.image.ndim > 2
            v_ul = self.transform([1, 1], center, scale, self.input_size).astype(np.int)
            v_br = self.transform([self.input_size, self.input_size],
                                  center,
                                  scale,
                                  self.input_size).astype(np.int)
            if is_color:
                new_dim = np.array([v_br[1] - v_ul[1],
                                    v_br[0] - v_ul[0],
                                    face.image.shape[2]],
                                   dtype=np.int32)
                new_img = np.zeros(new_dim, dtype=np.uint8)
            else:
                new_dim = np.array([v_br[1] - v_ul[1],
                                    v_br[0] - v_ul[0]],
                                   dtype=np.int)
                new_img = np.zeros(new_dim, dtype=np.uint8)
            height = face.image.shape[0]
            width = face.image.shape[1]
            new_x = np.array([max(1, -v_ul[0] + 1), min(v_br[0], width) - v_ul[0]],
                             dtype=np.int32)
            new_y = np.array([max(1, -v_ul[1] + 1),
                              min(v_br[1], height) - v_ul[1]],
                             dtype=np.int32)
            old_x = np.array([max(1, v_ul[0] + 1), min(v_br[0], width)],
                             dtype=np.int32)
            old_y = np.array([max(1, v_ul[1] + 1), min(v_br[1], height)],
                             dtype=np.int32)
            if is_color:
                new_img[new_y[0] - 1:new_y[1],
                        new_x[0] - 1:new_x[1]] = face.image[old_y[0] - 1:old_y[1],
                                                            old_x[0] - 1:old_x[1], :]
            else:
                new_img[new_y[0] - 1:new_y[1],
                        new_x[0] - 1:new_x[1]] = face.image[old_y[0] - 1:old_y[1],
                                                            old_x[0] - 1:old_x[1]]

            if new_img.shape[0] < self.input_size:
                interpolation = cv2.INTER_CUBIC  # pylint:disable=no-member
            else:
                interpolation = cv2.INTER_AREA  # pylint:disable=no-member

            new_images.append(cv2.resize(new_img,  # pylint:disable=no-member
                                         dsize=(int(self.input_size), int(self.input_size)),
                                         interpolation=interpolation))
        logger.trace("Cropped images")
        return new_images

    @staticmethod
    def transform(point, center, scale, resolution):
        """ Transform Image """
        logger.trace("Transforming Points")
        pnt = np.array([point[0], point[1], 1.0])
        hscl = 200.0 * scale
        eye = np.eye(3)
        eye[0, 0] = resolution / hscl
        eye[1, 1] = resolution / hscl
        eye[0, 2] = resolution * (-center[0] / hscl + 0.5)
        eye[1, 2] = resolution * (-center[1] / hscl + 0.5)
        eye = np.linalg.inv(eye)
        retval = np.matmul(eye, pnt)[0:2]
        logger.trace("Transformed Points: %s", retval)
        return retval

    def predict(self, batch):
        """ Predict the 68 point landmarks """
        logger.trace("Predicting Landmarks")
        batch["prediction"] = self.model.predict(batch["feed"])[-1]
        logger.trace([pred.shape for pred in batch["prediction"]])
        return batch

    def process_output(self, batch):
        """ Process the output from the model """
        self.get_pts_from_predict(batch)
        return batch

    def get_pts_from_predict(self, batch):
        """ Get points from predictor """
        logger.trace("Obtain points from prediction")
        landmarks = []
        for prediction, center, scale in zip(batch["prediction"], *batch["center_scale"]):
            var_b = prediction.reshape((prediction.shape[0],
                                        prediction.shape[1] * prediction.shape[2]))
            var_c = var_b.argmax(1).reshape((prediction.shape[0],
                                             1)).repeat(2,
                                                        axis=1).astype(np.float)
            var_c[:, 0] %= prediction.shape[2]
            var_c[:, 1] = np.apply_along_axis(
                lambda x: np.floor(x / prediction.shape[2]),
                0,
                var_c[:, 1])

            for i in range(prediction.shape[0]):
                pt_x, pt_y = int(var_c[i, 0]), int(var_c[i, 1])
                if 63 > pt_x > 0 and 63 > pt_y > 0:
                    diff = np.array([prediction[i, pt_y, pt_x+1]
                                     - prediction[i, pt_y, pt_x-1],
                                     prediction[i, pt_y+1, pt_x]
                                     - prediction[i, pt_y-1, pt_x]])

                    var_c[i] += np.sign(diff)*0.25

            var_c += 0.5
            landmarks = [self.transform(var_c[i], center, scale, prediction.shape[2])
                         for i in range(prediction.shape[0])]
            batch.setdefault("landmarks", []).append(landmarks)
        logger.trace("Obtained points from prediction: %s", batch["landmarks"])


class TorchBatchNorm2D(keras.engine.base_layer.Layer):
    # pylint:disable=too-many-instance-attributes
    """" Required for FAN_keras model """
    def __init__(self, axis=-1, momentum=0.99, epsilon=1e-3, **kwargs):
        super(TorchBatchNorm2D, self).__init__(**kwargs)
        self.supports_masking = True
        self.axis = axis
        self.momentum = momentum
        self.epsilon = epsilon
        self._epsilon_const = K.constant(self.epsilon, dtype='float32')

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
        invstd = (
            K.ones(shape=broadcast_shape, dtype='float32')
            / K.sqrt(broadcast_moving_variance + self._epsilon_const)
        )

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
