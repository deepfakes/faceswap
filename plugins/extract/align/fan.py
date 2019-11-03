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
import timeit

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
        self.reference_scale = 200. / 195.

    def init_model(self):
        """ Initialize FAN model """
        model_kwargs = dict(custom_objects={'TorchBatchNorm2D': TorchBatchNorm2D})
        self.model = KSession(self.name,
                              self.model_path,
                              model_kwargs=model_kwargs,
                              allow_growth=self.config["allow_growth"])
        self.model.load_model()
        # Feed a placeholder so Aligner is primed for Manual tool
        placeholder = np.zeros((self.batchsize, 3, self.input_size, self.input_size), dtype="float32")
        self.model.predict(placeholder)

    def process_input(self, batch):
        """ Compile the detected faces for prediction """
        logger.trace("Aligning faces around center")
        batch["center_scale"] = self.get_center_scale(batch["detected_faces"])
        faces = self.crop(batch)
        # start test comparision
        def wrapper(func, *args, **kwargs):
            def wrapped():
                return func(*args, **kwargs)
            return wrapped

        wrapped = wrapper(self.crop, batch)
        timer = timeit.timeit(wrapped, number=1000)
        print("\nNew Time: ", timer)
        # end test comparision
        logger.trace("Aligned image around center")
        faces = self._normalize_faces(faces)
        batch["feed"] = np.array(faces, dtype="float32")[..., :3].transpose((0, 3, 1, 2)) / 255.0
        return batch

    def get_center_scale(self, detected_faces):
        """ Get the center and set scale of bounding box """
        logger.trace("Calculating center and scale")
        center_scale = np.empty((len(detected_faces), 68, 3), dtype='float32')
        for index, face in enumerate(detected_faces):
            x_center = (face.left + face.right) / 2.0
            y_center = (face.top + face.bottom) / 2.0 - face.h * 0.12
            scale = (face.w + face.h) * self.reference_scale
            center_scale[index, :, 0] = np.full(68, x_center, dtype='float32')
            center_scale[index, :, 1] = np.full(68, y_center, dtype='float32')
            center_scale[index, :, 2] = np.full(68, scale, dtype='float32')
        logger.trace("Calculated center and scale: %s, %s", center_scale)
        return center_scale

    def crop(self, batch):  # pylint:disable=too-many-locals
        """ Crop image around the center point """
        logger.trace("Cropping images")
        sizes = (self.input_size, self.input_size)
        batch_shape = batch["center_scale"].shape[:2]
        resolutions = np.full(batch_shape, self.input_size, dtype='float32')
        matrix_ones = np.ones(batch_shape + (3,), dtype='float32')
        matrix_size = np.full(batch_shape + (3,), self.input_size, dtype='float32')
        matrix_size[..., 2] = 1.0
        upper_left = self.transform(matrix_ones, batch["center_scale"], resolutions).astype(np.int)
        bot_right = self.transform(matrix_size, batch["center_scale"], resolutions).astype(np.int)

        new_images = []
        for face, ul, br in zip(batch["detected_faces"], upper_left, bot_right):
            height, width = face.image.shape[:2]
            channels = 3 if face.image.ndim > 2 else 1
            br_width, br_height = br[0]
            ul_width, ul_height = ul[0]
            new_dim = [br_height - ul_height, br_width - ul_width, channels]
            new_img = np.zeros(new_dim, dtype=np.uint8)

            new_x = np.array([max(0, -ul_width), min(br_width, width) - ul_width], dtype=np.int32)
            new_y = np.array([max(0, -ul_height), min(br_height, height) - ul_height], dtype=np.int32)
            old_x = np.array([max(0, ul_width), min(br_width, width)], dtype=np.int32)
            old_y = np.array([max(0, ul_height), min(br_height, height)], dtype=np.int32)

            new_img[new_y[0]:new_y[1], new_x[0]:new_x[1]] = face.image[old_y[0]:old_y[1], old_x[0]:old_x[1]]

            interp = cv2.INTER_CUBIC if new_dim[0] < self.input_size else cv2.INTER_AREA
            new_images.append(cv2.resize(new_img, dsize=sizes, interpolation=interp))
        logger.trace("Cropped images")
        return new_images

    @staticmethod
    def transform(points, center_scales, resolutions):
        """ Transform Image """
        logger.trace("Transforming Points")
        centers = center_scales[..., :2]
        scales = center_scales[..., 2]
        eye = np.eye(3, dtype='float32')
        mat_eye = np.stack([eye] * points.shape[1], axis=0)
        mat_eye = np.stack([mat_eye] * points.shape[0], axis=0)
        x_scale = scales / resolutions
        y_scale = scales / resolutions
        x_translation = scales * -0.5 + centers[:, :, 0]
        y_translation = scales * -0.5 + centers[:, :, 1]
        mat_eye[:, :, 0, 0] = x_scale
        mat_eye[:, :, 1, 1] = y_scale
        mat_eye[:, :, 0, 2] = x_translation
        mat_eye[:, :, 1, 2] = y_translation
        retval = np.einsum('abij, abj -> abi', mat_eye, points, optimize='greedy')[:, :, :2]
        logger.trace("Transformed Points: %s", retval)
        return retval.astype('float32')

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
        num_images, num_landmarks, height, width = batch["prediction"].shape
        image_slice = np.repeat(np.arange(num_images)[:, None], num_landmarks, axis=1)
        landmark_slice = np.repeat(np.arange(num_landmarks)[None, :], num_images, axis=0)
        width = np.full((num_images, num_landmarks), 64, dtype=np.int32)
        subpixel_landmarks = np.ones((num_images, num_landmarks, 3), dtype='float32')
        indices = np.empty((num_images, num_landmarks, 2), dtype='uint32')
        for index, prediction in enumerate(batch["prediction"]):
            indices[index] = np.array([np.unravel_index(np.argmax(heatmap, axis=None), heatmap.shape)
                                      for heatmap in prediction], dtype='uint32')

        offsets = [(image_slice, landmark_slice, indices[:, :, 0], indices[:, :, 1] + 1),
                   (image_slice, landmark_slice, indices[:, :, 0], indices[:, :, 1] - 1),
                   (image_slice, landmark_slice, indices[:, :, 0] + 1, indices[:, :, 1]),
                   (image_slice, landmark_slice, indices[:, :, 0] - 1, indices[:, :, 1])]
        x_subpixel_shift = np.sign(batch["prediction"][offsets[0]] - batch["prediction"][offsets[1]]) * 0.25
        y_subpixel_shift = np.sign(batch["prediction"][offsets[2]] - batch["prediction"][offsets[3]]) * 0.25
        subpixel_landmarks[:, :, 0] = indices[:, :, 1] + x_subpixel_shift + 0.5
        subpixel_landmarks[:, :, 1] = indices[:, :, 0] + y_subpixel_shift + 0.5
        batch["landmarks"] = self.transform(subpixel_landmarks, batch["center_scale"], width)
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
