#!/usr/bin/env python3
""" Facial landmarks extractor for faceswap.py
    Code adapted and modified from:
    https://github.com/1adrianb/face-alignment
"""
from __future__ import annotations
import logging
import typing as T

import cv2
import numpy as np

from lib.model.session import KSession
from ._base import Aligner, AlignerBatch, BatchType

if T.TYPE_CHECKING:
    from lib.align import DetectedFace

logger = logging.getLogger(__name__)


class Align(Aligner):
    """ Perform transformation to align and get landmarks """
    def __init__(self, **kwargs) -> None:
        git_model_id = 13
        model_filename = "face-alignment-network_2d4_keras_v2.h5"
        super().__init__(git_model_id=git_model_id, model_filename=model_filename, **kwargs)
        self.model: KSession
        self.name = "FAN"
        self.input_size = 256
        self.color_format = "RGB"
        self.vram = 2240
        self.vram_warnings = 512  # Will run at this with warnings
        self.vram_per_batch = 64
        self.realign_centering = "head"
        self.batchsize: int = self.config["batch-size"]
        self.reference_scale = 200. / 195.

    def init_model(self) -> None:
        """ Initialize FAN model """
        assert isinstance(self.name, str)
        assert isinstance(self.model_path, str)
        self.model = KSession(self.name,
                              self.model_path,
                              allow_growth=self.config["allow_growth"],
                              exclude_gpus=self._exclude_gpus)
        self.model.load_model()
        # Feed a placeholder so Aligner is primed for Manual tool
        placeholder_shape = (self.batchsize, self.input_size, self.input_size, 3)
        placeholder = np.zeros(placeholder_shape, dtype="float32")
        self.model.predict(placeholder)

    def faces_to_feed(self, faces: np.ndarray) -> np.ndarray:
        """ Convert a batch of face images from UINT8 (0-255) to fp32 (0.0-1.0)

        Parameters
        ----------
        faces: :class:`numpy.ndarray`
            The batch of faces in UINT8 format

        Returns
        -------
        class: `numpy.ndarray`
            The batch of faces as fp32 in 0.0 to 1.0 range
        """
        return faces.astype("float32") / 255.

    def process_input(self, batch: BatchType) -> None:
        """ Compile the detected faces for prediction

        Parameters
        ----------
        batch: :class:`AlignerBatch`
            The current batch to process input for
        """
        assert isinstance(batch, AlignerBatch)
        logger.trace("Aligning faces around center")  # type:ignore[attr-defined]
        center_scale = self.get_center_scale(batch.detected_faces)
        batch.feed = np.array(self.crop(batch, center_scale))[..., :3]
        batch.data.append({"center_scale": center_scale})
        logger.trace("Aligned image around center")  # type:ignore[attr-defined]

    def get_center_scale(self, detected_faces: list[DetectedFace]) -> np.ndarray:
        """ Get the center and set scale of bounding box

        Parameters
        ----------
        detected_faces: list
            List of :class:`~lib.align.DetectedFace` objects for the batch

        Returns
        -------
        :class:`numpy.ndarray`
            The center and scale of the bounding box
        """
        logger.trace("Calculating center and scale")  # type:ignore[attr-defined]
        center_scale = np.empty((len(detected_faces), 68, 3), dtype='float32')
        for index, face in enumerate(detected_faces):
            x_ctr = (T.cast(int, face.left) + face.right) / 2.0
            y_ctr = (T.cast(int, face.top) + face.bottom) / 2.0 - T.cast(int, face.height) * 0.12
            scale = (T.cast(int, face.width) + T.cast(int, face.height)) * self.reference_scale
            center_scale[index, :, 0] = np.full(68, x_ctr, dtype='float32')
            center_scale[index, :, 1] = np.full(68, y_ctr, dtype='float32')
            center_scale[index, :, 2] = np.full(68, scale, dtype='float32')
        logger.trace("Calculated center and scale: %s", center_scale)  # type:ignore[attr-defined]
        return center_scale

    def _crop_image(self,
                    image: np.ndarray,
                    top_left: np.ndarray,
                    bottom_right: np.ndarray) -> np.ndarray:
        """ Crop a single image

        Parameters
        ----------
        image: :class:`numpy.ndarray`
            The image to crop
        top_left: :class:`numpy.ndarray`
            The top left (x, y) point to crop from
        bottom_right: :class:`numpy.ndarray`
            The bottom right (x, y) point to crop to

        Returns
        -------
        :class:`numpy.ndarray`
            The cropped image
        """
        bottom_right_width, bottom_right_height = bottom_right[0].astype('int32')
        top_left_width, top_left_height = top_left[0].astype('int32')
        new_dim = (bottom_right_height - top_left_height,
                   bottom_right_width - top_left_width,
                   3 if image.ndim > 2 else 1)
        new_img = np.zeros(new_dim, dtype=np.uint8)

        new_x = slice(max(0, -top_left_width),
                      min(bottom_right_width, image.shape[1]) - top_left_width)
        new_y = slice(max(0, -top_left_height),
                      min(bottom_right_height, image.shape[0]) - top_left_height)
        old_x = slice(max(0, top_left_width), min(bottom_right_width, image.shape[1]))
        old_y = slice(max(0, top_left_height), min(bottom_right_height, image.shape[0]))
        new_img[new_y, new_x] = image[old_y, old_x]

        interp = cv2.INTER_CUBIC if new_dim[0] < self.input_size else cv2.INTER_AREA
        return cv2.resize(new_img,
                          dsize=(self.input_size, self.input_size),
                          interpolation=interp)

    def crop(self, batch: AlignerBatch, center_scale: np.ndarray) -> list[np.ndarray]:
        """ Crop image around the center point

        Parameters
        ----------
        batch: :class:`AlignerBatch`
            The current batch to crop the image for
        center_scale: :class:`numpy.ndarray`
            The center and scale for the bounding box

        Returns
        -------
        list
            List of cropped images for the batch
        """
        logger.trace("Cropping images")  # type:ignore[attr-defined]
        batch_shape = center_scale.shape[:2]
        resolutions = np.full(batch_shape, self.input_size, dtype='float32')
        matrix_ones = np.ones(batch_shape + (3,), dtype='float32')
        matrix_size = np.full(batch_shape + (3,), self.input_size, dtype='float32')
        matrix_size[..., 2] = 1.0
        upper_left = self.transform(matrix_ones, center_scale, resolutions)
        bot_right = self.transform(matrix_size, center_scale, resolutions)

        # TODO second pass .. convert to matrix
        new_images = [self._crop_image(image, top_left, bottom_right)
                      for image, top_left, bottom_right in zip(batch.image, upper_left, bot_right)]
        logger.trace("Cropped images")  # type:ignore[attr-defined]
        return new_images

    @classmethod
    def transform(cls,
                  points: np.ndarray,
                  center_scales: np.ndarray,
                  resolutions: np.ndarray) -> np.ndarray:
        """ Transform Image

        Parameters
        ----------
        points: :class:`numpy.ndarray`
            The points to transform
        center_scales: :class:`numpy.ndarray`
            The calculated centers and scales for the batch
        resolutions: :class:`numpy.ndarray`
            The resolutions
        """
        logger.trace("Transforming Points")  # type:ignore[attr-defined]
        num_images, num_landmarks = points.shape[:2]
        transform_matrix = np.eye(3, dtype='float32')
        transform_matrix = np.repeat(transform_matrix[None, :], num_landmarks, axis=0)
        transform_matrix = np.repeat(transform_matrix[None, :, :], num_images, axis=0)
        scales = center_scales[:, :, 2] / resolutions
        translations = center_scales[..., 2:3] * -0.5 + center_scales[..., :2]
        transform_matrix[:, :, 0, 0] = scales  # x scale
        transform_matrix[:, :, 1, 1] = scales  # y scale
        transform_matrix[:, :, 0, 2] = translations[:, :, 0]  # x translation
        transform_matrix[:, :, 1, 2] = translations[:, :, 1]  # y translation
        new_points = np.einsum('abij, abj -> abi', transform_matrix, points, optimize='greedy')
        retval = new_points[:, :, :2].astype('float32')
        logger.trace("Transformed Points: %s", retval)  # type:ignore[attr-defined]
        return retval

    def predict(self, feed: np.ndarray) -> np.ndarray:
        """ Predict the 68 point landmarks

        Parameters
        ----------
        batch: :class:`numpy.ndarray`
            The batch to feed into the aligner

        Returns
        -------
        :class:`numpy.ndarray`
            The predictions from the aligner
        """
        logger.trace("Predicting Landmarks")  # type:ignore[attr-defined]
        # TODO Remove lazy transpose and change points from predict to use the correct
        # order
        retval = self.model.predict(feed)[-1].transpose(0, 3, 1, 2)
        logger.trace(retval.shape)  # type:ignore[attr-defined]
        return retval

    def process_output(self, batch: BatchType) -> None:
        """ Process the output from the model

        Parameters
        ----------
        batch: :class:`AlignerBatch`
            The current batch from the model with :attr:`predictions` populated
        """
        assert isinstance(batch, AlignerBatch)
        self.get_pts_from_predict(batch)

    def get_pts_from_predict(self, batch: AlignerBatch) -> None:
        """ Get points from predictor and populate the :attr:`landmarks` property of the
        :class:`AlignerBatch`

        Parameters
        ----------
        batch: :class:`AlignerBatch`
            The current batch from the model with :attr:`predictions` populated
        """
        logger.trace("Obtain points from prediction")  # type:ignore[attr-defined]
        num_images, num_landmarks = batch.prediction.shape[:2]
        image_slice = np.repeat(np.arange(num_images)[:, None], num_landmarks, axis=1)
        landmark_slice = np.repeat(np.arange(num_landmarks)[None, :], num_images, axis=0)
        resolution = np.full((num_images, num_landmarks), 64, dtype='int32')
        subpixel_landmarks = np.ones((num_images, num_landmarks, 3), dtype='float32')

        indices = np.array(np.unravel_index(batch.prediction.reshape(num_images,
                                                                     num_landmarks,
                                                                     -1).argmax(-1),
                                            (batch.prediction.shape[2],  # height
                                             batch.prediction.shape[3])))  # width
        min_clipped = np.minimum(indices + 1, batch.prediction.shape[2] - 1)
        max_clipped = np.maximum(indices - 1, 0)
        offsets = [(image_slice, landmark_slice, indices[0], min_clipped[1]),
                   (image_slice, landmark_slice, indices[0], max_clipped[1]),
                   (image_slice, landmark_slice, min_clipped[0], indices[1]),
                   (image_slice, landmark_slice, max_clipped[0], indices[1])]
        x_subpixel_shift = batch.prediction[offsets[0]] - batch.prediction[offsets[1]]
        y_subpixel_shift = batch.prediction[offsets[2]] - batch.prediction[offsets[3]]
        # TODO improve rudimentary sub-pixel logic to centroid of 3x3 window algorithm
        subpixel_landmarks[:, :, 0] = indices[1] + np.sign(x_subpixel_shift) * 0.25 + 0.5
        subpixel_landmarks[:, :, 1] = indices[0] + np.sign(y_subpixel_shift) * 0.25 + 0.5

        if batch.second_pass:  # Transformation handled by plugin parent for re-aligned faces
            batch.landmarks = subpixel_landmarks[..., :2] * 4.
        else:
            batch.landmarks = self.transform(subpixel_landmarks,
                                             batch.data[0]["center_scale"],
                                             resolution)
        logger.trace("Obtained points from prediction: %s",  # type:ignore[attr-defined]
                     batch.landmarks)
