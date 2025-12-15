#!/usr/bin/env python3
""" VGG Obstructed face mask plugin """
from __future__ import annotations
import logging
import typing as T

import numpy as np

from keras import layers as kl, Model

from lib.logger import parse_class_init
from lib.utils import get_module_objects
from ._base import BatchType, Masker, MaskerBatch
from . import vgg_obstructed_defaults as cfg

if T.TYPE_CHECKING:
    from keras import KerasTensor

logger = logging.getLogger(__name__)

# pylint:disable=duplicate-code


class Mask(Masker):
    """ Neural network to process face image into a segmentation mask of the face """
    def __init__(self, **kwargs) -> None:
        git_model_id = 5
        model_filename = "Nirkin_500_softmax_v1.h5"
        super().__init__(git_model_id=git_model_id, model_filename=model_filename, **kwargs)
        self.model: VGGObstructed
        self.name = "VGG Obstructed"
        self.input_size = 500
        self.vram = 1728  # 1710 in testing
        self.vram_per_batch = 896  # ~886 in testing
        self.batchsize = cfg.batch_size()

    def init_model(self) -> None:
        assert isinstance(self.model_path, str)
        self.model = VGGObstructed(self.model_path, self.batchsize)
        placeholder = np.zeros((self.batchsize, self.input_size, self.input_size, 3),
                               dtype="float32")
        self.model(placeholder)

    def process_input(self, batch: BatchType) -> None:
        """ Compile the detected faces for prediction """
        assert isinstance(batch, MaskerBatch)
        input_ = [T.cast(np.ndarray, feed.face)[..., :3] for feed in batch.feed_faces]
        batch.feed = input_ - np.mean(input_, axis=(1, 2))[:, None, None, :]
        logger.trace("feed shape: %s", batch.feed.shape)  # type:ignore[attr-defined]

    def predict(self, feed: np.ndarray) -> np.ndarray:
        """ Run model to get predictions """
        predictions = self.model(feed)
        assert isinstance(predictions, np.ndarray)
        return predictions[..., 0] * -1.0 + 1.0

    def process_output(self, batch: BatchType) -> None:
        """ Compile found faces for output """
        return


class VGGObstructed():
    """ VGG Obstructed mask for Faceswap.

    Caffe model re-implemented in Keras by Kyle Vrooman.
    Re-implemented for Keras by TorzDF

    Parameters
    ----------
    weights_path: str
        The path to the keras model file
    batch_size: int
        The batch size to feed the model

    References
    ----------
    On Face Segmentation, Face Swapping, and Face Perception (https://arxiv.org/abs/1704.06729)
    Source Implementation: https://github.com/YuvalNirkin/face_segmentation
    Model file sourced from:
    https://github.com/YuvalNirkin/face_segmentation/releases/download/1.0/face_seg_fcn8s.zip
    """
    def __init__(self, weights_path: str, batch_size: int) -> None:
        logger.debug(parse_class_init(locals()))
        self._batch_size = batch_size
        self._model = self._load_model(weights_path)
        logger.debug("Initialized: %s", self.__class__.__name__)

    @classmethod
    def _load_model(cls, weights_path: str) -> Model:
        """ Definition of the VGG Obstructed Model.

        Parameters
        ----------
        weights_path: str
            Full path to the model's weights

        Returns
        -------
        :class:`keras.models.Model`
            The VGG-Obstructed model
        """
        input_ = kl.Input(shape=(500, 500, 3))
        var_x = kl.ZeroPadding2D(padding=((100, 100), (100, 100)))(input_)

        var_x = _ConvBlock(1, 64, 2)(var_x)
        var_x = _ConvBlock(2, 128, 2)(var_x)
        var_x = _ConvBlock(3, 256, 3)(var_x)

        score_pool3 = _ScorePool(3, 0.0001, 9)(var_x)
        var_x = _ConvBlock(4, 512, 3)(var_x)
        score_pool4 = _ScorePool(4, 0.01, 5)(var_x)
        var_x = _ConvBlock(5, 512, 3)(var_x)

        var_x = kl.Conv2D(4096, 7, padding="valid", activation="relu", name="fc6")(var_x)
        var_x = kl.Dropout(rate=0.5)(var_x)
        var_x = kl.Conv2D(4096, 1, padding="valid", activation="relu", name="fc7")(var_x)
        var_x = kl.Dropout(rate=0.5)(var_x)

        var_x = kl.Conv2D(21, 1, padding="valid", activation="linear", name="score_fr")(var_x)
        var_x = kl.Conv2DTranspose(21,
                                   4,
                                   strides=2,
                                   activation="linear",
                                   use_bias=False,
                                   name="upscore2")(var_x)

        var_x = kl.Add()([var_x, score_pool4])
        var_x = kl.Conv2DTranspose(21,
                                   4,
                                   strides=2,
                                   activation="linear",
                                   use_bias=False,
                                   name="upscore_pool4")(var_x)

        var_x = kl.Add()([var_x, score_pool3])
        var_x = kl.Conv2DTranspose(21,
                                   16,
                                   strides=8,
                                   activation="linear",
                                   use_bias=False,
                                   name="upscore8")(var_x)
        var_x = kl.Cropping2D(cropping=((31, 37), (31, 37)), name="score")(var_x)
        var_x = kl.Activation("softmax", name="softmax")(var_x)

        retval = Model(input_, var_x)
        retval.load_weights(weights_path)
        retval.make_predict_function()
        return retval

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        """ Get predictions from the VGG-Clear model

        Parameters
        ----------
        inputs: :class:`numpy.ndarray`
            The input to VGG-Obstructed

        Returns
        -------
        :class:`numpy.ndarray`
            The output from VGG-Obstructed
        """
        return self._model.predict(inputs, verbose=0, batch_size=self._batch_size)


class _ConvBlock():
    """ Convolutional loop with max pooling layer for VGG Obstructed.

    Parameters
    ----------
    level: int
        For naming. The current level for this convolutional loop
    filters: int
        The number of filters that should appear in each Conv2D layer
    iterations: int
        The number of consecutive Conv2D layers to create
    """
    def __init__(self, level: int, filters: int, iterations: int) -> None:
        self._name = f"conv{level}_"
        self._level = level
        self._filters = filters
        self._iterator = range(1, iterations + 1)

    def __call__(self, inputs: KerasTensor) -> KerasTensor:
        """ Call the convolutional loop.

        Parameters
        ----------
        inputs: :class:`keras.KerasTensor`
            The input tensor to the block

        Returns
        -------
        :class:`keras.KerasTensor`
            The output tensor from the convolutional block
        """
        var_x = inputs
        for i in self._iterator:
            padding = "valid" if self._level == i == 1 else "same"
            var_x = kl.Conv2D(self._filters,
                              3,
                              padding=padding,
                              activation="relu",
                              name=f"{self._name}{i}")(var_x)
        var_x = kl.MaxPooling2D(padding="same",
                                strides=(2, 2),
                                name=f"pool{self._level}")(var_x)
        return var_x


class _ScorePool():
    """ Cropped scaling of the pooling layer.

    Parameters
    ----------
    level: int
        For naming. The current level for this score pool
    scale: float
        The scaling to apply to the pool
    crop: int
        The amount of 2D cropping to apply
    """
    def __init__(self, level: int, scale: float, crop: int) -> None:
        self._name = f"_pool{level}"
        self._cropping = ((crop, crop), (crop, crop))
        self._scale = scale

    def __call__(self, inputs: KerasTensor) -> KerasTensor:
        """ Score pool block.

        Parameters
        ----------
        inputs: :class:`keras.KerasTensor`
            The input tensor to the block

        Returns
        -------
        :class:`keras.KerasTensor`
            The output tensor from the score pool block
        """
        var_x = kl.Lambda(lambda x: x * self._scale, name="scale" + self._name)(inputs)
        var_x = kl.Conv2D(21,
                          1,
                          padding="valid",
                          activation="linear",
                          name="score" + self._name)(var_x)
        var_x = kl.Cropping2D(cropping=self._cropping, name="score" + self._name + "c")(var_x)
        return var_x


__all__ = get_module_objects(__name__)
