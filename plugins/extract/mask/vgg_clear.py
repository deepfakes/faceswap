#!/usr/bin/env python3
""" VGG Clear face mask plugin. """
from __future__ import annotations
import logging
import typing as T

import numpy as np

from keras import layers as kl,  Model

from lib.logger import parse_class_init
from lib.utils import get_module_objects
from ._base import BatchType, Masker, MaskerBatch
from . import vgg_clear_defaults as cfg

if T.TYPE_CHECKING:
    from keras import KerasTensor

logger = logging.getLogger(__name__)


class Mask(Masker):
    """ Neural network to process face image into a segmentation mask of the face """
    def __init__(self, **kwargs) -> None:
        git_model_id = 8
        model_filename = "Nirkin_300_softmax_v1.h5"
        super().__init__(git_model_id=git_model_id, model_filename=model_filename, **kwargs)
        self.model: VGGClear
        self.name = "VGG Clear"
        self.input_size = 300
        self.vram = 1344  # 1308 in testing
        self.vram_per_batch = 448  # ~402 in testing
        self.batchsize = cfg.batch_size()

    def init_model(self) -> None:
        assert isinstance(self.model_path, str)
        self.model = VGGClear(self.model_path, self.batchsize)
        placeholder = np.zeros((self.batchsize, self.input_size, self.input_size, 3),
                               dtype="float32")
        self.model(placeholder)

    def process_input(self, batch: BatchType) -> None:
        """ Compile the detected faces for prediction """
        assert isinstance(batch, MaskerBatch)
        input_ = np.array([T.cast(np.ndarray, feed.face)[..., :3]
                           for feed in batch.feed_faces], dtype="float32")
        batch.feed = input_ - np.mean(input_, axis=(1, 2))[:, None, None, :]
        logger.trace("feed shape: %s", batch.feed.shape)  # type: ignore

    def predict(self, feed: np.ndarray) -> np.ndarray:
        """ Run model to get predictions """
        predictions = self.model(feed)
        assert isinstance(predictions, np.ndarray)
        return predictions[..., -1]

    def process_output(self, batch: BatchType) -> None:
        """ Compile found faces for output """
        return


class VGGClear():
    """ VGG Clear mask for Faceswap.

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
    https://github.com/YuvalNirkin/face_segmentation/releases/download/1.1/face_seg_fcn8s_300_no_aug.zip

    """
    def __init__(self, weights_path: str, batch_size: int) -> None:
        logger.debug(parse_class_init(locals()))
        self._batch_size = batch_size
        self._model = self._load_model(weights_path)
        logger.debug("Initialized: %s", self.__class__.__name__)

    @classmethod
    def _load_model(cls, weights_path: str) -> Model:
        """ Definition of the VGG Clear Model.

        Parameters
        ----------
        weights_path: str
            Full path to the model's weights

        Returns
        -------
        :class:`keras.models.Model`
            The VGG-Clear model
        """
        input_ = kl.Input(shape=(300, 300, 3))
        var_x = kl.ZeroPadding2D(padding=((100, 100), (100, 100)), name="zero_padding2d_1")(input_)

        var_x = _ConvBlock(1, 64, 2)(var_x)
        var_x = _ConvBlock(2, 128, 2)(var_x)
        pool3 = _ConvBlock(3, 256, 3)(var_x)
        pool4 = _ConvBlock(4, 512, 3)(pool3)
        var_x = _ConvBlock(5, 512, 3)(pool4)

        score_pool3 = _ScorePool(3, 0.0001, (9, 8))(pool3)
        score_pool4 = _ScorePool(4, 0.01, (5, 5))(pool4)

        var_x = kl.Conv2D(4096, 7, activation="relu", name="fc6")(var_x)
        var_x = kl.Dropout(rate=0.5, name="drop6")(var_x)
        var_x = kl.Conv2D(4096, 1, activation="relu", name="fc7")(var_x)
        var_x = kl.Dropout(rate=0.5, name="drop7")(var_x)
        var_x = kl.Conv2D(2, 1, activation="linear", name="score_fr_r")(var_x)
        var_x = kl.Conv2DTranspose(2,
                                   4,
                                   strides=2,
                                   activation="linear",
                                   use_bias=False, name="upscore2_r")(var_x)

        var_x = kl.Add(name="fuse_pool4")([var_x, score_pool4])
        var_x = kl.Conv2DTranspose(2,
                                   4,
                                   strides=2,
                                   activation="linear",
                                   use_bias=False,
                                   name="upscore_pool4_r")(var_x)
        var_x = kl.Add(name="fuse_pool3")([var_x, score_pool3])
        var_x = kl.Conv2DTranspose(2,
                                   16,
                                   strides=8,
                                   activation="linear",
                                   use_bias=False,
                                   name="upscore8_r")(var_x)
        var_x = kl.Cropping2D(cropping=((31, 45), (31, 45)), name="score")(var_x)
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
            The input to VGG-Clear

        Returns
        -------
        :class:`numpy.ndarray`
            The output from VGG-Clear
        """
        return self._model.predict(inputs, verbose=0, batch_size=self._batch_size)


class _ConvBlock():
    """ Convolutional loop with max pooling layer for VGG Clear.

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
    crop: tuple
        The amount of 2D cropping to apply. Tuple of `ints`
    """
    def __init__(self, level: int, scale: float, crop: tuple[int, int]):
        self._name = f"_pool{level}"
        self._cropping = (crop, crop)
        self._scale = scale

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        """ Score pool block.

        Parameters
        ----------
        inputs: tensor
            The input tensor to the block

        Returns
        -------
        tensor
            The output tensor from the score pool block
        """
        var_x = kl.Lambda(lambda x: x * self._scale, name="scale" + self._name)(inputs)
        var_x = kl.Conv2D(2, 1, activation="linear", name="score" + self._name + "_r")(var_x)
        var_x = kl.Cropping2D(cropping=self._cropping, name="score" + self._name + "c")(var_x)
        return var_x


__all__ = get_module_objects(__name__)
