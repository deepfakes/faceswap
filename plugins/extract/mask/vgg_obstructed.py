#!/usr/bin/env python3
""" VGG Obstructed face mask plugin """
from __future__ import annotations
import logging
import typing as T

import numpy as np

# Ignore linting errors from Tensorflow's thoroughly broken import system
from tensorflow.keras.layers import (  # pylint:disable=import-error
    Add, Conv2D, Conv2DTranspose, Cropping2D, Dropout, Input, Lambda, MaxPooling2D,
    ZeroPadding2D)

from lib.model.session import KSession
from ._base import BatchType, Masker, MaskerBatch

if T.TYPE_CHECKING:
    from tensorflow import Tensor

logger = logging.getLogger(__name__)


class Mask(Masker):
    """ Neural network to process face image into a segmentation mask of the face """
    def __init__(self, **kwargs) -> None:
        git_model_id = 5
        model_filename = "Nirkin_500_softmax_v1.h5"
        super().__init__(git_model_id=git_model_id, model_filename=model_filename, **kwargs)
        self.model: KSession
        self.name = "VGG Obstructed"
        self.input_size = 500
        self.vram = 3936
        self.vram_warnings = 1088  # at BS 1. OOMs at higher batch sizes
        self.vram_per_batch = 304
        self.batchsize = self.config["batch-size"]

    def init_model(self) -> None:
        assert isinstance(self.model_path, str)
        self.model = VGGObstructed(self.model_path,
                                   allow_growth=self.config["allow_growth"],
                                   exclude_gpus=self._exclude_gpus)
        self.model.append_softmax_activation(layer_index=-1)
        placeholder = np.zeros((self.batchsize, self.input_size, self.input_size, 3),
                               dtype="float32")
        self.model.predict(placeholder)

    def process_input(self, batch: BatchType) -> None:
        """ Compile the detected faces for prediction """
        assert isinstance(batch, MaskerBatch)
        input_ = [T.cast(np.ndarray, feed.face)[..., :3] for feed in batch.feed_faces]
        batch.feed = input_ - np.mean(input_, axis=(1, 2))[:, None, None, :]
        logger.trace("feed shape: %s", batch.feed.shape)  # type:ignore

    def predict(self, feed: np.ndarray) -> np.ndarray:
        """ Run model to get predictions """
        predictions = self.model.predict(feed)
        assert isinstance(predictions, np.ndarray)
        return predictions[..., 0] * -1.0 + 1.0

    def process_output(self, batch: BatchType) -> None:
        """ Compile found faces for output """
        return


class VGGObstructed(KSession):
    """ VGG Obstructed mask for Faceswap.

    Caffe model re-implemented in Keras by Kyle Vrooman.
    Re-implemented for Tensorflow 2 by TorzDF

    Parameters
    ----------
    model_path: str
        The path to the keras model file
    allow_growth: bool
        Enable the Tensorflow GPU allow_growth configuration option. This option prevents
        Tensorflow from allocating all of the GPU VRAM, but can lead to higher fragmentation and
        slower performance
    exclude_gpus: list
        A list of indices correlating to connected GPUs that Tensorflow should not use. Pass
        ``None`` to not exclude any GPUs

    References
    ----------
    On Face Segmentation, Face Swapping, and Face Perception (https://arxiv.org/abs/1704.06729)
    Source Implementation: https://github.com/YuvalNirkin/face_segmentation
    Model file sourced from:
    https://github.com/YuvalNirkin/face_segmentation/releases/download/1.0/face_seg_fcn8s.zip
    """
    def __init__(self,
                 model_path: str,
                 allow_growth: bool,
                 exclude_gpus: list[int] | None) -> None:
        super().__init__("VGG Obstructed",
                         model_path,
                         allow_growth=allow_growth,
                         exclude_gpus=exclude_gpus)
        self.define_model(self._model_definition)
        self.load_model_weights()

    @classmethod
    def _model_definition(cls) -> tuple[Tensor, Tensor]:
        """ Definition of the VGG Obstructed Model.

        Returns
        -------
        tuple
            The tensor input to the model and tensor output to the model for compilation by
            :func`define_model`
        """
        input_ = Input(shape=(500, 500, 3))
        var_x = ZeroPadding2D(padding=((100, 100), (100, 100)))(input_)

        var_x = _ConvBlock(1, 64, 2)(var_x)
        var_x = _ConvBlock(2, 128, 2)(var_x)
        var_x = _ConvBlock(3, 256, 3)(var_x)

        score_pool3 = _ScorePool(3, 0.0001, 9)(var_x)
        var_x = _ConvBlock(4, 512, 3)(var_x)
        score_pool4 = _ScorePool(4, 0.01, 5)(var_x)
        var_x = _ConvBlock(5, 512, 3)(var_x)

        var_x = Conv2D(4096, 7, padding="valid", activation="relu", name="fc6")(var_x)
        var_x = Dropout(rate=0.5)(var_x)
        var_x = Conv2D(4096, 1, padding="valid", activation="relu", name="fc7")(var_x)
        var_x = Dropout(rate=0.5)(var_x)

        var_x = Conv2D(21, 1, padding="valid", activation="linear", name="score_fr")(var_x)
        var_x = Conv2DTranspose(21,
                                4,
                                strides=2,
                                activation="linear",
                                use_bias=False,
                                name="upscore2")(var_x)

        var_x = Add()([var_x, score_pool4])
        var_x = Conv2DTranspose(21,
                                4,
                                strides=2,
                                activation="linear",
                                use_bias=False,
                                name="upscore_pool4")(var_x)

        var_x = Add()([var_x, score_pool3])
        var_x = Conv2DTranspose(21,
                                16,
                                strides=8,
                                activation="linear",
                                use_bias=False,
                                name="upscore8")(var_x)
        var_x = Cropping2D(cropping=((31, 37), (31, 37)), name="score")(var_x)
        return input_, var_x


class _ConvBlock():  # pylint:disable=too-few-public-methods
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

    def __call__(self, inputs: Tensor) -> Tensor:
        """ Call the convolutional loop.

        Parameters
        ----------
        inputs: tensor
            The input tensor to the block

        Returns
        -------
        tensor
            The output tensor from the convolutional block
        """
        var_x = inputs
        for i in self._iterator:
            padding = "valid" if self._level == i == 1 else "same"
            var_x = Conv2D(self._filters,
                           3,
                           padding=padding,
                           activation="relu",
                           name=f"{self._name}{i}")(var_x)
        var_x = MaxPooling2D(padding="same",
                             strides=(2, 2),
                             name=f"pool{self._level}")(var_x)
        return var_x


class _ScorePool():  # pylint:disable=too-few-public-methods
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

    def __call__(self, inputs: Tensor) -> Tensor:
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
        var_x = Lambda(lambda x: x * self._scale, name="scale" + self._name)(inputs)
        var_x = Conv2D(21,
                       1,
                       padding="valid",
                       activation="linear",
                       name="score" + self._name)(var_x)
        var_x = Cropping2D(cropping=self._cropping, name="score" + self._name + "c")(var_x)
        return var_x
