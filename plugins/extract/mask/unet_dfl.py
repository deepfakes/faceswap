#!/usr/bin/env python3
""" UNET DFL face mask plugin

Architecture and Pre-Trained Model based on...
TernausNet: U-Net with VGG11 Encoder Pre-Trained on ImageNet for Image Segmentation
https://arxiv.org/abs/1801.05746
https://github.com/ternaus/TernausNet

Source Implementation and fine-tune training....
https://github.com/iperov/DeepFaceLab/blob/master/nnlib/TernausNet.py

Model file sourced from...
https://github.com/iperov/DeepFaceLab/blob/master/nnlib/FANSeg_256_full_face.h5
"""
from __future__ import annotations

import logging
import typing as T

import numpy as np
from keras import backend as K, layers as kl, Model

from lib.logger import parse_class_init
from lib.utils import get_module_objects
from ._base import BatchType, Masker, MaskerBatch
from . import unet_dfl_defaults as cfg

if T.TYPE_CHECKING:
    from keras import KerasTensor


logger = logging.getLogger(__name__)


class Mask(Masker):
    """ Neural network to process face image into a segmentation mask of the face """
    def __init__(self, **kwargs) -> None:
        git_model_id = 6
        model_filename = "DFL_256_sigmoid_v1.h5"
        super().__init__(git_model_id=git_model_id, model_filename=model_filename, **kwargs)
        self.model: UnetDFL
        self.name = "U-Net"
        self.input_size = 256
        self.vram = 320  # 276 in testing
        self.vram_per_batch = 256  # ~215 in testing
        self.batchsize = cfg.batch_size()
        self._storage_centering = "legacy"

    def init_model(self) -> None:
        assert self.name is not None and isinstance(self.model_path, str)
        self.model = UnetDFL(self.model_path, self.batchsize)
        placeholder = np.zeros((self.batchsize, self.input_size, self.input_size, 3),
                               dtype="float32")
        self.model(placeholder)

    def process_input(self, batch: BatchType) -> None:
        """ Compile the detected faces for prediction """
        assert isinstance(batch, MaskerBatch)
        batch.feed = np.array([T.cast(np.ndarray, feed.face)[..., :3]
                               for feed in batch.feed_faces], dtype="float32") / 255.0
        logger.trace("feed shape: %s", batch.feed.shape)  # type: ignore

    def predict(self, feed: np.ndarray) -> np.ndarray:
        """ Run model to get predictions """
        return self.model(feed)

    def process_output(self, batch: BatchType) -> None:
        """ Compile found faces for output """
        return


class UnetDFL:
    """ UNet DFL Definition for Keras 3 with PyTorch backend

    Parameters
    ----------
    weights_path: str
        Full path to the location of the weights file for the model
    batch_size: int
        The batch size to feed the model at

    Note
    ----
    Model definition is explicitly stated as there is an incompatibility for certain
    Conv2DTranspose combinations when model was trained on one backend but inferred on another:
    https://github.com/keras-team/keras-core/issues/774
    The effect of this misaligns the mask and peforms bad inference for this model.
    """
    def __init__(self, weights_path: str, batch_size: int) -> None:
        logger.debug(parse_class_init(locals()))
        self._batch_size = batch_size
        self._model = self._load_model(weights_path)
        logger.debug("Initialized: %s", self.__class__.__name__)

    @classmethod
    def conv_block(cls,
                   inputs: KerasTensor,
                   filters: int,
                   recursions: int,
                   idx: int) -> KerasTensor:
        """ Convolution block for UnetDFL downscales

        Parameters
        ----------
        inputs: :class:`keras.KerasTensor`
            The inputs to the block
        filters: int
            The number of filters for the convolution
        recursions: int
            The number of convolutions to run
        idx: The index id of the first convolution (used for naming)

        Returns
        -------
        :class:`keras.KerasTensor`
            The output from the convolution block
        """
        output = inputs

        for _ in range(recursions):
            output = kl.Conv2D(filters,
                               3,
                               padding="same",
                               activation="relu",
                               kernel_initializer="random_uniform",
                               name=f"features_{idx}")(output)
            idx += 2

        return output

    @classmethod
    def skip_block(cls,  # pylint:disable=too-many-positional-arguments
                   input_1: KerasTensor,
                   input_2: KerasTensor,
                   conv_filters: int,
                   trans_filters: int,
                   linear: bool,
                   idx: int) -> KerasTensor:
        """ Deconvolution + skip connection for UnetDFL upscales

        Parameters
        ----------
        input_1: :class:`keras.KerasTensor`
            The input to be upscaled
        input_2: :class:`keras.KerasTensor`
            The skip connection to be concatenated to the upscaled tensor
        conv_filters: int
            The number of filters to be used for the convolution
        trans_filters: int
            The number of filters to be used for the conv-transpose
        linear: bool
            ``True`` to use linear activation in the convolution, ``False`` to use ReLu
        idx: int
            The index for naming the layers

        Returns
        -------
        :class:`keras.KerasTensor`
            The output from the upscaled/skip connection
        """
        output = kl.Conv2D(conv_filters,
                           3,
                           padding="same",
                           activation="linear" if linear else "relu",
                           kernel_initializer="random_uniform",
                           name=f"conv2d_{idx}")(input_1)

        # TF vs PyTorch paddng is different. We need to negative pad the output for Torch
        padding = "valid" if K.backend() == "torch" else "same"
        output = kl.Conv2DTranspose(trans_filters,
                                    3,
                                    strides=2,
                                    padding=padding,
                                    activation="relu",
                                    kernel_initializer="random_uniform",
                                    name=f"conv2d_transpose_{idx}")(output)

        if K.backend() == "torch":
            output = output[:, :-1, :-1, :]

        return kl.Concatenate(name=f"concatenate_{idx}")([output, input_2])

    def _load_model(self, weights_path: str) -> Model:
        """ Definition of the UNet-DFL Model.

        Parameters
        ----------
        weights_path: str
            Full path to the model's weights

        Returns
        -------
        :class:`keras.models.Model`
            The VGG-Clear model
        """
        features = []
        input_ = kl.Input(shape=(256, 256, 3), name="input_1")

        features.append(self.conv_block(input_, 64, 1, 0))
        var_x = kl.MaxPool2D(pool_size=2, strides=2, name="max_pooling2d_1")(features[-1])

        features.append(self.conv_block(var_x, 128, 1, 3))
        var_x = kl.MaxPool2D(pool_size=2, strides=2, name="max_pooling2d_2")(features[-1])

        features.append(self.conv_block(var_x, 256, 2, 6))
        var_x = kl.MaxPool2D(pool_size=2, strides=2, name="max_pooling2d_3")(features[-1])

        features.append(self.conv_block(var_x, 512, 2, 11))
        var_x = kl.MaxPool2D(pool_size=2, strides=2, name="max_pooling2d_4")(features[-1])

        features.append(self.conv_block(var_x, 512, 2, 16))
        var_x = kl.MaxPool2D(pool_size=2, strides=2, name="max_pooling2d_5")(features[-1])

        convs = [512, 512, 512, 256, 128]
        for idx, (feats, filts) in enumerate(zip(reversed(features), convs)):
            linear = idx == 0
            trans_filts = filts // 2 if idx < 2 else filts // 4
            var_x = self.skip_block(var_x, feats, filts, trans_filts, linear, idx + 1)

        var_x = kl.Conv2D(64,
                          3,
                          padding="same",
                          activation="relu",
                          kernel_initializer="random_uniform",
                          name="conv2d_6")(var_x)
        output = kl.Conv2D(1,
                           3,
                           padding="same",
                           activation="sigmoid",
                           kernel_initializer="random_uniform",
                           name="conv2d_7")(var_x)

        model = Model(input_, output)
        model.load_weights(weights_path)
        model.make_predict_function()
        return model

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        """ Obtain predictions from the UNet-DFL Model

        Parameters
        ----------
        inputs: :class:`numpy.ndarray`
            The input to UNet-DFL

        Returns
        -------
        :class:`numpy.ndarray`
            The output from UNet-DFL
        """
        return self._model.predict(inputs, verbose=0, batch_size=self._batch_size)


__all__ = get_module_objects(__name__)
