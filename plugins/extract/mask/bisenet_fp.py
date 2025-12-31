#!/usr/bin/env python3
""" BiSeNet Face-Parsing mask plugin

Architecture and Pre-Trained Model ported from PyTorch to Keras by TorzDF from
https://github.com/zllrunning/face-parsing.PyTorch
"""
from __future__ import annotations
import logging
import typing as T

import numpy as np

import keras.backend as K
from keras.layers import (
    Activation, Add, BatchNormalization, Concatenate, Conv2D, GlobalAveragePooling2D, Input,
    MaxPooling2D, Multiply, Reshape, UpSampling2D, ZeroPadding2D)
from keras.models import Model

from lib.logger import parse_class_init
from lib.utils import get_module_objects
from plugins.extract.extract_config import load_config
from ._base import BatchType, Masker, MaskerBatch
from . import bisenet_fp_defaults as cfg

if T.TYPE_CHECKING:
    from keras import KerasTensor

logger = logging.getLogger(__name__)


class Mask(Masker):  # pylint:disable=too-many-instance-attributes
    """ Neural network to process face image into a segmentation mask of the face """
    def __init__(self, **kwargs) -> None:
        # We need access to user config prior to parent being initialized to correctly set the
        # model filename
        load_config(kwargs.get("configfile"))
        self._is_faceswap, version = self._check_weights_selection()

        git_model_id = 14
        model_filename = f"bisnet_face_parsing_v{version}.h5"
        super().__init__(git_model_id=git_model_id, model_filename=model_filename, **kwargs)

        self.model: BiSeNet
        self.name = "BiSeNet - Face Parsing"
        self.input_size = 512
        self.color_format = "RGB"
        self.vram = 384 if not cfg.cpu() else 0  # 378 in testing
        self.vram_per_batch = 384 if not cfg.cpu() else 0  # ~328 in testing
        self.batchsize = cfg.batch_size()

        self._segment_indices = self._get_segment_indices()
        self._storage_centering = "head" if cfg.include_hair() else "face"
        """ Literal["head", "face"] The mask type/storage centering to use """
        # Separate storage for face and head masks
        self._storage_name = f"{self._storage_name}_{self._storage_centering}"

    def _check_weights_selection(self) -> tuple[bool, int]:
        """ Check which weights have been selected.

        This is required for passing along the correct file name for the corresponding weights
        selection.

        Returns
        -------
        is_faceswap : bool
            ``True`` if `faceswap` trained weights have been selected. ``False`` if `original`
            weights have been selected.
        version : int
            ``1`` for non-faceswap, ``2`` if faceswap and full-head model is required. ``3`` if
            faceswap and full-face is required
        """
        is_faceswap = cfg.weights() == "faceswap"
        version = 1 if not is_faceswap else 2 if cfg.include_hair() else 3
        return is_faceswap, version

    def _get_segment_indices(self) -> list[int]:
        """ Obtain the segment indices to include within the face mask area based on user
        configuration settings.

        Returns
        -------
        list
            The segment indices to include within the face mask area

        Notes
        -----
        'original' Model segment indices:
        0: background, 1: skin, 2: left brow, 3: right brow, 4: left eye, 5: right eye, 6: glasses
        7: left ear, 8: right ear, 9: earing, 10: nose, 11: mouth, 12: upper lip, 13: lower_lip,
        14: neck, 15: neck ?, 16: cloth, 17: hair, 18: hat

        'faceswap' Model segment indices:
        0: background, 1: skin, 2: ears, 3: hair, 4: glasses
        """
        retval = [1] if self._is_faceswap else [1, 2, 3, 4, 5, 10, 11, 12, 13]

        if cfg.include_glasses():
            retval.append(4 if self._is_faceswap else 6)
        if cfg.include_ears():
            retval.extend([2] if self._is_faceswap else [7, 8, 9])
        if cfg.include_hair():
            retval.append(3 if self._is_faceswap else 17)
        logger.debug("Selected segment indices: %s", retval)
        return retval

    def init_model(self) -> None:
        """ Initialize the BiSeNet Face Parsing model. """
        assert isinstance(self.model_path, str)
        lbls = 5 if self._is_faceswap else 19
        placeholder = np.zeros((self.batchsize, self.input_size, self.input_size, 3),
                               dtype="float32")

        with self.get_device_context(cfg.cpu()):
            self.model = BiSeNet(self.model_path, self.batchsize, self.input_size, lbls)
            self.model(placeholder)

    def process_input(self, batch: BatchType) -> None:
        """ Compile the detected faces for prediction """
        assert isinstance(batch, MaskerBatch)
        mean = (0.384, 0.314, 0.279) if self._is_faceswap else (0.485, 0.456, 0.406)
        std = (0.324, 0.286, 0.275) if self._is_faceswap else (0.229, 0.224, 0.225)

        batch.feed = ((np.array([T.cast(np.ndarray, feed.face)[..., :3]
                                 for feed in batch.feed_faces],
                                dtype="float32") / 255.0) - mean) / std
        logger.trace("feed shape: %s", batch.feed.shape)  # type:ignore[attr-defined]

    def predict(self, feed: np.ndarray) -> np.ndarray:
        """ Run model to get predictions """
        with self.get_device_context(cfg.cpu()):
            return self.model(feed)[0]

    def process_output(self, batch: BatchType) -> None:
        """ Compile found faces for output """
        pred = batch.prediction.argmax(-1).astype("uint8")
        batch.prediction = np.isin(pred, self._segment_indices).astype("float32")

# BiSeNet Face-Parsing Model

# MIT License

# Copyright (c) 2019 zll

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


_NAME_TRACKER: set[str] = set()


def _get_name(name: str, start_idx: int = 1) -> str:
    """ Auto numbering to keep track of layer names.

    Names are kept the same as the PyTorch original model, to enable easier porting of weights.

    Names are tracked and auto-appended with an integer to ensure they are unique.

    Parameters
    ----------
    name: str
        The name of the layer to get auto named.
    start_idx
        The first index number to start auto naming layers with the same name. Usually 0 or 1.
        Pass -1 if the name should not be auto-named (i.e. should not have an integer appended
        to the end)

    Returns
    -------
    str
        A unique version of the original name
    """
    i = start_idx
    while True:
        retval = f"{name}{i}" if i != -1 else name
        if retval not in _NAME_TRACKER:
            break
        i += 1
    _NAME_TRACKER.add(retval)
    return retval


class ConvBn():
    """ Convolutional 3D with Batch Normalization block.

    Parameters
    ----------
    filters: int
        The dimensionality of the output space (i.e. the number of output filters in the
        convolution).
    kernel_size: int, optional
        The height and width of the 2D convolution window. Default: `3`
    strides: int, optional
        The strides of the convolution along the height and width. Default: `1`
    padding: int, optional
        The amount of padding to apply prior to the first Convolutional Layer. Default: `1`
    activation: bool
        Whether to include ReLu Activation at the end of the block. Default: ``True``
    prefix: str, optional
        The prefix to name the layers within the block. Default: ``""`` (empty string, i.e. no
        prefix)
    start_idx: int, optional
        The starting index for naming the layers within the block. See :func:`_get_name` for
        more information. Default: `1`
    """
    def __init__(self, filters: int,  # pylint:disable=too-many-positional-arguments
                 kernel_size: int = 3,
                 strides: int = 1,
                 padding: int = 1,
                 activation: int = True,
                 prefix: str = "",
                 start_idx: int = 1) -> None:
        self._filters = filters
        self._kernel_size = kernel_size
        self._strides = strides
        self._padding = padding
        self._activation = activation
        self._prefix = f"{prefix}-" if prefix else prefix
        self._start_idx = start_idx

    def __call__(self, inputs: KerasTensor) -> KerasTensor:
        """ Call the Convolutional Batch Normalization block.

        Parameters
        ----------
        inputs: :class:`keras.KerasTensor`
            The input to the block

        Returns
        -------
        :class:`keras.KerasTensor`
            The output from the block
        """
        var_x = inputs
        if self._padding > 0 and self._kernel_size != 1:
            var_x = ZeroPadding2D(self._padding,
                                  name=_get_name(f"{self._prefix}zeropad",
                                                 start_idx=self._start_idx))(var_x)
        padding = "valid" if self._padding != -1 else "same"
        var_x = Conv2D(self._filters,
                       self._kernel_size,
                       strides=self._strides,
                       padding=padding,
                       use_bias=False,
                       name=_get_name(f"{self._prefix}conv", start_idx=self._start_idx))(var_x)
        var_x = BatchNormalization(epsilon=1e-5,
                                   name=_get_name(f"{self._prefix}bn",
                                                  start_idx=self._start_idx))(var_x)
        if self._activation:
            var_x = Activation("relu",
                               name=_get_name(f"{self._prefix}relu",
                                              start_idx=self._start_idx))(var_x)
        return var_x


class ResNet18():
    """ ResNet 18 block. Used at the start of BiSeNet Face Parsing. """
    def __init__(self):
        self._feature_index = 1 if K.image_data_format() == "channels_first" else -1

    def _basic_block(self,
                     inputs: KerasTensor,
                     prefix: str,
                     filters: int,
                     strides: int = 1) -> KerasTensor:
        """ The basic building block for ResNet 18.

        Parameters
        ----------
        inputs: :class:`keras.KerasTensor`
            The input to the block
        prefix: str
            The prefix to name the layers within the block
        filters: int
            The dimensionality of the output space (i.e. the number of output filters in the
            convolution).
        strides: int, optional
            The strides of the convolution along the height and width. Default: `1`

        Returns
        -------
        :class:`keras.KerasTensor`
            The output from the block
        """
        res = ConvBn(filters, strides=strides, padding=1, prefix=prefix)(inputs)
        res = ConvBn(filters, strides=1, padding=1, activation=False, prefix=prefix)(res)

        shortcut = inputs
        filts = (shortcut.shape[self._feature_index], res.shape[self._feature_index])
        if strides != 1 or filts[0] != filts[1]:  # Downsample
            name = f"{prefix}-downsample-"
            shortcut = Conv2D(filters, 1,
                              strides=strides,
                              use_bias=False,
                              name=_get_name(f"{name}", start_idx=0))(shortcut)
            shortcut = BatchNormalization(epsilon=1e-5,
                                          name=_get_name(f"{name}", start_idx=0))(shortcut)

        var_x = Add(name=f"{prefix}-add")([res, shortcut])
        var_x = Activation("relu", name=f"{prefix}-relu")(var_x)
        return var_x

    def _basic_layer(self,  # pylint:disable=too-many-positional-arguments
                     inputs: KerasTensor,
                     prefix: str,
                     filters: int,
                     num_blocks: int,
                     strides: int = 1) -> KerasTensor:
        """ The basic layer for ResNet 18. Recursively builds from :func:`_basic_block`.

        Parameters
        ----------
        inputs: :class:`keras.KerasTensor`
            The input to the block
        prefix: str
            The prefix to name the layers within the block
        filters: int
            The dimensionality of the output space (i.e. the number of output filters in the
            convolution).
        num_blocks: int
            The number of basic blocks to recursively build
        strides: int, optional
            The strides of the convolution along the height and width. Default: `1`

        Returns
        -------
        :class:`keras.KerasTensor`
            The output from the block
        """
        var_x = self._basic_block(inputs, f"{prefix}-0", filters, strides=strides)
        for i in range(num_blocks - 1):
            var_x = self._basic_block(var_x, f"{prefix}-{i + 1}", filters, strides=1)
        return var_x

    def __call__(self, inputs: KerasTensor) -> KerasTensor:
        """ Call the ResNet 18 block.

        Parameters
        ----------
        inputs: :class:`keras.KerasTensor`
            The input to the block

        Returns
        -------
        :class:`keras.KerasTensor`
            The output from the block
        """
        var_x = ConvBn(64, kernel_size=7, strides=2, padding=3, prefix="cp-resnet")(inputs)
        var_x = ZeroPadding2D(1, name="cp-resnet-zeropad")(var_x)
        var_x = MaxPooling2D(pool_size=3, strides=2, name="cp-resnet-maxpool")(var_x)

        var_x = self._basic_layer(var_x, "cp-resnet-layer1", 64, 2)
        feat8 = self._basic_layer(var_x, "cp-resnet-layer2", 128, 2, strides=2)
        feat16 = self._basic_layer(feat8, "cp-resnet-layer3", 256, 2, strides=2)
        feat32 = self._basic_layer(feat16, "cp-resnet-layer4", 512, 2, strides=2)

        return feat8, feat16, feat32


class AttentionRefinementModule():
    """ The Attention Refinement block for BiSeNet Face Parsing

    Parameters
    ----------
    filters: int
        The dimensionality of the output space (i.e. the number of output filters in the
        convolution).
    """
    def __init__(self, filters: int) -> None:
        self._filters = filters

    def __call__(self, inputs: KerasTensor, feats: int) -> KerasTensor:
        """ Call the Attention Refinement block.

        Parameters
        ----------
        inputs: :class:`keras.KerasTensor`
            The input to the block
        feats: int
            The number of features. Used for naming.

        Returns
        -------
        :class:`keras.KerasTensor`
            The output from the block
        """
        prefix = f"cp-arm{feats}"
        feat = ConvBn(self._filters, prefix=f"{prefix}-conv", start_idx=-1, padding=-1)(inputs)
        atten = GlobalAveragePooling2D(name=f"{prefix}-avgpool")(feat)
        atten = Reshape((1, 1, atten.shape[-1]))(atten)
        atten = Conv2D(self._filters, 1, use_bias=False, name=f"{prefix}-conv_atten")(atten)
        atten = BatchNormalization(epsilon=1e-5, name=f"{prefix}-bn_atten")(atten)
        atten = Activation("sigmoid", name=f"{prefix}-sigmoid")(atten)
        var_x = Multiply(name=f"{prefix}.mul")([feat, atten])
        return var_x


class ContextPath():
    """ The Context Path block for BiSeNet Face Parsing. """
    def __init__(self):
        self._resnet = ResNet18()

    def __call__(self, inputs: KerasTensor) -> KerasTensor:
        """ Call the Context Path block.

        Parameters
        ----------
        inputs: :class:`keras.KerasTensor`
            The input to the block

        Returns
        -------
        :class:`keras.KerasTensor`
            The output from the block
        """
        feat8, feat16, feat32 = self._resnet(inputs)

        avg = GlobalAveragePooling2D(name="cp-avgpool")(feat32)
        avg = Reshape((1, 1, avg.shape[-1]))(avg)
        avg = ConvBn(128, kernel_size=1, padding=0, prefix="cp-conv_avg", start_idx=-1)(avg)

        avg_up = UpSampling2D(size=feat32.shape[1:3], name="cp-upsample")(avg)

        feat32 = AttentionRefinementModule(128)(feat32, 32)
        feat32 = Add(name="cp-add")([feat32, avg_up])
        feat32 = UpSampling2D(name="cp-upsample1")(feat32)
        feat32 = ConvBn(128, kernel_size=3, prefix="cp-conv_head32", start_idx=-1)(feat32)

        feat16 = AttentionRefinementModule(128)(feat16, 16)
        feat16 = Add(name="cp-add2")([feat16, feat32])
        feat16 = UpSampling2D(name="cp-upsample2")(feat16)
        feat16 = ConvBn(128, kernel_size=3, prefix="cp-conv_head16", start_idx=-1)(feat16)

        return feat8, feat16, feat32


class FeatureFusionModule():
    """ The Feature Fusion block for BiSeNet Face Parsing

    Parameters
    ----------
    filters: int
        The dimensionality of the output space (i.e. the number of output filters in the
        convolution).
    """
    def __init__(self, filters: int) -> None:
        self._filters = filters

    def __call__(self, inputs: KerasTensor) -> KerasTensor:
        """ Call the Feature Fusion block.

        Parameters
        ----------
        inputs: :class:`keras.KerasTensor`
            The input to the block

        Returns
        -------
        :class:`keras.KerasTensor`
            The output from the block
        """
        feat = Concatenate(name="ffm-concat")(inputs)
        feat = ConvBn(self._filters,
                      kernel_size=1,
                      padding=0,
                      prefix="ffm-convblk",
                      start_idx=-1)(feat)

        atten = GlobalAveragePooling2D(name="ffm-avgpool")(feat)
        atten = Reshape((1, 1, atten.shape[-1]))(atten)
        atten = Conv2D(self._filters // 4, 1, use_bias=False, name="ffm-conv1")(atten)
        atten = Activation("relu", name="ffm-relu")(atten)
        atten = Conv2D(self._filters, 1, use_bias=False, name="ffm-conv2")(atten)
        atten = Activation("sigmoid", name="ffm-sigmoid")(atten)

        var_x = Multiply(name="ffm-mul")([feat, atten])
        var_x = Add(name="ffm-add")([var_x, feat])
        return var_x


class BiSeNetOutput():
    """ The BiSeNet Output block for Face Parsing

    Parameters
    ----------
    filters: int
        The dimensionality of the output space (i.e. the number of output filters in the
        convolution).
    num_class: int
        The number of classes to generate
    label, str, optional
        The label for this output (for naming). Default: `""` (i.e. empty string, or no label)
    """
    def __init__(self, filters: int, num_classes: int, label: str = "") -> None:
        self._filters = filters
        self._num_classes = num_classes
        self._label = label

    def __call__(self, inputs: KerasTensor) -> KerasTensor:
        """ Call the BiSeNet Output block.

        Parameters
        ----------
        inputs: :class:`keras.KerasTensor`
            The input to the block

        Returns
        -------
        :class:`keras.KerasTensor`
            The output from the block
        """
        var_x = ConvBn(self._filters, prefix=f"conv_out{self._label}-conv", start_idx=-1)(inputs)
        var_x = Conv2D(self._num_classes, 1,
                       use_bias=False, name=f"conv_out{self._label}-conv_out")(var_x)
        return var_x


class BiSeNet():
    """ BiSeNet Face-Parsing Mask from https://github.com/zllrunning/face-parsing.PyTorch

    PyTorch model implemented in Keras by TorzDF

    Parameters
    ----------
    weights_path: str
        The path to the keras weights file
    batch_size: int
        The batch size to feed the model
    input_size: int
        The input size to the model
    num_classes: int
        The number of segmentation classes to create
    """
    def __init__(self,
                 weights_path: str,
                 batch_size: int,
                 input_size: int,
                 num_classes: int) -> None:
        logger.debug(parse_class_init(locals()))
        self._batch_size = batch_size
        self._input_size = input_size
        self._num_classes = num_classes
        self._cp = ContextPath()
        self._model = self._load_model(weights_path)
        logger.debug("Initialized: %s", self.__class__.__name__)

    def _load_model(self, weights_path: str) -> Model:
        """ Definition of the BiSeNet-FP  Model.

        Parameters
        ----------
        weights_path: str
            Full path to the model's weights

        Returns
        -------
        :class:`keras.models.Model`
            The BiSeNet-FP model
        """
        input_ = Input((self._input_size, self._input_size, 3))

        features = self._cp(input_)  # res8, cp8, cp16
        feat_fuse = FeatureFusionModule(256)([features[0], features[1]])

        feats = [BiSeNetOutput(256, self._num_classes)(feat_fuse),
                 BiSeNetOutput(64, self._num_classes, label="16")(features[1]),
                 BiSeNetOutput(64, self._num_classes, label="32")(features[2])]

        height, width = input_.shape[1:3]
        output = [UpSampling2D(size=(height // feat.shape[1], width // feat.shape[2]),
                               interpolation="bilinear")(feat)
                  for feat in feats]

        retval = Model(input_, output)
        retval.load_weights(weights_path)
        retval.make_predict_function()
        return retval

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        """ Get predictions from the BiSeNet-FP model

        Parameters
        ----------
        inputs: :class:`numpy.ndarray`
            The input to BiSeNet-FP

        Returns
        -------
        :class:`numpy.ndarray`
            The output from BiSeNet-FP
        """
        return self._model.predict(inputs, verbose=0, batch_size=self._batch_size)


__all__ = get_module_objects(__name__)
