#!/usr/bin/env python3
""" BiSeNet Face-Parsing mask plugin

Architecture and Pre-Trained Model ported from PyTorch to Keras by TorzDF from
https://github.com/zllrunning/face-parsing.PyTorch
"""

import numpy as np

from keras import backend as K
from keras.layers import (Activation, Add, BatchNormalization, Concatenate, Conv2D,
                          GlobalAveragePooling2D, Input, MaxPooling2D, Multiply, Reshape,
                          UpSampling2D, ZeroPadding2D)

from lib.model.session import KSession
from ._base import Masker, logger


class Mask(Masker):
    """ Neural network to process face image into a segmentation mask of the face """
    def __init__(self, **kwargs):
        git_model_id = 14
        model_filename = "bisnet_face_parsing_v1.h5"
        super().__init__(git_model_id=git_model_id, model_filename=model_filename, **kwargs)
        self.name = "BiSeNet - Face Parsing"
        self.input_size = 512
        self.color_format = "RGB"
        self.vram = 3424  # TODO
        self.vram_warnings = 256  # TODO
        self.vram_per_batch = 80  # TODO
        self.batchsize = self.config["batch-size"]
        self._segment_indices = self._get_segment_indices()

    def _get_segment_indices(self):
        """ Obtain the segment indices to include within the face mask area based on user
        configuration settings.

        Returns
        -------
        list
            The segment indices to include within the face mask area

        Notes
        -----
        Model segment indices:
        0: background, 1: skin, 2: left brow, 3: right brow, 4: left eye, 5: right eye, 6: glasses
        7: left ear, 8: right ear, 9: earings, 10: nose, 11: mouth, 12: upper lip, 13: lower_lip,
        14: neck, 15: neck ?, 16: cloth, 17: hair, 18: hat
        """
        retval = [1, 2, 3, 4, 5, 10, 11, 12, 13]
        if self.config["include_glasses"]:
            retval.append(6)
        if self.config["include_ears"]:
            retval.extend([7, 8, 9])
        if self.config["include_hair"]:
            retval.append(17)
        logger.debug("Selected segment indices: %s", retval)
        return retval

    def init_model(self):
        self.model = BiSeNet(self.model_path,
                             self.config["allow_growth"],
                             self._exclude_gpus,
                             self.input_size,
                             19)
        placeholder = np.zeros((self.batchsize, self.input_size, self.input_size, 3),
                               dtype="float32")
        self.model.predict(placeholder)

    def process_input(self, batch):
        """ Compile the detected faces for prediction """
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

        batch["feed"] = ((np.array([feed.face[..., :3]
                                    for feed in batch["feed_faces"]],
                                   dtype="float32") / 255.0) - mean) / std
        logger.trace("feed shape: %s", batch["feed"].shape)
        return batch

    def predict(self, batch):
        """ Run model to get predictions """
        batch["prediction"] = self.model.predict(batch["feed"])[0]
        return batch

    def process_output(self, batch):
        """ Compile found faces for output """
        pred = batch["prediction"].argmax(-1).astype("uint8")
        batch["prediction"] = np.isin(pred, self._segment_indices).astype("float32")
        return batch

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


_NAME_TRACKER = set()


def _get_name(name, start_idx=1):
    i = start_idx
    while True:
        retval = f"{name}{i}" if i != -1 else name
        if retval not in _NAME_TRACKER:
            break
        i += 1
    _NAME_TRACKER.add(retval)
    return retval


class ConvBn():
    def __init__(self, filters,
                 kernel_size=3, strides=1, padding=1, activation=True, prefix="", start_idx=1):
        self._filters = filters
        self._kernel_size = kernel_size
        self._strides = strides
        self._padding = padding
        self._activation = activation
        self._prefix = f"{prefix}." if prefix else prefix
        self._start_idx = start_idx

    def __call__(self, inputs):
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
    def __init__(self):
        self._feature_index = 1 if K.image_data_format() == "channels_first" else -1

    def _basic_block(self, inputs, prefix, filters, strides=1):
        res = ConvBn(filters, strides=strides, padding=1, prefix=prefix)(inputs)
        res = ConvBn(filters, strides=1, padding=1, activation=False, prefix=prefix)(res)

        shortcut = inputs
        filts = (K.int_shape(shortcut)[self._feature_index], K.int_shape(res)[self._feature_index])
        if strides != 1 or filts[0] != filts[1]:  # Downsample
            name = f"{prefix}.downsample."
            shortcut = Conv2D(filters, 1,
                              strides=strides,
                              use_bias=False,
                              name=_get_name(f"{name}", start_idx=0))(shortcut)
            shortcut = BatchNormalization(epsilon=1e-5,
                                          name=_get_name(f"{name}", start_idx=0))(shortcut)

        var_x = Add(name=f"{prefix}.add")([res, shortcut])
        var_x = Activation("relu", name=f"{prefix}.relu")(var_x)
        return var_x

    def _basic_layer(self, inputs, prefix, filters, bnum, strides=1):
        var_x = self._basic_block(inputs, f"{prefix}.0", filters, strides=strides)
        for i in range(bnum - 1):
            var_x = self._basic_block(var_x, f"{prefix}.{i + 1}", filters, strides=1)
        return var_x

    def __call__(self, inputs):
        var_x = ConvBn(64, kernel_size=7, strides=2, padding=3, prefix="cp.resnet")(inputs)
        var_x = ZeroPadding2D(1, name="cp.resnet.zeropad")(var_x)
        var_x = MaxPooling2D(pool_size=3, strides=2, name="cp.resnet.maxpool")(var_x)

        var_x = self._basic_layer(var_x, "cp.resnet.layer1", 64, 2)
        feat8 = self._basic_layer(var_x, "cp.resnet.layer2", 128, 2, strides=2)
        feat16 = self._basic_layer(feat8, "cp.resnet.layer3", 256, 2, strides=2)
        feat32 = self._basic_layer(feat16, "cp.resnet.layer4", 512, 2, strides=2)

        return feat8, feat16, feat32


class AttentionRefinementModule():
    def __init__(self, filters):
        self._filters = filters

    def __call__(self, inputs, feats):
        prefix = f"cp.arm{feats}"
        feat = ConvBn(self._filters, prefix=f"{prefix}.conv", start_idx=-1, padding=-1)(inputs)
        atten = GlobalAveragePooling2D(name=f"{prefix}.avgpool")(feat)
        atten = Reshape((1, 1, K.int_shape(atten)[-1]))(atten)
        atten = Conv2D(self._filters, 1, use_bias=False, name=f"{prefix}.conv_atten")(atten)
        atten = BatchNormalization(epsilon=1e-5, name=f"{prefix}.bn_atten")(atten)
        atten = Activation("sigmoid", name=f"{prefix}.sigmoid")(atten)
        var_x = Multiply(name=f"{prefix}.mul")([feat, atten])
        return var_x


class ContextPath():
    def __init__(self):
        self._resnet = ResNet18()

    def __call__(self, inputs):
        feat8, feat16, feat32 = self._resnet(inputs)

        avg = GlobalAveragePooling2D(name="cp.avgpool")(feat32)
        avg = Reshape((1, 1, K.int_shape(avg)[-1]))(avg)
        avg = ConvBn(128, kernel_size=1, padding=0, prefix="cp.conv_avg", start_idx=-1)(avg)

        avg_up = UpSampling2D(size=K.int_shape(feat32)[1:3], name="cp.upsample")(avg)

        feat32 = AttentionRefinementModule(128)(feat32, 32)
        feat32 = Add(name="cp.add")([feat32, avg_up])
        feat32 = UpSampling2D(name="cp.upsample1")(feat32)
        feat32 = ConvBn(128, kernel_size=3, prefix="cp.conv_head32", start_idx=-1)(feat32)

        feat16 = AttentionRefinementModule(128)(feat16, 16)
        feat16 = Add(name="cp.add2")([feat16, feat32])
        feat16 = UpSampling2D(name="cp.upsample2")(feat16)
        feat16 = ConvBn(128, kernel_size=3, prefix="cp.conv_head16", start_idx=-1)(feat16)

        return feat8, feat16, feat32


class FeatureFusionModule():
    def __init__(self, filters):
        self._filters = filters

    def __call__(self, inputs):
        feat = Concatenate(name="ffm.concat")(inputs)
        feat = ConvBn(self._filters,
                      kernel_size=1,
                      padding=0,
                      prefix="ffm.convblk",
                      start_idx=-1)(feat)

        atten = GlobalAveragePooling2D(name="ffm.avgpool")(feat)
        atten = Reshape((1, 1, K.int_shape(atten)[-1]))(atten)
        atten = Conv2D(self._filters // 4, 1, use_bias=False, name="ffm.conv1")(atten)
        atten = Activation("relu", name="ffm.relu")(atten)
        atten = Conv2D(self._filters, 1, use_bias=False, name="ffm.conv2")(atten)
        atten = Activation("sigmoid", name="ffm.sigmoid")(atten)

        var_x = Multiply(name="ffm.mul")([feat, atten])
        var_x = Add(name="ffm.add")([var_x, feat])
        return var_x


class BiSeNetOutput():
    def __init__(self, mid_chan, num_classes, label=""):
        self._mid_chan = mid_chan
        self._num_classes = num_classes
        self._label = label

    def __call__(self, inputs):
        var_x = ConvBn(self._mid_chan, prefix=f"conv_out{self._label}.conv", start_idx=-1)(inputs)
        var_x = Conv2D(self._num_classes, 1,
                       use_bias=False, name=f"conv_out{self._label}.conv_out")(var_x)
        return var_x


class BiSeNet(KSession):
    """ BiSeNet Face-Parsing Mask from https://github.com/zllrunning/face-parsing.PyTorch

    PyTorch model reimplemented in Keras by TorzDF

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
    input_size: int
        The input size to the model
    num_classes: int
        The number of segmentation classes to create
    """
    def __init__(self, model_path, allow_growth, exclude_gpus, input_size, num_classes):
        super().__init__("BiSeNet Face Parsing",
                         model_path,
                         allow_growth=allow_growth,
                         exclude_gpus=exclude_gpus)
        self._input_size = input_size
        self._num_classes = num_classes
        self._cp = ContextPath()
        self.define_model(self._model_definition)
        self.load_model_weights()

    def _model_definition(self):
        """ Definition of the VGG Obstructed Model.

        Returns
        -------
        tuple
            The tensor input to the model and tensor output to the model for compilation by
            :func`define_model`
        """
        input_ = Input((self._input_size, self._input_size, 3))

        feat_res8, feat_cp8, feat_cp16 = self._cp(input_)
        feat_fuse = FeatureFusionModule(256)([feat_res8, feat_cp8])

        feat_out = BiSeNetOutput(256, self._num_classes)(feat_fuse)
        feat_out16 = BiSeNetOutput(64, self._num_classes, label="16")(feat_cp8)
        feat_out32 = BiSeNetOutput(64, self._num_classes, label="32")(feat_cp16)

        height, width = K.int_shape(input_)[1:3]
        f_h, f_w = K.int_shape(feat_out)[1:3]
        f_h16, f_w16 = K.int_shape(feat_out16)[1:3]
        f_h32, f_w32 = K.int_shape(feat_out32)[1:3]

        feat_out = UpSampling2D(size=(height // f_h, width // f_w),
                                interpolation="bilinear")(feat_out)
        feat_out16 = UpSampling2D(size=(height // f_h16, width // f_w16),
                                  interpolation="bilinear")(feat_out16)
        feat_out32 = UpSampling2D(size=(height // f_h32, width // f_w32),
                                  interpolation="bilinear")(feat_out32)

        return input_, [feat_out, feat_out16, feat_out32]
