#!/usr/bin/env python3
"""BiSeNet Face-Parsing mask plugin

Architecture and Pre-Trained Model ported from PyTorch to Keras by TorzDF from
https://github.com/zllrunning/face-parsing.PyTorch
"""
from __future__ import annotations
import logging
import typing as T

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from lib.utils import get_module_objects, GetModel
from plugins.extract.base import FacePlugin
from . import bisenet_fp_defaults as cfg

if T.TYPE_CHECKING:
    from torch import Tensor

logger = logging.getLogger(__name__)
# pylint:disable=duplicate-code


class BiSeNetFP(FacePlugin):
    """Neural network to process face image into a segmentation mask of the face"""
    def __init__(self) -> None:
        super().__init__(input_size=512,
                         batch_size=cfg.batch_size(),
                         is_rgb=True,
                         dtype="float32",
                         scale=(0, 1),
                         force_cpu=cfg.cpu(),
                         centering="head" if cfg.include_hair() else "face")
        self.model: BiSeNet
        self._is_faceswap, self._git_version = self._check_weights_selection()
        self._segment_indices = self._get_segment_indices()
        self._storage_centering = "head" if cfg.include_hair() else "face"
        """The mask type/storage centering to use"""
        # Separate storage for face and head masks
        self.storage_name = f"{self.storage_name}_{self.centering}"

        mean = (0.384, 0.314, 0.279) if self._is_faceswap else (0.485, 0.456, 0.406)
        std = (0.324, 0.286, 0.275) if self._is_faceswap else (0.229, 0.224, 0.225)
        self._mean = np.array(mean, dtype="float32")
        self._std = np.array(std, dtype="float32")

    def _check_weights_selection(self) -> tuple[bool, int]:
        """Check which weights have been selected.

        This is required for passing along the correct file name for the corresponding weights
        selection.

        Returns
        -------
        is_faceswap
            ``True`` if `faceswap` trained weights have been selected. ``False`` if `original`
            weights have been selected.
        version
            ``1`` for non-faceswap, ``2`` if faceswap and full-head model is required. ``3`` if
            faceswap and full-face is required
        """
        is_faceswap = cfg.weights() == "faceswap"
        version = 4 if not is_faceswap else 5 if cfg.include_hair() else 6
        return is_faceswap, version

    def _get_segment_indices(self) -> list[int]:
        """Obtain the segment indices to include within the face mask area based on user
        configuration settings.

        Returns
        -------
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

    def load_model(self) -> BiSeNet:
        """Initialize the BiSeNet Face Parsing model.

        Returns
        -------
        The loaded BiSeNetFP model
        """
        weights = GetModel(f"bisenet_face_parsing_v{self._git_version}.pth", 14).model_path
        assert isinstance(weights, str)
        return T.cast(BiSeNet, self.load_torch_model(BiSeNet(5 if self._is_faceswap else 19),
                                                     weights,
                                                     return_indices=[0]))

    def pre_process(self, batch: np.ndarray) -> np.ndarray:
        """Format the detected faces for prediction

        Parameters
        ----------
        batch
            The batch of aligned faces in the correct format for the model

        Returns
        -------
        The updated images for feeding the model
        """
        return ((batch - self._mean) / self._std).transpose(0, 3, 1, 2)

    def process(self, batch: np.ndarray) -> np.ndarray:
        """Get the masks from the model

        Parameters
        ----------
        batch
            The batch to feed into the masker

        Returns
        -------

            The predicted masks from the plugin
        """
        return self.from_torch(batch).transpose(0, 2, 3, 1)

    def post_process(self, batch: np.ndarray) -> np.ndarray:
        """Process the output from the model

        Parameters
        ----------
        batch
            The predictions from the masker

        Returns
        -------
        The final masks
        """
        pred = batch.argmax(-1).astype("uint8")
        return np.isin(pred, self._segment_indices).astype("float32")

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


# Resnet18
class BasicBlock(nn.Module):
    """The basic building block for ResNet 18.

    Parameters
    ----------
    in_channels
        The number of input channels
    filters
        The dimensionality of the output space (i.e. the number of output filters in the
        convolution).
    stride
        The strides of the convolution along the height and width. Default: `1`
    """
    def __init__(self, in_channels: int, filters: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, filters, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(filters)
        self.conv2 = nn.Conv2d(filters, filters, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(filters)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if in_channels != filters or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, filters, 1, stride=stride, bias=False),
                nn.BatchNorm2d(filters),
                )

    def forward(self, inputs: Tensor) -> Tensor:
        """Call the ResNet 18 basic block.

        Parameters
        ----------
        inputs
            The input to the block

        Returns
        -------
        The output from the block
        """
        residual = F.relu(self.bn1(self.conv1(inputs)))
        residual = self.bn2(self.conv2(residual))
        shortcut = inputs if self.downsample is None else self.downsample(inputs)
        out = self.relu(shortcut + residual)
        return out


class ResNet18(nn.Module):
    """ResNet 18 block. Used at the start of BiSeNet Face Parsing. """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.layer1 = self._basic_layer(64, 64, 2, stride=1)
        self.layer2 = self._basic_layer(64, 128, 2, stride=2)
        self.layer3 = self._basic_layer(128, 256, 2, stride=2)
        self.layer4 = self._basic_layer(256, 512, 2, stride=2)

    @classmethod
    def _basic_layer(cls,
                     in_channels: int,
                     filters: int,
                     num_blocks: int,
                     stride: int = 1) -> nn.Sequential:
        """The basic layer for ResNet 18. Recursively builds from :func:`_basic_block`.

        Parameters
        ----------
        in_channels
            The number of input channels
        filters
            The dimensionality of the output space (i.e. the number of output filters in the
            convolution).
        num_blocks
            The number of basic blocks to recursively build
        stride
            The strides of the convolution along the height and width. Default: `1`

        Returns
        -------
        The basic layer module
        """
        layers = [BasicBlock(in_channels, filters, stride=stride)]
        for _ in range(num_blocks - 1):
            layers.append(BasicBlock(filters, filters, stride=1))
        return nn.Sequential(*layers)

    def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Call the ResNet 18 block.

        Parameters
        ----------
        inputs
            The input to the ResNet 18

        Returns
        -------
        The feature outputs from ResNet 18
        """
        x = self.maxpool(F.relu(self.bn1(self.conv1(inputs))))
        feat8 = self.layer2(self.layer1(x))
        feat16 = self.layer3(feat8)
        feat32 = self.layer4(feat16)
        return feat8, feat16, feat32


# bisenet
class ConvBNReLU(nn.Module):
    """Convolutional 3D with Batch Normalization block.

    Parameters
    ----------
    in_channels
        The number of input channels
    filters
        The dimensionality of the output space (i.e. the number of output filters in the
        convolution).
    kernel_size
        The height and width of the 2D convolution window. Default: `3`
    strides
        The strides of the convolution along the height and width. Default: `1`
    padding
        The amount of padding to apply prior to the first Convolutional Layer. Default: `1`
    """
    def __init__(self,
                 in_channels: int,
                 filters: int,
                 kernel_size: int = 3,
                 strides: int = 1,
                 padding: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels,
                              filters,
                              kernel_size,
                              stride=strides,
                              padding=padding,
                              bias=False)
        self.bn = nn.BatchNorm2d(filters)

    def forward(self, inputs: Tensor) -> Tensor:
        """Call the Convolutional Batch Normalization block.

        Parameters
        ----------
        inputs
            The input to the block

        Returns
        -------
        The output from the block
        """
        return F.relu(self.bn(self.conv(inputs)))


class BiSeNetOutput(nn.Module):
    """The BiSeNet Output block for Face Parsing

    Parameters
    ----------
    in_channels
        The number of input channels
    filters
        The dimensionality of the output space (i.e. the number of output filters in the
        convolution).
    num_class
        The number of classes to generate
    """
    def __init__(self, in_channels: int, filters: int, num_classes: int) -> None:
        super().__init__()
        self.conv1 = ConvBNReLU(in_channels, filters, kernel_size=3, strides=1, padding=1)
        self.conv_out = nn.Conv2d(filters, num_classes, 1, bias=False)

    def forward(self, inputs: Tensor) -> Tensor:
        """Call the BiSeNet Output block.

        Parameters
        ----------
        inputs
            The input to the block

        Returns
        -------
        The output from the block
        """
        return self.conv_out(self.conv1(inputs))


class AttentionRefinementModule(nn.Module):
    """The Attention Refinement block for BiSeNet Face Parsing

    Parameters
    ----------
    in_channels
        The number of input channels to the block
    filters
        The dimensionality of the output space (i.e. the number of output filters in the
        convolution).
    """
    def __init__(self, in_channels: int, filters: int) -> None:
        super().__init__()
        self.conv = ConvBNReLU(in_channels, filters, kernel_size=3, strides=1, padding=1)
        self.conv_atten = nn.Conv2d(filters, filters, kernel_size=1, bias=False)
        self.bn_atten = nn.BatchNorm2d(filters)
        self.sigmoid_atten = nn.Sigmoid()

    def forward(self, inputs: Tensor) -> Tensor:
        """Call the Attention Refinement block.

        Parameters
        ----------
        inputs
            The input to the block

        Returns
        -------
        The output from the block
        """
        feat = self.conv(inputs)
        attention = F.avg_pool2d(feat, feat.size()[2:])  # pylint:disable=not-callable
        attention = self.sigmoid_atten(self.bn_atten(self.conv_atten(attention)))
        out = torch.mul(feat, attention)
        return out


class ContextPath(nn.Module):
    """The Context Path block for BiSeNet Face Parsing. """
    def __init__(self):
        super().__init__()
        self.resnet = ResNet18()
        self.arm16 = AttentionRefinementModule(256, 128)
        self.arm32 = AttentionRefinementModule(512, 128)
        self.conv_head32 = ConvBNReLU(128, 128, kernel_size=3, strides=1, padding=1)
        self.conv_head16 = ConvBNReLU(128, 128, kernel_size=3, strides=1, padding=1)
        self.conv_avg = ConvBNReLU(512, 128, kernel_size=1, strides=1, padding=0)

    def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Call the Context Path block.

        Parameters
        ----------
        inputs
            The input to the block

        Returns
        -------
        The feature outputs from ResNet 18
        """
        feat8, feat16, feat32 = self.resnet(inputs)
        dim_8 = feat8.size()[2:]
        dim_16 = feat16.size()[2:]
        dim_32 = feat32.size()[2:]
        avg = F.interpolate(self.conv_avg(F.avg_pool2d(feat32,  # pylint:disable=not-callable
                                                       feat32.size()[2:])),
                            dim_32,
                            mode='nearest')
        feat32 = self.conv_head32(F.interpolate(self.arm32(feat32) + avg, dim_16, mode='nearest'))
        feat16 = self.conv_head16(F.interpolate(self.arm16(feat16) + feat32,
                                                dim_8,
                                                mode='nearest'))
        return feat8, feat16, feat32


class FeatureFusionModule(nn.Module):
    """The Feature Fusion block for BiSeNet Face Parsing

    Parameters
    ----------
    in_channels
        The number of input channels to the module
    filters
        The dimensionality of the output space (i.e. the number of output filters in the
        convolution).
    """
    def __init__(self, in_channels: int, filters: int) -> None:
        super().__init__()
        self.convblk = ConvBNReLU(in_channels, filters, kernel_size=1, strides=1, padding=0)
        self.conv1 = nn.Conv2d(filters, filters // 4, 1, stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(filters // 4, filters, 1, stride=1, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, feat_spatial: Tensor, feat_context: Tensor) -> Tensor:
        """Call the Feature Fusion block.

        Parameters
        ----------
        feat_spatial
            The spatial features input to the block
        feat_context
            The context features input to the block

        Returns
        -------
        The output from the block
        """
        feat = self.convblk(torch.cat([feat_spatial, feat_context], dim=1))
        attention = self.sigmoid(self.conv2(self.relu(self.conv1(
            F.avg_pool2d(feat, feat.size()[2:])))))  # pylint:disable=not-callable

        return torch.mul(feat, attention) + feat


class BiSeNet(nn.Module):
    """BiSeNet Face-Parsing Mask from https://github.com/zllrunning/face-parsing.PyTorch

    PyTorch model implemented in Keras and then back to pytorch by TorzDF, because why not?

    Parameters
    ----------
    num_classes
        The number of segmentation classes to create
    """
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.cp = ContextPath()
        self.ffm = FeatureFusionModule(256, 256)
        self.conv_out = BiSeNetOutput(256, 256, num_classes)
        self.conv_out16 = BiSeNetOutput(128, 64, num_classes)
        self.conv_out32 = BiSeNetOutput(128, 64, num_classes)
        logger.debug("Initialized: %s", self.__class__.__name__)

    def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Get predictions from the BiSeNet-FP model

        Parameters
        ----------
        inputs
            The input to BiSeNet-FP

        Returns
        -------
        The outputs from BiSeNet-FP
        """
        dims = inputs.size()[2:]
        feat_sp, feat_cp8, feat_cp16 = self.cp(inputs)
        feat_fuse = self.ffm(feat_sp, feat_cp8)

        feats = [self.conv_out(feat_fuse),
                 self.conv_out16(feat_cp8),
                 self.conv_out32(feat_cp16)]
        output = tuple(F.interpolate(feat, dims, mode='bilinear', align_corners=True)
                       for feat in feats)
        assert len(output) == 3
        return output


__all__ = get_module_objects(__name__)
