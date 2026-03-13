#!/usr/bin/env python3
"""UNET DFL face mask plugin

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
import torch
from torch import nn
from torch.nn import functional as F

from lib.utils import get_module_objects, GetModel
from plugins.extract.base import FacePlugin
from . import unet_dfl_defaults as cfg

if T.TYPE_CHECKING:
    from torch import Tensor


logger = logging.getLogger(__name__)
# pylint:disable=duplicate-code


class UNetDFL(FacePlugin):
    """Neural network to process face image into a segmentation mask of the face"""
    def __init__(self) -> None:
        super().__init__(input_size=256,
                         batch_size=cfg.batch_size(),
                         is_rgb=False,
                         dtype="float32",
                         scale=(0, 1),
                         centering="legacy")
        self.model: UnetDFL

    def load_model(self) -> UnetDFL:
        """Initialize the UNet-DFL Model

        Returns
        -------
        The loaded UnetDFL model
        """
        weights = GetModel("DFL_256_sigmoid_v2.pth", 6).model_path
        assert isinstance(weights, str)
        return T.cast(UnetDFL, self.load_torch_model(UnetDFL(), weights))

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
        return self.from_torch(batch.transpose(0, 3, 1, 2)).transpose(0, 2, 3, 1)


class ConvBlock(nn.Module):
    """Convolution block for UnetDFL down-scales

    Parameters
    ----------
    in_channels
        The number of input channels to the block
    filters
        The number of filters for the convolution
    recursions: int
        The number of convolutions to run
    """
    def __init__(self, in_channels: int, filters: int, recursions: int) -> None:
        super().__init__()
        layers = [nn.Conv2d(in_channels, filters, 3, padding=1),
                  nn.ReLU(inplace=True)]
        for _ in range(recursions - 1):
            layers.extend([nn.Conv2d(filters, filters, 3, padding=1),
                           nn.ReLU(inplace=True)])
        self.convs = nn.Sequential(*layers)

    def forward(self, inputs: Tensor) -> Tensor:
        """Convolution Block forward pass

        Parameters
        ----------
        inputs
            The input to the convolution block

        Returns
        -------
        The output from the convolution block
        """
        return self.convs(inputs)


class DecoderBlock(nn.Module):
    """Decoder Block for UnetDFL

    Parameters
    ----------
    in_channels
        The number of input channels to the block
    middle_channels
        The number of filters for the first convolution
    out_channels
        The number of filters for the second convolution
    relu
        ``True`` to use ReLU activation on the first conv. ``False`` to use no activation
    """
    def __init__(self,
                 in_channels: int,
                 middle_channels: int,
                 out_channels: int,
                 relu: bool) -> None:
        super().__init__()
        self._use_relu = relu
        self.conv = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.conv_trans = nn.ConvTranspose2d(middle_channels,
                                             out_channels,
                                             3,
                                             stride=2,
                                             padding=0,
                                             output_padding=0)

    def forward(self, inputs: Tensor) -> Tensor:
        """Decoder block forward pass

        Parameters
        ----------
        inputs
            The input to the decoder block

        Returns
        -------
        The output from the decoder block
        """
        x = self.conv(inputs)
        if self._use_relu:
            x = F.relu(x, inplace=True)
        x = F.relu(self.conv_trans(x), inplace=True)
        return x[:, :, :-1, :-1]


class UnetDFL(nn.Module):  # pylint:disable=too-many-instance-attributes
    """UNet DFL Definition for PyTorch"""
    def __init__(self) -> None:
        super().__init__()
        self.features_0 = ConvBlock(3, 64, 1)
        self.features_3 = ConvBlock(64, 128, 1)
        self.features_8 = ConvBlock(128, 256, 2)
        self.features_13 = ConvBlock(256, 512, 2)
        self.features_18 = ConvBlock(512, 512, 2)
        self.dec1 = DecoderBlock(512, 512, 256, False)
        self.dec2 = DecoderBlock(768, 512, 256, True)
        self.dec3 = DecoderBlock(768, 512, 128, True)
        self.dec4 = DecoderBlock(384, 256, 64, True)
        self.dec5 = DecoderBlock(192, 128, 32, True)
        self.conv2d_6 = nn.Conv2d(96, 64, 3, padding=1)
        self.conv2d_7 = nn.Conv2d(64, 1, 3, padding=1)

    def forward(self, inputs: Tensor) -> Tensor:
        """UnetDFL forward pass

        Parameters
        ----------
        inputs
            The input to UnetDFL

        Returns
        -------
        The output from UnetDFL
        """
        features = []
        features.append(self.features_0(inputs))
        x = F.max_pool2d(features[-1], 2, stride=2)
        features.append(self.features_3(x))
        x = F.max_pool2d(features[-1], 2, stride=2)
        features.append(self.features_8(x))
        x = F.max_pool2d(features[-1], 2, stride=2)
        features.append(self.features_13(x))
        x = F.max_pool2d(features[-1], 2, stride=2)
        features.append(self.features_18(x))
        x = F.max_pool2d(features[-1], 2, stride=2)

        x = torch.cat([self.dec1(x), features[4]], dim=1)
        x = torch.cat([self.dec2(x), features[3]], dim=1)
        x = torch.cat([self.dec3(x), features[2]], dim=1)
        x = torch.cat([self.dec4(x), features[1]], dim=1)
        x = torch.cat([self.dec5(x), features[0]], dim=1)

        x = F.relu(self.conv2d_6(x), inplace=True)
        return F.sigmoid(self.conv2d_7(x))


__all__ = get_module_objects(__name__)
