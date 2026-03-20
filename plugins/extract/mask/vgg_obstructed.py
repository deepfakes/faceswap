#!/usr/bin/env python3
"""VGG Obstructed face mask plugin"""
from __future__ import annotations
import logging
import typing as T

import numpy as np

from torch import nn
from torch.nn import functional as F

from lib.utils import get_module_objects, GetModel
from plugins.extract.base import FacePlugin
from . import vgg_obstructed_defaults as cfg

if T.TYPE_CHECKING:
    from torch import Tensor

logger = logging.getLogger(__name__)

# pylint:disable=duplicate-code


class VGGObstructed(FacePlugin):
    """Neural network to process face image into a segmentation mask of the face"""
    def __init__(self) -> None:
        super().__init__(input_size=500,
                         batch_size=cfg.batch_size(),
                         is_rgb=False,
                         dtype="float32",
                         scale=(0, 255),
                         centering="face")
        self.model: VGGObstructedModel

    def load_model(self) -> VGGObstructedModel:
        """Initialize the VGGObstructed Mask model.

        Returns
        -------
        The loaded VGGObstructed model
        """
        weights = GetModel("Nirkin_500_softmax_v2.pth", 8).model_path
        assert isinstance(weights, str)
        return T.cast(VGGObstructedModel, self.load_torch_model(VGGObstructedModel(),
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
        return (batch - np.mean(batch, axis=(1, 2))[:, None, None, :]).transpose(0, 3, 1, 2)

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
        return self.from_torch(batch) * -1.0 + 1.0


class ConvBlock(nn.Module):
    """Convolutional loop with max pooling layer for VGG Obstructed.

    Parameters
    ----------
    in_channels
        The number of input channels to the model
    filters
        The number of filters that should appear in each Conv2D layer
    iterations
        The number of consecutive Conv2D layers to create
    padding
        The amount of padding to apply to the first convolution. Default: 1
    pool_padding
        The amount of padding to apply to the max pooling layer. Default: 1
    """
    def __init__(self,
                 in_channels: int,
                 filters: int,
                 iterations: int,
                 padding: int = 1,
                 pool_padding: int = 1) -> None:
        super().__init__()
        layers = [nn.Conv2d(in_channels, filters, 3, padding=padding),
                  nn.ReLU(inplace=True)]
        for _ in range(iterations - 1):
            layers.append(nn.Conv2d(filters, filters, 3, padding=1))
            layers.append(nn.ReLU(inplace=True))
        self.convs = nn.Sequential(*layers)
        self._pool_padding = pool_padding

    def forward(self, inputs: Tensor) -> Tensor:
        """Call the convolutional loop.

        Parameters
        ----------
        inputs
            The input tensor to the block

        Returns
        -------
        The output tensor from the convolutional block
        """
        x = self.convs(inputs)
        x = F.max_pool2d(x, 2, stride=2, padding=self._pool_padding)
        return x


class ScorePool(nn.Module):
    """Cropped scaling of the pooling layer.

    Parameters
    ----------
    in_channels
        The number of input channels to the model
    scale : float
        The scaling to apply to the pool
    crop : tuple[int, int]
        The amount of 2D cropping to apply. Tuple of (Left/Top, Right/Bottom) `ints`
    """
    def __init__(self, in_channels: int, scale: float, crop: tuple[int, int]) -> None:
        super().__init__()
        self._scale = scale
        self.conv = nn.Conv2d(in_channels, 21, 1)
        self._crop = crop

    def forward(self, inputs: Tensor) -> Tensor:
        """Call the score pool layer.

        Parameters
        ----------
        inputs
            The input tensor to the block

        Returns
        -------
        The output tensor from the block
        """
        x = inputs * self._scale
        x = self.conv(x)
        x = x[:, :, self._crop[0]:-self._crop[1], self._crop[0]:-self._crop[1]]
        return x


class VGGObstructedModel(nn.Module):  # pylint:disable=too-many-instance-attributes
    """VGG Obstructed mask for Faceswap.

    Caffe model re-implemented in Keras by Kyle Vrooman.
    Re-implemented for Pytorch by TorzDF

    References
    ----------
    On Face Segmentation, Face Swapping, and Face Perception (https://arxiv.org/abs/1704.06729)
    Source Implementation: https://github.com/YuvalNirkin/face_segmentation
    Model file sourced from:
    https://github.com/YuvalNirkin/face_segmentation/releases/download/1.0/face_seg_fcn8s.zip
    """
    def __init__(self) -> None:
        super().__init__()
        self.zeropad = nn.ZeroPad2d(100)
        self.conv1 = ConvBlock(3, 64, 2, padding=0, pool_padding=0)
        self.conv2 = ConvBlock(64, 128, 2)
        self.conv3 = ConvBlock(128, 256, 3)
        self.conv4 = ConvBlock(256, 512, 3, pool_padding=0)
        self.conv5 = ConvBlock(512, 512, 3, pool_padding=0)
        self.fc6 = nn.Conv2d(512, 4096, 7)
        self.fc7 = nn.Conv2d(4096, 4096, 1)
        self.score_fr = nn.Conv2d(4096, 21, 1)
        self.upscore2 = nn.ConvTranspose2d(21, 21, 4, stride=2, bias=False)
        self.score_pool4 = ScorePool(512, 0.01, (5, 5))
        self.upscore_pool4 = nn.ConvTranspose2d(21, 21, 4, stride=2, bias=False)
        self.score_pool3 = ScorePool(256, 0.0001, (9, 9))
        self.upscore8 = nn.ConvTranspose2d(21, 21, 16, stride=8, bias=False)

    def forward(self, inputs: Tensor) -> Tensor:
        """Call the VGG Obstructed Model.

        Parameters
        ----------
        inputs
            The input to the model

        Returns
        -------
        The output from the VGG Obstructed model
        """
        x = self.zeropad(inputs)
        x = self.conv1(x)
        x = self.conv2(x)
        pool3 = self.conv3(x)
        pool4 = self.conv4(pool3)
        x = self.conv5(pool4)

        x = F.relu(self.fc6(x), inplace=True)
        x = F.dropout(x, 0.5)
        x = F.relu(self.fc7(x), inplace=True)
        x = F.dropout(x, 0.5)

        x = self.score_fr(x)
        x = self.upscore2(x)
        score_pool4 = self.score_pool4(pool4)
        x = x + score_pool4

        x = self.upscore_pool4(x)
        score_pool3 = self.score_pool3(pool3)
        x = x + score_pool3

        x = self.upscore8(x)
        x = x[:, :, 31:-37, 31:-37]
        return F.softmax(x, dim=1).swapaxes(0, 1)


__all__ = get_module_objects(__name__)
