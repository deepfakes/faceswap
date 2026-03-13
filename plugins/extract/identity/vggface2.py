#!/usr/bin python3
"""VGGFace inference"""

from __future__ import annotations
import logging
import typing as T

import numpy as np

from torch import nn
from torch.nn import functional as F

from lib.utils import get_module_objects, GetModel
from plugins.extract.base import FacePlugin
from . import vggface2_defaults as cfg

if T.TYPE_CHECKING:
    from torch import Tensor

logger = logging.getLogger(__name__)


class VGGFace2(FacePlugin):
    """VGGFace2 feature extraction.

    Extracts feature vectors from faces in order to compare similarity.

    Notes
    -----
    Input images should be in BGR Order

    Model exported from: https://github.com/WeidiXie/Keras-VGGFace2-ResNet50 which is based on:
    https://www.robots.ox.ac.uk/~vgg/software/vgg_face/


    Licensed under Creative Commons Attribution License.
    https://creativecommons.org/licenses/by-nc/4.0/
    """

    def __init__(self) -> None:
        super().__init__(input_size=224,
                         batch_size=cfg.batch_size(),
                         is_rgb=False,
                         dtype="float32",
                         scale=(0, 255),
                         force_cpu=cfg.cpu(),
                         centering="legacy")
        self.model: VGGFace2Model

        # Average image provided in https://github.com/ox-vgg/vgg_face2
        self._average_img = np.array([91.4953, 103.8827, 131.0912], dtype="float32")
        logger.debug("Initialized %s", self.__class__.__name__)

    def load_model(self) -> VGGFace2Model:
        """Initialize VGG Face 2 Model.

        Returns
        -------
        The loaded VGGFace2 model
        """
        # pylint:disable=duplicate-code
        weights = GetModel("vggface2_resnet50_v3.pth", 10).model_path
        assert isinstance(weights, str)
        return T.cast(VGGFace2Model, self.load_torch_model(VGGFace2Model(), weights))

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
        return (batch - self._average_img).transpose(0, 3, 1, 2)

    def process(self, batch: np.ndarray) -> np.ndarray:
        """Get the identity matrix from the model

        Parameters
        ----------
        batch
            The batch to feed into the recognition plugin

        Returns
        -------
        The predictions from the plugin
        """
        return self.from_torch(batch)


# Model definition
class ConvBlock(nn.Module):
    """Convolution block for ResNet50

    Parameters
    ----------
    in_channels
        The number of input channels
    filters
        The filters for the 1st and 2nd conv layers in the main path
    kernel
        The kernel size of middle conv layer of the block
    stride
        The stride length for the first and last convolution
    """
    def __init__(self, in_channels: int, filters: int, kernel: int, stride: int = 2) -> None:
        super().__init__()
        bottleneck = filters // 4
        self.reduce_conv = nn.Conv2d(in_channels, bottleneck, 1, stride=stride, bias=False)
        self.reduce_bn = nn.BatchNorm2d(bottleneck, eps=0.001, momentum=0.01)
        self.conv = nn.Conv2d(bottleneck, bottleneck, kernel, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(bottleneck, eps=0.001, momentum=0.01)
        self.increase_conv = nn.Conv2d(bottleneck, filters, 1, stride=1, bias=False)
        self.increase_bn = nn.BatchNorm2d(filters, eps=0.001, momentum=0.01)
        self.proj = nn.Conv2d(in_channels, filters, 1, stride=stride, bias=False)
        self.proj_bn = nn.BatchNorm2d(filters, eps=0.001, momentum=0.01)

    def forward(self, inputs: Tensor) -> Tensor:
        """Call the resnet50 ConvBlock

        Parameters
        ----------
        inputs
            Input tensor

        Returns
        -------
        Output tensor from the ConvBlock
        """
        x = F.relu(self.reduce_bn(self.reduce_conv(inputs)), inplace=True)
        x = F.relu(self.bn(self.conv(x)), inplace=True)
        x = self.increase_bn(self.increase_conv(x))
        residual = self.proj_bn(self.proj(inputs))
        return F.relu(x + residual, inplace=True)


class IdentityBlock(nn.Module):
    """Identity block for ResNet50

    Parameters
    ----------
    in_channels
        The number of input channels
    filters
        The filters for the 1st and 2nd conv layers in the main path
    kernel
        The kernel size of middle conv layer of the block
    """
    def __init__(self, in_channels: int, filters: int, kernel: int) -> None:
        super().__init__()
        self.reduce_conv = nn.Conv2d(in_channels, filters, 1, bias=False)
        self.reduce_bn = nn.BatchNorm2d(filters, eps=0.001, momentum=0.01)
        self.conv = nn.Conv2d(filters, filters, kernel, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(filters, eps=0.001, momentum=0.01)
        self.increase_conv = nn.Conv2d(filters, in_channels, 1, bias=False)
        self.increase_bn = nn.BatchNorm2d(in_channels, eps=0.001, momentum=0.01)

    def forward(self, inputs: Tensor) -> Tensor:
        """Call the resnet50 Identity block

        Parameters
        ----------
        inputs
            Input tensor

        Returns
        -------
        Output tensor from the Identity block
        """
        x = F.relu(self.reduce_bn(self.reduce_conv(inputs)))
        x = F.relu(self.bn(self.conv(x)))
        x = self.increase_bn(self.increase_conv(x))
        return F.relu(x + inputs, inplace=True)


class ResNet50(nn.Module):
    """ResNet50 imported for VGG-Face2 adapted from
    https://github.com/WeidiXie/Keras-VGGFace2-ResNet50
    """
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 64, 7, stride=2, bias=False)
        self.bn = nn.BatchNorm2d(64, eps=0.001, momentum=0.01)
        self.block1 = ConvBlock(64, 256, 3, stride=1)
        self.id1 = nn.Sequential(*[IdentityBlock(256, 64, 3) for _ in range(2)])
        self.block2 = ConvBlock(256, 512, 3, stride=2)
        self.id2 = nn.Sequential(*[IdentityBlock(512, 128, 3) for _ in range(3)])
        self.block3 = ConvBlock(512, 1024, 3, stride=2)
        self.id3 = nn.Sequential(*[IdentityBlock(1024, 256, 3) for _ in range(5)])
        self.block4 = ConvBlock(1024, 2048, 3, stride=2)
        self.id4 = nn.Sequential(*[IdentityBlock(2048, 512, 3) for _ in range(2)])

    def forward(self, inputs: Tensor) -> Tensor:
        """Call the resnet50 Network

        Parameters
        ----------
        inputs
            Input tensor

        Returns
        -------
        Output tensor from resnet50
        """
        x = F.pad(inputs, (2, 3, 2, 3), mode="constant")
        x = F.relu(self.bn(self.conv(x)), inplace=True)
        x = F.max_pool2d(x, 3, stride=2)
        x = self.id1(self.block1(x))
        x = self.id2(self.block2(x))
        x = self.id3(self.block3(x))
        return self.id4(self.block4(x))


class VGGFace2Model(nn.Module):
    """VGG-Face 2 model with resnet 50 backbone. Adapted from
    https://github.com/WeidiXie/Keras-VGGFace2-ResNet50
    """
    def __init__(self) -> None:
        super().__init__()
        self.resnet = ResNet50()
        self.dim_proj = nn.Linear(2048, 512)

    def forward(self, inputs: Tensor) -> Tensor:
        """Forward pass through the VGGFace2 model

        Parameters
        ----------
        inputs
            Input to the VGGFace2 Model

        Returns
        -------
        Output from the VGGFace2 Model
        """
        x = self.resnet(inputs)
        x = F.avg_pool2d(x, 7, stride=1)  # pylint:disable=not-callable
        x = F.relu(self.dim_proj(x.view(x.size(0), -1)), inplace=True)
        return F.normalize(x, p=2, dim=1)


__all__ = get_module_objects(__name__)
