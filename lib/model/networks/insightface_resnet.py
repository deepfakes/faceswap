"""InsightFace ResNet (IR) and InsightFace ResNet Squeeze + Excite (IRSE) for inference

From: https://github.com/deepinsight/insightface and  https://github.com/HuangYG123/CurricularFace

Released under MIT License
"""
import typing as T

import torch
from torch import nn

from lib.utils import get_module_objects


class SEModule(nn.Module):
    """Squeeze and Excite Block for IRNet

    Parameters
    ----------
    in_channels
        The number of input channels
    reduction
        The reduction factor for squeeze and excite
    """
    def __init__(self, in_channels: int, reduction: int) -> None:
        super().__init__()
        out_channels = in_channels // reduction
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, out_channels, 1, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(out_channels, in_channels, 1, padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass through the IRNet Squeeze and Excite Block"""
        x = self.avg_pool(inputs)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return inputs * x


class BasicBlockIR(nn.Module):
    """A Basic Block for InsightFace ResNet

    Parameters
    ----------
    in_channels
        The number of input channels to the layer
    depth
        The depth of the layer
    stride
        The Convolution stride
    use_se
        ``True`` to add squeeze and excite layer
    """
    def __init__(self, in_channels: int, depth: int, stride: int, use_se: bool) -> None:
        super().__init__()
        if in_channels == depth:
            self.shortcut_layer: nn.Sequential | nn.MaxPool2d = nn.MaxPool2d(1, stride)
        else:
            self.shortcut_layer = nn.Sequential(
                nn.Conv2d(in_channels, depth, 1, stride=stride, bias=False),
                nn.BatchNorm2d(depth))
        res_layer = [
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, depth, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(depth),
            nn.PReLU(depth),
            nn.Conv2d(depth, depth, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(depth)]
        if use_se:
            res_layer.append(SEModule(depth, 16))
        self.res_layer = nn.Sequential(*res_layer)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass through the IRNet basic block

        Parameters
        ----------
        inputs
            The input to the IRNet Block

        Returns
        -------
        The output from the IRNet Block
        """
        res = self.res_layer(inputs)
        shortcut = self.shortcut_layer(inputs)
        return res + shortcut


class BottleneckIR(nn.Module):
    """Bottleneck for IRNet

    Parameters
    ----------
    in_channels
        The number of input channels to the layer
    depth
        The depth of the layer
    stride
        The Convolution stride
    use_se
        ``True`` to add squeeze and excite layer
    """
    def __init__(self, in_channels: int, depth: int, stride: int, use_se: bool) -> None:
        super().__init__()
        super().__init__()
        shrink_channel = depth // 4
        if in_channels == depth:
            self.shortcut_layer: nn.Sequential | nn.MaxPool2d = nn.MaxPool2d(1, stride)
        else:
            self.shortcut_layer = nn.Sequential(
                nn.Conv2d(in_channels, depth, 1, stride=stride, bias=False),
                nn.BatchNorm2d(depth))
        res_layer = [nn.BatchNorm2d(in_channels),
                     nn.Conv2d(in_channels, shrink_channel, 1, stride=1, padding=0, bias=False),
                     nn.BatchNorm2d(shrink_channel),
                     nn.PReLU(shrink_channel),
                     nn.Conv2d(shrink_channel, shrink_channel, 3, stride=1, padding=1, bias=False),
                     nn.BatchNorm2d(shrink_channel),
                     nn.PReLU(shrink_channel),
                     nn.Conv2d(shrink_channel, depth, 1, stride=stride, padding=0, bias=False),
                     nn.BatchNorm2d(depth)]
        if use_se:
            res_layer.append(SEModule(depth, 16))
        self.res_layer = nn.Sequential(*res_layer)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass through the IRNet Bottleneck

        Parameters
        ----------
        inputs
            The input to the IRNet Bottleneck

        Returns
        -------
        The output from the IRNet Bottleneck
        """
        res = self.res_layer(inputs)
        shortcut = self.shortcut_layer(inputs)
        return res + shortcut


class Flatten(nn.Module):
    """Flatten layer for IRNet """
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Flatten the inbound layer

        Parameters
        ----------
        inputs
            The input layer to be flattened

        Returns
        -------
        The flattened input layer
        """
        return inputs.reshape(inputs.size(0), -1)


class IRNet(nn.Module):
    """Implementation if InsightFace ResNet with Squeeze + Excite support

    Parameters
    ----------
    input_size
        The input size to the model. Must be 112 or 224
    block_filters
        The number of in_channels to each block layer for each pass
    block_recursions
        The number of recursions within each block
    num_features
        The number of num_features to output. Default: 512
    use_se
        ``True`` to use Squeeze and Excite. ``False`` to use standard IR ResNet. Default: ``False``
    use_bottleneck
        ``True`` to use the Bottleneck block. ``False`` to use the Basic block. Default: ``False``
    """
    def __init__(self,
                 input_size: T.Literal[112, 224],
                 block_filters: tuple[int, int, int, int],
                 block_recursions: tuple[int, int, int, int],
                 num_features: int = 512,
                 use_se: bool = False,
                 use_bottleneck: bool = False) -> None:
        super().__init__()
        self.input_layer = nn.Sequential(nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False),
                                         nn.BatchNorm2d(64),
                                         nn.PReLU(64))
        self.body = self._get_blocks(block_filters, block_recursions, use_se, use_bottleneck)
        self.output_layer = self._get_output_layer(input_size, num_features)

    @classmethod
    def _get_blocks(cls,
                    block_filters: tuple[int, int, int, int],
                    block_recursions: tuple[int, int, int, int],
                    use_se: bool,
                    use_bottleneck: bool) -> nn.Sequential:
        """Obtain the IRNet Blocks for the given configuration

        Parameters
        ----------
        block_filters
            The number of in_channels to each block layer for each pass
        block_recursions
            The number of recursions within each block
        use_se
            ``True`` to build IRNetSE ``False`` to build IRNet
        use_bottleneck
            ``True`` to use the Bottleneck block. ``False`` to use the basic block

        Returns
        -------
        The configured blocks
        """
        depth = 64
        block = BottleneckIR if use_bottleneck else BasicBlockIR
        layers = []
        for in_channels, units in zip(block_filters, block_recursions):
            layers.append(block(in_channels, depth, 2, use_se))
            for _ in range(units - 1):
                layers.append(block(depth, depth, 1, use_se))
            depth *= 2
        return nn.Sequential(*layers)

    @classmethod
    def _get_output_layer(cls, input_size: T.Literal[112, 224], num_features: int
                          ) -> nn.Sequential:
        """Obtain the output layer of the model, based on input size and number of layers

        Parameters
        ----------
        input_size
            The input size to the model. Must be 112 or 224
        num_features
            The number of num_features to output

        Returns
        -------
        The output layer of the model
        """
        fc_scale = 7 * 7 if input_size == 112 else 14 * 14
        return nn.Sequential(nn.BatchNorm2d(num_features),
                             nn.Dropout(0.4),
                             Flatten(),
                             nn.Linear(num_features * fc_scale, 512),
                             nn.BatchNorm1d(512, affine=False))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass through IRNet

        Parameters
        ----------
        inputs
            The input to IRNet

        Returns
        -------
        The output from IRNet
        """
        x = self.input_layer(inputs)
        x = self.body(x)
        x = self.output_layer(x)
        return x


def ir_18(input_size: T.Literal[112, 224]):
    """Obtain an IRNet-18 model

    Parameters
    ----------
    input_size
        The input size to the model
    """
    return IRNet(input_size,
                 block_filters=(64, 64, 128, 256),
                 block_recursions=(2, 2, 2, 2),
                 num_features=512,
                 use_se=False,
                 use_bottleneck=False)


def ir_34(input_size: T.Literal[112, 224]) -> IRNet:
    """Obtain an IRNet-34 model

    Parameters
    ----------
    input_size
        The input size to the model
    """
    return IRNet(input_size,
                 block_filters=(64, 64, 128, 256),
                 block_recursions=(3, 4, 6, 3),
                 num_features=512,
                 use_se=False,
                 use_bottleneck=False)


def ir_50(input_size: T.Literal[112, 224]) -> IRNet:
    """Obtain an IRNet-50 model

    Parameters
    ----------
    input_size
        The input size to the model
    """
    return IRNet(input_size,
                 block_filters=(64, 64, 128, 256),
                 block_recursions=(3, 4, 14, 3),
                 num_features=512,
                 use_se=False,
                 use_bottleneck=False)


def ir_101(input_size: T.Literal[112, 224]) -> IRNet:
    """Obtain an IRNet-101 model

    Parameters
    ----------
    input_size
        The input size to the model
    """
    return IRNet(input_size,
                 block_filters=(64, 64, 128, 256),
                 block_recursions=(3, 13, 30, 3),
                 num_features=512,
                 use_se=False,
                 use_bottleneck=False)


def ir_152(input_size: T.Literal[112, 224]) -> IRNet:
    """Obtain an IRNet-152 model

    Parameters
    ----------
    input_size
        The input size to the model
    """
    return IRNet(input_size,
                 block_filters=(64, 256, 512, 1024),
                 block_recursions=(3, 8, 36, 3),
                 num_features=2048,
                 use_se=False,
                 use_bottleneck=True)


def ir_200(input_size: T.Literal[112, 224]) -> IRNet:
    """Obtain an IRNet-200 model

    Parameters
    ----------
    input_size
        The input size to the model
    """
    return IRNet(input_size,
                 block_filters=(64, 256, 512, 1024),
                 block_recursions=(3, 24, 36, 3),
                 num_features=2048,
                 use_se=False,
                 use_bottleneck=True)


def ir_se_50(input_size: T.Literal[112, 224]) -> IRNet:
    """Obtain an IRNetSE50 model

    Parameters
    ----------
    input_size
        The input size to the model
    """
    return IRNet(input_size,
                 block_filters=(64, 64, 128, 256),
                 block_recursions=(3, 4, 14, 3),
                 num_features=512,
                 use_se=True,
                 use_bottleneck=False)


def ir_se_101(input_size: T.Literal[112, 224]) -> IRNet:
    """Obtain an IRNetSE101 model

    Parameters
    ----------
    input_size
        The input size to the model
    """
    return IRNet(input_size,
                 block_filters=(64, 64, 128, 256),
                 block_recursions=(3, 13, 30, 3),
                 num_features=512,
                 use_se=True,
                 use_bottleneck=False)


def ir_se_152(input_size: T.Literal[112, 224]) -> IRNet:
    """Obtain an IRNetSE152 model

    Parameters
    ----------
    input_size
        The input size to the model
    """
    return IRNet(input_size,
                 block_filters=(64, 256, 512, 1024),
                 block_recursions=(3, 8, 36, 3),
                 num_features=2048,
                 use_se=True,
                 use_bottleneck=True)


def ir_se_200(input_size: T.Literal[112, 224]) -> IRNet:
    """Obtain an IRNetSE200 model

    Parameters
    ----------
    input_size
        The input size to the model
    """
    return IRNet(input_size,
                 block_filters=(64, 256, 512, 1024),
                 block_recursions=(3, 24, 36, 3),
                 num_features=2048,
                 use_se=True,
                 use_bottleneck=True)


__all__ = get_module_objects(__name__)
