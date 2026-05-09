#!/usr/bin/env python3
"""Custom Feature Map Loss Functions for faceswap.py"""
from __future__ import annotations
from dataclasses import dataclass, field
import logging
import typing as T

import torch
from torch import nn
from torchvision.models import alexnet, squeezenet1_1, vgg16, feature_extraction

from lib.logger import parse_class_init
from lib.utils import get_module_objects, GetModel

if T.TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)


@dataclass
class NetInfo:
    """Data class for holding information about Trunk and Linear Layer nets.

    Parameters
    ----------
    model_id
        The model ID for the model stored in the deepfakes Model repo
    model_name
        The filename of the decompressed model/weights file
    net
        The net definition to load, if any. Default:``None``
    outputs
        For trunk networks the name of the output feature layers. For linear networks the number of
        input channels to each layer
    pad_amount
        For trunk networks, the amount of zero padding applied to each feature output
    """
    model_id: int = 0
    model_name: str = ""
    net: Callable | None = None
    outputs: list[str] | list[int] = field(default_factory=list)
    pad_amount: list[int] | int = 0


_NETS = {"alex": NetInfo(model_id=15,
                         model_name="alexnet_imagenet_no_top_v2.pth",
                         net=alexnet,
                         outputs=[f"features.{i}" for i in (0, 3, 6, 8, 10)],
                         pad_amount=[2, 2, 1, 1, 1]),
         "squeeze": NetInfo(model_id=16,
                            model_name="squeezenet_imagenet_no_top_v2.pth",
                            net=squeezenet1_1,
                            outputs=[f"features.{i}" for i in (0, 4, 7, 9, 10, 11, 12)],
                            pad_amount=1),
         "vgg16": NetInfo(model_id=17,
                          model_name="vgg16_imagenet_no_top_v2.pth",
                          net=vgg16,
                          outputs=[f"features.{i}" for i in (2, 7, 14, 21, 29)],
                          pad_amount=1)}

_LINEAR = {"alex": NetInfo(model_id=18,
                           model_name="alexnet_lpips_v2.pth",
                           outputs=[64, 192, 384, 256, 256]),
           "squeeze": NetInfo(model_id=19,
                              model_name="squeezenet_lpips_v2.pth",
                              outputs=[64, 128, 256, 384, 384, 512, 512]),
           "vgg16": NetInfo(model_id=20,
                            model_name="vgg16_lpips_v2.pth",
                            outputs=[64, 128, 256, 512, 512])}


class _LPIPSTrunkNet(nn.Module):
    """Trunk neural network loader for LPIPS Loss function. Loads the trunk network and the
    weights and selects the feature layers for output

    Parameters
    ----------
    net_name
        The name of the trunk network to load. One of "alex", "squeeze" or "vgg16"
    eval_mode
        ``True`` for evaluation mode, ``False`` for training mode
    load_weights
        ``True`` if pretrained trunk network weights should be loaded, otherwise ``False``
    """
    def __init__(self,
                 net_name: T.Literal["alex", "squeeze", "vgg16"],
                 eval_mode: bool,
                 load_weights: bool) -> None:
        logger.debug(parse_class_init(locals()))
        super().__init__()
        self._net_name = net_name
        self._eval_mode = eval_mode
        self._load_weights = load_weights
        self._net_name = net_name
        self.net = self._get_net()
        logger.debug("Initialized: %s ", self.__class__.__name__)

    def __repr__(self) -> str:
        """Pretty print for logging"""
        _repr = super().__repr__()
        params = ", ".join(f"{k[1:]}={repr(v)}" for k, v in self.__dict__.items()
                           if k.startswith(("_net_name", "_eval_mode", "_load_weights")))
        pfx = f"{self.__class__.__name__}("
        return f"{pfx}{params})({_repr[len(pfx):]}"

    def _get_net(self) -> nn.Module:
        """Load the trunk, set the weights and feature outputs

        Returns
        -------
        The loaded trunk network with feature extractor outputs set
        """
        net_info = _NETS[self._net_name]
        model_def = net_info.net
        assert model_def is not None
        net = feature_extraction.create_feature_extractor(model_def(),
                                                          return_nodes=T.cast(list[str],
                                                                              net_info.outputs))
        if self._load_weights:
            weights_path = GetModel(net_info.model_name, net_info.model_id).model_path
            assert isinstance(weights_path, str)
            weights = torch.load(weights_path)
            net.load_state_dict(weights)

        if self._eval_mode:
            net.eval()
        return net

    @classmethod
    def _normalize_output(cls, inputs: torch.Tensor, epsilon: float = 1e-10) -> torch.Tensor:
        """Normalize the output tensors from the trunk network.

        Parameters
        ----------
        inputs: :class:`keras.KerasTensor`
            An output tensor from the trunk model
        epsilon: float, optional
            Epsilon to apply to the normalization operation. Default: `1e-10`
        """
        norm_factor = torch.sqrt(torch.sum(torch.square(inputs), dim=1, keepdim=True))
        return inputs / (norm_factor + epsilon)

    def forward(self, inputs: torch.Tensor) -> list[torch.Tensor]:
        """Obtain the normalized features from the trunk net

        Returns
        -------
        The normalized feature outputs from the trunk net
        """
        outputs = [self._normalize_output(x) for x in self.net(inputs).values()]
        return outputs


class _LPIPSLinearNet(nn.Module):
    """The Linear Network to be applied to the difference between the true and predicted outputs
    of the trunk network.

    Parameters
    ----------
    net_name
        The name of the trunk network in use. One of "alex", "squeeze" or "vgg16"
    eval_mode
        ``True`` for evaluation mode, ``False`` for training mode
    load_weights
        ``True`` if pretrained linear network weights should be loaded, otherwise ``False``
    use_dropout
        ``True`` if a dropout layer should be used in the Linear network otherwise ``False``
    """
    def __init__(self,
                 net_name: T.Literal["alex", "squeeze", "vgg16"],
                 eval_mode: bool,
                 load_weights: bool,
                 use_dropout: bool) -> None:
        logger.debug(parse_class_init(locals()))
        super().__init__()
        self._net_name = net_name
        self._eval_mode = eval_mode
        self._load_weights = load_weights
        self._use_dropout = use_dropout
        self.net = self._get_net()

    def _get_net(self) -> nn.ModuleList:
        """Load the linear network, set the weights

        Returns
        -------
        The Linear network for the given trunk network
        """
        net_info = _LINEAR[self._net_name]
        layers: list[nn.Module] = []
        for in_channels in net_info.outputs:
            assert isinstance(in_channels, int)
            conv = nn.Conv2d(in_channels, 1, 1, stride=1, padding=0, bias=False)
            if self._use_dropout:
                layers.append(nn.Sequential(nn.Dropout(), conv))
            else:
                layers.append(conv)

        net = nn.ModuleList(layers)

        if self._load_weights:
            weights_path = GetModel(net_info.model_name, net_info.model_id).model_path
            assert isinstance(weights_path, str)
            weights = torch.load(weights_path)
            state = net.state_dict()
            assert len(weights) == len(state)
            for key, val in zip(list(state), weights.values()):
                state[key] = val

            net.load_state_dict(state)

        if self._eval_mode:
            net.eval()
        return net

    def forward(self, inputs: list[torch.Tensor]) -> list[torch.Tensor]:
        """Run the linear layers over each trunk network's feature output

        Parameters
        ----------
        inputs
            The feature maps output from the trunk network

        Returns
        -------
        The output of the linear layers applied to the feature map outputs
        """
        return [self.net[i](inp) for i, inp in enumerate(inputs)]


class LPIPSLoss(nn.Module):  # pylint:disable=too-many-instance-attributes
    """LPIPS Loss Function.

    A perceptual loss function that uses linear outputs from pretrained CNNs feature layers.

    Notes
    -----
    Channels Last implementation. All trunks implemented from the original paper.

    References
    ----------
    https://richzhang.github.io/PerceptualSimilarity/

    Parameters
    ----------
    trunk_network
        The name of the trunk network to use. One of "alex", "squeeze" or "vgg16"
    trunk_pretrained
        ``True`` Load the imagenet pretrained weights for the trunk network. ``False`` randomly
        initialize the trunk network. Default: ``True``
    trunk_eval_mode
        ``True`` for running inference on the trunk network (standard mode), ``False`` for training
        the trunk network. Default: ``True``
    linear_pretrained
        ``True`` loads the pretrained weights for the linear network layers. ``False`` randomly
        initializes the layers. Default: ``True``
    linear_eval_mode
        ``True`` for running inference on the linear network (standard mode), ``False`` for
        training the linear network. Default: ``True``
    linear_use_dropout
        ``True`` if a dropout layer should be used in the Linear network otherwise ``False``.
        Default: ``True``
    lpips
        ``True`` to use linear network on top of the trunk network. ``False`` to just average the
        output from the trunk network. Default ``True``
    spatial_output
        ``True`` output the loss in the spatial domain (i.e. as a grayscale tensor of height and
        width of the input image). ``Bool`` reduce the spatial dimensions for loss calculation.
        Default: ``True``
    normalize
        ``True`` if the input Tensor needs to be normalized from the 0. to 1. range to the -1. to
        1. range. Default: ``True``
    ret_per_layer
        ``True`` to return the loss value per feature output layer otherwise ``False``.
        Default: ``False``
    crop
        Crop the zero-padded borders from the feature maps. Can help reduce moire pattern.
        Default: ``False``
    color_order
        The RGB/BGR order of the input images
    """
    _shift: torch.Tensor
    _scale: torch.Tensor

    def __init__(self,  # pylint:disable=too-many-arguments,too-many-positional-arguments
                 trunk_network: T.Literal["alex", "squeeze", "vgg16"],
                 trunk_pretrained: bool = True,
                 trunk_eval_mode: bool = True,
                 linear_pretrained: bool = True,
                 linear_eval_mode: bool = True,
                 linear_use_dropout: bool = True,
                 lpips: bool = True,
                 spatial_output: bool = True,
                 normalize: bool = True,
                 ret_per_layer: bool = False,
                 crop: bool = False,
                 color_order: T.Literal["bgr", "rgb"] = "bgr") -> None:
        super().__init__()
        logger.debug(parse_class_init(locals()))
        self._spatial = spatial_output
        self._use_lpips = lpips
        self._normalize = normalize
        self._ret_per_layer = ret_per_layer
        self._crop_amount = self._get_crop_amount(crop, trunk_network)

        self._is_rgb = color_order == "rgb"

        self.register_buffer("_shift",
                             torch.Tensor([-.030, -.088, -.188]).float()[None, :, None, None])
        self.register_buffer("_scale",
                             torch.Tensor([.458, .448, .450]).float()[None, :, None, None])
        self._trunk_net = _LPIPSTrunkNet(trunk_network, trunk_eval_mode, trunk_pretrained)
        self._linear_net = _LPIPSLinearNet(trunk_network,
                                           linear_eval_mode,
                                           linear_pretrained,
                                           linear_use_dropout)
        if trunk_eval_mode and linear_eval_mode:
            self.eval()

    @classmethod
    def _get_crop_amount(cls,
                         do_crop: bool,
                         trunk_network: T.Literal["alex", "squeeze", "vgg16"]) -> list[int]:
        """Obtain the amount to crop from the side of each feature map output when cropping is
        selected

        Parameters
        ----------
        do_crop
            ``True`` if cropping is enabled otherwise ``False``
        trunk_network
            The truck network to obtain the cropping amount for

        Returns
        -------
        The amount to crop from each side of the feature map outputs. Empty list if no cropping to
        be performed
        """
        if not do_crop:
            retval = []
        else:
            info = _NETS[trunk_network]
            if isinstance(info.pad_amount, list):
                retval = info.pad_amount
            elif not info.pad_amount:
                retval = []
            else:
                retval = [info.pad_amount for _ in range(len(info.outputs))]
        logger.debug("[LPIPSLoss] Crop amounts for '%s' do_crop=%s: %s",
                     trunk_network, do_crop, retval)
        return retval

    def _process_diffs(self, inputs: list[torch.Tensor]) -> list[torch.Tensor]:
        """Perform processing on the Trunk Network outputs.

        If :attr:`use_lpips` is enabled, process the diff values through the linear network,
        otherwise return the diff values summed on the channels axis.

        Parameters
        ----------
        List of the squared difference of the true and predicted outputs from the trunk network

        Returns
        -------
        List of either the linear network outputs (when using lpips) or summed network outputs
        """
        if self._use_lpips:
            return self._linear_net(inputs)
        return [torch.sum(x, dim=1) for x in inputs]

    def _process_output(self, inputs: torch.Tensor, output_dims: tuple) -> torch.Tensor:
        """Process an individual output based on whether :attr:`is_spatial` has been selected.

        When spatial output is selected, all outputs are sized to the shape of the original True
        input Tensor. When not selected, the mean across the spatial axes (h, w) are returned

        Parameters
        ----------
        inputs
            An individual diff output tensor from the linear network or summed output
        output_dims
            The (height, width) of the original true image

        Returns
        -------
        Either the original tensor resized to the true image dimensions, or the mean value across
        the height, width axes.
        """
        if self._spatial:
            return nn.Upsample(output_dims, mode="bilinear", align_corners=False)(inputs)
        return torch.mean(inputs, dim=(2, 3), keepdim=True)

    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor
                ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """Perform the LPIPS Loss Function.

        Parameters
        ----------
        y_true
            The ground truth batch of images
        y_pred
            The predicted batch of images

        Returns
        -------
        The final loss value for each item in the batch
        """
        if not self._is_rgb:
            y_true = torch.flip(y_true, dims=[1])
            y_pred = torch.flip(y_pred, dims=[1])

        if self._normalize:
            y_true = (y_true * 2.0) - 1.0
            y_pred = (y_pred * 2.0) - 1.0

        y_true = (y_true - self._shift) / self._scale
        y_pred = (y_pred - self._shift) / self._scale

        net_true = self._trunk_net(y_true)
        net_pred = self._trunk_net(y_pred)

        diffs = [(out_true - out_pred) ** 2
                 for out_true, out_pred in zip(net_true, net_pred)]

        dims = y_true.shape[2:4]
        if self._crop_amount:
            diffs = [d[:, :, i:-i, i: -i] if i else d
                     for d, i in zip(diffs, self._crop_amount)]

        dims = dims if self._spatial else y_true.shape[2:4]
        res = [self._process_output(diff, dims) for diff in self._process_diffs(diffs)]

        if self._spatial:
            val = torch.stack(res, dim=0).sum(dim=0)
        else:
            val = torch.stack([r.sum(dim=(1, 2, 3)) for r in res]).sum(dim=0)

        val *= 0.1  # Reduce by factor of 10 'cos this loss is STRONG. # TODO config

        retval = (val, res) if self._ret_per_layer else val
        return retval


__all__ = get_module_objects(__name__)
