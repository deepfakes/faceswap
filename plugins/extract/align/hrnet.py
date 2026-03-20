#!/usr/bin/env python3
"""Facial landmarks extractor for faceswap.pnt_y
   Code adapted and modified from:
   https://github.com/1adrianb/face-alignment
"""
from __future__ import annotations
import logging
import typing as T
from dataclasses import dataclass

import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

from lib.utils import get_module_objects, GetModel
from plugins.extract.base import ExtractPlugin
from . import hrnet_defaults as cfg
from .dark_decoder import Dark

if T.TYPE_CHECKING:
    import numpy.typing as npt


logger = logging.getLogger(__name__)
# pylint:disable=duplicate-code


@dataclass
class HRNetStageConfig:
    """Configuration settings for each stage of HRNet"""
    num_modules: int
    num_branches: int
    num_blocks: list[int]
    num_channels: list[int]
    use_bottleneck: bool


class HRNet(ExtractPlugin):
    """HRNet Face alignment"""
    def __init__(self) -> None:
        super().__init__(input_size=256,
                         batch_size=cfg.batch_size(),
                         is_rgb=True,
                         dtype="float32",
                         scale=(0, 1))
        self._stage_2_config = HRNetStageConfig(num_modules=1,
                                                num_branches=2,
                                                num_blocks=[4, 4],
                                                num_channels=[18, 36],
                                                use_bottleneck=False)
        self._stage_3_config = HRNetStageConfig(num_modules=4,
                                                num_branches=3,
                                                num_blocks=[4, 4, 4],
                                                num_channels=[18, 36, 72],
                                                use_bottleneck=False)
        self._stage_4_config = HRNetStageConfig(num_modules=3,
                                                num_branches=4,
                                                num_blocks=[4, 4, 4, 4],
                                                num_channels=[18, 36, 72, 144],
                                                use_bottleneck=False)

        self.model: HighResolutionNet
        self.realign_centering = "legacy"

        self._mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self._std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        self._dark = Dark(68, 64) if cfg.dark_decoder() else None

    def load_model(self) -> HighResolutionNet:
        """Load the HRNet model

        Returns
        -------
        The loaded HRNet model
        """
        weights = GetModel("hrnet_landmark_v1.pth", 34).model_path
        assert isinstance(weights, str)
        model = T.cast(HighResolutionNet, self.load_torch_model(
            HighResolutionNet(num_joints=68,
                              final_conv_kernel=1,
                              stage_2_config=self._stage_2_config,
                              stage_3_config=self._stage_3_config,
                              stage_4_config=self._stage_4_config),
            weights))
        return model

    def pre_process(self, batch: np.ndarray) -> np.ndarray:
        """Format the ROI faces detection boxes for prediction

        Parameters
        ----------
        batch
            The batch of face detection bounding boxes as (bs, l, t, r, b)

        Returns
        -------
        The face detection bounding boxes formatted to take an image patch for prediction
        """
        heights = batch[:, 3] - batch[:, 1]
        widths = batch[:, 2] - batch[:, 0]
        ctr_x = np.rint((batch[:, 0] + batch[:, 2]) * 0.5).astype("int32")
        ctr_y = np.rint((batch[:, 1] + batch[:, 3]) * 0.5).astype("int32")
        size = np.maximum(widths, heights)
        half = np.rint(size * 0.5).astype("int32")
        retval = np.empty((batch.shape[0], 4), dtype=np.int32)
        retval[:, 0] = ctr_x - half
        retval[:, 1] = ctr_y - half
        retval[:, 2] = ctr_x + half
        retval[:, 3] = ctr_y + half
        return retval

    def process(self, batch: np.ndarray) -> np.ndarray:
        """Predict the 68 point landmarks

        Parameters
        ----------
        batch
            The batch to feed into the aligner

        Returns
        -------
        The predictions from the aligner
        """
        batch -= self._mean
        batch /= self._std
        return self.from_torch(batch.transpose(0, 3, 1, 2))

    def _get_predictions(self, scores: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Convert the score maps from the model into predictions

        Parameters
        ----------
        scores
            The score maps from the model

        Returns
        -------
        Predictions from the score maps
        """
        batch_size, num_points, size = scores.shape[:3]
        scores_r = scores.reshape((batch_size, num_points, -1))

        max_val: npt.NDArray[np.float32] = scores_r.max(axis=2)
        idx: npt.NDArray[np.int64] = scores_r.argmax(axis=2)

        retval = np.empty((batch_size, num_points, 2), dtype=np.float32)
        retval[:, :, 0] = (idx - 1) % size + 1
        retval[:, :, 1] = np.floor((idx - 1) / size) + 1

        mask = max_val[..., None] > 0.
        retval *= mask
        return retval

    def post_process(self, batch: np.ndarray) -> np.ndarray:  # pylint:disable=too-many-locals
        """Process the output from the model

        Parameters
        ----------
        batch
            The predictions from the aligner

        Returns
        -------
        The final landmarks in 0-1 space
        """
        if self._dark is not None:
            return self._dark(batch) / 64.
        batch_size, num_points, height, width = batch.shape
        assert height == width, "Heatmaps must be square"
        resolution = height

        coords = self._get_predictions(batch)
        pnt_x = coords[..., 0].astype(np.int32)
        pnt_y = coords[..., 1].astype(np.int32)
        mask = (pnt_x > 1) & (pnt_x < resolution) & (pnt_y > 1) & (pnt_y < resolution)
        pnt_x = np.clip(pnt_x, 2, resolution - 1)
        pnt_y = np.clip(pnt_x, 2, resolution - 1)
        idx_batch = np.arange(batch_size)[:, None]
        idx_pnt = np.arange(num_points)[None, :]
        delta_x = (batch[idx_batch, idx_pnt, pnt_y - 1, pnt_x] -
                   batch[idx_batch, idx_pnt, pnt_y - 1, pnt_x - 2])
        delta_y = (batch[idx_batch, idx_pnt, pnt_y, pnt_x - 1] -
                   batch[idx_batch, idx_pnt, pnt_y - 2, pnt_x - 1])
        diff = np.stack([delta_x, delta_y], axis=-1)
        coords += (np.sign(diff) * 0.25 * mask[..., None]) + 0.5
        coords /= resolution
        return coords


class BasicBlock(nn.Module):
    """ Basic block for HRNet

    Parameters
    ----------
    in_channels
        The number of in channels
    out_channels
        The number of out channels
    stride
        The stride for the first 3x3 conv block. Default: 1
    downsample
        The module to use for downsampling or ``None`` for no downsample. Default: ``None``
    """
    expansion = 1

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 stride: int = 1,
                 downsample: nn.Module | None = None) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels, momentum=0.01)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels, momentum=0.01)
        self.downsample = downsample
        self.stride = stride

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass through HRNet's basic block

        Parameters
        ----------
        inputs
            Input to the conv block

        Returns
        -------
        Output from the conv block
        """
        residual = inputs

        out = self.conv1(inputs)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(inputs)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """ Bottleneck for HRNet

    Parameters
    ----------
    in_channels
        The number of in channels
    out_channels
        The number of out channels
    stride
        The stride for the first 3x3 conv block. Default: 1
    downsample
        The module to use for downsampling or ``None`` for no downsample. Default: ``None``
    """
    expansion = 4

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 stride: int = 1,
                 downsample: nn.Module | None = None) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels, momentum=0.01)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels, momentum=0.01)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion, momentum=0.01)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass through HRNet's basic block

        Parameters
        ----------
        inputs
            Input to the conv block

        Returns
        -------
        Output from the conv block
        """
        residual = inputs

        out = self.conv1(inputs)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(inputs)

        out += residual
        out = self.relu(out)

        return out


class HighResolutionModule(nn.Module):
    """ High Resolution Module for HRNet

    Parameters
    ----------
    num_branches
        The number of branches to use
    blocks
        The block object to use
    num_blocks
        The number of blocks in each branch
    num_in_channels
        The number of input channels in each branch
    num_channels
        The number of channels in each branch
    multi_scale_output
        ``True`` to output multi-scaled
    """
    def __init__(self,
                 num_branches: int,
                 block: type[Bottleneck] | type[BasicBlock],
                 num_blocks: list[int],
                 num_in_channels: list[int],
                 num_channels: list[int],
                 multi_scale_output: bool = True) -> None:
        super().__init__()
        self._check_branches(num_branches,
                             num_blocks,
                             num_in_channels,
                             num_channels)

        self.num_in_channels = num_in_channels
        self.num_branches = num_branches
        self.multi_scale_output = multi_scale_output
        self.branches = nn.ModuleList(self._make_one_branch(i,
                                                            block,
                                                            num_blocks[i],
                                                            num_channels[i])
                                      for i in range(num_branches))

        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(inplace=True)

    def _check_branches(self,
                        num_branches: int,
                        num_blocks: list[int],
                        num_in_channels: list[int],
                        num_channels: list[int]) -> None:
        """Check that the branch configuration is valid

        Parameters
        ----------
        num_branches
            The number of branches to use
        num_blocks
            The number of blocks in each branch
        num_in_channels
            The number of input channels in each branch
        num_channels
            The number of channels in each branch

        Raises
        ------
        ValueError
            On an invalid configuration
        """
        if num_branches != len(num_blocks):
            raise ValueError(f"NUM_BRANCHES({num_branches}) <> NUM_BLOCKS({len(num_blocks)})")
        if num_branches != len(num_channels):
            raise ValueError(f"NUM_BRANCHES({num_branches}) <> NUM_CHANNELS({len(num_channels)})")
        if num_branches != len(num_in_channels):
            raise ValueError(f"NUM_BRANCHES({num_branches}) <> "
                             f"NUM_IN_CHANNELS({len(num_in_channels)})")

    def _make_one_branch(self,
                         branch_index: int,
                         block: type[Bottleneck] | type[BasicBlock],
                         num_blocks: int,
                         num_channels: int,
                         stride: int = 1) -> nn.Sequential:
        """ Make a single branch

        Parameters
        ----------
        branch_index
            The index of the branch to make
        block
            The block object to use
        num_blocks
            The number of blocks in each branch
        num_in_channels
            The number of input channels in each branch
        num_channels
            The number of channels in each branch
        multi_scale_output
            ``True`` to output multi-scaled

        Returns
        -------
        The sequential modules for the branch
        """
        downsample = None
        if stride != 1 or self.num_in_channels[branch_index] != num_channels * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.num_in_channels[branch_index],
                                                 num_channels * block.expansion,
                                                 1,
                                                 stride=stride,
                                                 bias=False),
                                       nn.BatchNorm2d(num_channels * block.expansion,
                                                      momentum=0.01))

        layers = []
        layers.append(block(self.num_in_channels[branch_index],
                            num_channels,
                            stride,
                            downsample=downsample))
        self.num_in_channels[branch_index] = num_channels * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.num_in_channels[branch_index], num_channels))

        return nn.Sequential(*layers)

    def _make_fuse_layers(self) -> nn.ModuleList | None:
        """Make the fuse layers for the HR Module

        Returns
        -------
        The fuse layers module list or ``None`` if layers are not to be fused
        """
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_in_channels = self.num_in_channels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(nn.Sequential(nn.Conv2d(num_in_channels[j],
                                                              num_in_channels[i],
                                                              1,
                                                              stride=1,
                                                              padding=0,
                                                              bias=False),
                                                    nn.BatchNorm2d(num_in_channels[i],
                                                                   momentum=0.01)))
                elif j == i:
                    fuse_layer.append(None)  # type:ignore[arg-type]
                else:
                    conv3x3s = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            num_out_channels_conv3x3 = num_in_channels[i]
                            conv3x3s.append(nn.Sequential(nn.Conv2d(num_in_channels[j],
                                                                    num_out_channels_conv3x3,
                                                                    3,
                                                                    stride=2,
                                                                    padding=1,
                                                                    bias=False),
                                                          nn.BatchNorm2d(num_out_channels_conv3x3,
                                                                         momentum=0.01)))
                        else:
                            num_out_channels_conv3x3 = num_in_channels[j]
                            conv3x3s.append(nn.Sequential(nn.Conv2d(num_in_channels[j],
                                                                    num_out_channels_conv3x3,
                                                                    3,
                                                                    stride=2,
                                                                    padding=1,
                                                                    bias=False),
                                                          nn.BatchNorm2d(num_out_channels_conv3x3,
                                                                         momentum=0.01),
                                                          nn.ReLU(inplace=True)))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_in_channels(self) -> list[int]:
        """Obtain the number of input channels to the module

        Returns
        -------
        The number of input channels to the module
        """
        return self.num_in_channels

    def forward(self, inputs: list[torch.Tensor]) -> list[torch.Tensor]:
        """Forward pass through the HR Module

        Parameters
        ----------
        inputs
            Input to the HR Module

        Returns
        -------
        Output from the HR Module
        """
        x = inputs
        if self.num_branches == 1:
            return [self.branches[0](x[0])]
        assert self.fuse_layers is not None

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []
        for i, fuse_layer in enumerate(T.cast(list[nn.ModuleList], self.fuse_layers)):
            y = x[0] if i == 0 else fuse_layer[0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                elif j > i:
                    y = y + F.interpolate(fuse_layer[j](x[j]),
                                          size=[x[i].shape[2], x[i].shape[3]],
                                          mode="bilinear")
                else:
                    y = y + fuse_layer[j](x[j])
            x_fuse.append(self.relu(y))
        return x_fuse


class HighResolutionNet(nn.Module):  # pylint:disable=too-many-instance-attributes
    """The HRNet Landmark Detection model

    Parameters
    ----------
    num_joints
        The number of joints in the model
    final_conv_kernel
        Kernel size of the final convolution
    stage_2_config
        Configuration settings for stage 2 layers
    stage_3_config
        Configuration settings for stage 3 layers
    stage_4_config
        Configuration settings for stage 4 layers
    """
    def __init__(self,
                 num_joints: int,
                 final_conv_kernel: int,
                 stage_2_config: HRNetStageConfig,
                 stage_3_config: HRNetStageConfig,
                 stage_4_config: HRNetStageConfig) -> None:
        self.in_channels = 64
        super().__init__()
        self.stage_2_config = stage_2_config
        self.stage_3_config = stage_3_config
        self.stage_4_config = stage_4_config

        # stem net
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=0.01)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=0.01)
        self.relu = nn.ReLU(inplace=True)
        self.sf = nn.Softmax(dim=1)
        self.layer1 = self._make_layer(Bottleneck, 64, 64, 4)

        num_channels = stage_2_config.num_channels
        block = Bottleneck if stage_2_config.use_bottleneck else BasicBlock
        num_channels = [num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition1 = self._make_transition_layer([256], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(stage_2_config, num_channels)

        num_channels = stage_3_config.num_channels
        block = Bottleneck if stage_3_config.use_bottleneck else BasicBlock
        num_channels = [num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition2 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(stage_3_config, num_channels)

        num_channels = stage_4_config.num_channels
        block = Bottleneck if stage_4_config.use_bottleneck else BasicBlock
        num_channels = [num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition3 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(stage_4_config,
                                                           num_channels,
                                                           multi_scale_output=True)

        final_inp_channels = sum(pre_stage_channels)

        self.head = nn.Sequential(nn.Conv2d(final_inp_channels,
                                            final_inp_channels,
                                            1,
                                            stride=1,
                                            padding=1 if final_conv_kernel == 3 else 0),
                                  nn.BatchNorm2d(final_inp_channels, momentum=0.01),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(final_inp_channels,
                                            num_joints,
                                            final_conv_kernel,
                                            stride=1,
                                            padding=1 if final_conv_kernel == 3 else 0))

    def _make_transition_layer(self,
                               num_channels_pre_layer: list[int],
                               num_channels_cur_layer: list[int]) -> nn.ModuleList:
        """Make an HRNet transition layer

        Parameters
        ----------
        num_channels_pre_layer
            The number of channels from the previous layer
        num_channels_cur_layer
            The number of channels from the current layer

        Returns
        -------
        The transition layer module list
        """
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(
                        nn.Conv2d(num_channels_pre_layer[i],
                                  num_channels_cur_layer[i],
                                  3,
                                  stride=1,
                                  padding=1,
                                  bias=False),
                        nn.BatchNorm2d(num_channels_cur_layer[i], momentum=0.01),
                        nn.ReLU(inplace=True)))
                else:
                    transition_layers.append(None)  # type:ignore[arg-type]
                continue
            conv3x3s = []
            for j in range(i + 1 - num_branches_pre):
                in_channels = num_channels_pre_layer[-1]
                out_channels = (num_channels_cur_layer[i] if j == i - num_branches_pre
                                else in_channels)
                conv3x3s.append(nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(out_channels, momentum=0.01),
                    nn.ReLU(inplace=True)))
            transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_layer(self,
                    block: type[BasicBlock] | type[Bottleneck],
                    in_channels: int,
                    out_channels: int,
                    blocks: int,
                    stride: int = 1) -> nn.Sequential:
        """Make an HRNet layer

        Parameters
        ----------
        block
            The type of block to use for the layer
        in_channels
            The number of input channels
        out_channels
            The number of output channels
        blocks
            The number of blocks
        stride
            The stride size. Default: 1

        Returns
        -------
        The sequential layer
        """
        downsample = None
        if stride != 1 or in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(in_channels,
                                                 out_channels * block.expansion,
                                                 1,
                                                 stride=stride,
                                                 bias=False),
                                       nn.BatchNorm2d(out_channels * block.expansion,
                                                      momentum=0.01))

        layers = []
        layers.append(block(in_channels, out_channels, stride, downsample))
        in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(in_channels, out_channels))

        return nn.Sequential(*layers)

    def _make_stage(self,
                    layer_config: HRNetStageConfig,
                    num_in_channels: list[int],
                    multi_scale_output: bool = True) -> tuple[nn.Sequential, list[int]]:
        """Make a stage for HRNet

        Parameters
        ----------
        layer_config
            The configuration for the stage
        num_in_channels
            The input channels for the stage
        multi_scale_output
            ``True`` to output multi scale

        Returns
        -------
        sequential
            The stage Sequential Modules
        num_in_channels
            The number of input channels from the final module
        """
        num_modules = layer_config.num_modules
        num_branches = layer_config.num_branches
        num_blocks = layer_config.num_blocks
        num_channels = layer_config.num_channels
        block = Bottleneck if layer_config.use_bottleneck else BasicBlock

        modules: list[HighResolutionModule] = []
        for i in range(num_modules):
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True
            modules.append(HighResolutionModule(num_branches,
                                                block,
                                                num_blocks,
                                                num_in_channels,
                                                num_channels,
                                                reset_multi_scale_output))
            num_in_channels = modules[-1].get_num_in_channels()

        return nn.Sequential(*modules), num_in_channels

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass through HRNet

        Parameters
        ----------
        inputs
            Input to HRNet

        Returns
        -------
        Output from HRNet
        """
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)

        x_list = [self.transition1[i](x) if self.transition1[i] is not None else x
                  for i in range(self.stage_2_config.num_branches)]
        y_list = self.stage2(x_list)

        x_list = [self.transition2[i](y_list[-1]) if self.transition2[i] is not None
                  else y_list[i]
                  for i in range(self.stage_3_config.num_branches)]
        y_list = self.stage3(x_list)

        x_list = [self.transition3[i](y_list[-1]) if self.transition3[i] is not None
                  else y_list[i]
                  for i in range(self.stage_4_config.num_branches)]
        x = self.stage4(x_list)

        # Head Part
        height, width = x[0].size(2), x[0].size(3)
        x1 = F.interpolate(x[1], size=(height, width), mode="bilinear", align_corners=False)
        x2 = F.interpolate(x[2], size=(height, width), mode="bilinear", align_corners=False)
        x3 = F.interpolate(x[3], size=(height, width), mode="bilinear", align_corners=False)
        x = torch.cat([x[0], x1, x2, x3], 1)
        x = self.head(x)

        return x


__all__ = get_module_objects(__name__)
