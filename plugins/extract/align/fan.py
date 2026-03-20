#!/usr/bin/env python3
"""Facial landmarks extractor for faceswap.py
   Code adapted and modified from:
   https://github.com/1adrianb/face-alignment
"""
from __future__ import annotations
import logging
import typing as T

import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

from lib.utils import get_module_objects, GetModel
from plugins.extract.base import ExtractPlugin

from . import fan_defaults as cfg
from . dark_decoder import Dark


logger = logging.getLogger(__name__)


class FAN(ExtractPlugin):
    """FAN Face alignment"""
    def __init__(self) -> None:
        super().__init__(input_size=256,
                         batch_size=cfg.batch_size(),
                         is_rgb=True,
                         dtype="float32",
                         scale=(0, 1))
        self.model: FaceAlignmentNetwork
        self.realign_centering = "head"
        # Original reference scale leads to some fairly unsatisfying landmarks so tightened up
        # self._reference_scale = 200. / 195.
        self._reference_scale = 0.8
        self._dark = Dark(68, 64) if cfg.dark_decoder() else None

    def load_model(self) -> FaceAlignmentNetwork:
        """Load the FAN model

        Returns
        -------
        The loaded FAN model
        """
        weights = GetModel("face-alignment-network_2d4_v4.pth", 13).model_path
        assert isinstance(weights, str)
        model = T.cast(FaceAlignmentNetwork,
                       self.load_torch_model(FaceAlignmentNetwork(num_stack=4,
                                                                  num_modules=1,
                                                                  hg_depth=4,
                                                                  num_features=256,
                                                                  num_classes=68),
                                             weights,
                                             return_indices=[-1]))
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
        # This y-shift only really makes sense for derived bounding boxes, so removed:
        # ctr_y = np.rint((batch[:, 1] + batch[:, 3]) * 0.5 - heights * 0.12).astype("int32")
        ctr_y = np.rint((batch[:, 1] + batch[:, 3]) * 0.5).astype("int32")
        size = (widths + heights) * self._reference_scale
        half = np.rint(size * 0.5).astype("int32")
        # Original implementation is (1, 1) top left, not (0, 0)
        tl_offset = np.rint(size / self.input_size).astype("int32")

        retval = np.empty((batch.shape[0], 4), dtype=np.int32)
        retval[:, 0] = ctr_x - half + tl_offset
        retval[:, 1] = ctr_y - half + tl_offset
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
        return self.from_torch(batch.transpose(0, 3, 1, 2))

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
        num_images, num_landmarks, height, width = batch.shape
        assert height == width, "Heatmaps must be square"
        resolution = height

        image_slice = np.arange(num_images)[:, None]
        landmark_slice = np.arange(num_landmarks)[None, :]

        subpixel_landmarks = np.ones((num_images, num_landmarks, 2), dtype='float32')

        indices = np.array(np.unravel_index(batch.reshape(num_images,
                                                          num_landmarks,
                                                          -1).argmax(-1),
                                            (batch.shape[2],  # height
                                             batch.shape[3])))  # width
        min_clipped = np.minimum(indices + 1, batch.shape[2] - 1)
        max_clipped = np.maximum(indices - 1, 0)

        offsets = [(image_slice, landmark_slice, indices[0], min_clipped[1]),  # Right
                   (image_slice, landmark_slice, indices[0], max_clipped[1]),  # Left
                   (image_slice, landmark_slice, min_clipped[0], indices[1]),  # Down
                   (image_slice, landmark_slice, max_clipped[0], indices[1])]  # Up
        right, left = batch[offsets[0]], batch[offsets[1]]
        down, up = batch[offsets[2]], batch[offsets[3]]
        epsilon = 1e-6  # Small epsilon to avoid zero div
        x_delta = np.clip((right - left) / (right + left + epsilon), -0.5, 0.5)
        y_delta = np.clip((down - up) / (down + up + epsilon), -0.5, 0.5)

        subpixel_landmarks[..., 0] = indices[1] + x_delta + 0.5
        subpixel_landmarks[..., 1] = indices[0] + y_delta + 0.5
        subpixel_landmarks /= resolution
        return subpixel_landmarks


class ConvBlock(nn.Module):
    """Convolution block for FAN

    Parameters
    ----------
    num_in
        The number of in channels
    num_out
        The number of out channels
    """
    def __init__(self, num_in: int, num_out: int) -> None:
        super().__init__()
        self.bn1 = nn.BatchNorm2d(num_in)
        self.conv1 = nn.Conv2d(num_in, num_out // 2, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_out // 2)
        self.conv2 = nn.Conv2d(num_out // 2, num_out // 4, 3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(num_out // 4)
        self.conv3 = nn.Conv2d(num_out // 4, num_out // 4, 3, stride=1, padding=1, bias=False)
        self.downsample = None
        if num_in != num_out:
            self.downsample = nn.Sequential(nn.BatchNorm2d(num_in),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(num_in, num_out, 1, stride=1, bias=False))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass through FAN's conv block

        Parameters
        ----------
        inputs
            Input to the conv block

        Returns
        -------
        Output from the conv block
        """
        residual = inputs if self.downsample is None else self.downsample(inputs)
        var_x = self.conv1(F.relu(self.bn1(inputs), inplace=True))
        var_y = self.conv2(F.relu(self.bn2(var_x), inplace=True))
        var_z = self.conv3(F.relu(self.bn3(var_y), inplace=True))
        out = torch.cat((var_x, var_y, var_z), dim=1) + residual
        return out


class HourGlass(nn.Module):
    """Hour-glass module for FAN

    Parameters
    ----------
    num_modules
        The number of modules in the hour-glass network
    depth
        The depth of the hour-glass network
    num_features
        The number of features to generate
    """
    def __init__(self, num_modules: int, depth: int, num_features: int) -> None:
        super().__init__()
        self._num_modules = num_modules
        self._num_features = num_features
        self._depth = depth
        self._generate_network(depth)

    def _generate_network(self, level: int) -> None:
        """Recursively generate the hour-glass network

        Parameters
        ----------
        level
            The depth of the hour-glass network
        """
        for i in range(self._num_modules):
            self.add_module(f"b1_{level}_{i}", ConvBlock(self._num_features, self._num_features))
        for i in range(self._num_modules):
            self.add_module(f"b2_{level}_{i}", ConvBlock(self._num_features, self._num_features))

        if level > 1:
            self._generate_network(level - 1)
        else:
            for i in range(self._num_modules):
                self.add_module(f"b2_plus_{level}_{i}",
                                ConvBlock(self._num_features, self._num_features))

        for i in range(self._num_modules):
            self.add_module(f"b3_{level}_{i}", ConvBlock(self._num_features, self._num_features))

    def _forward(self, level: int, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass through FAN's hour-glass network

        Parameters
        ----------
        inputs
            Input to the hour-glass network

        Returns
        -------
        Output from the hour-glass network
        """
        up1 = inputs
        for i in range(self._num_modules):
            up1 = getattr(self, f"b1_{level}_{i}")(up1)

        lo1 = F.avg_pool2d(inputs, 2, stride=2)  # pylint:disable=not-callable
        for i in range(self._num_modules):
            lo1 = getattr(self, f"b2_{level}_{i}")(lo1)

        if level > 1:
            lo2 = self._forward(level - 1, lo1)
        else:
            lo2 = lo1
            for i in range(self._num_modules):
                lo2 = getattr(self, f"b2_plus_{level}_{i}")(lo2)

        lo3 = lo2
        for i in range(self._num_modules):
            lo3 = getattr(self, f"b3_{level}_{i}")(lo3)

        up2 = F.interpolate(lo3, scale_factor=2, mode="nearest")
        return up1 + up2

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass through FAN's hour-glass network

        Parameters
        ----------
        inputs
            Input to the hour-glass network

        Returns
        -------
        Output from the hour-glass network
        """
        return self._forward(self._depth, inputs)


class FaceAlignmentNetwork(nn.Module):
    """2D FAN alignment for faceswap"""
    def __init__(self,
                 num_stack: int = 4,
                 num_modules: int = 1,
                 hg_depth: int = 4,
                 num_features: int = 256,
                 num_classes: int = 68) -> None:
        super().__init__()
        self._num_stacks = num_stack
        self._num_modules = num_modules

        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = ConvBlock(64, 128)
        self.conv3 = ConvBlock(128, 128)
        self.conv4 = ConvBlock(128, num_features)

        for i in range(self._num_stacks):
            self.add_module(f"m{i}", HourGlass(num_modules, hg_depth, num_features))
            for j in range(num_modules):
                # backwards labelled in original impl:
                self.add_module(f"top_m{j}_{i}", ConvBlock(num_features, num_features))
            self.add_module(f"conv_last{i}", nn.Conv2d(num_features, num_features, 1))
            self.add_module(f"bn_end{i}", nn.BatchNorm2d(num_features))
            self.add_module(f"l{i}", nn.Conv2d(num_features, num_classes, 1))
            if i == self._num_stacks - 1:
                continue
            self.add_module(f"bl{i}", nn.Conv2d(num_features, num_features, 1))
            self.add_module(f"al{i}", nn.Conv2d(num_classes, num_features, 1))

    def forward(self, inputs: torch.Tensor) -> list[torch.Tensor]:
        """Forward pass through FAN face alignment

        Parameters
        ----------
        inputs
            Input to FAN

        Returns
        -------
        Output from FAN
        """
        var_x = F.relu(self.bn1(self.conv1(inputs)), inplace=True)
        var_x = F.avg_pool2d(self.conv2(var_x), 2, stride=2)  # pylint:disable=not-callable
        var_x = self.conv4(self.conv3(var_x))

        out = []
        inter = var_x

        for i in range(self._num_stacks):
            hg = getattr(self, f"m{i}")(inter)
            ll = hg
            for j in range(self._num_modules):
                ll = getattr(self, f"top_m{j}_{i}")(ll)

            ll = F.relu(getattr(self, f"bn_end{i}")(getattr(self, f"conv_last{i}")(ll)),
                        inplace=True)

            out.append(getattr(self, f"l{i}")(ll))

            if i == self._num_stacks - 1:
                continue
            ll = getattr(self, f"bl{i}")(ll)
            inter = inter + ll + getattr(self, f"al{i}")(out[-1])

        return out


__all__ = get_module_objects(__name__)
