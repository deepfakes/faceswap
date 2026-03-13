#! /usr/env/bin/python3
"""Retina face detector adapted from: https://github.com/biubug6/Pytorch_Retinaface

MIT License

Copyright (c) 2019 Sefik Ilkin Serengil

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
from __future__ import annotations

import typing as T
from itertools import product
from math import ceil

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import torchvision.models as tv_models
import torchvision.models._utils as tv_utils

from lib.utils import get_module_objects, GetModel
from plugins.extract.base import ExtractPlugin

from . import retinaface_defaults as cfg

if T.TYPE_CHECKING:
    import numpy.typing as npt
# pylint:disable=duplicate-code


class RetinaFace(ExtractPlugin):
    """RetinaFace detector for face detection"""
    def __init__(self) -> None:
        super().__init__(input_size=640,
                         batch_size=cfg.batch_size(),
                         is_rgb=True,
                         dtype="float32",
                         scale=(0, 255),
                         force_cpu=cfg.cpu())
        self.model: RetinaFaceModel
        self._average_img = np.array([[104.0, 117.0, 123.0]], dtype="float32")
        self._confidence = cfg.confidence() / 100
        self._variance = [0.1, 0.2]
        self._priors = self._generate_priors()
        self._keep_top_k = 750
        self._nms_threshold = 0.4

    def _generate_priors(self, clip: bool = False  # pylint:disable=too-many-locals
                         ) -> npt.NDArray[np.float32]:
        """Generate the anchor boxes for the image size

        Parameters
        ----------
        clip
            ``True`` to clip the output to 0-1. Default: ``False``

        Returns
        -------
        The pre-computed priors in center-offset form shape: (1, num_priors, 4)
        """
        steps = [8, 16, 32]
        min_sizes = [[16, 32], [64, 128], [256, 512]]
        feature_maps = [[ceil(self.input_size / step), ceil(self.input_size / step)]
                        for step in steps]
        anchors = []

        for sizes, feats, step in zip(min_sizes, feature_maps, steps):
            for i, j in product(range(feats[0]), range(feats[1])):
                for min_size in sizes:
                    s_kx = min_size / self.input_size
                    s_ky = min_size / self.input_size
                    dense_cx = [x * step / self.input_size for x in [j + 0.5]]
                    dense_cy = [y * step / self.input_size for y in [i + 0.5]]
                    for cy, cx in product(dense_cy, dense_cx):
                        anchors += [cx, cy, s_kx, s_ky]

        output = np.array(anchors, dtype="float32").reshape(-1, 4)
        if clip:
            output.clip(0, 1)
        return output[None]

    def load_model(self) -> RetinaFaceModel:
        """Initialize RetinaFace Model

        Returns
        -------
        The loaded RetinaFace model
        """
        backbone = T.cast(T.Literal["resnet", "mobilenet"], cfg.backbone())
        assert backbone in ("resnet", "mobilenet")
        vers = 1 if backbone == "resnet" else 2
        weights = GetModel(f"retinaface_v{vers}.pth", 32).model_path
        assert isinstance(weights, str)
        return T.cast(RetinaFaceModel, self.load_torch_model(RetinaFaceModel(backbone), weights))

    def pre_process(self, batch: np.ndarray) -> np.ndarray:
        """Compile the detection image(s) for prediction

        Parameters
        ----------
        batch
            The input batch of images at model input size in the correct color order, dtype and
            scale

        Returns
        -------
        The batch of images ready for feeding the model
        """
        return (batch - self._average_img).transpose(0, 3, 1, 2)

    def process(self, batch: np.ndarray) -> np.ndarray:
        """Run model to get predictions

        Parameters
        ----------
        batch
            A batch of images ready to feed the model

        Returns
        -------
        The batch of detection results from the model
        """
        return self.from_torch(batch)

    def _decode(self, locations: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Decode locations from the model using priors to undo the encoding we did for offset
        regression at train time.

        Parameters
        ----------
        locations
            batch of location predictions from the model filtered by score, Shape:
            [batch_size, filtered_num_priors, 4]
        priors
            The pre-computed priors filtered by score, Shape: [1, filtered_num_priors, 4]

        Returns
        -------
        Decoded bounding box predictions
        """
        boxes = np.concatenate([
            self._priors[..., :2] + locations[..., :2] * self._variance[0] * self._priors[..., 2:],
            self._priors[..., 2:] * np.exp(locations[..., 2:] * self._variance[1])], axis=2)
        boxes[..., :2] -= boxes[..., 2:] / 2
        boxes[..., 2:] += boxes[..., :2]
        return T.cast("npt.NDArray[np.float32]", boxes)

    def _nms(self, boxes: npt.NDArray[np.float32]  # pylint:disable=too-many-locals
             ) -> npt.NDArray[np.float32]:
        """Perform Non-Maximum Suppression

        Parameters
        ----------
        boxes
            The detection bounding boxes to process

        Returns
        -------
        The final bounding boxes
        """
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        scores = boxes[:, 4]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            order = order[1:][ovr <= self._nms_threshold]

        return boxes[keep]

    def post_process(self, batch: np.ndarray) -> np.ndarray:
        """Process the output from the model to bounding boxes

        Parameters
        ----------
        batch
            The output predictions from the S3FD model

        Returns
        -------
        The processed detection bounding box from the model at model input size
        """
        locs, confidence = batch
        batch_boxes = self._decode(locs) * self.input_size
        batch_scores = T.cast("npt.NDArray[np.float32]", confidence[:, :, 1])
        batch_mask = batch_scores > self._confidence
        final_boxes = []
        for boxes, scores, mask in zip(batch_boxes, batch_scores, batch_mask):
            scores = scores[mask]
            if scores.size == 0:
                final_boxes.append(np.empty((0, 5), dtype="float32"))
                continue

            boxes = boxes[mask]
            order = np.argsort(scores)[::-1][:self._keep_top_k]
            detections = np.hstack([boxes[order], scores[order][:, None]])
            final_boxes.append(self._nms(detections)[..., :4])

        retval = np.empty(len(final_boxes), dtype=object)
        retval[:] = final_boxes
        return retval


def conv_bn(in_channels: int,
            out_channels: int,
            kernel: int = 3,
            stride: int = 1,
            padding: int = 1,
            use_relu: bool = False,
            leaky: float = 0.0
            ) -> torch.nn.Sequential:
    """Generates a Conv Batch Norm sequential module for RetinaFace

    Parameters
    ----------
    in_channels
        The number of input channels
    out_channels
        The number of output channels
    kernel
        The kernel size. Default: 3
    stride
        The number of strides. Default: 1
    padding
        The padding to apply. Default: 1
    use_relu
        ``True`` to use LeakyReLU activation
    leaky
        The negative float value for the LeakyReLU. Default: 0.0

    Returns
    -------
    The built sequential module
    """
    layers = [nn.Conv2d(in_channels, out_channels, kernel, stride, padding, bias=False),
              nn.BatchNorm2d(out_channels)]
    if use_relu:
        layers.append(nn.LeakyReLU(negative_slope=leaky, inplace=True))
    return nn.Sequential(*layers)


def conv_dw(in_channels: int, out_channels: int, stride: int, leaky=0.1) -> torch.nn.Sequential:
    """Generates a double Conv Batch Norm sequential module for RetinaFace

    Parameters
    ----------
    in_channels
        The number of input channels
    out_channels
        The number of output channels
    stride
        The number of strides. Default: 1
    leaky
        The negative float value for the LeakyReLU. Default: 0.0

    Returns
    -------
    The built sequential module
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, in_channels, 3, stride, 1, groups=in_channels, bias=False),
        nn.BatchNorm2d(in_channels),
        nn.LeakyReLU(negative_slope=leaky, inplace=True),
        nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(negative_slope=leaky, inplace=True),
    )


class MobileNetV1(nn.Module):
    """MobileNet V1 for use with RetinaFace"""
    def __init__(self) -> None:
        super().__init__()
        self.stage1 = nn.Sequential(
            conv_bn(3, 8, kernel=3, stride=2, use_relu=True, leaky=0.1),
            conv_dw(8, 16, 1),
            conv_dw(16, 32, 2),
            conv_dw(32, 32, 1),
            conv_dw(32, 64, 2),
            conv_dw(64, 64, 1),
        )
        self.stage2 = nn.Sequential(
            conv_dw(64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 128, 1),
            conv_dw(128, 128, 1),
            conv_dw(128, 128, 1),
            conv_dw(128, 128, 1),
        )
        self.stage3 = nn.Sequential(
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
        )
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, 1000)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass through MobileNetV1

        Parameters
        ----------
        inputs
            The input to MobileNetV1

        Returns
        -------
        The output from MobileNetV1
        """
        x = self.stage1(inputs)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.avg(x)
        x = x.view(-1, 256)
        return self.fc(x)


class SSH(nn.Module):
    """SSH Module for RetinaFace

    Parameters
    ----------
    in_channels
        The number of input channels
    out_channels
        The number of output channels
    """
    def __init__(self, in_channels: int, out_channel: int) -> None:
        super().__init__()
        assert out_channel % 4 == 0
        leaky = 0.0
        if out_channel <= 64:
            leaky = 0.1
        self.conv3X3 = conv_bn(in_channels,  # pylint:disable=invalid-name
                               out_channel // 2,
                               stride=1,
                               use_relu=False)
        self.conv5X5_1 = conv_bn(in_channels,  # pylint:disable=invalid-name
                                 out_channel // 4,
                                 stride=1,
                                 use_relu=True,
                                 leaky=leaky)
        self.conv5X5_2 = conv_bn(out_channel // 4,  # pylint:disable=invalid-name
                                 out_channel // 4,
                                 stride=1,
                                 use_relu=False)
        self.conv7X7_2 = conv_bn(out_channel // 4,  # pylint:disable=invalid-name
                                 out_channel // 4,
                                 stride=1,
                                 use_relu=True,
                                 leaky=leaky)
        self.conv7x7_3 = conv_bn(out_channel // 4, out_channel // 4, stride=1, use_relu=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass through SSH Module

        Parameters
        ----------
        inputs
            The input to the SSH Module

        Returns
        -------
        The output from SSH Module
        """
        conv3x3 = self.conv3X3(inputs)

        conv5x5_1 = self.conv5X5_1(inputs)
        conv5x5 = self.conv5X5_2(conv5x5_1)

        conv7x7_2 = self.conv7X7_2(conv5x5_1)
        conv7x7 = self.conv7x7_3(conv7x7_2)

        out = torch.cat([conv3x3, conv5x5, conv7x7], dim=1)
        return F.relu(out)


class FPN(nn.Module):
    """FPN Module for RetinaFace

    Parameters
    ----------
    in_channels_list
        The number of input channels
    out_channels
        The number of output channels
    """
    def __init__(self, in_channels_list: list[int], out_channels: int) -> None:
        super().__init__()
        leaky = 0.0
        if out_channels <= 64:
            leaky = 0.1
        self.output1 = conv_bn(in_channels_list[0],
                               out_channels,
                               kernel=1,
                               stride=1,
                               padding=0,
                               use_relu=True,
                               leaky=leaky)
        self.output2 = conv_bn(in_channels_list[1],
                               out_channels,
                               kernel=1,
                               stride=1,
                               padding=0,
                               use_relu=True,
                               leaky=leaky)
        self.output3 = conv_bn(in_channels_list[2],
                               out_channels,
                               kernel=1,
                               stride=1,
                               padding=0,
                               use_relu=True,
                               leaky=leaky)

        self.merge1 = conv_bn(out_channels, out_channels, use_relu=True, leaky=leaky)
        self.merge2 = conv_bn(out_channels, out_channels, use_relu=True, leaky=leaky)

    def forward(self, inputs: torch.Tensor) -> list[torch.Tensor]:
        """Forward pass through FPN Module

        Parameters
        ----------
        inputs
            The input to the FPN Module

        Returns
        -------
        The output from FPN Module
        """
        l_inputs = list(inputs.values())

        output1 = self.output1(l_inputs[0])
        output2 = self.output2(l_inputs[1])
        output3 = self.output3(l_inputs[2])

        up3 = F.interpolate(output3, size=[output2.size(2), output2.size(3)], mode="nearest")
        output2 = output2 + up3
        output2 = self.merge2(output2)

        up2 = F.interpolate(output2, size=[output1.size(2), output1.size(3)], mode="nearest")
        output1 = output1 + up2
        output1 = self.merge1(output1)
        return [output1, output2, output3]


class ClassHead(nn.Module):
    """Class Head Module for RetinaFace

    Parameters
    ----------
    in_channels
        The number of input channels. Default: 512
    num_anchors
        The number of anchors. Default: 3
    """
    def __init__(self, in_channels: int = 512, num_anchors: int = 3) -> None:
        super().__init__()
        self.num_anchors = num_anchors
        self.conv1x1 = nn.Conv2d(in_channels,
                                 self.num_anchors * 2,
                                 kernel_size=(1, 1),
                                 stride=1,
                                 padding=0)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass through ClassHead Module

        Parameters
        ----------
        inputs
            The input to the ClassHead Module

        Returns
        -------
        The output from ClassHead Module
        """
        x = self.conv1x1(inputs)
        x = x.permute(0, 2, 3, 1).contiguous()
        return x.view(x.shape[0], -1, 2)


class BboxHead(nn.Module):
    """Bounding Box Head Module for RetinaFace

    Parameters
    ----------
    in_channels
        The number of input channels. Default: 512
    num_anchors
        The number of anchors. Default: 3
    """
    def __init__(self, in_channels: int = 512, num_anchors: int = 3) -> None:
        super().__init__()
        self.conv1x1 = nn.Conv2d(in_channels,
                                 num_anchors * 4,
                                 kernel_size=(1, 1),
                                 stride=1,
                                 padding=0)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass through BboxHead Module

        Parameters
        ----------
        inputs
            The input to the BboxHead Module

        Returns
        -------
        The output from BboxHead Module
        """
        x = self.conv1x1(inputs)
        x = x.permute(0, 2, 3, 1).contiguous()
        return x.view(x.shape[0], -1, 4)


class RetinaFaceModel(nn.Module):
    """RetinaFace Model

    Parameters
    ----------
    backbone
        The backbone to use for RetinaFace
    """
    def __init__(self, backbone: T.Literal["mobilenet", "resnet"]) -> None:
        super().__init__()
        b_bone_cfg = {"mobilenet": {"in_channels": 32,
                                    "out_channel": 64,
                                    "return_layers": {'stage1': 1, 'stage2': 2, 'stage3': 3}},
                      "resnet": {"in_channels": 256,
                                 "out_channel": 256,
                                 'return_layers': {'layer2': 1, 'layer3': 2, 'layer4': 3}}}
        self._config = b_bone_cfg[backbone]
        self.body = tv_utils.IntermediateLayerGetter(
            tv_models.resnet50() if backbone == "resnet" else MobileNetV1(),
            self._config["return_layers"]
        )
        in_channels_stage2 = T.cast(int, self._config["in_channels"])
        in_channels_list = [
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ]
        out_channels = T.cast(int, self._config["out_channel"])
        self.fpn = FPN(in_channels_list, out_channels)
        self.ssh1 = SSH(out_channels, out_channels)
        self.ssh2 = SSH(out_channels, out_channels)
        self.ssh3 = SSH(out_channels, out_channels)

        self.ClassHead = self._make_class_head(  # pylint:disable=invalid-name
            fpn_num=3, in_channels=out_channels)
        self.BboxHead = self._make_bbox_head(  # pylint:disable=invalid-name
            fpn_num=3, in_channels=out_channels)

    def _make_class_head(self, fpn_num: int = 3, in_channels: int = 64, anchor_num: int = 2
                         ) -> torch.nn.ModuleList:
        """Make the Class Head for RetinaFace

        Parameters
        ----------
        fpn_num
            The number of FPN modules. Default: 3
        in_channels
            The number of input channels. Default: 64
        num_anchors
            The number of anchors. Default: 2

        Returns
        -------
        The Class Head module list
        """
        class_head = nn.ModuleList()
        for _ in range(fpn_num):
            class_head.append(ClassHead(in_channels, anchor_num))
        return class_head

    def _make_bbox_head(self, fpn_num: int = 3, in_channels: int = 64, anchor_num: int = 2
                        ) -> torch.nn.ModuleList:
        """Make the Bounding Box Head for RetinaFace

        Parameters
        ----------
        fpn_num
            The number of FPN modules. Default: 3
        in_channels
            The number of input channels. Default: 64
        num_anchors
            The number of anchors. Default: 2

        Returns
        -------
        The Bounding Box Head module list
        """
        bbox_head = nn.ModuleList()
        for _ in range(fpn_num):
            bbox_head.append(BboxHead(in_channels, anchor_num))
        return bbox_head

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through RetinaFace

        Parameters
        ----------
        inputs
            The input to the RetinaFace Module

        Returns
        -------
        The output from RetinaFace Module
        """
        out = self.body(inputs)

        # FPN
        fpn = self.fpn(out)

        # SSH
        feature1 = self.ssh1(fpn[0])
        feature2 = self.ssh2(fpn[1])
        feature3 = self.ssh3(fpn[2])
        features = [feature1, feature2, feature3]

        bbox_regressions = torch.cat([self.BboxHead[i](feature)
                                      for i, feature in enumerate(features)], dim=1)
        classifications = torch.cat([self.ClassHead[i](feature)
                                     for i, feature in enumerate(features)], dim=1)
        output = (bbox_regressions, F.softmax(classifications, dim=-1))
        return output


__all__ = get_module_objects(__name__)
