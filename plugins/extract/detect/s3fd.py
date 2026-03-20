#!/usr/bin/env python3
"""S3FD Face detection plugin
https://arxiv.org/abs/1708.05237

Adapted from S3FD Port in FAN:
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
from . import s3fd_defaults as cfg


logger = logging.getLogger(__name__)
# pylint:disable=duplicate-code


class S3FD(ExtractPlugin):
    """S3FD detector for face detection"""
    def __init__(self) -> None:
        super().__init__(input_size=640,
                         batch_size=cfg.batch_size(),
                         is_rgb=False,
                         dtype="float32",
                         scale=(0, 255))
        self.model: S3FDModel
        self._model_path = self._get_weights_path()
        self._average_img = np.array([104.0, 117.0, 123.0], dtype="float32")
        self._confidence = cfg.confidence() / 100

    def _get_weights_path(self) -> str:
        """Download the weights, if required, and return the path to the weights files

        Returns
        -------
        The path to the downloaded S3FD weights file
        """
        model = GetModel(model_filename="s3fd_torch_v3.pth", git_model_id=11)
        model_path = model.model_path
        assert isinstance(model_path, str)
        return model_path

    def load_model(self) -> S3FDModel:
        """Load the S3FD Model

        Returns
        -------
        The loaded S3FD model
        """
        weights = GetModel(model_filename="s3fd_torch_v3.pth", git_model_id=11).model_path
        assert isinstance(weights, str)
        return T.cast(S3FDModel, self.load_torch_model(S3FDModel(), weights))

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

    @staticmethod
    def decode(location: np.ndarray, priors: np.ndarray) -> np.ndarray:
        """Decode locations from predictions using priors to undo the encoding we did for offset
        regression at train time.

        Parameters
        ----------
        location
            location predictions for location layers,
        priors
            Prior boxes in center-offset form.

        Returns
        -------
        Decoded bounding box predictions
        """
        variances = [0.1, 0.2]
        boxes = np.concatenate((priors[:, :2] + location[:, :2] * variances[0] * priors[:, 2:],
                                priors[:, 2:] * np.exp(location[:, 2:] * variances[1])), axis=1)
        boxes[:, :2] -= boxes[:, 2:] / 2
        boxes[:, 2:] += boxes[:, :2]
        return boxes

    def _process_bbox(self,  # pylint:disable=too-many-locals
                      o_cls: np.ndarray,
                      o_reg: np.ndarray,
                      stride: int) -> list[list[np.ndarray]]:
        """Process a bounding box

        Parameters
        ----------
        o_cls
            The class outputs from S3FD
        o_reg
            The reg outputs from S3FD
        stride
            The stride to use

        Returns
        -------
        The bounding boxes with scores
        """
        retval = []
        for _, h_idx, w_idx in zip(*np.where(o_cls[:, 1, :, :] > 0.05)):
            axc, ayc = stride / 2 + w_idx * stride, stride / 2 + h_idx * stride
            score = o_cls[0, 1, h_idx, w_idx]
            if score < self._confidence:
                continue
            loc = o_reg[:, :, h_idx, w_idx].copy()
            priors = np.array([[axc / 1.0, ayc / 1.0, stride * 4 / 1.0, stride * 4 / 1.0]])
            box = self.decode(loc, priors)
            x_1, y_1, x_2, y_2 = box[0] * 1.0
            retval.append([x_1, y_1, x_2, y_2, score])
        return retval

    def _post_process(self, bbox_list: list[np.ndarray]) -> np.ndarray:
        """Perform post processing on output

        Parameters
        ----------
        bbox_list
            The class and reg outputs from the S3FD model

        Returns
        -------
        The [N, left, top, right, bottom, score] bounding boxes from the model
        """
        retval = []
        for i in range(len(bbox_list) // 2):
            o_cls, o_reg = bbox_list[i * 2], bbox_list[i * 2 + 1]
            stride = 2 ** (i + 2)    # 4,8,16,32,64,128
            retval.extend(self._process_bbox(o_cls, o_reg, stride))

        return_numpy = np.array(retval) if len(retval) != 0 else np.zeros((1, 5))
        return return_numpy

    @staticmethod
    def _nms(boxes: np.ndarray, threshold: float) -> np.ndarray:
        """Perform Non-Maximum Suppression

        Parameters
        ----------
        boxes
            The detection bounding boxes to process
        threshold
            The threshold to accept boxes

        Returns
        -------
        The final bounding boxes
        """
        retained_box_indices = []

        areas = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
        ranked_indices = boxes[:, 4].argsort()[::-1]
        while ranked_indices.size > 0:
            best_rest = ranked_indices[0], ranked_indices[1:]

            max_of_xy = np.maximum(boxes[best_rest[0], :2], boxes[best_rest[1], :2])
            min_of_xy = np.minimum(boxes[best_rest[0], 2:4], boxes[best_rest[1], 2:4])
            width_height = np.maximum(0, min_of_xy - max_of_xy + 1)
            intersection_areas = width_height[:, 0] * width_height[:, 1]
            iou = intersection_areas / (areas[best_rest[0]] +
                                        areas[best_rest[1]] - intersection_areas)

            overlapping_boxes = (iou > threshold).nonzero()[0]
            if len(overlapping_boxes) != 0:
                overlap_set = ranked_indices[overlapping_boxes + 1]
                vote = np.average(boxes[overlap_set, :4], axis=0, weights=boxes[overlap_set, 4])
                boxes[best_rest[0], :4] = vote
            retained_box_indices.append(best_rest[0])

            non_overlapping_boxes = (iou <= threshold).nonzero()[0]
            ranked_indices = ranked_indices[non_overlapping_boxes + 1]
        return boxes[retained_box_indices]

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
        ret = []
        batch_size = range(batch[0].shape[0])
        for img in batch_size:
            bbox_list = [scale[img:img+1] for scale in batch]
            boxes = self._post_process(bbox_list)
            final_list = self._nms(boxes, 0.5)
            ret.append(final_list[..., :4])
        retval = np.empty(len(ret), dtype=object)
        retval[:] = ret
        return retval


################################################################################
# S3FD Net
################################################################################
class L2Norm(nn.Module):
    """L2 Normalization layer for S3FD.

    Parameters
    ----------
    n_channels
        The number of channels to normalize
    scale
        The scaling for initial weights. Default: `1.0`
    """
    def __init__(self, n_channels: int, scale: float) -> None:
        super().__init__()
        self.n_channels = n_channels
        self.gamma = scale
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))

    def forward(self, inputs: torch.Tensor):
        """Call the L2 Normalization Layer.

        Parameters
        ----------
        inputs
            The input to the L2 Normalization Layer

        Returns
        -------
        The output from the L2 Normalization Layer
        """
        norm = inputs.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps
        x = inputs / norm * self.weight.view(1, -1, 1, 1)
        return x


class S3FDModel(nn.Module):  # pylint:disable=too-many-instance-attributes
    """The S3FD Model, adapted from https://github.com/1adrianb/face-alignment"""
    def __init__(self) -> None:
        super().__init__()
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.fc6 = nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=3)
        self.fc7 = nn.Conv2d(1024, 1024, kernel_size=1, stride=1, padding=0)

        self.conv6_1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.conv6_2 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)

        self.conv7_1 = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0)
        self.conv7_2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)

        self.conv3_3_norm = L2Norm(256, scale=10)
        self.conv4_3_norm = L2Norm(512, scale=8)
        self.conv5_3_norm = L2Norm(512, scale=5)

        self.conv3_3_norm_mbox_conf = nn.Conv2d(256, 4, kernel_size=3, stride=1, padding=1)
        self.conv3_3_norm_mbox_loc = nn.Conv2d(256, 4, kernel_size=3, stride=1, padding=1)
        self.conv4_3_norm_mbox_conf = nn.Conv2d(512, 2, kernel_size=3, stride=1, padding=1)
        self.conv4_3_norm_mbox_loc = nn.Conv2d(512, 4, kernel_size=3, stride=1, padding=1)
        self.conv5_3_norm_mbox_conf = nn.Conv2d(512, 2, kernel_size=3, stride=1, padding=1)
        self.conv5_3_norm_mbox_loc = nn.Conv2d(512, 4, kernel_size=3, stride=1, padding=1)

        self.fc7_mbox_conf = nn.Conv2d(1024, 2, kernel_size=3, stride=1, padding=1)
        self.fc7_mbox_loc = nn.Conv2d(1024, 4, kernel_size=3, stride=1, padding=1)
        self.conv6_2_mbox_conf = nn.Conv2d(512, 2, kernel_size=3, stride=1, padding=1)
        self.conv6_2_mbox_loc = nn.Conv2d(512, 4, kernel_size=3, stride=1, padding=1)
        self.conv7_2_mbox_conf = nn.Conv2d(256, 2, kernel_size=3, stride=1, padding=1)
        self.conv7_2_mbox_loc = nn.Conv2d(256, 4, kernel_size=3, stride=1, padding=1)

    def forward(self,  # pylint:disable=too-many-locals,too-many-statements
                inputs: torch.Tensor) -> list[torch.Tensor]:
        """Run the forward pass through S3FD

        Parameters
        ----------
        inputs
            The (N, C, H, W) batch of images to process

        Returns
        -------
        The predictions from the S3FD model
        """
        h = F.relu(self.conv1_1(inputs), inplace=True)
        h = F.relu(self.conv1_2(h), inplace=True)
        h = F.max_pool2d(h, 2, 2)

        h = F.relu(self.conv2_1(h), inplace=True)
        h = F.relu(self.conv2_2(h), inplace=True)
        h = F.max_pool2d(h, 2, 2)

        h = F.relu(self.conv3_1(h), inplace=True)
        h = F.relu(self.conv3_2(h), inplace=True)
        h = F.relu(self.conv3_3(h), inplace=True)
        f3_3 = h
        h = F.max_pool2d(h, 2, 2)

        h = F.relu(self.conv4_1(h), inplace=True)
        h = F.relu(self.conv4_2(h), inplace=True)
        h = F.relu(self.conv4_3(h), inplace=True)
        f4_3 = h
        h = F.max_pool2d(h, 2, 2)

        h = F.relu(self.conv5_1(h), inplace=True)
        h = F.relu(self.conv5_2(h), inplace=True)
        h = F.relu(self.conv5_3(h), inplace=True)
        f5_3 = h
        h = F.max_pool2d(h, 2, 2)

        h = F.relu(self.fc6(h), inplace=True)
        h = F.relu(self.fc7(h), inplace=True)
        ffc7 = h
        h = F.relu(self.conv6_1(h), inplace=True)
        h = F.relu(self.conv6_2(h), inplace=True)
        f6_2 = h
        h = F.relu(self.conv7_1(h), inplace=True)
        h = F.relu(self.conv7_2(h), inplace=True)
        f7_2 = h

        f3_3 = self.conv3_3_norm(f3_3)
        f4_3 = self.conv4_3_norm(f4_3)
        f5_3 = self.conv5_3_norm(f5_3)

        cls1 = self.conv3_3_norm_mbox_conf(f3_3)
        reg1 = self.conv3_3_norm_mbox_loc(f3_3)
        cls2 = self.conv4_3_norm_mbox_conf(f4_3)
        reg2 = self.conv4_3_norm_mbox_loc(f4_3)
        cls3 = self.conv5_3_norm_mbox_conf(f5_3)
        reg3 = self.conv5_3_norm_mbox_loc(f5_3)
        cls4 = self.fc7_mbox_conf(ffc7)
        reg4 = self.fc7_mbox_loc(ffc7)
        cls5 = self.conv6_2_mbox_conf(f6_2)
        reg5 = self.conv6_2_mbox_loc(f6_2)
        cls6 = self.conv7_2_mbox_conf(f7_2)
        reg6 = self.conv7_2_mbox_loc(f7_2)

        # max-out background label
        chunk = torch.chunk(cls1, 4, 1)
        b_max = torch.max(torch.max(chunk[0], chunk[1]), chunk[2])
        cls1 = torch.cat([b_max, chunk[3]], dim=1)

        outputs = [cls1, reg1, cls2, reg2, cls3, reg3, cls4, reg4, cls5, reg5, cls6, reg6]
        for i in range(len(outputs) // 2):
            outputs[i * 2] = F.softmax(outputs[i * 2], dim=1)

        return outputs


__all__ = get_module_objects(__name__)
