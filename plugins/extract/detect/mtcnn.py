#!/usr/bin/env python3
"""MTCNN Face detection plugin"""
from __future__ import annotations
import logging

import cv2
import numpy as np

import torch
from torch import nn

from lib.logger import parse_class_init
from lib.utils import get_module_objects, GetModel
from plugins.extract.base import ExtractPlugin

from . import mtcnn_defaults as cfg


logger = logging.getLogger(__name__)


class MTCNN(ExtractPlugin):
    """MTCNN detector for face recognition."""
    def __init__(self) -> None:
        super().__init__(input_size=640,
                         batch_size=cfg.batch_size(),
                         is_rgb=True,
                         dtype="float32",
                         scale=(-1, 1),
                         force_cpu=cfg.cpu())
        self.model: MTCNNModel
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate that config options are correct. If not reset to default"""
        if cfg.min_size() < 10:
            logger.warning("Invalid MTCNN config value 'min_size': %s. Reset to %s",
                           cfg.min_size(), cfg.min_size.default)
            cfg.min_size.set(cfg.min_size.default)

        for idx, threshold in enumerate((cfg.threshold_1, cfg.threshold_2, cfg.threshold_3)):
            if not 0.0 < threshold() <= 1.0:
                logger.warning("Invalid MTCNN config value 'threshold_%s': %s. Reset to %s",
                               idx + 1, threshold(), threshold.default)
                threshold.set(threshold.default)

        if not 0.0 < cfg.scalefactor() < 1.0:
            logger.warning("Invalid MTCNN config value 'scalefactor': %s. Reset to %s",
                           cfg.scalefactor(), cfg.scalefactor.default)
            cfg.scalefactor.set(cfg.scalefactor.default)

    def _get_weights_path(self) -> list[str]:
        """Download the weights, if required, and return the path to the weights files

        Returns
        -------
        The paths to the downloaded MTCNN weights files
        """
        model = GetModel(
            model_filename=["mtcnn_det_v3.1.pt", "mtcnn_det_v3.2.pt", "mtcnn_det_v3.3.pt"],
            git_model_id=2)
        model_path = model.model_path
        assert isinstance(model_path, list)
        return model_path

    def load_model(self) -> MTCNNModel:
        """Load the model

        Returns
        -------
        The loaded MTCNN model
        """
        weights = self._get_weights_path()
        threshold = [cfg.threshold_1(), cfg.threshold_2(), cfg.threshold_3()]
        model = MTCNNModel(weights,
                           self.device,
                           input_size=self.input_size,
                           min_size=cfg.min_size(),
                           threshold=threshold,
                           factor=cfg.scalefactor())

        placeholder_shape = (self.batch_size, self.input_size, self.input_size, 3)
        placeholder = np.zeros(placeholder_shape, dtype="float32")

        model.detect_faces(placeholder)
        logger.debug("[%s] Loaded model", self.name)
        return model

    def pre_process(self, batch: np.ndarray) -> np.ndarray:
        """Compile the detection image(s) for prediction. No further pre-processing required for
        MTCNN

        Parameters
        ----------
        batch
            The input batch of images at model input size in the correct color order

        Returns
        -------
        The batch of images ready for feeding the model
        """
        return batch

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
        prediction, points = self.model.detect_faces(batch)
        logger.trace("prediction: %s, mtcnn_points: %s",  # type:ignore[attr-defined]
                     prediction, points)
        return prediction

    def post_process(self, batch: np.ndarray) -> np.ndarray:
        """Remove confidences from output

        Parameters
        ----------
        batch
            The detection results for the model

        Returns
        -------
        The processed detection bounding box from the model at model input size
        """
        return np.array([p[..., :4] for p in batch], dtype="object")


# MTCNN Detector
# Code adapted from: https://github.com/xiangrufan/keras-mtcnn and
# https://github.com/timesler/facenet-pytorch/blob/master/models/mtcnn.py
#
# Keras implementation of the face detection / alignment algorithm also
# found at
# https://github.com/kpzhang93/MTCNN_face_detection_alignment
#
# MIT License
#
# Copyright (c) 2016 Kaipeng Zhang
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


class PNet(nn.Module):
    """PyTorch P-Net model for MTCNN

    Parameters
    ----------
    weights_path
        The path to the keras model file
    """
    def __init__(self, weights_path: str) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 10, 3)
        self.prelu1 = nn.PReLU(10)
        self.pool1 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.conv2 = nn.Conv2d(10, 16, 3)
        self.prelu2 = nn.PReLU(16)
        self.conv3 = nn.Conv2d(16, 32, 3)
        self.prelu3 = nn.PReLU(32)
        self.conv4_1 = nn.Conv2d(32, 2, 1)
        self.softmax4_1 = nn.Softmax(dim=1)
        self.conv4_2 = nn.Conv2d(32, 4, 1)
        self.load_state_dict(torch.load(weights_path, map_location="cpu"))

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """PyTorch P-Network Definition for MTCNN

        Parameters
        ----------
        inputs
            The input tensor to PNet

        Returns
        -------
        classifier
           The result from PNet classifier
        bbox_regress
            The result from PNet bbox regression
        """
        var_x = self.pool1(self.prelu1(self.conv1(inputs)))
        var_x = self.prelu2(self.conv2(var_x))
        var_x = self.prelu3(self.conv3(var_x))

        classifier = self.softmax4_1(self.conv4_1(var_x))
        bbox_regress = self.conv4_2(var_x)

        return classifier, bbox_regress


class PNetRunner():
    """Runner for PyTorch P-Net model for MTCNN

    Parameters
    ----------
    weights_path
        The path to the keras model file
    device
        The device to use for model inference
    input_size
        The input size of the model
    min_size
        The minimum size of a face to accept as a detection. Default: `20`
    threshold
        Threshold for P-Net
    """
    def __init__(self,
                 weights_path: str,
                 device: torch.device,
                 input_size: int,
                 min_size: int,
                 factor: float,
                 threshold: float) -> None:
        logger.debug(parse_class_init(locals()))
        self._model = PNet(weights_path)
        self._model.to(device,
                       memory_format=torch.channels_last)  # type:ignore[call-overload]
        self.device = device

        self._input_size = input_size
        self._threshold = threshold

        self._pnet_scales = self._calculate_scales(min_size, factor)
        self._pnet_sizes = [(int(input_size * scale), int(input_size * scale))
                            for scale in self._pnet_scales]
        logger.debug("Initialized: %s", self.__class__.__name__)

    def _calculate_scales(self, min_size: int, factor: float) -> list[float]:
        """Calculate multi-scale

        Parameters
        ----------
        min_size
            Minimum size for a face to be accepted
        factor
            Scaling factor

        Returns
        -------
        List of scale floats
        """
        factor_count = 0
        var_m = 12.0 / min_size
        min_l = self._input_size * var_m
        # create scale pyramid
        scales = []
        while min_l >= 12:
            scales += [var_m * np.power(factor, factor_count)]
            min_l = min_l * factor
            factor_count += 1
        logger.trace(scales)  # type:ignore[attr-defined]
        return scales

    def _detect_face_12net(self,
                           class_probabilities: np.ndarray,
                           roi: np.ndarray,
                           size: int,
                           scale: float) -> tuple[np.ndarray, np.ndarray]:
        """Detect face position and calibrate bounding box on 12net feature map(matrix version)

        Parameters
        ----------
        class_probabilities
            softmax feature map for face classify
        roi
            feature map for regression
        size
            feature map's largest size
        scale
            current input image scale in multi-scales

        Returns
        -------
        Calibrated face candidates
        """
        in_side = 2 * size + 11
        stride = 0. if size == 1 else float(in_side - 12) / (size - 1)
        (var_x, var_y) = np.nonzero(class_probabilities >= self._threshold)
        bbox = np.array([var_x, var_y]).T

        bbox = np.concatenate((np.fix((stride * (bbox) + 0) * scale),
                               np.fix((stride * (bbox) + 11) * scale)), axis=1)
        offset = roi[:4, var_x, var_y].T
        bbox = bbox + offset * 12.0 * scale
        rectangles = np.concatenate((bbox,
                                     np.array([class_probabilities[var_x, var_y]]).T), axis=1)
        rectangles = rect2square(rectangles)

        np.clip(rectangles[..., :4], 0., self._input_size, out=rectangles[..., :4])
        pick = np.where(np.logical_and(rectangles[..., 2] > rectangles[..., 0],
                                       rectangles[..., 3] > rectangles[..., 1]))[0]
        rect = rectangles[pick, :4].astype("int")
        scores = rectangles[pick, 4]

        return nms(rect, scores, 0.3, "iou")

    def __call__(self, images: np.ndarray) -> list[np.ndarray]:  # pylint:disable=too-many-locals
        """first stage - fast proposal network (p-net) to obtain face candidates

        Parameters
        ----------
        images
            The batch of images to detect faces in

        Returns
        -------
        List of face candidates from P-Net
        """
        batch_size = images.shape[0]
        rectangles: list[list[list[int | float]]] = [[] for _ in range(batch_size)]
        scores: list[list[np.ndarray]] = [[] for _ in range(batch_size)]

        pnet_input = [np.empty((batch_size, r_height, r_width, 3), dtype="float32")
                      for r_height, r_width in self._pnet_sizes]

        for scale, batch, (r_height, r_width) in zip(self._pnet_scales,
                                                     pnet_input,
                                                     self._pnet_sizes):
            for idx in range(batch_size):
                cv2.resize(images[idx], (r_width, r_height), dst=batch[idx])

            feed = torch.from_numpy(batch.transpose(0, 3, 1, 2)).to(
                self.device,
                memory_format=torch.channels_last)
            with torch.inference_mode():
                cls_prob, roi = (t.cpu().numpy() for t in self._model(feed))
            cls_prob = cls_prob[:, 1]
            out_side = max(cls_prob.shape[1:3])
            cls_prob = np.swapaxes(cls_prob, 1, 2)
            for idx in range(batch_size):
                # first index 0 = class score, 1 = one hot representation
                rect, score = self._detect_face_12net(cls_prob[idx, ...],
                                                      roi[idx, ...],
                                                      out_side,
                                                      1 / scale)
                rectangles[idx].extend(rect)
                scores[idx].extend(score)

        return [nms(np.array(rect), np.array(score), 0.7, "iou")[0]  # don't output scores
                for rect, score in zip(rectangles, scores)]


class RNet(nn.Module):  # pylint:disable=too-many-instance-attributes
    """PyTorch R-Net model Definition for MTCNN

    Parameters
    ----------
    weights_path
        The path to the torch weights file
    """
    def __init__(self, weights_path: str) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 28, 3)
        self.prelu1 = nn.PReLU(28)
        self.pool1 = nn.MaxPool2d(3, 2, ceil_mode=True)
        self.conv2 = nn.Conv2d(28, 48, 3)
        self.prelu2 = nn.PReLU(48)
        self.pool2 = nn.MaxPool2d(3, 2, ceil_mode=True)
        self.conv3 = nn.Conv2d(48, 64, 2)
        self.prelu3 = nn.PReLU(64)
        self.dense4 = nn.Linear(576, 128)
        self.prelu4 = nn.PReLU(128)
        self.dense5_1 = nn.Linear(128, 2)
        self.softmax5_1 = nn.Softmax(dim=1)
        self.dense5_2 = nn.Linear(128, 4)
        self.load_state_dict(torch.load(weights_path, map_location="cpu"))

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Keras R-Network Definition for MTCNN

        Parameters
        ----------
        inputs
            The input to RNet

        Returns
        -------
        classifier
           The result from RNet classifier
        bbox_regress
            The result from RNet bbox regression
        """
        var_x = self.pool1(self.prelu1(self.conv1(inputs)))
        var_x = self.pool2(self.prelu2(self.conv2(var_x)))
        var_x = self.prelu3(self.conv3(var_x))
        var_x = var_x.permute(0, 3, 2, 1).contiguous()
        var_x = self.prelu4(self.dense4(var_x.view(var_x.shape[0], -1)))
        classifier = self.softmax5_1(self.dense5_1(var_x))
        bbox_regress = self.dense5_2(var_x)
        return classifier, bbox_regress


class RNetRunner():
    """Runner for PyTorch R-Net for MTCNN

    Parameters
    ----------
    weights_path
        The path to the keras model file
    device
        The device to run inference on
    input_size
        The input size of the model
    threshold
        Threshold for R-Net
    """
    def __init__(self,
                 weights_path: str,
                 device: torch.device,
                 input_size: int,
                 threshold: float) -> None:
        logger.debug(parse_class_init(locals()))
        self._model = RNet(weights_path)
        self._model.to(device,
                       memory_format=torch.channels_last)  # type:ignore[call-overload]
        self.device = device
        self._input_size = input_size
        self._threshold = threshold
        logger.debug("Initialized: %s", self.__class__.__name__)

    def _filter_face_24net(self,
                           class_probabilities: np.ndarray,
                           roi: np.ndarray,
                           rectangles: np.ndarray,
                           ) -> np.ndarray:
        """Filter face position and calibrate bounding box on 12net's output

        Parameters
        ----------
        class_probabilities
            Softmax feature map for face classify
        roi
            Feature map for regression
        rectangles
            12net's predict

        Returns
        -------
        Rectangles in the format [[x, y, x1, y1, score]]
        """
        prob = class_probabilities[:, 1]
        pick = np.nonzero(prob >= self._threshold)

        bbox = rectangles.T[:4, pick]
        scores = np.array([prob[pick]]).T.ravel()
        deltas = roi.T[:4, pick]

        dims = np.tile([bbox[2] - bbox[0], bbox[3] - bbox[1]], (2, 1, 1))
        bbox = np.transpose(bbox + deltas * dims).reshape(-1, 4)
        bbox = np.clip(rect2square(bbox), 0, self._input_size).astype("int")
        return nms(bbox, scores, 0.3, "iou")[0]

    def __call__(self,
                 images: np.ndarray,
                 rectangle_batch: list[np.ndarray],
                 ) -> list[np.ndarray]:
        """second stage - refinement of face candidates with r-net

        Parameters
        ----------
        images
            The batch of images to detect faces in
        rectangle_batch
            face candidates from P-Net

        Returns
        -------
        Refined face candidates from R-Net
        """
        ret: list[np.ndarray] = []
        for idx, (rectangles, image) in enumerate(zip(rectangle_batch, images)):
            if not np.any(rectangles):
                ret.append(np.array([]))
                continue

            batch = np.empty((rectangles.shape[0], 24, 24, 3), dtype="float32")

            for idx, rect in enumerate(rectangles):
                cv2.resize(image[rect[1]: rect[3], rect[0]: rect[2]], (24, 24), dst=batch[idx])

            feed = torch.from_numpy(batch.transpose(0, 3, 1, 2)).to(
                self.device,
                memory_format=torch.channels_last)
            with torch.inference_mode():
                cls_prob, roi_prob = (t.cpu().numpy() for t in self._model(feed))

            ret.append(self._filter_face_24net(cls_prob, roi_prob, rectangles))
        return ret


class ONet(nn.Module):  # pylint:disable=too-many-instance-attributes
    """PyTorch O-Net model Definition for MTCNN

    Parameters
    ----------
    weights_path
        The path to the torch weights file
    """
    def __init__(self, weights_path: str) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.prelu1 = nn.PReLU(32)
        self.pool1 = nn.MaxPool2d(3, 2, ceil_mode=True)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.prelu2 = nn.PReLU(64)
        self.pool2 = nn.MaxPool2d(3, 2, ceil_mode=True)
        self.conv3 = nn.Conv2d(64, 64, 3)
        self.prelu3 = nn.PReLU(64)
        self.pool3 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.conv4 = nn.Conv2d(64, 128, 2)
        self.prelu4 = nn.PReLU(128)
        self.dense5 = nn.Linear(1152, 256)
        self.prelu5 = nn.PReLU(256)
        self.dense6_1 = nn.Linear(256, 2)
        self.softmax6_1 = nn.Softmax(dim=1)
        self.dense6_2 = nn.Linear(256, 4)
        self.dense6_3 = nn.Linear(256, 10)
        self.load_state_dict(torch.load(weights_path, map_location="cpu"))

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Keras O-Network Definition for MTCNN

        Parameters
        ----------
        inputs
            The input to ONet

        Returns
        -------
        classifier
           The result from ONet classifier
        bbox_regress
            The result from ONet bbox regression
        landmark_regress
            The result from ONet landmark regression
        """
        var_x = self.pool1(self.prelu1(self.conv1(inputs)))
        var_x = self.pool2(self.prelu2(self.conv2(var_x)))
        var_x = self.pool3(self.prelu3(self.conv3(var_x)))
        var_x = self.prelu4(self.conv4(var_x))
        var_x = var_x.permute(0, 3, 2, 1).contiguous()
        var_x = self.prelu5(self.dense5(var_x.view(var_x.shape[0], -1)))
        classifier = self.softmax6_1(self.dense6_1(var_x))
        bbox_regress = self.dense6_2(var_x)
        landmark_regress = self.dense6_3(var_x)
        return classifier, bbox_regress, landmark_regress


class ONetRunner():
    """Keras O-Net model for MTCNN

    Parameters
    ----------
    weights_path
        The path to the keras model file
    device
        The device to run inference on
    input_size
        The input size of the model
    threshold
        Threshold for O-Net
    """
    def __init__(self,
                 weights_path: str,
                 device: torch.device,
                 input_size: int,
                 threshold: float) -> None:
        logger.debug(parse_class_init(locals()))
        self._model = ONet(weights_path)
        self._model.to(device,
                       memory_format=torch.channels_last)  # type:ignore[call-overload]
        self.device = device
        self._input_size = input_size
        self._threshold = threshold
        logger.debug("Initialized: %s", self.__class__.__name__)

    def _filter_face_48net(self, class_probabilities: np.ndarray,
                           roi: np.ndarray,
                           points: np.ndarray,
                           rectangles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Filter face position and calibrate bounding box on 12net's output

        Parameters
        ----------
        class_probabilities
            class_probabilities[1] is face possibility. Array of face probabilities
        roi
            offset
        points
            5 point face landmark
        rectangles
            12net's predict, rectangles[i][0:3] is the position, rectangles[i][4] is score

        Returns
        -------
        boxes
            The [l, t, r, b, score] bounding boxes
        points
            The 5 point landmarks
        """
        prob = class_probabilities[:, 1]
        pick = np.nonzero(prob >= self._threshold)[0]
        scores = np.array([prob[pick]]).T.ravel()

        bbox = rectangles[pick]
        dims = np.array([bbox[..., 2] - bbox[..., 0], bbox[..., 3] - bbox[..., 1]]).T

        pts = np.vstack(
            np.hsplit(points[pick], 2)).reshape(2, -1, 5).transpose(1, 2, 0).reshape(-1, 10)
        pts = np.tile(dims, (1, 5)) * pts + np.tile(bbox[..., :2], (1, 5))

        bbox = np.clip(np.floor(bbox + roi[pick] * np.tile(dims, (1, 2))),
                       0.,
                       self._input_size)

        indices = np.where(
            np.logical_and(bbox[..., 2] > bbox[..., 0], bbox[..., 3] > bbox[..., 1]))[0]
        picks = np.concatenate([bbox[indices], pts[indices]], axis=-1)

        results, scores = nms(picks, scores, 0.3, "iom")
        return np.concatenate([results[..., :4], scores[..., None]], axis=-1), results[..., 4:].T

    def __call__(self,
                 images: np.ndarray,
                 rectangle_batch: list[np.ndarray]
                 ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Third stage - further refinement and facial landmarks positions with o-net

        Parameters
        ----------
        images
            The batch of images to detect faces in
        rectangle_batch
            List of :class:`numpy.ndarray` face candidates from R-Net

        Returns
        -------
        List of refined final candidates, scores and landmark points from O-Net
        """
        ret: list[tuple[np.ndarray, np.ndarray]] = []
        for idx, rectangles in enumerate(rectangle_batch):
            if not np.any(rectangles):
                ret.append((np.empty((0, 5)), np.empty(0)))
                continue
            image = images[idx]
            batch = np.empty((rectangles.shape[0], 48, 48, 3), dtype="float32")

            for i, rect in enumerate(rectangles):
                cv2.resize(image[rect[1]: rect[3], rect[0]: rect[2]], (48, 48), dst=batch[i])

            feed = torch.from_numpy(batch.transpose(0, 3, 1, 2)).to(
                self.device,
                memory_format=torch.channels_last)
            with torch.inference_mode():
                cls_probs, roi_probs, pts_probs = (t.cpu().numpy()
                                                   for t in self._model(feed))
            ret.append(self._filter_face_48net(cls_probs, roi_probs, pts_probs, rectangles))
        return ret


class MTCNNModel():
    """MTCNN Detector for face alignment

    Parameters
    ----------
    weights_path
        List of paths to the 3 MTCNN subnet weights
    device
        The device to run inference on
    input_size
        The height, width input size to the model. Default: 640
    min_size
        The minimum size of a face to accept as a detection. Default: `20`
    threshold
        List of floats for the three steps, Default: `[0.6, 0.7, 0.7]`
    factor
        The factor used to create a scaling pyramid of face sizes to detect in the image.
        Default: `0.709`
    """
    def __init__(self,
                 weights_path: list[str],
                 device: torch.device,
                 input_size: int = 640,
                 min_size: int = 20,
                 threshold: list[float] | None = None,
                 factor: float = 0.709) -> None:
        logger.debug(parse_class_init(locals()))
        threshold = [0.6, 0.7, 0.7] if threshold is None else threshold
        self._pnet = PNetRunner(weights_path[0],
                                device,
                                input_size,
                                min_size,
                                factor,
                                threshold[0])
        self._rnet = RNetRunner(weights_path[1],
                                device,
                                input_size,
                                threshold[1])
        self._onet = ONetRunner(weights_path[2],
                                device,
                                input_size,
                                threshold[2])
        logger.debug("Initialized: %s", self.__class__.__name__)

    def detect_faces(self, batch: np.ndarray) -> tuple[np.ndarray, tuple[np.ndarray]]:
        """Detects faces in an image, and returns bounding boxes and points for them.

        Parameters
        ----------
        batch
            The input batch of images to detect face in

        Returns
        -------
        List of numpy arrays containing the bounding box and 5 point landmarks of detected faces
        """
        rectangles = self._pnet(batch)
        rectangles = self._rnet(batch, rectangles)

        ret_boxes, ret_points = zip(*self._onet(batch, rectangles))
        return np.array(ret_boxes, dtype="object"), ret_points


def nms(rectangles: np.ndarray,
        scores: np.ndarray,
        threshold: float,
        method: str = "iom") -> tuple[np.ndarray, np.ndarray]:
    """Apply non-maximum suppression on ROIs in same scale(matrix version)

    Parameters
    ----------
    rectangles
        The [b, l, t, r, b] bounding box detection candidates
    threshold
        Threshold for successful match
    method
        "iom" method or default. default: "iom"

    Returns
    -------
    rectangles
        The [b, l, t, r, b] bounding boxes
    scores
        The associated scores for the rectangles
    """
    if not np.any(rectangles):
        return rectangles, scores
    bboxes = rectangles[..., :4].T
    area = np.multiply(bboxes[2] - bboxes[0] + 1, bboxes[3] - bboxes[1] + 1)
    s_sort = scores.argsort()

    pick = []
    while len(s_sort) > 0:
        s_bboxes = np.concatenate([  # s_sort[-1] have highest prob score, s_sort[0:-1]->others
            np.maximum(bboxes[:2, s_sort[-1], None], bboxes[:2, s_sort[0:-1]]),
            np.minimum(bboxes[2:, s_sort[-1], None], bboxes[2:, s_sort[0:-1]])], axis=0)

        inter = (np.maximum(0.0, s_bboxes[2] - s_bboxes[0] + 1) *
                 np.maximum(0.0, s_bboxes[3] - s_bboxes[1] + 1))

        if method == "iom":
            var_o = inter / np.minimum(area[s_sort[-1]], area[s_sort[0:-1]])
        else:
            var_o = inter / (area[s_sort[-1]] + area[s_sort[0:-1]] - inter)

        pick.append(s_sort[-1])
        s_sort = s_sort[np.where(var_o <= threshold)[0]]

    result_rectangle = rectangles[pick]
    result_scores = scores[pick]
    return result_rectangle, result_scores


def rect2square(rectangles: np.ndarray) -> np.ndarray:
    """change rectangles into squares (matrix version)

    Parameters
    ----------
    rectangles
        [b, x, y, x1, y1] rectangles

    Return
    ------
    Original rectangle changed to a square
    """
    width = rectangles[:, 2] - rectangles[:, 0]
    height = rectangles[:, 3] - rectangles[:, 1]
    length = np.maximum(width, height).T
    rectangles[:, 0] = rectangles[:, 0] + width * 0.5 - length * 0.5
    rectangles[:, 1] = rectangles[:, 1] + height * 0.5 - length * 0.5
    rectangles[:, 2:4] = rectangles[:, 0:2] + np.repeat([length], 2, axis=0).T
    return rectangles


__all__ = get_module_objects(__name__)
