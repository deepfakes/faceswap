#!/usr/bin/env python3
""" MTCNN Face detection plugin """
from __future__ import annotations
import logging
import typing as T

import cv2
import numpy as np

# Ignore linting errors from Tensorflow's thoroughly broken import system
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Input, MaxPool2D, Permute, PReLU  # noqa:E501  # pylint:disable=import-error

from lib.model.session import KSession
from ._base import BatchType, Detector

if T.TYPE_CHECKING:
    from tensorflow import Tensor

logger = logging.getLogger(__name__)


class Detect(Detector):
    """ MTCNN detector for face recognition. """
    def __init__(self, **kwargs) -> None:
        git_model_id = 2
        model_filename = ["mtcnn_det_v2.1.h5", "mtcnn_det_v2.2.h5", "mtcnn_det_v2.3.h5"]
        super().__init__(git_model_id=git_model_id, model_filename=model_filename, **kwargs)
        self.name = "MTCNN"
        self.input_size = 640
        self.vram = 320 if not self.config["cpu"] else 0
        self.vram_warnings = 64 if not self.config["cpu"] else 0  # Will run at this with warnings
        self.vram_per_batch = 32 if not self.config["cpu"] else 0
        self.batchsize = self.config["batch-size"]
        self.kwargs = self._validate_kwargs()
        self.color_format = "RGB"

    def _validate_kwargs(self) -> dict[str, int | float | list[float]]:
        """ Validate that config options are correct. If not reset to default """
        valid = True
        threshold = [self.config["threshold_1"],
                     self.config["threshold_2"],
                     self.config["threshold_3"]]
        kwargs = {"minsize": self.config["minsize"],
                  "threshold": threshold,
                  "factor": self.config["scalefactor"],
                  "input_size": self.input_size}

        if kwargs["minsize"] < 10:
            valid = False
        elif not all(0.0 < threshold <= 1.0 for threshold in kwargs['threshold']):
            valid = False
        elif not 0.0 < kwargs['factor'] < 1.0:
            valid = False

        if not valid:
            kwargs = {}
            logger.warning("Invalid MTCNN options in config. Running with defaults")

        logger.debug("Using mtcnn kwargs: %s", kwargs)
        return kwargs

    def init_model(self) -> None:
        """ Initialize MTCNN Model. """
        assert isinstance(self.model_path, list)
        self.model = MTCNN(self.model_path,
                           self.config["allow_growth"],
                           self._exclude_gpus,
                           self.config["cpu"],
                           **self.kwargs)  # type:ignore

    def process_input(self, batch: BatchType) -> None:
        """ Compile the detection image(s) for prediction

        Parameters
        ----------
        batch: :class:`~plugins.extract.detect._base.DetectorBatch`
            Contains the batch that is currently being passed through the plugin process
        """
        batch.feed = (np.array(batch.image, dtype="float32") - 127.5) / 127.5

    def predict(self, feed: np.ndarray) -> np.ndarray:
        """ Run model to get predictions

        Parameters
        ----------
        batch: :class:`~plugins.extract.detect._base.DetectorBatch`
            Contains the batch to pass through the MTCNN model

        Returns
        -------
        dict
            The batch with the predictions added to the dictionary
        """
        assert isinstance(self.model, MTCNN)
        prediction, points = self.model.detect_faces(feed)
        logger.trace("prediction: %s, mtcnn_points: %s",  # type:ignore
                     prediction, points)
        return prediction

    def process_output(self, batch: BatchType) -> None:
        """ MTCNN performs no post processing so the original batch is returned

        Parameters
        ----------
        batch: :class:`~plugins.extract.detect._base.DetectorBatch`
            Contains the batch to apply postprocessing to
        """
        return


# MTCNN Detector
# Code adapted from: https://github.com/xiangrufan/keras-mtcnn
#
# Keras implementation of the face detection / alignment algorithm
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


class PNet(KSession):
    """ Keras P-Net model for MTCNN

    Parameters
    ----------
    model_path: str
        The path to the keras model file
    allow_growth: bool, optional
        Enable the Tensorflow GPU allow_growth configuration option. This option prevents
        Tensorflow from allocating all of the GPU VRAM, but can lead to higher fragmentation and
        slower performance. Default: ``False``
    exclude_gpus: list, optional
        A list of indices correlating to connected GPUs that Tensorflow should not use. Pass
        ``None`` to not exclude any GPUs. Default: ``None``
    cpu_mode: bool, optional
        ``True`` run the model on CPU. Default: ``False``
    input_size: int
        The input size of the model
    minsize: int, optional
        The minimum size of a face to accept as a detection. Default: `20`
    threshold: list, optional
        Threshold for P-Net
    """
    def __init__(self,
                 model_path: str,
                 allow_growth: bool,
                 exclude_gpus: list[int] | None,
                 cpu_mode: bool,
                 input_size: int,
                 min_size: int,
                 factor: float,
                 threshold: float) -> None:
        super().__init__("MTCNN-PNet",
                         model_path,
                         allow_growth=allow_growth,
                         exclude_gpus=exclude_gpus,
                         cpu_mode=cpu_mode)

        self.define_model(self.model_definition)
        self.load_model_weights()

        self._input_size = input_size
        self._threshold = threshold

        self._pnet_scales = self._calculate_scales(min_size, factor)
        self._pnet_sizes = [(int(input_size * scale), int(input_size * scale))
                            for scale in self._pnet_scales]
        self._pnet_input: list[np.ndarray] | None = None

    @staticmethod
    def model_definition() -> tuple[list[Tensor], list[Tensor]]:
        """ Keras P-Network Definition for MTCNN """
        input_ = Input(shape=(None, None, 3))
        var_x = Conv2D(10, (3, 3), strides=1, padding='valid', name='conv1')(input_)
        var_x = PReLU(shared_axes=[1, 2], name='PReLU1')(var_x)
        var_x = MaxPool2D(pool_size=2)(var_x)
        var_x = Conv2D(16, (3, 3), strides=1, padding='valid', name='conv2')(var_x)
        var_x = PReLU(shared_axes=[1, 2], name='PReLU2')(var_x)
        var_x = Conv2D(32, (3, 3), strides=1, padding='valid', name='conv3')(var_x)
        var_x = PReLU(shared_axes=[1, 2], name='PReLU3')(var_x)
        classifier = Conv2D(2, (1, 1), activation='softmax', name='conv4-1')(var_x)
        bbox_regress = Conv2D(4, (1, 1), name='conv4-2')(var_x)
        return [input_], [classifier, bbox_regress]

    def _calculate_scales(self,
                          minsize: int,
                          factor: float) -> list[float]:
        """ Calculate multi-scale

        Parameters
        ----------
        minsize: int
            Minimum size for a face to be accepted
        factor: float
            Scaling factor

        Returns
        -------
        list
            List of scale floats
        """
        factor_count = 0
        var_m = 12.0 / minsize
        minl = self._input_size * var_m
        # create scale pyramid
        scales = []
        while minl >= 12:
            scales += [var_m * np.power(factor, factor_count)]
            minl = minl * factor
            factor_count += 1
        logger.trace(scales)  # type:ignore
        return scales

    def __call__(self, images: np.ndarray) -> list[np.ndarray]:
        """ first stage - fast proposal network (p-net) to obtain face candidates

        Parameters
        ----------
        images: :class:`numpy.ndarray`
            The batch of images to detect faces in

        Returns
        -------
        List
            List of face candidates from P-Net
        """
        batch_size = images.shape[0]
        rectangles: list[list[list[int | float]]] = [[] for _ in range(batch_size)]
        scores: list[list[np.ndarray]] = [[] for _ in range(batch_size)]

        if self._pnet_input is None:
            self._pnet_input = [np.empty((batch_size, rheight, rwidth, 3), dtype="float32")
                                for rheight, rwidth in self._pnet_sizes]

        for scale, batch, (rheight, rwidth) in zip(self._pnet_scales,
                                                   self._pnet_input,
                                                   self._pnet_sizes):
            _ = [cv2.resize(images[idx], (rwidth, rheight), dst=batch[idx])
                 for idx in range(batch_size)]
            cls_prob, roi = self.predict(batch)
            cls_prob = cls_prob[..., 1]
            out_side = max(cls_prob.shape[1:3])
            cls_prob = np.swapaxes(cls_prob, 1, 2)
            roi = np.swapaxes(roi, 1, 3)
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

    def _detect_face_12net(self,
                           class_probabilities: np.ndarray,
                           roi: np.ndarray,
                           size: int,
                           scale: float) -> tuple[np.ndarray, np.ndarray]:
        """ Detect face position and calibrate bounding box on 12net feature map(matrix version)

        Parameters
        ----------
        class_probabilities: :class:`numpy.ndarray`
            softmax feature map for face classify
        roi: :class:`numpy.ndarray`
            feature map for regression
        size: int
            feature map's largest size
        scale: float
            current input image scale in multi-scales

        Returns
        -------
        list
            Calibrated face candidates
        """
        in_side = 2 * size + 11
        stride = 0. if size == 1 else float(in_side - 12) / (size - 1)
        (var_x, var_y) = np.nonzero(class_probabilities >= self._threshold)
        boundingbox = np.array([var_x, var_y]).T

        boundingbox = np.concatenate((np.fix((stride * (boundingbox) + 0) * scale),
                                      np.fix((stride * (boundingbox) + 11) * scale)), axis=1)
        offset = roi[:4, var_x, var_y].T
        boundingbox = boundingbox + offset * 12.0 * scale
        rectangles = np.concatenate((boundingbox,
                                     np.array([class_probabilities[var_x, var_y]]).T), axis=1)
        rectangles = rect2square(rectangles)

        np.clip(rectangles[..., :4], 0., self._input_size, out=rectangles[..., :4])
        pick = np.where(np.logical_and(rectangles[..., 2] > rectangles[..., 0],
                                       rectangles[..., 3] > rectangles[..., 1]))[0]
        rects = rectangles[pick, :4].astype("int")
        scores = rectangles[pick, 4]

        return nms(rects, scores, 0.3, "iou")


class RNet(KSession):
    """ Keras R-Net model Definition for MTCNN

    Parameters
    ----------
    model_path: str
        The path to the keras model file
    allow_growth: bool, optional
        Enable the Tensorflow GPU allow_growth configuration option. This option prevents
        Tensorflow from allocating all of the GPU VRAM, but can lead to higher fragmentation and
        slower performance. Default: ``False``
    exclude_gpus: list, optional
        A list of indices correlating to connected GPUs that Tensorflow should not use. Pass
        ``None`` to not exclude any GPUs. Default: ``None``
    cpu_mode: bool, optional
        ``True`` run the model on CPU. Default: ``False``
    input_size: int
        The input size of the model
    threshold: list, optional
        Threshold for R-Net

    """
    def __init__(self,
                 model_path: str,
                 allow_growth: bool,
                 exclude_gpus: list[int] | None,
                 cpu_mode: bool,
                 input_size: int,
                 threshold: float) -> None:
        super().__init__("MTCNN-RNet",
                         model_path,
                         allow_growth=allow_growth,
                         exclude_gpus=exclude_gpus,
                         cpu_mode=cpu_mode)
        self.define_model(self.model_definition)
        self.load_model_weights()

        self._input_size = input_size
        self._threshold = threshold

    @staticmethod
    def model_definition() -> tuple[list[Tensor], list[Tensor]]:
        """ Keras R-Network Definition for MTCNN """
        input_ = Input(shape=(24, 24, 3))
        var_x = Conv2D(28, (3, 3), strides=1, padding='valid', name='conv1')(input_)
        var_x = PReLU(shared_axes=[1, 2], name='prelu1')(var_x)
        var_x = MaxPool2D(pool_size=3, strides=2, padding='same')(var_x)

        var_x = Conv2D(48, (3, 3), strides=1, padding='valid', name='conv2')(var_x)
        var_x = PReLU(shared_axes=[1, 2], name='prelu2')(var_x)
        var_x = MaxPool2D(pool_size=3, strides=2)(var_x)

        var_x = Conv2D(64, (2, 2), strides=1, padding='valid', name='conv3')(var_x)
        var_x = PReLU(shared_axes=[1, 2], name='prelu3')(var_x)
        var_x = Permute((3, 2, 1))(var_x)
        var_x = Flatten()(var_x)
        var_x = Dense(128, name='conv4')(var_x)
        var_x = PReLU(name='prelu4')(var_x)
        classifier = Dense(2, activation='softmax', name='conv5-1')(var_x)
        bbox_regress = Dense(4, name='conv5-2')(var_x)
        return [input_], [classifier, bbox_regress]

    def __call__(self,
                 images: np.ndarray,
                 rectangle_batch: list[np.ndarray],
                 ) -> list[np.ndarray]:
        """ second stage - refinement of face candidates with r-net

        Parameters
        ----------
        images: :class:`numpy.ndarray`
            The batch of images to detect faces in
        rectangle_batch:
            List of :class:`numpy.ndarray` face candidates from P-Net

        Returns
        -------
        List
            List of :class:`numpy.ndarray` refined face candidates from R-Net
        """
        ret: list[np.ndarray] = []
        for idx, (rectangles, image) in enumerate(zip(rectangle_batch, images)):
            if not np.any(rectangles):
                ret.append(np.array([]))
                continue

            feed_batch = np.empty((rectangles.shape[0], 24, 24, 3), dtype="float32")

            _ = [cv2.resize(image[rect[1]: rect[3], rect[0]: rect[2]],
                            (24, 24),
                            dst=feed_batch[idx])
                 for idx, rect in enumerate(rectangles)]

            cls_prob, roi_prob = self.predict(feed_batch)
            ret.append(self._filter_face_24net(cls_prob, roi_prob, rectangles))
        return ret

    def _filter_face_24net(self,
                           class_probabilities: np.ndarray,
                           roi: np.ndarray,
                           rectangles: np.ndarray,
                           ) -> np.ndarray:
        """ Filter face position and calibrate bounding box on 12net's output

        Parameters
        ----------
        class_probabilities: class:`np.ndarray`
            Softmax feature map for face classify
        roi: :class:`numpy.ndarray`
            Feature map for regression
        rectangles: list
            12net's predict

        Returns
        -------
        list
            rectangles in the format [[x, y, x1, y1, score]]
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


class ONet(KSession):
    """ Keras O-Net model for MTCNN

    Parameters
    ----------
    model_path: str
        The path to the keras model file
    allow_growth: bool, optional
        Enable the Tensorflow GPU allow_growth configuration option. This option prevents
        Tensorflow from allocating all of the GPU VRAM, but can lead to higher fragmentation and
        slower performance. Default: ``False``
    exclude_gpus: list, optional
        A list of indices correlating to connected GPUs that Tensorflow should not use. Pass
        ``None`` to not exclude any GPUs. Default: ``None``
    cpu_mode: bool, optional
        ``True`` run the model on CPU. Default: ``False``
    input_size: int
        The input size of the model
    threshold: list, optional
        Threshold for O-Net
    """
    def __init__(self,
                 model_path: str,
                 allow_growth: bool,
                 exclude_gpus: list[int] | None,
                 cpu_mode: bool,
                 input_size: int,
                 threshold: float) -> None:
        super().__init__("MTCNN-ONet",
                         model_path,
                         allow_growth=allow_growth,
                         exclude_gpus=exclude_gpus,
                         cpu_mode=cpu_mode)
        self.define_model(self.model_definition)
        self.load_model_weights()

        self._input_size = input_size
        self._threshold = threshold

    @staticmethod
    def model_definition() -> tuple[list[Tensor], list[Tensor]]:
        """ Keras O-Network for MTCNN """
        input_ = Input(shape=(48, 48, 3))
        var_x = Conv2D(32, (3, 3), strides=1, padding='valid', name='conv1')(input_)
        var_x = PReLU(shared_axes=[1, 2], name='prelu1')(var_x)
        var_x = MaxPool2D(pool_size=3, strides=2, padding='same')(var_x)
        var_x = Conv2D(64, (3, 3), strides=1, padding='valid', name='conv2')(var_x)
        var_x = PReLU(shared_axes=[1, 2], name='prelu2')(var_x)
        var_x = MaxPool2D(pool_size=3, strides=2)(var_x)
        var_x = Conv2D(64, (3, 3), strides=1, padding='valid', name='conv3')(var_x)
        var_x = PReLU(shared_axes=[1, 2], name='prelu3')(var_x)
        var_x = MaxPool2D(pool_size=2)(var_x)
        var_x = Conv2D(128, (2, 2), strides=1, padding='valid', name='conv4')(var_x)
        var_x = PReLU(shared_axes=[1, 2], name='prelu4')(var_x)
        var_x = Permute((3, 2, 1))(var_x)
        var_x = Flatten()(var_x)
        var_x = Dense(256, name='conv5')(var_x)
        var_x = PReLU(name='prelu5')(var_x)

        classifier = Dense(2, activation='softmax', name='conv6-1')(var_x)
        bbox_regress = Dense(4, name='conv6-2')(var_x)
        landmark_regress = Dense(10, name='conv6-3')(var_x)
        return [input_], [classifier, bbox_regress, landmark_regress]

    def __call__(self,
                 images: np.ndarray,
                 rectangle_batch: list[np.ndarray]
                 ) -> list[tuple[np.ndarray, np.ndarray]]:
        """ Third stage - further refinement and facial landmarks positions with o-net

        Parameters
        ----------
        images: :class:`numpy.ndarray`
            The batch of images to detect faces in
        rectangle_batch:
            List of :class:`numpy.ndarray` face candidates from R-Net

        Returns
        -------
        List
            List of refined final candidates, scores and landmark points from O-Net
        """
        ret: list[tuple[np.ndarray, np.ndarray]] = []
        for idx, rectangles in enumerate(rectangle_batch):
            if not np.any(rectangles):
                ret.append((np.empty((0, 5)), np.empty(0)))
                continue
            image = images[idx]
            feed_batch = np.empty((rectangles.shape[0], 48, 48, 3), dtype="float32")

            _ = [cv2.resize(image[rect[1]: rect[3], rect[0]: rect[2]],
                            (48, 48),
                            dst=feed_batch[idx])
                 for idx, rect in enumerate(rectangles)]

            cls_probs, roi_probs, pts_probs = self.predict(feed_batch)
            ret.append(self._filter_face_48net(cls_probs, roi_probs, pts_probs, rectangles))
        return ret

    def _filter_face_48net(self, class_probabilities: np.ndarray,
                           roi: np.ndarray,
                           points: np.ndarray,
                           rectangles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """ Filter face position and calibrate bounding box on 12net's output

        Parameters
        ----------
        class_probabilities: :class:`numpy.ndarray`  : class_probabilities[1] is face possibility
            Array of face probabilities
        roi: :class:`numpy.ndarray`
            offset
        points: :class:`numpy.ndarray`
            5 point face landmark
        rectangles: :class:`numpy.ndarray`
            12net's predict, rectangles[i][0:3] is the position, rectangles[i][4] is score

        Returns
        -------
        boxes: :class:`numpy.ndarray`
            The [l, t, r, b, score] bounding boxes
        points: :class:`numpy.ndarray`
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


class MTCNN():  # pylint:disable=too-few-public-methods
    """ MTCNN Detector for face alignment

    Parameters
    ----------
    model_path: list
        List of paths to the 3 MTCNN subnet weights
    allow_growth: bool, optional
        Enable the Tensorflow GPU allow_growth configuration option. This option prevents
        Tensorflow from allocating all of the GPU VRAM, but can lead to higher fragmentation and
        slower performance. Default: ``False``
    exclude_gpus: list, optional
        A list of indices correlating to connected GPUs that Tensorflow should not use. Pass
        ``None`` to not exclude any GPUs. Default: ``None``
    cpu_mode: bool, optional
        ``True`` run the model on CPU. Default: ``False``
    input_size: int, optional
        The height, width input size to the model. Default: 640
    minsize: int, optional
        The minimum size of a face to accept as a detection. Default: `20`
    threshold: list, optional
        List of floats for the three steps, Default: `[0.6, 0.7, 0.7]`
    factor: float, optional
        The factor used to create a scaling pyramid of face sizes to detect in the image.
        Default: `0.709`
    """
    def __init__(self,
                 model_path: list[str],
                 allow_growth: bool,
                 exclude_gpus: list[int] | None,
                 cpu_mode: bool,
                 input_size: int = 640,
                 minsize: int = 20,
                 threshold: list[float] | None = None,
                 factor: float = 0.709) -> None:
        logger.debug("Initializing: %s: (model_path: '%s', allow_growth: %s, exclude_gpus: %s, "
                     "input_size: %s, minsize: %s, threshold: %s, factor: %s)",
                     self.__class__.__name__, model_path, allow_growth, exclude_gpus,
                     input_size, minsize, threshold, factor)

        threshold = [0.6, 0.7, 0.7] if threshold is None else threshold
        self._pnet = PNet(model_path[0],
                          allow_growth,
                          exclude_gpus,
                          cpu_mode,
                          input_size,
                          minsize,
                          factor,
                          threshold[0])
        self._rnet = RNet(model_path[1],
                          allow_growth,
                          exclude_gpus,
                          cpu_mode,
                          input_size,
                          threshold[1])
        self._onet = ONet(model_path[2],
                          allow_growth,
                          exclude_gpus,
                          cpu_mode,
                          input_size,
                          threshold[2])

        logger.debug("Initialized: %s", self.__class__.__name__)

    def detect_faces(self, batch: np.ndarray) -> tuple[np.ndarray, tuple[np.ndarray]]:
        """Detects faces in an image, and returns bounding boxes and points for them.

        Parameters
        ----------
        batch: :class:`numpy.ndarray`
            The input batch of images to detect face in

        Returns
        -------
        List
            list of numpy arrays containing the bounding box and 5 point landmarks
            of detected faces
        """
        rectangles = self._pnet(batch)
        rectangles = self._rnet(batch, rectangles)

        ret_boxes, ret_points = zip(*self._onet(batch, rectangles))
        return np.array(ret_boxes, dtype="object"), ret_points


def nms(rectangles: np.ndarray,
        scores: np.ndarray,
        threshold: float,
        method: str = "iom") -> tuple[np.ndarray, np.ndarray]:
    """ apply non-maximum suppression on ROIs in same scale(matrix version)

    Parameters
    ----------
    rectangles: :class:`np.ndarray`
        The [b, l, t, r, b] bounding box detection candidates
    threshold: float
        Threshold for succesful match
    method: str, optional
        "iom" method or default. Defalt: "iom"

    Returns
    -------
    rectangles: :class:`np.ndarray`
        The [b, l, t, r, b] bounding boxes
    scores :class:`np.ndarray`
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
    """ change rectangles into squares (matrix version)

    Parameters
    ----------
    rectangles: :class:`numpy.ndarray`
        [b, x, y, x1, y1] rectangles

    Return
    ------
    list
        Original rectangle changed to a square
    """
    width = rectangles[:, 2] - rectangles[:, 0]
    height = rectangles[:, 3] - rectangles[:, 1]
    length = np.maximum(width, height).T
    rectangles[:, 0] = rectangles[:, 0] + width * 0.5 - length * 0.5
    rectangles[:, 1] = rectangles[:, 1] + height * 0.5 - length * 0.5
    rectangles[:, 2:4] = rectangles[:, 0:2] + np.repeat([length], 2, axis=0).T
    return rectangles
