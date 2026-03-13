#!/usr/bin/env python3
"""OpenCV DNN Face detection plugin"""
import logging

import cv2
import numpy as np

from lib.utils import get_module_objects, GetModel
from plugins.extract.base import ExtractPlugin
from . import cv2_dnn_defaults as cfg


logger = logging.getLogger(__name__)


class CV2DNNDetect(ExtractPlugin):
    """CV2 DNN detector for face recognition"""
    def __init__(self) -> None:
        super().__init__(input_size=300,
                         batch_size=1,
                         is_rgb=False,
                         dtype="float32",
                         scale=(0, 255))
        self.model: cv2.dnn.Net
        self.confidence = cfg.confidence() / 100
        self._average_image = np.array([104, 117, 123], dtype="float32")

    def load_model(self) -> cv2.dnn.Net:
        """Load the CV2 DNN Detector Model

        Returns
        -------
        The loaded cv2-DNN model
        """
        weights = GetModel(model_filename=["resnet_ssd_v1.caffemodel", "resnet_ssd_v1.prototxt"],
                           git_model_id=4)
        model_path = weights.model_path
        assert isinstance(model_path, list)
        model = cv2.dnn.readNetFromCaffe(model_path[1], model_path[0])
        model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        return model

    def pre_process(self, batch: np.ndarray) -> np.ndarray:
        """Compile the detection image(s) for prediction

        Parameters
        ----------
        batch
            The input batch of images at model input size in the correct color order

        Returns
        -------
        The batch of images ready for feeding the model
        """
        return (batch - self._average_image).transpose(0, 3, 1, 2)

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
        self.model.setInput(batch)
        result = self.model.forward()
        return result.reshape(batch.shape[0], 200, 7)

    def post_process(self, batch: np.ndarray) -> np.ndarray:
        """Compile found faces for output

        Parameters
        ----------
        batch
            The detection results for the model

        Returns
        -------
        The processed detection bounding box from the model at model input size
        """
        confidence_mask = batch[..., 2] >= self.confidence
        boxes = [batch[b, ..., 3:7][confidence_mask[b]] * self.input_size
                 for b in range(batch.shape[0])]
        return np.array(boxes, dtype="object")


__all__ = get_module_objects(__name__)
