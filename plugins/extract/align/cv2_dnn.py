#!/usr/bin/env python3
"""CV2 DNN landmarks extractor for faceswap.py
Adapted from: https://github.com/yinguobing/cnn-facial-landmark
MIT License

Copyright (c) 2017 Yin Guobing

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
import logging

import cv2
import numpy as np

from lib.utils import get_module_objects, GetModel
from plugins.extract.base import ExtractPlugin

logger = logging.getLogger(__name__)


class CV2DNNAlign(ExtractPlugin):
    """CV2 DNN Plugin for face alignment """
    def __init__(self) -> None:
        # pylint:disable=duplicate-code
        super().__init__(input_size=128,
                         batch_size=1,
                         is_rgb=True,
                         dtype="float32",
                         scale=(0, 255))
        self.model: cv2.dnn.Net

    def load_model(self) -> cv2.dnn.Net:
        """Load the CV2 DNN Aligner Model

        Returns
        -------
        The loaded cv2-DNN model
        """
        weights = GetModel(model_filename="cnn-facial-landmark_v1.pb", git_model_id=1)
        model_path = weights.model_path
        assert isinstance(model_path, str)
        model = cv2.dnn.readNetFromTensorflow(model_path)
        model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
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
        heights = batch[..., 3] - batch[..., 1]
        widths = batch[..., 2] - batch[..., 0]

        diff_height_width = widths - heights
        offset = np.abs(diff_height_width // 2)
        batch[:, [1, 3]] += offset[:, None]

        cx = (batch[:, 0] + batch[:, 2]) // 2
        cy = (batch[:, 1] + batch[:, 3]) // 2

        size = np.maximum(widths, heights)
        half = size // 2

        retval = batch.copy()
        retval[:, 0] = cx - half
        retval[:, 1] = cy - half
        retval[:, 2] = retval[:, 0] + size
        retval[:, 3] = retval[:, 1] + size
        return retval

    def process(self, batch: np.ndarray) -> np.ndarray:
        """Predict the 68 point landmarks

        Parameters
        ----------
        feed
            The batch to feed into the aligner

        Returns
        -------
        The predictions from the aligner
        """
        self.model.setInput(batch.transpose((0, 3, 1, 2)))
        return self.model.forward().reshape(batch.shape[0], -1, 2)


__all__ = get_module_objects(__name__)
