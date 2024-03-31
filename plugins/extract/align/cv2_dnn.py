#!/usr/bin/env python3
""" CV2 DNN landmarks extractor for faceswap.py
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
import typing as T

import cv2
import numpy as np

from ._base import Aligner, AlignerBatch, BatchType

if T.TYPE_CHECKING:
    from lib.align.detected_face import DetectedFace

logger = logging.getLogger(__name__)


class Align(Aligner):
    """ Perform transformation to align and get landmarks """
    def __init__(self, **kwargs) -> None:
        git_model_id = 1
        model_filename = "cnn-facial-landmark_v1.pb"
        super().__init__(git_model_id=git_model_id, model_filename=model_filename, **kwargs)

        self.model: cv2.dnn.Net
        self.model_path: str
        self.name = "cv2-DNN Aligner"
        self.input_size = 128
        self.color_format = "RGB"
        self.vram = 0  # Doesn't use GPU
        self.vram_per_batch = 0
        self.batchsize = 1
        self.realign_centering = "legacy"

    def init_model(self) -> None:
        """ Initialize CV2 DNN Detector Model"""
        self.model = cv2.dnn.readNetFromTensorflow(self.model_path)
        self.model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def faces_to_feed(self, faces: np.ndarray) -> np.ndarray:
        """ Convert a batch of face images from UINT8 (0-255) to fp32 (0.0-255.0)

        Parameters
        ----------
        faces: :class:`numpy.ndarray`
            The batch of faces in UINT8 format

        Returns
        -------
        class: `numpy.ndarray`
            The batch of faces as fp32
        """
        return faces.astype("float32").transpose((0, 3, 1, 2))

    def process_input(self, batch: BatchType) -> None:
        """ Compile the detected faces for prediction

        Parameters
        ----------
        batch: :class:`AlignerBatch`
            The current batch to process input for

        Returns
        -------
        :class:`AlignerBatch`
            The batch item with the :attr:`feed` populated and any required :attr:`data` added
        """
        assert isinstance(batch, AlignerBatch)
        lfaces, roi, offsets = self.align_image(batch)
        batch.feed = np.array(lfaces)[..., :3]
        batch.data.append({"roi": roi, "offsets": offsets})

    def _get_box_and_offset(self, face: DetectedFace) -> tuple[list[int], int]:
        """Obtain the bounding box and offset from a detected face.


        Parameters
        ----------
        face: :class:`~lib.align.DetectedFace`
            The detected face object to obtain the bounding box and offset from

        Returns
        -------
        box: list
            The [left, top, right, bottom] bounding box
        offset: int
            The offset of the box (difference between half width vs height)
        """

        box = T.cast(list[int], [face.left,
                                 face.top,
                                 face.right,
                                 face.bottom])
        diff_height_width = T.cast(int, face.height) - T.cast(int, face.width)
        offset = int(abs(diff_height_width / 2))
        return box, offset

    def align_image(self, batch: AlignerBatch) -> tuple[list[np.ndarray],
                                                        list[list[int]],
                                                        list[tuple[int, int]]]:
        """ Align the incoming image for prediction

        Parameters
        ----------
        batch: :class:`AlignerBatch`
            The current batch to align the input for

        Returns
        -------
        faces: list
            List of feed faces for the aligner
        rois: list
            List of roi's for the faces
        offsets: list
            List of offsets for the faces
        """
        logger.trace("Aligning image around center")  # type:ignore[attr-defined]
        sizes = (self.input_size, self.input_size)
        rois = []
        faces = []
        offsets = []
        for det_face, image in zip(batch.detected_faces, batch.image):
            box, offset_y = self._get_box_and_offset(det_face)
            box_moved = self.move_box(box, (0, offset_y))
            # Make box square.
            roi = self.get_square_box(box_moved)

            # Pad the image and adjust roi if face is outside of boundaries
            image, offset = self.pad_image(roi, image)
            face = image[roi[1] + abs(offset[1]): roi[3] + abs(offset[1]),
                         roi[0] + abs(offset[0]): roi[2] + abs(offset[0])]
            interpolation = cv2.INTER_CUBIC if face.shape[0] < self.input_size else cv2.INTER_AREA
            face = cv2.resize(face, dsize=sizes, interpolation=interpolation)
            faces.append(face)
            rois.append(roi)
            offsets.append(offset)
        return faces, rois, offsets

    @classmethod
    def move_box(cls,
                 box: list[int],
                 offset: tuple[int, int]) -> list[int]:
        """Move the box to direction specified by vector offset

        Parameters
        ----------
        box: list
            The (`left`, `top`, `right`, `bottom`) box positions
        offset: tuple
            (x, y) offset to move the box

        Returns
        -------
        list
            The original box shifted by the offset
        """
        left = box[0] + offset[0]
        top = box[1] + offset[1]
        right = box[2] + offset[0]
        bottom = box[3] + offset[1]
        return [left, top, right, bottom]

    @staticmethod
    def get_square_box(box: list[int]) -> list[int]:
        """Get a square box out of the given box, by expanding it.

        Parameters
        ----------
        box: list
            The (`left`, `top`, `right`, `bottom`) box positions

        Returns
        -------
        list
            The original box but made square
        """
        left = box[0]
        top = box[1]
        right = box[2]
        bottom = box[3]

        box_width = right - left
        box_height = bottom - top

        # Check if box is already a square. If not, make it a square.
        diff = box_height - box_width
        delta = int(abs(diff) / 2)

        if diff == 0:                   # Already a square.
            return box
        if diff > 0:                    # Height > width, a slim box.
            left -= delta
            right += delta
            if diff % 2 == 1:
                right += 1
        else:                           # Width > height, a short box.
            top -= delta
            bottom += delta
            if diff % 2 == 1:
                bottom += 1

        # Make sure box is always square.
        assert ((right - left) == (bottom - top)), 'Box is not square.'

        return [left, top, right, bottom]

    @classmethod
    def pad_image(cls, box: list[int], image: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
        """Pad image if face-box falls outside of boundaries

        Parameters
        ----------
        box: list
            The (`left`, `top`, `right`, `bottom`) roi box positions
        image: :class:`numpy.ndarray`
            The image to be padded

        Returns
        -------
        :class:`numpy.ndarray`
            The padded image
        """
        height, width = image.shape[:2]
        pad_l = 1 - box[0] if box[0] < 0 else 0
        pad_t = 1 - box[1] if box[1] < 0 else 0
        pad_r = box[2] - width if box[2] > width else 0
        pad_b = box[3] - height if box[3] > height else 0
        logger.trace("Padding: (l: %s, t: %s, r: %s, b: %s)",  # type:ignore[attr-defined]
                     pad_l, pad_t, pad_r, pad_b)
        padded_image = cv2.copyMakeBorder(image.copy(),
                                          pad_t,
                                          pad_b,
                                          pad_l,
                                          pad_r,
                                          cv2.BORDER_CONSTANT,
                                          value=(0, 0, 0))
        offsets = (pad_l - pad_r, pad_t - pad_b)
        logger.trace("image_shape: %s, Padded shape: %s, box: %s, "  # type:ignore[attr-defined]
                     "offsets: %s",
                     image.shape, padded_image.shape, box, offsets)
        return padded_image, offsets

    def predict(self, feed: np.ndarray) -> np.ndarray:
        """ Predict the 68 point landmarks

        Parameters
        ----------
        feed: :class:`numpy.ndarray`
            The batch to feed into the aligner

        Returns
        -------
        :class:`numpy.ndarray`
            The predictions from the aligner
        """
        assert isinstance(self.model, cv2.dnn.Net)
        self.model.setInput(feed)
        retval = self.model.forward()
        return retval

    def process_output(self, batch: BatchType) -> None:
        """ Process the output from the model

        Parameters
        ----------
        batch: :class:`AlignerBatch`
            The current batch from the model with :attr:`predictions` populated
        """
        assert isinstance(batch, AlignerBatch)
        self.get_pts_from_predict(batch)

    def get_pts_from_predict(self, batch: AlignerBatch):
        """ Get points from predictor and populates the :attr:`landmarks` property

        Parameters
        ----------
        batch: :class:`AlignerBatch`
            The current batch from the model with :attr:`predictions` populated
        """
        landmarks = []
        if batch.second_pass:
            batch.landmarks = batch.prediction.reshape(self.batchsize, -1, 2) * self.input_size
        else:
            for prediction, roi, offset in zip(batch.prediction,
                                               batch.data[0]["roi"],
                                               batch.data[0]["offsets"]):
                points = np.reshape(prediction, (-1, 2))
                points *= (roi[2] - roi[0])
                points[:, 0] += (roi[0] - offset[0])
                points[:, 1] += (roi[1] - offset[1])
                landmarks.append(points)
            batch.landmarks = np.array(landmarks)
        logger.trace("Predicted Landmarks: %s", batch.landmarks)  # type:ignore[attr-defined]
