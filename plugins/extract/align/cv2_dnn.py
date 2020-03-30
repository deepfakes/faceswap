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

import cv2
import numpy as np

from ._base import Aligner, logger


class Align(Aligner):
    """ Perform transformation to align and get landmarks """
    def __init__(self, **kwargs):
        git_model_id = 1
        model_filename = "cnn-facial-landmark_v1.pb"
        super().__init__(git_model_id=git_model_id, model_filename=model_filename, **kwargs)

        self.name = "cv2-DNN Aligner"
        self.input_size = 128
        self.color_format = "RGB"
        self.vram = 0  # Doesn't use GPU
        self.vram_per_batch = 0
        self.batchsize = 1

    def init_model(self):
        """ Initialize CV2 DNN Detector Model"""
        self.model = cv2.dnn.readNetFromTensorflow(self.model_path)  # pylint: disable=no-member
        self.model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)  # pylint: disable=no-member

    def process_input(self, batch):
        """ Compile the detected faces for prediction """
        faces, batch["roi"], batch["offsets"] = self.align_image(batch)
        faces = self._normalize_faces(faces)
        batch["feed"] = np.array(faces, dtype="float32")[..., :3].transpose((0, 3, 1, 2))
        return batch

    def align_image(self, batch):
        """ Align the incoming image for prediction """
        logger.trace("Aligning image around center")
        sizes = (self.input_size, self.input_size)
        rois = []
        faces = []
        offsets = []
        for det_face, image in zip(batch["detected_faces"], batch["image"]):
            box = (det_face.left,
                   det_face.top,
                   det_face.right,
                   det_face.bottom)
            diff_height_width = det_face.h - det_face.w
            offset_y = int(abs(diff_height_width / 2))
            box_moved = self.move_box(box, [0, offset_y])
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

    @staticmethod
    def move_box(box, offset):
        """Move the box to direction specified by vector offset"""
        left = box[0] + offset[0]
        top = box[1] + offset[1]
        right = box[2] + offset[0]
        bottom = box[3] + offset[1]
        return [left, top, right, bottom]

    @staticmethod
    def get_square_box(box):
        """Get a square box out of the given box, by expanding it."""
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

    @staticmethod
    def pad_image(box, image):
        """Pad image if face-box falls outside of boundaries """
        height, width = image.shape[:2]
        pad_l = 1 - box[0] if box[0] < 0 else 0
        pad_t = 1 - box[1] if box[1] < 0 else 0
        pad_r = box[2] - width if box[2] > width else 0
        pad_b = box[3] - height if box[3] > height else 0
        logger.trace("Padding: (l: %s, t: %s, r: %s, b: %s)", pad_l, pad_t, pad_r, pad_b)
        padded_image = cv2.copyMakeBorder(image.copy(),
                                          pad_t,
                                          pad_b,
                                          pad_l,
                                          pad_r,
                                          cv2.BORDER_CONSTANT,
                                          value=(0, 0, 0))
        offsets = (pad_l - pad_r, pad_t - pad_b)
        logger.trace("image_shape: %s, Padded shape: %s, box: %s, offsets: %s",
                     image.shape, padded_image.shape, box, offsets)
        return padded_image, offsets

    def predict(self, batch):
        """ Predict the 68 point landmarks """
        logger.trace("Predicting Landmarks")
        self.model.setInput(batch["feed"])
        batch["prediction"] = self.model.forward()
        return batch

    def process_output(self, batch):
        """ Process the output from the model """
        self.get_pts_from_predict(batch)
        return batch

    @staticmethod
    def get_pts_from_predict(batch):
        """ Get points from predictor """
        for prediction, roi, offset in zip(batch["prediction"], batch["roi"], batch["offsets"]):
            points = np.reshape(prediction, (-1, 2))
            points *= (roi[2] - roi[0])
            points[:, 0] += (roi[0] - offset[0])
            points[:, 1] += (roi[1] - offset[1])
            batch.setdefault("landmarks", []).append(points)
        logger.trace("Predicted Landmarks: %s", batch["landmarks"])
