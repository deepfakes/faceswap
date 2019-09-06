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
        super().__init__(git_model_id=git_model_id,
                         model_filename=model_filename,
                         colorspace="RGB",
                         input_size=128,
                         **kwargs)
        self.vram = 0  # Doesn't use GPU
        self.model = None

    def initialize(self, *args, **kwargs):
        """ Initialization tasks to run prior to alignments """
        try:
            super().initialize(*args, **kwargs)
            logger.info("Initializing cv2 DNN Aligner...")
            logger.debug("cv2 DNN initialize: (args: %s kwargs: %s)", args, kwargs)
            logger.verbose("Using CPU for alignment")

            self.model = cv2.dnn.readNetFromTensorflow(  # pylint: disable=no-member
                self.model_path)
            self.model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)  # pylint: disable=no-member
            self.init.set()
            logger.info("Initialized cv2 DNN Aligner.")
        except Exception as err:
            self.error.set()
            raise err

    def align_image(self, detected_face, image):
        """ Align the incoming image for prediction """
        logger.trace("Aligning image around center")

        box = (detected_face["left"],
               detected_face["top"],
               detected_face["right"],
               detected_face["bottom"])
        height = detected_face["bottom"] - detected_face["top"]
        width = detected_face["right"] - detected_face["left"]
        diff_height_width = height - width
        offset_y = int(abs(diff_height_width / 2))
        box_moved = self.move_box(box, [0, offset_y])

        # Make box square.
        roi = self.get_square_box(box_moved)
        # Pad the image if face is outside of boundaries
        image = self.pad_image(roi, image)
        face = image[roi[1]: roi[3], roi[0]: roi[2]]

        if face.shape[0] < self.input_size:
            interpolation = cv2.INTER_CUBIC  # pylint:disable=no-member
        else:
            interpolation = cv2.INTER_AREA  # pylint:disable=no-member

        face = cv2.resize(face,  # pylint:disable=no-member
                          dsize=(int(self.input_size), int(self.input_size)),
                          interpolation=interpolation)
        return dict(image=face, roi=roi)

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

        # Shift the box if any points fall below zero
        if left < 0:
            right += abs(left)
            left += abs(left)
        if top < 0:
            bottom += abs(top)
            top += abs(top)

        # Make sure box is always square.
        assert ((right - left) == (bottom - top)), 'Box is not square.'

        return [left, top, right, bottom]

    @staticmethod
    def pad_image(box, image):
        """Pad image if facebox falls outside of boundaries """
        width, height = image.shape[:2]
        pad_l = 1 - box[0] if box[0] < 0 else 0
        pad_t = 1 - box[1] if box[1] < 0 else 0
        pad_r = box[2] - width if box[2] > width else 0
        pad_b = box[3] - height if box[3] > height else 0
        logger.trace("Padding: (l: %s, t: %s, r: %s, b: %s)", pad_l, pad_t, pad_r, pad_b)
        retval = cv2.copyMakeBorder(image.copy(),  # pylint: disable=no-member
                                    pad_t,
                                    pad_b,
                                    pad_l,
                                    pad_r,
                                    cv2.BORDER_CONSTANT,  # pylint: disable=no-member
                                    value=(0, 0, 0))
        logger.trace("Padded shape: %s", retval.shape)
        return retval

    def predict_landmarks(self, feed_dict):
        """ Predict the 68 point landmarks """
        logger.trace("Predicting Landmarks")
        image = np.expand_dims(np.transpose(feed_dict["image"], (2, 0, 1)), 0).astype("float32")
        self.model.setInput(image)
        prediction = self.model.forward()
        pts_img = self.get_pts_from_predict(prediction, feed_dict["roi"])
        return pts_img

    @staticmethod
    def get_pts_from_predict(prediction, roi):
        """ Get points from predictor """
        logger.trace("Obtain points from prediction")
        points = np.array(prediction).flatten()
        points = np.reshape(points, (-1, 2))
        points *= (roi[2] - roi[0])
        points[:, 0] += roi[0]
        points[:, 1] += roi[1]
        retval = np.rint(points).astype("uint").tolist()
        logger.trace("Predicted Landmarks: %s", retval)
        return retval
