#!/usr/bin/env python3
""" Components Mask for faceswap.py """

import cv2
import numpy as np
from ._base import Masker, logger


class Mask(Masker):
    """ Perform transformation to align and get landmarks """
    def __init__(self, **kwargs):
        git_model_id = None
        model_filename = None
        super().__init__(git_model_id=git_model_id, model_filename=model_filename, **kwargs)
        self.input_size = 256
        self.name = "Components"
        self.vram = 0  # Doesn't use GPU
        self.vram_per_batch = 0
        self.batchsize = 1

    def init_model(self):
        logger.debug("No mask model to initialize")

    def process_input(self, batch):
        """ Compile the detected faces for prediction """
        batch["feed"] = np.zeros((self.batchsize, self.input_size, self.input_size, 1),
                                 dtype="float32")
        return batch

    def predict(self, batch):
        """ Run model to get predictions """
        for mask, face in zip(batch["feed"], batch["detected_faces"]):
            parts = self.parse_parts(np.array(face.feed_landmarks))
            for item in parts:
                item = np.rint(np.concatenate(item)).astype("int32")
                hull = cv2.convexHull(item)
                cv2.fillConvexPoly(mask, hull, 1.0, lineType=cv2.LINE_AA)
        batch["prediction"] = batch["feed"]
        return batch

    def process_output(self, batch):
        """ Compile found faces for output """
        return batch

    @staticmethod
    def parse_parts(landmarks):
        """ Component face hull mask """
        r_jaw = (landmarks[0:9], landmarks[17:18])
        l_jaw = (landmarks[8:17], landmarks[26:27])
        r_cheek = (landmarks[17:20], landmarks[8:9])
        l_cheek = (landmarks[24:27], landmarks[8:9])
        nose_ridge = (landmarks[19:25], landmarks[8:9],)
        r_eye = (landmarks[17:22],
                 landmarks[27:28],
                 landmarks[31:36],
                 landmarks[8:9])
        l_eye = (landmarks[22:27],
                 landmarks[27:28],
                 landmarks[31:36],
                 landmarks[8:9])
        nose = (landmarks[27:31], landmarks[31:36])
        parts = [r_jaw, l_jaw, r_cheek, l_cheek, nose_ridge, r_eye, l_eye, nose]
        return parts
