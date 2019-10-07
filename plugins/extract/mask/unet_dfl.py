#!/usr/bin/env python3

import cv2
import keras
import numpy as np
from lib.model.session import KSession
from ._base import Masker, logger


class Mask(Masker):
    """ Perform transformation to align and get landmarks """
    def __init__(self, **kwargs):
        git_model_id = 6
        model_filename = "DFL_256_sigmoid_v1.h5"
        super().__init__(git_model_id=git_model_id, model_filename=model_filename, **kwargs)
        self.name = "U-Net Mask Network(256)"
        self.mask_in_size = 256
        self.colorformat = "BGR"
        self.vram = 3440
        self.vram_warnings = 1024  # TODO determine
        self.vram_per_batch = 64  # TODO determine
        self.batchsize = self.config["batch-size"]

    def init_model(self):
        self.model = KSession(self.name, self.model_path, model_kwargs=dict())
        self.model.load_model()
        self.input = np.zeros((self.batchsize, self.mask_in_size, self.mask_in_size, 3),
                              dtype="float32")
        self.model.predict(self.input)

    def process_input(self, batch):
        """ Compile the detected faces for prediction """
        for index, face in enumerate(batch["detected_faces"]):
            face.load_aligned(face.image,
                              size=self.mask_in_size,
                              dtype='float32')
            self.input[index] = face.aligned["face"][..., :3]
        batch["feed"] = self.input / 255.
        return batch

    def predict(self, batch):
        """ Run model to get predictions """
        predictions = self.model.predict(batch["feed"])
        batch["prediction"] = predictions * 255.
        return batch

    def process_output(self, batch):
        """ Compile found faces for output """
        for idx, (face, predicts) in enumerate(zip(batch["detected_faces"], batch["prediction"])):
            generator = (cv2.GaussianBlur(mask, (7, 7), 0) for mask in predicts)
            predicted = np.array(tuple(generator))
            predicted[predicted < 10.] = 0.
            predicted[predicted > 245.] = 255.

            face.load_feed_face(face.image,
                                size=self.input_size,
                                coverage_ratio=self.coverage_ratio)
            feed_face = face.feed["face"][..., :3]
            feed_mask = self._resize(predicted, self.input_size).astype('uint8')
            batch["detected_faces"][idx].feed["face"] = np.concatenate((feed_face,
                                                                        feed_mask),
                                                                       axis=-1)
            face.load_reference_face(face.image,
                                     size=self.output_size,
                                     coverage_ratio=self.coverage_ratio)
            ref_face = face.reference["face"][..., :3]
            ref_mask = self._resize(predicted, self.output_size).astype('uint8')
            batch["detected_faces"][idx].reference["face"] = np.concatenate((ref_face,
                                                                             ref_mask),
                                                                            axis=-1)
        return batch
