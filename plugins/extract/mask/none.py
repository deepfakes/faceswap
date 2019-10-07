#!/usr/bin/env python3

import numpy as np
from ._base import Masker, logger


class Mask(Masker):
    """ Perform transformation to align and get landmarks """
    def __init__(self, **kwargs):
        git_model_id = None
        model_filename = None
        super().__init__(git_model_id=git_model_id, model_filename=model_filename, **kwargs)
        self.name = "without a mask"
        self.colorformat = "BGR"
        self.vram = 0
        self.vram_warnings = 0
        self.vram_per_batch = 30
        self.batchsize = self.config["batch-size"]

    def init_model(self):
        logger.debug("No mask model to initialize")

    def process_input(self, batch):
        """ Compile the detected faces for prediction """
        batch["feed"] = np.array([face.image for face in batch["detected_faces"]])
        return batch

    def predict(self, batch):
        """ Run model to get predictions """
        batch["prediction"] = np.full(batch["feed"].shape[:-1] + (1,),
                                      fill_value=255,
                                      dtype='uint8')
        return batch

    def process_output(self, batch):
        """ Compile found faces for output """
        generator = zip(batch["feed"], batch["detected_faces"], batch["prediction"])
        for feed, face, prediction in generator:
            face.image = np.concatenate((feed, prediction), axis=-1)
            face.load_feed_face(face.image,
                                size=self.input_size,
                                coverage_ratio=self.coverage_ratio)
            face.load_reference_face(face.image,
                                     size=self.output_size,
                                     coverage_ratio=self.coverage_ratio)
        return batch
