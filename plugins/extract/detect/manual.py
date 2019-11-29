#!/usr/bin/env python3
""" Manual face detection plugin """

import numpy as np
from ._base import Detector


class Detect(Detector):
    """ Manual Detector """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Manual"
        self.input_size = 1440  # Arbitrary size for manual tool
        self.vram = 0
        self.vram_warnings = 0
        self.vram_per_batch = 1
        self.batchsize = 1

    def _compile_detection_image(self, input_image):
        """ Override compile detection image for manual. No face is actually fed into a model """
        return input_image, 1, (0, 0)

    def init_model(self):
        """ No model for Manual """
        return

    def process_input(self, batch):
        """ No pre-processing for Manual. Just set a dummy feed """
        batch["feed"] = batch["image"]
        return batch

    def predict(self, batch):
        """ No prediction for Manual """
        batch["prediction"] = [np.array(batch["manual_face"])]
        return batch

    def process_output(self, batch):
        """ Post process the detected faces """
        return batch
