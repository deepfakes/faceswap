#!/usr/bin/env python3
""" Dummy empty Mask for faceswap.py """

import numpy as np
from ._base import Masker, logger


class Mask(Masker):
    """ Perform transformation to align and get landmarks """
    def __init__(self, **kwargs):
        git_model_id = None
        model_filename = None
        super().__init__(git_model_id=git_model_id, model_filename=model_filename, **kwargs)
        self.input_size = 256
        self.blur_kernel = None
        self.name = "None"
        self.vram = 0
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
        batch["prediction"] = np.ones_like(batch["feed"], dtype="float32")
        return batch

    def process_output(self, batch):
        """ Compile found faces for output """
        return batch
