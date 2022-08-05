#!/usr/bin/env python3
""" Components Mask for faceswap.py """

import numpy as np
from ._base import Masker, logger


class Mask(Masker):
    """ A mask that fills the whole face area with 1s or 0s (depending on user selected settings)
    for custom editing. """
    def __init__(self, **kwargs):
        git_model_id = None
        model_filename = None
        super().__init__(git_model_id=git_model_id, model_filename=model_filename, **kwargs)
        self.input_size = 256
        self.name = "Custom"
        self.vram = 0  # Doesn't use GPU
        self.vram_per_batch = 0
        self.batchsize = self.config["batch-size"]
        self._storage_centering = self.config["centering"]
        # Separate storage for face and head masks
        self._storage_name = f"{self._storage_name}_{self._storage_centering}"

    def init_model(self):
        logger.debug("No mask model to initialize")

    def process_input(self, batch):
        """ Compile the detected faces for prediction """
        batch["feed"] = np.zeros((self.batchsize, self.input_size, self.input_size, 1),
                                 dtype="float32")
        return batch

    def predict(self, batch):
        """ Run model to get predictions """
        if self.config["fill"]:
            batch["feed"][:] = 1.0
        batch["prediction"] = batch["feed"]
        return batch

    def process_output(self, batch):
        """ Compile found faces for output """
        return batch
