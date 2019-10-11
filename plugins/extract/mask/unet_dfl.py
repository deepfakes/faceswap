#!/usr/bin/env python3
""" UNET DFL face mask plugin """

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
        self.input_size = 256
        self.blur_kernel = 5
        self.vram = 3440
        self.vram_warnings = 1024  # TODO determine
        self.vram_per_batch = 64  # TODO determine
        self.batchsize = self.config["batch-size"]

    def init_model(self):
        self.model = KSession(self.name, self.model_path, model_kwargs=dict())
        self.model.load_model()
        placeholder = np.zeros((self.batchsize, self.input_size, self.input_size, 3),
                               dtype="float32")
        self.model.predict(placeholder)

    def process_input(self, batch):
        """ Compile the detected faces for prediction """
        batch["feed"] = np.array([face.feed_face[..., :3]
                                  for face in batch["detected_faces"]], dtype="float32") / 255.0
        logger.trace("feed shape: %s", batch["feed"].shape)
        return batch

    def predict(self, batch):
        """ Run model to get predictions """
        batch["prediction"] = self.model.predict(batch["feed"])
        return batch

    def process_output(self, batch):
        """ Compile found faces for output """
        return batch
