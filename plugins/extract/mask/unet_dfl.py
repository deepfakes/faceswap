#!/usr/bin/env python3
""" UNET DFL face mask plugin

Architecture and Pre-Trained Model based on...
TernausNet: U-Net with VGG11 Encoder Pre-Trained on ImageNet for Image Segmentation
https://arxiv.org/abs/1801.05746
https://github.com/ternaus/TernausNet

Source Implementation and fine-tune training....
https://github.com/iperov/DeepFaceLab/blob/master/nnlib/TernausNet.py

Model file sourced from...
https://github.com/iperov/DeepFaceLab/blob/master/nnlib/FANSeg_256_full_face.h5
"""

import numpy as np
from lib.model.session import KSession
from ._base import Masker, logger


class Mask(Masker):
    """ Neural network to process face image into a segmentation mask of the face """
    def __init__(self, **kwargs):
        git_model_id = 6
        model_filename = "DFL_256_sigmoid_v1.h5"
        super().__init__(git_model_id=git_model_id, model_filename=model_filename, **kwargs)
        self.name = "U-Net"
        self.input_size = 256
        self.vram = 3424
        self.vram_warnings = 256
        self.vram_per_batch = 80
        self.batchsize = self.config["batch-size"]

    def init_model(self):
        self.model = KSession(self.name, self.model_path,
                              model_kwargs=dict(), allow_growth=self.config["allow_growth"])
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
