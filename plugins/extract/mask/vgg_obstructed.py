#!/usr/bin/env python3
""" VGG Obstructed face mask plugin

Architecture and Pre-Trained Model based on...
On Face Segmentation, Face Swapping, and Face Perception
https://arxiv.org/abs/1704.06729

Source Implementation...
https://github.com/YuvalNirkin/face_segmentation

Model file sourced from...
https://github.com/YuvalNirkin/face_segmentation/releases/download/1.0/face_seg_fcn8s.zip

Caffe model re-implemented in Keras by Kyle Vrooman
"""

import numpy as np
from lib.model.session import KSession
from ._base import Masker, logger


class Mask(Masker):
    """ Neural network to process face image into a segmentation mask of the face """
    def __init__(self, **kwargs):
        git_model_id = 5
        model_filename = "Nirkin_500_softmax_v1.h5"
        super().__init__(git_model_id=git_model_id, model_filename=model_filename, **kwargs)
        self.name = "VGG Obstructed"
        self.input_size = 500
        self.vram = 3936
        self.vram_warnings = 1088  # at BS 1. OOMs at higher batch sizes
        self.vram_per_batch = 304
        self.batchsize = self.config["batch-size"]

    def init_model(self):
        self.model = KSession(self.name, self.model_path,
                              model_kwargs=dict(), allow_growth=self.config["allow_growth"])
        self.model.load_model()
        self.model.append_softmax_activation(layer_index=-1)
        placeholder = np.zeros((self.batchsize, self.input_size, self.input_size, 3),
                               dtype="float32")
        self.model.predict(placeholder)

    def process_input(self, batch):
        """ Compile the detected faces for prediction """
        input_ = [face.feed_face[..., :3] for face in batch["detected_faces"]]
        batch["feed"] = input_ - np.mean(input_, axis=(1, 2))[:, None, None, :]
        logger.trace("feed shape: %s", batch["feed"].shape)
        return batch

    def predict(self, batch):
        """ Run model to get predictions """
        predictions = self.model.predict(batch["feed"])
        batch["prediction"] = predictions[..., 0] * -1.0 + 1.0
        return batch

    def process_output(self, batch):
        """ Compile found faces for output """
        return batch
