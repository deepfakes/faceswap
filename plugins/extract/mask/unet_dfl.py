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
import logging
import typing as T

import numpy as np
from lib.model.session import KSession
from ._base import BatchType, Masker, MaskerBatch

logger = logging.getLogger(__name__)


class Mask(Masker):
    """ Neural network to process face image into a segmentation mask of the face """
    def __init__(self, **kwargs) -> None:
        git_model_id = 6
        model_filename = "DFL_256_sigmoid_v1.h5"
        super().__init__(git_model_id=git_model_id, model_filename=model_filename, **kwargs)
        self.model: KSession
        self.name = "U-Net"
        self.input_size = 256
        self.vram = 3424
        self.vram_warnings = 256
        self.vram_per_batch = 80
        self.batchsize = self.config["batch-size"]
        self._storage_centering = "legacy"

    def init_model(self) -> None:
        assert self.name is not None and isinstance(self.model_path, str)
        self.model = KSession(self.name,
                              self.model_path,
                              model_kwargs={},
                              allow_growth=self.config["allow_growth"],
                              exclude_gpus=self._exclude_gpus)
        self.model.load_model()
        placeholder = np.zeros((self.batchsize, self.input_size, self.input_size, 3),
                               dtype="float32")
        self.model.predict(placeholder)

    def process_input(self, batch: BatchType) -> None:
        """ Compile the detected faces for prediction """
        assert isinstance(batch, MaskerBatch)
        batch.feed = np.array([T.cast(np.ndarray, feed.face)[..., :3]
                               for feed in batch.feed_faces], dtype="float32") / 255.0
        logger.trace("feed shape: %s", batch.feed.shape)  # type: ignore

    def predict(self, feed: np.ndarray) -> np.ndarray:
        """ Run model to get predictions """
        retval = self.model.predict(feed)
        assert isinstance(retval, np.ndarray)
        return retval

    def process_output(self, batch: BatchType) -> None:
        """ Compile found faces for output """
        return
