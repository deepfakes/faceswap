#!/usr/bin/env python3
""" OpenCV DNN Face detection plugin """
import logging

import numpy as np

from ._base import BatchType, cv2, Detector, DetectorBatch


logger = logging.getLogger(__name__)


class Detect(Detector):
    """ CV2 DNN detector for face recognition """
    def __init__(self, **kwargs) -> None:
        git_model_id = 4
        model_filename = ["resnet_ssd_v1.caffemodel", "resnet_ssd_v1.prototxt"]
        super().__init__(git_model_id=git_model_id, model_filename=model_filename, **kwargs)
        self.name = "cv2-DNN Detector"
        self.input_size = 300
        self.vram = 0  # CPU Only. Doesn't use VRAM
        self.vram_per_batch = 0
        self.batchsize = 1
        self.confidence = self.config["confidence"] / 100

    def init_model(self) -> None:
        """ Initialize CV2 DNN Detector Model"""
        assert isinstance(self.model_path, list)
        self.model = cv2.dnn.readNetFromCaffe(self.model_path[1],
                                              self.model_path[0])
        self.model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def process_input(self, batch: BatchType) -> None:
        """ Compile the detection image(s) for prediction """
        assert isinstance(batch, DetectorBatch)
        batch.feed = cv2.dnn.blobFromImages(batch.image,
                                            scalefactor=1.0,
                                            size=(self.input_size, self.input_size),
                                            mean=[104, 117, 123],
                                            swapRB=False,
                                            crop=False)

    def predict(self, feed: np.ndarray) -> np.ndarray:
        """ Run model to get predictions """
        assert isinstance(self.model, cv2.dnn.Net)
        self.model.setInput(feed)
        predictions = self.model.forward()
        return self.finalize_predictions(predictions)

    def finalize_predictions(self, predictions: np.ndarray) -> np.ndarray:
        """ Filter faces based on confidence level """
        faces = []
        for i in range(predictions.shape[2]):
            confidence = predictions[0, 0, i, 2]
            if confidence >= self.confidence:
                logger.trace("Accepting due to confidence %s >= %s",  # type:ignore[attr-defined]
                             confidence, self.confidence)
                faces.append([(predictions[0, 0, i, 3] * self.input_size),
                              (predictions[0, 0, i, 4] * self.input_size),
                              (predictions[0, 0, i, 5] * self.input_size),
                              (predictions[0, 0, i, 6] * self.input_size)])
        logger.trace("faces: %s", faces)  # type:ignore[attr-defined]
        return np.array(faces)[None, ...]

    def process_output(self, batch: BatchType) -> None:
        """ Compile found faces for output """
        return
