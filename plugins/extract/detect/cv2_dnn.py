#!/usr/bin/env python3
""" OpenCV DNN Face detection plugin """

import numpy as np

from ._base import cv2, Detector, logger


class Detect(Detector):
    """ CV2 DNN detector for face recognition """
    def __init__(self, **kwargs):
        git_model_id = 4
        model_filename = ["resnet_ssd_v1.caffemodel", "resnet_ssd_v1.prototxt"]
        super().__init__(git_model_id=git_model_id, model_filename=model_filename, **kwargs)
        self.name = "cv2-DNN Detector"
        self.input_size = 300
        self.vram = 0  # CPU Only. Doesn't use VRAM
        self.vram_per_batch = 0
        self.batchsize = 1
        self.confidence = self.config["confidence"] / 100

    def init_model(self):
        """ Initialize CV2 DNN Detector Model"""
        self.model = cv2.dnn.readNetFromCaffe(self.model_path[1],  # pylint: disable=no-member
                                              self.model_path[0])
        self.model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)  # pylint: disable=no-member

    def process_input(self, batch):
        """ Compile the detection image(s) for prediction """
        batch["feed"] = cv2.dnn.blobFromImages(batch["image"],  # pylint: disable=no-member
                                               scalefactor=1.0,
                                               size=(self.input_size, self.input_size),
                                               mean=[104, 117, 123],
                                               swapRB=False,
                                               crop=False)
        return batch

    def predict(self, batch):
        """ Run model to get predictions """
        self.model.setInput(batch["feed"])
        predictions = self.model.forward()
        batch["prediction"] = self.finalize_predictions(predictions)
        return batch

    def finalize_predictions(self, predictions):
        """ Filter faces based on confidence level """
        faces = list()
        for i in range(predictions.shape[2]):
            confidence = predictions[0, 0, i, 2]
            if confidence >= self.confidence:
                logger.trace("Accepting due to confidence %s >= %s",
                             confidence, self.confidence)
                faces.append([(predictions[0, 0, i, 3] * self.input_size),
                              (predictions[0, 0, i, 4] * self.input_size),
                              (predictions[0, 0, i, 5] * self.input_size),
                              (predictions[0, 0, i, 6] * self.input_size)])
        logger.trace("faces: %s", faces)
        return [np.array(faces)]

    def process_output(self, batch):
        """ Compile found faces for output """
        return batch
