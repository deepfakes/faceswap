#!/usr/bin/env python3
""" DLib landmarks extractor for faceswap.py """
import face_recognition_models
import dlib

from ._base import Aligner, logger


class Align(Aligner):
    """ Perform transformation to align and get landmarks """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.vram = 0  # Doesn't use GPU
        self.model = None

    def set_model_path(self):
        """ Model path handled by face_recognition_models """
        model_path = face_recognition_models.pose_predictor_model_location()
        logger.debug("Loading model: '%s'", model_path)
        return model_path

    def initialize(self, *args, **kwargs):
        """ Initialization tasks to run prior to alignments """
        try:
            super().initialize(*args, **kwargs)
            logger.info("Initializing Dlib Pose Predictor...")
            logger.debug("dlib initialize: (args: %s kwargs: %s)", args, kwargs)
            self.model = dlib.shape_predictor(self.model_path)  # pylint: disable=c-extension-no-member
            self.init.set()
            logger.info("Initialized Dlib Pose Predictor.")
        except Exception as err:
            self.error.set()
            raise err

    def align(self, *args, **kwargs):
        """ Perform alignments on detected faces """
        super().align(*args, **kwargs)
        for item in self.get_item():
            if item == "EOF":
                self.finalize(item)
                break
            image = item["image"][:, :, ::-1].copy()

            logger.trace("Algning faces")
            item["landmarks"] = self.process_landmarks(image, item["detected_faces"])
            logger.trace("Algned faces: %s", item["landmarks"])

            self.finalize(item)
        logger.debug("Completed Align")

    def process_landmarks(self, image, detected_faces):
        """ Align image and process landmarks """
        logger.trace("Processing Landmarks")
        retval = list()
        for detected_face in detected_faces:
            pts = self.model(image, detected_face).parts()
            landmarks = [(point.x, point.y) for point in pts]
            retval.append(landmarks)
        logger.trace("Processed Landmarks: %s", retval)
        return retval
