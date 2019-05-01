#!/usr/bin/env python3
""" OpenCV DNN Face detection plugin """
import os
from time import sleep

import numpy as np

from ._base import cv2, Detector, dlib, logger


class Detect(Detector):
    """ CV2 DNN detector for face recognition """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.parent_is_pool = True
        self.target = (300, 300)  # Doesn't use VRAM
        self.vram = 0
        self.detector = None
        self.confidence = self.config["confidence"] / 100

    def set_model_path(self):
        """ CV2 DNN model file """
        model_path = os.path.join(self.cachepath, "res10_300x300_ssd_iter_140000_fp16.caffemodel")
        if not os.path.exists(model_path):
            raise Exception("Error: Unable to find {}, reinstall "
                            "the lib!".format(model_path))
        logger.debug("Loading model: '%s'", model_path)
        return model_path

    def initialize(self, *args, **kwargs):
        """ Calculate batch size """
        super().initialize(*args, **kwargs)
        logger.info("Initializing CV2-DNN Detector...")
        logger.verbose("Using CPU for detection")

        config_file = os.path.join(self.cachepath, "deploy.prototxt")
        self.detector = cv2.dnn.readNetFromCaffe(config_file,  # pylint: disable=no-member
                                                 self.model_path)
        self.detector.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)  # pylint: disable=no-member
        self.init = True
        logger.info("Initialized CV2-DNN Detector...")

    def detect_faces(self, *args, **kwargs):
        """ Detect faces in grayscale image """
        super().detect_faces(*args, **kwargs)
        while True:
            item = self.get_item()
            if item == "EOF":
                break
            logger.trace("Detecting faces: %s", item["filename"])
            [detect_image, scale] = self.compile_detection_image(item["image"],
                                                                 is_square=True,
                                                                 scale_up=True)
            height, width = detect_image.shape[:2]
            for angle in self.rotation:
                current_image, rotmat = self.rotate_image(detect_image, angle)
                logger.trace("Detecting faces")

                blob = cv2.dnn.blobFromImage(current_image,  # pylint: disable=no-member
                                             1.0,
                                             self.target,
                                             [104, 117, 123],
                                             False,
                                             False)
                self.detector.setInput(blob)
                detected = self.detector.forward()
                faces = list()
                for i in range(detected.shape[2]):
                    confidence = detected[0, 0, i, 2]
                    if confidence >= self.confidence:
                        logger.trace("Accepting due to confidence %s >= %s",
                                     confidence, self.confidence)
                        faces.append([(detected[0, 0, i, 3] * width),
                                      (detected[0, 0, i, 4] * height),
                                      (detected[0, 0, i, 5] * width),
                                      (detected[0, 0, i, 6] * height)])

                logger.trace("Detected faces: %s", [face for face in faces])

                if angle != 0 and faces:
                    logger.verbose("found face(s) by rotating image %s degrees", angle)

                if faces:
                    break

            detected_faces = self.process_output(faces, rotmat, scale)
            item["detected_faces"] = detected_faces
            self.finalize(item)

        if item == "EOF":
            sleep(3)  # Wait for all processes to finish before EOF (hacky!)
            self.queues["out"].put("EOF")
        logger.debug("Detecting Faces Complete")

    def process_output(self, faces, rotation_matrix, scale):
        """ Compile found faces for output """
        logger.trace("Processing Output: (faces: %s, rotation_matrix: %s)",
                     faces, rotation_matrix)

        faces = [dlib.rectangle(  # pylint: disable=c-extension-no-member
            int(face[0]), int(face[1]), int(face[2]), int(face[3])) for face in faces]
        if isinstance(rotation_matrix, np.ndarray):
            faces = [self.rotate_rect(face, rotation_matrix)
                     for face in faces]
        detected = [dlib.rectangle(  # pylint: disable=c-extension-no-member
            int(face.left() / scale), int(face.top() / scale),
            int(face.right() / scale), int(face.bottom() / scale))
                    for face in faces]

        logger.trace("Processed Output: %s", detected)
        return detected
