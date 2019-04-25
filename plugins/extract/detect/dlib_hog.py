#!/usr/bin/env python3
""" DLIB CNN Face detection plugin """
from time import sleep

import numpy as np

from ._base import Detector, dlib, logger


class Detect(Detector):
    """ Dlib detector for face recognition """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.parent_is_pool = True
        self.target = (2048, 2048)  # Doesn't use VRAM
        self.vram = 0
        self.detector = dlib.get_frontal_face_detector()  # pylint: disable=c-extension-no-member
        self.iterator = None

    def set_model_path(self):
        """ No model for dlib Hog """
        pass

    def initialize(self, *args, **kwargs):
        """ Calculate batch size """
        super().initialize(*args, **kwargs)
        logger.info("Initializing Dlib-HOG Detector...")
        logger.verbose("Using CPU for detection")
        self.init = True
        logger.info("Initialized Dlib-HOG Detector...")

    def detect_faces(self, *args, **kwargs):
        """ Detect faces in rgb image """
        super().detect_faces(*args, **kwargs)
        while True:
            item = self.get_item()
            if item == "EOF":
                break
            logger.trace("Detecting faces: %s", item["filename"])
            [detect_image, scale] = self.compile_detection_image(item["image"], True, True, True)

            for angle in self.rotation:
                current_image, rotmat = self.rotate_image(detect_image, angle)

                logger.trace("Detecting faces")
                faces = self.detector(current_image, 0)
                logger.trace("Detected faces: %s", [face for face in faces])

                if angle != 0 and faces.any():
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
        if isinstance(rotation_matrix, np.ndarray):
            faces = [self.rotate_rect(face, rotation_matrix)
                     for face in faces]
        detected = [dlib.rectangle(  # pylint: disable=c-extension-no-member
            int(face.left() / scale), int(face.top() / scale),
            int(face.right() / scale), int(face.bottom() / scale))
                    for face in faces]
        logger.trace("Processed Output: %s", detected)
        return detected
