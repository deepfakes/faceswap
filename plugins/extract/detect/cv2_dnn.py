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
        self.target = (300, 300)  # Doesn't use VRAM
        self.vram = 0
        self.detector = None
        self.confidence = self.config["confidence"] / 100

    def initialize(self, *args, **kwargs):
        """ Calculate batch size """
        super().initialize(*args, **kwargs)
        logger.info("Initializing cv2 DNN Detector...")
        logger.verbose("Using CPU for detection")
        self.detector = cv2.dnn.readNetFromCaffe(self.model_path[1],  # pylint: disable=no-member
                                                 self.model_path[0])
        self.detector.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)  # pylint: disable=no-member
        self.init.set()
        logger.info("Initialized cv2 DNN Detector.")

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

        self.queues["out"].put("EOF")
        logger.debug("Detecting Faces Complete")

    def process_output(self, faces, rotation_matrix, scale):
        """ Compile found faces for output """
        logger.trace("Processing Output: (faces: %s, rotation_matrix: %s)",
                     faces, rotation_matrix)

        faces = [self.to_bounding_box_dict(face[0], face[1], face[2], face[3]) for face in faces]
        if isinstance(rotation_matrix, np.ndarray):
            faces = [self.rotate_rect(face, rotation_matrix)
                     for face in faces]
        detected = [self.to_bounding_box_dict(face["left"] / scale, face["top"] / scale,
                                              face["right"] / scale, face["bottom"] / scale)
                    for face in faces]

        logger.trace("Processed Output: %s", detected)
        return detected
