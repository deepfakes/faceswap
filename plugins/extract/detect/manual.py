#!/usr/bin/env python3
""" Manual face detection plugin """

from ._base import Detector, dlib, logger


class Detect(Detector):
    """ Manual Detector """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def set_model_path(self):
        """ No model required for Manual Detector """
        return None

    def initialize(self, *args, **kwargs):
        """ Create the mtcnn detector """
        super().initialize(*args, **kwargs)
        logger.info("Initializing Manual Detector...")
        self.init.set()
        logger.info("Initialized Manual Detector.")

    def detect_faces(self, *args, **kwargs):
        """ Return the given bounding box in a dlib rectangle """
        super().detect_faces(*args, **kwargs)
        while True:
            item = self.get_item()
            if item == "EOF":
                break
            face = item["face"]

            bounding_box = [dlib.rectangle(  # pylint: disable=c-extension-no-member
                int(face[0]), int(face[1]), int(face[2]), int(face[3]))]
            item["detected_faces"] = bounding_box
            self.finalize(item)

        self.queues["out"].put("EOF")
