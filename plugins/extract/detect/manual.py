#!/usr/bin/env python3
""" Manual face detection plugin """

from ._base import Detector, logger


class Detect(Detector):
    """ Manual Detector """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def initialize(self, *args, **kwargs):
        """ Create the mtcnn detector """
        super().initialize(*args, **kwargs)
        logger.info("Initializing Manual Detector...")
        self.init.set()
        logger.info("Initialized Manual Detector.")

    def detect_faces(self, *args, **kwargs):
        """ Return the given bounding box in a bounding box dict """
        super().detect_faces(*args, **kwargs)
        while True:
            item = self.get_item()
            if item == "EOF":
                break
            face = item["face"]

            bounding_box = [self.to_bounding_box_dict(face[0], face[1], face[2], face[3])]
            item["detected_faces"] = bounding_box
            self.finalize(item)

        self.queues["out"].put("EOF")
