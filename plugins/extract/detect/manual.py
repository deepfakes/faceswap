#!/usr/bin/env python3
""" Manual face detection plugin """

from ._base import Detector, dlib


class Detect(Detector):
    """ Manual Detector """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def set_model_path(self):
        """ No model required for Manual Detector """
        return None

    def initialize(self, *args, **kwargs):
        """ Create the mtcnn detector """
        print("Initializing Manual Detector...")
        super().initialize(*args, **kwargs)
        self.init.set()
        print("Initialized Manual Detector.")

    def detect_faces(self, *args, **kwargs):
        """ Return the given bounding box in a dlib rectangle """
        super().detect_faces(*args, **kwargs)
        while True:
            item = self.queues["in"].get()
            if item == "EOF":
                break
            image, face = item

            bounding_box = [dlib.rectangle(int(face[0]), int(face[1]),
                                           int(face[2]), int(face[3]))]
            retval = {"image": image,
                      "detected_faces": bounding_box}
            self.finalize(retval)

        self.queues["out"].put("EOF")
