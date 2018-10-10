#!/usr/bin/env python3
""" Manual face detection plugin """

from .base import Detector, dlib


class Detect(Detector):
    """ Manual Detector """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "manual"

    def set_model_path(self):
        """ No model required for Manual Detector """
        return None

    def initialize(self, **kwargs):
        """ Create the mtcnn detector """
        pass

    def detect_faces(self, bounding_box):
        """ Return the given bounding box in a dlib rectangle """
        face = bounding_box
        return [dlib.rectangle(int(face[0]), int(face[1]),
                               int(face[2]), int(face[3]))]
