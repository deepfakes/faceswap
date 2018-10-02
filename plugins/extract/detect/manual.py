#!/usr/bin/env python3
""" Manual face detection plugin """

from .base import Detector, dlib


class Detect(Detector):
    """ Manual Detector """
    def set_data_path(self):
        return None

    def create_detector(self, verbose):
        """ Create the mtcnn detector """
        self.verbose = verbose

        if self.verbose:
            print("Adding Manual detector")

    def detect_faces(self, bounding_box):
        """ Return the given bounding box in a dlib rectangle """
        face = bounding_box
        self.detected_faces = [dlib.rectangle(int(face[0]), int(face[1]),
                                              int(face[2]), int(face[3]))]
