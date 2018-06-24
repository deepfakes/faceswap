#!/usr/bin python3
""" DLIB Detector for face alignment
    Code adapted and modified from:
    https://github.com/1adrianb/face-alignment """

import os
import dlib


class DLibDetector(object):
    """ Dlib detector for face recognition """
    def __init__(self):
        self.verbose = False
        self.data_path = self.set_data_path()
        self.detectors = list()
        self.detected_faces = None

    @staticmethod
    def set_data_path():
        """ Load the face detector data """
        data_path = os.path.join(os.path.dirname(__file__),
                                 ".cache",
                                 "mmod_human_face_detector.dat")
        if not os.path.exists(data_path):
            raise Exception("Error: Unable to find {}, reinstall "
                            "the lib!".format(data_path))
        return data_path

    def add_detectors(self, detector):
        """ Add the requested detectors """
        if self.detectors:
            return

        if detector == 'cnn' or detector == "all":
            self.detectors.append(dlib.cnn_face_detection_model_v1(
                self.data_path))

        if detector == "hog" or detector == "all":
            self.detectors.append(dlib.get_frontal_face_detector())

    def detect_faces(self, images):
        """ Detect faces in images """
        self.detected_faces = None
        for current_detector, current_image in(
                (current_detector, current_image)
                for current_detector in self.detectors
                for current_image in images):
            self.detected_faces = current_detector(current_image, 0)

            if self.detected_faces:
                break

    def set_predetected(self, width, height):
        """ Set a dlib rectangle for predetected faces """
        # Predetected_face is used for sort tool.
        # Landmarks should not be extracted again from predetected faces,
        # because face data is lost, resulting in a large variance
        # against extract from original image
        self.detected_faces = [dlib.rectangle(0, 0, width, height)]

    @staticmethod
    def is_mmod_rectangle(d_rectangle):
        """ Return whether the passed in object is
            a dlib.mmod_rectangle """
        return isinstance(d_rectangle, dlib.mmod_rectangle)
