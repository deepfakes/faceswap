#!/usr/bin/env python3
""" DLIB Face detection plugin """

import os
from .base import Detector, dlib


class Detect(Detector):
    """ Dlib detector for face recognition """
    def __init__(self):
        super().__init__()
        self.detectors = list()
        self.target = (1920, 1920)  # Uses approx 1805MB of VRAM
        self.vram = 1805

    @staticmethod
    def compiled_for_cuda():
        """ Return a message on DLIB Cuda Compilation status """
        msg = "DLib IS "
        if not dlib.DLIB_USE_CUDA:
            msg += "NOT "
        msg += "compiled to use CUDA"
        return msg

    def set_data_path(self):
        """ Load the face detector data """
        data_path = os.path.join(self.cachepath,
                                 "mmod_human_face_detector.dat")
        if not os.path.exists(data_path):
            raise Exception("Error: Unable to find {}, reinstall "
                            "the lib!".format(data_path))
        return data_path

    def create_detector(self, verbose, detector, placeholder):
        """ Add the requested detectors """
        self.verbose = verbose

        if detector == "dlib-cnn" or detector == "dlib-all":
            if self.verbose:
                print("Adding DLib - CNN detector")
            self.detectors.append(dlib.cnn_face_detection_model_v1(
                self.data_path))

        if detector == "dlib-hog" or detector == "dlib-all":
            if self.verbose:
                print("Adding DLib - HOG detector")
            self.detectors.append(dlib.get_frontal_face_detector())

        for current_detector in self.detectors:
            current_detector(placeholder, 0)

        self.initialized = True

    def detect_faces(self, image):
        """ Detect faces in rgb image """
        self.set_scale(image, is_square=True, scale_up=True)
        image = self.set_detect_image(image)
        self.detected_faces = None
        for current_detector in self.detectors:
            self.detected_faces = current_detector(image, 0)

            if self.detected_faces:
                break
        for d_rect in self.detected_faces:
            d_rect = d_rect.rect if self.is_mmod_rectangle(d_rect) else d_rect
            d_rect = dlib.rectangle(int(d_rect.left() / self.scale),
                                    int(d_rect.top() / self.scale),
                                    int(d_rect.right() / self.scale),
                                    int(d_rect.bottom() / self.scale))
