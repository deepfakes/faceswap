#!/usr/bin/env python3
""" DLIB CNN Face detection plugin """

import numpy as np

from lib.multithreading import PoolProcess
from .base import Detector, dlib


class Detect(Detector):
    """ Dlib detector for face recognition """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.target = (2048, 2048)  # Doesn't use VRAM
        self.vram = 0
        self.detector = dlib.get_frontal_face_detector()
        self.iterator = None

    def set_model_path(self):
        """ No model for dlib Hog """
        pass

    def initialize(self, **kwargs):
        """ Calculate batch size """
        if self.verbose:
            print("Using CPU for detection")
        self.iterator = PoolProcess(self.detect_process, verbose=self.verbose)

    def detect_faces(self, image_queue):
        """ Dlib Hog is CPU bound so multi process """
        for output in self.iterator.process(self.feed_queue(image_queue)):
            yield output

    def detect_process(self, detect_item):
        """ Detect faces in rgb image """
        filename, image = detect_item
        detect_image = self.compile_detection_image(image, True, True)

        for angle in self.rotation:
            current_image, rotmat = self.rotate_image(detect_image, angle)

            faces = self.detector(current_image, 0)

            if self.verbose and angle != 0 and faces:
                print("found face(s) by rotating image {} degrees".format(
                    angle))

            if faces:
                break

        detected_faces = self.process_output(faces, rotmat)
        return filename, image, detected_faces

    def process_output(self, faces, rotation_matrix):
        """ Compile found faces for output """
        if isinstance(rotation_matrix, np.ndarray):
            faces = [self.rotate_rect(face, rotation_matrix)
                     for face in faces]
        return [dlib.rectangle(int(face.left() / self.scale),
                               int(face.top() / self.scale),
                               int(face.right() / self.scale),
                               int(face.bottom() / self.scale))
                for face in faces]
