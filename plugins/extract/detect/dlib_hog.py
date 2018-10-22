#!/usr/bin/env python3
""" DLIB CNN Face detection plugin """
from time import sleep

import numpy as np

from ._base import Detector, dlib


class Detect(Detector):
    """ Dlib detector for face recognition """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.parent_is_pool = True
        self.target = (2048, 2048)  # Doesn't use VRAM
        self.vram = 0
        self.detector = dlib.get_frontal_face_detector()
        self.iterator = None

    def set_model_path(self):
        """ No model for dlib Hog """
        pass

    def initialize(self, *args, **kwargs):
        """ Calculate batch size """
        print("Initializing Dlib-HOG Detector...")
        super().initialize(*args, **kwargs)
        if self.verbose:
            print("Using CPU for detection")
        self.init = True
        print("Initialized Dlib-HOG Detector...")

    def detect_faces(self, *args, **kwargs):
        """ Detect faces in rgb image """
        super().detect_faces(*args, **kwargs)
        try:
            while True:
                item = self.queues["in"].get()
                if item in ("EOF", "END"):
                    self.queues["in"].put("END")
                    break

                filename, image = item
                detect_image = self.compile_detection_image(image, True, True)

                for angle in self.rotation:
                    current_image, rotmat = self.rotate_image(detect_image,
                                                              angle)

                    faces = self.detector(current_image, 0)

                    if self.verbose and angle != 0 and faces.any():
                        print("found face(s) by rotating image {} "
                              "degrees".format(angle))

                    if faces:
                        break

                detected_faces = self.process_output(faces, rotmat)
                retval = {"filename": filename,
                          "image": image,
                          "detected_faces": detected_faces}
                self.finalize(retval)
        except:
            retval = {"exception": True}
            self.queues["out"].put(retval)
            raise

        if item == "EOF":
            sleep(3)  # Wait for all processes to finish before EOF (hacky!)
            self.queues["out"].put("EOF")

    def process_output(self, faces, rotation_matrix):
        """ Compile found faces for output """
        if isinstance(rotation_matrix, np.ndarray):
            faces = [self.rotate_rect(face, rotation_matrix)
                     for face in faces]
        detected = [dlib.rectangle(int(face.left() / self.scale),
                                   int(face.top() / self.scale),
                                   int(face.right() / self.scale),
                                   int(face.bottom() / self.scale))
                    for face in faces]
        return detected
