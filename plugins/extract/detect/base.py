#!/usr/bin/env python3
""" Base class for Face Detector plugins
    Plugins should inherit from this class """

import os
import cv2
import dlib


class Detector():
    """ Detector object """
    def __init__(self):
        self.cachepath = os.path.dirname(__file__)
        self.initialized = False
        self.verbose = False
        self.data_path = self.set_data_path()
        self.detected_faces = None
        self.target = None  # Set to tuple of dims or int of pixel count
        self.scale = 1.0
        # Approximate VRAM used for the set target. Used to calculate
        # how many parallel processes can be run. Be conservative to avoid OOM
        # Override for detector. A small buffer will be built in.
        self.vram = None

    @staticmethod
    def set_data_path():
        """ path to data file/models
            override for specific detector """
        raise NotImplementedError()

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

    def set_scale(self, image, is_square=False, scale_up=False):
        """ Set the scale factor for incoming image """
        height, width = image.shape[:2]
        if is_square:
            if isinstance(self.target, int):
                dims = (self.target ** 0.5, self.target ** 0.5)
                self.target = dims
            source = max(height, width)
            target = max(self.target)
        else:
            if isinstance(self.target, tuple):
                self.target = self.target[0] * self.target[1]
            source = width * height
            target = self.target

        if scale_up or target < source:
            self.scale = target / source
        else: 
            self.scale = 1.0

    def set_detect_image(self, input_image):
        """ Convert the image to RGB and scale """
        image = input_image[:, :, ::-1].copy()
        if self.scale == 1.0:
            return image

        height, width = image.shape[:2]
        interpln = cv2.INTER_LINEAR if self.scale > 1.0 else cv2.INTER_AREA
        dims = (int(width * self.scale), int(height * self.scale))

        if self.verbose and self.scale < 1.0:
            print("Resizing image from {}x{} to {}.".format(
                str(width), str(height), "x".join(str(i) for i in dims)))

        image = cv2.resize(image, dims, interpolation=interpln)
        return image

    def create_detector(self, verbose, *args):
        """ Create the detector
            Override for specific detector """
        raise NotImplementedError()

    def detect_faces(self, *args):
        """ Detect faces in rgb image
            Override for specific detector """
        raise NotImplementedError()
