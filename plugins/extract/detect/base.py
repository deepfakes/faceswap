#!/usr/bin/env python3
""" Base class for Face Detector plugins
    Plugins should inherit from this class

    See the override methods for which methods are
    required.

    The plugin must yield:
    <filename> <source_image> <detected_faces in image as a dlib.rectangle>>
    """

import os

import cv2
import numpy as np
import dlib

from lib.gpu_stats import GPUStats
from lib.utils import rotate_image_by_angle


class Detector():
    """ Detector object """
    def __init__(self, rotation=None, verbose=False):
        self.cachepath = os.path.dirname(__file__)
        self.verbose = verbose
        self.rotation = self.get_rotation_angles(rotation)
        self.feed_empty = False

        # Scaling factor for image
        self.scale = 1.0

        #  Path to model if required
        self.model_path = self.set_model_path()

        # Target image size for passing images through the detector
        # Set to tuple of dimensions (x, y) or int of pixel count
        self.target = None

        # Approximate VRAM used for the set target. Used to calculate
        # how many parallel processes / batches can be run.
        # Be conservative to avoid OOM.
        self.vram = None

        # For detectors that support batching, this should be set to
        # the calculated batch size that the amount of available VRAM
        # will support. It is also used for holding the number of threads/
        # processes for parallel processing plugins
        self.batch_size = 1

    # <<< OVERRIDE METHODS >>> #
    # These methods must be overriden when creating a plugin
    @staticmethod
    def set_model_path():
        """ path to data file/models
            override for specific detector """
        raise NotImplementedError()

    def initialize(self, **kwargs):
        """ Inititalize the detector
            Tasks to be run before any detection is performed.
            Override for specific detector """
        raise NotImplementedError()

    def detect_faces(self, image_queue):
        """ Detect faces in rgb image
            Override for specific detector
            Must return a list of dlib rects"""
        raise NotImplementedError()

    # <<< DETECTION IMAGE COMPILATION METHODS >>> #
    def compile_detection_image(self, image, is_square, scale_up):
        """ Compile the detection image """
        self.set_scale(image, is_square=is_square, scale_up=scale_up)
        return self.set_detect_image(image)

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

    # <<< IMAGE ROTATION METHODS >>> #
    @staticmethod
    def get_rotation_angles(rotation):
        """ Set the rotation angles. Includes backwards compatibility for the
            'on' and 'off' options:
                - 'on' - increment 90 degrees
                - 'off' - disable
                - 0 is prepended to the list, as whatever happens, we want to
                  scan the image in it's upright state """
        rotation_angles = [0]

        if not rotation or rotation.lower() == "off":
            return rotation_angles

        if rotation.lower() == "on":
            rotation_angles.extend(range(90, 360, 90))
        else:
            passed_angles = [int(angle)
                             for angle in rotation.split(",")]
            if len(passed_angles) == 1:
                rotation_step_size = passed_angles[0]
                rotation_angles.extend(range(rotation_step_size,
                                             360,
                                             rotation_step_size))
            elif len(passed_angles) > 1:
                rotation_angles.extend(passed_angles)

        return rotation_angles

    @staticmethod
    def rotate_image(image, angle):
        """ Rotate the image by given angle and return
            Image with rotation matrix """
        if angle == 0:
            return image, None
        return rotate_image_by_angle(image, angle)

    @staticmethod
    def rotate_rect(d_rect, rotation_matrix):
        """ Rotate a dlib rect based on the rotation_matrix"""
        rotation_matrix = cv2.invertAffineTransform(rotation_matrix)

        corners = [[d_rect.left(), d_rect.top()],
                   [d_rect.right(), d_rect.bottom()]]

        points = np.array(corners, np.int32)
        points = np.expand_dims(points, axis=0)
        transformed = cv2.transform(points,
                                    rotation_matrix).astype(np.int32)
        transformed = transformed.squeeze().ravel()

        d_rect = dlib.rectangle(*transformed)
        return d_rect

    # << QUEUE ITERATION METHODS >> #
    def feed_queue(self, input_queue):
        """ Feed the items from the queue as an iterator """
        for item in iter(input_queue.get, "EOF"):
            yield item
        self.feed_empty = True

    def feed_batch(self, input_queue):
        """ Feed the items from the queue iterator in batches

            First item in output tuple indicates whether the
            generator is exhausted.
            Second item is the batch """
        exhausted = False
        batch = list()
        for _ in range(self.batch_size):
            try:
                item = next(self.feed_queue(input_queue))
                batch.append(item)
            except StopIteration:
                exhausted = True
                break
        return (exhausted, batch)

    # <<< DLIB RECTANGLE METHODS >>> #
    @staticmethod
    def is_mmod_rectangle(d_rectangle):
        """ Return whether the passed in object is
            a dlib.mmod_rectangle """
        return isinstance(d_rectangle, dlib.mmod_rectangle)

    def convert_to_dlib_rectangle(self, d_rect):
        """ Convert detected mmod_rects to dlib_rectangle """
        if self.is_mmod_rectangle(d_rect):
            return d_rect.rect
        return d_rect

    # <<< MISC METHODS >>> #
    def get_vram_free(self):
        """ Return total free VRAM on largest card """
        stats = GPUStats()
        free_vram = stats.get_free()
        # Get card with most available vram
        vram_free = max(free_vram)
        card = free_vram.index(vram_free)
        if self.verbose:
            print("Using device {} with {}MB free of {}MB".format(
                stats.devices[card],
                int(vram_free),
                int(stats.vram[card])))
        return vram_free

    @staticmethod
    def set_predetected(width, height):
        """ Set a dlib rectangle for predetected faces """
        # Predetected_face is used for sort tool.
        # Landmarks should not be extracted again from predetected faces,
        # because face data is lost, resulting in a large variance
        # against extract from original image
        return [dlib.rectangle(0, 0, width, height)]
