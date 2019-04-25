#!/usr/bin/env python3
""" Base class for Face Detector plugins
    Plugins should inherit from this class

    See the override methods for which methods are
    required.

    For each source frame, the plugin must pass a dict to finalize containing:
    {"filename": <filename of source frame>,
     "image": <source image>,
     "detected_faces": <list of dlib.rectangles>}
    """

import logging
import os
import traceback
from io import StringIO

import cv2
import dlib

from lib.gpu_stats import GPUStats
from lib.utils import rotate_landmarks
from plugins.extract._config import Config

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def get_config(plugin_name):
    """ Return the config for the requested model """
    return Config(plugin_name).config_dict


class Detector():
    """ Detector object """
    def __init__(self, loglevel, rotation=None, min_size=0):
        logger.debug("Initializing %s: (rotation: %s, min_size: %s)",
                     self.__class__.__name__, rotation, min_size)
        self.config = get_config(".".join(self.__module__.split(".")[-2:]))
        self.loglevel = loglevel
        self.cachepath = os.path.join(os.path.dirname(__file__), ".cache")
        self.rotation = self.get_rotation_angles(rotation)
        self.min_size = min_size
        self.parent_is_pool = False
        self.init = None
        self.error = None

        # The input and output queues for the plugin.
        # See lib.queue_manager.QueueManager for getting queues
        self.queues = {"in": None, "out": None}

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
        logger.debug("Initialized _base %s", self.__class__.__name__)

    # <<< OVERRIDE METHODS >>> #
    # These methods must be overriden when creating a plugin
    @staticmethod
    def set_model_path():
        """ path to data file/models
            override for specific detector """
        raise NotImplementedError()

    def initialize(self, *args, **kwargs):
        """ Inititalize the detector
            Tasks to be run before any detection is performed.
            Override for specific detector """
        logger.debug("initialize %s (PID: %s, args: %s, kwargs: %s)",
                     self.__class__.__name__, os.getpid(), args, kwargs)
        self.init = kwargs.get("event", False)
        self.error = kwargs.get("error", False)
        self.queues["in"] = kwargs["in_queue"]
        self.queues["out"] = kwargs["out_queue"]

    def detect_faces(self, *args, **kwargs):
        """ Detect faces in rgb image
            Override for specific detector
            Must return a list of dlib rects"""
        try:
            if not self.init:
                self.initialize(*args, **kwargs)
        except ValueError as err:
            logger.error(err)
            exit(1)
        logger.debug("Detecting Faces (args: %s, kwargs: %s)", args, kwargs)

    # <<< DETECTION WRAPPER >>> #
    def run(self, *args, **kwargs):
        """ Parent detect process.
            This should always be called as the entry point so exceptions
            are passed back to parent.
            Do not override """
        try:
            self.detect_faces(*args, **kwargs)
        except Exception as err:  # pylint: disable=broad-except
            logger.error("Caught exception in child process: %s: %s", os.getpid(), str(err))
            # Display traceback if in initialization stage
            if not self.init.is_set():
                logger.exception("Traceback:")
            tb_buffer = StringIO()
            traceback.print_exc(file=tb_buffer)
            logger.trace(tb_buffer.getvalue())
            exception = {"exception": (os.getpid(), tb_buffer)}
            self.queues["out"].put(exception)
            exit(1)

    # <<< FINALIZE METHODS>>> #
    def finalize(self, output):
        """ This should be called as the final task of each plugin
            Performs fianl processing and puts to the out queue """
        if isinstance(output, dict):
            logger.trace("Item out: %s", {key: val
                                          for key, val in output.items()
                                          if key != "image"})
            if self.min_size > 0 and output.get("detected_faces", None):
                output["detected_faces"] = self.filter_small_faces(output["detected_faces"])
        else:
            logger.trace("Item out: %s", output)
        self.queues["out"].put(output)

    def filter_small_faces(self, detected_faces):
        """ Filter out any faces smaller than the min size threshold """
        retval = list()
        for face in detected_faces:
            face_size = ((face.right() - face.left()) ** 2 +
                         (face.bottom() - face.top()) ** 2) ** 0.5
            if face_size < self.min_size:
                logger.debug("Removing detected face: (face_size: %s, min_size: %s",
                             face_size, self.min_size)
                continue
            retval.append(face)
        return retval

    # <<< DETECTION IMAGE COMPILATION METHODS >>> #
    def compile_detection_image(self, input_image, is_square, scale_up, to_rgb):
        """ Compile the detection image """
        image = input_image[:, :, ::-1].copy() if to_rgb else input_image.copy()
        scale = self.set_scale(image, is_square=is_square, scale_up=scale_up)
        image = self.scale_image(image, scale)
        return [image, scale]

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
            source = (width * height) ** 0.5
            if isinstance(self.target, tuple):
                self.target = self.target[0] * self.target[1]
            target = self.target ** 0.5

        if scale_up or target < source:
            scale = target / source
        else:
            scale = 1.0
        logger.trace("Detector scale: %s", scale)

        return scale

    @staticmethod
    def scale_image(image, scale):
        """ Scale the image """
        # pylint: disable=no-member
        if scale == 1.0:
            return image

        height, width = image.shape[:2]
        interpln = cv2.INTER_LINEAR if scale > 1.0 else cv2.INTER_AREA
        dims = (int(width * scale), int(height * scale))

        if scale < 1.0:
            logger.verbose("Resizing image from %sx%s to %s.",
                           width, height, "x".join(str(i) for i in dims))

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
            logger.debug("Not setting rotation angles")
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

        logger.debug("Rotation Angles: %s", rotation_angles)
        return rotation_angles

    def rotate_image(self, image, angle):
        """ Rotate the image by given angle and return
            Image with rotation matrix """
        if angle == 0:
            return image, None
        return self.rotate_image_by_angle(image, angle)

    @staticmethod
    def rotate_rect(d_rect, rotation_matrix):
        """ Rotate a dlib rect based on the rotation_matrix"""
        logger.trace("Rotating d_rectangle")
        d_rect = rotate_landmarks(d_rect, rotation_matrix)
        return d_rect

    @staticmethod
    def rotate_image_by_angle(image, angle,
                              rotated_width=None, rotated_height=None):
        """ Rotate an image by a given angle.
            From: https://stackoverflow.com/questions/22041699 """

        logger.trace("Rotating image: (angle: %s, rotated_width: %s, rotated_height: %s)",
                     angle, rotated_width, rotated_height)
        height, width = image.shape[:2]
        image_center = (width/2, height/2)
        rotation_matrix = cv2.getRotationMatrix2D(  # pylint: disable=no-member
            image_center, -1.*angle, 1.)
        if rotated_width is None or rotated_height is None:
            abs_cos = abs(rotation_matrix[0, 0])
            abs_sin = abs(rotation_matrix[0, 1])
            if rotated_width is None:
                rotated_width = int(height*abs_sin + width*abs_cos)
            if rotated_height is None:
                rotated_height = int(height*abs_cos + width*abs_sin)
        rotation_matrix[0, 2] += rotated_width/2 - image_center[0]
        rotation_matrix[1, 2] += rotated_height/2 - image_center[1]
        logger.trace("Rotated image: (rotation_matrix: %s", rotation_matrix)
        return (cv2.warpAffine(image,  # pylint: disable=no-member
                               rotation_matrix,
                               (rotated_width, rotated_height)),
                rotation_matrix)

    # << QUEUE METHODS >> #
    def get_item(self):
        """ Yield one item from the queue """
        item = self.queues["in"].get()
        if isinstance(item, dict):
            logger.trace("Item in: %s", item["filename"])
        else:
            logger.trace("Item in: %s", item)
        if item == "EOF":
            logger.debug("In Queue Exhausted")
            # Re-put EOF into queue for other threads
            self.queues["in"].put(item)
        return item

    def get_batch(self):
        """ Get items from the queue in batches of
            self.batch_size

            First item in output tuple indicates whether the
            queue is exhausted.
            Second item is the batch

            Remember to put "EOF" to the out queue after processing
            the final batch """
        exhausted = False
        batch = list()
        for _ in range(self.batch_size):
            item = self.get_item()
            if item == "EOF":
                exhausted = True
                break
            batch.append(item)
        logger.trace("Returning batch size: %s", len(batch))
        return (exhausted, batch)

    # <<< DLIB RECTANGLE METHODS >>> #
    @staticmethod
    def is_mmod_rectangle(d_rectangle):
        """ Return whether the passed in object is
            a dlib.mmod_rectangle """
        return isinstance(
            d_rectangle,
            dlib.mmod_rectangle)  # pylint: disable=c-extension-no-member

    def convert_to_dlib_rectangle(self, d_rect):
        """ Convert detected mmod_rects to dlib_rectangle """
        if self.is_mmod_rectangle(d_rect):
            return d_rect.rect
        return d_rect

    # <<< MISC METHODS >>> #
    @staticmethod
    def get_vram_free():
        """ Return free and total VRAM on card with most VRAM free"""
        stats = GPUStats()
        vram = stats.get_card_most_free()
        logger.verbose("Using device %s with %sMB free of %sMB",
                       vram["device"],
                       int(vram["free"]),
                       int(vram["total"]))
        return int(vram["card_id"]), int(vram["free"]), int(vram["total"])

    @staticmethod
    def set_predetected(width, height):
        """ Set a dlib rectangle for predetected faces """
        # Predetected_face is used for sort tool.
        # Landmarks should not be extracted again from predetected faces,
        # because face data is lost, resulting in a large variance
        # against extract from original image
        logger.debug("Setting predetected face")
        return [dlib.rectangle(0, 0, width, height)]  # pylint: disable=c-extension-no-member
