#!/usr/bin/env python3
""" Base class for Face Aligner plugins
    Plugins should inherit from this class

    See the override methods for which methods are
    required.

    The plugin will receive a dict containing:
    {"filename": <filename of source frame>,
     "image": <source image>,
     "detected_faces": <list of bounding box dicts as defined in lib/plugins/extract/detect/_base>}

    For each source item, the plugin must pass a dict to finalize containing:
    {"filename": <filename of source frame>,
     "image": <source image>,
     "detected_faces": <list of bounding box dicts as defined in lib/plugins/extract/detect/_base>,
     "landmarks": <list of landmarks>}
    """

import logging
import os
import traceback

from io import StringIO

import cv2

from lib.aligner import Extract
from lib.gpu_stats import GPUStats
from lib.utils import GetModel

logger = logging.getLogger(__name__)  # pylint:disable=invalid-name


class Aligner():
    """ Landmarks Aligner Object """
    def __init__(self, loglevel, configfile=None, normalize_method=None,
                 git_model_id=None, model_filename=None, colorspace="BGR", input_size=256):
        logger.debug("Initializing %s: (loglevel: %s, configfile: %s, normalize_method: %s, "
                     "git_model_id: %s, model_filename: '%s', colorspace: '%s'. input_size: %s)",
                     self.__class__.__name__, loglevel, configfile, normalize_method, git_model_id,
                     model_filename, colorspace, input_size)
        self.loglevel = loglevel
        self.normalize_method = normalize_method
        self.colorspace = colorspace.upper()
        self.input_size = input_size
        self.extract = Extract()
        self.init = None
        self.error = None

        # The input and output queues for the plugin.
        # See lib.queue_manager.QueueManager for getting queues
        self.queues = {"in": None, "out": None}

        #  Get model if required
        self.model_path = self.get_model(git_model_id, model_filename)

        # Approximate VRAM required for aligner. Used to calculate
        # how many parallel processes / batches can be run.
        # Be conservative to avoid OOM.
        self.vram = None

        # Set to true if the plugin supports PlaidML
        self.supports_plaidml = False

        logger.debug("Initialized %s", self.__class__.__name__)

    # <<< OVERRIDE METHODS >>> #
    # These methods must be overriden when creating a plugin
    def initialize(self, *args, **kwargs):
        """ Inititalize the aligner
            Tasks to be run before any alignments are performed.
            Override for specific detector """
        logger.debug("_base initialize %s: (PID: %s, args: %s, kwargs: %s)",
                     self.__class__.__name__, os.getpid(), args, kwargs)
        self.init = kwargs["event"]
        self.error = kwargs["error"]
        self.queues["in"] = kwargs["in_queue"]
        self.queues["out"] = kwargs["out_queue"]

    def align_image(self, detected_face, image):
        """ Align the incoming image for feeding into aligner
            Override for aligner specific processing """
        raise NotImplementedError

    def predict_landmarks(self, feed_dict):
        """ Predict the 68 point landmarks
            Override for aligner specific landmark prediction """
        raise NotImplementedError

    # <<< GET MODEL >>> #
    @staticmethod
    def get_model(git_model_id, model_filename):
        """ Check if model is available, if not, download and unzip it """
        if model_filename is None:
            logger.debug("No model_filename specified. Returning None")
            return None
        if git_model_id is None:
            logger.debug("No git_model_id specified. Returning None")
            return None
        cache_path = os.path.join(os.path.dirname(__file__), ".cache")
        model = GetModel(model_filename, cache_path, git_model_id)
        return model.model_path

    # <<< ALIGNMENT WRAPPER >>> #
    def run(self, *args, **kwargs):
        """ Parent align process.
            This should always be called as the entry point so exceptions
            are passed back to parent.
            Do not override """
        try:
            self.align(*args, **kwargs)
        except Exception:  # pylint:disable=broad-except
            logger.error("Caught exception in child process: %s", os.getpid())
            # Display traceback if in initialization stage
            if not self.init.is_set():
                logger.exception("Traceback:")
            tb_buffer = StringIO()
            traceback.print_exc(file=tb_buffer)
            exception = {"exception": (os.getpid(), tb_buffer)}
            self.queues["out"].put(exception)
            exit(1)

    def align(self, *args, **kwargs):
        """ Process landmarks """
        if not self.init:
            self.initialize(*args, **kwargs)
        logger.debug("Launching Align: (args: %s kwargs: %s)", args, kwargs)

        for item in self.get_item():
            if item == "EOF":
                self.finalize(item)
                break
            image = self.convert_color(item["image"])

            logger.trace("Aligning faces")
            try:
                item["landmarks"] = self.process_landmarks(image, item["detected_faces"])
                logger.trace("Aligned faces: %s", item["landmarks"])
            except ValueError as err:
                logger.warning("Image '%s' could not be processed. This may be due to corrupted "
                               "data: %s", item["filename"], str(err))
                item["detected_faces"] = list()
                item["landmarks"] = list()
                # UNCOMMENT THIS CODE BLOCK TO PRINT TRACEBACK ERRORS
                # import sys
                # exc_info = sys.exc_info()
                # traceback.print_exception(*exc_info)
            self.finalize(item)
        logger.debug("Completed Align")

    def convert_color(self, image):
        """ Convert the image to the correct colorspace """
        logger.trace("Converting image to colorspace: %s", self.colorspace)
        if self.colorspace == "RGB":
            cvt_image = image[:, :, ::-1].copy()
        elif self.colorspace == "GRAY":
            cvt_image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)  # pylint:disable=no-member
        else:
            cvt_image = image.copy()
        return cvt_image

    def process_landmarks(self, image, detected_faces):
        """ Align image and process landmarks """
        logger.trace("Processing landmarks")
        retval = list()
        for detected_face in detected_faces:
            feed_dict = self.align_image(detected_face, image)
            self.normalize_face(feed_dict)
            landmarks = self.predict_landmarks(feed_dict)
            retval.append(landmarks)
        logger.trace("Processed landmarks: %s", retval)
        return retval

    # <<< FACE NORMALIZATION METHODS >>> #
    def normalize_face(self, feed_dict):
        """ Normalize the face for feeding into model """
        if self.normalize_method is None:
            return
        logger.trace("Normalizing face")
        meth = getattr(self, "normalize_{}".format(self.normalize_method.lower()))
        feed_dict["image"] = meth(feed_dict["image"])
        logger.trace("Normalized face")

    @staticmethod
    def normalize_mean(face):
        """ Normalize Face to the Mean """
        face = face / 255.0
        for chan in range(3):
            layer = face[:, :, chan]
            layer = (layer - layer.min()) / (layer.max() - layer.min())
            face[:, :, chan] = layer
        return face * 255.0

    @staticmethod
    def normalize_hist(face):
        """ Equalize the RGB histogram channels """
        for chan in range(3):
            face[:, :, chan] = cv2.equalizeHist(face[:, :, chan])  # pylint: disable=no-member
        return face

    @staticmethod
    def normalize_clahe(face):
        """ Perform Contrast Limited Adaptive Histogram Equalization """
        clahe = cv2.createCLAHE(clipLimit=2.0,  # pylint: disable=no-member
                                tileGridSize=(4, 4))
        for chan in range(3):
            face[:, :, chan] = clahe.apply(face[:, :, chan])
        return face

    # <<< FINALIZE METHODS >>> #
    def finalize(self, output):
        """ This should be called as the final task of each plugin
            aligns faces and puts to the out queue """
        if output == "EOF":
            logger.trace("Item out: %s", output)
            self.queues["out"].put("EOF")
            return
        logger.trace("Item out: %s", {key: val
                                      for key, val in output.items()
                                      if key != "image"})
        self.queues["out"].put((output))

    # <<< MISC METHODS >>> #
    def get_vram_free(self):
        """ Return free and total VRAM on card with most VRAM free"""
        stats = GPUStats()
        vram = stats.get_card_most_free(supports_plaidml=self.supports_plaidml)
        logger.verbose("Using device %s with %sMB free of %sMB",
                       vram["device"],
                       int(vram["free"]),
                       int(vram["total"]))
        return int(vram["card_id"]), int(vram["free"]), int(vram["total"])

    def get_item(self):
        """ Yield one item from the queue """
        while True:
            item = self.queues["in"].get()
            if isinstance(item, dict):
                logger.trace("Item in: %s", {key: val
                                             for key, val in item.items()
                                             if key != "image"})
                # Pass Detector failures straight out and quit
                if item.get("exception", None):
                    self.queues["out"].put(item)
                    exit(1)
            else:
                logger.trace("Item in: %s", item)
            yield item
            if item == "EOF":
                break
