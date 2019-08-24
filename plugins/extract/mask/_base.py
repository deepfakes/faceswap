#!/usr/bin/env python3
""" Base class for Face Masker plugins
    Plugins should inherit from this class

    See the override methods for which methods are required.

    The plugin will receive a dict containing:
    {"filename": <filename of source frame>,
     "image": <source image>,
     "detected_faces": <list of bounding box dicts as defined in lib/plugins/extract/detect/_base>}

    For each source item, the plugin must pass a dict to finalize containing:
    {"filename": <filename of source frame>,
     "image": <four channel source image>,
     "detected_faces": <list of bounding box dicts as defined in lib/plugins/extract/detect/_base>,
     "mask": <one channel mask image>}
    """

import logging
import os
import traceback
import cv2
import numpy as np

from io import StringIO

from lib.aligner import Extract
from lib.gpu_stats import GPUStats
from lib.utils import GetModel

logger = logging.getLogger(__name__)  # pylint:disable=invalid-name


def get_available_masks():
    """ Return a list of the available masks for cli """
    masks = ["none", "components", "extended", "vgg_300", "vgg_500", "unet_256"]
    logger.debug(masks)
    return masks


def get_default_mask():
    """ Set the default mask for cli """
    masks = get_available_masks()
    default = "components" if "components" in masks else masks[0]
    logger.debug("Default mask is %s", default)
    return default


class Masker():
    """ Face Mask Object
    Faces may be of shape (batch_size, height, width, 3) or (height, width, 3)
        of dtype float32 and with range[0., 1.]
        Landmarks may be of shape (batch_size, 68, 2) or (68, 2)
        Produced mask will be in range [0, 1.]
        the output mask will be <mask_type>.mask
        channels: 1, 3 or 4:
                    1 - Returns a single channel mask
                    3 - Returns a 3 channel mask
                    4 - Returns the original image with the mask in the alpha channel """
    def __init__(self, loglevel, configfile=None, git_model_id=None,
                 model_filename=None, input_size=256):
        logger.debug("Initializing %s: (loglevel: %s, configfile: %s, git_model_id: %s, "
                     "model_filename: '%s', input_size: %s)", self.__class__.__name__, loglevel,
                     configfile, git_model_id, model_filename, input_size)
        self.loglevel = loglevel
        self.input_size = input_size
        self.extract = Extract()
        self.init = None
        self.error = None

        # The input and output queues for the plugin.
        # See lib.queue_manager.QueueManager for getting queues
        self.queues = {"in": None, "out": None}

        #  Get model if required
        self.model_path = self.get_model(git_model_id, model_filename)

        # Approximate VRAM required for masker. Used to calculate
        # how many parallel processes / batches can be run.
        # Be conservative to avoid OOM.
        self.vram = None

        # Set to true if the plugin supports PlaidML
        self.supports_plaidml = False

        logger.debug("Initialized %s", self.__class__.__name__)

    # <<< OVERRIDE METHODS >>> #
    # These methods must be overriden when creating a plugin
    def initialize(self, *args, **kwargs):
        """ Inititalize the masker
            Tasks to be run before any masking is performed.
            Override for specific masker """
        logger.debug("_base initialize %s: (PID: %s, args: %s, kwargs: %s)",
                     self.__class__.__name__, os.getpid(), args, kwargs)
        self.init = kwargs["event"]
        self.error = kwargs["error"]
        self.queues["in"] = kwargs["in_queue"]
        self.queues["out"] = kwargs["out_queue"]

    def build_masks(self, faces, means, landmarks):
        """ Override to build the mask """
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

    # <<< MASKING WRAPPER >>> #
    def run(self, *args, **kwargs):
        """ Parent align process.
            This should always be called as the entry point so exceptions
            are passed back to parent.
            Do not override """
        try:
            self.mask(*args, **kwargs)
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

    def mask(self, *args, **kwargs):
        """ Process masks """
        if not self.init:
            self.initialize(*args, **kwargs)
        logger.debug("Launching Mask: (args: %s kwargs: %s)", args, kwargs)

        for item in self.get_item():
            if item == "EOF":
                self.finalize(item)
                break

            logger.trace("Masking faces")
            try:
                item["image"], item["mask"] = self.process_masks(item["image"], item["landmarks"])
                logger.trace("Masked faces: %s", item["filename"])
            except ValueError as err:
                logger.warning("Image '%s' could not be processed. This may be due to corrupted "
                               "data: %s", item["filename"], str(err))
                item["detected_faces"] = list()
                item["mask"] = list()
                # UNCOMMENT THIS CODE BLOCK TO PRINT TRACEBACK ERRORS
                import sys
                exc_info = sys.exc_info()
                traceback.print_exception(*exc_info)
            self.finalize(item)
        logger.debug("Completed Mask")

    def process_masks(self, image, landmarks):
        """ Align image and process landmarks """
        logger.trace("Processing masks")
        retval = list()
        faces, masks = self.build_masks(image, np.array(landmarks))
        logger.trace("Processed masks")
        return faces, masks

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
