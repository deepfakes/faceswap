#!/usr/bin/env python3
""" Base class for Face Aligner plugins
    Plugins should inherit from this class

    See the override methods for which methods are
    required.

    The plugin will receive a dict containing:
    {"filename": <filename of source frame>,
     "image": <source image>,
     "detected_faces": <list of DlibRectangles>}

    For each source item, the plugin must pass a dict to finalize containing:
    {"filename": <filename of source frame>,
     "image": <source image>,
     "detected_faces": <list of dlibRectangles>,
     "landmarks": <list of landmarks>}
    """

import logging
import os
import traceback

from io import StringIO

from lib.aligner import Extract
from lib.gpu_stats import GPUStats

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class Aligner():
    """ Landmarks Aligner Object """
    def __init__(self, loglevel):
        logger.debug("Initializing %s", self.__class__.__name__)
        self.loglevel = loglevel
        self.cachepath = os.path.join(os.path.dirname(__file__), ".cache")
        self.extract = Extract()
        self.init = None
        self.error = None

        # The input and output queues for the plugin.
        # See lib.queue_manager.QueueManager for getting queues
        self.queues = {"in": None, "out": None}

        #  Path to model if required
        self.model_path = self.set_model_path()

        # Approximate VRAM required for aligner. Used to calculate
        # how many parallel processes / batches can be run.
        # Be conservative to avoid OOM.
        self.vram = None
        logger.debug("Initialized %s", self.__class__.__name__)

    # <<< OVERRIDE METHODS >>> #
    # These methods must be overriden when creating a plugin
    @staticmethod
    def set_model_path():
        """ path to data file/models
            override for specific detector """
        raise NotImplementedError()

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

    def align(self, *args, **kwargs):
        """ Process landmarks
            Override for specific detector
            Must return a list of dlib rects"""
        if not self.init:
            self.initialize(*args, **kwargs)
        logger.debug("Launching Align: (args: %s kwargs: %s)", args, kwargs)

    # <<< DETECTION WRAPPER >>> #
    def run(self, *args, **kwargs):
        """ Parent align process.
            This should always be called as the entry point so exceptions
            are passed back to parent.
            Do not override """
        try:
            self.align(*args, **kwargs)
        except Exception:  # pylint: disable=broad-except
            logger.error("Caught exception in child process: %s", os.getpid())
            # Display traceback if in initialization stage
            if not self.init.is_set():
                logger.exception("Traceback:")
            tb_buffer = StringIO()
            traceback.print_exc(file=tb_buffer)
            exception = {"exception": (os.getpid(), tb_buffer)}
            self.queues["out"].put(exception)
            exit(1)

    # <<< FINALIZE METHODS>>> #
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
