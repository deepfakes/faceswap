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
     "detected_faces": <list of tuples containing (dlibRectangle, Landmarks)>}
    """

import os

from lib.aligner import Extract
from lib.gpu_stats import GPUStats


class Aligner():
    """ Landmarks Aligner Object """
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.cachepath = os.path.join(os.path.dirname(__file__), ".cache")
        self.extract = Extract()
        self.init = None

        # The input and output queues for the plugin.
        # See lib.multithreading.QueueManager for getting queues
        self.queues = {"in": None, "out": None}

        #  Path to model if required
        self.model_path = self.set_model_path()

        # Approximate VRAM required for aligner. Used to calculate
        # how many parallel processes / batches can be run.
        # Be conservative to avoid OOM.
        self.vram = None

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
        self.init = kwargs["event"]
        self.queues["in"] = kwargs["in_queue"]
        self.queues["out"] = kwargs["out_queue"]

    def align(self, *args, **kwargs):
        """ Process landmarks
            Override for specific detector
            Must return a list of dlib rects"""
        try:
            if not self.init:
                self.initialize(*args, **kwargs)
        except ValueError as err:
            print("ERROR: {}".format(err))
            exit(1)

    # <<< FINALIZE METHODS>>> #
    def finalize(self, output):
        """ This should be called as the final task of each plugin
            aligns faces and puts to the out queue """
        if output == "EOF":
            self.queues["out"].put("EOF")
            return
        self.queues["out"].put((output))

    # <<< MISC METHODS >>> #
    def get_vram_free(self):
        """ Return free and total VRAM on card with most VRAM free"""
        stats = GPUStats()
        vram = stats.get_card_most_free()
        if self.verbose:
            print("Using device {} with {}MB free of {}MB".format(
                vram["device"],
                int(vram["free"]),
                int(vram["total"])))
        return int(vram["card_id"]), int(vram["free"]), int(vram["total"])
