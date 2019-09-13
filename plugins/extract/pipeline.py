#!/usr/bin/env python3
"""
Return a requested detector/aligner pipeline

Tensorflow does not like to release GPU VRAM, so these are launched in subprocesses
so that the vram is released on subprocess exit """

import logging

from lib.gpu_stats import GPUStats
from lib.queue_manager import queue_manager, QueueEmpty
from lib.utils import get_backend
from plugins.plugin_loader import PluginLoader

logger = logging.getLogger(__name__)  # pylint:disable=invalid-name


class Extractor():
    """ Creates a detect/align pipeline and returns results from a generator

        Input queue is dynamically set depending on the current phase of extraction
        and can be accessed from:
            Extractor.input_queue
    """
    def __init__(self, detector, aligner,
                 configfile=None, multiprocess=False, rotate_images=None, min_size=20,
                 normalize_method=None):
        logger.debug("Initializing %s: (detector: %s, aligner: %s, configfile: %s, "
                     "multiprocess: %s, rotate_images: %s, min_size: %s, "
                     "normalize_method: %s)", self.__class__.__name__, detector, aligner,
                     configfile, multiprocess, rotate_images, min_size, normalize_method)
        self.vram_buffer = 320
        self.phase = "detect"
        self.queue_size = 32
        self.detector = self.load_detector(detector, rotate_images, min_size, configfile)
        self.aligner = self.load_aligner(aligner, configfile, normalize_method)
        self.is_parallel = self.set_parallel_processing(multiprocess)
        self.queues = self.add_queues()
        self.threads = []
        logger.debug("Initialized %s", self.__class__.__name__)

    @property
    def input_queue(self):
        """ Return the correct input queue depending on the current phase """
        if self.is_parallel or self.phase == "detect":
            qname = "extract_detect_in"
        else:
            qname = "extract_align_in"
        retval = self.queues[qname]
        logger.trace("%s: %s", qname, retval)
        return retval

    @property
    def output_queue(self):
        """ Return the correct output queue depending on the current phase """
        qname = "extract_align_out" if self.final_pass else "extract_align_in"
        retval = self.queues[qname]
        logger.trace("%s: %s", qname, retval)
        return retval

    @property
    def passes(self):
        """ Return the number of passes the extractor needs to make """
        retval = 1 if self.is_parallel else 2
        logger.trace(retval)
        return retval

    @property
    def final_pass(self):
        """ Return true if this is the final extractor pass """
        retval = self.is_parallel or self.phase == "align"
        logger.trace(retval)
        return retval

    @property
    def active_plugins(self):
        """ Return the plugins that are currently active based on pass """
        if self.passes == 1:
            retval = [self.detector, self.aligner]
        elif self.passes == 2 and not self.final_pass:
            retval = [self.detector]
        else:
            retval = [self.aligner]
        logger.trace("Active plugins: %s", retval)
        return retval

    @staticmethod
    def load_detector(detector, rotation, min_size, configfile):
        """ Set global arguments and load detector plugin """
        detector_name = detector.replace("-", "_").lower()
        logger.debug("Loading Detector: '%s'", detector_name)
        detector = PluginLoader.get_detector(detector_name)(rotation=rotation,
                                                            min_size=min_size,
                                                            configfile=configfile)
        return detector

    @staticmethod
    def load_aligner(aligner, configfile, normalize_method):
        """ Set global arguments and load aligner plugin """
        aligner_name = aligner.replace("-", "_").lower()
        logger.debug("Loading Aligner: '%s'", aligner_name)
        aligner = PluginLoader.get_aligner(aligner_name)(configfile=configfile,
                                                         normalize_method=normalize_method)
        return aligner

    def set_parallel_processing(self, multiprocess):
        """ Set whether to run detect and align together or separately """
        if self.detector.vram == 0 or self.aligner.vram == 0:
            logger.debug("At least one of aligner or detector have no VRAM requirement. "
                         "Enabling parallel processing.")
            return True

        if not multiprocess:
            logger.debug("Parallel processing disabled by cli.")
            return False

        gpu_stats = GPUStats()
        if gpu_stats.device_count == 0:
            logger.debug("No GPU detected. Enabling parallel processing.")
            return True

        if get_backend() == "amd":
            logger.debug("Parallel processing discabled by amd")
            return False

        vram_required = self.detector.vram + self.aligner.vram + self.vram_buffer
        stats = gpu_stats.get_card_most_free()
        vram_free = int(stats["free"])
        logger.verbose("%s - %sMB free of %sMB",
                       stats["device"],
                       vram_free,
                       int(stats["total"]))
        if vram_free <= vram_required:
            logger.warning("Not enough free VRAM for parallel processing. "
                           "Switching to serial")
            return False

        self.set_extractor_batchsize(vram_required, vram_free)
        return True

    def set_extractor_batchsize(self, vram_required, vram_free):
        """ Sets the batchsize of the used plugins based on their vram and
            vram_per_batch_requirements """
        batch_required = ((self.aligner.vram_per_batch * self.aligner.batchsize) +
                          (self.detector.vram_per_batch * self.detector.batchsize))
        plugin_required = vram_required + batch_required
        if plugin_required <= vram_free:
            logger.verbose("Plugin requirements within threshold: (plugin_required: %sMB, "
                           "vram_free: %sMB)", plugin_required, vram_free)
            return
        # Hacky split across 2 plugins
        available_for_batching = (vram_free - vram_required) // 2
        self.aligner.batchsize = max(1, available_for_batching // self.aligner.vram_per_batch)
        self.detector.batchsize = max(1, available_for_batching // self.detector.vram_per_batch)
        logger.verbose("Reset batchsizes: (aligner: %s, detector: %s)",
                       self.aligner.batchsize, self.detector.batchsize)

    def add_queues(self):
        """ Add the required processing queues to Queue Manager """
        queues = dict()
        for task in ("extract_detect_in", "extract_align_in", "extract_align_out"):
            # Limit queue size to avoid stacking ram
            self.queue_size = 32
            if task == "extract_detect_in" or (not self.is_parallel
                                               and task == "extract_align_in"):
                self.queue_size = 64
            queue_manager.add_queue(task, maxsize=self.queue_size, multiprocessing_queue=False)
            queues[task] = queue_manager.get_queue(task)
        logger.debug("Queues: %s", queues)
        return queues

    def launch(self):
        """ Launches the plugins
            This can be called multiple times depending on the phase/whether multiprocessing
            is enabled.

            If multiprocessing:
                launches both plugins, but aligner first so that it's VRAM can be allocated
                prior to giving the remaining to the detector
            If not multiprocessing:
                Launches the relevant plugin for the current phase """
        if self.is_parallel:
            self.launch_aligner()
            self.launch_detector()
        elif self.phase == "detect":
            self.launch_detector()
        else:
            self.launch_aligner()

    def launch_aligner(self):
        """ Launch the face aligner """
        logger.debug("Launching Aligner")
        kwargs = dict(in_queue=self.queues["extract_align_in"],
                      out_queue=self.queues["extract_align_out"],
                      queue_size=self.queue_size)
        self.aligner.initialize(**kwargs)
        self.aligner.start()
        logger.debug("Launched Aligner")

    def launch_detector(self):
        """ Launch the face detector """
        logger.debug("Launching Detector")
        kwargs = dict(in_queue=self.queues["extract_detect_in"],
                      out_queue=self.queues["extract_align_in"],
                      queue_size=self.queue_size)
        self.detector.initialize(**kwargs)
        self.detector.start()
        logger.debug("Launched Detector")

    def detected_faces(self):
        """ Detect faces from in an image """
        logger.debug("Running Detection. Phase: '%s'", self.phase)
        # If not multiprocessing, intercept the align in queue for
        # detection phase
        out_queue = self.output_queue
        while True:
            try:
                if self.check_and_raise_error():
                    break
                faces = out_queue.get(True, 1)
                if faces == "EOF":
                    break
            except QueueEmpty:
                continue

            yield faces
        self.join_threads()
        if self.final_pass:
            # Cleanup queues
            for q_name in self.queues.keys():
                queue_manager.del_queue(q_name)
            logger.debug("Detection Complete")
        else:
            logger.debug("Switching to align phase")
            self.phase = "align"

    def check_and_raise_error(self):
        """ Check all threads for errors and raise if one occurs """
        for plugin in self.active_plugins:
            if plugin.check_and_raise_error():
                return True
        return False

    def join_threads(self):
        """ Join threads for current pass """
        for plugin in self.active_plugins:
            plugin.join()
