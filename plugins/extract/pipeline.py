#!/usr/bin/env python3
"""
Return a requested detector/aligner pipeline

Tensorflow does not like to release GPU VRAM, so these are launched in subprocesses
so that the vram is released on subprocess exit """

import logging

from lib.gpu_stats import GPUStats
from lib.multithreading import PoolProcess, SpawnProcess
from lib.queue_manager import queue_manager, QueueEmpty
from plugins.plugin_loader import PluginLoader

logger = logging.getLogger(__name__)  # pylint:disable=invalid-name


class Extractor():
    """ Creates a detect/align pipeline and returns results from a generator

        Input queue is dynamically set depending on the current phase of extraction
        and can be accessed from:
            Extractor.input_queue
    """
    def __init__(self, detector, aligner, masker, loglevel,
                 configfile=None, multiprocess=False, rotate_images=None, min_size=20,
                 normalize_method=None, input_size=256, output_size=256, coverage_ratio=1.):
        logger.debug("Initializing %s: (detector: %s, aligner: %s, masker: %s, loglevel: %s, "
                     "configfile: %s, multiprocess: %s, rotate_images: %s, min_size: %s, "
                     "normalize_method: %s, input_size: %s, output_size: %s, coverage_ratio: %s)",
                     self.__class__.__name__, detector, aligner, masker, loglevel, configfile,
                     multiprocess, rotate_images, min_size, normalize_method, input_size,
                     output_size, coverage_ratio)
        self.phase = "detect"
        self.detector = self.load_detector(detector, loglevel, rotate_images, min_size, configfile)
        self.aligner = self.load_aligner(aligner, loglevel, configfile, normalize_method)
        self.masker = self.load_masker(masker,
                                       loglevel,
                                       configfile,
                                       input_size,
                                       output_size,
                                       coverage_ratio)
        self.is_parallel = self.set_parallel_processing(multiprocess)
        self.processes = list()
        self.queues = self.add_queues()
        logger.debug("Initialized %s", self.__class__.__name__)

    @property
    def input_queue(self):
        """ Return the correct input queue depending on the current phase """
        qname_dict = {"detect":  "extract_detect_in",
                      "align":   "extract_align_in",
                      "mask":    "extract_mask_in"}
        qname = "extract_detect_in" if self.is_parallel else qname_dict[self.phase]
        retval = self.queues[qname]
        logger.trace("%s: %s", qname, retval)
        return retval

    @property
    def output_queue(self):
        """ Return the correct output queue depending on the current phase """
        qname_dict = {"detect":  "extract_align_in",
                      "align":   "extract_mask_in",
                      "mask":    "extract_mask_out"}
        qname = "extract_mask_out" if self.final_pass else qname_dict[self.phase]
        retval = self.queues[qname]
        logger.trace("%s: %s", qname, retval)
        return retval

    @property
    def passes(self):
        """ Return the number of passes the extractor needs to make """
        retval = 1 if self.is_parallel else 3
        logger.trace(retval)
        return retval

    @property
    def final_pass(self):
        """ Return true if this is the final extractor pass """
        retval = self.is_parallel or self.phase == "mask"
        logger.trace(retval)
        return retval

    @staticmethod
    def load_detector(detector, loglevel, rotation, min_size, configfile):
        """ Set global arguments and load detector plugin """
        detector_name = detector.replace("-", "_").lower()
        logger.debug("Loading Detector: '%s'", detector_name)
        detector = PluginLoader.get_detector(detector_name)(loglevel=loglevel,
                                                            rotation=rotation,
                                                            min_size=min_size,
                                                            configfile=configfile)
        return detector

    @staticmethod
    def load_aligner(aligner, loglevel, configfile, normalize_method):
        """ Set global arguments and load aligner plugin """
        aligner_name = aligner.replace("-", "_").lower()
        logger.debug("Loading Aligner: '%s'", aligner_name)
        aligner = PluginLoader.get_aligner(aligner_name)(loglevel=loglevel,
                                                         configfile=configfile,
                                                         normalize_method=normalize_method)
        return aligner

    @staticmethod
    def load_masker(masker, loglevel, configfile, input_size, output_size, coverage_ratio):
        """ Set global arguments and load masker plugin """
        masker_name = masker.replace("-", "_").lower()
        logger.debug("Loading Masker: '%s'", masker_name)
        masker = PluginLoader.get_masker(masker_name)(loglevel=loglevel,
                                                      configfile=configfile,
                                                      input_size=input_size,
                                                      output_size=output_size,
                                                      coverage_ratio=coverage_ratio)
        return masker

    def set_parallel_processing(self, multiprocess):
        """ Set whether to run detect, align, and mask together or separately """
        detector_vram = self.detector.vram
        aligner_vram = self.aligner.vram
        masker_vram = self.masker.vram

        if not multiprocess:
            logger.debug("Parallel processing disabled by cli.")
            return False

        if detector_vram == 0 or aligner_vram == 0 or masker_vram == 0:
            logger.debug("At least one of aligner, detector or masker have no VRAM requirements. "
                         "Enabling parallel processing.")
            return True

        gpu_stats = GPUStats()
        if gpu_stats.device_count == 0:
            logger.debug("No GPU detected. Enabling parallel processing.")
            return True

        if gpu_stats.is_plaidml:
            return (bool(self.detector.supports_plaidml) +
                    bool(self.aligner.supports_plaidml) +
                    bool(self.masker.supports_plaidml)) <= 1

        required_vram = detector_vram + aligner_vram + masker_vram + 320  # 320MB buffer
        stats = gpu_stats.get_card_most_free()
        free_vram = int(stats["free"])
        logger.verbose("%s - %sMB free of %sMB", stats["device"], free_vram, int(stats["total"]))
        if free_vram <= required_vram:
            logger.warning("Not enough free VRAM for parallel processing. Switching to serial")
            return False
        return True

    def add_queues(self):
        """ Add the required processing queues to Queue Manager """
        queues = dict()
        tasks = ["extract_detect_in", "extract_align_in", "extract_mask_in", "extract_mask_out"]
        for task in tasks:
            # Limit queue size to avoid stacking ram
            size = 32
            if task == "extract_detect_in" or (not self.is_parallel and
                                               task == "extract_align_in"):
                size = 64
            queue_manager.add_queue(task, maxsize=size)
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
        print('in launch')
        if self.is_parallel:
            print('in parallel')
            logger.debug("Launching aligner and detector")
            self.launch_aligner()
            self.launch_detector()
            self.launch_masker()
        elif self.phase == "detect":
            print('in detector')
            logger.debug("Launching detector")
            self.launch_detector()
        elif self.phase == "align":
            logger.debug("Launching aligner")
            self.launch_aligner()
        else:
            print('in masker')
            logger.debug("Launching masker")
            self.launch_masker()

    def launch_masker(self):
        """ Launch the face masker """
        logger.debug("Launching Masker")
        kwargs = {"in_queue": self.queues["extract_mask_in"],
                  "out_queue": self.queues["extract_mask_out"]}

        mp_func = PoolProcess if self.masker.parent_is_pool else SpawnProcess
        process = mp_func(self.masker.run, **kwargs)
        event = process.event if hasattr(process, "event") else None
        error = process.error if hasattr(process, "error") else None
        process.start()
        self.processes.append(process)

        if event is None:
            logger.debug("Launched Masker")
            return

        # Wait for Masker to take it's VRAM
        for mins in reversed(range(5)):
            for seconds in range(60):
                event.wait(seconds)
                if event.is_set():
                    break
                if error.is_set():
                    break
            if event.is_set():
                break
            if mins == 0 or error.is_set():
                raise ValueError("Error initializing Masker")
            logger.info("Waiting for Masker... Time out in %s minutes", mins)

        logger.debug("Launched Masker")

    def launch_aligner(self):
        """ Launch the face aligner """
        logger.debug("Launching Aligner")
        kwargs = {"in_queue": self.queues["extract_align_in"],
                  "out_queue": self.queues["extract_mask_in"]}

        process = SpawnProcess(self.aligner.run, **kwargs)
        event = process.event
        error = process.error
        process.start()
        self.processes.append(process)

        # Wait for Aligner to take it's VRAM
        for mins in reversed(range(5)):
            for seconds in range(60):
                event.wait(seconds)
                if event.is_set():
                    break
                if error.is_set():
                    break
            if event.is_set():
                break
            if mins == 0 or error.is_set():
                raise ValueError("Error initializing Aligner")
            logger.info("Waiting for Aligner... Time out in %s minutes", mins)

        logger.debug("Launched Aligner")

    def launch_detector(self):
        """ Launch the face detector """
        logger.debug("Launching Detector")
        kwargs = {"in_queue": self.queues["extract_detect_in"],
                  "out_queue": self.queues["extract_align_in"]}
        mp_func = PoolProcess if self.detector.parent_is_pool else SpawnProcess
        process = mp_func(self.detector.run, **kwargs)

        event = process.event if hasattr(process, "event") else None
        error = process.error if hasattr(process, "error") else None
        process.start()
        self.processes.append(process)

        if event is None:
            logger.debug("Launched Detector")
            return

        # Wait for Detector to take it's VRAM
        for mins in reversed(range(5)):
            for seconds in range(60):
                event.wait(seconds)
                if event.is_set():
                    break
                if error and error.is_set():
                    break
            if event.is_set():
                break
            if mins == 0 or (error and error.is_set()):
                raise ValueError("Error initializing Detector")
            logger.info("Waiting for Detector... Time out in %s minutes", mins)

        logger.debug("Launched Detector")

    def detected_faces(self):
        """ Detect faces from in an image """
        logger.debug("Running Detection. Phase: '%s'", self.phase)
        # If not multiprocessing, intercept the align in queue for
        # detection phase
        out_queue = self.output_queue
        while True:
            try:
                faces = out_queue.get(True, 1)
                if faces == "EOF":
                    break
                if isinstance(faces, dict) and faces.get("exception"):
                    pid = faces["exception"][0]
                    t_back = faces["exception"][1].getvalue()
                    err = "Error in child process {}. {}".format(pid, t_back)
                    raise Exception(err)
            except QueueEmpty:
                continue

            yield faces
        for process in self.processes:
            logger.trace("Joining process: %s", process)
            process.join()
            del process
        if self.final_pass:
            # Cleanup queues
            for q_name in self.queues.keys():
                queue_manager.del_queue(q_name)
            logger.debug("Detection Complete")
        else:
            self.phase = "align" if self.phase == "detect" else "mask"
            logger.debug("Switching to %s phase", self.phase)
