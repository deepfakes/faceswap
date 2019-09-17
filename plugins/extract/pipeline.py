#!/usr/bin/env python3
"""
Return a requested detector/aligner pipeline

Tensorflow does not like to release GPU VRAM, so parallel plugins need to be managed to work
together.

This module sets up a pipeline for the extraction workflow, loading align and detect plugins
either in parallal or in series, giving easy access to input and output.

 """

import logging

from lib.gpu_stats import GPUStats
from lib.queue_manager import queue_manager, QueueEmpty
from lib.utils import get_backend
from plugins.plugin_loader import PluginLoader

logger = logging.getLogger(__name__)  # pylint:disable=invalid-name


class Extractor():
    """ Creates a :mod:`~plugins.extract.detect`/:mod:`~plugins.extract.align` pipeline and yields
    results frame by frame from the :attr:`detected_faces` generator

    :attr:`input_queue` is dynamically set depending on the current :attr:`phase` of extraction

    Parameters
    ----------
    detector: str
        The name of a detector plugin as exists in :mod:`plugins.extract.detect`
    aligner: str
        The name of an aligner plugin as exists in :mod:`plugins.extract.align`
    configfile: str, optional
        The path to a custom ``extract.ini`` configfile. If ``None`` then the system
        :file:`config/extract.ini` file will be used.
    multiprocess: bool, optional
        Whether to attempt processing the plugins in parallel. This may get overridden
        internally depending on the plugin combination. Default: ``False``
    rotate_images: str, optional
        Used to set the :attr:`~plugins.extract.detect.rotation` attribute. Pass in a single number
        to use increments of that size up to 360, or pass in a ``list`` of ``ints`` to enumerate
        exactly what angles to check. Can also pass in ``'on'`` to increment at 90 degree
        intervals. Default: ``None``
    min_size: int, optional
        Used to set the :attr:`~plugins.extract.detect.min_size` attribute Filters out faces
        detected below this size. Length, in pixels across the diagonal of the bounding box. Set
        to ``0`` for off. Default: ``0``
    normalize_method: {`None`, 'clahe', 'hist', 'mean'}, optional
        Used to set the :attr:`~plugins.extract.align.normalize_method` attribute. Normalize the
        images fed to the aligner.Default: ``None``

    Attributes
    ----------
    phase: str
        The current phase that the pipeline is running. Used in conjunction with :attr:`passes` and
        :attr:`final_pass` to indicate to the caller which phase is being processed
    """
    def __init__(self, detector, aligner,
                 configfile=None, multiprocess=False, rotate_images=None, min_size=20,
                 normalize_method=None):
        logger.debug("Initializing %s: (detector: %s, aligner: %s, configfile: %s, "
                     "multiprocess: %s, rotate_images: %s, min_size: %s, "
                     "normalize_method: %s)", self.__class__.__name__, detector, aligner,
                     configfile, multiprocess, rotate_images, min_size, normalize_method)
        self.phase = "detect"
        self._queue_size = 32
        self._vram_buffer = 320  # Leave a buffer for VRAM allocation
        self._detector = self._load_detector(detector, rotate_images, min_size, configfile)
        self._aligner = self._load_aligner(aligner, configfile, normalize_method)
        self._is_parallel = self._set_parallel_processing(multiprocess)
        self._queues = self._add_queues()
        logger.debug("Initialized %s", self.__class__.__name__)

    @property
    def input_queue(self):
        """ queue: Return the correct input queue depending on the current phase

        The input queue is the entry point into the extraction pipeline. A ``dict`` should
        be put to the queue in the following format(s):

        For detect/single phase operations:

        >>> {'filename': <path to the source image that is to be extracted from>,
        >>>  'image': <the source image as a numpy.array in BGR color format>}

        For align (2nd pass operations):

        >>> {'filename': <path to the source image that is to be extracted from>,
        >>>  'image': <the source image as a numpy.array in BGR color format>,
        >>>  'detected_faces: [<list of DetectedFace objects as generated from detect>]}

        """
        if self._is_parallel or self.phase == "detect":
            qname = "extract_detect_in"
        else:
            qname = "extract_align_in"
        retval = self._queues[qname]
        logger.trace("%s: %s", qname, retval)
        return retval

    @property
    def passes(self):
        """ int: Returns the total number of passes the extractor needs to make.

        This is calculated on several factors (vram available, plugin choice,
        :attr:`multiprocess` etc.). It is useful for iterating over the pipeline
        and handling accordingly.

        Example
        -------
        >>> for phase in extractor.passes:
        >>>     if phase == 1:
        >>>         extractor.input_queue.put({"filename": "path/to/image/file",
        >>>                                    "image": np.array(image)})
        >>>     else:
        >>>         extractor.input_queue.put({"filename": "path/to/image/file",
        >>>                                    "image": np.array(image),
        >>>                                    "detected_faces": [<DetectedFace objects]})
        """
        retval = 1 if self._is_parallel else 2
        logger.trace(retval)
        return retval

    @property
    def final_pass(self):
        """ bool, Return ``True`` if this is the final extractor pass otherwise ``False``

        Useful for iterating over the pipeline :attr:`passes` or :func:`detected_faces` and
        handling accordingly.

        Example
        -------
        >>> for face in extractor.detected_faces():
        >>>     if extractor.final_pass:
        >>>         <do final processing>
        >>>     else:
        >>>         <do intermediate processing>
        >>>         extractor.input_queue.put({"filename": "path/to/image/file",
        >>>                                    "image": np.array(image),
        >>>                                    "detected_faces": [<DetectedFace objects]})
        """
        retval = self._is_parallel or self.phase == "align"
        logger.trace(retval)
        return retval

    def set_batchsize(self, plugin_type, batchsize):
        """ Set the batchsize of a given :attr:`plugin_type` to the given :attr:`batchsize`.

        This should be set prior to :func:`launch` if the batchsize is to be manually overriden

        Parameters
        ----------
        plugin_type: {'aligner', 'detector'}
            The plugin_type to be overriden
        batchsize: int
            The batchsize to use for this plugin type
        """
        logger.debug("Overriding batchsize for plugin_type: %s to: %s", plugin_type, batchsize)
        plugin = getattr(self, "_{}".format(plugin_type))
        plugin.batchsize = batchsize

    def launch(self):
        """ Launches the plugin(s)

        This launches the plugins held in the pipeline, and should be called at the beginning
        of each :attr:`phase`. To ensure VRAM is conserved, It will only launch the plugin(s)
        required for the currently running phase

        Example
        -------
        >>> for phase in extractor.passes:
        >>>     extractor.launch():
        >>>         <do processing>
        """

        if self._is_parallel:
            self._launch_aligner()
            self._launch_detector()
        elif self.phase == "detect":
            self._launch_detector()
        else:
            self._launch_aligner()

    def detected_faces(self):
        """ Generator that returns results, frame by frame from the extraction pipeline

        This is the exit point for the extraction pipeline and is used to obtain the output
        of any pipeline :attr:`phase`

        Yields
        ------
        faces: dict
            regardless of phase, the returned dictinary will contain, exclusively, ``filename``:
            the filename of the source image, ``image``: the ``numpy.array`` of the source image
            in BGR color format, ``detected_faces``: a list of
            :class:`~lib.faces_detect.Detected_Face` objects.

        Example
        -------
        >>> for face in extractor.detected_faces():
        >>>     filename = face["filename"]
        >>>     image = face["image"]
        >>>     detected_faces = face["detected_faces"]
        """
        logger.debug("Running Detection. Phase: '%s'", self.phase)
        # If not multiprocessing, intercept the align in queue for
        # detection phase
        out_queue = self._output_queue
        while True:
            try:
                if self._check_and_raise_error():
                    break
                faces = out_queue.get(True, 1)
                if faces == "EOF":
                    break
            except QueueEmpty:
                continue

            yield faces
        self._join_threads()
        if self.final_pass:
            # Cleanup queues
            for q_name in self._queues.keys():
                queue_manager.del_queue(q_name)
            logger.debug("Detection Complete")
        else:
            logger.debug("Switching to align phase")
            self.phase = "align"

    # <<< INTERNAL METHODS >>> #
    @property
    def _output_queue(self):
        """ Return the correct output queue depending on the current phase """
        qname = "extract_align_out" if self.final_pass else "extract_align_in"
        retval = self._queues[qname]
        logger.trace("%s: %s", qname, retval)
        return retval

    @property
    def _active_plugins(self):
        """ Return the plugins that are currently active based on pass """
        if self.passes == 1:
            retval = [self._detector, self._aligner]
        elif self.passes == 2 and not self.final_pass:
            retval = [self._detector]
        else:
            retval = [self._aligner]
        logger.trace("Active plugins: %s", retval)
        return retval

    def _add_queues(self):
        """ Add the required processing queues to Queue Manager """
        queues = dict()
        for task in ("extract_detect_in", "extract_align_in", "extract_align_out"):
            # Limit queue size to avoid stacking ram
            self._queue_size = 32
            if task == "extract_detect_in" or (not self._is_parallel
                                               and task == "extract_align_in"):
                self._queue_size = 64
            queue_manager.add_queue(task, maxsize=self._queue_size)
            queues[task] = queue_manager.get_queue(task)
        logger.debug("Queues: %s", queues)
        return queues

    def _set_parallel_processing(self, multiprocess):
        """ Set whether to run detect and align together or separately """
        if self._detector.vram == 0 or self._aligner.vram == 0:
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

        vram_required = self._detector.vram + self._aligner.vram + self._vram_buffer
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

        self._set_extractor_batchsize(vram_required, vram_free)
        return True

    # << INTERNAL PLUGIN HANDLING >> #
    @staticmethod
    def _load_detector(detector, rotation, min_size, configfile):
        """ Set global arguments and load detector plugin """
        detector_name = detector.replace("-", "_").lower()
        logger.debug("Loading Detector: '%s'", detector_name)
        detector = PluginLoader.get_detector(detector_name)(rotation=rotation,
                                                            min_size=min_size,
                                                            configfile=configfile)
        return detector

    @staticmethod
    def _load_aligner(aligner, configfile, normalize_method):
        """ Set global arguments and load aligner plugin """
        aligner_name = aligner.replace("-", "_").lower()
        logger.debug("Loading Aligner: '%s'", aligner_name)
        aligner = PluginLoader.get_aligner(aligner_name)(configfile=configfile,
                                                         normalize_method=normalize_method)
        return aligner

    def _launch_aligner(self):
        """ Launch the face aligner """
        logger.debug("Launching Aligner")
        kwargs = dict(in_queue=self._queues["extract_align_in"],
                      out_queue=self._queues["extract_align_out"])
        self._aligner.initialize(**kwargs)
        self._aligner.start()
        logger.debug("Launched Aligner")

    def _launch_detector(self):
        """ Launch the face detector """
        logger.debug("Launching Detector")
        kwargs = dict(in_queue=self._queues["extract_detect_in"],
                      out_queue=self._queues["extract_align_in"])
        self._detector.initialize(**kwargs)
        self._detector.start()
        logger.debug("Launched Detector")

    def _set_extractor_batchsize(self, vram_required, vram_free):
        """ Sets the batchsize of the used plugins based on their vram and
            vram_per_batch_requirements """
        batch_required = ((self._aligner.vram_per_batch * self._aligner.batchsize) +
                          (self._detector.vram_per_batch * self._detector.batchsize))
        plugin_required = vram_required + batch_required
        if plugin_required <= vram_free:
            logger.verbose("Plugin requirements within threshold: (plugin_required: %sMB, "
                           "vram_free: %sMB)", plugin_required, vram_free)
            return
        # Hacky split across 2 plugins
        available_for_batching = (vram_free - vram_required) // 2
        self._aligner.batchsize = max(1, available_for_batching // self._aligner.vram_per_batch)
        self._detector.batchsize = max(1, available_for_batching // self._detector.vram_per_batch)
        logger.verbose("Reset batchsizes: (aligner: %s, detector: %s)",
                       self._aligner.batchsize, self._detector.batchsize)

    def _join_threads(self):
        """ Join threads for current pass """
        for plugin in self._active_plugins:
            plugin.join()

    def _check_and_raise_error(self):
        """ Check all threads for errors and raise if one occurs """
        for plugin in self._active_plugins:
            if plugin.check_and_raise_error():
                return True
        return False
