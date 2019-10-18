#!/usr/bin/env python3
"""
Return a requested detector/aligner pipeline

Tensorflow does not like to release GPU VRAM, so parallel plugins need to be managed to work
together.

This module sets up a pipeline for the extraction workflow, loading align and detect plugins
either in parallel or in series, giving easy access to input and output.

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
    def __init__(self, detector, aligner, masker, configfile=None,
                 multiprocess=False, rotate_images=None, min_size=20,
                 normalize_method=None):
        logger.debug("Initializing %s: (detector: %s, aligner: %s, masker: %s, "
                     "configfile: %s, multiprocess: %s, rotate_images: %s, min_size: %s, "
                     "normalize_method: %s)",
                     self.__class__.__name__, detector, aligner, masker, configfile,
                     multiprocess, rotate_images, min_size, normalize_method)
        self._flow = ["detect", "align", "mask"]
        self.phase = self._flow[0]
        self._queue_size = 32
        self._vram_buffer = 320  # Leave a buffer for VRAM allocation
        self._detect = self._load_detect(detector, rotate_images, min_size, configfile)
        self._align = self._load_align(aligner, configfile, normalize_method)
        self._mask = self._load_mask(masker, configfile)
        self._is_parallel = self._set_parallel_processing(multiprocess)
        self._set_extractor_batchsize()
        self._queues = self._add_queues()
        logger.debug("Initialized %s", self.__class__.__name__)

    @property
    def input_queue(self):
        """ queue: Return the correct input queue depending on the current phase

        The input queue is the entry point into the extraction pipeline. A ``dict`` should
        be put to the queue in the following format(s):

        For detect/single phase operations:

        >>> {'filename': <path to the source image that is to be extracted from>,
        >>>  'image': <the source image as a numpy.ndarray in BGR color format>}

        For align (2nd pass operations):

        >>> {'filename': <path to the source image that is to be extracted from>,
        >>>  'image': <the source image as a numpy.ndarray in BGR color format>,
        >>>  'detected_faces: [<list of DetectedFace objects as generated from detect>]}

        """
        qname = "extract_{}_in".format(self.phase)
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
        >>>                                    "image": numpy.array(image)})
        >>>     else:
        >>>         extractor.input_queue.put({"filename": "path/to/image/file",
        >>>                                    "image": numpy.array(image),
        >>>                                    "detected_faces": [<DetectedFace objects]})
        """
        retval = 1 if self._is_parallel else len(self._flow)
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
        >>>                                    "image": numpy.array(image),
        >>>                                    "detected_faces": [<DetectedFace objects]})
        """
        retval = self._is_parallel or self.phase == self._final_phase
        logger.trace(retval)
        return retval

    def set_batchsize(self, plugin_type, batchsize):
        """ Set the batch size of a given :attr:`plugin_type` to the given :attr:`batchsize`.

        This should be set prior to :func:`launch` if the batch size is to be manually overridden

        Parameters
        ----------
        plugin_type: {'aligner', 'detector'}
            The plugin_type to be overridden
        batchsize: int
            The batch size to use for this plugin type
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
            for phase in self._flow:
                self._launch_plugin(phase)
        else:
            self._launch_plugin(self.phase)

    def detected_faces(self):
        """ Generator that returns results, frame by frame from the extraction pipeline

        This is the exit point for the extraction pipeline and is used to obtain the output
        of any pipeline :attr:`phase`

        Yields
        ------
        faces: dict
            regardless of phase, the returned dictionary will contain, exclusively, ``filename``:
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
            for q_name in self._queues:
                queue_manager.del_queue(q_name)
            logger.debug("Detection Complete")
        else:
            self.phase = self._next_phase
            logger.debug("Switching to %s phase", self.phase)

    # <<< INTERNAL METHODS >>> #
    @property
    def _next_phase(self):
        """ Return the next phase from the flow list """
        retval = self._flow[self._flow.index(self.phase) + 1]
        logger.trace(retval)
        return retval

    @property
    def _final_phase(self):
        """ Return the final phase from the flow list """
        retval = self._flow[-1]
        logger.trace(retval)
        return retval

    @property
    def _total_vram_required(self):
        """ Return vram required for all phases plus the buffer """
        retval = sum([getattr(self, "_{}".format(p)).vram for p in self._flow]) + self._vram_buffer
        logger.trace(retval)
        return retval

    @property
    def _output_queue(self):
        """ Return the correct output queue depending on the current phase """
        if self.final_pass:
            qname = "extract_{}_out".format(self._final_phase)
        else:
            qname = "extract_{}_in".format(self._next_phase)
        retval = self._queues[qname]
        logger.trace("%s: %s", qname, retval)
        return retval

    @property
    def _all_plugins(self):
        """ Return list of all plugin objects in this pipeline """
        retval = [getattr(self, "_{}".format(phase)) for phase in self._flow]
        logger.trace("All Plugins: %s", retval)
        return retval

    @property
    def _active_plugins(self):
        """ Return the plugins that are currently active based on pass """
        if self.passes == 1:
            retval = self._all_plugins
        else:
            retval = [getattr(self, "_{}".format(self.phase))]
        logger.trace("Active plugins: %s", retval)
        return retval

    def _add_queues(self):
        """ Add the required processing queues to Queue Manager """
        queues = dict()
        tasks = ["extract_{}_in".format(phase) for phase in self._flow]
        tasks.append("extract_{}_out".format(self._final_phase))
        for task in tasks:
            # Limit queue size to avoid stacking ram
            self._queue_size = 32
            if task == "extract_{}_in".format(self._flow[0]) or (not self._is_parallel
                                                                 and not task.endswith("_out")):
                self._queue_size = 64
            queue_manager.add_queue(task, maxsize=self._queue_size)
            queues[task] = queue_manager.get_queue(task)
        logger.debug("Queues: %s", queues)
        return queues

    def _set_parallel_processing(self, multiprocess):
        """ Set whether to run detect, align, and mask together or separately """

        if not multiprocess:
            logger.debug("Parallel processing disabled by cli.")
            return False

        gpu_stats = GPUStats()
        if gpu_stats.device_count == 0:
            logger.debug("No GPU detected. Enabling parallel processing.")
            return True

        if get_backend() == "amd":
            logger.debug("Parallel processing disabled by amd")
            return False

        stats = gpu_stats.get_card_most_free()
        vram_free = int(stats["free"])
        logger.verbose("%s - %sMB free of %sMB",
                       stats["device"],
                       vram_free,
                       int(stats["total"]))
        if vram_free <= self._total_vram_required:
            logger.warning("Not enough free VRAM for parallel processing. "
                           "Switching to serial")
            return False
        return True

    # << INTERNAL PLUGIN HANDLING >> #
    @staticmethod
    def _load_align(aligner, configfile, normalize_method):
        """ Set global arguments and load aligner plugin """
        aligner_name = aligner.replace("-", "_").lower()
        logger.debug("Loading Aligner: '%s'", aligner_name)
        aligner = PluginLoader.get_aligner(aligner_name)(configfile=configfile,
                                                         normalize_method=normalize_method)
        return aligner

    @staticmethod
    def _load_detect(detector, rotation, min_size, configfile):
        """ Set global arguments and load detector plugin """
        detector_name = detector.replace("-", "_").lower()
        logger.debug("Loading Detector: '%s'", detector_name)
        detector = PluginLoader.get_detector(detector_name)(rotation=rotation,
                                                            min_size=min_size,
                                                            configfile=configfile)
        return detector

    @staticmethod
    def _load_mask(masker, configfile):
        """ Set global arguments and load masker plugin """
        masker_name = masker.replace("-", "_").lower()
        logger.debug("Loading Masker: '%s'", masker_name)
        masker = PluginLoader.get_masker(masker_name)(configfile=configfile)
        return masker

    def _launch_plugin(self, phase):
        """ Launch an extraction plugin """
        logger.debug("Launching %s plugin", phase)
        in_qname = "extract_{}_in".format(phase)
        if phase == self._final_phase:
            out_qname = "extract_{}_out".format(self._final_phase)
        else:
            next_phase = self._flow[self._flow.index(phase) + 1]
            out_qname = "extract_{}_in".format(next_phase)
        logger.debug("in_qname: %s, out_qname: %s", in_qname, out_qname)
        kwargs = dict(in_queue=self._queues[in_qname], out_queue=self._queues[out_qname])

        plugin = getattr(self, "_{}".format(phase))
        plugin.initialize(**kwargs)
        plugin.start()
        logger.debug("Launched %s plugin", phase)

    def _set_extractor_batchsize(self):
        """
        Sets the batch size of the requested plugins based on their vram and
        vram_per_batch_requirements if the the configured batch size requires more
        vram than is available. Nvidia only.
        """
        if get_backend() != "nvidia":
            logger.debug("Backend is not Nvidia. Not updating batchsize requirements")
            return
        if sum([plugin.vram for plugin in self._all_plugins]) == 0:
            logger.debug("No plugins use VRAM. Not updating batchsize requirements.")
            return

        stats = GPUStats().get_card_most_free()
        vram_free = int(stats["free"])
        if self._is_parallel:
            batch_required = sum([plugin.vram_per_batch * plugin.batchsize
                                  for plugin in self._all_plugins])
            plugin_required = self._total_vram_required + batch_required
            if plugin_required <= vram_free:
                logger.debug("Plugin requirements within threshold: (plugin_required: %sMB, "
                             "vram_free: %sMB)", plugin_required, vram_free)
                return
            # Hacky split across 3 plugins
            available_vram = (vram_free - self._total_vram_required) // 3
            for plugin in self._all_plugins:
                self._set_plugin_batchsize(plugin, available_vram)
        else:
            for plugin in self._all_plugins:
                vram_required = plugin.vram + self._vram_buffer
                batch_required = plugin.vram_per_batch * plugin.batchsize
                plugin_required = vram_required + batch_required
                if plugin_required <= vram_free:
                    logger.debug("%s requirements within threshold: (plugin_required: %sMB, "
                                 "vram_free: %sMB)", plugin.name, plugin_required, vram_free)
                    continue
                available_vram = vram_free - vram_required
                self._set_plugin_batchsize(plugin, available_vram)

    @staticmethod
    def _set_plugin_batchsize(plugin, available_vram):
        """ Set the batch size for the given plugin based on given available vram """
        plugin.batchsize = max(1, available_vram // plugin.vram_per_batch)
        logger.verbose("Reset batchsize for %s to %s", plugin.name, plugin.batchsize)

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
