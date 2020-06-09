#!/usr/bin/env python3
"""
Return a requested detector/aligner/masker pipeline

Tensorflow does not like to release GPU VRAM, so parallel plugins need to be managed to work
together.

This module sets up a pipeline for the extraction workflow, loading detect, align and mask
plugins either in parallel or in series, giving easy access to input and output.

 """

import logging

import cv2

from lib.gpu_stats import GPUStats
from lib.queue_manager import queue_manager, QueueEmpty
from lib.utils import get_backend
from plugins.plugin_loader import PluginLoader

logger = logging.getLogger(__name__)  # pylint:disable=invalid-name
_INSTANCES = -1  # Tracking for multiple instances of pipeline


def _get_instance():
    """ Increment the global :attr:`_INSTANCES` and obtain the current instance value """
    global _INSTANCES  # pylint:disable=global-statement
    _INSTANCES += 1
    return _INSTANCES


class Extractor():
    """ Creates a :mod:`~plugins.extract.detect`/:mod:`~plugins.extract.align``/\
    :mod:`~plugins.extract.mask` pipeline and yields results frame by frame from the
    :attr:`detected_faces` generator

    :attr:`input_queue` is dynamically set depending on the current :attr:`phase` of extraction

    Parameters
    ----------
    detector: str
        The name of a detector plugin as exists in :mod:`plugins.extract.detect`
    aligner: str
        The name of an aligner plugin as exists in :mod:`plugins.extract.align`
    masker: str or list
        The name of a masker plugin(s) as exists in :mod:`plugins.extract.mask`.
        This can be a single masker or a list of multiple maskers
    configfile: str, optional
        The path to a custom ``extract.ini`` configfile. If ``None`` then the system
        :file:`config/extract.ini` file will be used.
    multiprocess: bool, optional
        Whether to attempt processing the plugins in parallel. This may get overridden
        internally depending on the plugin combination. Default: ``False``
    rotate_images: str, optional
        Used to set the :attr:`plugins.extract.detect.rotation` attribute. Pass in a single number
        to use increments of that size up to 360, or pass in a ``list`` of ``ints`` to enumerate
        exactly what angles to check. Can also pass in ``'on'`` to increment at 90 degree
        intervals. Default: ``None``
    min_size: int, optional
        Used to set the :attr:`plugins.extract.detect.min_size` attribute Filters out faces
        detected below this size. Length, in pixels across the diagonal of the bounding box. Set
        to ``0`` for off. Default: ``0``
    normalize_method: {`None`, 'clahe', 'hist', 'mean'}, optional
        Used to set the :attr:`plugins.extract.align.normalize_method` attribute. Normalize the
        images fed to the aligner.Default: ``None``
    image_is_aligned: bool, optional
        Used to set the :attr:`plugins.extract.mask.image_is_aligned` attribute. Indicates to the
        masker that the fed in image is an aligned face rather than a frame. Default: ``False``

    Attributes
    ----------
    phase: str
        The current phase that the pipeline is running. Used in conjunction with :attr:`passes` and
        :attr:`final_pass` to indicate to the caller which phase is being processed
    """
    def __init__(self, detector, aligner, masker, configfile=None,
                 multiprocess=False, rotate_images=None, min_size=20,
                 normalize_method=None, image_is_aligned=False):
        logger.debug("Initializing %s: (detector: %s, aligner: %s, masker: %s, "
                     "configfile: %s, multiprocess: %s, rotate_images: %s, min_size: %s, "
                     "normalize_method: %s, image_is_aligned: %s)",
                     self.__class__.__name__, detector, aligner, masker, configfile,
                     multiprocess, rotate_images, min_size, normalize_method, image_is_aligned)
        self._instance = _get_instance()
        masker = [masker] if not isinstance(masker, list) else masker
        self._flow = self._set_flow(detector, aligner, masker)
        # We only ever need 1 item in each queue. This is 2 items cached (1 in queue 1 waiting
        # for queue) at each point. Adding more just stacks RAM with no speed benefit.
        self._queue_size = 1
        self._vram_stats = self._get_vram_stats()
        self._detect = self._load_detect(detector, rotate_images, min_size, configfile)
        self._align = self._load_align(aligner, configfile, normalize_method)
        self._mask = [self._load_mask(mask, image_is_aligned, configfile) for mask in masker]
        self._is_parallel = self._set_parallel_processing(multiprocess)
        self._phases = self._set_phases(multiprocess)
        self._phase_index = 0
        self._set_extractor_batchsize()
        self._queues = self._add_queues()
        logger.debug("Initialized %s", self.__class__.__name__)

    @property
    def input_queue(self):
        """ queue: Return the correct input queue depending on the current phase

        The input queue is the entry point into the extraction pipeline. An :class:`ExtractMedia`
        object should be put to the queue.

        For detect/single phase operations the :attr:`ExtractMedia.filename` and
        :attr:`~ExtractMedia.image` attributes should be populated.

        For align/mask (2nd/3rd pass operations) the :attr:`ExtractMedia.detected_faces` should
        also be populated by calling :func:`ExtractMedia.set_detected_faces`.
        """
        qname = "extract{}_{}_in".format(self._instance, self._current_phase[0])
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
        >>>         extract_media = ExtractMedia("path/to/image/file", image)
        >>>         extractor.input_queue.put(extract_media)
        >>>     else:
        >>>         extract_media.set_image(image)
        >>>         extractor.input_queue.put(extract_media)
        """
        retval = len(self._phases)
        logger.trace(retval)
        return retval

    @property
    def phase_text(self):
        """ str: The plugins that are running in the current phase, formatted for info text
        output. """
        plugin_types = set(self._get_plugin_type_and_index(phase)[0]
                           for phase in self._current_phase)
        retval = ", ".join(plugin_type.title() for plugin_type in list(plugin_types))
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
        >>>         extract_media.set_image(image)
        >>>         <do intermediate processing>
        >>>         extractor.input_queue.put(extract_media)
        """
        retval = self._phase_index == len(self._phases) - 1
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
        for phase in self._current_phase:
            self._launch_plugin(phase)

    def detected_faces(self):
        """ Generator that returns results, frame by frame from the extraction pipeline

        This is the exit point for the extraction pipeline and is used to obtain the output
        of any pipeline :attr:`phase`

        Yields
        ------
        faces: :class:`ExtractMedia`
            The populated extracted media object.

        Example
        -------
        >>> for extract_media in extractor.detected_faces():
        >>>     filename = extract_media.filename
        >>>     image = extract_media.image
        >>>     detected_faces = extract_media.detected_faces
        """
        logger.debug("Running Detection. Phase: '%s'", self._current_phase)
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
            self._phase_index += 1
            logger.debug("Switching to phase: %s", self._current_phase)

    # <<< INTERNAL METHODS >>> #
    @property
    def _parallel_scaling(self):
        """ dict: key is number of parallel plugins being loaded, value is the scaling factor that
        the total base vram for those plugins should be scaled by

        Notes
        -----
        VRAM for parallel plugins does not stack in a linear manner. Calculating the precise
        scaling for any given plugin combination is non trivial, however the following are
        calculations based on running 2-5 plugins in parallel using s3fd, fan, unet, vgg-clear
        and vgg-obstructed. The worst ratio is selected for each combination, plus a little extra
        to ensure that vram is not used up.

        If OOM errors are being reported, then these ratios should be relaxed some more
        """
        retval = {0: 1.0,
                  1: 1.0,
                  2: 0.7,
                  3: 0.55,
                  4: 0.5,
                  5: 0.4}
        logger.trace(retval)
        return retval

    @property
    def _vram_per_phase(self):
        """ dict: The amount of vram required for each phase in :attr:`_flow`. """
        retval = dict()
        for phase in self._flow:
            plugin_type, idx = self._get_plugin_type_and_index(phase)
            attr = getattr(self, "_{}".format(plugin_type))
            attr = attr[idx] if idx is not None else attr
            retval[phase] = attr.vram
        logger.trace(retval)
        return retval

    @property
    def _total_vram_required(self):
        """ Return vram required for all phases plus the buffer """
        vrams = self._vram_per_phase
        vram_required_count = sum(1 for p in vrams.values() if p > 0)
        logger.debug("VRAM requirements: %s. Plugins requiring VRAM: %s",
                     vrams, vram_required_count)
        retval = (sum(vrams.values()) *
                  self._parallel_scaling[vram_required_count])
        logger.debug("Total VRAM required: %s", retval)
        return retval

    @property
    def _current_phase(self):
        """ list: The current phase from :attr:`_phases` that is running through the extractor. """
        retval = self._phases[self._phase_index]
        logger.trace(retval)
        return retval

    @property
    def _final_phase(self):
        """ Return the final phase from the flow list """
        retval = self._flow[-1]
        logger.trace(retval)
        return retval

    @property
    def _output_queue(self):
        """ Return the correct output queue depending on the current phase """
        if self.final_pass:
            qname = "extract{}_{}_out".format(self._instance, self._final_phase)
        else:
            qname = "extract{}_{}_in".format(self._instance,
                                             self._phases[self._phase_index + 1][0])
        retval = self._queues[qname]
        logger.trace("%s: %s", qname, retval)
        return retval

    @property
    def _all_plugins(self):
        """ Return list of all plugin objects in this pipeline """
        retval = []
        for phase in self._flow:
            plugin_type, idx = self._get_plugin_type_and_index(phase)
            attr = getattr(self, "_{}".format(plugin_type))
            attr = attr[idx] if idx is not None else attr
            retval.append(attr)
        logger.trace("All Plugins: %s", retval)
        return retval

    @property
    def _active_plugins(self):
        """ Return the plugins that are currently active based on pass """
        retval = []
        for phase in self._current_phase:
            plugin_type, idx = self._get_plugin_type_and_index(phase)
            attr = getattr(self, "_{}".format(plugin_type))
            retval.append(attr[idx] if idx is not None else attr)
        logger.trace("Active plugins: %s", retval)
        return retval

    @staticmethod
    def _set_flow(detector, aligner, masker):
        """ Set the flow list based on the input plugins """
        logger.debug("detector: %s, aligner: %s, masker: %s", detector, aligner, masker)
        retval = []
        if detector is not None and detector.lower() != "none":
            retval.append("detect")
        if aligner is not None and aligner.lower() != "none":
            retval.append("align")
        retval.extend(["mask_{}".format(idx)
                       for idx, mask in enumerate(masker)
                       if mask is not None and mask.lower() != "none"])
        logger.debug("flow: %s", retval)
        return retval

    @staticmethod
    def _get_plugin_type_and_index(flow_phase):
        """ Obtain the plugin type and index for the plugin for the given flow phase.

        When multiple plugins for the same phase are allowed (e.g. Mask) this will return
        the plugin type and the index of the plugin required. If only one plugin is allowed
        then the plugin type will be returned and the index will be ``None``.

        Parameters
        ----------
        flow_phase: str
            The phase within :attr:`_flow` that is to have the plugin type and index returned

        Returns
        -------
        plugin_type: str
            The plugin type for the given flow phase
        index: int
            The index of this plugin type within the flow, if there are multiple plugins in use
            otherwise ``None`` if there is only 1 plugin in use for the given phase
        """
        idx = flow_phase.split("_")[-1]
        if idx.isdigit():
            idx = int(idx)
            plugin_type = "_".join(flow_phase.split("_")[:-1])
        else:
            plugin_type = flow_phase
            idx = None
        return plugin_type, idx

    def _add_queues(self):
        """ Add the required processing queues to Queue Manager """
        queues = dict()
        tasks = ["extract{}_{}_in".format(self._instance, phase) for phase in self._flow]
        tasks.append("extract{}_{}_out".format(self._instance, self._final_phase))
        for task in tasks:
            # Limit queue size to avoid stacking ram
            queue_manager.add_queue(task, maxsize=self._queue_size)
            queues[task] = queue_manager.get_queue(task)
        logger.debug("Queues: %s", queues)
        return queues

    @staticmethod
    def _get_vram_stats():
        """ Obtain statistics on available VRAM and subtract a constant buffer from available vram.

        Returns
        -------
        dict
            Statistics on available VRAM
        """
        vram_buffer = 256  # Leave a buffer for VRAM allocation
        gpu_stats = GPUStats()
        stats = gpu_stats.get_card_most_free()
        retval = dict(count=gpu_stats.device_count,
                      device=stats["device"],
                      vram_free=int(stats["free"] - vram_buffer),
                      vram_total=int(stats["total"]))
        logger.debug(retval)
        return retval

    def _set_parallel_processing(self, multiprocess):
        """ Set whether to run detect, align, and mask together or separately.

        Parameters
        ----------
        multiprocess: bool
            ``True`` if the single-process command line flag has not been set otherwise ``False``
        """
        if not multiprocess:
            logger.debug("Parallel processing disabled by cli.")
            return False

        if self._vram_stats["count"] == 0:
            logger.debug("No GPU detected. Enabling parallel processing.")
            return True

        if get_backend() == "amd":
            logger.debug("Parallel processing disabled by amd")
            return False

        logger.verbose("%s - %sMB free of %sMB",
                       self._vram_stats["device"],
                       self._vram_stats["vram_free"],
                       self._vram_stats["vram_total"])
        if self._vram_stats["vram_free"] <= self._total_vram_required:
            logger.warning("Not enough free VRAM for parallel processing. "
                           "Switching to serial")
            return False
        return True

    def _set_phases(self, multiprocess):
        """ If not enough VRAM is available, then chunk :attr:`_flow` up into phases that will fit
        into VRAM, otherwise return the single flow.

        Parameters
        ----------
        multiprocess: bool
            ``True`` if the single-process command line flag has not been set otherwise ``False``

        Returns
        -------
        list:
            The jobs to be undertaken split into phases that fit into GPU RAM
        """
        force_single_process = not multiprocess or get_backend() == "amd"
        phases = []
        current_phase = []
        available = self._vram_stats["vram_free"]
        for phase in self._flow:
            num_plugins = len([p for p in current_phase if self._vram_per_phase[p] > 0])
            num_plugins += 1 if self._vram_per_phase[phase] > 0 else 0
            scaling = self._parallel_scaling[num_plugins]
            required = sum(self._vram_per_phase[p] for p in current_phase + [phase]) * scaling
            logger.debug("Num plugins for phase: %s, scaling: %s, vram required: %s",
                         num_plugins, scaling, required)
            if required <= available and not force_single_process:
                logger.debug("Required: %s, available: %s. Adding phase '%s' to current phase: %s",
                             required, available, phase, current_phase)
                current_phase.append(phase)
            elif len(current_phase) == 0 or force_single_process:
                # Amount of VRAM required to run a single plugin is greater than available. We add
                # it anyway, and hope it will run with warnings, as the alternative is to not run
                # at all.
                # This will also run if forcing single process
                logger.debug("Required: %s, available: %s. Single plugin has higher requirements "
                             "than available or forcing single process: '%s'",
                             required, available, phase)
                phases.append([phase])
            else:
                logger.debug("Required: %s, available: %s. Adding phase to flow: %s",
                             required, available, current_phase)
                phases.append(current_phase)
                current_phase = [phase]
        if current_phase:
            phases.append(current_phase)
        logger.debug("Total phases: %s, Phases: %s", len(phases), phases)
        return phases

    # << INTERNAL PLUGIN HANDLING >> #
    def _load_align(self, aligner, configfile, normalize_method):
        """ Set global arguments and load aligner plugin """
        if aligner is None or aligner.lower() == "none":
            logger.debug("No aligner selected. Returning None")
            return None
        aligner_name = aligner.replace("-", "_").lower()
        logger.debug("Loading Aligner: '%s'", aligner_name)
        aligner = PluginLoader.get_aligner(aligner_name)(configfile=configfile,
                                                         normalize_method=normalize_method,
                                                         instance=self._instance)
        return aligner

    def _load_detect(self, detector, rotation, min_size, configfile):
        """ Set global arguments and load detector plugin """
        if detector is None or detector.lower() == "none":
            logger.debug("No detector selected. Returning None")
            return None
        detector_name = detector.replace("-", "_").lower()
        logger.debug("Loading Detector: '%s'", detector_name)
        detector = PluginLoader.get_detector(detector_name)(rotation=rotation,
                                                            min_size=min_size,
                                                            configfile=configfile,
                                                            instance=self._instance)
        return detector

    def _load_mask(self, masker, image_is_aligned, configfile):
        """ Set global arguments and load masker plugin """
        if masker is None or masker.lower() == "none":
            logger.debug("No masker selected. Returning None")
            return None
        masker_name = masker.replace("-", "_").lower()
        logger.debug("Loading Masker: '%s'", masker_name)
        masker = PluginLoader.get_masker(masker_name)(image_is_aligned=image_is_aligned,
                                                      configfile=configfile,
                                                      instance=self._instance)
        return masker

    def _launch_plugin(self, phase):
        """ Launch an extraction plugin """
        logger.debug("Launching %s plugin", phase)
        in_qname = "extract{}_{}_in".format(self._instance, phase)
        if phase == self._final_phase:
            out_qname = "extract{}_{}_out".format(self._instance, self._final_phase)
        else:
            next_phase = self._flow[self._flow.index(phase) + 1]
            out_qname = "extract{}_{}_in".format(self._instance, next_phase)
        logger.debug("in_qname: %s, out_qname: %s", in_qname, out_qname)
        kwargs = dict(in_queue=self._queues[in_qname], out_queue=self._queues[out_qname])

        plugin_type, idx = self._get_plugin_type_and_index(phase)
        plugin = getattr(self, "_{}".format(plugin_type))
        plugin = plugin[idx] if idx is not None else plugin
        plugin.initialize(**kwargs)
        plugin.start()
        logger.debug("Launched %s plugin", phase)

    def _set_extractor_batchsize(self):
        """
        Sets the batch size of the requested plugins based on their vram, their
        vram_per_batch_requirements and the number of plugins being loaded in the current phase.
        Only adjusts if the the configured batch size requires more vram than is available. Nvidia
        only.
        """
        if get_backend() != "nvidia":
            logger.debug("Backend is not Nvidia. Not updating batchsize requirements")
            return
        if sum([plugin.vram for plugin in self._active_plugins]) == 0:
            logger.debug("No plugins use VRAM. Not updating batchsize requirements.")
            return

        batch_required = sum([plugin.vram_per_batch * plugin.batchsize
                              for plugin in self._active_plugins])
        gpu_plugins = [p for p in self._current_phase if self._vram_per_phase[p] > 0]
        plugins_required = sum([self._vram_per_phase[p]
                                for p in gpu_plugins]) * self._parallel_scaling[len(gpu_plugins)]
        if plugins_required + batch_required <= self._vram_stats["vram_free"]:
            logger.debug("Plugin requirements within threshold: (plugins_required: %sMB, "
                         "vram_free: %sMB)", plugins_required, self._vram_stats["vram_free"])
            return
        # Hacky split across plugins that use vram
        available_vram = (self._vram_stats["vram_free"] - plugins_required) // len(gpu_plugins)
        self._set_plugin_batchsize(gpu_plugins, available_vram)

    def set_aligner_normalization_method(self, method):
        """ Change the normalization method for faces fed into the aligner.

        Parameters
        ----------
        method: {"none", "clahe", "hist", "mean"}
            The normalization method to apply to faces prior to feeding into the aligner's model
        """
        logger.debug("Setting to: '%s'", method)
        self._align.set_normalize_method(method)

    def _set_plugin_batchsize(self, gpu_plugins, available_vram):
        """ Set the batch size for the given plugin based on given available vram.
        Do not update plugins which have a vram_per_batch of 0 (CPU plugins) due to
        zero division error.
        """
        plugins = [self._active_plugins[idx]
                   for idx, plugin in enumerate(self._current_phase)
                   if plugin in gpu_plugins]
        vram_per_batch = [plugin.vram_per_batch for plugin in plugins]
        ratios = [vram / sum(vram_per_batch) for vram in vram_per_batch]
        requested_batchsizes = [plugin.batchsize for plugin in plugins]
        batchsizes = [min(requested, max(1, int((available_vram * ratio) / plugin.vram_per_batch)))
                      for ratio, plugin, requested in zip(ratios, plugins, requested_batchsizes)]
        remaining = available_vram - sum(batchsize * plugin.vram_per_batch
                                         for batchsize, plugin in zip(batchsizes, plugins))
        sorted_indices = [i[0] for i in sorted(enumerate(plugins),
                                               key=lambda x: x[1].vram_per_batch, reverse=True)]

        logger.debug("requested_batchsizes: %s, batchsizes: %s, remaining vram: %s",
                     requested_batchsizes, batchsizes, remaining)

        while remaining > min(plugin.vram_per_batch
                              for plugin in plugins) and requested_batchsizes != batchsizes:
            for idx in sorted_indices:
                plugin = plugins[idx]
                if plugin.vram_per_batch > remaining:
                    logger.debug("Not enough VRAM to increase batch size of %s. Required: %sMB, "
                                 "Available: %sMB", plugin, plugin.vram_per_batch, remaining)
                    continue
                if plugin.batchsize == batchsizes[idx]:
                    logger.debug("Threshold reached for %s. Batch size: %s",
                                 plugin, plugin.batchsize)
                    continue
                logger.debug("Incrementing batch size of %s to %s", plugin, batchsizes[idx] + 1)
                batchsizes[idx] += 1
                remaining -= plugin.vram_per_batch
                logger.debug("Remaining VRAM to allocate: %sMB", remaining)

        if batchsizes != requested_batchsizes:
            text = ", ".join(["{}: {}".format(plugin.__class__.__name__, batchsize)
                              for plugin, batchsize in zip(plugins, batchsizes)])
            for plugin, batchsize in zip(plugins, batchsizes):
                plugin.batchsize = batchsize
            logger.info("Reset batch sizes due to available VRAM: %s", text)

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


class ExtractMedia():
    """ An object that passes through the :class:`~plugins.extract.pipeline.Extractor` pipeline.

    Parameters
    ----------
    filename: str
        The base name of the original frame's filename
    image: :class:`numpy.ndarray`
        The original frame
    detected_faces: list, optional
        A list of :class:`~lib.faces_detect.DetectedFace` objects. Detected faces can be added
        later with :func:`add_detected_faces`. Default: None
    """

    def __init__(self, filename, image, detected_faces=None):
        logger.trace("Initializing %s: (filename: '%s', image shape: %s, detected_faces: %s)",
                     self.__class__.__name__, filename, image.shape, detected_faces)
        self._filename = filename
        self._image = image
        self._detected_faces = detected_faces

    @property
    def filename(self):
        """ str: The base name of the :attr:`image` filename. """
        return self._filename

    @property
    def image(self):
        """ :class:`numpy.ndarray`: The source frame for this object. """
        return self._image

    @property
    def image_shape(self):
        """ tuple: The shape of the stored :attr:`image`. """
        return self._image.shape

    @property
    def image_size(self):
        """ tuple: The (`height`, `width`) of the stored :attr:`image`. """
        return self._image.shape[:2]

    @property
    def detected_faces(self):
        """list: A list of :class:`~lib.faces_detect.DetectedFace` objects in the
        :attr:`image`. """
        return self._detected_faces

    def get_image_copy(self, color_format):
        """ Get a copy of the image in the requested color format.

        Parameters
        ----------
        color_format: ['BGR', 'RGB', 'GRAY']
            The requested color format of :attr:`image`

        Returns
        -------
        :class:`numpy.ndarray`:
            A copy of :attr:`image` in the requested :attr:`color_format`
        """
        logger.trace("Requested color format '%s' for frame '%s'", color_format, self._filename)
        image = getattr(self, "_image_as_{}".format(color_format.lower()))()
        return image

    def add_detected_faces(self, faces):
        """ Add detected faces to the object. Called at the end of each extraction phase.

        Parameters
        ----------
        faces: list
            A list of :class:`~lib.faces_detect.DetectedFace` objects
        """
        logger.trace("Adding detected faces for filename: '%s'. (faces: %s, lrtb: %s)",
                     self._filename, faces,
                     [(face.left, face.right, face.top, face.bottom) for face in faces])
        self._detected_faces = faces

    def remove_image(self):
        """ Delete the image and reset :attr:`image` to ``None``.

        Required for multi-phase extraction to avoid the frames stacking RAM.
        """
        logger.trace("Removing image for filename: '%s'", self._filename)
        del self._image
        self._image = None

    def set_image(self, image):
        """ Add the image back into :attr:`image`

        Required for multi-phase extraction adds the image back to this object.

        Parameters
        ----------
        image: :class:`numpy.ndarry`
            The original frame to be re-applied to for this :attr:`filename`
        """
        logger.trace("Reapplying image: (filename: `%s`, image shape: %s)",
                     self._filename, image.shape)
        self._image = image

    def _image_as_bgr(self):
        """ Get a copy of the source frame in BGR format.

        Returns
        -------
        :class:`numpy.ndarray`:
            A copy of :attr:`image` in BGR color format """
        return self._image[..., :3].copy()

    def _image_as_rgb(self):
        """ Get a copy of the source frame in RGB format.

        Returns
        -------
        :class:`numpy.ndarray`:
            A copy of :attr:`image` in RGB color format """
        return self._image[..., 2::-1].copy()

    def _image_as_gray(self):
        """ Get a copy of the source frame in gray-scale format.

        Returns
        -------
        :class:`numpy.ndarray`:
            A copy of :attr:`image` in gray-scale color format """
        return cv2.cvtColor(self._image.copy(), cv2.COLOR_BGR2GRAY)
