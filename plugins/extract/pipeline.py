#!/usr/bin/env python3
"""
Return a requested detector/aligner/masker pipeline

Tensorflow does not like to release GPU VRAM, so parallel plugins need to be managed to work
together.

This module sets up a pipeline for the extraction workflow, loading detect, align and mask
plugins either in parallel or in series, giving easy access to input and output.

 """

import logging
import sys
from typing import Any, cast, Dict, Generator, List, Optional, Tuple, TYPE_CHECKING, Union

import cv2

from lib.gpu_stats import GPUStats
from lib.queue_manager import EventQueue, queue_manager, QueueEmpty
from lib.utils import get_backend
from plugins.plugin_loader import PluginLoader

if sys.version_info < (3, 8):
    from typing_extensions import Literal
else:
    from typing import Literal

if TYPE_CHECKING:
    import numpy as np
    from lib.align.detected_face import DetectedFace
    from plugins.extract._base import Extractor as PluginExtractor
    from plugins.extract.detect._base import Detector
    from plugins.extract.align._base import Aligner
    from plugins.extract.mask._base import Masker

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
    detector: str or ``None``
        The name of a detector plugin as exists in :mod:`plugins.extract.detect`
    aligner: str or ``None
        The name of an aligner plugin as exists in :mod:`plugins.extract.align`
    masker: str or list or ``None
        The name of a masker plugin(s) as exists in :mod:`plugins.extract.mask`.
        This can be a single masker or a list of multiple maskers
    configfile: str, optional
        The path to a custom ``extract.ini`` configfile. If ``None`` then the system
        :file:`config/extract.ini` file will be used.
    multiprocess: bool, optional
        Whether to attempt processing the plugins in parallel. This may get overridden
        internally depending on the plugin combination. Default: ``False``
    exclude_gpus: list, optional
        A list of indices correlating to connected GPUs that Tensorflow should not use. Pass
        ``None`` to not exclude any GPUs. Default: ``None``
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
    re_feed: int
        The number of times to re-feed a slightly adjusted bounding box into the aligner.
        Default: `0`
    image_is_aligned: bool, optional
        Used to set the :attr:`plugins.extract.mask.image_is_aligned` attribute. Indicates to the
        masker that the fed in image is an aligned face rather than a frame. Default: ``False``

    Attributes
    ----------
    phase: str
        The current phase that the pipeline is running. Used in conjunction with :attr:`passes` and
        :attr:`final_pass` to indicate to the caller which phase is being processed
    """
    def __init__(self,
                 detector: Optional[str],
                 aligner: Optional[str],
                 masker: Optional[Union[str, List[str]]],
                 configfile: Optional[str] = None,
                 multiprocess: bool = False,
                 exclude_gpus: Optional[List[int]] = None,
                 rotate_images: Optional[List[int]] = None,
                 min_size: int = 20,
                 normalize_method: Optional[str] = None,
                 re_feed: int = 0,
                 image_is_aligned: bool = False) -> None:
        logger.debug("Initializing %s: (detector: %s, aligner: %s, masker: %s, configfile: %s, "
                     "multiprocess: %s, exclude_gpus: %s, rotate_images: %s, min_size: %s, "
                     "normalize_method: %s, re_feed: %s, image_is_aligned: %s)",
                     self.__class__.__name__, detector, aligner, masker, configfile, multiprocess,
                     exclude_gpus, rotate_images, min_size, normalize_method, re_feed,
                     image_is_aligned)
        self._instance = _get_instance()
        maskers = [cast(Optional[str],
                   masker)] if not isinstance(masker, list) else cast(List[Optional[str]], masker)
        self._flow = self._set_flow(detector, aligner, maskers)
        self._exclude_gpus = exclude_gpus
        # We only ever need 1 item in each queue. This is 2 items cached (1 in queue 1 waiting
        # for queue) at each point. Adding more just stacks RAM with no speed benefit.
        self._queue_size = 1
        # TODO Calculate scaling for more plugins than currently exist in _parallel_scaling
        self._scaling_fallback = 0.4
        self._vram_stats = self._get_vram_stats()
        self._detect = self._load_detect(detector, rotate_images, min_size, configfile)
        self._align = self._load_align(aligner, configfile, normalize_method, re_feed)
        self._mask = [self._load_mask(mask, image_is_aligned, configfile) for mask in maskers]
        self._is_parallel = self._set_parallel_processing(multiprocess)
        self._phases = self._set_phases(multiprocess)
        self._phase_index = 0
        self._set_extractor_batchsize()
        self._queues = self._add_queues()
        logger.debug("Initialized %s", self.__class__.__name__)

    @property
    def input_queue(self) -> EventQueue:
        """ queue: Return the correct input queue depending on the current phase

        The input queue is the entry point into the extraction pipeline. An :class:`ExtractMedia`
        object should be put to the queue.

        For detect/single phase operations the :attr:`ExtractMedia.filename` and
        :attr:`~ExtractMedia.image` attributes should be populated.

        For align/mask (2nd/3rd pass operations) the :attr:`ExtractMedia.detected_faces` should
        also be populated by calling :func:`ExtractMedia.set_detected_faces`.
        """
        qname = f"extract{self._instance}_{self._current_phase[0]}_in"
        retval = self._queues[qname]
        logger.trace("%s: %s", qname, retval)  # type: ignore
        return retval

    @property
    def passes(self) -> int:
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
        logger.trace(retval)  # type: ignore
        return retval

    @property
    def phase_text(self) -> str:
        """ str: The plugins that are running in the current phase, formatted for info text
        output. """
        plugin_types = set(self._get_plugin_type_and_index(phase)[0]
                           for phase in self._current_phase)
        retval = ", ".join(plugin_type.title() for plugin_type in list(plugin_types))
        logger.trace(retval)  # type: ignore
        return retval

    @property
    def final_pass(self) -> bool:
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
        logger.trace(retval)  # type: ignore
        return retval

    def reset_phase_index(self) -> None:
        """ Reset the current phase index back to 0. Used for when batch processing is used in
        extract. """
        self._phase_index = 0

    def set_batchsize(self,
                      plugin_type: Literal["align", "detect"],
                      batchsize: int) -> None:
        """ Set the batch size of a given :attr:`plugin_type` to the given :attr:`batchsize`.

        This should be set prior to :func:`launch` if the batch size is to be manually overridden

        Parameters
        ----------
        plugin_type: {'align', 'detect'}
            The plugin_type to be overridden
        batchsize: int
            The batch size to use for this plugin type
        """
        logger.debug("Overriding batchsize for plugin_type: %s to: %s", plugin_type, batchsize)
        plugin = getattr(self, f"_{plugin_type}")
        plugin.batchsize = batchsize

    def launch(self) -> None:
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

    def detected_faces(self) -> Generator["ExtractMedia", None, None]:
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
            logger.debug("Detection Complete")
        else:
            self._phase_index += 1
            logger.debug("Switching to phase: %s", self._current_phase)

    # <<< INTERNAL METHODS >>> #
    @property
    def _parallel_scaling(self) -> Dict[int, float]:
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
        logger.trace(retval)  # type: ignore
        return retval

    @property
    def _vram_per_phase(self) -> Dict[str, float]:
        """ dict: The amount of vram required for each phase in :attr:`_flow`. """
        retval = {}
        for phase in self._flow:
            plugin_type, idx = self._get_plugin_type_and_index(phase)
            attr = getattr(self, f"_{plugin_type}")
            attr = attr[idx] if idx is not None else attr
            retval[phase] = attr.vram
        logger.trace(retval)  # type: ignore
        return retval

    @property
    def _total_vram_required(self) -> float:
        """ Return vram required for all phases plus the buffer """
        vrams = self._vram_per_phase
        vram_required_count = sum(1 for p in vrams.values() if p > 0)
        logger.debug("VRAM requirements: %s. Plugins requiring VRAM: %s",
                     vrams, vram_required_count)
        retval = (sum(vrams.values()) *
                  self._parallel_scaling.get(vram_required_count, self._scaling_fallback))
        logger.debug("Total VRAM required: %s", retval)
        return retval

    @property
    def _current_phase(self) -> List[str]:
        """ list: The current phase from :attr:`_phases` that is running through the extractor. """
        retval = self._phases[self._phase_index]
        logger.trace(retval)  # type: ignore
        return retval

    @property
    def _final_phase(self) -> str:
        """ Return the final phase from the flow list """
        retval = self._flow[-1]
        logger.trace(retval)  # type: ignore
        return retval

    @property
    def _output_queue(self) -> EventQueue:
        """ Return the correct output queue depending on the current phase """
        if self.final_pass:
            qname = f"extract{self._instance}_{self._final_phase}_out"
        else:
            qname = f"extract{self._instance}_{self._phases[self._phase_index + 1][0]}_in"
        retval = self._queues[qname]
        logger.trace("%s: %s", qname, retval)  # type: ignore
        return retval

    @property
    def _all_plugins(self) -> List["PluginExtractor"]:
        """ Return list of all plugin objects in this pipeline """
        retval = []
        for phase in self._flow:
            plugin_type, idx = self._get_plugin_type_and_index(phase)
            attr = getattr(self, f"_{plugin_type}")
            attr = attr[idx] if idx is not None else attr
            retval.append(attr)
        logger.trace("All Plugins: %s", retval)  # type: ignore
        return retval

    @property
    def _active_plugins(self) -> List["PluginExtractor"]:
        """ Return the plugins that are currently active based on pass """
        retval = []
        for phase in self._current_phase:
            plugin_type, idx = self._get_plugin_type_and_index(phase)
            attr = getattr(self, f"_{plugin_type}")
            retval.append(attr[idx] if idx is not None else attr)
        logger.trace("Active plugins: %s", retval)  # type: ignore
        return retval

    @staticmethod
    def _set_flow(detector: Optional[str],
                  aligner: Optional[str],
                  masker: List[Optional[str]]) -> List[str]:
        """ Set the flow list based on the input plugins """
        logger.debug("detector: %s, aligner: %s, masker: %s", detector, aligner, masker)
        retval = []
        if detector is not None and detector.lower() != "none":
            retval.append("detect")
        if aligner is not None and aligner.lower() != "none":
            retval.append("align")
        retval.extend([f"mask_{idx}"
                       for idx, mask in enumerate(masker)
                       if mask is not None and mask.lower() != "none"])
        logger.debug("flow: %s", retval)
        return retval

    @staticmethod
    def _get_plugin_type_and_index(flow_phase: str) -> Tuple[str, Optional[int]]:
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
        sidx = flow_phase.split("_")[-1]
        if sidx.isdigit():
            idx: Optional[int] = int(sidx)
            plugin_type = "_".join(flow_phase.split("_")[:-1])
        else:
            plugin_type = flow_phase
            idx = None
        return plugin_type, idx

    def _add_queues(self) -> Dict[str, EventQueue]:
        """ Add the required processing queues to Queue Manager """
        queues = {}
        tasks = [f"extract{self._instance}_{phase}_in" for phase in self._flow]
        tasks.append(f"extract{self._instance}_{self._final_phase}_out")
        for task in tasks:
            # Limit queue size to avoid stacking ram
            queue_manager.add_queue(task, maxsize=self._queue_size)
            queues[task] = queue_manager.get_queue(task)
        logger.debug("Queues: %s", queues)
        return queues

    @staticmethod
    def _get_vram_stats() -> Dict[str, Union[int, str]]:
        """ Obtain statistics on available VRAM and subtract a constant buffer from available vram.

        Returns
        -------
        dict
            Statistics on available VRAM
        """
        vram_buffer = 256  # Leave a buffer for VRAM allocation
        gpu_stats = GPUStats()
        stats = gpu_stats.get_card_most_free()
        retval: Dict[str, Union[int, str]] = dict(count=gpu_stats.device_count,
                                                  device=stats["device"],
                                                  vram_free=int(stats["free"] - vram_buffer),
                                                  vram_total=int(stats["total"]))
        logger.debug(retval)
        return retval

    def _set_parallel_processing(self, multiprocess: bool) -> bool:
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

        logger.verbose("%s - %sMB free of %sMB",  # type: ignore
                       self._vram_stats["device"],
                       self._vram_stats["vram_free"],
                       self._vram_stats["vram_total"])
        if cast(int, self._vram_stats["vram_free"]) <= self._total_vram_required:
            logger.warning("Not enough free VRAM for parallel processing. "
                           "Switching to serial")
            return False
        return True

    def _set_phases(self, multiprocess: bool) -> List[List[str]]:
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
        phases: List[List[str]] = []
        current_phase: List[str] = []
        available = cast(int, self._vram_stats["vram_free"])
        for phase in self._flow:
            num_plugins = len([p for p in current_phase if self._vram_per_phase[p] > 0])
            num_plugins += 1 if self._vram_per_phase[phase] > 0 else 0
            scaling = self._parallel_scaling.get(num_plugins, self._scaling_fallback)
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
    def _load_align(self,
                    aligner: Optional[str],
                    configfile: Optional[str],
                    normalize_method: Optional[str],
                    re_feed: int) -> Optional["Aligner"]:
        """ Set global arguments and load aligner plugin """
        if aligner is None or aligner.lower() == "none":
            logger.debug("No aligner selected. Returning None")
            return None
        aligner_name = aligner.replace("-", "_").lower()
        logger.debug("Loading Aligner: '%s'", aligner_name)
        plugin = PluginLoader.get_aligner(aligner_name)(exclude_gpus=self._exclude_gpus,
                                                        configfile=configfile,
                                                        normalize_method=normalize_method,
                                                        re_feed=re_feed,
                                                        instance=self._instance)
        return plugin

    def _load_detect(self,
                     detector: Optional[str],
                     rotation: Optional[List[int]],
                     min_size: int,
                     configfile: Optional[str]) -> Optional["Detector"]:
        """ Set global arguments and load detector plugin """
        if detector is None or detector.lower() == "none":
            logger.debug("No detector selected. Returning None")
            return None
        detector_name = detector.replace("-", "_").lower()
        logger.debug("Loading Detector: '%s'", detector_name)
        plugin = PluginLoader.get_detector(detector_name)(exclude_gpus=self._exclude_gpus,
                                                          rotation=rotation,
                                                          min_size=min_size,
                                                          configfile=configfile,
                                                          instance=self._instance)
        return plugin

    def _load_mask(self,
                   masker: Optional[str],
                   image_is_aligned: bool,
                   configfile: Optional[str]) -> Optional["Masker"]:
        """ Set global arguments and load masker plugin """
        if masker is None or masker.lower() == "none":
            logger.debug("No masker selected. Returning None")
            return None
        masker_name = masker.replace("-", "_").lower()
        logger.debug("Loading Masker: '%s'", masker_name)
        plugin = PluginLoader.get_masker(masker_name)(exclude_gpus=self._exclude_gpus,
                                                      image_is_aligned=image_is_aligned,
                                                      configfile=configfile,
                                                      instance=self._instance)
        return plugin

    def _launch_plugin(self, phase: str) -> None:
        """ Launch an extraction plugin """
        logger.debug("Launching %s plugin", phase)
        in_qname = f"extract{self._instance}_{phase}_in"
        if phase == self._final_phase:
            out_qname = f"extract{self._instance}_{self._final_phase}_out"
        else:
            next_phase = self._flow[self._flow.index(phase) + 1]
            out_qname = f"extract{self._instance}_{next_phase}_in"
        logger.debug("in_qname: %s, out_qname: %s", in_qname, out_qname)
        kwargs = dict(in_queue=self._queues[in_qname], out_queue=self._queues[out_qname])

        plugin_type, idx = self._get_plugin_type_and_index(phase)
        plugin = getattr(self, f"_{plugin_type}")
        plugin = plugin[idx] if idx is not None else plugin
        plugin.initialize(**kwargs)
        plugin.start()
        logger.debug("Launched %s plugin", phase)

    def _set_extractor_batchsize(self) -> None:
        """
        Sets the batch size of the requested plugins based on their vram, their
        vram_per_batch_requirements and the number of plugins being loaded in the current phase.
        Only adjusts if the the configured batch size requires more vram than is available. Nvidia
        only.
        """
        if get_backend() != "nvidia":
            logger.debug("Backend is not Nvidia. Not updating batchsize requirements")
            return
        if sum(plugin.vram for plugin in self._active_plugins) == 0:
            logger.debug("No plugins use VRAM. Not updating batchsize requirements.")
            return

        batch_required = sum(plugin.vram_per_batch * plugin.batchsize
                             for plugin in self._active_plugins)
        gpu_plugins = [p for p in self._current_phase if self._vram_per_phase[p] > 0]
        scaling = self._parallel_scaling.get(len(gpu_plugins), self._scaling_fallback)
        plugins_required = sum(self._vram_per_phase[p] for p in gpu_plugins) * scaling
        if plugins_required + batch_required <= cast(int, self._vram_stats["vram_free"]):
            logger.debug("Plugin requirements within threshold: (plugins_required: %sMB, "
                         "vram_free: %sMB)", plugins_required, self._vram_stats["vram_free"])
            return
        # Hacky split across plugins that use vram
        available_vram = (cast(int, self._vram_stats["vram_free"])
                          - plugins_required) // len(gpu_plugins)
        self._set_plugin_batchsize(gpu_plugins, available_vram)

    def set_aligner_normalization_method(self, method: str) -> None:
        """ Change the normalization method for faces fed into the aligner.

        Parameters
        ----------
        method: {"none", "clahe", "hist", "mean"}
            The normalization method to apply to faces prior to feeding into the aligner's model
        """
        assert self._align is not None
        logger.debug("Setting to: '%s'", method)
        self._align.set_normalize_method(method)

    def _set_plugin_batchsize(self, gpu_plugins: List[str], available_vram: float) -> None:
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
            text = ", ".join([f"{plugin.__class__.__name__}: {batchsize}"
                              for plugin, batchsize in zip(plugins, batchsizes)])
            for plugin, batchsize in zip(plugins, batchsizes):
                plugin.batchsize = batchsize
            logger.info("Reset batch sizes due to available VRAM: %s", text)

    def _join_threads(self):
        """ Join threads for current pass """
        for plugin in self._active_plugins:
            plugin.join()

    def _check_and_raise_error(self) -> bool:
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
        A list of :class:`~lib.align.DetectedFace` objects. Detected faces can be added
        later with :func:`add_detected_faces`. Setting ``None`` will default to an empty list.
        Default: ``None``
    """

    def __init__(self,
                 filename: str,
                 image: "np.ndarray",
                 detected_faces: Optional[List["DetectedFace"]] = None) -> None:
        logger.trace("Initializing %s: (filename: '%s', image shape: %s, "  # type: ignore
                     "detected_faces: %s)", self.__class__.__name__, filename, image.shape,
                     detected_faces)
        self._filename = filename
        self._image: Optional["np.ndarray"] = image
        self._image_shape = cast(Tuple[int, int, int], image.shape)
        self._detected_faces: List["DetectedFace"] = ([] if detected_faces is None
                                                      else detected_faces)
        self._frame_metadata: Dict[str, Any] = {}

    @property
    def filename(self) -> str:
        """ str: The base name of the :attr:`image` filename. """
        return self._filename

    @property
    def image(self) -> "np.ndarray":
        """ :class:`numpy.ndarray`: The source frame for this object. """
        assert self._image is not None
        return self._image

    @property
    def image_shape(self) -> Tuple[int, int, int]:
        """ tuple: The shape of the stored :attr:`image`. """
        return self._image_shape

    @property
    def image_size(self) -> Tuple[int, int]:
        """ tuple: The (`height`, `width`) of the stored :attr:`image`. """
        return self._image_shape[:2]

    @property
    def detected_faces(self) -> List["DetectedFace"]:
        """list: A list of :class:`~lib.align.DetectedFace` objects in the :attr:`image`. """
        return self._detected_faces

    @property
    def frame_metadata(self) -> dict:
        """ dict: The frame metadata that has been added from an aligned image. This property
        should only be called after :func:`add_frame_metadata` has been called when processing
        an aligned face. For all other instances an assertion error will be raised.

        Raises
        ------
        AssertionError
            If frame metadata has not been populated from an aligned image
        """
        assert self._frame_metadata is not None
        return self._frame_metadata

    def get_image_copy(self, color_format: Literal["BGR", "RGB", "GRAY"]) -> "np.ndarray":
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
        logger.trace("Requested color format '%s' for frame '%s'",  # type: ignore
                     color_format, self._filename)
        image = getattr(self, f"_image_as_{color_format.lower()}")()
        return image

    def add_detected_faces(self, faces: List["DetectedFace"]) -> None:
        """ Add detected faces to the object. Called at the end of each extraction phase.

        Parameters
        ----------
        faces: list
            A list of :class:`~lib.align.DetectedFace` objects
        """
        logger.trace("Adding detected faces for filename: '%s'. "  # type: ignore
                     "(faces: %s, lrtb: %s)", self._filename, faces,
                     [(face.left, face.right, face.top, face.bottom) for face in faces])
        self._detected_faces = faces

    def remove_image(self) -> None:
        """ Delete the image and reset :attr:`image` to ``None``.

        Required for multi-phase extraction to avoid the frames stacking RAM.
        """
        logger.trace("Removing image for filename: '%s'", self._filename)  # type: ignore
        del self._image
        self._image = None

    def set_image(self, image: "np.ndarray") -> None:
        """ Add the image back into :attr:`image`

        Required for multi-phase extraction adds the image back to this object.

        Parameters
        ----------
        image: :class:`numpy.ndarry`
            The original frame to be re-applied to for this :attr:`filename`
        """
        logger.trace("Reapplying image: (filename: `%s`, image shape: %s)",  # type: ignore
                     self._filename, image.shape)
        self._image = image

    def add_frame_metadata(self, metadata: Dict[str, Any]) -> None:
        """ Add the source frame metadata from an aligned PNG's header data.

        metadata: dict
            The contents of the 'source' field in the PNG header
        """
        logger.trace("Adding PNG Source data for '%s': %s",  # type:ignore
                     self._filename, metadata)
        dims: Tuple[int, int] = metadata["source_frame_dims"]
        self._image_shape = (*dims, 3)
        self._frame_metadata = metadata

    def _image_as_bgr(self) -> "np.ndarray":
        """ Get a copy of the source frame in BGR format.

        Returns
        -------
        :class:`numpy.ndarray`:
            A copy of :attr:`image` in BGR color format """
        return self.image[..., :3].copy()

    def _image_as_rgb(self) -> "np.ndarray":
        """ Get a copy of the source frame in RGB format.

        Returns
        -------
        :class:`numpy.ndarray`:
            A copy of :attr:`image` in RGB color format """
        return self.image[..., 2::-1].copy()

    def _image_as_gray(self) -> "np.ndarray":
        """ Get a copy of the source frame in gray-scale format.

        Returns
        -------
        :class:`numpy.ndarray`:
            A copy of :attr:`image` in gray-scale color format """
        return cv2.cvtColor(self.image.copy(), cv2.COLOR_BGR2GRAY)
