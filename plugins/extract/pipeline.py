#!/usr/bin/env python3
"""
Return a requested detector/aligner/masker pipeline

Tensorflow does not like to release GPU VRAM, so parallel plugins need to be managed to work
together.

This module sets up a pipeline for the extraction workflow, loading detect, align and mask
plugins either in parallel or in series, giving easy access to input and output.
"""
from __future__ import annotations
import logging
import os
import typing as T

from lib.align import LandmarkType
from lib.gpu_stats import GPUStats
from lib.logger import parse_class_init
from lib.queue_manager import EventQueue, queue_manager, QueueEmpty
from lib.serializer import get_serializer
from lib.utils import get_backend, FaceswapError
from plugins.plugin_loader import PluginLoader

if T.TYPE_CHECKING:
    from collections.abc import Generator
    from ._base import Extractor as PluginExtractor
    from .align._base import Aligner
    from .align.external import Align as AlignImport
    from .detect._base import Detector
    from .detect.external import Detect as DetectImport
    from .mask._base import Masker
    from .recognition._base import Identity
    from . import ExtractMedia

logger = logging.getLogger(__name__)
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
    aligner: str or ``None``
        The name of an aligner plugin as exists in :mod:`plugins.extract.align`
    masker: str or list or ``None``
        The name of a masker plugin(s) as exists in :mod:`plugins.extract.mask`.
        This can be a single masker or a list of multiple maskers
    recognition: str or ``None``
        The name of the recognition plugin to use. ``None`` to not do face recognition.
        Default: ``None``
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
        Used to set the :attr:`plugins.extract.detect.min_size` attribute. Filters out faces
        detected below this size. Length, in pixels across the diagonal of the bounding box. Set
        to ``0`` for off. Default: ``0``
    normalize_method: {`None`, 'clahe', 'hist', 'mean'}, optional
        Used to set the :attr:`plugins.extract.align.normalize_method` attribute. Normalize the
        images fed to the aligner.Default: ``None``
    re_feed: int
        The number of times to re-feed a slightly adjusted bounding box into the aligner.
        Default: `0`
    re_align: bool, optional
        ``True`` to obtain landmarks by passing the initially aligned face back through the
        aligner. Default ``False``
    disable_filter: bool, optional
        Disable all aligner filters regardless of config option. Default: ``False``

    Attributes
    ----------
    phase: str
        The current phase that the pipeline is running. Used in conjunction with :attr:`passes` and
        :attr:`final_pass` to indicate to the caller which phase is being processed
    """
    def __init__(self,
                 detector: str | None,
                 aligner: str | None,
                 masker: str | list[str] | None,
                 recognition: str | None = None,
                 configfile: str | None = None,
                 multiprocess: bool = False,
                 exclude_gpus: list[int] | None = None,
                 rotate_images: str | None = None,
                 min_size: int = 0,
                 normalize_method:  T.Literal["none", "clahe", "hist", "mean"] | None = None,
                 re_feed: int = 0,
                 re_align: bool = False,
                 disable_filter: bool = False) -> None:
        logger.debug(parse_class_init(locals()))
        self._instance = _get_instance()
        maskers = [T.cast(str | None,
                   masker)] if not isinstance(masker, list) else T.cast(list[str | None],
                                                                        masker)
        self._flow = self._set_flow(detector, aligner, maskers, recognition)
        self._exclude_gpus = exclude_gpus
        # We only ever need 1 item in each queue. This is 2 items cached (1 in queue 1 waiting
        # for queue) at each point. Adding more just stacks RAM with no speed benefit.
        self._queue_size = 1
        # TODO Calculate scaling for more plugins than currently exist in _parallel_scaling
        self._scaling_fallback = 0.4
        self._vram_stats = self._get_vram_stats()
        self._detect = self._load_detect(detector, aligner, rotate_images, min_size, configfile)
        self._align = self._load_align(aligner,
                                       configfile,
                                       normalize_method,
                                       re_feed,
                                       re_align,
                                       disable_filter)
        self._recognition = self._load_recognition(recognition, configfile)
        self._mask = [self._load_mask(mask, configfile) for mask in maskers]
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
        logger.trace(retval)  # type:ignore[attr-defined]
        return retval

    @property
    def aligner(self) -> Aligner:
        """ The currently selected aligner plugin """
        assert self._align is not None
        return self._align

    @property
    def recognition(self) -> Identity:
        """ The currently selected recognition plugin """
        assert self._recognition is not None
        return self._recognition

    def reset_phase_index(self) -> None:
        """ Reset the current phase index back to 0. Used for when batch processing is used in
        extract. """
        self._phase_index = 0

    def set_batchsize(self,
                      plugin_type: T.Literal["align", "detect"],
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

    def detected_faces(self) -> Generator[ExtractMedia, None, None]:
        """ Generator that returns results, frame by frame from the extraction pipeline

        This is the exit point for the extraction pipeline and is used to obtain the output
        of any pipeline :attr:`phase`

        Yields
        ------
        faces: :class:`~plugins.extract.extract_media.ExtractMedia`
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
                self._check_and_raise_error()
                faces = out_queue.get(True, 1)
                if faces == "EOF":
                    break
            except QueueEmpty:
                continue
            yield faces

        self._join_threads()
        if self.final_pass:
            for plugin in self._all_plugins:
                plugin.on_completion()
            logger.debug("Detection Complete")
        else:
            self._phase_index += 1
            logger.debug("Switching to phase: %s", self._current_phase)

    def _disable_lm_maskers(self) -> None:
        """ Disable any 68 point landmark based maskers if alignment data is not 2D 68
        point landmarks and update the process flow/phases accordingly """
        logger.warning("Alignment data is not 68 point 2D landmarks. Some Faceswap functionality "
                       "will be unavailable for these faces")

        rem_maskers = [m.name for m in self._mask
                       if m is not None and m.landmark_type == LandmarkType.LM_2D_68]
        self._mask = [m for m in self._mask if m is None or m.name not in rem_maskers]

        self._flow = [
            item for item in self._flow
            if not item.startswith("mask")
            or item.startswith("mask") and int(item.rsplit("_", maxsplit=1)[-1]) < len(self._mask)]

        self._phases = [[s for s in p if s in self._flow] for p in self._phases
                        if any(t in p for t in self._flow)]

        for queue in self._queues:
            queue_manager.del_queue(queue)
        del self._queues
        self._queues = self._add_queues()

        logger.warning("The following maskers have been disabled due to unsupported landmarks: %s",
                       rem_maskers)

    def import_data(self, input_location: str) -> None:
        """ Import json data to the detector and/or aligner if 'import' plugin has been selected

        Parameters
        ----------
        input_location: str
            Full path to the input location for the extract process
        """
        assert self._detect is not None
        import_plugins: list[DetectImport | AlignImport] = [
            p for p in (self._detect, self.aligner)  # type:ignore[misc]
            if T.cast(str, p.name).lower() == "external"]

        if not import_plugins:
            return

        align_origin = None
        assert self.aligner.name is not None
        if self.aligner.name.lower() == "external":
            align_origin = self.aligner.config["origin"]

        logger.info("Importing external data for %s from json file...",
                    " and ".join([p.__class__.__name__ for p in import_plugins]))

        folder = input_location
        folder = folder if os.path.isdir(folder) else os.path.dirname(folder)

        last_fname = ""
        is_68_point = True
        for plugin in import_plugins:
            plugin_type = plugin.__class__.__name__
            path = os.path.join(folder, plugin.config["file_name"])
            if not os.path.isfile(path):
                raise FaceswapError(f"{plugin_type} import file could not be found at '{path}'")

            if path != last_fname:  # Different import file for aligner data
                last_fname = path
                data = get_serializer("json").load(path)

            if plugin_type == "Detect":
                plugin.import_data(data, align_origin)  # type:ignore[call-arg]
            else:
                plugin.import_data(data)  # type:ignore[call-arg]
                is_68_point = plugin.landmark_type == LandmarkType.LM_2D_68  # type:ignore[union-attr]  # noqa:E501  # pylint:disable="line-too-long"

        if not is_68_point:
            self._disable_lm_maskers()

        logger.info("Imported external data")

    # <<< INTERNAL METHODS >>> #
    @property
    def _parallel_scaling(self) -> dict[int, float]:
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
    def _vram_per_phase(self) -> dict[str, float]:
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
    def _current_phase(self) -> list[str]:
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
    def _all_plugins(self) -> list[PluginExtractor]:
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
    def _active_plugins(self) -> list[PluginExtractor]:
        """ Return the plugins that are currently active based on pass """
        retval = []
        for phase in self._current_phase:
            plugin_type, idx = self._get_plugin_type_and_index(phase)
            attr = getattr(self, f"_{plugin_type}")
            retval.append(attr[idx] if idx is not None else attr)
        logger.trace("Active plugins: %s", retval)  # type: ignore
        return retval

    @staticmethod
    def _set_flow(detector: str | None,
                  aligner: str | None,
                  masker: list[str | None],
                  recognition: str | None) -> list[str]:
        """ Set the flow list based on the input plugins

        Parameters
        ----------
        detector: str or ``None``
            The name of a detector plugin as exists in :mod:`plugins.extract.detect`
        aligner: str or ``None
            The name of an aligner plugin as exists in :mod:`plugins.extract.align`
        masker: str or list or ``None
            The name of a masker plugin(s) as exists in :mod:`plugins.extract.mask`.
            This can be a single masker or a list of multiple maskers
        recognition: str or ``None``
            The name of the recognition plugin to use. ``None`` to not do face recognition.
        """
        logger.debug("detector: %s, aligner: %s, masker: %s recognition: %s",
                     detector, aligner, masker, recognition)
        retval = []
        if detector is not None and detector.lower() != "none":
            retval.append("detect")
        if aligner is not None and aligner.lower() != "none":
            retval.append("align")
        if recognition is not None and recognition.lower() != "none":
            retval.append("recognition")
        retval.extend([f"mask_{idx}"
                       for idx, mask in enumerate(masker)
                       if mask is not None and mask.lower() != "none"])
        logger.debug("flow: %s", retval)
        return retval

    @staticmethod
    def _get_plugin_type_and_index(flow_phase: str) -> tuple[str, int | None]:
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
            idx: int | None = int(sidx)
            plugin_type = "_".join(flow_phase.split("_")[:-1])
        else:
            plugin_type = flow_phase
            idx = None
        return plugin_type, idx

    def _add_queues(self) -> dict[str, EventQueue]:
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
    def _get_vram_stats() -> dict[str, int | str]:
        """ Obtain statistics on available VRAM and subtract a constant buffer from available vram.

        Returns
        -------
        dict
            Statistics on available VRAM
        """
        vram_buffer = 256  # Leave a buffer for VRAM allocation
        gpu_stats = GPUStats()
        stats = gpu_stats.get_card_most_free()
        retval: dict[str, int | str] = {"count": gpu_stats.device_count,
                                        "device": stats.device,
                                        "vram_free": int(stats.free - vram_buffer),
                                        "vram_total": int(stats.total)}
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

        logger.verbose("%s - %sMB free of %sMB",  # type: ignore
                       self._vram_stats["device"],
                       self._vram_stats["vram_free"],
                       self._vram_stats["vram_total"])
        if T.cast(int, self._vram_stats["vram_free"]) <= self._total_vram_required:
            logger.warning("Not enough free VRAM for parallel processing. "
                           "Switching to serial")
            return False
        return True

    def _set_phases(self, multiprocess: bool) -> list[list[str]]:
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
        phases: list[list[str]] = []
        current_phase: list[str] = []
        available = T.cast(int, self._vram_stats["vram_free"])
        for phase in self._flow:
            num_plugins = len([p for p in current_phase if self._vram_per_phase[p] > 0])
            num_plugins += 1 if self._vram_per_phase[phase] > 0 else 0
            scaling = self._parallel_scaling.get(num_plugins, self._scaling_fallback)
            required = sum(self._vram_per_phase[p] for p in current_phase + [phase]) * scaling
            logger.debug("Num plugins for phase: %s, scaling: %s, vram required: %s",
                         num_plugins, scaling, required)
            if required <= available and multiprocess:
                logger.debug("Required: %s, available: %s. Adding phase '%s' to current phase: %s",
                             required, available, phase, current_phase)
                current_phase.append(phase)
            elif len(current_phase) == 0 or not multiprocess:
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
                    aligner: str | None,
                    configfile: str | None,
                    normalize_method: T.Literal["none", "clahe", "hist", "mean"] | None,
                    re_feed: int,
                    re_align: bool,
                    disable_filter: bool) -> Aligner | None:
        """ Set global arguments and load aligner plugin

        Parameters
        ----------
        aligner: str
            The aligner plugin to load or ``None`` for no aligner
        configfile: str
            Optional full path to custom config file
        normalize_method: str
            Optional normalization method to use
        re_feed: int
            The number of times to adjust the image and re-feed to get an average score
        re_align: bool
            ``True`` to obtain landmarks by passing the initially aligned face back through the
            aligner.
        disable_filter: bool
            Disable all aligner filters regardless of config option

        Returns
        -------
        Aligner plugin if one is specified otherwise ``None``
        """
        if aligner is None or aligner.lower() == "none":
            logger.debug("No aligner selected. Returning None")
            return None
        aligner_name = aligner.replace("-", "_").lower()
        logger.debug("Loading Aligner: '%s'", aligner_name)
        plugin = PluginLoader.get_aligner(aligner_name)(exclude_gpus=self._exclude_gpus,
                                                        configfile=configfile,
                                                        normalize_method=normalize_method,
                                                        re_feed=re_feed,
                                                        re_align=re_align,
                                                        disable_filter=disable_filter,
                                                        instance=self._instance)
        return plugin

    def _load_detect(self,
                     detector: str | None,
                     aligner: str | None,
                     rotation: str | None,
                     min_size: int,
                     configfile: str | None) -> Detector | None:
        """ Set global arguments and load detector plugin

        Parameters
        ----------
        detector: str | None
            The name of the face detection plugin to use. ``None`` for no detection
        aligner: str | None
            The name of the face aligner plugin to use. ``None`` for no aligner
        rotation: str | None
            The rotation to perform on detection. ``None`` for no rotation
        min_size: int
            The minimum size of detected faces to accept
        configfile: str | None
            Full path to a custom config file to use. ``None`` for default config

        Returns
        -------
        :class:`~plugins.extract.detect._base.Detector` | None
            The face detection plugin to use, or ``None`` if no detection to be performed
        """
        if detector is None or detector.lower() == "none":
            logger.debug("No detector selected. Returning None")
            return None
        detector_name = detector.replace("-", "_").lower()

        if aligner == "external" and detector_name != "external":
            logger.warning("Unsupported '%s' detector selected for 'External' aligner. Switching "
                           "detector to 'External'", detector_name)
            detector_name = aligner

        logger.debug("Loading Detector: '%s'", detector_name)
        plugin = PluginLoader.get_detector(detector_name)(exclude_gpus=self._exclude_gpus,
                                                          rotation=rotation,
                                                          min_size=min_size,
                                                          configfile=configfile,
                                                          instance=self._instance)
        return plugin

    def _load_mask(self,
                   masker: str | None,
                   configfile: str | None) -> Masker | None:
        """ Set global arguments and load masker plugin

        Parameters
        ----------
        masker: str or ``none``
            The name of the masker plugin to use or ``None`` if no masker
        configfile: str
            Full path to custom config.ini file or ``None`` to use default

        Returns
        -------
        :class:`~plugins.extract.mask._base.Masker` or ``None``
            The masker plugin to use or ``None`` if no masker selected
        """
        if masker is None or masker.lower() == "none":
            logger.debug("No masker selected. Returning None")
            return None
        masker_name = masker.replace("-", "_").lower()
        logger.debug("Loading Masker: '%s'", masker_name)
        plugin = PluginLoader.get_masker(masker_name)(exclude_gpus=self._exclude_gpus,
                                                      configfile=configfile,
                                                      instance=self._instance)
        return plugin

    def _load_recognition(self,
                          recognition: str | None,
                          configfile: str | None) -> Identity | None:
        """ Set global arguments and load recognition plugin """
        if recognition is None or recognition.lower() == "none":
            logger.debug("No recognition selected. Returning None")
            return None
        recognition_name = recognition.replace("-", "_").lower()
        logger.debug("Loading Recognition: '%s'", recognition_name)
        plugin = PluginLoader.get_recognition(recognition_name)(exclude_gpus=self._exclude_gpus,
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
        kwargs = {"in_queue": self._queues[in_qname], "out_queue": self._queues[out_qname]}

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
        backend = get_backend()
        if backend not in ("nvidia", "directml", "rocm"):
            logger.debug("Not updating batchsize requirements for backend: '%s'", backend)
            return
        if sum(plugin.vram for plugin in self._active_plugins) == 0:
            logger.debug("No plugins use VRAM. Not updating batchsize requirements.")
            return

        batch_required = sum(plugin.vram_per_batch * plugin.batchsize
                             for plugin in self._active_plugins)
        gpu_plugins = [p for p in self._current_phase if self._vram_per_phase[p] > 0]
        scaling = self._parallel_scaling.get(len(gpu_plugins), self._scaling_fallback)
        plugins_required = sum(self._vram_per_phase[p] for p in gpu_plugins) * scaling
        if plugins_required + batch_required <= T.cast(int, self._vram_stats["vram_free"]):
            logger.debug("Plugin requirements within threshold: (plugins_required: %sMB, "
                         "vram_free: %sMB)", plugins_required, self._vram_stats["vram_free"])
            return
        # Hacky split across plugins that use vram
        available_vram = (T.cast(int, self._vram_stats["vram_free"])
                          - plugins_required) // len(gpu_plugins)
        self._set_plugin_batchsize(gpu_plugins, available_vram)

    def _set_plugin_batchsize(self, gpu_plugins: list[str], available_vram: float) -> None:
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

    def _check_and_raise_error(self) -> None:
        """ Check all threads for errors and raise if one occurs """
        for plugin in self._active_plugins:
            plugin.check_and_raise_error()
