#!/usr/bin/env python3
""" Base class for Faceswap :mod:`~plugins.extract.detect`, :mod:`~plugins.extract.align` and
:mod:`~plugins.extract.mask` Plugins
"""
from __future__ import annotations
import logging
import typing as T

from dataclasses import dataclass, field

import numpy as np
from tensorflow.python.framework import errors_impl as tf_errors  # pylint:disable=no-name-in-module  # noqa

from lib.multithreading import MultiThread
from lib.queue_manager import queue_manager
from lib.utils import GetModel, FaceswapError
from ._config import Config
from . import ExtractMedia

if T.TYPE_CHECKING:
    from collections.abc import Callable, Generator, Sequence
    from queue import Queue
    import cv2
    from lib.align import DetectedFace
    from lib.model.session import KSession
    from .align._base import AlignerBatch
    from .detect._base import DetectorBatch
    from .mask._base import MaskerBatch
    from .recognition._base import RecogBatch

logger = logging.getLogger(__name__)
# TODO Run with warnings mode


def _get_config(plugin_name: str, configfile: str | None = None) -> dict[str, T.Any]:
    """ Return the configuration for the requested model

    Parameters
    ----------
    plugin_name: str
        The module name of the child plugin.
    configfile: str, optional
        Path to a :file:`./config/<plugin_type>.ini` file for this plugin. Default: use system
        configuration.

    Returns
    -------
    config_dict, dict
       A dictionary of configuration items from the configuration file
    """
    return Config(plugin_name, configfile=configfile).config_dict


BatchType = T.Union["DetectorBatch", "AlignerBatch", "MaskerBatch", "RecogBatch"]


@dataclass
class ExtractorBatch:
    """ Dataclass for holding a batch flowing through post Detector plugins.

    The batch size for post Detector plugins is not the same as the overall batch size.
    An image may contain 0 or more detected faces, and these need to be split and recombined
    to be able to utilize a plugin's internal batch size.

    Plugin types will inherit from this class and add required keys.

    Parameters
    ----------
    image: list
        List of :class:`numpy.ndarray` containing the original frames
    detected_faces: list
        List of :class:`~lib.align.DetectedFace` objects
    filename: list
        List of original frame filenames for the batch
    feed: :class:`numpy.ndarray`
        Batch of feed images to feed the net with
    prediction: :class:`numpy.nd.array`
        Batch of predictions. Direct output from the aligner net
    data: dict
        Any specific data required during the processing phase for a particular plugin
    """
    image: list[np.ndarray] = field(default_factory=list)
    detected_faces: Sequence[DetectedFace | list[DetectedFace]] = field(default_factory=list)
    filename: list[str] = field(default_factory=list)
    feed: np.ndarray = np.array([])
    prediction: np.ndarray = np.array([])
    data: list[dict[str, T.Any]] = field(default_factory=list)

    def __repr__(self) -> str:
        """ Prettier repr for debug printing """
        data = [{k: (v.shape, v.dtype) if isinstance(v, np.ndarray) else v for k, v in dat.items()}
                for dat in self.data]
        return (f"{self.__class__.__name__}("
                f"image={[(img.shape, img.dtype) for img in self.image]}, "
                f"detected_faces={self.detected_faces}, "
                f"filename={self.filename}, "
                f"feed={[(f.shape, f.dtype) for f in self.feed]}, "
                f"prediction=({self.prediction.shape}, {self.prediction.dtype}), "
                f"data={data}")


class Extractor():
    """ Extractor Plugin Object

    All ``_base`` classes for Aligners, Detectors and Maskers inherit from this class.

    This class sets up a pipeline for working with ML plugins.

    Plugins are split into 3 threads, to utilize Numpy and CV2s parallel processing, as well as
    allow the predict function of the model to sit in a dedicated thread.
    A plugin is expected to have 3 core functions, each in their own thread:
    - :func:`process_input()` - Prepare the data for feeding into a model
    - :func:`predict` - Feed the data through the model
    - :func:`process_output()` - Perform any data post-processing

    Parameters
    ----------
    git_model_id: int
        The second digit in the github tag that identifies this model. See
        https://github.com/deepfakes-models/faceswap-models for more information
    model_filename: str
        The name of the model file to be loaded
    exclude_gpus: list, optional
        A list of indices correlating to connected GPUs that Tensorflow should not use. Pass
        ``None`` to not exclude any GPUs. Default: ``None``
    configfile: str, optional
        Path to a custom configuration ``ini`` file. Default: Use system configfile
    instance: int, optional
        If this plugin is being executed multiple times (i.e. multiple pipelines have been
        launched), the instance of the plugin must be passed in for naming convention reasons.
        Default: 0


    The following attributes should be set in the plugin's :func:`__init__` method after
    initializing the parent.

    Attributes
    ----------
    name: str
        Name of this plugin. Used for display purposes.
    input_size: int
        The input size to the model in pixels across one edge. The input size should always be
        square.
    color_format: str
        Color format for model. Must be ``'BGR'``, ``'RGB'`` or ``'GRAY'``. Defaults to ``'BGR'``
        if not explicitly set.
    vram: int
        Approximate VRAM used by the model at :attr:`input_size`. Used to calculate the
        :attr:`batchsize`. Be conservative to avoid OOM.
    vram_warnings: int
        Approximate VRAM used by the model at :attr:`input_size` that will still run, but generates
        warnings. Used to calculate the :attr:`batchsize`. Be conservative to avoid OOM.
    vram_per_batch: int
        Approximate additional VRAM used by the model for each additional batch. Used to calculate
        the :attr:`batchsize`. Be conservative to avoid OOM.

    See Also
    --------
    plugins.extract.detect._base : Detector parent class for extraction plugins.
    plugins.extract.align._base : Aligner parent class for extraction plugins.
    plugins.extract.mask._base : Masker parent class for extraction plugins.
    plugins.extract.pipeline : The extract pipeline that configures and calls all plugins

    """
    def __init__(self,
                 git_model_id: int | None = None,
                 model_filename: str | list[str] | None = None,
                 exclude_gpus: list[int] | None = None,
                 configfile: str | None = None,
                 instance: int = 0) -> None:
        logger.debug("Initializing %s: (git_model_id: %s, model_filename: %s, exclude_gpus: %s, "
                     "configfile: %s, instance: %s, )", self.__class__.__name__, git_model_id,
                     model_filename, exclude_gpus, configfile, instance)
        self._is_initialized = False
        self._instance = instance
        self._exclude_gpus = exclude_gpus
        self.config = _get_config(".".join(self.__module__.split(".")[-2:]), configfile=configfile)
        """ dict: Config for this plugin, loaded from ``extract.ini`` configfile """

        self.model_path = self._get_model(git_model_id, model_filename)
        """ str or list: Path to the model file(s) (if required). Multiple model files should
        be a list of strings """

        # << SET THE FOLLOWING IN PLUGINS __init__ IF DIFFERENT FROM DEFAULT >> #
        self.name: str | None = None
        self.input_size = 0
        self.color_format: T.Literal["BGR", "RGB", "GRAY"] = "BGR"
        self.vram = 0
        self.vram_warnings = 0  # Will run at this with warnings
        self.vram_per_batch = 0

        # << THE FOLLOWING ARE SET IN self.initialize METHOD >> #
        self.queue_size = 1
        """ int: Queue size for all internal queues. Set in :func:`initialize()` """

        self.model: KSession | cv2.dnn.Net | None = None
        """varies: The model for this plugin. Set in the plugin's :func:`init_model()` method """

        # For detectors that support batching, this should be set to  the calculated batch size
        # that the amount of available VRAM will support.
        self.batchsize = 1
        """ int: Batchsize for feeding this model. The number of images the model should
        feed through at once. """

        self._queues: dict[str, Queue] = {}
        """ dict: in + out queues and internal queues for this plugin, """

        self._threads: list[MultiThread] = []
        """ list: Internal threads for this plugin """

        self._extract_media: dict[str, ExtractMedia] = {}
        """ dict: The :class:`~plugins.extract.extract_media.ExtractMedia` objects currently being
        processed. Stored at input for pairing back up on output of extractor process """

        # << THE FOLLOWING PROTECTED ATTRIBUTES ARE SET IN PLUGIN TYPE _base.py >>> #
        self._plugin_type: T.Literal["align", "detect", "recognition", "mask"] | None = None
        """ str: Plugin type. ``detect`, ``align``, ``recognise`` or ``mask`` set in
        ``<plugin_type>._base`` """

        # << Objects for splitting frame's detected faces and rejoining them >>
        # << for post-detector pliugins                                      >>
        self._faces_per_filename: dict[str, int] = {}  # Tracking for recompiling batches
        self._rollover: ExtractMedia | None = None  # batch rollover items
        self._output_faces: list[DetectedFace] = []  # Recompiled output faces from plugin

        logger.debug("Initialized _base %s", self.__class__.__name__)

    # <<< OVERIDABLE METHODS >>> #
    def init_model(self) -> None:
        """ **Override method**

        Override this method to execute the specific model initialization method """
        raise NotImplementedError

    def process_input(self, batch: BatchType) -> None:
        """ **Override method**

        Override this method for specific extractor pre-processing of image

        Parameters
        ----------
        batch : :class:`ExtractorBatch`
            Contains the batch that is currently being passed through the plugin process
        """
        raise NotImplementedError

    def predict(self, feed: np.ndarray) -> np.ndarray:
        """ **Override method**

        Override this method for specific extractor model prediction function

        Parameters
        ----------
        feed: :class:`numpy.ndarray`
            The feed images for the batch

        Notes
        -----
        Input for :func:`predict` should have been set in :func:`process_input`

        Output from the model should populate the key :attr:`prediction` of the :attr:`batch`.

        For Detect:
            the expected output for the :attr:`prediction` of the :attr:`batch` should be a
            ``list`` of :attr:`batchsize` of detected face points. These points should be either
            a ``list``, ``tuple`` or ``numpy.ndarray`` with the first 4 items being the `left`,
            `top`, `right`, `bottom` points, in that order
        """
        raise NotImplementedError

    def process_output(self, batch: BatchType) -> None:
        """ **Override method**

        Override this method for specific extractor model post predict function

        Parameters
        ----------
        batch: :class:`ExtractorBatch`
            Contains the batch that is currently being passed through the plugin process

        Notes
        -----
        For Align:
            The :attr:`landmarks` must be populated in :attr:`batch` from this method.
            This should be a ``list`` or :class:`numpy.ndarray` of :attr:`batchsize` containing a
            ``list``, ``tuple`` or :class:`numpy.ndarray` of `(x, y)` coordinates of the 68 point
            landmarks as calculated from the :attr:`model`.
        """
        raise NotImplementedError

    def on_completion(self) -> None:
        """ Override to perform an action when the extract process has completed. By default, no
        action is undertaken """
        return

    def _predict(self, batch: BatchType) -> BatchType:
        """ **Override method** (at `<plugin_type>` level)

        This method should be overridden at the `<plugin_type>` level (IE.
        ``plugins.extract.detect._base`` or ``plugins.extract.align._base``) and should not
        be overridden within plugins themselves.

        It acts as a wrapper for the plugin's ``self.predict`` method and handles any
        predict processing that is consistent for all plugins within the `plugin_type`

        Parameters
        ----------
        batch: :class:`ExtractorBatch`
            Contains the batch that is currently being passed through the plugin process
        """
        raise NotImplementedError

    def _process_input(self, batch: BatchType) -> BatchType:
        """ **Override method** (at `<plugin_type>` level)

        This method should be overridden at the `<plugin_type>` level (IE.
        ``plugins.extract.detect._base`` or ``plugins.extract.align._base``) and should not
        be overridden within plugins themselves.

        It acts as a wrapper for the plugin's :func:`process_input` method and handles any
        input processing that is consistent for all plugins within the `plugin_type`.

        If this method is not overridden then the plugin's :func:`process_input` is just called.

        Parameters
        ----------
        batch: :class:`ExtractorBatch`
            Contains the batch that is currently being passed through the plugin process

        Notes
        -----
        When preparing an input to the model a the attribute :attr:`feed` must be added
        to the :attr:`batch` which contains this input.
        """
        self.process_input(batch)
        return batch

    def _process_output(self, batch: BatchType) -> BatchType:
        """ **Override method** (at `<plugin_type>` level)

        This method should be overridden at the `<plugin_type>` level (IE.
        ``plugins.extract.detect._base`` or ``plugins.extract.align._base``) and should not
        be overridden within plugins themselves.

        It acts as a wrapper for the plugin's :func:`process_output` method and handles any
        output processing that is consistent for all plugins within the `plugin_type`.

        If this method is not overridden then the plugin's :func:`process_output` is just called.

        Parameters
        ----------
        batch: :class:`ExtractorBatch`
            Contains the batch that is currently being passed through the plugin process
        """
        self.process_output(batch)
        return batch

    def finalize(self, batch: BatchType) -> Generator[ExtractMedia, None, None]:
        """ **Override method** (at `<plugin_type>` level)

        This method should be overridden at the `<plugin_type>` level (IE.
        :mod:`plugins.extract.detect._base`, :mod:`plugins.extract.align._base` or
        :mod:`plugins.extract.mask._base`) and should not be overridden within plugins themselves.

        Handles consistent finalization for all plugins that exist within that plugin type. Its
        input is always the output from :func:`process_output()`

        Parameters
        ----------
        batch: :class:`ExtractorBatch`
            Contains the batch that is currently being passed through the plugin process
        """
        raise NotImplementedError

    def get_batch(self, queue: Queue) -> tuple[bool, BatchType]:
        """ **Override method** (at `<plugin_type>` level)

        This method should be overridden at the `<plugin_type>` level (IE.
        :mod:`plugins.extract.detect._base`, :mod:`plugins.extract.align._base` or
        :mod:`plugins.extract.mask._base`) and should not be overridden within plugins themselves.

        Get :class:`~plugins.extract.extract_media.ExtractMedia` items from the queue in batches of
        :attr:`batchsize`

        Parameters
        ----------
        queue : queue.Queue()
            The ``queue`` that the batch will be fed from. This will be the input to the plugin.
        """
        raise NotImplementedError

    # <<< THREADING METHODS >>> #
    def start(self) -> None:
        """ Start all threads

        Exposed for :mod:`~plugins.extract.pipeline` to start plugin's threads
        """
        for thread in self._threads:
            thread.start()

    def join(self) -> None:
        """ Join all threads

        Exposed for :mod:`~plugins.extract.pipeline` to join plugin's threads
        """
        for thread in self._threads:
            thread.join()

    def check_and_raise_error(self) -> None:
        """ Check all threads for errors

        Exposed for :mod:`~plugins.extract.pipeline` to check plugin's threads for errors
        """
        for thread in self._threads:
            thread.check_and_raise_error()

    def rollover_collector(self, queue: Queue) -> T.Literal["EOF"] | ExtractMedia:
        """ For extractors after the Detectors, the number of detected faces per frame vs extractor
        batch size mean that faces will need to be split/re-joined with frames. The rollover
        collector can be used to rollover items that don't fit in a batch.

        Collect the item from the :attr:`_rollover` dict or from the queue. Add face count per
        frame to self._faces_per_filename for joining batches back up in finalize

        Parameters
        ----------
        queue: :class:`queue.Queue`
            The input queue to the aligner. Should contain
            :class:`~plugins.extract.extract_media.ExtractMedia` objects

        Returns
        -------
        :class:`~plugins.extract.extract_media.ExtractMedia` or EOF
            The next extract media object, or EOF if pipe has ended
        """
        if self._rollover is not None:
            logger.trace("Getting from _rollover: (filename: `%s`, faces: %s)",  # type:ignore
                         self._rollover.filename, len(self._rollover.detected_faces))
            item: T.Literal["EOF"] | ExtractMedia = self._rollover
            self._rollover = None
        else:
            next_item = self._get_item(queue)
            # Rollover collector should only be used at entry to plugin
            assert isinstance(next_item, (ExtractMedia, str))
            item = next_item
            if item != "EOF":
                logger.trace("Getting from queue: (filename: %s, faces: %s)",  # type:ignore
                             item.filename, len(item.detected_faces))
                self._faces_per_filename[item.filename] = len(item.detected_faces)
        return item

    # <<< PROTECTED ACCESS METHODS >>> #
    # <<< INIT METHODS >>> #
    @classmethod
    def _get_model(cls,
                   git_model_id: int | None,
                   model_filename: str | list[str] | None) -> str | list[str] | None:
        """ Check if model is available, if not, download and unzip it """
        if model_filename is None:
            logger.debug("No model_filename specified. Returning None")
            return None
        if git_model_id is None:
            logger.debug("No git_model_id specified. Returning None")
            return None
        model = GetModel(model_filename, git_model_id)
        return model.model_path

    # <<< PLUGIN INITIALIZATION >>> #
    def initialize(self, *args, **kwargs) -> None:
        """ Initialize the extractor plugin

            Should be called from :mod:`~plugins.extract.pipeline`
        """
        logger.debug("initialize %s: (args: %s, kwargs: %s)",
                     self.__class__.__name__, args, kwargs)
        assert self._plugin_type is not None and self.name is not None
        if self._is_initialized:
            # When batch processing, plugins will be initialized on first job in batch
            logger.debug("Plugin already initialized: %s (%s)",
                         self.name, self._plugin_type.title())
            return

        logger.info("Initializing %s (%s)...", self.name, self._plugin_type.title())
        self.queue_size = 1
        name = self.name.replace(" ", "_").lower()
        self._add_queues(kwargs["in_queue"],
                         kwargs["out_queue"],
                         [f"predict_{name}", f"post_{name}"])
        self._compile_threads()
        try:
            self.init_model()
        except tf_errors.UnknownError as err:
            if "failed to get convolution algorithm" in str(err).lower():
                msg = ("Tensorflow raised an unknown error. This is most likely caused by a "
                       "failure to launch cuDNN which can occur for some GPU/Tensorflow "
                       "combinations. You should enable `allow_growth` to attempt to resolve this "
                       "issue:"
                       "\nGUI: Go to Settings > Extract Plugins > Global and enable the "
                       "`allow_growth` option."
                       "\nCLI: Go to `faceswap/config/extract.ini` and change the `allow_growth "
                       "option to `True`.")
                raise FaceswapError(msg) from err
            raise err
        self._is_initialized = True
        logger.info("Initialized %s (%s) with batchsize of %s",
                    self.name, self._plugin_type.title(), self.batchsize)

    def _add_queues(self,
                    in_queue: Queue,
                    out_queue: Queue,
                    queues: list[str]) -> None:
        """ Add the queues
            in_queue and out_queue should be previously created queue manager queues.
            queues should be a list of queue names """
        self._queues["in"] = in_queue
        self._queues["out"] = out_queue
        for q_name in queues:
            self._queues[q_name] = queue_manager.get_queue(
                name=f"{self._plugin_type}{self._instance}_{q_name}",
                maxsize=self.queue_size)

    # <<< THREAD METHODS >>> #
    def _compile_threads(self) -> None:
        """ Compile the threads into self._threads list """
        assert self.name is not None
        logger.debug("Compiling %s threads", self._plugin_type)
        name = self.name.replace(" ", "_").lower()
        base_name = f"{self._plugin_type}_{name}"
        self._add_thread(f"{base_name}_input",
                         self._process_input,
                         self._queues["in"],
                         self._queues[f"predict_{name}"])
        self._add_thread(f"{base_name}_predict",
                         self._predict,
                         self._queues[f"predict_{name}"],
                         self._queues[f"post_{name}"])
        self._add_thread(f"{base_name}_output",
                         self._process_output,
                         self._queues[f"post_{name}"],
                         self._queues["out"])
        logger.debug("Compiled %s threads: %s", self._plugin_type, self._threads)

    def _add_thread(self,
                    name: str,
                    function: Callable[[BatchType], BatchType],
                    in_queue: Queue,
                    out_queue: Queue) -> None:
        """ Add a MultiThread thread to self._threads """
        logger.debug("Adding thread: (name: %s, function: %s, in_queue: %s, out_queue: %s)",
                     name, function, in_queue, out_queue)
        self._threads.append(MultiThread(target=self._thread_process,
                                         name=name,
                                         function=function,
                                         in_queue=in_queue,
                                         out_queue=out_queue))
        logger.debug("Added thread: %s", name)

    def _obtain_batch_item(self, function: Callable[[BatchType], BatchType],
                           in_queue: Queue,
                           out_queue: Queue) -> BatchType | None:
        """ Obtain the batch item from the in queue for the current process.

        Parameters
        ----------
        function: callable
            The current plugin function being run
        in_queue: :class:`queue.Queue`
            The input queue for the function
        out_queue: :class:`queue.Queue`
            The output queue from the function

        Returns
        -------
        :class:`ExtractorBatch` or ``None``
            The batch, if one exists, or ``None`` if queue is exhausted
        """
        batch: T.Literal["EOF"] | BatchType | ExtractMedia
        if function.__name__ == "_process_input":  # Process input items to batches
            exhausted, batch = self.get_batch(in_queue)
            if exhausted:
                if batch.filename:
                    # Put the final batch
                    batch = function(batch)
                    out_queue.put(batch)
                return None
        else:
            batch = self._get_item(in_queue)
            if batch == "EOF":
                return None

        # ExtractMedia should only ever be the output of _get_item at the entry to a
        # plugin's pipeline (ie in _process_input)
        assert not isinstance(batch, ExtractMedia)
        return batch

    def _thread_process(self,
                        function: Callable[[BatchType], BatchType],
                        in_queue: Queue,
                        out_queue: Queue) -> None:
        """ Perform a plugin function in a thread

        Parameters
        ----------
        function: callable
            The current plugin function being run
        in_queue: :class:`queue.Queue`
            The input queue for the function
        out_queue: :class:`queue.Queue`
            The output queue from the function
         """
        logger.debug("threading: (function: '%s')", function.__name__)
        while True:
            batch = self._obtain_batch_item(function, in_queue, out_queue)
            if batch is None:
                break
            if not batch.filename:  # Batch not populated. Possible during re-aligns
                continue
            try:
                batch = function(batch)
            except tf_errors.UnknownError as err:
                if "failed to get convolution algorithm" in str(err).lower():
                    msg = ("Tensorflow raised an unknown error. This is most likely caused by a "
                           "failure to launch cuDNN which can occur for some GPU/Tensorflow "
                           "combinations. You should enable `allow_growth` to attempt to resolve "
                           "this issue:"
                           "\nGUI: Go to Settings > Extract Plugins > Global and enable the "
                           "`allow_growth` option."
                           "\nCLI: Go to `faceswap/config/extract.ini` and change the "
                           "`allow_growth option to `True`.")
                    raise FaceswapError(msg) from err
                raise err
            if function.__name__ == "_process_output":
                # Process output items to individual items from batch
                for item in self.finalize(batch):
                    out_queue.put(item)
            else:
                out_queue.put(batch)
        logger.debug("Putting EOF")
        out_queue.put("EOF")

    # <<< QUEUE METHODS >>> #
    def _get_item(self, queue: Queue) -> T.Literal["EOF"] | ExtractMedia | BatchType:
        """ Yield one item from a queue """
        item = queue.get()
        if isinstance(item, ExtractMedia):
            logger.trace("filename: '%s', image shape: %s, detected_faces: %s, "  # type:ignore
                         "queue: %s, item: %s",
                         item.filename, item.image_shape, item.detected_faces, queue, item)
            self._extract_media[item.filename] = item
        else:
            logger.trace("item: %s, queue: %s", item, queue)  # type:ignore
        return item
