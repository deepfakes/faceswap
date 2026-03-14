#! /usr/env/bin/python3
"""Handles extract plugins and runners """
from __future__ import annotations

import logging
import typing as T
from queue import Queue, Empty as QueueEmpty, Full as QueueFull
from threading import current_thread, main_thread
from time import sleep
from uuid import uuid4

import numpy as np
import numpy.typing as npt

from lib.align.constants import LandmarkType
from lib.logger import parse_class_init
from lib.multithreading import ErrorState, FSThread
from lib.utils import get_module_objects
from .iterator import InboundIterator, InputIterator, InterimIterator, OutputIterator
from .objects import ExtractBatch, FrameFaces, ExtractSignal


if T.TYPE_CHECKING:
    from .handler import ExtractHandler, ExtractHandlerFace
    from lib.align.alignments import PNGHeaderSourceDict
    from lib.align.detected_face import DetectedFace

logger = logging.getLogger(__name__)


_PLUGIN_REGISTER: dict[str, list[ExtractRunner]] = {}
"""uuid of the input runner to list of runners in the chain. Used to assert build order and when
calling the runner in passthrough mode and tracking multiple pipelines """


class PluginThreads:
    """Handles the holding of threads that will run a plugin's various subprocesses.

    Parameters
    ----------
    name
        The name of the plugin that the threads are being created for
    """
    def __init__(self, name: str) -> None:
        self._name = name
        self._threads: dict[str, FSThread] = {}
        self._backup_error_state = ErrorState()
        """This is used when a plugin has no threads to run. Specifically the File handler never
        has threads, so there will never be a thread error. If running in the main thread it is
        safe to return an unused object"""
        self._external_error_state: ErrorState | None = None

    @property
    def error_state(self) -> ErrorState:
        """The global FSThread error state object"""
        if not self._threads and self._external_error_state is None:
            return self._backup_error_state
        if self._external_error_state is not None:
            return self._external_error_state
        return list(self._threads.values())[0].error_state

    @property
    def enabled(self) -> list[str]:
        """The thread names that have been registered within this group"""
        return list(self._threads)

    def __repr__(self) -> str:
        """Pretty print for logging"""
        obj = f"{self.__class__.__name__}(name={self._name})"
        threads = self.enabled
        alive = [x.is_alive() for x in self._threads.values()]
        error = None if not threads else list(self._threads.values())[0].error_state.has_error
        info = f"[threads: {threads}, alive: {alive}, error: {error}]"
        return f"{obj} {info}"

    def register_thread(self,
                        name: str,
                        target: T.Callable[[T.Literal["pre_process", "process", "post_process"]],
                                           None]) -> None:
        """Register a thread

        Parameters
        ----------
        name
            The name of the plugin handler's processor that is running in the thread
        target
            The function to run within the thread
        """
        full_name = f"{self._name}.{name}"
        logger.debug("[%s] Registering thread: '%s'", self._name, name)
        self._threads[name] = FSThread(target=target, name=full_name, args=(name, ))

    def start(self) -> None:
        """Start the plugin's threads"""
        for key, thread in self._threads.items():
            logger.debug("[%s] Starting thread: '%s'", self._name, key)
            thread.start()

    def join(self) -> None:
        """Join all of the plugin's threads"""
        for key, thread in self._threads.items():
            logger.debug("[%s] Joining thread: '%s'", self._name, key)
            thread.join()

    def is_alive(self) -> bool:
        """Test if any thread is alive

        Returns
        -------
        ``True`` if any thread is alive otherwise False
        """
        return any(t.is_alive() for t in self._threads.values())

    def register_external_error_state(self, state: ErrorState) -> None:
        """Register an external error state object.

        If we are not running any threads (specifically, file handler), the pipeline can hang the
        calling thread. The error state from the calling thread can be populated here. This can
        only be called if no threads have been registered

        Parameters
        ----------
        state
            The ErrorState object to register

        Raises
        ------
        RuntimeError
            If an ErrorState object is registered when this object already contains threads
        """
        logger.debug("Registering external ErrorState: %s", state)
        if self._external_error_state is not None:
            logger.debug("Error state already registered: %s", state)
            return
        if self._threads:
            raise RuntimeError("You cannot register an ErrorState object when threads exist")
        self._external_error_state = state


HandlerT = T.TypeVar("HandlerT", "ExtractHandler", "ExtractHandlerFace")


class ExtractRunner(T.Generic[HandlerT]):
    """Runs an extract plugin

    Parameters
    ----------
    handler
        The plugin handler that this runner will execute
    """
    def __init__(self, handler: HandlerT) -> None:
        logger.debug(parse_class_init(locals()))
        self._handler: HandlerT = handler
        self._plugin_name = handler.plugin_name
        self._queues: dict[str, Queue] = {}
        self._is_first = False
        self._uuid: str | None = None
        """Unique identifier for plugin ordering and multi-plugin tracking. Populated on __call__
        to ensure a plugin is not called prior to it's input runner being called"""
        self._threads = self._get_threads()
        self._inbound_iterator: InboundIterator | InputIterator
        self._output_iterator: OutputIterator | None = None

    def __repr__(self) -> str:
        """Pretty print for logging"""
        return f"{self.__class__.__name__}(handler={self.handler})"

    def __iter__(self) -> T.Self:
        """This is an iterator"""
        return self

    def __next__(self) -> FrameFaces:
        """Obtain the next item from the plugin's output

        Returns
        -------
        The media object with populated detected faces for a frame
        """
        if self._output_iterator is None:
            raise RuntimeError(f"[{self._plugin_name}] You can only iterate the final runner in a "
                               "pipeline chain.")
        retval = next((self._output_iterator), None)
        if self._threads.error_state.has_error:
            current = current_thread()
            if current is main_thread():
                self._threads.error_state.re_raise()
            else:
                logger.debug("[%s.%s] Thread error detected in worker thread",
                             current.name, self.__class__.__name__)
                retval = None
        if retval is None:
            raise StopIteration
        return retval

    @property
    def handler(self) -> HandlerT:
        """The plugin handler that this runner is executing"""
        return self._handler

    @property
    def out_queue(self) -> Queue[ExtractBatch]:
        """The output queue from this plugin runner"""
        return self._queues["out"]

    @property
    def uuid(self) -> str:
        """Unique identifier for plugin ordering and multi-plugin tracking"""
        assert self._uuid is not None
        return self._uuid

    def _delete_images(self, batch: ExtractBatch) -> None:
        """Delete any images from the batch where there are no faces

        Parameters
        ----------
        batch
            The batch of data to delete images without faces from
        """
        no_boxes = [i for i in range(len(batch.images)) if i not in batch.frame_ids]
        if not no_boxes:
            return
        logger.trace(  # type:ignore[attr-defined]
            "[%s.out] Deleting %s of %s images with no bounding boxes",
            self._plugin_name, len(no_boxes), len(batch.images))
        for idx in no_boxes:
            batch.images[idx] = np.empty(shape=(0, 0, 3), dtype=np.uint8)

    def _clean_output(self,
                      batch: ExtractBatch | ExtractSignal,
                      next_process: T.Literal["process", "post_process", "out"]) -> None:
        """Remove any images from the batch that have no detected faces and delete any internal
        plugin attributes when outputting from the plugin

        Parameters
        ----------
        batch
            The batch of data to potentially delete data from or ``None`` for EOF
        next_process
            The next process for the plugin
        """
        if next_process != "out" or isinstance(batch, ExtractSignal):
            return
        self._delete_images(batch)
        if hasattr(batch, "matrices"):
            del batch.matrices
        if hasattr(batch, "data"):
            del batch.data

    def _put_data(self, process: str, batch: ExtractBatch | ExtractSignal) -> None:
        """Put data from a plugin's process into the next queue. If this is the first plugin in
        the pipeline and we are queueing data out from the plugin, then remove any images which
        have no detected faces.

        Parameters
        ----------
        process
            The name of the process that wishes to output data
        batch
            The batch of data to put to the next queue or an ExtractSignal after the final
            iteration
        """
        queue_names = list(self._queues)
        queue_index = queue_names.index(process) + 1
        next_process = T.cast(T.Literal["process", "post_process", "out"],
                              queue_names[queue_index])
        assert next_process in ("process", "post_process", "out")
        queue = self._queues[next_process]
        self._clean_output(batch, next_process)
        logger.trace("[%s.%s] Outputting to '%s': %s",  # type:ignore[attr-defined]
                     self._plugin_name,
                     process,
                     next_process,
                     batch.name if isinstance(batch, ExtractSignal) else batch)

        while True:
            if self._threads.error_state.has_error:
                logger.debug("[%s.%s] thread error detected. Not putting",
                             self._plugin_name, process)
                return
            try:
                logger.trace("[%s.%s] Putting to out queue: %s",  # type:ignore[attr-defined]
                             self._plugin_name,
                             process,
                             batch.name if isinstance(batch, ExtractSignal) else batch)
                queue.put(batch, timeout=0.2)
                break
            except QueueFull:
                logger.trace("[%s.%s] Waiting to put item",  # type:ignore[attr-defined]
                             self._plugin_name, process)
                continue

        if next_process == "out" and isinstance(batch, ExtractSignal):
            sleep(1)  # Wait for downstream plugins to flush
            self.handler.output_info()

    def _handle_zero_detections(self, process, batch: ExtractBatch) -> bool:
        """Check if the given batch is not a Detect batch and has detected faces. If not, skip the
        handler and pass it straight through to the next queue

        Parameters
        ----------
        process
            The name of the process that is checking for zero detections
        batch
            The batch of data to check for zero detections

        Returns
        -------
        ``True`` if the batch has no face detections and has been passed on. ``False`` if the batch
        contains data to be processed
        """
        if self.handler.plugin_type == "detect" or batch.frame_ids.size:
            return False
        logger.trace(  # type:ignore[attr-defined]
            "[%s.%s] Passing through batch with no detections",  self._plugin_name, process)
        self._put_data(process, batch)
        return True

    def _get_data(self, process: str) -> T.Generator[ExtractBatch, None, None]:
        """Get the next batch of data for the thread's process."""
        queue = self._queues[process]
        name = f"{self._plugin_name}_{process}"
        if list(self._queues).index(process) == 0:
            iterator: InboundIterator | InputIterator | InterimIterator = self._inbound_iterator
        else:
            iterator = InterimIterator(queue,
                                       name,
                                       self.handler.plugin_type,
                                       self.handler.batch_size,
                                       self._threads.error_state)
        for batch in iterator:
            if batch == ExtractSignal.FLUSH:  # pass flush downstream
                self._put_data(process, batch)
                continue
            assert isinstance(batch, ExtractBatch)
            if self._handle_zero_detections(process, batch):
                continue
            yield batch

    def _process_passthrough(self, batch: ExtractBatch) -> ExtractBatch:
        """When processing a passthrough batch, it is possible for the batch object to hold more
        than the plugin's batch size. In these instances, split the batch to the plugin's batch
        size and merge the results back

        Parameters
        ----------
        batch : ExtractBatch
            The passthrough batch to potentially split

        Returns
        -------
        The passthrough batch with the processed predictions
        """
        in_size = len(batch.bboxes)
        batch_size = self._handler.batch_size
        if in_size <= batch_size:
            self._handler.process(batch)
            return batch

        logger.debug("[%s.process] Splitting passthrough batch of size %s for plugin size of %s",
                     self._plugin_name, in_size, batch_size)
        retval = batch[0:batch_size]
        self._handler.process(retval)

        for start in range(batch_size, in_size, batch_size):
            feed = batch[start:start + batch_size]
            self._handler.process(feed)
            retval.append(feed)
        return retval

    def _process_batches(self, process: T.Literal["pre_process", "process", "post_process"]
                         ) -> None:
        """Obtain items from inbound queue for the process, pass to the relevant handler's
        processor and for  output to the next queue

        Parameters
        ----------
        process
            The handler's processor that will be handling the iterated batch items
        """
        if process == "process" and not self.handler.do_compile:
            # Non-compiled models launch quicker in the thread
            self.handler.init_model()
        logger.debug("[%s.%s] Starting process", self._plugin_name, process)
        processor = getattr(self.handler, process)
        for batch in self._get_data(process):
            if process == "process" and batch.passthrough:
                batch = self._process_passthrough(batch)
            else:
                processor(batch)
            self._put_data(process, batch)
        logger.debug("[%s.%s] Finished process", self._plugin_name, process)
        self._put_data(process, ExtractSignal.SHUTDOWN)

    def _get_threads(self) -> PluginThreads:
        """Obtain the threads required to each enabled plugin process.

        Returns
        -------
        The object that manages the threads for this plugin
        """
        retval = PluginThreads(self._plugin_name)
        for process in self.handler.processors:
            logger.debug("[%s] Adding thread for '%s'", self._plugin_name, process)
            retval.register_thread(name=process, target=self._process_batches)
        logger.debug("[%s] Threads: %s", self._plugin_name, retval)
        return retval

    def _get_queues(self, input_runner: ExtractRunner | None) -> dict[str, Queue]:
        """Obtain the in queue to the model and the output queues from each of this plugin's
        processes

        Parameters
        ----------
        input_runner
            The input plugin or queue that feeds this plugin. ``None`` if data is to be fed
            through the runner's `put` method.

        Returns
        -------
        The plugin inbound queue and the output queue for each of this plugin's processes in
        processing order
        """
        retval: dict[str, Queue] = {}
        in_queue = Queue(maxsize=1) if input_runner is None else input_runner.out_queue
        for idx, thread in enumerate(self._threads.enabled):
            queue = in_queue if idx == 0 else Queue(maxsize=1)
            logger.debug("[%s] Adding in queue for thread '%s'", self._plugin_name, thread)
            retval[thread] = queue
        logger.debug("[%s] Adding out queue", self._plugin_name)
        retval["out"] = Queue(maxsize=1)
        logger.debug("[%s] Queues: %s", self._plugin_name, retval)
        return retval

    def _get_inbound_iterator(self) -> InboundIterator | InputIterator:
        """Obtain the inbound iterator. If this is the first/only plugin in the pipeline, this
        will be an InputIterator that splits FrameFaces frame objects into appropriate batches
        for the plugin.

        If this is a subsequent plugin, then an InboundIterator will be returned, which takes
        already batched data from the previous plugin and re-batches for the current plugin

        Returns
        -------
        The iterator to process inbound data for the plugin
        """
        retval: InputIterator | InboundIterator
        if self._is_first:
            retval = InputIterator(list(self._queues.values())[0],
                                   f"{self._plugin_name}",
                                   self.handler.plugin_type,
                                   self.handler.batch_size,
                                   self._threads.error_state)
        else:
            retval = InboundIterator(list(self._queues.values())[0],
                                     f"{self._plugin_name}",
                                     self.handler.plugin_type,
                                     self.handler.batch_size,
                                     self._threads.error_state)
        logger.debug("[%s.in] Got inbound iterator: %s", self._plugin_name, retval)
        return retval

    def _put_to_input(self, data: FrameFaces | ExtractBatch | ExtractSignal) -> None:
        """Put data to the runner's input queue, monitoring for errors

        Parameters
        ----------
        data
            The object to put into the runner's in queue
        """
        while True:
            if self._threads.error_state.has_error:
                logger.debug("[%s] Error in worker thread", self._plugin_name)
                return
            try:
                self._queues[list(self._queues)[0]].put(data, timeout=0.2)
                break
            except QueueFull:
                logger.debug("[%s] Waiting on queue", self._plugin_name)
                continue

    def put_direct(self,  # noqa[C901]
                   filename: str,
                   image: npt.NDArray[np.uint8],
                   detected_faces: list[DetectedFace],
                   is_aligned: bool = False,
                   frame_size: tuple[int, int] | None = None) -> ExtractBatch:
        """Put an item directly into this runner's plugin and return the result

        Parameters
        ----------
        filename
            The filename of the frame
        image
            The loaded frame as UINT8 BGR array
        detected_faces
            The detected face objects for the frame
        is_aligned
            ``True`` if the image being passed into the pipeline is an aligned faceswap face.
            Default: ``False``
        frame_size
            The (height, width) size of the original frame if passing in an aligned image

        Raises
        ------
        ValueError
            If attempting to put an ExtractBatch object to the first runner in the pipeline or if
            providing an aligned image with insufficient data

        Returns
        -------
        ExtractBatch
            The output from this plugin for the given input
        """
        if isinstance(self._inbound_iterator, InputIterator):
            raise ValueError("'put_direct' should not be used on the first runner in a "
                             "pipeline. Use the runner's `put` method")
        if self.handler.plugin_type not in ("detect", "align") and not is_aligned:
            raise ValueError(f"'{self.handler.plugin_type}' requires aligned input")
        if self.handler.plugin_type in ("detect", "align") and is_aligned:
            raise ValueError(f"'{self.handler.plugin_type}' requires non-aligned input")
        if is_aligned and not frame_size:
            raise ValueError("Aligned input must provide the original frame_size")
        batch = ExtractBatch(filenames=[filename], images=[image], is_aligned=is_aligned)
        batch.bboxes = np.array([[f.left, f.top, f.right, f.bottom]
                                 for f in detected_faces], dtype=np.int32)
        batch.frame_ids = np.zeros((batch.bboxes.shape[0], ), dtype=np.int32)
        batch.frame_sizes = [frame_size] if frame_size else None
        if self.handler.plugin_type not in ("detect", "align"):
            landmarks = np.array([f.landmarks_xy for f in detected_faces], dtype=np.float32)
            batch.landmarks = landmarks
            batch.landmark_type = LandmarkType.from_shape(T.cast(tuple[int, int],
                                                                 landmarks.shape[1:]))
        original_out = self._queues["out"]  # Unhook queue from next runner
        self._queues["out"] = Queue(maxsize=1)
        self._put_to_input(batch)
        self._put_to_input(ExtractSignal.FLUSH)

        result: list[ExtractBatch] = []
        while True:
            if self._threads.error_state.has_error and current_thread() == main_thread():
                self._threads.error_state.re_raise()
            if self._threads.error_state.has_error:
                logger.debug("[%s.%s] Thread error detected in worker thread",
                             current_thread().name, self.__class__.__name__)
                break
            try:
                out = self._queues["out"].get(timeout=0.2)
            except QueueEmpty:
                continue
            if out == ExtractSignal.FLUSH:
                break
            result.append(out)

        self._queues["out"] = original_out  # Re-attach queue to next runner

        retval = result[0]
        if len(result) > 1:
            for remain in result[1:]:
                retval.append(remain)
        return retval

    @T.overload
    def put(self,
            filename: str,
            image: npt.NDArray[np.uint8],
            detected_faces: list[DetectedFace] | None = None,
            source: str | None = None,
            is_aligned: bool = False,
            frame_metadata: PNGHeaderSourceDict | None = None,
            passthrough: T.Literal[False] = False) -> None: ...

    @T.overload
    def put(self,
            filename: str,
            image: npt.NDArray[np.uint8],
            detected_faces: list[DetectedFace] | None = None,
            source: str | None = None,
            is_aligned: bool = False,
            frame_metadata: PNGHeaderSourceDict | None = None,
            *,
            passthrough: T.Literal[True]) -> FrameFaces: ...

    def put(self,
            filename: str,
            image: npt.NDArray[np.uint8],
            detected_faces: list[DetectedFace] | None = None,
            source: str | None = None,
            is_aligned: bool = False,
            frame_metadata: PNGHeaderSourceDict | None = None,
            passthrough: bool = False) -> None | FrameFaces:
        """Put a frame into the pipeline.

        Note
        ----
        When a pipeline is built using the __call__ method, this method will always put items into
        the first plugin in the pipeline

        Parameters
        ----------
        filename
            The filename of the frame
        image
            The loaded frame as UINT8 BGR array
        detected_faces
            The detected face objects for the frame. ``None`` if not any. Default: ``None``
        source
            The full path to the source folder or video file. Default: ``None`` (Not provided)
        is_aligned
            ``True`` if the image being passed into the pipeline is an aligned faceswap face.
            Default: ``False``
        frame_metadata
            If the image is aligned then the original frame metadata can be added here. Some
            plugins (eg: mask) require this to be populated for aligned inputs. Default: ``None``
        passthrough
            ``True`` if this item is meant to be passed straight through the extraction pipeline
            with no caching, for immediate return. Default: ``False``

        Returns
        -------
        If passthrough is ``True`` returns the output FrameFaces object, otherwise ``None``
        """
        item = FrameFaces(filename=filename,
                          image=image,
                          source=source,
                          is_aligned=is_aligned,
                          frame_metadata=frame_metadata,
                          passthrough=passthrough)
        if detected_faces is not None:
            item.detected_faces = detected_faces
        self._put_to_input(item)
        if passthrough:
            return next(_PLUGIN_REGISTER[self.uuid][-1])
        return None

    def put_media(self, media: FrameFaces) -> None | FrameFaces:
        """Put a frame into the pipeline that is within a FrameFaces object.

        Note
        ----
        When a pipeline is built using the __call__ method, this method will always put items into
        the first plugin in the pipeline

        Parameters
        ----------
        media
            The FrameFaces object to put into the pipeline

        Returns
        -------
        If the FrameFaces's passthrough is ``True`` returns the output FrameFaces object,
        otherwise ``None``
        """
        self._put_to_input(media)
        if media.passthrough:
            return next(_PLUGIN_REGISTER[self.uuid][-1])
        return None

    def stop(self) -> None:
        """Indicate to the runner that there is no more data to be ingested"""
        logger.debug("[%s] Putting EOF to runner", self._plugin_name)
        self._put_to_input(ExtractSignal.SHUTDOWN)
        logger.debug("[%s] Removing pipeline '%s'", self._plugin_name, self.uuid)
        del _PLUGIN_REGISTER[self.uuid]

    def flush(self) -> None:
        """Flush all data currently within the pipeline"""
        logger.debug("[%s] Putting FLUSH to runner", self._plugin_name)
        self._put_to_input(ExtractSignal.FLUSH)

    def _cascade_interfaces(self, input_runner: ExtractRunner | None) -> None:
        """On this runner's call method, cascade the public interfaces to be the input runner's
        public interfaces, such that calling them from the final plugin in the pipeline actually
        interacts with the first plugin in the pipeline.

        Similarly remove the output iterator from the input runner so that attempting to iterate a
        runner that is not the final runner in the chain results in a RuntimeError

        Parameters
        ----------
        input_runner
            The input runner to this runner or ``None`` if this is the first runner in the pipeline
        """
        if input_runner is None:
            return
        setattr(self, "put", input_runner.put)
        setattr(self, "put_media", input_runner.put_media)
        setattr(self, "stop", input_runner.stop)
        setattr(self, "flush", input_runner.flush)

        logger.debug(
            "[%s] Set pipeline interfaces to %s",
            self.__class__.__name__,
            [f"{f.__self__.__class__.__name__}.{f.__func__.__name__}"  # type:ignore[union-attr]
             for f in (self.put, self.put_media, self.stop, self.flush)]
        )

        del input_runner._output_iterator
        input_runner._output_iterator = None  # pylint:disable=protected-access
        logger.debug("[%s] Removed output iterator from %s",
                     self.__class__.__name__, input_runner.__class__.__name__)

    def _register_plugin(self, input_runner: ExtractRunner | None = None) -> None:
        """Register the plugin into the plugin tracker

        Parameters
        ----------
        input_runner
            The input plugin that feeds this plugin or ``None`` if data is to be fed through the
            runner's `put` method. Default: ``None``
        """
        name = f"{self.__class__.__name__}.{self._plugin_name}"
        if input_runner is None:
            logger.debug("[%s] Registering new pipeline: '%s'", name, self.uuid)
            _PLUGIN_REGISTER[self.uuid] = [self]
            return
        uid, chain = next((k, v) for k, v in _PLUGIN_REGISTER.items() if input_runner in v)
        logger.debug("[%s] Adding to existing pipeline: '%s'", name, uid)
        chain.insert(chain.index(input_runner) + 1, self)

    def start(self) -> None:
        """Start the threads. Callback for when the profiler has finished executing"""
        if self._threads.is_alive():
            logger.warning("Start called on runner '%s' when threads are already active. This is "
                           "almost definitely not desired", self.__class__.__name__)
            return
        if self._uuid is None:
            raise ValueError(f"Runner '{self.__class__.__name__}' must be called before starting")

        if self.handler.do_compile:
            self.handler.init_model()  # Need to compile the model in main thread
        self._threads.start()

    def __call__(self, input_runner: ExtractRunner | None, profile: bool) -> None:
        """Build and start the plugin runner

        Parameters
        ----------
        input_runner
            The input plugin that feeds this plugin or ``None`` if data is to be fed through the
            runner's `put` method.
        profile
            ``True`` if the runner is to be profiled, indicating that threads will not be started

        Raises
        ------
        ValueError
            If the input runner has not been called and assigned a UUID or if this runner has
            already been called
        """
        if input_runner is not None and input_runner._uuid is None:
            raise ValueError(f"Input runner '{input_runner.__class__.__name__}' must be called "
                             f"prior to adding to '{self.__class__.__name__}'")
        if self._uuid is not None:
            raise ValueError(f"Runner '{self.__class__.__name__}' has already been called")
        self._uuid = uuid4().hex

        self._is_first = input_runner is None
        self._queues = self._get_queues(input_runner)

        self._inbound_iterator = self._get_inbound_iterator()
        self._output_iterator = OutputIterator(self._queues["out"],
                                               f"{self._plugin_name}_out",
                                               self.handler.plugin_type,
                                               self.handler.batch_size,
                                               self._threads.error_state)
        self._cascade_interfaces(input_runner)
        self._register_plugin(input_runner)
        if not profile:
            self.start()

    def register_external_error_state(self, state: ErrorState) -> None:
        """Register an external error state object.

        If we are not running any threads (specifically, file handler), the pipeline can hang the
        calling thread. The error state from the calling thread can be populated here. This can
        only be called if no threads have been registered for the runner

        Parameters
        ----------
        state
            The ErrorState object to register
        """
        self._threads.register_external_error_state(state)


def get_pipeline(runner: ExtractRunner) -> list[ExtractRunner]:
    """Obtain a list of runners in order of input to output of the extraction chain that the given
    runner belongs to

    Parameters
    ----------
    runner
        The initialized runner to obtain the full chain for

    Returns
    -------
    The ordered list of runners if the inference chain that the given runner belongs to
    """
    retval = next(v for v in _PLUGIN_REGISTER.values() if runner in v)
    logger.debug("Obtained plugin chain for runner '%s': %s", runner, retval)
    return retval


__all__ = get_module_objects(__name__)
