#!/usr/bin python3
""" Main entry point to the convert process of FaceSwap """
from __future__ import annotations
from dataclasses import dataclass, field
import logging
import re
import os
import sys
import typing as T
from threading import Event
from time import sleep

import cv2
import numpy as np
from tqdm import tqdm

from scripts.fsmedia import Alignments, PostProcess, finalize
from lib.serializer import get_serializer
from lib.convert import Converter
from lib.align import AlignedFace, DetectedFace, update_legacy_png_header
from lib.gpu_stats import GPUStats
from lib.image import read_image_meta_batch, ImagesLoader
from lib.multithreading import MultiThread, total_cpus
from lib.queue_manager import queue_manager
from lib.utils import FaceswapError, get_folder, get_image_paths, handle_deprecated_cliopts
from plugins.extract import ExtractMedia, Extractor
from plugins.plugin_loader import PluginLoader

if T.TYPE_CHECKING:
    from argparse import Namespace
    from collections.abc import Callable
    from plugins.convert.writer._base import Output
    from plugins.train.model._base import ModelBase
    from lib.align.aligned_face import CenteringType
    from lib.queue_manager import EventQueue


logger = logging.getLogger(__name__)


@dataclass
class ConvertItem:
    """ A single frame with associated objects passing through the convert process.

    Parameters
    ----------
    input: :class:`~plugins.extract.extract_media.ExtractMedia`
        The ExtractMedia object holding the :attr:`filename`, :attr:`image` and attr:`list` of
        :class:`~lib.align.DetectedFace` objects loaded from disk
    feed_faces: list, Optional
        list of :class:`lib.align.AlignedFace` objects for feeding into the model's predict
        function
    reference_faces: list, Optional
        list of :class:`lib.align.AlignedFace` objects at model output sized for using as reference
        in the convert functionfor feeding into the model's predict
    swapped_faces: :class:`np.ndarray`
        The swapped faces returned from the model's predict function
    """
    inbound: ExtractMedia
    feed_faces: list[AlignedFace] = field(default_factory=list)
    reference_faces: list[AlignedFace] = field(default_factory=list)
    swapped_faces: np.ndarray = np.array([])


class Convert():
    """ The Faceswap Face Conversion Process.

    The conversion process is responsible for swapping the faces on source frames with the output
    from a trained model.

    It leverages a series of user selected post-processing plugins, executed from
    :class:`lib.convert.Converter`.

    The convert process is self contained and should not be referenced by any other scripts, so it
    contains no public properties.

    Parameters
    ----------
    arguments: :class:`argparse.Namespace`
        The arguments to be passed to the convert process as generated from Faceswap's command
        line arguments
    """
    def __init__(self, arguments: Namespace) -> None:
        logger.debug("Initializing %s: (args: %s)", self.__class__.__name__, arguments)
        self._args = handle_deprecated_cliopts(arguments)

        self._images = ImagesLoader(self._args.input_dir, fast_count=True)
        self._alignments = self._get_alignments()
        self._opts = OptionalActions(self._args, self._images.file_list, self._alignments)

        self._add_queues()
        self._predictor = Predict(self._queue_size, arguments)
        self._disk_io = DiskIO(self._alignments, self._images, self._predictor, arguments)
        self._predictor.launch(self._disk_io.load_queue)
        self._validate()
        get_folder(self._args.output_dir)

        configfile = self._args.configfile if hasattr(self._args, "configfile") else None
        self._converter = Converter(self._predictor.output_size,
                                    self._predictor.coverage_ratio,
                                    self._predictor.centering,
                                    self._disk_io.draw_transparent,
                                    self._disk_io.pre_encode,
                                    arguments,
                                    configfile=configfile)
        self._patch_threads = self._get_threads()
        logger.debug("Initialized %s", self.__class__.__name__)

    @property
    def _queue_size(self) -> int:
        """ int: Size of the converter queues. 2 for single process otherwise 4 """
        retval = 2 if self._args.singleprocess or self._args.jobs == 1 else 4
        logger.debug(retval)
        return retval

    @property
    def _pool_processes(self) -> int:
        """ int: The number of threads to run in parallel. Based on user options and number of
        available processors. """
        if self._args.singleprocess:
            retval = 1
        elif self._args.jobs > 0:
            retval = min(self._args.jobs, total_cpus(), self._images.count)
        else:
            retval = min(total_cpus(), self._images.count)
        retval = 1 if retval == 0 else retval
        logger.debug(retval)
        return retval

    def _get_alignments(self) -> Alignments:
        """ Perform validation checks and legacy updates and return alignemnts object

        Returns
        -------
        :class:`~lib.align.alignments.Alignments`
            The alignments file for the extract job
        """
        retval = Alignments(self._args, False, self._images.is_video)
        if retval.version == 1.0:
            logger.error("The alignments file format has been updated since the given alignments "
                         "file was generated. You need to update the file to proceed.")
            logger.error("To do this run the 'Alignments Tool' > 'Extract' Job.")
            sys.exit(1)

        retval.update_legacy_has_source(os.path.basename(self._args.input_dir))
        return retval

    def _validate(self) -> None:
        """ Validate the Command Line Options.

        Ensure that certain cli selections are valid and won't result in an error. Checks:
            * If frames have been passed in with video output, ensure user supplies reference
            video.
            * If "on-the-fly" and a Neural Network mask is selected, warn and switch to 'extended'
            * If a mask-type is selected, ensure it exists in the alignments file.
            * If a predicted mask-type is selected, ensure model has been trained with a mask
            otherwise attempt to select first available masks, otherwise raise error.

        Raises
        ------
        FaceswapError
            If an invalid selection has been found.

        """
        if (self._args.writer == "ffmpeg" and
                not self._images.is_video and
                self._args.reference_video is None):
            raise FaceswapError("Output as video selected, but using frames as input. You must "
                                "provide a reference video ('-ref', '--reference-video').")

        if (self._args.on_the_fly and
                self._args.mask_type not in ("none", "extended", "components")):
            logger.warning("You have selected an incompatible mask type ('%s') for On-The-Fly "
                           "conversion. Switching to 'extended'", self._args.mask_type)
            self._args.mask_type = "extended"

        if (not self._args.on_the_fly and
                self._args.mask_type not in ("none", "predicted") and
                not self._alignments.mask_is_valid(self._args.mask_type)):
            msg = (f"You have selected the Mask Type `{self._args.mask_type}` but at least one "
                   "face does not have this mask stored in the Alignments File.\nYou should "
                   "generate the required masks with the Mask Tool or set the Mask Type option to "
                   "an existing Mask Type.\nA summary of existing masks is as follows:\nTotal "
                   f"faces: {self._alignments.faces_count}, "
                   f"Masks: {self._alignments.mask_summary}")
            raise FaceswapError(msg)

        if self._args.mask_type == "predicted" and not self._predictor.has_predicted_mask:
            available_masks = [k for k, v in self._alignments.mask_summary.items()
                               if k != "none" and v == self._alignments.faces_count]
            if not available_masks:
                msg = ("Predicted Mask selected, but the model was not trained with a mask and no "
                       "masks are stored in the Alignments File.\nYou should generate the "
                       "required masks with the Mask Tool or set the Mask Type to `none`.")
                raise FaceswapError(msg)
            mask_type = available_masks[0]
            logger.warning("Predicted Mask selected, but the model was not trained with a "
                           "mask. Selecting first available mask: '%s'", mask_type)
            self._args.mask_type = mask_type

    def _add_queues(self) -> None:
        """ Add the queues for in, patch and out. """
        logger.debug("Adding queues. Queue size: %s", self._queue_size)
        for qname in ("convert_in", "convert_out", "patch"):
            queue_manager.add_queue(qname, self._queue_size)

    def _get_threads(self) -> MultiThread:
        """ Get the threads for patching the converted faces onto the frames.

        Returns
        :class:`lib.multithreading.MultiThread`
            The threads that perform the patching of swapped faces onto the output frames
        """
        save_queue = queue_manager.get_queue("convert_out")
        patch_queue = queue_manager.get_queue("patch")
        return MultiThread(self._converter.process, patch_queue, save_queue,
                           thread_count=self._pool_processes, name="patch")

    def process(self) -> None:
        """ The entry point for triggering the Conversion Process.

        Should only be called from  :class:`lib.cli.launcher.ScriptExecutor`

        Raises
        ------
        FaceswapError
            Error raised if the process runs out of memory
        """
        logger.debug("Starting Conversion")
        # queue_manager.debug_monitor(5)
        try:
            self._convert_images()
            self._disk_io.save_thread.join()
            queue_manager.terminate_queues()

            finalize(self._images.count,
                     self._predictor.faces_count,
                     self._predictor.verify_output)
            logger.debug("Completed Conversion")
        except MemoryError as err:
            msg = ("Faceswap ran out of RAM running convert. Conversion is very system RAM "
                   "heavy, so this can happen in certain circumstances when you have a lot of "
                   "cpus but not enough RAM to support them all."
                   "\nYou should lower the number of processes in use by either setting the "
                   "'singleprocess' flag (-sp) or lowering the number of parallel jobs (-j).")
            raise FaceswapError(msg) from err

    def _convert_images(self) -> None:
        """ Start the multi-threaded patching process, monitor all threads for errors and join on
        completion. """
        logger.debug("Converting images")
        self._patch_threads.start()
        while True:
            self._check_thread_error()
            if self._disk_io.completion_event.is_set():
                logger.debug("DiskIO completion event set. Joining Pool")
                break
            if self._patch_threads.completed():
                logger.debug("All patch threads completed")
                break
            sleep(1)
        self._patch_threads.join()

        logger.debug("Putting EOF")
        queue_manager.get_queue("convert_out").put("EOF")
        logger.debug("Converted images")

    def _check_thread_error(self) -> None:
        """ Monitor all running threads for errors, and raise accordingly.

        Raises
        ------
        Error
            Re-raises any error encountered within any of the running threads
        """
        for thread in (self._predictor.thread,
                       self._disk_io.load_thread,
                       self._disk_io.save_thread,
                       self._patch_threads):
            thread.check_and_raise_error()


class DiskIO():
    """ Disk Input/Output for the converter process.

    Background threads to:
        * Load images from disk and get the detected faces
        * Save images back to disk

    Parameters
    ----------
    alignments: :class:`lib.alignmnents.Alignments`
        The alignments for the input video
    images: :class:`lib.image.ImagesLoader`
        The input images
    predictor: :class:`Predict`
        The object for generating predictions from the model
    arguments: :class:`argparse.Namespace`
        The arguments that were passed to the convert process as generated from Faceswap's command
        line arguments
    """

    def __init__(self,
                 alignments: Alignments,
                 images: ImagesLoader,
                 predictor: Predict,
                 arguments: Namespace) -> None:
        logger.debug("Initializing %s: (alignments: %s, images: %s, predictor: %s, arguments: %s)",
                     self.__class__.__name__, alignments, images, predictor, arguments)
        self._alignments = alignments
        self._images = images
        self._args = arguments
        self._pre_process = PostProcess(arguments)
        self._completion_event = Event()

        # For frame skipping
        self._imageidxre = re.compile(r"(\d+)(?!.*\d\.)(?=\.\w+$)")
        self._frame_ranges = self._get_frame_ranges()
        self._writer = self._get_writer(predictor)

        # Extractor for on the fly detection
        self._extractor = self._load_extractor()

        self._queues: dict[T.Literal["load", "save"], EventQueue] = {}
        self._threads: dict[T.Literal["load", "save"], MultiThread] = {}
        self._init_threads()
        logger.debug("Initialized %s", self.__class__.__name__)

    @property
    def completion_event(self) -> Event:
        """ :class:`event.Event`: Event is set when the DiskIO Save task is complete """
        return self._completion_event

    @property
    def draw_transparent(self) -> bool:
        """ bool: ``True`` if the selected writer's Draw_transparent configuration item is set
        otherwise ``False`` """
        return self._writer.config.get("draw_transparent", False)

    @property
    def pre_encode(self) -> Callable[[np.ndarray, T.Any], list[bytes]] | None:
        """ python function: Selected writer's pre-encode function, if it has one,
        otherwise ``None`` """
        dummy = np.zeros((20, 20, 3), dtype="uint8")
        test = self._writer.pre_encode(dummy)
        retval: Callable | None = None if test is None else self._writer.pre_encode
        logger.debug("Writer pre_encode function: %s", retval)
        return retval

    @property
    def save_thread(self) -> MultiThread:
        """ :class:`lib.multithreading.MultiThread`: The thread that is running the image writing
        operation. """
        return self._threads["save"]

    @property
    def load_thread(self) -> MultiThread:
        """ :class:`lib.multithreading.MultiThread`: The thread that is running the image loading
        operation. """
        return self._threads["load"]

    @property
    def load_queue(self) -> EventQueue:
        """ :class:`~lib.queue_manager.EventQueue`: The queue that images and detected faces are "
        "loaded into. """
        return self._queues["load"]

    @property
    def _total_count(self) -> int:
        """ int: The total number of frames to be converted """
        if self._frame_ranges and not self._args.keep_unchanged:
            retval = sum(fr[1] - fr[0] + 1 for fr in self._frame_ranges)
        else:
            retval = self._images.count
        logger.debug(retval)
        return retval

    # Initialization
    def _get_writer(self, predictor: Predict) -> Output:
        """ Load the selected writer plugin.

        Parameters
        ----------
        predictor: :class:`Predict`
            The object for generating predictions from the model

        Returns
        -------
        :mod:`plugins.convert.writer` plugin
            The requested writer plugin
        """
        args = [self._args.output_dir]
        if self._args.writer in ("ffmpeg", "gif"):
            args.extend([self._total_count, self._frame_ranges])
        if self._args.writer == "ffmpeg":
            if self._images.is_video:
                args.append(self._args.input_dir)
            else:
                args.append(self._args.reference_video)
        if self._args.writer == "patch":
            args.append(predictor.output_size)
        logger.debug("Writer args: %s", args)
        configfile = self._args.configfile if hasattr(self._args, "configfile") else None
        return PluginLoader.get_converter("writer", self._args.writer)(*args,
                                                                       configfile=configfile)

    def _get_frame_ranges(self) -> list[tuple[int, int]] | None:
        """ Obtain the frame ranges that are to be converted.

        If frame ranges have been specified, then split the command line formatted arguments into
        ranges that can be used.

        Returns
        list or ``None``
            A list of  frames to be processed, or ``None`` if the command line argument was not
            used
        """
        if not self._args.frame_ranges:
            logger.debug("No frame range set")
            return None

        minframe, maxframe = None, None
        if self._images.is_video:
            minframe, maxframe = 1, self._images.count
        else:
            indices = [int(self._imageidxre.findall(os.path.basename(filename))[0])
                       for filename in self._images.file_list]
            if indices:
                minframe, maxframe = min(indices), max(indices)
        logger.debug("minframe: %s, maxframe: %s", minframe, maxframe)

        if minframe is None or maxframe is None:
            raise FaceswapError("Frame Ranges specified, but could not determine frame numbering "
                                "from filenames")

        retval = []
        for rng in self._args.frame_ranges:
            if "-" not in rng:
                raise FaceswapError("Frame Ranges not specified in the correct format")
            start, end = rng.split("-")
            retval.append((max(int(start), minframe), min(int(end), maxframe)))
        logger.debug("frame ranges: %s", retval)
        return retval

    def _load_extractor(self) -> Extractor | None:
        """ Load the CV2-DNN Face Extractor Chain.

        For On-The-Fly conversion we use a CPU based extractor to avoid stacking the GPU.
        Results are poor.

        Returns
        -------
        :class:`plugins.extract.Pipeline.Extractor`
            The face extraction chain to be used for on-the-fly conversion
        """
        if not self._alignments.have_alignments_file and not self._args.on_the_fly:
            logger.error("No alignments file found. Please provide an alignments file for your "
                         "destination video (recommended) or enable on-the-fly conversion (not "
                         "recommended).")
            sys.exit(1)
        if self._alignments.have_alignments_file:
            if self._args.on_the_fly:
                logger.info("On-The-Fly conversion selected, but an alignments file was found. "
                            "Using pre-existing alignments file: '%s'", self._alignments.file)
            else:
                logger.debug("Alignments file found: '%s'", self._alignments.file)
            return None

        logger.debug("Loading extractor")
        logger.warning("On-The-Fly conversion selected. This will use the inferior cv2-dnn for "
                       "extraction and will produce poor results.")
        logger.warning("It is recommended to generate an alignments file for your destination "
                       "video with Extract first for superior results.")
        extractor = Extractor(detector="cv2-dnn",
                              aligner="cv2-dnn",
                              masker=self._args.mask_type,
                              multiprocess=True,
                              rotate_images=None,
                              min_size=20)
        extractor.launch()
        logger.debug("Loaded extractor")
        return extractor

    def _init_threads(self) -> None:
        """ Initialize queues and threads.

        Creates the load and save queues and the load and save threads. Starts the threads.
        """
        logger.debug("Initializing DiskIO Threads")
        for task in T.get_args(T.Literal["load", "save"]):
            self._add_queue(task)
            self._start_thread(task)
        logger.debug("Initialized DiskIO Threads")

    def _add_queue(self, task: T.Literal["load", "save"]) -> None:
        """ Add the queue to queue_manager and to :attr:`self._queues` for the given task.

        Parameters
        ----------
        task: {"load", "save"}
            The task that the queue is to be added for
        """
        logger.debug("Adding queue for task: '%s'", task)
        if task == "load":
            q_name = "convert_in"
        elif task == "save":
            q_name = "convert_out"
        else:
            q_name = task
        self._queues[task] = queue_manager.get_queue(q_name)
        logger.debug("Added queue for task: '%s'", task)

    def _start_thread(self, task: T.Literal["load", "save"]) -> None:
        """ Create the thread for the given task, add it it :attr:`self._threads` and start it.

        Parameters
        ----------
        task: {"load", "save"}
            The task that the thread is to be created for
        """
        logger.debug("Starting thread: '%s'", task)
        args = self._completion_event if task == "save" else None
        func = getattr(self, f"_{task}")
        io_thread = MultiThread(func, args, thread_count=1)
        io_thread.start()
        self._threads[task] = io_thread
        logger.debug("Started thread: '%s'", task)

    # Loading tasks
    def _load(self, *args) -> None:  # pylint:disable=unused-argument
        """ Load frames from disk.

        In a background thread:
            * Loads frames from disk.
            * Discards or passes through cli selected skipped frames
            * Pairs the frame with its :class:`~lib.align.DetectedFace` objects
            * Performs any pre-processing actions
            * Puts the frame and detected faces to the load queue
        """
        logger.debug("Load Images: Start")
        idx = 0
        for filename, image in self._images.load():
            idx += 1
            if self._queues["load"].shutdown.is_set():
                logger.debug("Load Queue: Stop signal received. Terminating")
                break
            if image is None or (not image.any() and image.ndim not in (2, 3)):
                # All black frames will return not numpy.any() so check dims too
                logger.warning("Unable to open image. Skipping: '%s'", filename)
                continue
            if self._check_skipframe(filename):
                if self._args.keep_unchanged:
                    logger.trace("Saving unchanged frame: %s", filename)  # type:ignore
                    out_file = os.path.join(self._args.output_dir, os.path.basename(filename))
                    self._queues["save"].put((out_file, image))
                else:
                    logger.trace("Discarding frame: '%s'", filename)  # type:ignore
                continue

            detected_faces = self._get_detected_faces(filename, image)
            item = ConvertItem(ExtractMedia(filename, image, detected_faces))
            self._pre_process.do_actions(item.inbound)
            self._queues["load"].put(item)

        logger.debug("Putting EOF")
        self._queues["load"].put("EOF")
        logger.debug("Load Images: Complete")

    def _check_skipframe(self, filename: str) -> bool:
        """ Check whether a frame is to be skipped.

        Parameters
        ----------
        filename: str
            The filename of the frame to check

        Returns
        -------
        bool
            ``True`` if the frame is to be skipped otherwise ``False``
        """
        if not self._frame_ranges:
            return False
        indices = self._imageidxre.findall(filename)
        if not indices:
            logger.warning("Could not determine frame number. Frame will be converted: '%s'",
                           filename)
            return False
        idx = int(indices[0])
        skipframe = not any(map(lambda b: b[0] <= idx <= b[1], self._frame_ranges))
        logger.trace("idx: %s, skipframe: %s", idx, skipframe)  # type: ignore[attr-defined]
        return skipframe

    def _get_detected_faces(self, filename: str, image: np.ndarray) -> list[DetectedFace]:
        """ Return the detected faces for the given image.

        If we have an alignments file, then the detected faces are created from that file. If
        we're running On-The-Fly then they will be extracted from the extractor.

        Parameters
        ----------
        filename: str
            The filename to return the detected faces for
        image: :class:`numpy.ndarray`
            The frame that the detected faces exist in

        Returns
        -------
        list
            List of :class:`lib.align.DetectedFace` objects
        """
        logger.trace("Getting faces for: '%s'", filename)  # type:ignore
        if not self._extractor:
            detected_faces = self._alignments_faces(os.path.basename(filename), image)
        else:
            detected_faces = self._detect_faces(filename, image)
        logger.trace("Got %s faces for: '%s'", len(detected_faces), filename)  # type:ignore
        return detected_faces

    def _alignments_faces(self, frame_name: str, image: np.ndarray) -> list[DetectedFace]:
        """ Return detected faces from an alignments file.

        Parameters
        ----------
        frame_name: str
            The name of the frame to return the detected faces for
        image: :class:`numpy.ndarray`
            The frame that the detected faces exist in

        Returns
        -------
        list
            List of :class:`lib.align.DetectedFace` objects
        """
        if not self._check_alignments(frame_name):
            return []

        faces = self._alignments.get_faces_in_frame(frame_name)
        detected_faces = []

        for rawface in faces:
            face = DetectedFace()
            face.from_alignment(rawface, image=image)
            detected_faces.append(face)
        return detected_faces

    def _check_alignments(self, frame_name: str) -> bool:
        """ Ensure that we have alignments for the current frame.

        If we have no alignments for this image, skip it and output a message.

        Parameters
        ----------
        frame_name: str
            The name of the frame to check that we have alignments for

        Returns
        -------
        bool
            ``True`` if we have alignments for this face, otherwise ``False``
        """
        have_alignments = self._alignments.frame_exists(frame_name)
        if not have_alignments:
            tqdm.write(f"No alignment found for {frame_name}, skipping")
        return have_alignments

    def _detect_faces(self, filename: str, image: np.ndarray) -> list[DetectedFace]:
        """ Extract the face from a frame for On-The-Fly conversion.

        Pulls detected faces out of the Extraction pipeline.

        Parameters
        ----------
        filename: str
            The filename to return the detected faces for
        image: :class:`numpy.ndarray`
            The frame that the detected faces exist in

        Returns
        -------
        list
            List of :class:`lib.align.DetectedFace` objects
         """
        assert self._extractor is not None
        self._extractor.input_queue.put(ExtractMedia(filename, image))
        faces = next(self._extractor.detected_faces())
        return faces.detected_faces

    # Saving tasks
    def _save(self, completion_event: Event) -> None:
        """ Save the converted images.

        Puts the selected writer into a background thread and feeds it from the output of the
        patch queue.

        Parameters
        ----------
        completion_event: :class:`event.Event`
            An even that this process triggers when it has finished saving
        """
        logger.debug("Save Images: Start")
        write_preview = self._args.redirect_gui and self._writer.is_stream
        preview_image = os.path.join(self._writer.output_folder, ".gui_preview.jpg")
        logger.debug("Write preview for gui: %s", write_preview)
        for idx in tqdm(range(self._total_count), desc="Converting", file=sys.stdout):
            if self._queues["save"].shutdown.is_set():
                logger.debug("Save Queue: Stop signal received. Terminating")
                break
            item: tuple[str, np.ndarray | bytes] | T.Literal["EOF"] = self._queues["save"].get()
            if item == "EOF":
                logger.debug("EOF Received")
                break
            filename, image = item
            # Write out preview image for the GUI every 10 frames if writing to stream
            if write_preview and idx % 10 == 0 and not os.path.exists(preview_image):
                logger.debug("Writing GUI Preview image: '%s'", preview_image)
                assert isinstance(image, np.ndarray)
                cv2.imwrite(preview_image, image)
            self._writer.write(filename, image)
        self._writer.close()
        completion_event.set()
        logger.debug("Save Faces: Complete")


class Predict():
    """ Obtains the output from the Faceswap model.

    Parameters
    ----------
    queue_size: int
        The maximum size of the input queue
    arguments: :class:`argparse.Namespace`
        The arguments that were passed to the convert process as generated from Faceswap's command
        line arguments
    """
    def __init__(self, queue_size: int, arguments: Namespace) -> None:
        logger.debug("Initializing %s: (args: %s, queue_size: %s)",
                     self.__class__.__name__, arguments, queue_size)
        self._args = arguments
        self._in_queue: EventQueue | None = None
        self._out_queue = queue_manager.get_queue("patch")
        self._serializer = get_serializer("json")
        self._faces_count = 0
        self._verify_output = False

        self._model = self._load_model()
        self._batchsize = self._get_batchsize(queue_size)
        self._sizes = self._get_io_sizes()
        self._coverage_ratio = self._model.coverage_ratio
        self._centering = self._model.config["centering"]

        self._thread: MultiThread | None = None
        logger.debug("Initialized %s: (out_queue: %s)", self.__class__.__name__, self._out_queue)

    @property
    def thread(self) -> MultiThread:
        """ :class:`~lib.multithreading.MultiThread`: The thread that is running the prediction
        function from the Faceswap model. """
        assert self._thread is not None
        return self._thread

    @property
    def in_queue(self) -> EventQueue:
        """ :class:`~lib.queue_manager.EventQueue`: The input queue to the predictor. """
        assert self._in_queue is not None
        return self._in_queue

    @property
    def out_queue(self) -> EventQueue:
        """ :class:`~lib.queue_manager.EventQueue`: The output queue from the predictor. """
        return self._out_queue

    @property
    def faces_count(self) -> int:
        """ int: The total number of faces seen by the Predictor. """
        return self._faces_count

    @property
    def verify_output(self) -> bool:
        """ bool: ``True`` if multiple faces have been found in frames, otherwise ``False``. """
        return self._verify_output

    @property
    def coverage_ratio(self) -> float:
        """ float: The coverage ratio that the model was trained at. """
        return self._coverage_ratio

    @property
    def centering(self) -> CenteringType:
        """ str: The centering that the model was trained on (`"head", "face"` or `"legacy"`) """
        return self._centering

    @property
    def has_predicted_mask(self) -> bool:
        """ bool: ``True`` if the model was trained to learn a mask, otherwise ``False``. """
        return bool(self._model.config.get("learn_mask", False))

    @property
    def output_size(self) -> int:
        """ int: The size in pixels of the Faceswap model output. """
        return self._sizes["output"]

    def _get_io_sizes(self) -> dict[str, int]:
        """ Obtain the input size and output size of the model.

        Returns
        -------
        dict
            input_size in pixels and output_size in pixels
        """
        input_shape = self._model.model.input_shape
        input_shape = [input_shape] if not isinstance(input_shape, list) else input_shape
        output_shape = self._model.model.output_shape
        output_shape = [output_shape] if not isinstance(output_shape, list) else output_shape
        retval = {"input": input_shape[0][1], "output": output_shape[-1][1]}
        logger.debug(retval)
        return retval

    def _load_model(self) -> ModelBase:
        """ Load the Faceswap model.

        Returns
        -------
        :mod:`plugins.train.model` plugin
            The trained model in the specified model folder
        """
        logger.debug("Loading Model")
        model_dir = get_folder(self._args.model_dir, make_folder=False)
        if not model_dir:
            raise FaceswapError(f"{self._args.model_dir} does not exist.")
        trainer = self._get_model_name(model_dir)
        model = PluginLoader.get_model(trainer)(model_dir, self._args, predict=True)
        model.build()
        logger.debug("Loaded Model")
        return model

    def _get_batchsize(self, queue_size: int) -> int:
        """ Get the batch size for feeding the model.

        Sets the batch size to 1 if inference is being run on CPU, otherwise the minimum of the
        input queue size and the model's `convert_batchsize` configuration option.

        Parameters
        ----------
        queue_size: int
            The queue size that is feeding the predictor

        Returns
        -------
        int
            The batch size that the model is to be fed at.
        """
        logger.debug("Getting batchsize")
        is_cpu = GPUStats().device_count == 0
        batchsize = 1 if is_cpu else self._model.config["convert_batchsize"]
        batchsize = min(queue_size, batchsize)
        logger.debug("Got batchsize: %s", batchsize)
        return batchsize

    def _get_model_name(self, model_dir: str) -> str:
        """ Return the name of the Faceswap model used.

        Retrieve the name of the model from the model's state file.

        Parameters
        ----------
        model_dir: str
            The folder that contains the trained Faceswap model

        Returns
        -------
        str
            The name of the Faceswap model being used.

        """
        statefiles = [fname for fname in os.listdir(str(model_dir))
                      if fname.endswith("_state.json")]
        if len(statefiles) != 1:
            raise FaceswapError("There should be 1 state file in your model folder. "
                                f"{len(statefiles)} were found.")
        statefile = os.path.join(str(model_dir), statefiles[0])

        state = self._serializer.load(statefile)
        trainer = state.get("name", None)

        if not trainer:
            raise FaceswapError("Trainer name could not be read from state file.")
        logger.debug("Trainer from state file: '%s'", trainer)
        return trainer

    def launch(self, load_queue: EventQueue) -> None:
        """ Launch the prediction process in a background thread.

        Starts the prediction thread and returns the thread.

        Parameters
        ----------
        load_queue: :class:`~lib.queue_manager.EventQueue`
            The queue that contains images and detected faces for feeding the model
        """
        self._in_queue = load_queue
        self._thread = MultiThread(self._predict_faces, thread_count=1)
        self._thread.start()

    def _predict_faces(self) -> None:
        """ Run Prediction on the Faceswap model in a background thread.

        Reads from the :attr:`self._in_queue`, prepares images for prediction
        then puts the predictions back to the :attr:`self.out_queue`
        """
        faces_seen = 0
        consecutive_no_faces = 0
        batch: list[ConvertItem] = []
        assert self._in_queue is not None
        while True:
            item: T.Literal["EOF"] | ConvertItem = self._in_queue.get()
            if item == "EOF":
                logger.debug("EOF Received")
                if batch:  # Process out any remaining items
                    self._process_batch(batch, faces_seen)
                break
            logger.trace("Got from queue: '%s'", item.inbound.filename)  # type:ignore
            faces_count = len(item.inbound.detected_faces)

            # Safety measure. If a large stream of frames appear that do not have faces,
            # these will stack up into RAM. Keep a count of consecutive frames with no faces.
            # If self._batchsize number of frames appear, force the current batch through
            # to clear RAM.
            consecutive_no_faces = consecutive_no_faces + 1 if faces_count == 0 else 0
            self._faces_count += faces_count
            if faces_count > 1:
                self._verify_output = True
                logger.verbose("Found more than one face in an image! '%s'",  # type:ignore
                               os.path.basename(item.inbound.filename))

            self.load_aligned(item)
            faces_seen += faces_count

            batch.append(item)

            if faces_seen < self._batchsize and consecutive_no_faces < self._batchsize:
                logger.trace("Continuing. Current batchsize: %s, "  # type:ignore
                             "consecutive_no_faces: %s", faces_seen, consecutive_no_faces)
                continue

            self._process_batch(batch, faces_seen)

            consecutive_no_faces = 0
            faces_seen = 0
            batch = []

        logger.debug("Putting EOF")
        self._out_queue.put("EOF")
        logger.debug("Load queue complete")

    def _process_batch(self, batch: list[ConvertItem], faces_seen: int):
        """ Predict faces on the given batch of images and queue out to patch thread

        Parameters
        ----------
        batch: list
            List of :class:`ConvertItem` objects for the current batch
        faces_seen: int
            The number of faces seen in the current batch

        Returns
        -------
        :class:`np.narray`
            The predicted faces for the current batch
        """
        logger.trace("Batching to predictor. Frames: %s, Faces: %s",  # type:ignore
                     len(batch), faces_seen)
        feed_batch = [feed_face for item in batch for feed_face in item.feed_faces]
        if faces_seen != 0:
            feed_faces = self._compile_feed_faces(feed_batch)
            batch_size = None
            predicted = self._predict(feed_faces, batch_size)
        else:
            predicted = np.array([])

        self._queue_out_frames(batch, predicted)

    def load_aligned(self, item: ConvertItem) -> None:
        """ Load the model's feed faces and the reference output faces.

        For each detected face in the incoming item, load the feed face and reference face
        images, correctly sized for input and output respectively.

        Parameters
        ----------
        item: :class:`ConvertMedia`
            The convert media object, containing the ExctractMedia for the current image
        """
        logger.trace("Loading aligned faces: '%s'", item.inbound.filename)  # type:ignore
        feed_faces = []
        reference_faces = []
        for detected_face in item.inbound.detected_faces:
            feed_face = AlignedFace(detected_face.landmarks_xy,
                                    image=item.inbound.image,
                                    centering=self._centering,
                                    size=self._sizes["input"],
                                    coverage_ratio=self._coverage_ratio,
                                    dtype="float32")
            if self._sizes["input"] == self._sizes["output"]:
                reference_faces.append(feed_face)
            else:
                reference_faces.append(AlignedFace(detected_face.landmarks_xy,
                                                   image=item.inbound.image,
                                                   centering=self._centering,
                                                   size=self._sizes["output"],
                                                   coverage_ratio=self._coverage_ratio,
                                                   dtype="float32"))
            feed_faces.append(feed_face)
        item.feed_faces = feed_faces
        item.reference_faces = reference_faces
        logger.trace("Loaded aligned faces: '%s'", item.inbound.filename)  # type:ignore

    @staticmethod
    def _compile_feed_faces(feed_faces: list[AlignedFace]) -> np.ndarray:
        """ Compile a batch of faces for feeding into the Predictor.

        Parameters
        ----------
        feed_faces: list
            List of :class:`~lib.align.AlignedFace` objects sized for feeding into the model

        Returns
        -------
        :class:`numpy.ndarray`
            A batch of faces ready for feeding into the Faceswap model.
        """
        logger.trace("Compiling feed face. Batchsize: %s", len(feed_faces))  # type:ignore
        retval = np.stack([T.cast(np.ndarray, feed_face.face)[..., :3]
                           for feed_face in feed_faces]) / 255.0
        logger.trace("Compiled Feed faces. Shape: %s", retval.shape)  # type:ignore
        return retval

    def _predict(self, feed_faces: np.ndarray, batch_size: int | None = None) -> np.ndarray:
        """ Run the Faceswap models' prediction function.

        Parameters
        ----------
        feed_faces: :class:`numpy.ndarray`
            The batch to be fed into the model
        batch_size: int, optional
            Used for plaidml only. Indicates to the model what batch size is being processed.
            Default: ``None``

        Returns
        -------
        :class:`numpy.ndarray`
            The swapped faces for the given batch
        """
        logger.trace("Predicting: Batchsize: %s", len(feed_faces))  # type:ignore

        if self._model.color_order.lower() == "rgb":
            feed_faces = feed_faces[..., ::-1]

        feed = [feed_faces]
        logger.trace("Input shape(s): %s", [item.shape for item in feed])  # type:ignore

        inbound = self._model.model.predict(feed, verbose=0, batch_size=batch_size)
        predicted: list[np.ndarray] = inbound if isinstance(inbound, list) else [inbound]

        if self._model.color_order.lower() == "rgb":
            predicted[0] = predicted[0][..., ::-1]

        logger.trace("Output shape(s): %s",  # type:ignore
                     [predict.shape for predict in predicted])

        # Only take last output(s)
        if predicted[-1].shape[-1] == 1:  # Merge mask to alpha channel
            retval = np.concatenate(predicted[-2:], axis=-1).astype("float32")
        else:
            retval = predicted[-1].astype("float32")

        logger.trace("Final shape: %s", retval.shape)  # type:ignore
        return retval

    def _queue_out_frames(self, batch: list[ConvertItem], swapped_faces: np.ndarray) -> None:
        """ Compile the batch back to original frames and put to the Out Queue.

        For batching, faces are split away from their frames. This compiles all detected faces
        back to their parent frame before putting each frame to the out queue in batches.

        Parameters
        ----------
        batch: dict
            The batch that was used as the input for the model predict function
        swapped_faces: :class:`numpy.ndarray`
            The predictions returned from the model's predict function
        """
        logger.trace("Queueing out batch. Batchsize: %s", len(batch))  # type:ignore
        pointer = 0
        for item in batch:
            num_faces = len(item.inbound.detected_faces)
            if num_faces != 0:
                item.swapped_faces = swapped_faces[pointer:pointer + num_faces]

            logger.trace("Putting to queue. ('%s', detected_faces: %s, "  # type:ignore
                         "reference_faces: %s, swapped_faces: %s)", item.inbound.filename,
                         len(item.inbound.detected_faces), len(item.reference_faces),
                         item.swapped_faces.shape[0])
            pointer += num_faces
        self._out_queue.put(batch)
        logger.trace("Queued out batch. Batchsize: %s", len(batch))  # type:ignore


class OptionalActions():  # pylint:disable=too-few-public-methods
    """ Process specific optional actions for Convert.

    Currently only handles skip faces. This class should probably be (re)moved.

    Parameters
    ----------
    arguments: :class:`argparse.Namespace`
        The arguments that were passed to the convert process as generated from Faceswap's command
        line arguments
    input_images: list
        List of input image files
    alignments: :class:`lib.align.Alignments`
        The alignments file for this conversion
    """
    def __init__(self,
                 arguments: Namespace,
                 input_images: list[np.ndarray],
                 alignments: Alignments) -> None:
        logger.debug("Initializing %s", self.__class__.__name__)
        self._args = arguments
        self._input_images = input_images
        self._alignments = alignments

        self._remove_skipped_faces()
        logger.debug("Initialized %s", self.__class__.__name__)

    # SKIP FACES #
    def _remove_skipped_faces(self) -> None:
        """ If the user has specified an input aligned directory, remove any non-matching faces
        from the alignments file. """
        logger.debug("Filtering Faces")
        accept_dict = self._get_face_metadata()
        if not accept_dict:
            logger.debug("No aligned face data. Not skipping any faces")
            return
        pre_face_count = self._alignments.faces_count
        self._alignments.filter_faces(accept_dict, filter_out=False)
        logger.info("Faces filtered out: %s", pre_face_count - self._alignments.faces_count)

    def _get_face_metadata(self) -> dict[str, list[int]]:
        """ Check for the existence of an aligned directory for identifying which faces in the
        target frames should be swapped. If it exists, scan the folder for face's metadata

        Returns
        -------
        dict
            Dictionary of source frame names with a list of associated face indices to be skipped
        """
        retval: dict[str, list[int]] = {}
        input_aligned_dir = self._args.input_aligned_dir

        if input_aligned_dir is None:
            logger.verbose("Aligned directory not specified. All faces listed in "  # type:ignore
                           "the alignments file will be converted")
            return retval
        if not os.path.isdir(input_aligned_dir):
            logger.warning("Aligned directory not found. All faces listed in the "
                           "alignments file will be converted")
            return retval

        log_once = False
        filelist = get_image_paths(input_aligned_dir)
        for fullpath, metadata in tqdm(read_image_meta_batch(filelist),
                                       total=len(filelist),
                                       desc="Reading Face Data",
                                       leave=False):
            if "itxt" not in metadata or "source" not in metadata["itxt"]:
                # UPDATE LEGACY FACES FROM ALIGNMENTS FILE
                if not log_once:
                    logger.warning("Legacy faces discovered in '%s'. These faces will be updated",
                                   input_aligned_dir)
                    log_once = True
                data = update_legacy_png_header(fullpath, self._alignments)
                if not data:
                    raise FaceswapError(
                        f"Some of the faces being passed in from '{input_aligned_dir}' could not "
                        f"be matched to the alignments file '{self._alignments.file}'\n"
                        "Please double check your sources and try again.")
                meta = data["source"]
            else:
                meta = metadata["itxt"]["source"]
            retval.setdefault(meta["source_filename"], []).append(meta["face_index"])

        if not retval:
            raise FaceswapError("Aligned directory is empty, no faces will be converted!")
        if len(retval) <= len(self._input_images) / 3:
            logger.warning("Aligned directory contains far fewer images than the input "
                           "directory, are you sure this is the right folder?")
        return retval
