#!/usr/bin python3
""" Main entry point to the extract process of FaceSwap """

from __future__ import annotations

import logging
import os
import sys
from argparse import Namespace
from typing import List, Dict, Optional

from tqdm import tqdm

from lib.image import encode_image, generate_thumbnail, ImagesLoader, ImagesSaver
from lib.multithreading import MultiThread
from lib.utils import get_folder, _image_extensions, _video_extensions
from plugins.extract.pipeline import Extractor, ExtractMedia
from scripts.fsmedia import Alignments, PostProcess, finalize


tqdm.monitor_interval = 0  # workaround for TqdmSynchronisationWarning
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class Extract():  # pylint:disable=too-few-public-methods
    """ The Faceswap Face Extraction Process.

    The extraction process is responsible for detecting faces in a series of images/video, aligning
    these faces and then generating a mask.

    It leverages a series of user selected plugins, chained together using
    :mod:`plugins.extract.pipeline`.

    The extract process is self contained and should not be referenced by any other scripts, so it
    contains no public properties.

    Parameters
    ----------
    arguments: :class:`argparse.Namespace`
        The arguments to be passed to the extraction process as generated from Faceswap's command
        line arguments
    """
    def __init__(self, arguments: Namespace) -> None:
        logger.debug("Initializing %s: (args: %s", self.__class__.__name__, arguments)
        self._args = arguments
        self._input_locations = self._get_input_locations()
        self._validate_batchmode()

        configfile = self._args.configfile if hasattr(self._args, "configfile") else None
        normalization = None if self._args.normalization == "none" else self._args.normalization
        maskers = ["components", "extended"]
        maskers += self._args.masker if self._args.masker else []
        self._extractor = Extractor(self._args.detector,
                                    self._args.aligner,
                                    maskers,
                                    configfile=configfile,
                                    multiprocess=not self._args.singleprocess,
                                    exclude_gpus=self._args.exclude_gpus,
                                    rotate_images=self._args.rotate_images,
                                    min_size=self._args.min_size,
                                    normalize_method=normalization,
                                    re_feed=self._args.re_feed)

    def _get_input_locations(self) -> List[str]:
        """ Obtain the full path to input locations. Will be a list of locations if batch mode is
        selected, or a containing a single location if batch mode is not selected.

        Returns
        -------
        list:
            The list of input location paths
        """
        if not self._args.batch_mode or os.path.isfile(self._args.input_dir):
            return [self._args.input_dir]  # Not batch mode or a single file

        retval = [os.path.join(self._args.input_dir, fname)
                  for fname in os.listdir(self._args.input_dir)
                  if (os.path.isdir(os.path.join(self._args.input_dir, fname))  # folder images
                      and any(os.path.splitext(iname)[-1].lower() in _image_extensions
                              for iname in os.listdir(os.path.join(self._args.input_dir, fname))))
                  or os.path.splitext(fname)[-1].lower() in _video_extensions]  # video

        logger.debug("Input locations: %s", retval)
        return retval

    def _validate_batchmode(self):
        """ Validate the command line arguments.

        If batch-mode selected and there is only one object to extract from, then batch mode is
        disabled

        If processing in batch mode, some of the given arguments may not make sense, in which case
        a warning is shown and those options are reset.

        """
        if not self._args.batch_mode:
            return

        if os.path.isfile(self._args.input_dir):
            logger.warning("Batch mode selected but input is not a folder. Switching to normal "
                           "mode")
            self._args.batch_mode = False

        if not self._input_locations:
            logger.error("Batch mode selected, but no valid files found in input location: '%s'. "
                         "Exiting.", self._args.input_dir)
            sys.exit(1)

        if self._args.alignments_path:
            logger.warning("Custom alignments path not supported for batch mode. "
                           "Reverting to default.")
            self._args.alignments_path = None

    def _output_for_input(self, input_location: str) -> str:
        """ Obtain the path to an output folder for faces for a given input location.

        If not running in batch mode, then the user supplied output location will be returned,
        otherwise a sub-folder within the user supplied output location will be returned based on
        the input filename

        Parameters
        ----------
        input_location: str
            The full path to an input video or folder of images
        """
        if not self._args.batch_mode:
            return self._args.output_dir

        retval = os.path.join(self._args.output_dir,
                              os.path.splitext(os.path.basename(input_location))[0])
        logger.debug("Returning output: '%s' for input: '%s'", retval, input_location)
        return retval

    def process(self):
        """ The entry point for triggering the Extraction Process.

        Should only be called from  :class:`lib.cli.launcher.ScriptExecutor`
        """
        logger.info('Starting, this may take a while...')
        inputs = self._input_locations
        if self._args.batch_mode:
            logger.info("Batch mode selected processing: %s", self._input_locations)
        for job_no, location in enumerate(self._input_locations):
            if self._args.batch_mode:
                logger.info("Processing job %s of %s: '%s'", job_no + 1, len(inputs), location)
                arguments = Namespace(**self._args.__dict__)
                arguments.input_dir = location
                arguments.output_dir = self._output_for_input(location)
            else:
                arguments = self._args
            extract = _Extract(self._extractor, arguments)
            extract.process()
            self._extractor.reset_phase_index()


class _Extract():  # pylint:disable=too-few-public-methods
    """ The Actual extraction process.

    This class is called by the parent :class:`Extract` process

    Parameters
    ----------
    extractor: :class:`~plugins.extract.pipeline.Extractor`
        The extractor pipeline for running extractions
    arguments: :class:`argparse.Namespace`
        The arguments to be passed to the extraction process as generated from Faceswap's command
        line arguments
    """
    def __init__(self,
                 extractor: Extractor,
                 arguments: Namespace) -> None:
        logger.debug("Initializing %s: (extractor: %s, args: %s)", self.__class__.__name__,
                     extractor, arguments)
        self._args = arguments
        self._output_dir = None if self._args.skip_saving_faces else get_folder(
            self._args.output_dir)

        logger.info("Output Directory: %s", self._output_dir)
        self._images = ImagesLoader(self._args.input_dir, fast_count=True)
        self._alignments = Alignments(self._args, True, self._images.is_video)
        self._extractor = extractor

        self._existing_count = 0
        self._set_skip_list()

        self._post_process = PostProcess(arguments)
        self._threads: List[MultiThread] = []
        self._verify_output = False
        logger.debug("Initialized %s", self.__class__.__name__)

    @property
    def _save_interval(self) -> Optional[int]:
        """ int: The number of frames to be processed between each saving of the alignments file if
        it has been provided, otherwise ``None`` """
        if hasattr(self._args, "save_interval"):
            return self._args.save_interval
        return None

    @property
    def _skip_num(self) -> int:
        """ int: Number of frames to skip if extract_every_n has been provided """
        return self._args.extract_every_n if hasattr(self._args, "extract_every_n") else 1

    def _set_skip_list(self) -> None:
        """ Add the skip list to the image loader

        Checks against `extract_every_n` and the existence of alignments data (can exist if
        `skip_existing` or `skip_existing_faces` has been provided) and compiles a list of frame
        indices that should not be processed, providing these to :class:`lib.image.ImagesLoader`.
        """
        if self._skip_num == 1 and not self._alignments.data:
            logger.debug("No frames to be skipped")
            return
        skip_list = []
        for idx, filename in enumerate(self._images.file_list):
            if idx % self._skip_num != 0:
                logger.trace("Adding image '%s' to skip list due to "  # type: ignore
                             "extract_every_n = %s", filename, self._skip_num)
                skip_list.append(idx)
            # Items may be in the alignments file if skip-existing[-faces] is selected
            elif os.path.basename(filename) in self._alignments.data:
                self._existing_count += 1
                logger.trace("Removing image: '%s' due to previously existing",  # type: ignore
                             filename)
                skip_list.append(idx)
        if self._existing_count != 0:
            logger.info("Skipping %s frames due to skip_existing/skip_existing_faces.",
                        self._existing_count)
        logger.debug("Adding skip list: %s", skip_list)
        self._images.add_skip_list(skip_list)

    def process(self) -> None:
        """ The entry point for triggering the Extraction Process.

        Should only be called from  :class:`lib.cli.launcher.ScriptExecutor`
        """
        # from lib.queue_manager import queue_manager ; queue_manager.debug_monitor(3)
        self._threaded_redirector("load")
        self._run_extraction()
        for thread in self._threads:
            thread.join()
        self._alignments.save()
        finalize(self._images.process_count + self._existing_count,
                 self._alignments.faces_count,
                 self._verify_output)

    def _threaded_redirector(self, task: str, io_args: Optional[tuple] = None) -> None:
        """ Redirect image input/output tasks to relevant queues in background thread

        Parameters
        ----------
        task: str
            The name of the task to be put into a background thread
        io_args: tuple, optional
            Any arguments that need to be provided to the background function
        """
        logger.debug("Threading task: (Task: '%s')", task)
        io_args = tuple() if io_args is None else io_args
        func = getattr(self, f"_{task}")
        io_thread = MultiThread(func, *io_args, thread_count=1)
        io_thread.start()
        self._threads.append(io_thread)

    def _load(self) -> None:
        """ Load the images

        Loads images from :class:`lib.image.ImagesLoader`, formats them into a dict compatible
        with :class:`plugins.extract.Pipeline.Extractor` and passes them into the extraction queue.
        """
        logger.debug("Load Images: Start")
        load_queue = self._extractor.input_queue
        for filename, image in self._images.load():
            if load_queue.shutdown.is_set():
                logger.debug("Load Queue: Stop signal received. Terminating")
                break
            item = ExtractMedia(filename, image[..., :3])
            load_queue.put(item)
        load_queue.put("EOF")
        logger.debug("Load Images: Complete")

    def _reload(self, detected_faces: Dict[str, ExtractMedia]) -> None:
        """ Reload the images and pair to detected face

        When the extraction pipeline is running in serial mode, images are reloaded from disk,
        paired with their extraction data and passed back into the extraction queue

        Parameters
        ----------
        detected_faces: dict
            Dictionary of :class:`plugins.extract.pipeline.ExtractMedia` with the filename as the
            key for repopulating the image attribute.
        """
        logger.debug("Reload Images: Start. Detected Faces Count: %s", len(detected_faces))
        load_queue = self._extractor.input_queue
        for filename, image in self._images.load():
            if load_queue.shutdown.is_set():
                logger.debug("Reload Queue: Stop signal received. Terminating")
                break
            logger.trace("Reloading image: '%s'", filename)  # type: ignore
            extract_media = detected_faces.pop(filename, None)
            if not extract_media:
                logger.warning("Couldn't find faces for: %s", filename)
                continue
            extract_media.set_image(image)
            load_queue.put(extract_media)
        load_queue.put("EOF")
        logger.debug("Reload Images: Complete")

    def _run_extraction(self) -> None:
        """ The main Faceswap Extraction process

        Receives items from :class:`plugins.extract.Pipeline.Extractor` and either saves out the
        faces and data (if on the final pass) or reprocesses data through the pipeline for serial
        processing.
        """
        size = self._args.size if hasattr(self._args, "size") else 256
        saver = None if self._args.skip_saving_faces else ImagesSaver(self._output_dir,
                                                                      as_bytes=True)
        exception = False

        for phase in range(self._extractor.passes):
            if exception:
                break
            is_final = self._extractor.final_pass
            detected_faces = {}
            self._extractor.launch()
            self._check_thread_error()
            ph_desc = "Extraction" if self._extractor.passes == 1 else self._extractor.phase_text
            desc = f"Running pass {phase + 1} of {self._extractor.passes}: {ph_desc}"
            for idx, extract_media in enumerate(tqdm(self._extractor.detected_faces(),
                                                     total=self._images.process_count,
                                                     file=sys.stdout,
                                                     desc=desc)):
                self._check_thread_error()
                if is_final:
                    self._output_processing(extract_media, size)
                    self._output_faces(saver, extract_media)
                    if self._save_interval and (idx + 1) % self._save_interval == 0:
                        self._alignments.save()
                else:
                    extract_media.remove_image()
                    # cache extract_media for next run
                    detected_faces[extract_media.filename] = extract_media

            if not is_final:
                logger.debug("Reloading images")
                self._threaded_redirector("reload", (detected_faces, ))
        if saver is not None:
            saver.close()

    def _check_thread_error(self) -> None:
        """ Check if any errors have occurred in the running threads and their errors """
        for thread in self._threads:
            thread.check_and_raise_error()

    def _output_processing(self, extract_media: ExtractMedia, size: int) -> None:
        """ Prepare faces for output

        Loads the aligned face, generate the thumbnail, perform any processing actions and verify
        the output.

        Parameters
        ----------
        extract_media: :class:`plugins.extract.pipeline.ExtractMedia`
            Output from :class:`plugins.extract.pipeline.Extractor`
        size: int
            The size that the aligned face should be created at
        """
        for face in extract_media.detected_faces:
            face.load_aligned(extract_media.image,
                              size=size,
                              centering="head")
            face.thumbnail = generate_thumbnail(face.aligned.face, size=96, quality=60)
        self._post_process.do_actions(extract_media)
        extract_media.remove_image()

        faces_count = len(extract_media.detected_faces)
        if faces_count == 0:
            logger.verbose("No faces were detected in image: %s",  # type: ignore
                           os.path.basename(extract_media.filename))

        if not self._verify_output and faces_count > 1:
            self._verify_output = True

    def _output_faces(self, saver: Optional[ImagesSaver], extract_media: ExtractMedia) -> None:
        """ Output faces to save thread

        Set the face filename based on the frame name and put the face to the
        :class:`~lib.image.ImagesSaver` save queue and add the face information to the alignments
        data.

        Parameters
        ----------
        saver: :class:`lib.images.ImagesSaver` or ``None``
            The background saver for saving the image or ``None`` if faces are not to be saved
        extract_media: :class:`~plugins.extract.pipeline.ExtractMedia`
            The output from :class:`~plugins.extract.Pipeline.Extractor`
        """
        logger.trace("Outputting faces for %s", extract_media.filename)  # type: ignore
        final_faces = []
        filename = os.path.splitext(os.path.basename(extract_media.filename))[0]
        extension = ".png"

        for idx, face in enumerate(extract_media.detected_faces):
            output_filename = f"{filename}_{idx}{extension}"
            meta = dict(alignments=face.to_png_meta(),
                        source=dict(alignments_version=self._alignments.version,
                                    original_filename=output_filename,
                                    face_index=idx,
                                    source_filename=os.path.basename(extract_media.filename),
                                    source_is_video=self._images.is_video,
                                    source_frame_dims=extract_media.image_size))
            image = encode_image(face.aligned.face, extension, metadata=meta)

            if saver is not None:
                saver.save(output_filename, image)
            final_faces.append(face.to_alignment())
        self._alignments.data[os.path.basename(extract_media.filename)] = dict(faces=final_faces)
        del extract_media
