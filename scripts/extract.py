#!/usr/bin python3
""" Main entry point to the extract process of FaceSwap """

import logging
import os
import sys

from tqdm import tqdm

from lib.image import encode_image_with_hash, ImagesLoader, ImagesSaver
from lib.multithreading import MultiThread
from lib.utils import get_folder
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
    arguments: argparse.Namespace
        The arguments to be passed to the extraction process as generated from Faceswap's command
        line arguments
    """
    def __init__(self, arguments):
        logger.debug("Initializing %s: (args: %s", self.__class__.__name__, arguments)
        self._args = arguments
        self._output_dir = str(get_folder(self._args.output_dir))

        logger.info("Output Directory: %s", self._args.output_dir)
        self._images = ImagesLoader(self._args.input_dir, fast_count=True)
        self._alignments = Alignments(self._args, True, self._images.is_video)

        self._existing_count = 0
        self._set_skip_list()

        self._post_process = PostProcess(arguments)
        configfile = self._args.configfile if hasattr(self._args, "configfile") else None
        normalization = None if self._args.normalization == "none" else self._args.normalization

        maskers = ["components", "extended"]
        maskers += self._args.masker if self._args.masker else []
        self._extractor = Extractor(self._args.detector,
                                    self._args.aligner,
                                    maskers,
                                    configfile=configfile,
                                    multiprocess=not self._args.singleprocess,
                                    rotate_images=self._args.rotate_images,
                                    min_size=self._args.min_size,
                                    normalize_method=normalization)
        self._threads = list()
        self._verify_output = False
        logger.debug("Initialized %s", self.__class__.__name__)

    @property
    def _save_interval(self):
        """ int: The number of frames to be processed between each saving of the alignments file if
        it has been provided, otherwise ``None`` """
        if hasattr(self._args, "save_interval"):
            return self._args.save_interval
        return None

    @property
    def _skip_num(self):
        """ int: Number of frames to skip if extract_every_n has been provided """
        return self._args.extract_every_n if hasattr(self._args, "extract_every_n") else 1

    def _set_skip_list(self):
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
                logger.trace("Adding image '%s' to skip list due to extract_every_n = %s",
                             filename, self._skip_num)
                skip_list.append(idx)
            # Items may be in the alignments file if skip-existing[-faces] is selected
            elif os.path.basename(filename) in self._alignments.data:
                self._existing_count += 1
                logger.trace("Removing image: '%s' due to previously existing", filename)
                skip_list.append(idx)
        if self._existing_count != 0:
            logger.info("Skipping %s frames due to skip_existing/skip_existing_faces.",
                        self._existing_count)
        logger.debug("Adding skip list: %s", skip_list)
        self._images.add_skip_list(skip_list)

    def process(self):
        """ The entry point for triggering the Extraction Process.

        Should only be called from  :class:`lib.cli.launcher.ScriptExecutor`
        """
        logger.info('Starting, this may take a while...')
        # from lib.queue_manager import queue_manager ; queue_manager.debug_monitor(3)
        self._threaded_redirector("load")
        self._run_extraction()
        for thread in self._threads:
            thread.join()
        self._alignments.save()
        finalize(self._images.process_count + self._existing_count,
                 self._alignments.faces_count,
                 self._verify_output)

    def _threaded_redirector(self, task, io_args=None):
        """ Redirect image input/output tasks to relevant queues in background thread

        Parameters
        ----------
        task: str
            The name of the task to be put into a background thread
        io_args: tuple, optional
            Any arguments that need to be provided to the background function
        """
        logger.debug("Threading task: (Task: '%s')", task)
        io_args = tuple() if io_args is None else (io_args, )
        func = getattr(self, "_{}".format(task))
        io_thread = MultiThread(func, *io_args, thread_count=1)
        io_thread.start()
        self._threads.append(io_thread)

    def _load(self):
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

    def _reload(self, detected_faces):
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
            logger.trace("Reloading image: '%s'", filename)
            extract_media = detected_faces.pop(filename, None)
            if not extract_media:
                logger.warning("Couldn't find faces for: %s", filename)
                continue
            extract_media.set_image(image)
            load_queue.put(extract_media)
        load_queue.put("EOF")
        logger.debug("Reload Images: Complete")

    def _run_extraction(self):
        """ The main Faceswap Extraction process

        Receives items from :class:`plugins.extract.Pipeline.Extractor` and either saves out the
        faces and data (if on the final pass) or reprocesses data through the pipeline for serial
        processing.
        """
        size = self._args.size if hasattr(self._args, "size") else 256
        saver = ImagesSaver(self._output_dir, as_bytes=True)
        exception = False

        for phase in range(self._extractor.passes):
            if exception:
                break
            is_final = self._extractor.final_pass
            detected_faces = dict()
            self._extractor.launch()
            self._check_thread_error()
            ph_desc = "Extraction" if self._extractor.passes == 1 else self._extractor.phase_text
            desc = "Running pass {} of {}: {}".format(phase + 1,
                                                      self._extractor.passes,
                                                      ph_desc)
            status_bar = tqdm(self._extractor.detected_faces(),
                              total=self._images.process_count,
                              file=sys.stdout,
                              desc=desc)
            for idx, extract_media in enumerate(status_bar):
                self._check_thread_error()
                if is_final:
                    self._output_processing(extract_media, size)
                    if not self._args.skip_saving_faces:
                        self._output_faces(saver, extract_media)
                    if self._save_interval and (idx + 1) % self._save_interval == 0:
                        self._alignments.save()
                else:
                    extract_media.remove_image()
                    # cache extract_media for next run
                    detected_faces[extract_media.filename] = extract_media
                status_bar.update(1)

            if not is_final:
                logger.debug("Reloading images")
                self._threaded_redirector("reload", detected_faces)
        saver.close()

    def _check_thread_error(self):
        """ Check if any errors have occurred in the running threads and their errors """
        for thread in self._threads:
            thread.check_and_raise_error()

    def _output_processing(self, extract_media, size):
        """ Prepare faces for output

        Loads the aligned face, perform any processing actions and verify the output.

        Parameters
        ----------
        extract_media: :class:`plugins.extract.pipeline.ExtractMedia`
            Output from :class:`plugins.extract.pipeline.Extractor`
        size: int
            The size that the aligned face should be created at
        """
        for face in extract_media.detected_faces:
            face.load_aligned(extract_media.image, size=size)

        self._post_process.do_actions(extract_media)
        extract_media.remove_image()

        faces_count = len(extract_media.detected_faces)
        if faces_count == 0:
            logger.verbose("No faces were detected in image: %s",
                           os.path.basename(extract_media.filename))

        if not self._verify_output and faces_count > 1:
            self._verify_output = True

    def _output_faces(self, saver, extract_media):
        """ Output faces to save thread

        Set the face filename based on the frame name and put the face to the
        :class:`~lib.image.ImagesSaver` save queue and add the face information to the alignments
        data.

        Parameters
        ----------
        saver: lib.images.ImagesSaver
            The background saver for saving the image
        extract_media: :class:`~plugins.extract.pipeline.ExtractMedia`
            The output from :class:`~plugins.extract.Pipeline.Extractor`
        """
        logger.trace("Outputting faces for %s", extract_media.filename)
        final_faces = list()
        filename, extension = os.path.splitext(os.path.basename(extract_media.filename))
        for idx, face in enumerate(extract_media.detected_faces):
            output_filename = "{}_{}{}".format(filename, str(idx), extension)
            face.hash, image = encode_image_with_hash(face.aligned_face, extension)

            saver.save(output_filename, image)
            final_faces.append(face.to_alignment())
        self._alignments.data[os.path.basename(extract_media.filename)] = dict(faces=final_faces)
        del extract_media
