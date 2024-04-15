#!/usr/bin python3
""" Main entry point to the extract process of FaceSwap """

from __future__ import annotations
import logging
import os
import sys
import typing as T

from argparse import Namespace
from multiprocessing import Process

import numpy as np
from tqdm import tqdm
from lib.align.alignments import PNGHeaderDict

from lib.image import encode_image, generate_thumbnail, ImagesLoader, ImagesSaver, read_image_meta
from lib.multithreading import MultiThread
from lib.utils import get_folder, handle_deprecated_cliopts, IMAGE_EXTENSIONS, VIDEO_EXTENSIONS
from plugins.extract import ExtractMedia, Extractor
from scripts.fsmedia import Alignments, PostProcess, finalize

if T.TYPE_CHECKING:
    from lib.align.alignments import PNGHeaderAlignmentsDict

# tqdm.monitor_interval = 0  # workaround for TqdmSynchronisationWarning  # TODO?
logger = logging.getLogger(__name__)


class Extract():
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
        self._args = handle_deprecated_cliopts(arguments)
        self._input_locations = self._get_input_locations()
        self._validate_batchmode()

        configfile = self._args.configfile if hasattr(self._args, "configfile") else None
        normalization = None if self._args.normalization == "none" else self._args.normalization
        maskers = ["components", "extended"]
        maskers += self._args.masker if self._args.masker else []
        recognition = ("vgg_face2"
                       if arguments.identity or arguments.filter or arguments.nfilter
                       else None)
        self._extractor = Extractor(self._args.detector,
                                    self._args.aligner,
                                    maskers,
                                    recognition=recognition,
                                    configfile=configfile,
                                    multiprocess=not self._args.singleprocess,
                                    exclude_gpus=self._args.exclude_gpus,
                                    rotate_images=self._args.rotate_images,
                                    min_size=self._args.min_size,
                                    normalize_method=normalization,
                                    re_feed=self._args.re_feed,
                                    re_align=self._args.re_align)
        self._filter = Filter(self._args.ref_threshold,
                              self._args.filter,
                              self._args.nfilter,
                              self._extractor)

    def _get_input_locations(self) -> list[str]:
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
                      and any(os.path.splitext(iname)[-1].lower() in IMAGE_EXTENSIONS
                              for iname in os.listdir(os.path.join(self._args.input_dir, fname))))
                  or os.path.splitext(fname)[-1].lower() in VIDEO_EXTENSIONS]  # video

        logger.debug("Input locations: %s", retval)
        return retval

    def _validate_batchmode(self) -> None:
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

    def process(self) -> None:
        """ The entry point for triggering the Extraction Process.

        Should only be called from  :class:`lib.cli.launcher.ScriptExecutor`
        """
        logger.info('Starting, this may take a while...')
        if self._args.batch_mode:
            logger.info("Batch mode selected processing: %s", self._input_locations)
        for job_no, location in enumerate(self._input_locations):
            if self._args.batch_mode:
                logger.info("Processing job %s of %s: '%s'",
                            job_no + 1, len(self._input_locations), location)
                arguments = Namespace(**self._args.__dict__)
                arguments.input_dir = location
                arguments.output_dir = self._output_for_input(location)
            else:
                arguments = self._args
            extract = _Extract(self._extractor, arguments)
            if sys.platform == "linux" and len(self._input_locations) > 1:
                # TODO - Running this in a process is hideously hacky. However, there is a memory
                # leak in some instances when running in batch mode. Many days have been spent
                # trying to track this down to no avail (most likely coming from C-code.) Running
                # the extract job inside a process prevents the memory leak in testing. This should
                # be replaced if/when the memory leak is found
                # Only done for Linux as not reported elsewhere and this new process won't work in
                # Windows because it can't fork.
                proc = Process(target=extract.process)
                proc.start()
                proc.join()
            else:
                extract.process()
            self._extractor.reset_phase_index()


class Filter():
    """ Obtains and holds face identity embeddings for any filter/nfilter image files
    passed in from the command line.

    Parameters
    ----------
    filter_files: list or ``None``
        The list of filter file(s) passed in as command line arguments
    nfilter_files: list or ``None``
        The list of nfilter file(s) passed in as command line arguments
    extractor: :class:`~plugins.extract.pipeline.Extractor`
        The extractor pipeline for obtaining face identity from images
    """
    def __init__(self,
                 threshold: float,
                 filter_files: list[str] | None,
                 nfilter_files: list[str] | None,
                 extractor: Extractor) -> None:
        logger.debug("Initializing %s: (threshold: %s, filter_files: %s, nfilter_files: %s "
                     "extractor: %s)", self.__class__.__name__, threshold, filter_files,
                     nfilter_files, extractor)
        self._threshold = threshold
        self._filter_files, self._nfilter_files = self._validate_inputs(filter_files,
                                                                        nfilter_files)

        if not self._filter_files and not self._nfilter_files:
            logger.debug("Filter not selected. Exiting %s", self.__class__.__name__)
            return

        self._embeddings: list[np.ndarray] = [np.array([]) for _ in self._filter_files]
        self._nembeddings: list[np.ndarray] = [np.array([]) for _ in self._nfilter_files]
        self._extractor = extractor

        self._get_embeddings()
        self._extractor.recognition.add_identity_filters(self.embeddings,
                                                         self.n_embeddings,
                                                         self._threshold)
        logger.debug("Initialized %s", self.__class__.__name__)

    @property
    def active(self):
        """ bool: ``True`` if filter files have been passed in command line arguments. ``False`` if
        no filter files have been provided """
        return bool(self._filter_files) or bool(self._nfilter_files)

    @property
    def embeddings(self) -> np.ndarray:
        """ :class:`numpy.ndarray`: The filter embeddings"""
        if self._embeddings and all(np.any(e) for e in self._embeddings):
            retval = np.concatenate(self._embeddings, axis=0)
        else:
            retval = np.array([])
        return retval

    @property
    def n_embeddings(self) -> np.ndarray:
        """ :class:`numpy.ndarray`: The n-filter embeddings"""
        if self._nembeddings and all(np.any(e) for e in self._nembeddings):
            retval = np.concatenate(self._nembeddings, axis=0)
        else:
            retval = np.array([])
        return retval

    @classmethod
    def _files_from_folder(cls, input_location: list[str]) -> list[str]:
        """ Test whether the input location is a folder and if so, return the list of contained
        image files, otherwise return the original input location

        Parameters
        ---------
        input_files: list
            A list of full paths to individual files or to a folder location

        Returns
        -------
        bool
            Either the original list of files provided, or the image files that exist in the
            provided folder location
        """
        if not input_location or len(input_location) > 1:
            return input_location

        test_folder = input_location[0]
        if not os.path.isdir(test_folder):
            logger.debug("'%s' is not a folder. Returning original list", test_folder)
            return input_location

        retval = [os.path.join(test_folder, fname)
                  for fname in os.listdir(test_folder)
                  if os.path.splitext(fname)[-1].lower() in IMAGE_EXTENSIONS]
        logger.info("Collected files from folder '%s': %s", test_folder,
                    [os.path.basename(f) for f in retval])
        return retval

    def _validate_inputs(self,
                         filter_files: list[str] | None,
                         nfilter_files: list[str] | None) -> tuple[list[str], list[str]]:
        """ Validates that the given filter/nfilter files exist, are image files and are unique

        Parameters
        ----------
        filter_files: list or ``None``
            The list of filter file(s) passed in as command line arguments
        nfilter_files: list or ``None``
            The list of nfilter file(s) passed in as command line arguments

        Returns
        -------
        filter_files: list
            List of full paths to filter files
        nfilter_files: list
            List of full paths to nfilter files
        """
        error = False
        retval: list[list[str]] = []

        for files in (filter_files, nfilter_files):
            filt_files = [] if files is None else self._files_from_folder(files)
            for file in filt_files:
                if (not os.path.isfile(file) or
                        os.path.splitext(file)[-1].lower() not in IMAGE_EXTENSIONS):
                    logger.warning("Filter file '%s' does not exist or is not an image file", file)
                    error = True
            retval.append(filt_files)

        filters = retval[0]
        nfilters = retval[1]
        f_fnames = set(os.path.basename(fname) for fname in filters)
        n_fnames = set(os.path.basename(fname) for fname in nfilters)
        if f_fnames.intersection(n_fnames):
            error = True
            logger.warning("filter and nfilter filenames should be unique. The following "
                           "filenames exist in both folders: %s", f_fnames.intersection(n_fnames))

        if error:
            logger.error("There was a problem processing filter files. See the above warnings for "
                         "details")
            sys.exit(1)
        logger.debug("filter_files: %s, nfilter_files: %s", retval[0], retval[1])

        return filters, nfilters

    @classmethod
    def _identity_from_extracted(cls, filename) -> tuple[np.ndarray, bool]:
        """ Test whether the given image is a faceswap extracted face and contains identity
        information. If so, return the identity embedding

        Parameters
        ----------
        filename: str
            Full path to the image file to load

        Returns
        -------
        :class:`numpy.ndarray`
            The identity embeddings, if they can be obtained from the image header, otherwise an
            empty array
        bool
            ``True`` if the image is a faceswap extracted image otherwise ``False``
        """
        if os.path.splitext(filename)[-1].lower() != ".png":
            logger.debug("'%s' not a png. Returning empty array", filename)
            return np.array([]), False

        meta = read_image_meta(filename)
        if "itxt" not in meta or "alignments" not in meta["itxt"]:
            logger.debug("'%s' does not contain faceswap data. Returning empty array", filename)
            return np.array([]), False

        align: "PNGHeaderAlignmentsDict" = meta["itxt"]["alignments"]
        if "identity" not in align or "vggface2" not in align["identity"]:
            logger.debug("'%s' does not contain identity data. Returning empty array", filename)
            return np.array([]), True

        retval = np.array(align["identity"]["vggface2"])
        logger.debug("Obtained identity for '%s'. Shape: %s", filename, retval.shape)

        return retval, True

    def _process_extracted(self, item: ExtractMedia) -> None:
        """ Process the output from the extraction pipeline.

        If no face has been detected, or multiple faces are detected for the inclusive filter,
        embeddings and filenames are removed from the filter.

        if a single face is detected or multiple faces are detected for the exclusive filter,
        embeddings are added to the relevent filter list

        Parameters
        ----------
        item: :class:`plugins.extract.Pipeline.ExtracMedia`
            The output from the extraction pipeline containing the identity encodings
        """
        is_filter = item.filename in self._filter_files
        lbl = "filter" if is_filter else "nfilter"
        filelist = self._filter_files if is_filter else self._nfilter_files
        embeddings = self._embeddings if is_filter else self._nembeddings
        identities = np.array([face.identity["vggface2"] for face in item.detected_faces])
        idx = filelist.index(item.filename)

        if len(item.detected_faces) == 0:
            logger.warning("No faces detected for %s in file '%s'. Image will not be used",
                           lbl, os.path.basename(item.filename))
            filelist.pop(idx)
            embeddings.pop(idx)
            return

        if len(item.detected_faces) == 1:
            logger.debug("Adding identity for %s from file '%s'", lbl, item.filename)
            embeddings[idx] = identities
            return

        if len(item.detected_faces) > 1 and is_filter:
            logger.warning("%s faces detected for filter in '%s'. These identies will not be used",
                           len(item.detected_faces), os.path.basename(item.filename))
            filelist.pop(idx)
            embeddings.pop(idx)
            return

        if len(item.detected_faces) > 1 and not is_filter:
            logger.warning("%s faces detected for nfilter in '%s'. All of these identies will be "
                           "used", len(item.detected_faces), os.path.basename(item.filename))
            embeddings[idx] = identities
            return

    def _identity_from_extractor(self, file_list: list[str], aligned: list[str]) -> None:
        """ Obtain the identity embeddings from the extraction pipeline

        Parameters
        ----------
        filesile_list: list
            List of full path to images to run through the extraction pipeline
        aligned: list
            List of full path to images that exist in attr:`filelist` that are faceswap aligned
            images
        """
        logger.info("Extracting faces to obtain identity from images")
        logger.debug("Files requiring full extraction: %s",
                     [fname for fname in file_list if fname not in aligned])
        logger.debug("Aligned files requiring identity info: %s", aligned)

        loader = PipelineLoader(file_list, self._extractor, aligned_filenames=aligned)
        loader.launch()

        for phase in range(self._extractor.passes):
            is_final = self._extractor.final_pass
            detected_faces: dict[str, ExtractMedia] = {}
            self._extractor.launch()
            desc = "Obtaining reference face Identity"
            if self._extractor.passes > 1:
                desc = (f"{desc } pass {phase + 1} of {self._extractor.passes}: "
                        f"{self._extractor.phase_text}")
            for extract_media in tqdm(self._extractor.detected_faces(),
                                      total=len(file_list),
                                      file=sys.stdout,
                                      desc=desc):
                if is_final:
                    self._process_extracted(extract_media)
                else:
                    extract_media.remove_image()
                    # cache extract_media for next run
                    detected_faces[extract_media.filename] = extract_media

            if not is_final:
                logger.debug("Reloading images")
                loader.reload(detected_faces)

        self._extractor.reset_phase_index()

    def _get_embeddings(self) -> None:
        """ Obtain the embeddings for the given filter lists """
        needs_extraction: list[str] = []
        aligned: list[str] = []

        for files, embed in zip((self._filter_files, self._nfilter_files),
                                (self._embeddings, self._nembeddings)):
            for idx, file in enumerate(files):
                identity, is_aligned = self._identity_from_extracted(file)
                if np.any(identity):
                    logger.debug("Obtained identity from png header: '%s'", file)
                    embed[idx] = identity[None, ...]
                    continue

                needs_extraction.append(file)
                if is_aligned:
                    aligned.append(file)

        if needs_extraction:
            self._identity_from_extractor(needs_extraction, aligned)

        if not self._nfilter_files and not self._filter_files:
            logger.error("No faces were detected from your selected identity filter files")
            sys.exit(1)

        logger.debug("Filter: (filenames: %s, shape: %s), nFilter: (filenames: %s, shape: %s)",
                     [os.path.basename(f) for f in self._filter_files],
                     self.embeddings.shape,
                     [os.path.basename(f) for f in self._nfilter_files],
                     self.n_embeddings.shape)


class PipelineLoader():
    """ Handles loading and reloading images into the extraction pipeline.

    Parameters
    ----------
    path: str or list of str
        Full path to a folder of images or a video file or a list of image files
    extractor: :class:`~plugins.extract.pipeline.Extractor`
        The extractor pipeline for obtaining face identity from images
    aligned_filenames: list, optional
        Used for when the loader is used for getting face filter embeddings. List of full path to
        image files that exist in :attr:`path` that are aligned faceswap images
    """
    def __init__(self,
                 path: str | list[str],
                 extractor: Extractor,
                 aligned_filenames: list[str] | None = None) -> None:
        logger.debug("Initializing %s: (path: %s, extractor: %s, aligned_filenames: %s)",
                     self.__class__.__name__, path, extractor, aligned_filenames)
        self._images = ImagesLoader(path, fast_count=True)
        self._extractor = extractor
        self._threads: list[MultiThread] = []
        self._aligned_filenames = [] if aligned_filenames is None else aligned_filenames
        logger.debug("Initialized %s", self.__class__.__name__)

    @property
    def is_video(self) -> bool:
        """ bool: ``True`` if the input location is a video file, ``False`` if it is a folder of
        images """
        return self._images.is_video

    @property
    def file_list(self) -> list[str]:
        """ list: A full list of files in the source location. If the input is a video
        then this is a list of dummy filenames as corresponding to an alignments file """
        return self._images.file_list

    @property
    def process_count(self) -> int:
        """ int: The number of images or video frames to be processed (IE the total count less
        items that are to be skipped from the :attr:`skip_list`)"""
        return self._images.process_count

    def add_skip_list(self, skip_list: list[int]) -> None:
        """ Add a skip list to the :class:`ImagesLoader`

        Parameters
        ----------
        skip_list: list
            A list of indices corresponding to the frame indices that should be skipped by the
            :func:`load` function.
        """
        self._images.add_skip_list(skip_list)

    def launch(self) -> None:
        """ Launch the image loading pipeline """
        self._threaded_redirector("load")

    def reload(self, detected_faces: dict[str, ExtractMedia]) -> None:
        """ Reload images for multiple pipeline passes """
        self._threaded_redirector("reload", (detected_faces, ))

    def check_thread_error(self) -> None:
        """ Check if any errors have occurred in the running threads and raise their errors """
        for thread in self._threads:
            thread.check_and_raise_error()

    def join(self) -> None:
        """ Join all open loader threads """
        for thread in self._threads:
            thread.join()

    def _threaded_redirector(self, task: str, io_args: tuple | None = None) -> None:
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
            is_aligned = filename in self._aligned_filenames
            item = ExtractMedia(filename, image[..., :3], is_aligned=is_aligned)
            load_queue.put(item)
        load_queue.put("EOF")
        logger.debug("Load Images: Complete")

    def _reload(self, detected_faces: dict[str, ExtractMedia]) -> None:
        """ Reload the images and pair to detected face

        When the extraction pipeline is running in serial mode, images are reloaded from disk,
        paired with their extraction data and passed back into the extraction queue

        Parameters
        ----------
        detected_faces: dict
            Dictionary of :class:`~plugins.extract.extract_media.ExtractMedia` with the filename as
            the key for repopulating the image attribute.
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


class _Extract():
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
        self._loader = PipelineLoader(self._args.input_dir, extractor)

        self._alignments = Alignments(self._args, True, self._loader.is_video)
        self._extractor = extractor
        self._extractor.import_data(self._args.input_dir)

        self._existing_count = 0
        self._set_skip_list()

        self._post_process = PostProcess(arguments)
        self._verify_output = False
        logger.debug("Initialized %s", self.__class__.__name__)

    @property
    def _save_interval(self) -> int | None:
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
        for idx, filename in enumerate(self._loader.file_list):
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
        self._loader.add_skip_list(skip_list)

    def process(self) -> None:
        """ The entry point for triggering the Extraction Process.

        Should only be called from  :class:`lib.cli.launcher.ScriptExecutor`
        """
        # from lib.queue_manager import queue_manager ; queue_manager.debug_monitor(3)
        self._loader.launch()
        self._run_extraction()
        self._loader.join()
        self._alignments.save()
        finalize(self._loader.process_count + self._existing_count,
                 self._alignments.faces_count,
                 self._verify_output)

    def _run_extraction(self) -> None:
        """ The main Faceswap Extraction process

        Receives items from :class:`plugins.extract.Pipeline.Extractor` and either saves out the
        faces and data (if on the final pass) or reprocesses data through the pipeline for serial
        processing.
        """
        size = self._args.size if hasattr(self._args, "size") else 256
        saver = None if self._args.skip_saving_faces else ImagesSaver(self._output_dir,
                                                                      as_bytes=True)
        for phase in range(self._extractor.passes):
            is_final = self._extractor.final_pass
            detected_faces: dict[str, ExtractMedia] = {}
            self._extractor.launch()
            self._loader.check_thread_error()
            ph_desc = "Extraction" if self._extractor.passes == 1 else self._extractor.phase_text
            desc = f"Running pass {phase + 1} of {self._extractor.passes}: {ph_desc}"
            for idx, extract_media in enumerate(tqdm(self._extractor.detected_faces(),
                                                     total=self._loader.process_count,
                                                     file=sys.stdout,
                                                     desc=desc,
                                                     leave=False)):
                self._loader.check_thread_error()
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
                self._loader.reload(detected_faces)
        if saver is not None:
            saver.close()

    def _output_processing(self, extract_media: ExtractMedia, size: int) -> None:
        """ Prepare faces for output

        Loads the aligned face, generate the thumbnail, perform any processing actions and verify
        the output.

        Parameters
        ----------
        extract_media: :class:`~plugins.extract.extract_media.ExtractMedia`
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

    def _output_faces(self, saver: ImagesSaver | None, extract_media: ExtractMedia) -> None:
        """ Output faces to save thread

        Set the face filename based on the frame name and put the face to the
        :class:`~lib.image.ImagesSaver` save queue and add the face information to the alignments
        data.

        Parameters
        ----------
        saver: :class:`lib.images.ImagesSaver` or ``None``
            The background saver for saving the image or ``None`` if faces are not to be saved
        extract_media: :class:`~plugins.extract.extract_media.ExtractMedia`
            The output from :class:`~plugins.extract.Pipeline.Extractor`
        """
        logger.trace("Outputting faces for %s", extract_media.filename)  # type: ignore
        final_faces = []
        filename = os.path.splitext(os.path.basename(extract_media.filename))[0]

        skip_idx = 0
        for face_id, face in enumerate(extract_media.detected_faces):
            real_face_id = face_id - skip_idx
            output_filename = f"{filename}_{real_face_id}.png"
            aligned = face.aligned.face
            assert aligned is not None
            meta: PNGHeaderDict = {
                "alignments": face.to_png_meta(),
                "source": {"alignments_version": self._alignments.version,
                           "original_filename": output_filename,
                           "face_index": real_face_id,
                           "source_filename": os.path.basename(extract_media.filename),
                           "source_is_video": self._loader.is_video,
                           "source_frame_dims": extract_media.image_size}}
            image = encode_image(aligned, ".png", metadata=meta)

            sub_folder = extract_media.sub_folders[face_id]
            # Binned faces shouldn't risk filename clash, so just use original id
            out_name = output_filename if not sub_folder else f"{filename}_{face_id}.png"

            if saver is not None:
                saver.save(out_name, image, sub_folder)

            if sub_folder:  # This is a filtered out face being binned
                skip_idx += 1
                continue
            final_faces.append(face.to_alignment())

        self._alignments.data[os.path.basename(extract_media.filename)] = {"faces": final_faces,
                                                                           "video_meta": {}}
        del extract_media
