#!/usr/bin/env python3
""" Tools for manipulating the alignments serialized file """
import logging
import os
import sys
import typing as T

from argparse import Namespace
from multiprocessing import Process

from lib.utils import FaceswapError, handle_deprecated_cliopts, VIDEO_EXTENSIONS
from .media import AlignmentData
from .jobs import Check, Export, Sort, Spatial  # noqa pylint:disable=unused-import
from .jobs_faces import FromFaces, RemoveFaces, Rename  # noqa pylint:disable=unused-import
from .jobs_frames import Draw, Extract  # noqa pylint:disable=unused-import


logger = logging.getLogger(__name__)


class Alignments():
    """ The main entry point for Faceswap's Alignments Tool. This tool is part of the Faceswap
    Tools suite and should be called from the ``python tools.py alignments`` command.

    The tool allows for manipulation, and working with Faceswap alignments files.

    This parent class handles creating the individual job arguments when running in batch-mode or
    triggers the job when not running in batch mode

    Parameters
    ----------
    arguments: :class:`argparse.Namespace`
        The :mod:`argparse` arguments as passed in from :mod:`tools.py`
    """
    def __init__(self, arguments: Namespace) -> None:
        logger.debug("Initializing %s: (arguments: %s)", self.__class__.__name__, arguments)
        self._requires_alignments = ["export", "sort", "spatial"]
        self._requires_faces = ["extract", "from-faces"]
        self._requires_frames = ["draw",
                                 "extract",
                                 "missing-alignments",
                                 "missing-frames",
                                 "no-faces"]

        self._args = handle_deprecated_cliopts(arguments)
        self._batch_mode = self._validate_batch_mode()
        self._locations = self._get_locations()

    def _validate_batch_mode(self) -> bool:
        """ Validate that the selected job supports batch processing

        Returns
        -------
        bool
            ``True`` if batch mode has been selected otherwise ``False``
        """
        batch_mode: bool = self._args.batch_mode
        if not batch_mode:
            logger.debug("Running in standard mode")
            return batch_mode
        valid = self._requires_alignments + self._requires_faces + self._requires_frames
        if self._args.job not in valid:
            logger.error("Job '%s' does not support batch mode. Please select a job from %s or "
                         "disable batch mode", self._args.job, valid)
            sys.exit(1)
        logger.debug("Running in batch mode")
        return batch_mode

    def _get_alignments_locations(self) -> dict[str, list[str | None]]:
        """ Obtain the full path to alignments files in a parent (batch) location

        These are jobs that only require an alignments file as input, so frames and face locations
        are returned as a list of ``None`` values corresponding to the number of alignments files
        detected

        Returns
        -------
        dict[str, list[Optional[str]]]:
            The list of alignments location paths and None lists for frames and faces locations
        """
        if not self._args.alignments_file:
            logger.error("Please provide an 'alignments_file' location for '%s' job",
                         self._args.job)
            sys.exit(1)

        alignments = [os.path.join(self._args.alignments_file, fname)
                      for fname in os.listdir(self._args.alignments_file)
                      if os.path.splitext(fname)[-1].lower() == ".fsa"
                      and os.path.splitext(fname)[0].endswith("alignments")]
        if not alignments:
            logger.error("No alignment files found in '%s'", self._args.alignments_file)
            sys.exit(1)

        logger.info("Batch mode selected. Processing alignments: %s", alignments)
        retval = {"alignments_file": alignments,
                  "faces_dir": [None for _ in range(len(alignments))],
                  "frames_dir": [None for _ in range(len(alignments))]}
        return retval

    def _get_frames_locations(self) -> dict[str, list[str | None]]:
        """ Obtain the full path to frame locations along with corresponding alignments file
        locations contained within the parent (batch) location

        Returns
        -------
        dict[str, list[Optional[str]]]:
            list of frames and alignments location paths. If the job requires an output faces
            location then the faces folders are also returned, otherwise the faces will be a list
            of ``Nones`` corresponding to the number of jobs to run
        """
        if not self._args.frames_dir:
            logger.error("Please provide a 'frames_dir' location for '%s' job", self._args.job)
            sys.exit(1)

        frames: list[str] = []
        alignments: list[str] = []
        candidates = [os.path.join(self._args.frames_dir, fname)
                      for fname in os.listdir(self._args.frames_dir)
                      if os.path.isdir(os.path.join(self._args.frames_dir, fname))
                      or os.path.splitext(fname)[-1].lower() in VIDEO_EXTENSIONS]
        logger.debug("Frame candidates: %s", candidates)

        for candidate in candidates:
            fname = os.path.join(candidate, "alignments.fsa")
            if os.path.isdir(candidate) and os.path.exists(fname):
                frames.append(candidate)
                alignments.append(fname)
                continue
            fname = f"{os.path.splitext(candidate)[0]}_alignments.fsa"
            if os.path.isfile(candidate) and os.path.exists(fname):
                frames.append(candidate)
                alignments.append(fname)
                continue
            logger.warning("Can't locate alignments file for '%s'. Skipping.", candidate)

        if not frames:
            logger.error("No valid videos or frames folders found in '%s'", self._args.frames_dir)
            sys.exit(1)

        if self._args.job not in self._requires_faces:  # faces not required for frames input
            faces: list[str | None] = [None for _ in range(len(frames))]
        else:
            if not self._args.faces_dir:
                logger.error("Please provide a 'faces_dir' location for '%s' job", self._args.job)
                sys.exit(1)
            faces = [os.path.join(self._args.faces_dir, os.path.basename(os.path.splitext(frm)[0]))
                     for frm in frames]

        logger.info("Batch mode selected. Processing frames: %s",
                    [os.path.basename(frame) for frame in frames])

        return {"alignments_file": T.cast(list[str | None], alignments),
                "frames_dir": T.cast(list[str | None], frames),
                "faces_dir": faces}

    def _get_locations(self) -> dict[str, list[str | None]]:
        """ Obtain the full path to any frame, face and alignments input locations for the
        selected job when running in batch mode. If not running in batch mode, then the original
        passed in values are returned in lists

        Returns
        -------
        dict[str, list[Optional[str]]]
            A dictionary corresponding to the alignments, frames_dir and faces_dir arguments
            with a list of full paths for each job
        """
        job: str = self._args.job
        if not self._batch_mode:  # handle with given arguments
            retval = {"alignments_file": [self._args.alignments_file],
                      "faces_dir": [self._args.faces_dir],
                      "frames_dir": [self._args.frames_dir]}

        elif job in self._requires_alignments:  # Jobs only requiring an alignments file location
            retval = self._get_alignments_locations()

        elif job in self._requires_frames:  # Jobs that require a frames folder
            retval = self._get_frames_locations()

        elif job in self._requires_faces and job not in self._requires_frames:
            # Jobs that require faces as input
            faces = [os.path.join(self._args.faces_dir, folder)
                     for folder in os.listdir(self._args.faces_dir)
                     if os.path.isdir(os.path.join(self._args.faces_dir, folder))]
            if not faces:
                logger.error("No folders found in '%s'", self._args.faces_dir)
                sys.exit(1)

            retval = {"faces_dir": faces,
                      "frames_dir": [None for _ in range(len(faces))],
                      "alignments_file": [None for _ in range(len(faces))]}
            logger.info("Batch mode selected. Processing faces: %s",
                        [os.path.basename(folder) for folder in faces])
        else:
            raise FaceswapError(f"Unhandled job: {self._args.job}. This is a bug. Please report "
                                "to the developers")

        logger.debug("File locations: %s", retval)
        return retval

    @staticmethod
    def _run_process(arguments) -> None:
        """ The alignements tool process to be run in a spawned process.

        In some instances, batch-mode memory leaks. Launching each job in a separate process
        prevents this leak.

        Parameters
        ----------
        arguments: :class:`argparse.Namespace`
            The :mod:`argparse` arguments to be used for the given job
        """
        logger.debug("Starting process: (arguments: %s)", arguments)
        tool = _Alignments(arguments)
        tool.process()
        logger.debug("Finished process: (arguments: %s)", arguments)

    def process(self):
        """ The entry point for the Alignments tool from :mod:`lib.tools.alignments.cli`.

        Launches the selected alignments job.
        """
        num_jobs = len(self._locations["frames_dir"])
        for idx, (frames, faces, alignments) in enumerate(zip(self._locations["frames_dir"],
                                                              self._locations["faces_dir"],
                                                              self._locations["alignments_file"])):
            if num_jobs > 1:
                logger.info("Processing job %s of %s", idx + 1, num_jobs)

            args = Namespace(**self._args.__dict__)
            args.frames_dir = frames
            args.faces_dir = faces
            args.alignments_file = alignments

            if num_jobs > 1:
                proc = Process(target=self._run_process, args=(args, ))
                proc.start()
                proc.join()
            else:
                self._run_process(args)


class _Alignments():
    """ The main entry point for Faceswap's Alignments Tool. This tool is part of the Faceswap
    Tools suite and should be called from the ``python tools.py alignments`` command.

    The tool allows for manipulation, and working with Faceswap alignments files.

    Parameters
    ----------
    arguments: :class:`argparse.Namespace`
        The :mod:`argparse` arguments as passed in from :mod:`tools.py`
    """
    def __init__(self, arguments: Namespace) -> None:
        logger.debug("Initializing %s: (arguments: '%s'", self.__class__.__name__, arguments)
        self._args = arguments
        job = self._args.job

        if job == "from-faces":
            self.alignments = None
        else:
            self.alignments = AlignmentData(self._find_alignments())

        if (self.alignments is not None and
                arguments.frames_dir and
                os.path.isfile(arguments.frames_dir)):
            self.alignments.update_legacy_has_source(os.path.basename(arguments.frames_dir))

        logger.debug("Initialized %s", self.__class__.__name__)

    def _find_alignments(self) -> str:
        """ If an alignments folder is required and hasn't been provided, scan for a file based on
        the video folder.

        Exits if an alignments file cannot be located

        Returns
        -------
        str
            The full path to an alignments file
        """
        fname = self._args.alignments_file
        frames = self._args.frames_dir
        if fname and os.path.isfile(fname) and os.path.splitext(fname)[-1].lower() == ".fsa":
            return fname
        if fname:
            logger.error("Not a valid alignments file: '%s'", fname)
            sys.exit(1)

        if not frames or not os.path.exists(frames):
            logger.error("Not a valid frames folder: '%s'. Can't scan for alignments.", frames)
            sys.exit(1)

        fname = "alignments.fsa"
        if os.path.isdir(frames) and os.path.exists(os.path.join(frames, fname)):
            return fname

        if os.path.isdir(frames) or os.path.splitext(frames)[-1] not in VIDEO_EXTENSIONS:
            logger.error("Can't find a valid alignments file in location: %s", frames)
            sys.exit(1)

        fname = f"{os.path.splitext(frames)[0]}_{fname}"
        if not os.path.exists(fname):
            logger.error("Can't find a valid alignments file for video: %s", frames)
            sys.exit(1)

        return fname

    def process(self) -> None:
        """ The entry point for the Alignments tool from :mod:`lib.tools.alignments.cli`.

        Launches the selected alignments job.
        """
        if self._args.job in ("missing-alignments", "missing-frames", "multi-faces", "no-faces"):
            job: T.Any = Check
        else:
            job = globals()[self._args.job.title().replace("-", "")]
        job = job(self.alignments, self._args)
        logger.debug(job)
        job.process()
