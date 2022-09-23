#!/usr/bin/env python3
""" Tools for manipulating the alignments serialized file """
import logging
import os
import sys

from typing import Any, TYPE_CHECKING

from lib.utils import _video_extensions
from .media import AlignmentData
from .jobs import (Check, Draw, Extract, FromFaces, Rename,  # noqa pylint: disable=unused-import
                   RemoveFaces, Sort, Spatial)


if TYPE_CHECKING:
    from argparse import Namespace

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class Alignments():  # pylint:disable=too-few-public-methods
    """ The main entry point for Faceswap's Alignments Tool. This tool is part of the Faceswap
    Tools suite and should be called from the ``python tools.py alignments`` command.

    The tool allows for manipulation, and working with Faceswap alignments files.

    Parameters
    ----------
    arguments: :class:`argparse.Namespace`
        The :mod:`argparse` arguments as passed in from :mod:`tools.py`
    """
    def __init__(self, arguments: "Namespace") -> None:
        logger.debug("Initializing %s: (arguments: '%s'", self.__class__.__name__, arguments)
        self._args = arguments
        job = self._args.job

        if job == "from-faces":
            self.alignments = None
        else:
            self.alignments = AlignmentData(self._find_alignments())

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

        if os.path.isdir(frames) or os.path.splitext(frames)[-1] not in _video_extensions:
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
            job: Any = Check
        else:
            job = globals()[self._args.job.title().replace("-", "")]
        job = job(self.alignments, self._args)
        logger.debug(job)
        job.process()
