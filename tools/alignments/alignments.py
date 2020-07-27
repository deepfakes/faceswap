#!/usr/bin/env python3
""" Tools for manipulating the alignments serialized file """
import sys
import logging

from .media import AlignmentData
from .jobs import (Check, Dfl, Draw, Extract, Fix, Merge,  # noqa pylint: disable=unused-import
                   Rename, RemoveAlignments, Sort, Spatial, UpdateHashes)
from .jobs_manual import Manual  # noqa pylint: disable=unused-import

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
    def __init__(self, arguments):
        logger.debug("Initializing %s: (arguments: '%s'", self.__class__.__name__, arguments)
        self.args = arguments
        self.alignments = self._load_alignments()
        logger.debug("Initialized %s", self.__class__.__name__)

    def _load_alignments(self):
        """ Loads the given alignments file(s) prior to running the selected job.

        Returns
        -------
        :class:`~tools.alignments.media.AlignmentData` or list
            The alignments data formatted for use by the alignments tool. If multiple alignments
            files have been selected, then this will be a list of
            :class:`~tools.alignments.media.AlignmentData` objects
        """
        logger.debug("Loading alignments")
        if len(self.args.alignments_file) > 1 and self.args.job != "merge":
            logger.error("Multiple alignments files are only permitted for merging")
            sys.exit(0)
        if len(self.args.alignments_file) == 1 and self.args.job == "merge":
            logger.error("More than one alignments file required for merging")
            sys.exit(0)

        if len(self.args.alignments_file) == 1:
            retval = AlignmentData(self.args.alignments_file[0])
        else:
            retval = [AlignmentData(a_file) for a_file in self.args.alignments_file]
        logger.debug("Alignments: %s", retval)
        return retval

    def process(self):
        """ The entry point for the Alignments tool from :mod:`lib.tools.alignments.cli`.

        Launches the selected alignments job.
        """
        if self.args.job == "manual":
            logger.warning("The 'manual' job is deprecated and will be removed from a future "
                           "update. Please use the new 'manual' tool.")
        if self.args.job == "update-hashes":
            job = UpdateHashes
        elif self.args.job.startswith("remove-"):
            job = RemoveAlignments
        elif self.args.job in ("missing-alignments", "missing-frames",
                               "multi-faces", "leftover-faces", "no-faces"):
            job = Check
        else:
            job = globals()[self.args.job.title()]
        job = job(self.alignments, self.args)
        logger.debug(job)
        job.process()
