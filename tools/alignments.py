#!/usr/bin/env python3
""" Tools for manipulating the alignments seralized file """
import logging

from lib.utils import set_system_verbosity
from .lib_alignments import (AlignmentData, Check, Draw, # noqa pylint: disable=unused-import
                             Extract, Legacy, Manual, Merge, Reformat, Rename,
                             RemoveAlignments, Sort, Spatial, UpdateHashes)

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class Alignments():
    """ Perform tasks relating to alignments file """
    def __init__(self, arguments):
        logger.debug("Initializing %s: (arguments: '%s'", self.__class__.__name__, arguments)
        self.args = arguments
        set_system_verbosity(self.args.loglevel)
        self.alignments = self.load_alignments()
        logger.debug("Initialized %s", self.__class__.__name__)

    def load_alignments(self):
        """ Loading alignments """
        logger.debug("Loading alignments")
        if len(self.args.alignments_file) > 1 and self.args.job != "merge":
            logger.error("Multiple alignments files are only permitted for merging")
            exit(0)
        if len(self.args.alignments_file) == 1 and self.args.job == "merge":
            logger.error("More than one alignments file required for merging")
            exit(0)

        dest_format = self.get_dest_format()
        if len(self.args.alignments_file) == 1:
            retval = AlignmentData(self.args.alignments_file[0], dest_format)
        else:
            retval = [AlignmentData(a_file, dest_format) for a_file in self.args.alignments_file]
        logger.debug("Alignments: %s", retval)
        return retval

    def get_dest_format(self):
        """ Set the destination format for Alignments """
        dest_format = None
        if hasattr(self.args, 'alignment_format') and self.args.alignment_format:
            dest_format = self.args.alignment_format
        logger.debug(dest_format)
        return dest_format

    def process(self):
        """ Main processing function of the Align tool """
        if self.args.job.startswith("extract"):
            job = Extract
        elif self.args.job == "update-hashes":
            job = UpdateHashes
        elif self.args.job.startswith("remove-"):
            job = RemoveAlignments
        elif self.args.job.startswith("sort-"):
            job = Sort
        elif self.args.job in("missing-alignments", "missing-frames",
                              "multi-faces", "leftover-faces", "no-faces"):
            job = Check
        else:
            job = globals()[self.args.job.title()]
        job = job(self.alignments, self.args)
        logger.debug(job)
        job.process()
