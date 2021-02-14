#!/usr/bin/env python3
""" Tools for manipulating the alignments serialized file """
import logging

from .media import AlignmentData
from .jobs import (Check, Draw, Extract, Rename,  # noqa pylint: disable=unused-import
                   RemoveFaces, Sort, Spatial)

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
        self.alignments = AlignmentData(self.args.alignments_file)
        logger.debug("Initialized %s", self.__class__.__name__)

    def process(self):
        """ The entry point for the Alignments tool from :mod:`lib.tools.alignments.cli`.

        Launches the selected alignments job.
        """
        if self.args.job in ("missing-alignments", "missing-frames", "multi-faces", "no-faces"):
            job = Check
        else:
            job = globals()[self.args.job.title().replace("-", "")]
        job = job(self.alignments, self.args)
        logger.debug(job)
        job.process()
