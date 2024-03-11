#!/usr/bin/env python3
""" Import mask processing for faceswap's mask tool """
from __future__ import annotations

import logging
import os
import sys

from lib.image import ImagesLoader

logger = logging.getLogger(__name__)


class Import:  # pylint:disable=too-few-public-methods
    """ Import masks from disk into an Alignments file """
    def __init__(self, import_path: str) -> None:
        logger.debug("Initializing %s (import_path: %s)", self.__class__.__name__, import_path)
        self._loader = self._get_loader(import_path)
        logger.debug("Initialized %s", self.__class__.__name__)

    @classmethod
    def _get_loader(cls, import_path: str | None) -> ImagesLoader:
        """ Get an images loader for the masks to import

        Parameters
        ----------
        import_path: str | None
            Full path to the folder where the masks are to be imported from. ``None`` if it has not
            been provided

        Returns
        -------
        :class:`lib.image.ImagesLoader`
            Images loader instance for loading the masks
        """
        if import_path is None:
            logger.error("A mask input location must be provided when importing masks.")
            sys.exit(1)
        if not os.path.isdir(import_path):
            logger.error("Mask input location cannot be found: '%s'", import_path)
            sys.exit(1)

        retval = ImagesLoader(import_path)

        if retval.is_video:
            logger.error("Mask input must be a folder of mask images, not a video: '%s'",
                         import_path)
            sys.exit(1)
        if retval.count < 1:
            logger.error("No image files located: '%s'", import_path)
            sys.exit(1)

        return retval
