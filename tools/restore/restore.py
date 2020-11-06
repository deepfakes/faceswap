#!/usr/bin/env python3
""" Tool to restore models from backup """

import logging
import os
import sys

from lib.model.backup_restore import Backup

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class Restore():
    """ Restore a model from backup """

    def __init__(self, arguments):
        logger.debug("Initializing %s: (arguments: '%s'", self.__class__.__name__, arguments)
        self.model_dir = arguments.model_dir
        self.model_name = None

    def process(self):
        """ Perform the Restore process """
        logger.info("Starting Model Restore...")
        self.validate()
        backup = Backup(self.model_dir, self.model_name)
        backup.restore()
        logger.info("Completed Model Restore")

    def validate(self):
        """ Make sure there is only one model in the target folder """
        if not os.path.exists(self.model_dir):
            logger.error("Folder does not exist: '%s'", self.model_dir)
            sys.exit(1)
        chkfiles = [fname for fname in os.listdir(self.model_dir) if fname.endswith("_state.json")]
        bkfiles = [fname for fname in os.listdir(self.model_dir) if fname.endswith(".bk")]
        if not chkfiles:
            logger.error("Could not find a model in the supplied folder: '%s'", self.model_dir)
            sys.exit(1)
        if len(chkfiles) > 1:
            logger.error("More than one model found in the supplied folder: '%s'", self.model_dir)
            sys.exit(1)
        if not bkfiles:
            logger.error("Could not find any backup files in the supplied folder: '%s'",
                         self.model_dir)
            sys.exit(1)
        self.model_name = chkfiles[0].replace("_state.json", "")
        logger.info("%s Model found", self.model_name.title())
        logger.verbose("Backup files: %s)", bkfiles)
