#!/usr/bin/env python3
""" Installs any required third party libs for faceswap.py

    Checks for installed Conda / Pip packages and updates accordingly
"""
import logging
import os
import sys

from lib.logger import log_setup
from setup import Environment, Install

logger = logging.getLogger(__name__)


def main(is_gui=False) -> None:
    """ Check for and update dependencies

    Parameters
    ----------
    is_gui: bool, optional
        ``True`` if being called by the GUI. Prevents the updater from outputting progress bars
        which get scrambled in the GUI
    """
    logger.info("Updating dependencies...")
    update = Environment(updater=True)
    Install(update, is_gui=is_gui)
    logger.info("Dependencies updated")


if __name__ == "__main__":
    logfile = os.path.join(os.path.dirname(os.path.realpath(sys.argv[0])), "faceswap_update.log")
    log_setup("INFO", logfile, "setup")
    main()
