#!/usr/bin/env python3
"""Installs required third-party libraries for faceswap.py.

    Checks for installed Conda/Pip packages and updates accordingly.
"""
import logging
import os
import sys

from lib.logger import log_setup
from setup import Environment, Install

logger = logging.getLogger(__name__)

def main(is_gui=False) -> None:
    """Check for and update dependencies.

    Parameters
    ----------
    is_gui: bool, optional
        ``True`` if being called by the GUI. Prevents the updater from outputting progress bars
        which can get scrambled in the GUI.
    """
    try:
        logger.info("Starting dependency update...")

        # Initialize Environment and Install classes
        update = Environment(updater=True)
        Install(update, is_gui=is_gui)

        logger.info("Dependencies updated successfully.")
    except Exception as e:
        logger.error(f"An error occurred while updating dependencies: {e}")
        sys.exit(1)  # Exit with error code to indicate failure

if __name__ == "__main__":
    try:
        # Set up logging to file
        logfile = os.path.join(os.path.dirname(os.path.realpath(sys.argv[0])), "faceswap_update.log")
        log_setup("INFO", logfile, "setup")
        
        # Run the main function to update dependencies
        main()
    except Exception as e:
        logger.error(f"An error occurred during setup: {e}")
        sys.exit(1)  # Exit with error code
