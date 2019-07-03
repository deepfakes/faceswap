#!/usr/bin/env python3
""" Installs any required third party libs for faceswap.py

    Checks for installed Conda / Pip packages and updates accordingly
"""

from setup import Environment, Install, Output

_LOGGER = None


def output(msg):
    """ Output to print or logger """
    if _LOGGER is not None:
        _LOGGER.info(msg)
    else:
        Output().info(msg)


def main(logger=None):
    """ Check for and update dependencies """
    if logger is not None:
        global _LOGGER  # pylint:disable=global-statement
        _LOGGER = logger
    output("Updating dependencies...")
    update = Environment(logger=logger, updater=True)
    Install(update)
    output("Dependencies updated")


if __name__ == "__main__":
    main()
