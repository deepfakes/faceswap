#!/usr/bin/env python3
""" Image output writer for faceswap.py converter """

import cv2
from ._base import Writer, logger


class Output(Writer):
    """ Images output writer using cv2 """

    def write(self, filename, image):
        logger.trace("Outputting: (filename: '%s', shape: %s", filename, image.shape)
        try:
            cv2.imwrite(filename, image)  # pylint: disable=no-member
        except Exception as err:  # pylint: disable=broad-except
            logger.error("Failed to save image '%s'. Original Error: %s", filename, err)

    def close(self):
        """ Image writer does not need a close method """
        return
