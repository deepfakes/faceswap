#!/usr/bin/env python3
""" Default configurations for extract """

import logging

from lib.config import FaceswapConfig

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class Config(FaceswapConfig):
    """ Config File for Models """

    def set_defaults(self):
        """ Set the default values for config """
        logger.debug("Setting defaults")

        # << GLOBAL OPTIONS >> #
#        section = "global"
#        self.add_section(title=section,
#                         info="Options that apply to all models")

        # << MTCNN DETECTOR OPTIONS >> #
        section = "detect.mtcnn"
        self.add_section(title=section,
                         info="MTCNN Detector options")
        self.add_item(
            section=section, title="minsize", datatype=int, default=20, rounding=10,
            min_max=(20, 1000),
            info="The minimum size of a face (in pixels) to be accepted as a positive match.\n"
                 "Lower values use significantly more VRAM and will detect more false positives")
        self.add_item(
            section=section, title="threshold_1", datatype=float, default=0.6, rounding=2,
            min_max=(0.1, 0.9),
            info="First stage threshold for face detection. This stage obtains face candidates")
        self.add_item(
            section=section, title="threshold_2", datatype=float, default=0.7, rounding=2,
            min_max=(0.1, 0.9),
            info="Second stage threshold for face detection. This stage refines face candidates")
        self.add_item(
            section=section, title="threshold_3", datatype=float, default=0.7, rounding=2,
            min_max=(0.1, 0.9),
            info="Third stage threshold for face detection. This stage further refines face "
                 "candidates")
        self.add_item(
            section=section, title="scalefactor", datatype=float, default=0.709, rounding=3,
            min_max=(0.1, 0.9),
            info="The scale factor for the image pyramid")

        # << S3FD DETECTOR OPTIONS >> #
        section = "detect.s3fd"
        self.add_section(title=section,
                         info="S3FD Detector options")
        self.add_item(
            section=section, title="confidence", datatype=int, default=50, rounding=5,
            min_max=(25, 100),
            info="The confidence level at which the detector has succesfully found a face.\n"
                 "Higher levels will be more discriminating, lower levels will have more false "
                 "positives")
