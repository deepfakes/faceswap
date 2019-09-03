#!/usr/bin/env python3

import cv2
import numpy as np

from ._base import Masker, logger


class Mask(Masker):
    """ Perform transformation to align and get landmarks """
    def __init__(self, **kwargs):
        git_model_id = None
        model_filename = None
        super().__init__(git_model_id=git_model_id,
                         model_filename=model_filename,
                         **kwargs)
        self.vram = 0
        self.model = None
        self.supports_plaidml = True

    def initialize(self, *args, **kwargs):
        """ Initialization tasks to run prior to alignments """
        try:
            super().initialize(*args, **kwargs)
            logger.info("Initializing Components Mask Model...")
            logger.debug("Components initialize: (args: %s kwargs: %s)", args, kwargs)
            self.init.set()
            logger.info("Initialized Components Mask Model")
        except Exception as err:
            self.error.set()
            raise err

    # MASK PROCESSING
    def build_masks(self, image, detected_face):
        """ Function for creating facehull masks
            Faces may be of shape (batch_size, height, width, 3)
        """
        image = np.array(image)
        landmarks = np.array(detected_face.landmarksXY)
        mask = np.zeros(image.shape[:-1] + (1,), dtype='uint8')
        parts = self.parse_parts(landmarks)
        for item in parts:
            item  = np.concatenate(item)
            hull = cv2.convexHull(item).astype("int32")  # pylint: disable=no-member
            cv2.fillConvexPoly(mask, hull, 255)  # pylint: disable=no-member
        masked_img = np.concatenate((image[..., :3], mask), axis=-1)
        detected_face.load_aligned(masked_img, size=self.crop_size, align_eyes=False)
        return detected_face

    @staticmethod
    def parse_parts(landmarks):
        """ Component facehull mask """
        r_jaw = (landmarks[0:9], landmarks[17:18])
        l_jaw = (landmarks[8:17], landmarks[26:27])
        r_cheek = (landmarks[17:20], landmarks[8:9])
        l_cheek = (landmarks[24:27], landmarks[8:9])
        nose_ridge = (landmarks[19:25], landmarks[8:9],)
        r_eye = (landmarks[17:22],
                 landmarks[27:28],
                 landmarks[31:36],
                 landmarks[8:9])
        l_eye = (landmarks[22:27],
                 landmarks[27:28],
                 landmarks[31:36],
                 landmarks[8:9])
        nose = (landmarks[27:31], landmarks[31:36])
        parts = [r_jaw, l_jaw, r_cheek, l_cheek, nose_ridge, r_eye, l_eye, nose]
        return parts
