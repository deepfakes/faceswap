#!/usr/bin/env python3

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
            logger.info("Initializing Dummy Mask Model...")
            logger.debug("Dummy initialize: (args: %s kwargs: %s)", args, kwargs)
            self.init.set()
            logger.info("Initialized Dummy Mask Model")
        except Exception as err:
            self.error.set()
            raise err

    # MASK PROCESSING
    def build_masks(self, image, detected_face, input_size, output_size, coverage_ratio):
        """ Function for creating facehull masks
            Faces may be of shape (batch_size, height, width, 3) or (height, width, 3)
        """
        masks = np.full(image.shape[:-1] + (1,), fill_value=255, dtype='uint8')
        masked_img = np.concatenate((image[..., :3], masks), axis=-1)
        detected_face.image = masked_img
        detected_face.load_feed_face(masked_img,
                                     size=input_size,
                                     coverage_ratio=coverage_ratio)
        if input_size != output_size:
            detected_face.load_reference_face(masked_img,
                                              size=output_size,
                                              coverage_ratio=coverage_ratio)
        return detected_face
