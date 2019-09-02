#!/us#!/usr/bin/env python3

import cv2
import keras
import numpy as np
from ._base import Masker, logger


class Mask(Masker):
    """ Perform transformation to align and get landmarks """
    def __init__(self, **kwargs):
        git_model_id = 6
        model_filename = "DFL_256_sigmoid_v1.h5"
        super().__init__(git_model_id=git_model_id,
                         model_filename=model_filename,
                         **kwargs)
        self.vram = 3000
        self.min_vram = 1024
        self.input_size = 256
        self.model = None
        self.supports_plaidml = True

    def initialize(self, *args, **kwargs):
        """ Initialization tasks to run prior to alignments """
        try:
            super().initialize(*args, **kwargs)
            logger.info("Initializing U-Net Mask Network(256)...")
            logger.debug("U-Net initialize: (args: %s kwargs: %s)", args, kwargs)
            self.model = keras.models.load_model(self.model_path)
            self.init.set()
            logger.info("Initialized U-Net Mask Network(256)")
        except Exception as err:
            self.error.set()
            raise err

    # MASK PROCESSING
    def build_masks(self, image, detected_face):
        """ Function for creating facehull masks
            Faces may be of shape (batch_size, height, width, 3) or (height, width, 3)
        """
        # pylint: disable=no-member
        postprocess_test = False
        image = np.array(image)
        detected_face.load_aligned(image, size=self.input_size, align_eyes=False, dtype='float32')
        feed_face = detected_face.aligned["face"]
        mask = np.zeros(feed_face.shape[:-1] + (1, ), dtype='float32')
        model_input = feed_face / 255.
        
        results = self.model.predict_on_batch(model_input[None, :, :, :3])
        generator = (cv2.GaussianBlur(mask, (7, 7), 0) for mask in results)
        if postprocess_test:
            generator = (self.postprocessing(mask[..., None]) for mask in results)
        results = np.array(tuple(generator))[..., None]
        results[results < 0.05] = 0.
        results[results > 0.95] = 1.
        results *= 255.

        detected_face.aligned.clear()
        detected_face.load_aligned(image, size=self.crop_size, align_eyes=False)
        output_face = detected_face.aligned["face"]
        resized_masked = self.resize_inputs(results, self.crop_size).astype('uint8')
        resized_masked = np.squeeze(resized_masked, axis=0)
        masked_img = np.concatenate((output_face[..., :3], resized_masked), axis=-1)
        detected_face.aligned["face"] = masked_img
        return detected_face

