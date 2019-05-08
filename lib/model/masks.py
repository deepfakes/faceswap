#!/usr/bin/env python3
""" Masks functions for faceswap.py """

import logging
from pathlib import Path
import cv2
import keras
import numpy as np
from lib.utils import GetModel

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

def get_available_masks():
    """ Return a list of the available masks for cli """
    masks = ["components", "dfl_full", "facehull", "vgg_300", "vgg_500", "unet_256", "none"]
    logger.debug(masks)
    return masks

def get_default_mask():
    """ Set the default mask for cli """
    default = "dfl_full"
    logger.debug("Default mask is %s", default)
    return default

class Mask():
    """ Parent class for masks

        the output mask will be <mask_type>.mask
        channels: 1, 3 or 4:
                    1 - Returns a single channel mask
                    3 - Returns a 3 channel mask
                    4 - Returns the original image with the mask in the alpha channel """

    def __init__(self, mask_type, channels=4):
        logger.trace("Initializing %s: (mask_type: %s, channels: %s)",
                     self.__class__.__name__, mask_type, channels)
        assert channels in (1, 3, 4), "Channels should be 1, 3 or 4"
        self.mask_type = mask_type
        self.channels = channels
        self.mask_function = self.build_function()
        logger.trace("Initialized %s", self.__class__.__name__)

    def build_function(self):
        """ Build the mask function """
        build_dict = {"facehull":    (self.facehull, 1),
                      "dfl_full":    (self.facehull, 3),
                      "components":  (self.facehull, 8),
                      None:          (self.facehull, 3),
                      "vgg_300":     (self.smart, "Nirkin_300_softmax_v1.h5"),
                      "vgg_500":     (self.smart, "Nirkin_500_softmax_v1.h5"),
                      "unet_256":    (self.smart, "DFL_256_sigmoid_v1.h5"),
                      "none":        (self.dummy, None)}
        build_function, arg = build_dict[self.mask_type]
        mask_function = build_function(arg)
        return mask_function

    def merge_masks(self, faces, masks):
        """ Return the masks in the requested shape """
        logger.trace("faces_shape: %s, masks_shape: %s", faces.shape, masks.shape)
        masks = np.squeeze(masks, axis=0) if masks.shape[0] == 1 else masks
        if self.channels == 3:
            retval = np.repeat(masks, 3, axis=-1)
        elif self.channels == 4:
            retval = np.concatenate((faces, masks), axis=-1)
        else:
            retval = masks
        logger.trace("Final masks shape: %s", retval.shape)
        return retval

    def mask(self, faces, landmarks):
        logger.trace("Masking: (faces_shape: %s)",faces.shape)
        masks = self.mask_function(faces, landmarks)
        merged_masks = self.merge_masks(faces, masks)
        return merged_masks

    @staticmethod
    def facehull(part_number):
        """ Compute the facehull """

        @staticmethod
        def one():
            """ Basic facehull mask """
            parts = [(self.landmarks)]
            return parts

        @staticmethod
        def three():
            """ DFL facehull mask """
            nose_ridge = (self.landmarks[27:31], self.landmarks[33:34])
            jaw = (self.landmarks[0:17],
                   self.landmarks[48:68],
                   self.landmarks[0:1],
                   self.landmarks[8:9],
                   self.landmarks[16:17])
            eyes = (self.landmarks[17:27],
                    self.landmarks[0:1],
                    self.landmarks[27:28],
                    self.landmarks[16:17],
                    self.landmarks[33:34])
            parts = [jaw, nose_ridge, eyes]
            return parts

        @staticmethod
        def eight():
            """ Component facehull mask """
            r_jaw = (self.landmarks[0:9], self.landmarks[17:18])
            l_jaw = (self.landmarks[8:17], self.landmarks[26:27])
            r_cheek = (self.landmarks[17:20], self.landmarks[8:9])
            l_cheek = (self.landmarks[24:27], self.landmarks[8:9])
            nose_ridge = (self.landmarks[19:25], self.landmarks[8:9],)
            r_eye = (self.landmarks[17:22],
                     self.landmarks[27:28],
                     self.landmarks[31:36],
                     self.landmarks[8:9])
            l_eye = (self.landmarks[22:27],
                     self.landmarks[27:28],
                     self.landmarks[31:36],
                     self.landmarks[8:9])
            nose = (self.landmarks[27:31], self.landmarks[31:36])
            parts = [r_jaw, l_jaw, r_cheek, l_cheek, nose_ridge, r_eye, l_eye, nose]
            return parts

        part_dict = {1: one, 3: three, 8: eight}
        part_function = part_dict[part_number]

        @staticmethod
        def mask_function(faces, landmarks):
            """
            Function for creating facehull masks
            Faces may be of shape (batch_size, height, width, 3) or (height, width, 3)
            Landmarks may be of shape (batch_size, 68, 2) or (68, 2)
            """
            masks = np.array(np.zeros(faces.shape[:-1] + (1, )), dtype='float32', ndim=4)
            parts = part_function(landmark)
            for mask in masks:
                for item in parts:
                    # pylint: disable=no-member
                    hull = cv2.convexHull(np.concatenate(item))
                    cv2.fillConvexPoly(mask, hull, 1., lineType=cv2.LINE_AA)
            return masks

        return mask_function

    @staticmethod
    def smart(model_type):
        """ Compute the facehull """

        @staticmethod
        def get_model(model_filename):
            """ Check if model is available, if not, download and unzip it """
            cache_path = os.path.join(os.path.dirname(__file__), ".cache")
            model = GetModel(model_filename, cache_path)
            return model.model_path

        model_path = get_model(model_type)
        model = keras.models.load_model(model_path)

        #TODO finish here

        return mask_function

    @staticmethod
    def dummy(dummy):
        """ Basic facehull mask """
        def mask_function(faces, landmarks):
            masks = np.ones_like(faces)
            return masks

        return mask_function

