#!/usr/bin/env python3
""" Masks functions for faceswap.py """

import logging

from pathlib import Path
import cv2
import os
import keras
import numpy as np
from lib.utils import GetModel

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def get_available_masks():
    """ Return a list of the available masks for cli """
    masks = ["components", "dfl_full", "facehull", "none", "vgg_300", "vgg_500", "unet_256"]
    logger.debug(masks)
    return masks

def get_default_mask():
    """ Set the default mask for cli """
    masks = get_available_masks()
    default = "dfl_full" if "dfl_full" in masks else masks[0]
    logger.debug("Default mask is %s", default)
    return default

class Mask():
    """ Parent class for masks
        Faces may be of shape (batch_size, height, width, 3) or (height, width, 3) 
        of dtype float32 and with range[0., 1.]
        Landmarks may be of shape (batch_size, 68, 2) or (68, 2)
        Produced mask will be in range [0, 1.]
        the output mask will be <mask_type>.mask
        channels: 1, 3 or 4:
                    1 - Returns a single channel mask
                    3 - Returns a 3 channel mask
                    4 - Returns the original image with the mask in the alpha channel """

    def __init__(self, mask_type, faces, landmarks, channels=4):
        logger.trace("Initializing %s: (mask_type: %s, channels: %s)",
                     self.__class__.__name__, mask_type, channels)
        assert channels in (1, 3, 4), "Channels should be 1, 3 or 4"
        self.mask_type = mask_type
        self.channels = channels
        masks = self.build_masks(mask_type, faces, landmarks)
        self.masks = self.merge_masks(faces, masks)
        logger.trace("Initialized %s", self.__class__.__name__)

    def build_masks(self, mask_type, faces, landmarks):
        """ Override to build the mask """
        raise NotImplementedError

    def merge_masks(self, faces, masks):
        """ Return the masks in the requested shape """
        logger.trace("faces_shape: %s, masks_shape: %s", faces.shape, masks.shape)
        if self.channels == 3:
            retval = np.repeat(masks, 3, axis=-1)
        elif self.channels == 4:
            retval = np.concatenate((faces[..., :3], masks), axis=-1)
        else:
            retval = masks
        logger.trace("Final masks shape: %s", retval.shape)
        return retval


class Facehull(Mask):

    def one(self, landmarks):
        """ Basic facehull mask """
        parts = [(landmarks)]
        return parts

    def three(self, landmarks):
        """ DFL facehull mask """
        nose_ridge = (landmarks[27:31], landmarks[33:34])
        jaw = (landmarks[0:17],
               landmarks[48:68],
               landmarks[0:1],
               landmarks[8:9],
               landmarks[16:17])
        eyes = (landmarks[17:27],
                landmarks[0:1],
                landmarks[27:28],
                landmarks[16:17],
                landmarks[33:34])
        parts = [jaw, nose_ridge, eyes]
        return parts

    def eight(self, landmarks):
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

    def build_masks(self, mask_type, faces, landmarks):
        """
        Function for creating facehull masks
        Faces may be of shape (batch_size, height, width, 3) or (height, width, 3)
        Landmarks may be of shape (batch_size, 68, 2) or (68, 2)
        """
        build_dict = {"facehull":    self.one,
                      "dfl_full":    self.three,
                      "components":  self.eight,
                      None:          self.three}
        masks = np.array(np.zeros(faces.shape[:-1] + (1,)), dtype='float32', ndmin=4)
        if landmarks.ndim == 2:
            landmarks = landmarks[None, ...]
        for i, landmark in enumerate(landmarks):
            parts = build_dict[mask_type](landmark)
            for item in parts:
                # pylint: disable=no-member
                hull = cv2.convexHull(np.concatenate(item))
                try:
                    cv2.fillConvexPoly(masks[i], hull, 1.)
                except:
                    print("cv2 error")
        return masks


class Smart(Mask):

    def build_masks(self, mask_type, faces, landmarks=None):
        """
        Function for creating facehull masks
        Faces may be of shape (batch_size, height, width, 3) or (height, width, 3)
        Check if model is available, if not, download and unzip it
        """

        build_dict = {"vgg_300":     "Nirkin_300_softmax_v1.h5",
                      "vgg_500":     "Nirkin_500_softmax_v1.h5",
                      "unet_256":    "DFL_256_sigmoid_v1.h5",
                      None:          "Nirkin_500_softmax_v1.h5"}
        model_name = build_dict[mask_type]
        cache_path = os.path.join(os.path.dirname(__file__), ".cache")
        model = GetModel(model_name, cache_path)
        model = keras.models.load_model(model_path.model_path)
        
        masks = np.array(np.zeros(faces.shape[:-1] + (1, )), dtype='float32', ndim=4)

        # TODO finish here
        def segment(self, model_type, out_dir='C:/data/masked/'):
            """ doc string """
            i = 0
            for img_batch in self.memmapped_images:
                if  model_type=='DFL':
                    model_input = (faces - self.means)
                    masks = self.model.predict(model_input)
                    low = masks < 0.1
                    masks[low] = 0.
                if model_type=='Nirkin':
                    model_input = (faces - self.means)
                    masks = self.model.predict_on_batch(model_input)[..., 1:2]
                    generator = (cv2.GaussianBlur(mask, (7,7), 0) for mask in masks)
                    # generator = (self.postprocessing(mask[:, :, None]) for mask in masks)
                    masks = np.array(tuple(generator))[..., None]
                    low = masks < 0.01
                    high = masks > 0.975
                    results[masks] = 0.
                    results[masks] = 1.

            return masks

        @staticmethod
        def postprocessing(mask):
            # pylint: disable=no-member
            """ doc string """
            #Select_largest_segment
            pop_small_segments = False # Don't do this right now
            if pop_small_segments:
                results = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)
                _, labels, stats, _ = results
                segments_ranked_by_area = np.argsort(stats[:, -1])[::-1]
                mask[labels != segments_ranked_by_area[0, 0]] = 0.

            #Smooth contours
            smooth_contours = False # Don't do this right now
            if smooth_contours:
                iters = 2
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
                cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=iters)
                cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iters)
                cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iters)
                cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=iters)

            #Fill holes
            fill_holes = True
            if fill_holes:
                not_holes = mask.copy()
                not_holes = np.pad(not_holes, ((2, 2), (2, 2), (0, 0)), 'constant')
                cv2.floodFill(not_holes, None, (0, 0), 255)
                holes = cv2.bitwise_not(not_holes)[2:-2,2:-2]
                mask = cv2.bitwise_or(mask, holes)
                mask = np.expand_dims(mask, axis=-1)
                
            return mask

        return masks


class Dummy(Mask):

    def build_masks(self, mask_type, faces, landmarks=None):
        """ Dummy mask of all ones """
        masks = np.ones_like(faces)
        return masks

