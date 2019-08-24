#!/usr/bin/env python3
""" Masks functions for faceswap.py """

import logging

import os
import sys
import cv2
import numpy as np
from lib.utils import GetModel

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def get_available_masks():
    """ Return a list of the available masks for cli """
    masks = ["none", "components", "extended", "vgg_300", "vgg_500", "unet_256"]
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

    def __init__(self, mask_type):
        logger.trace("Initializing %s: (mask_type: %s)", self.__class__.__name__, mask_type)
        self.mask_type = mask_type
        self.target_size, self.mask_model = self.build_model(mask_type)
        logger.trace("Initialized %s", self.__class__.__name__)

    def build_model(self, mask_type):
        """ Override to build the mask """
        raise NotImplementedError

    def build_masks(self, faces, means, landmarks):
        """ Override to build the mask """
        raise NotImplementedError
        
    def merge_masks(self, faces, masks, channels):
        """ Return the masks in the requested shape """
        logger.trace("faces_shape: %s, masks_shape: %s", faces.shape, masks.shape)
        if channels == 3:
            retval = np.repeat(masks, 3, axis=-1)
        elif channels == 4:
            retval = np.concatenate((faces[..., :3], masks), axis=-1)
        else:
            retval = masks
        logger.trace("Final masks shape: %s", retval.shape)
        return retval


class Facehull(Mask):
    """ Face masks designed from facehulls of facial landmark points """

    @staticmethod
    def one(landmarks):
        """ Basic facehull mask """
        parts = [(landmarks)]
        return parts

    @staticmethod
    def three(landmarks):
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

    @staticmethod
    def eight(landmarks):
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

    def extended(self, landmarks):
        """ Component facehull mask with forehead extended"""
        # mid points between the side of face and eye point
        ml_pnt = (landmarks[36] + landmarks[0]) // 2
        mr_pnt = (landmarks[16] + landmarks[45]) // 2

        # mid points between the mid points and eye
        ql_pnt = (landmarks[36] + ml_pnt) // 2
        qr_pnt = (landmarks[45] + mr_pnt) // 2

        # Top of the eye arrays
        bot_l = np.array((ql_pnt, landmarks[36], landmarks[37], landmarks[38], landmarks[39]))
        bot_r = np.array((landmarks[42], landmarks[43], landmarks[44], landmarks[45], qr_pnt))

        # Eyebrow arrays
        top_l = landmarks[17:22]
        top_r = landmarks[22:27]

        # Adjust eyebrow arrays
        landmarks[17:22] = top_l + ((top_l - bot_l) // 2)
        landmarks[22:27] = top_r + ((top_r - bot_r) // 2)

        parts = self.eight(landmarks)
        return parts

    def build_model(self, mask_type):
        """
        Function for creating facehull masks
        Faces may be of shape (batch_size, height, width, 3) or (height, width, 3)
        Landmarks may be of shape (batch_size, 68, 2) or (68, 2)
        """
        build_dict = {"facehull":    self.one,
                      "dfl_full":    self.three,
                      "components":  self.eight,
                      "extended":    self.extended,
                      None:          self.three}
        return None, build_dict[mask_type]

    def build_masks(self, faces, means, landmarks, channels):
                      
        if faces.ndim < 4:
            faces = np.expand_dims(faces, axis=0)
        masks = np.array(np.zeros(faces.shape[:-1] + (1,)), dtype='float32', ndmin=4)
        if landmarks.ndim == 2:
            landmarks = landmarks[None, ...]
        for i, landmark in enumerate(landmarks):
            parts = self.mask_model(landmark)
            for item in parts:
                # pylint: disable=no-member
                hull = cv2.convexHull(np.concatenate(item)).astype("int32")
                try:
                    cv2.fillConvexPoly(masks[i], hull, 1.)
                except Exception as error:
                    print("CV2 Error '{0}' occured.".format(error.message))
                    print("Error Arguments {1}.".format(error.args))
        return self.merge_masks(faces, masks, channels)


class Smart(Mask):
    """ Neural net trained segmentation masks for face areas """

    def build_model(self, mask_type):
        """ Check if model is available, if not, download and unzip it """
        import keras
        build_dict = {"vgg_300":     (300, 8, ["Nirkin_300_softmax_v1.h5"]),
                      "vgg_500":     (500, 5, ["Nirkin_500_softmax_v1.h5"]),
                      "unet_256":    (256, 6, ["DFL_256_sigmoid_v1.h5"]),
                      None:          (500, 5, ["Nirkin_500_softmax_v1.h5"])}
        input_size, git_model_id, model_filename = build_dict[mask_type]
        root_path = os.path.abspath(os.path.dirname(sys.argv[0]))
        cache_path = os.path.join(root_path, "plugins", "extract", ".cache")
        model = GetModel(model_filename, cache_path, git_model_id)
        with keras.backend.tf.device("/cpu:0"):
            mask_model = keras.models.load_model(model.model_path)
        print(mask_model.summary())
        return input_size, mask_model
    
    def build_masks(self, faces, means, landmarks, channels):
        """
        Function for creating facehull masks
        Faces may be of shape (batch_size, height, width, 3) or (height, width, 3)
        """
        import keras

        postprocess_test = False
        masks = np.array(np.zeros(faces.shape[:-1] + (1, )), dtype='float32', ndmin=4)
        original_size, faces = self.resize_inputs(faces, self.target_size)
        batch_size = min(faces.shape[0], 8)
        batches = ((faces.shape[0] - 1) // batch_size) + 1
        even_split_section = batches * batch_size
        faces_batched = np.split(faces[:even_split_section], batches)
        means_batched = np.split(means[:even_split_section], batches)
        masks_batched = np.split(masks[:even_split_section], batches)
        if faces.shape[0] % batch_size != 0:
            faces_batched.append(faces[even_split_section:])
            means_batched.append(means[even_split_section:])
            masks_batched.append(masks[even_split_section:])
        batched = zip(faces_batched, means_batched)
        print("faces: ", faces.shape,"means: ", means.shape,"masks: ", masks.shape,"faces_batched: ", len(faces_batched),"means_batched: ", len(means_batched))
        print("faces: ", faces.shape,"means: ", means.shape,"masks: ", masks.shape,"faces_batched: ", faces_batched[0].shape,"means_batched: ", means_batched[0].shape)
        for i, (faces, means) in enumerate(batched):
            if  self.target_size == 256:
                model_input = faces
                with keras.backend.tf.device("/cpu:0"):
                    results = self.mask_model.predict(model_input)
                results = np.swapaxes(results, 2, 0)
            else:
                # pylint: disable=no-member
                model_input = faces - means[:, None, None, :]
                with keras.backend.tf.device("/cpu:0"):
                    results = self.mask_model.predict_on_batch(model_input)
                print("done prediction")
                results = results[..., 1:2]
                generator = (cv2.GaussianBlur(mask, (7, 7), 0) for mask in results)
                if postprocess_test:
                    generator = (self.postprocessing(mask[:, :, None]) for mask in results)
                results = np.array(tuple(generator))[..., None]
            results[results < 0.05] = 0.
            results[results > 0.95] = 1.
            print("results......................: ",results.shape)
            _, results = self.resize_inputs(results, original_size)
            batch_slice = slice(i * batch_size, (i + 1) * batch_size)
            #results = results * 255.
            masks[batch_slice] = results[..., None]
        print("done batch")
        return self.merge_masks(faces, masks, channels)

    @staticmethod
    def resize_inputs(faces, target_size):
        """ resize input and output of mask models appropriately """
        _, height, width, _ = faces.shape
        image_size = min(height, width)
        if image_size != target_size:
            method = cv2.INTER_CUBIC if image_size < target_size else cv2.INTER_AREA
            generator = (cv2.resize(face, (target_size, target_size), method) for face in faces)
            faces = np.array(tuple(generator))

        return image_size, faces

    @staticmethod
    def postprocessing(mask):
        """ Post-processing of Nirkin style segmentation masks """
        # pylint: disable=no-member
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
            holes = cv2.bitwise_not(not_holes)[2:-2, 2:-2]
            mask = cv2.bitwise_or(mask, holes)
            mask = np.expand_dims(mask, axis=-1)

        return mask


class Dummy(Mask):
    """ Dummy mask to enable full crop training of face and background """

    def build_masks(self, faces, means, landmarks, channels):
        """ Dummy mask of all ones """
        masks = np.array(np.ones(faces.shape[:-1] + (1,)), dtype='float32', ndmin=4)
        return self.merge_masks(faces, masks, channels)
