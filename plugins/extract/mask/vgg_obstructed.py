#!/us#!/usr/bin/env python3
""" xxxxxxx for faceswap.py
Adapted from: xxxxxxxxxxxx
MIT License

Copyright (c) xxxx xxx xxxxxxx

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import cv2
import keras
import sys
import numpy as np

from ._base import Masker, logger


class Mask(Masker):
    """ Perform transformation to align and get landmarks """
    def __init__(self, **kwargs):
        git_model_id = 5
        model_filename = "Nirkin_500_softmax_v1.h5"
        super().__init__(git_model_id=git_model_id,
                         model_filename=model_filename,
                         input_size=500,
                         **kwargs)
        self.vram = 0
        self.model = None
        self.supports_plaidml = True

    def initialize(self, *args, **kwargs):
        """ Initialization tasks to run prior to alignments """
        try:
            super().initialize(*args, **kwargs)
            logger.info("Initializing VGG Mask Network(500)...")
            logger.debug("VGG initialize: (args: %s kwargs: %s)", args, kwargs)
            with keras.backend.tf.device("/cpu:0"):
                self.model = keras.models.load_model(self.model_path)
            o = keras.layers.core.Activation('softmax', name='softmax')(self.model.layers[-1].output)
            self.model = keras.models.Model(input=self.model.input, output=[o])
            self.init.set()
            logger.info("Initialized VGG Mask Network(500)")
        except Exception as err:
            self.error.set()
            raise err

    # MASK PROCESSING
    def build_masks(self, faces, landmarks):
        """ Function for creating facehull masks
            Faces may be of shape (batch_size, height, width, 3) or (height, width, 3)
        """
        postprocess_test = False
        images = faces.astype("float32")
        means = np.mean(images, axis=(0,1), dtype='float32')
        images = images[None, ...] if images.ndim ==3 else images
        means = means[None, ...] if means.ndim ==1 else means
        masks = np.array(np.zeros(images.shape[:-1] + (1, )), dtype='uint8', ndmin=4)
        
        original_size, resized_faces = self.resize_inputs(images, self.input_size)
        batch_size = min(resized_faces.shape[0], 8)
        batches = ((resized_faces.shape[0] - 1) // batch_size) + 1
        even_split_section = batches * batch_size
        faces_batched = np.split(resized_faces[:even_split_section], batches)
        means_batched = np.split(means[:even_split_section], batches)
        masks_batched = np.split(masks[:even_split_section], batches)
        if resized_faces.shape[0] % batch_size != 0:
            faces_batched.append(resized_faces[even_split_section:])
            means_batched.append(means[even_split_section:])
            masks_batched.append(masks[even_split_section:])
        batched = zip(faces_batched, means_batched)
        for i, (face, mean) in enumerate(batched):
            # pylint: disable=no-member
            model_input = face - mean[:, None, None, :]
            results = self.model.predict_on_batch(model_input)
            print("shape: ", results.shape)
            sys.stdout.flush()
            print(np.mean(results, axis=(0,1,2)))
            sys.stdout.flush()
            results = results[..., 0:1] * -1. + 1.
            #results = np.clip(results.argmax(axis=-1), 0, 1).astype('float32')
            generator = (cv2.GaussianBlur(mask, (7, 7), 0) for mask in results)
            if postprocess_test:
                generator = (self.postprocessing(mask[:, :, None]) for mask in results)
            results = np.array(tuple(generator))[..., None]
            print("shape: ", results.shape)
            sys.stdout.flush()
            results[results < 0.05]= 0.
            results[results > 0.95] = 1.
            results *= 255.
            _, results = self.resize_inputs(results, original_size)
            print("shape: ", masks.shape, results.shape)
            sys.stdout.flush()
            batch_slice = slice(i * batch_size, (i + 1) * batch_size)
            masks[batch_slice] = results[..., None].astype('uint8')
        faces = np.concatenate((faces[..., :3], masks[0]), axis=-1)
        return faces, masks

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

