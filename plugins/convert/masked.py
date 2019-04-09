#!/usr/bin/env python3
""" Masked converter for faceswap.py
    Based on: https://gist.github.com/anonymous/d3815aba83a8f79779451262599b0955
    found on https://www.reddit.com/r/deepfakes/ """

import logging
import os
from pathlib import Path

import cv2
import numpy as np
from lib.model import masks as model_masks

from . import Box, Mask, Face, Scaling

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

# TODO Move this module


class Convert():
    """ Swap a source face with a target """
    def __init__(self, output_dir, output_size, arguments):
        logger.debug("Initializing %s: (output_dir: '%s', output_size: %s,  arguments: %s)",
                     self.__class__.__name__, output_dir, output_size, arguments)
        self.output_dir = output_dir
        self.args = arguments
        self.box = Box(arguments, output_size)
        self.mask = Mask(arguments, output_size)
        self.pre_adjustments = Face(arguments)
        self.post_adjustments = Scaling(arguments)
        logger.debug("Initialized %s", self.__class__.__name__)

    def process(self, in_queue, out_queue):
        """ Process items from the queue """
        logger.debug("Starting convert process. (in_queue: %s, out_queue: %s)",
                     in_queue, out_queue)
        while True:
            item = in_queue.get()
            if item == "EOF":
                logger.debug("Patch queue finished")
                # Signal EOF to other processes in pool
                in_queue.put(item)
                break
            logger.trace("Patch queue got: '%s'", item["filename"])

            try:
                image = self.patch_image(item)
            except Exception as err:  # pylint: disable=broad-except
                # Log error and output original frame
                logger.error("Failed to convert image: '%s'. Reason: %s",
                             item["filename"], str(err))
                image = item["image"]
                # TODO Remove this debugging code
                import sys
                import traceback
                exc_info = sys.exc_info()
                traceback.print_exception(*exc_info)

            out_file = str(self.output_dir / Path(item["filename"]).name)
            if self.args.draw_transparent:
                out_file = "{}.png".format(os.path.splitext(out_file)[0])
                logger.trace("Set extension to png: `%s`", out_file)

            logger.trace("Out queue put: %s", out_file)
            out_queue.put((out_file, image))

        out_queue.put("EOF")
        logger.debug("Completed convert process")

    def patch_image(self, predicted):
        """ Patch the image """
        logger.trace("Patching image: '%s'", predicted["filename"])
        frame_size = (predicted["image"].shape[1], predicted["image"].shape[0])
        new_image = self.get_new_image(predicted, frame_size)
        patched_face = self.apply_post_warp_fixes(predicted, new_image)
        logger.trace("Patched image: '%s'", predicted["filename"])
        return patched_face

    def get_new_image(self, predicted, frame_size):
        """ Get the new face from the predictor and apply box manipulations """
        logger.trace("Getting: (filename: '%s', faces: %s)",
                     predicted["filename"], len(predicted["swapped_faces"]))

        placeholder = predicted["image"] / 255.0
        placeholder = np.concatenate((placeholder,
                                      np.zeros((frame_size[1], frame_size[0], 1))),
                                     axis=-1).astype("float32")
        for new_face, detected_face in zip(predicted["swapped_faces"],
                                           predicted["detected_faces"]):
            new_face = new_face[:, :, :3]
            src_face = detected_face.reference_face

            predicted_mask = new_face[:, :, :-1] if new_face.ndim == 4 else None
            interpolator = detected_face.reference_interpolators[1]

            new_face = self.box.do_actions(old_face=src_face, new_face=new_face)
            new_face, raw_mask = self.get_image_mask(new_face, detected_face, predicted_mask)
            new_face = self.pre_adjustments.do_actions(old_face=src_face,
                                                       new_face=new_face,
                                                       raw_mask=raw_mask)

            # Warp face with the mask
            placeholder = cv2.warpAffine(  # pylint: disable=no-member
                new_face,
                detected_face.reference_matrix,
                frame_size,
                placeholder,
                flags=cv2.WARP_INVERSE_MAP | interpolator,  # pylint: disable=no-member
                borderMode=cv2.BORDER_TRANSPARENT)  # pylint: disable=no-member

            placeholder = np.clip(placeholder, 0.0, 1.0)

        logger.trace("Got filename: '%s'. (placeholders: %s)",
                     predicted["filename"], placeholder.shape)

        return placeholder

    def get_image_mask(self, new_face, detected_face, predicted_mask):
        """ Get the image mask """
        logger.trace("Getting mask. Image shape: %s", new_face.shape)
        mask, raw_mask = self.mask.do_actions(detected_face=detected_face,
                                              predicted_mask=predicted_mask)
        if new_face.shape[2] == 4:
            logger.trace("Combining mask with alpha channel box mask")
            new_face[:, :, -1] = np.minimum(new_face[:, :, -1], mask.squeeze())
        else:
            logger.trace("Adding mask to alpha channel")
            new_face = np.concatenate((new_face, mask), -1)
        new_face = np.clip(new_face, 0.0, 1.0)
        logger.trace("Got mask. Image shape: %s", new_face.shape)
        return new_face, raw_mask

    def apply_post_warp_fixes(self, predicted, new_image):
        """ Apply fixes to the image prior to warping """
        new_image = self.post_adjustments.do_actions(new_face=new_image)

        mask = np.repeat(new_image[:, :, -1][:, :, np.newaxis], 3, axis=-1)
        foreground = new_image[:, :, :3]
        background = (predicted["image"][:, :, :3] / 255.0) * (1.0 - mask)

        foreground *= mask
        frame = foreground + background
        frame = self.draw_transparent(frame, predicted)

        np.clip(frame, 0.0, 1.0, out=frame)
        return np.rint(frame * 255.0).astype("uint8")

    def draw_transparent(self, frame, predicted):
        """ Adding a 4th channel should happen after all other channel operations
            Add the default mask as 4th channel for saving as png with alpha channel """
        if not self.args.draw_transparent:
            return frame
        logger.trace("Creating transparent image: '%s'", predicted["filename"])
        mask_type = getattr(model_masks, model_masks.get_default_mask())
        final_mask = np.zeros(frame.shape[:2] + (1, ), dtype="float32")

        for detected_face in predicted["detected_faces"]:
            landmarks = detected_face.landmarks_as_xy
            final_mask = cv2.bitwise_or(final_mask,  # pylint: disable=no-member
                                        mask_type(landmarks, frame, channels=1).mask)
        frame = np.concatenate((frame, np.expand_dims(final_mask, axis=-1)), axis=-1)
        logger.trace("Created transparent image: '%s'", predicted["filename"])
        return frame
