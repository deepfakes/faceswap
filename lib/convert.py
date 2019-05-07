#!/usr/bin/env python3
""" Converter for faceswap.py
    Based on: https://gist.github.com/anonymous/d3815aba83a8f79779451262599b0955
    found on https://www.reddit.com/r/deepfakes/ """

import logging

import cv2
import numpy as np
from lib.model.masks import get_default_mask
from lib.model.masks import Mask as mask_class
from plugins.plugin_loader import PluginLoader

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class Converter():
    """ Swap a source face with a target """
    def __init__(self, output_dir, output_size, output_has_mask,
                 draw_transparent, pre_encode, arguments):
        logger.debug("Initializing %s: (output_dir: '%s', output_size: %s,  output_has_mask: %s, "
                     "draw_transparent: %s, pre_encode: %s, arguments: %s)",
                     self.__class__.__name__, output_dir, output_size, output_has_mask,
                     draw_transparent, pre_encode, arguments)
        self.output_dir = output_dir
        self.draw_transparent = draw_transparent
        self.writer_pre_encode = pre_encode
        self.scale = arguments.output_scale / 100.
        self.args = arguments
        self.adjustments = dict(box=None, mask=None, color=None, seamless=None, scaling=None)
        self.load_plugins(output_size, output_has_mask)
        logger.debug("Initialized %s", self.__class__.__name__)

    def load_plugins(self, output_size, output_has_mask):
        """ Load the requested adjustment plugins """
        logger.debug("Loading plugins")
        self.adjustments["box"] = PluginLoader.get_converter("mask", "box_blend")(
            "none",
            output_size)

        self.adjustments["mask"] = PluginLoader.get_converter("mask", "mask_blend")(
            self.args.mask_type,
            output_size,
            output_has_mask)

        if self.args.color_adjustment != "none" and self.args.color_adjustment is not None:
            self.adjustments["color"] = PluginLoader.get_converter("color",
                                                                   self.args.color_adjustment)()

        if self.args.scaling != "none" and self.args.scaling is not None:
            self.adjustments["scaling"] = PluginLoader.get_converter("scaling",
                                                                     self.args.scaling)()
        logger.debug("Loaded plugins: %s", self.adjustments)

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
                # UNCOMMENT THIS CODE BLOCK TO PRINT TRACEBACK ERRORS
                # import sys
                # import traceback
                # exc_info = sys.exc_info()
                # traceback.print_exception(*exc_info)

            logger.trace("Out queue put: %s", item["filename"])
            out_queue.put((item["filename"], image))
        logger.debug("Completed convert process")

    def patch_image(self, predicted):
        """ Patch the image """
        logger.trace("Patching image: '%s'", predicted["filename"])
        frame_size = (predicted["image"].shape[1], predicted["image"].shape[0])
        new_image = self.get_new_image(predicted, frame_size)
        patched_face = self.post_warp_adjustments(predicted, new_image)
        patched_face = self.scale_image(patched_face)
        patched_face = np.rint(patched_face).astype("uint8")
        if self.writer_pre_encode is not None:
            patched_face = self.writer_pre_encode(patched_face)
        logger.trace("Patched image: '%s'", predicted["filename"])
        return patched_face

    def get_new_image(self, predicted, frame_size):
        """ Get the new face from the predictor and apply box manipulations """
        logger.trace("Getting: (filename: '%s', faces: %s)",
                     predicted["filename"], len(predicted["swapped_faces"]))
        #TODO finish
        print(placeholder.dtype)
        placeholder = (predicted["image"] / 255.).astype("float32")
        zeros = np.zeros((frame_size[1], frame_size[0], 1), dtype="float32")
        placeholder = np.concatenate((placeholder, zeros), axis=-1)
        dual_generator = zip(predicted["swapped_faces"], predicted["detected_faces"])
        for new_face, detected_face in dual_generator:
            predicted_mask = new_face[:, :, -1] if new_face.shape[2] == 4 else None
            new_face = new_face[:, :, :3]
            src_face = detected_face.reference_face
            interpolator = detected_face.reference_interpolators[1]

            new_face = self.pre_warp_adjustments(src_face, new_face, detected_face, predicted_mask)

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

    def pre_warp_adjustments(self, old_face, new_face, detected_face, predicted_mask):
        """ Run the pre-warp adjustments """
        logger.trace("old_face shape: %s, new_face shape: %s, predicted_mask shape: %s",
                     old_face.shape, new_face.shape,
                     predicted_mask.shape if predicted_mask is not None else None)
        new_face = self.adjustments["box"].run(new_face)
        new_face, raw_mask = self.get_image_mask(new_face, detected_face, predicted_mask)
        if self.adjustments["color"] is not None:
            new_face = self.adjustments["color"].run(old_face, new_face, raw_mask)
        if self.adjustments["seamless"] is not None:
            new_face = self.adjustments["seamless"].run(old_face, new_face, raw_mask)
        logger.trace("returning: new_face shape %s", new_face.shape)
        return new_face

    def get_image_mask(self, new_face, detected_face, predicted_mask):
        """ Get the image mask """
        logger.trace("Getting mask. Image shape: %s", new_face.shape)
        mask, raw_mask = self.adjustments["mask"].run(detected_face, predicted_mask)
        if new_face.shape[2] == 4:
            logger.trace("Combining mask with alpha channel box mask")
            new_face[:, :, -1] = np.minimum(new_face[:, :, -1:], mask)
        else:
            logger.trace("Adding mask to alpha channel")
            new_face = np.concatenate((new_face, mask), axis=-1)
        new_face = np.clip(new_face, 0., 1.)
        logger.trace("Got mask. Image shape: %s", new_face.shape)
        return new_face, raw_mask

    def post_warp_adjustments(self, predicted, new_image):
        """ Apply fixes to the image after warping """
        if self.adjustments["scaling"] is not None:
            new_image = self.adjustments["scaling"].run(new_image)
        mask = new_image[:, :, -1:]
        foreground = new_image[:, :, :3] * 255.
        background = predicted["image"][:, :, :3]
        frame = foreground * mask + background * (1. - mask)
        frame = self.add_alpha_mask(frame, predicted)
        frame = np.clip(frame, 0., 255.)
        return frame

    def add_alpha_mask(self, frame, predicted):
        """ Adding a 4th channel should happen after all other channel operations
            Add the default mask as 4th channel for saving an image with alpha channel """
        if self.draw_transparent:
            logger.trace("Creating transparent image: '%s'", predicted["filename"])
            mask_type = get_default_mask()
            final_mask = np.zeros(frame.shape[:2] + (1, ), dtype="float32")
            for detected_face in predicted["detected_faces"]:
                landmarks = detected_face.landmarks_as_xy
                mask = mask_class(landmarks, frame, mask_type, channels=1).mask
                final_mask = np.bitwise_or(final_mask, mask)
            frame = np.concatenate((frame, np.expand_dims(final_mask, axis=-1)), axis=-1)
            logger.trace("Created transparent image: '%s'", predicted["filename"])
        return frame

    def scale_image(self, frame):
        """ Scale the image if requested """
        if self.scale != 1.:
            logger.trace("source frame: %s", frame.shape)
            interp = cv2.INTER_CUBIC if self.scale > 1 else cv2.INTER_AREA  # pylint: disable=no-member
            frame = cv2.resize(frame, fx=self.scale, fy=self.scale,  # pylint: disable=no-member
                               interpolation=interp)
            logger.trace("resized frame: %s", frame.shape)
        return frame
