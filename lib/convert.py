#!/usr/bin/env python3
""" Converter for faceswap.py
    Based on: https://gist.github.com/anonymous/d3815aba83a8f79779451262599b0955
    found on https://www.reddit.com/r/deepfakes/ """

import logging

import cv2
import numpy as np

from plugins.plugin_loader import PluginLoader

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

class Converter():
    """ Swap a source face with a target """
    def __init__(self, output_dir, output_size, output_has_mask,
                 draw_transparent, pre_encode, arguments, configfile=None):
        logger.debug("Initializing %s: (output_dir: '%s', output_size: %s,  output_has_mask: %s, "
                     "draw_transparent: %s, pre_encode: %s, arguments: %s, configfile: %s)",
                     self.__class__.__name__, output_dir, output_size, output_has_mask,
                     draw_transparent, pre_encode, arguments, configfile)
        self.output_dir = output_dir
        self.draw_transparent = draw_transparent
        self.writer_pre_encode = pre_encode
        self.scale = arguments.output_scale / 100
        self.output_size = output_size
        self.output_has_mask = output_has_mask
        self.args = arguments
        self.configfile = configfile
        self.adjustments = dict(box=None, mask=None, color=None, seamless=None, scaling=None)
        self.load_plugins()
        logger.debug("Initialized %s", self.__class__.__name__)

    def reinitialize(self, config):
        """ reinitialize converter """
        logger.debug("Reinitializing converter")
        self.adjustments = dict(box=None, mask=None, color=None, seamless=None, scaling=None)
        self.load_plugins(config=config, disable_logging=True)
        logger.debug("Reinitialized converter")

    def load_plugins(self, config=None, disable_logging=False):
        """ Load the requested adjustment plugins """
        logger.debug("Loading plugins. config: %s", config)
        self.adjustments["box"] = PluginLoader.get_converter(
            "mask",
            "box_blend",
            disable_logging=disable_logging)("none",
                                             self.output_size,
                                             configfile=self.configfile,
                                             config=config)

        self.adjustments["mask"] = PluginLoader.get_converter(
            "mask",
            "mask_blend",
            disable_logging=disable_logging)(self.args.mask_type,
                                             self.output_size,
                                             self.output_has_mask,
                                             configfile=self.configfile,
                                             config=config)

        if self.args.color_adjustment != "none" and self.args.color_adjustment is not None:
            self.adjustments["color"] = PluginLoader.get_converter(
                "color",
                self.args.color_adjustment,
                disable_logging=disable_logging)(configfile=self.configfile, config=config)

        if self.args.scaling != "none" and self.args.scaling is not None:
            self.adjustments["scaling"] = PluginLoader.get_converter(
                "scaling",
                self.args.scaling,
                disable_logging=disable_logging)(configfile=self.configfile, config=config)
        logger.debug("Loaded plugins: %s", self.adjustments)

    def process(self, in_queue, out_queue, completion_queue=None):
        """ Process items from the queue """
        logger.debug("Starting convert process. (in_queue: %s, out_queue: %s, completion_queue: "
                     "%s)", in_queue, out_queue, completion_queue)
        while True:
            items = in_queue.get()
            if items == "EOF":
                logger.debug("EOF Received")
                logger.debug("Patch queue finished")
                # Signal EOF to other processes in pool
                logger.debug("Putting EOF back to in_queue")
                in_queue.put(items)
                break

            if isinstance(items, dict):
                items = [items]
            for item in items:
                logger.trace("Patch queue got: '%s'", item["filename"])
                try:
                    image = self.patch_image(item)
                except Exception as err:  # pylint: disable=broad-except
                    # Log error and output original frame
                    logger.error("Failed to convert image: '%s'. Reason: %s",
                                 item["filename"], str(err))
                    image = item["image"]
                    # UNCOMMENT THIS CODE BLOCK TO PRINT TRACEBACK ERRORS
                    import sys
                    import traceback
                    exc_info = sys.exc_info()
                    traceback.print_exception(*exc_info)

                logger.trace("Out queue put: %s", item["filename"])
                out_queue.put((item["filename"], image))
        logger.debug("Completed convert process")
        # Signal that this process has finished
        if completion_queue is not None:
            completion_queue.put(1)

    def patch_image(self, predicted):
        """ Patch the image """
        logger.trace("Patching image: '%s'", predicted["filename"])
        frame_size = (predicted["image"].shape[1], predicted["image"].shape[0])
        new_image, original_frame = self.get_new_image(predicted, frame_size)
        patched_face = self.post_warp_adjustments(original_frame, new_image)
        patched_face = self.scale_image(patched_face)
        patched_face = np.rint(patched_face,
                               out=np.empty(patched_face.shape, dtype="uint8"),
                               casting='unsafe')
        if self.writer_pre_encode is not None:
            patched_face = self.writer_pre_encode(patched_face)
        logger.trace("Patched image: '%s'", predicted["filename"])
        return patched_face

    def get_new_image(self, predicted, frame_size):
        """ Get the new face from the predictor and apply box manipulations """
        logger.trace("Getting: (filename: '%s', faces: %s)",
                     predicted["filename"], len(predicted["swapped_faces"]))
        original_frame = predicted["image"].astype("float32") / 255.
        blank_mask = np.zeros(original_frame.shape[:-1] + (1,), dtype="float32")
        new_image = np.concatenate([original_frame, blank_mask], axis=-1)
        frame_size = (original_frame.shape[1], original_frame.shape[0])

        dual_generator = zip(predicted["swapped_faces"], predicted["detected_faces"])
        for new_face, old_face in dual_generator:
            new_face = self.pre_warp_adjustments(old_face.reference_face / 255., new_face)
            interpolator = old_face.reference_interpolators[1]
            flags = cv2.WARP_INVERSE_MAP + interpolator  # pylint: disable=no-member
            new_image = cv2.warpAffine(new_face,
                           old_face.reference_matrix,
                           frame_size,
                           new_image,
                           flags=cv2.WARP_INVERSE_MAP | old_face.reference_interpolators[1],
                           borderMode=cv2.BORDER_TRANSPARENT)
            np.clip(new_image, 0., 1., out=new_image)
        logger.trace("Got filename: '%s'. (new_image: %s)", predicted["filename"], new_image.shape)
        return new_image, original_frame

    def pre_warp_adjustments(self, old_face, new_face):
        """ Run the pre-warp adjustments """
        logger.trace("old_face shape: %s, new_face shape: %s", old_face.shape, new_face.shape)
        old_face, new_face, mask = self.get_image_mask(old_face, new_face)
        if self.adjustments["color"] is not None:
            new_face = self.adjustments["color"].run(old_face, new_face, mask)
        if self.adjustments["seamless"] is not None:
            new_face = self.adjustments["seamless"].run(old_face, new_face, mask)
        new_face = np.concatenate([new_face, mask], axis=-1)
        logger.trace("returning: new_face shape %s", new_face.shape)
        return new_face

    def get_image_mask(self, old_face, new_face):
        """ Get the image mask """
        logger.trace("Getting mask. Image shape: %s", new_face.shape)
        new_face = self.adjustments["box"].run(new_face)
        mask = self.adjustments["mask"].run(new_face)
        np.clip(new_face, 0.0, 1.0, out=new_face)
        logger.trace("Got mask. Image shape: %s, Mask shape: %s", new_face.shape, mask.shape)
        return old_face[..., :3], new_face[..., :3], mask

    def post_warp_adjustments(self, background, new_image):
        """ Apply fixes to the image after warping """
        logger.trace("Compositing face into frame")
        if self.adjustments["scaling"] is not None:
            new_image = self.adjustments["scaling"].run(new_image)

        mask = new_image[..., -1:]
        foreground = new_image[..., :3]
        frame = foreground * mask + background * (1. - mask)
        if self.draw_transparent:
            frame = np.concatenate((frame, mask), axis=-1)
        np.clip(frame * 255., 0., 255., out=frame)
        logger.trace("Swapped frame created")
        return frame

    def scale_image(self, frame):
        """ Scale the image if requested """
        if self.scale != 1.:
            logger.trace("source frame: %s", frame.shape)
            interp = cv2.INTER_CUBIC if self.scale > 1. else cv2.INTER_AREA  # pylint: disable=no-member
            frame = cv2.resize(frame,  # pylint: disable=no-member
                               (0, 0), fx=self.scale, fy=self.scale, interpolation=interp)
            logger.trace("resized frame: %s", frame.shape)
        return frame
