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
            item = in_queue.get()
            if item == "EOF":
                logger.debug("EOF Received")
                logger.debug("Patch queue finished")
                # Signal EOF to other processes in pool
                logger.debug("Putting EOF back to in_queue")
                in_queue.put(item)
                break
            logger.trace("Patch queue got: '%s'", item["filename"])

            try:
                image = self.patch_image(item)
            except Exception as err:  # pylint: disable=broad-except
                # Log error and output original frame
                logger.error("Failed to convert: '%s'. Reason: %s", item["filename"], str(err))
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
        new_image = self.get_new_image(predicted)
        patched_face = self.post_warp_adjustments(predicted, new_image)
        patched_face = self.scale_image(patched_face)
        print("patched mean: ", patched_face.shape, "---", np.mean(patched_face, axis=(0,1)))
        patched_face = np.rint(patched_face).astype("uint8")
        if self.writer_pre_encode is not None:
            patched_face = self.writer_pre_encode(patched_face)
        logger.trace("Patched image: '%s'", predicted["filename"])
        return patched_face

    def get_new_image(self, predicted):
        """ Get the new face from the predictor and apply box manipulations """
        # pylint: disable=no-member
        logger.trace("Getting: (filename: '%s', faces: %s)",
                     predicted["filename"], len(predicted["swapped_faces"]))
        print("predicted[image]: ",predicted["image"].shape, predicted["image"].dtype)
        frame = predicted["image"].astype("float32") / 255.
        print("before: ",frame.shape)
        frame_size = (frame.shape[1], frame.shape[0])
        # print("old: ", [item.reference_face.shape for item in predicted["detected_faces"]], "   new: ", [item.shape for item in predicted["swapped_faces"]])
        dual_generator = zip(predicted["swapped_faces"], predicted["detected_faces"])
        for new_face, detected_face in dual_generator:
            print("new_facw: ", new_face.shape)
            src_face = detected_face.reference_face
            print("src_facw: ", src_face.shape)
            new_face = self.pre_warp_adjustments(src_face, new_face)
            interpolator = detected_face.reference_interpolators[1]
            # Warp face with the mask
            new_image = cv2.warpAffine(new_face,
                                       detected_face.reference_matrix,
                                       frame_size,
                                       frame,
                                       flags=cv2.WARP_INVERSE_MAP | interpolator,
                                       borderMode=cv2.BORDER_TRANSPARENT)
            new_image = np.clip(new_image, 0., 1.)
        logger.trace("Got filename: '%s'. (new_image: %s)", predicted["filename"], new_image.shape)
        return new_image

    def pre_warp_adjustments(self, old_face, new_face):
        """ Run the pre-warp adjustments """
        logger.trace("old_face shape: %s, new_face shape: %s", old_face.shape, new_face.shape)
        new_face, mask = self.get_image_mask(new_face)
        if self.adjustments["color"] is not None:
            new_face = self.adjustments["color"].run(old_face, new_face, mask)
        if self.adjustments["seamless"] is not None:
            new_face = self.adjustments["seamless"].run(old_face, new_face, mask)
        logger.trace("returning: new_face shape %s", new_face.shape)
        return new_face

    def get_image_mask(self, new_face):
        """ Get the image mask """
        logger.trace("Getting mask. Image shape: %s", new_face.shape)
        print("face_new: ", new_face.shape)
        new_face = self.adjustments["box"].run(new_face)
        mask = self.adjustments["mask"].run(new_face)
        logger.trace("Got mask. Image shape: %s, Mask shape: %s", new_face.shape, mask.shape)
        return new_face, mask

    def post_warp_adjustments(self, predicted, new_image):
        """ Apply fixes to the image after warping """
        logger.trace("Compositing face into frame")
        if self.adjustments["scaling"] is not None:
            new_image = self.adjustments["scaling"].run(new_image)
        print("after: ",new_image.shape)        
        mask = new_image[..., -1:]
        foreground = new_image[..., :3] * 255.
        background = predicted["image"][..., :3]
        frame = foreground * mask + background * (1. - mask)
        if self.draw_transparent:
            frame = np.concatenate((frame, mask * 255.), axis=-1)
        np.clip(frame, 0., 255.)
        logger.trace("Swapped frame created")
        return frame

    def scale_image(self, frame):
        """ Scale the image if requested """
        # pylint: disable=no-member
        if self.scale != 1.:
            logger.trace("source frame: %s", frame.shape)
            interp = cv2.INTER_CUBIC if self.scale > 1. else cv2.INTER_AREA
            frame = cv2.resize(frame, fx=self.scale, fy=self.scale, interpolation=interp)
            logger.trace("resized frame: %s", frame.shape)
        return frame
