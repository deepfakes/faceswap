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
                    # import sys
                    # import traceback
                    # exc_info = sys.exc_info()
                    # traceback.print_exception(*exc_info)

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
        new_image, background = self.get_new_image(predicted, frame_size)
        patched_face = self.post_warp_adjustments(background, new_image)
        patched_face = self.scale_image(patched_face)
        patched_face *= 255.0
        patched_face = np.rint(
            patched_face,
            out=np.empty(patched_face.shape, dtype="uint8"),
            casting='unsafe'
        )
        if self.writer_pre_encode is not None:
            patched_face = self.writer_pre_encode(patched_face)
        logger.trace("Patched image: '%s'", predicted["filename"])
        return patched_face

    def get_new_image(self, predicted, frame_size):
        """ Get the new face from the predictor and apply box manipulations """
        logger.trace("Getting: (filename: '%s', faces: %s)",
                     predicted["filename"], len(predicted["swapped_faces"]))

        placeholder = np.zeros((frame_size[1], frame_size[0], 4), dtype="float32")
        background = predicted["image"] / np.array(255.0, dtype="float32")
        placeholder[:, :, :3] = background

        for new_face, detected_face in zip(predicted["swapped_faces"],
                                           predicted["detected_faces"]):
            predicted_mask = new_face[:, :, -1] if new_face.shape[2] == 4 else None
            new_face = new_face[:, :, :3]
            src_face = detected_face.reference_face[..., :3] / 255.0
            interpolator = detected_face.reference_interpolators[1]

            new_face = self.pre_warp_adjustments(src_face, new_face, detected_face, predicted_mask)

            # Warp face with the mask
            cv2.warpAffine(  # pylint: disable=no-member
                new_face,
                detected_face.reference_matrix,
                frame_size,
                placeholder,
                flags=cv2.WARP_INVERSE_MAP | interpolator,  # pylint: disable=no-member
                borderMode=cv2.BORDER_TRANSPARENT)  # pylint: disable=no-member

        np.clip(placeholder, 0.0, 1.0, out=placeholder)
        logger.trace("Got filename: '%s'. (placeholders: %s)",
                     predicted["filename"], placeholder.shape)

        return placeholder, background

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
            new_face[:, :, -1] = np.minimum(new_face[:, :, -1], mask.squeeze())
        else:
            logger.trace("Adding mask to alpha channel")
            new_face = np.concatenate((new_face, mask), -1)
        np.clip(new_face, 0.0, 1.0, out=new_face)
        logger.trace("Got mask. Image shape: %s", new_face.shape)
        return new_face, raw_mask

    def post_warp_adjustments(self, background, new_image):
        """ Apply fixes to the image after warping """
        if self.adjustments["scaling"] is not None:
            new_image = self.adjustments["scaling"].run(new_image)

        if self.draw_transparent:
            frame = new_image
        else:
            foreground, mask = np.split(new_image, (3, ), axis=-1)
            foreground *= mask
            background *= (1.0 - mask)
            background += foreground
            frame = background
        np.clip(frame, 0.0, 1.0, out=frame)
        return frame

    def scale_image(self, frame):
        """ Scale the image if requested """
        if self.scale == 1:
            return frame
        logger.trace("source frame: %s", frame.shape)
        interp = cv2.INTER_CUBIC if self.scale > 1 else cv2.INTER_AREA  # pylint: disable=no-member
        dims = (round((frame.shape[1] / 2 * self.scale) * 2),
                round((frame.shape[0] / 2 * self.scale) * 2))
        frame = cv2.resize(frame, dims, interpolation=interp)  # pylint: disable=no-member
        logger.trace("resized frame: %s", frame.shape)
        np.clip(frame, 0.0, 1.0, out=frame)
        return frame
