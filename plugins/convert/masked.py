#!/usr/bin/env python3
""" Masked converter for faceswap.py
    Based on: https://gist.github.com/anonymous/d3815aba83a8f79779451262599b0955
    found on https://www.reddit.com/r/deepfakes/ """

import logging
import os
from pathlib import Path

import cv2
import numpy as np
from lib.model.masks import dfl_full

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class Convert():
    """ Swap a source face with a target """
    def __init__(self, output_dir, training_size, padding, crop, arguments):
        logger.debug("Initializing %s: (output_dir: '%s', training_size: %s, padding: %s, "
                     "crop: %s, arguments: %s)", self.__class__.__name__, output_dir,
                     training_size, padding, crop, arguments)
        self.crop = crop
        self.output_dir = output_dir
        self.args = arguments
        self.mask = Mask(arguments.mask_type, training_size, padding, crop)
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
            except Exception as err:
                # Log error and output original frame
                logger.error("Failed to convert image: '%s'. Reason: %s",
                             item["filename"], str(err))
                image = item["image"]

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
        image_mask = self.get_image_mask(predicted, frame_size)
        patched_face = self.apply_fixes(predicted, new_image, image_mask)
        logger.trace("Patched image: '%s'", predicted["filename"])
        return patched_face

    def get_new_image(self, predicted, frame_size):
        """ Get the new face from the predictor """
        new_image = predicted["image"].copy()
        source_faces = [face.aligned_face[:, :, :3] for face in predicted["detected_faces"]]
        interpolators = [face.adjusted_interpolators for face in predicted["detected_faces"]]

        for idx, new_face in enumerate(predicted["swapped_faces"]):
            src_face = source_faces[idx]
            old_face = predicted["original_faces"][idx]
            interpolator = interpolators[idx][1]

            if self.args.smooth_box:
                self.smooth_box(old_face, new_face)

            src_face[self.crop, self.crop] = new_face

            new_image = cv2.warpAffine(  # pylint: disable=no-member
                src_face,
                predicted["detected_faces"][idx].adjusted_matrix,
                frame_size,
                new_image,
                flags=cv2.WARP_INVERSE_MAP | interpolator,  # pylint: disable=no-member
                borderMode=cv2.BORDER_TRANSPARENT)  # pylint: disable=no-member
        return new_image

    @staticmethod
    def smooth_box(old_face, new_face):
        """ Perform gaussian blur on the edges of the output rect """
        height = new_face.shape[0]
        crop = slice(0, height)
        erode = slice(height // 15, -height // 15)
        sigma = height / 16  # 10 for the default 160 size
        window = int(np.ceil(sigma * 3.0))
        window = window + 1 if window % 2 == 0 else window
        mask = np.zeros_like(new_face)
        mask[erode, erode] = 1.0
        mask = cv2.GaussianBlur(mask,  # pylint: disable=no-member
                                (window, window),
                                sigma)
        new_face[crop, crop] = (mask * new_face + (1.0 - mask) * old_face)

    def get_image_mask(self, predicted, frame_size):
        """ Get the image mask """
        logger.trace("Getting image mask: '%s'", predicted["filename"])
        masks = list()
        for detected_face in predicted["detected_faces"]:
            mask = self.mask.get_mask(detected_face, frame_size)
            if self.args.erosion_size != 0:
                kwargs = {"src": mask,
                          "kernel": self.set_erosion_kernel(mask),
                          "iterations": 1}
                if self.args.erosion_size > 0:
                    mask = cv2.erode(**kwargs)  # pylint: disable=no-member
                else:
                    mask = cv2.dilate(**kwargs)  # pylint: disable=no-member

            if self.args.blur_size != 0:
                blur_size = self.set_blur_size(mask)
                for _ in [1, 2, 3, 4]:
                    mask = cv2.blur(mask, (blur_size, blur_size))   # pylint: disable=no-member
            masks.append(np.clip(mask, 0.0, 1.0, out=mask))
        logger.trace("Got image mask: '%s'", predicted["filename"])
        return masks

    def set_erosion_kernel(self, mask):
        """ Set the erosion kernel """
        erosion_ratio = self.args.erosion_size / 100
        mask_radius = np.sqrt(np.sum(mask)) / 2
        percent_erode = max(1, int(abs(erosion_ratio * mask_radius)))
        erosion_kernel = cv2.getStructuringElement(  # pylint: disable=no-member
            cv2.MORPH_ELLIPSE,  # pylint: disable=no-member
            (percent_erode, percent_erode))
        logger.trace("erosion_kernel shape: %s", erosion_kernel.shape)
        return erosion_kernel

    def set_blur_size(self, mask):
        """ Set the blur size to absolute or percentage """
        blur_ratio = self.args.blur_size / 100 / 1.6
        mask_radius = np.sqrt(np.sum(mask)) / 2
        blur_size = int(max(1, blur_ratio * mask_radius))
        logger.trace("blur_size: %s", blur_size)
        return blur_size

    def apply_fixes(self, predicted, new_image, masks):
        """ Apply fixes """
        frame = predicted["image"][:, :, :3]
        new_image = new_image[:, :, :3]
        for idx, detected_face in enumerate(predicted["detected_faces"]):
            image_mask = masks[idx][:, :, :3].copy()
            landmarks = detected_face.landmarks_as_xy

            # TODO - force default for args.sharpen_image to ensure it isn't None
            if self.args.sharpen_image is not None and self.args.sharpen_image.lower() != "none":
                new_image = self.sharpen(new_image, self.args.sharpen_image)

            if self.args.avg_color_adjust:
                new_image = self.color_adjust(new_image, frame, image_mask)

            if self.args.match_histogram:
                new_image = self.color_hist_match(new_image, frame, image_mask)

            if self.args.seamless_clone:
                frame = self.seamless_clone(new_image, frame, image_mask)

            else:
                foreground = new_image * image_mask
                background = frame * (1.0 - image_mask)
                frame = foreground + background

            np.clip(frame, 0.0, 255.0, out=frame)
            if self.args.draw_transparent:
                # Adding a 4th channel should happen after all other channel operations
                # Add mask as 4th channel for saving as alpha on supported output formats
                frame = dfl_full(landmarks, frame, channels=4)
        return np.rint(frame).astype('uint8')

    @staticmethod
    def sharpen(new, method):
        """ Sharpen using the unsharp=mask technique , subtracting a blurried image """
        np.clip(new, 0.0, 255.0, out=new)
        if method == "box_filter":
            kernel = np.ones((3, 3)) * (-1)
            kernel[1, 1] = 9
            new = cv2.filter2D(new, -1, kernel)  # pylint: disable=no-member
        elif method == "gaussian_filter":
            blur = cv2.GaussianBlur(new, (0, 0), 3.0)   # pylint: disable=no-member
            new = cv2.addWeighted(new, 1.5, blur, -.5, 0, new)  # pylint: disable=no-member

        return new

    @staticmethod
    def color_adjust(new, frame, img_mask):
        """ Adjust the mean of the color channels to be the same for the swap and old frame """
        for _ in [0, 1]:
            np.clip(new, 0.0, 255.0, out=new)
            diff = frame - new
            avg_diff = np.sum(diff * img_mask, axis=(0, 1))
            adjustment = avg_diff / np.sum(img_mask, axis=(0, 1))
            new = new + adjustment

        return new

    def color_hist_match(self, new, frame, img_mask):
        """ Match the histogram of the color intensity of each channel """
        np.clip(new, 0.0, 255.0, out=new)
        new = [self.hist_match(new[:, :, c], frame[:, :, c], img_mask[:, :, c]) for c in [0, 1, 2]]
        new = np.stack(new, axis=2)

        return new

    @staticmethod
    def hist_match(new, frame, img_mask):
        """  Construct the histogram of the color intensity of a channel
             for the swap and the original. Match the histogram of the original
             by interpolation
        """

        mask_indices = img_mask.nonzero()
        if len(mask_indices[0]) == 0:
            return new

        m_new = new[mask_indices].ravel()
        m_frame = frame[mask_indices].ravel()
        _, bin_idx, s_counts = np.unique(m_new, return_inverse=True, return_counts=True)
        t_values, t_counts = np.unique(m_frame, return_counts=True)
        s_quants = np.cumsum(s_counts, dtype='float32')
        t_quants = np.cumsum(t_counts, dtype='float32')
        s_quants /= s_quants[-1]  # cdf
        t_quants /= t_quants[-1]  # cdf
        interp_s_values = np.interp(s_quants, t_quants, t_values)
        new.put(mask_indices, interp_s_values[bin_idx])

        return new

    @staticmethod
    def seamless_clone(new, frame, img_mask):
        """ Seamless clone the swapped image into the old frame with cv2 """
        np.clip(new, 0.0, 255.0, out=new)
        height, width, _ = frame.shape
        height = height // 2
        width = width // 2
        y_indices, x_indices, _ = np.nonzero(img_mask)
        y_crop = slice(np.min(y_indices), np.max(y_indices))
        x_crop = slice(np.min(x_indices), np.max(x_indices))
        y_center = int(np.rint((np.max(y_indices) + np.min(y_indices)) / 2 + height))
        x_center = int(np.rint((np.max(x_indices) + np.min(x_indices)) / 2 + width))

        insertion = np.rint(new[y_crop, x_crop]).astype('uint8')
        insertion_mask = img_mask[y_crop, x_crop]
        insertion_mask[insertion_mask != 0] = 255
        insertion_mask = insertion_mask.astype('uint8')
        prior = np.pad(frame, ((height, height), (width, width), (0, 0)), 'constant')
        prior = prior.astype('uint8')

        blended = cv2.seamlessClone(insertion,  # pylint: disable=no-member
                                    prior,
                                    insertion_mask,
                                    (x_center, y_center),
                                    cv2.NORMAL_CLONE)  # pylint: disable=no-member
        blended = blended[height:-height, width:-width]

        return blended


class Mask():
    """ Return the requested mask """

    def __init__(self, mask_type, training_size, padding, crop):
        """ Set requested mask """
        logger.debug("Initializing %s: (mask_type: '%s', training_size: %s, padding: %s)",
                     self.__class__.__name__, mask_type, training_size, padding)

        self.training_size = training_size
        self.padding = padding
        self.mask_type = mask_type
        self.crop = crop

        logger.debug("Initialized %s", self.__class__.__name__)

    def get_mask(self, detected_face, image_size):
        """ Return a face mask """
        kwargs = {"matrix": detected_face.adjusted_matrix,
                  "interpolators": detected_face.adjusted_interpolators,
                  "landmarks": detected_face.landmarks_as_xy,
                  "image_size": image_size}
        logger.trace("kwargs: %s", kwargs)
        mask = getattr(self, self.mask_type)(**kwargs)
        mask = self.finalize_mask(mask)
        logger.trace("mask shape: %s", mask.shape)
        return mask

    def cnn(self, **kwargs):
        """ CNN Mask """
        # Insert FCN-VGG16 segmentation mask model here
        logger.info("cnn not yet implemented, using facehull instead")
        return self.facehull(**kwargs)

    def rect(self, **kwargs):
        """ Namespace for rect mask. This is the same as 'none' in the cli """
        return self.none(**kwargs)

    def none(self, **kwargs):
        """ Rect Mask """
        logger.trace("Getting mask")
        interpolator = kwargs["interpolators"][1]
        ones = np.zeros((self.training_size, self.training_size, 3), dtype='float32')
        mask = np.zeros((kwargs["image_size"][1], kwargs["image_size"][0], 3), dtype='float32')
        # central_core = slice(self.padding, -self.padding)
        ones[self.crop, self.crop] = 1.0
        cv2.warpAffine(ones,  # pylint: disable=no-member
                       kwargs["matrix"],
                       kwargs["image_size"],
                       mask,
                       flags=cv2.WARP_INVERSE_MAP | interpolator,  # pylint: disable=no-member
                       borderMode=cv2.BORDER_CONSTANT,  # pylint: disable=no-member
                       borderValue=0.0)
        return mask

    def dfl(self, **kwargs):
        """ DFaker Mask """
        logger.trace("Getting mask")
        dummy = np.zeros((kwargs["image_size"][1], kwargs["image_size"][0], 3), dtype='float32')
        mask = dfl_full(kwargs["landmarks"], dummy, channels=3)
        mask = self.intersect_rect(mask, **kwargs)
        return mask

    def facehull(self, **kwargs):
        """ Facehull Mask """
        logger.trace("Getting mask")
        mask = np.zeros((kwargs["image_size"][1], kwargs["image_size"][0], 3), dtype='float32')
        hull = cv2.convexHull(  # pylint: disable=no-member
            np.array(kwargs["landmarks"]).reshape((-1, 2)))
        cv2.fillConvexPoly(mask,  # pylint: disable=no-member
                           hull,
                           (1.0, 1.0, 1.0),
                           lineType=cv2.LINE_AA)  # pylint: disable=no-member
        mask = self.intersect_rect(mask, **kwargs)
        return mask

    @staticmethod
    def ellipse(**kwargs):
        """ Ellipse Mask """
        logger.trace("Getting mask")
        mask = np.zeros((kwargs["image_size"][1], kwargs["image_size"][0], 3), dtype='float32')
        ell = cv2.fitEllipse(  # pylint: disable=no-member
            np.array(kwargs["landmarks"]).reshape((-1, 2)))
        cv2.ellipse(mask,  # pylint: disable=no-member
                    box=ell,
                    color=(1.0, 1.0, 1.0),
                    thickness=-1)
        return mask

    def intersect_rect(self, hull_mask, **kwargs):
        """ Intersect the given hull mask with the roi """
        logger.trace("Intersecting rect")
        mask = self.rect(**kwargs)
        mask *= hull_mask
        return mask

    @staticmethod
    def finalize_mask(mask):
        """ Finalize the mask """
        logger.trace("Finalizing mask")
        np.nan_to_num(mask, copy=False)
        np.clip(mask, 0.0, 1.0, out=mask)
        return mask
