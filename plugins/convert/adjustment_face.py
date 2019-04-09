#!/usr/bin/env python3
""" Adjustments for the swap box for faceswap.py converter
    Based on: https://gist.github.com/anonymous/d3815aba83a8f79779451262599b0955
    found on https://www.reddit.com/r/deepfakes/ """

import cv2

from ._base import Adjustment, logger, np


class PreWarpFace(Adjustment):
    """ Adjustments for the face applied prior to warp """

    @property
    def func_names(self):
        return ["avg_color_adjust", "match_histogram", "seamless_clone"]

    # Add Functions
    def add_avg_color_adjust_func(self, action):
        """ Add the average color adjust function to funcs if requested """
        do_add = hasattr(self.args, action) and getattr(self.args, action)
        self.add_function(action, do_add)

    def add_match_histogram_func(self, action):
        """ Add the match histogram function to funcs if requested """
        do_add = hasattr(self.args, action) and getattr(self.args, action)
        self.add_function(action, do_add)

    def add_seamless_clone_func(self, action):
        """ Add the seamless clone function to funcs if requested """
        do_add = hasattr(self.args, action) and getattr(self.args, action)
        self.add_function(action, do_add)

    # IMAGE MANIPULATIONS
    @staticmethod
    def avg_color_adjust(old_face, new_face, raw_mask):
        """ Adjust the mean of the color channels to be the same for the swap and old frame """
        for _ in [0, 1]:
            diff = old_face - new_face
            avg_diff = np.sum(diff * raw_mask, axis=(0, 1))
            adjustment = avg_diff / np.sum(raw_mask, axis=(0, 1))
            new_face += adjustment
        return new_face

    def match_histogram(self, old_face, new_face, raw_mask):
        """ Match the histogram of the color intensity of each channel """
        mask_indices = np.nonzero(raw_mask.squeeze())
        new_face = [self.hist_match(old_face[:, :, c],
                                    new_face[:, :, c],
                                    mask_indices,
                                    self.config["threshold"] / 100)
                    for c in range(3)]
        new_face = np.stack(new_face, axis=-1)
        return new_face

    @staticmethod
    def hist_match(old_channel, new_channel, mask_indices, threshold):
        """  Construct the histogram of the color intensity of a channel
             for the swap and the original. Match the histogram of the original
             by interpolation
        """
        if mask_indices[0].size == 0:
            return new_channel

        old_masked = old_channel[mask_indices]
        new_masked = new_channel[mask_indices]
        _, bin_idx, s_counts = np.unique(new_masked, return_inverse=True, return_counts=True)
        t_values, t_counts = np.unique(old_masked, return_counts=True)
        s_quants = np.cumsum(s_counts, dtype='float32')
        t_quants = np.cumsum(t_counts, dtype='float32')
        s_quants = threshold * s_quants / s_quants[-1]  # cdf
        t_quants /= t_quants[-1]  # cdf
        interp_s_values = np.interp(s_quants, t_quants, t_values)
        new_channel[mask_indices] = interp_s_values[bin_idx]
        return new_channel

    @staticmethod
    def seamless_clone(old_face, new_face, raw_mask):
        """ Seamless clone the swapped face into the old face with cv2 """
        height, width, _ = old_face.shape
        height = height // 2
        width = width // 2

        y_indices, x_indices, _ = np.nonzero(raw_mask)
        y_crop = slice(np.min(y_indices), np.max(y_indices))
        x_crop = slice(np.min(x_indices), np.max(x_indices))
        y_center = int(np.rint((np.max(y_indices) + np.min(y_indices)) / 2 + height))
        x_center = int(np.rint((np.max(x_indices) + np.min(x_indices)) / 2 + width))

        insertion = np.rint(new_face[y_crop, x_crop] * 255.0).astype("uint8")
        insertion_mask = np.rint(raw_mask[y_crop, x_crop] * 255.0).astype("uint8")
        insertion_mask[insertion_mask != 0] = 255
        prior = np.rint(np.pad(old_face * 255.0,
                               ((height, height), (width, width), (0, 0)),
                               'constant')).astype("uint8")

        blended = cv2.seamlessClone(insertion,  # pylint: disable=no-member
                                    prior,
                                    insertion_mask,
                                    (x_center, y_center),
                                    cv2.NORMAL_CLONE)  # pylint: disable=no-member
        blended = blended[height:-height, width:-width]

        return blended.astype("float32") / 255.0


class PostWarpFace(Adjustment):
    """ Adjustments for the face applied after warp to final frame """
    @property
    def func_names(self):
        return ["sharpen_image"]

    # Add functions
    def add_sharpen_image_func(self, action):
        """ Add the sharpen image function to funcs if requested """
        do_add = self.config["sharpen_image"].get("method", None) is not None
        self.add_function(action, do_add)

    # Functions
    def sharpen_image(self, new_face):
        """ Sharpen using the unsharp=mask technique, subtracting a blurried image """
        config = self.config["sharpen_image"]
        amount = config["amount"] / 100.0
        kernel_size, center = self.get_kernel_size(new_face, config["radius"])

        if config["method"] == "box":
            kernel = np.zeros(kernel_size, dtype="float32")
            kernel[center, center] = 1.0
            box_filter = np.ones(kernel_size, dtype="float32") / kernel_size[0]**2
            kernel = kernel + (kernel - box_filter) * amount
            new_face = cv2.filter2D(new_face, -1, kernel)  # pylint: disable=no-member
        elif config["method"] == "gaussian":
            blur = cv2.GaussianBlur(new_face, kernel_size, 0)  # pylint: disable=no-member
            new_face = cv2.addWeighted(new_face,  # pylint: disable=no-member
                                       1.0 + (0.5 * amount),
                                       blur,
                                       -(0.5 * amount),
                                       0)
        elif config["method"] == "unsharp_mask":
            threshold = config["threshold"] / 255.0
            amount = config["amount"] / 100.0
            blur = cv2.GaussianBlur(new_face, kernel_size, 0)  # pylint: disable=no-member
            low_contrast_mask = (abs(new_face - blur) < threshold).astype("float32")
            sharpened = (new_face * (1.0 + amount)) + (blur * -amount)
            new_face = (new_face * (1.0 - low_contrast_mask)) + (sharpened * low_contrast_mask)
        return new_face

    @staticmethod
    def get_kernel_size(new_face, radius_percent):
        """ Return the kernel size and central point for the given radius
            relative to frame width """
        radius = round(new_face.shape[1] * radius_percent / 100)
        radius = 1 if radius < 1 else radius
        kernel_size = (radius * 2) + 1
        kernel_size = (kernel_size, kernel_size)
        logger.trace(kernel_size)
        return kernel_size, radius
