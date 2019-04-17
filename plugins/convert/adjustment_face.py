#!/usr/bin/env python3
""" Adjustments for the swap box for faceswap.py converter
    Based on: https://gist.github.com/anonymous/d3815aba83a8f79779451262599b0955
    found on https://www.reddit.com/r/deepfakes/ """

import cv2

from ._base import Adjustment, logger, np


class Face(Adjustment):
    """ Adjustments for the face applied prior to warp """

    @property
    def func_names(self):
        return ["color_adjustment", "seamless_clone"]

    # Add Functions
    def add_color_adjustment_func(self, action):
        """ Add the color adjustment function to funcs if requested """
        do_add = getattr(self.args, action).lower() != "none"
        if do_add:
            action = getattr(self.args, action).lower().replace("-", "_")
        self.add_function(action, do_add)

    def add_seamless_clone_func(self, action):
        """ Add the seamless clone function to funcs if requested """
        do_add = hasattr(self.args, action) and getattr(self.args, action)
        self.add_function(action, do_add)

    # <<< IMAGE MANIPULATIONS >>>
    # Average color adjust
    @staticmethod
    def avg_color(old_face, new_face, raw_mask):
        """ Adjust the mean of the color channels to be the same for the swap and old frame """
        for _ in [0, 1]:
            diff = old_face - new_face
            avg_diff = np.sum(diff * raw_mask, axis=(0, 1))
            adjustment = avg_diff / np.sum(raw_mask, axis=(0, 1))
            new_face += adjustment
        return new_face

    # Match Histogram
    def match_hist(self, old_face, new_face, raw_mask):
        """ Match the histogram of the color intensity of each channel """
        config = self.config["match_hist"]
        mask_indices = np.nonzero(raw_mask.squeeze())
        new_face = [self.hist_match(old_face[:, :, c],
                                    new_face[:, :, c],
                                    mask_indices,
                                    config["threshold"] / 100)
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

    # Seamless Clone
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

    # Color Transfer
    def color_transfer(self, old_face, new_face, raw_mask):
        """
        source: https://github.com/jrosebr1/color_transfer
        The MIT License (MIT)

        Copyright (c) 2014 Adrian Rosebrock, http://www.pyimagesearch.com

        Permission is hereby granted, free of charge, to any person obtaining a copy
        of this software and associated documentation files (the "Software"), to deal
        in the Software without restriction, including without limitation the rights
        to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
        copies of the Software, and to permit persons to whom the Software is
        furnished to do so, subject to the following conditions:

        The above copyright notice and this permission notice shall be included in
        all copies or substantial portions of the Software.

        THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
        IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
        FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
        AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
        LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
        OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
        THE SOFTWARE.

        Transfers the color distribution from the source to the target
        image using the mean and standard deviations of the L*a*b*
        color space.

        This implementation is (loosely) based on to the "Color Transfer
        between Images" paper by Reinhard et al., 2001.

        Parameters:
        -------
        source: NumPy array
            OpenCV image in BGR color space (the source image)
        target: NumPy array
            OpenCV image in BGR color space (the target image)
        clip: Should components of L*a*b* image be scaled by np.clip before
            converting back to BGR color space?
            If False then components will be min-max scaled appropriately.
            Clipping will keep target image brightness truer to the input.
            Scaling will adjust image brightness to avoid washed out portions
            in the resulting color transfer that can be caused by clipping.
        preserve_paper: Should color transfer strictly follow methodology
            layed out in original paper? The method does not always produce
            aesthetically pleasing results.
            If False then L*a*b* components will scaled using the reciprocal of
            the scaling factor proposed in the paper.  This method seems to produce
            more consistently aesthetically pleasing results

        Returns:
        -------
        transfer: NumPy array
            OpenCV image (w, h, 3) NumPy array (uint8)
        """
        config = self.config["color_transfer"]
        clip = config.get("clip", True)
        preserve_paper = config.get("preserve_paper", True)

        # convert the images from the RGB to L*ab* color space, being
        # sure to utilizing the floating point data type (note: OpenCV
        # expects floats to be 32-bit, so use that instead of 64-bit)
        source = cv2.cvtColor(  # pylint: disable=no-member
            np.rint(old_face * raw_mask * 255.0).astype("uint8"),
            cv2.COLOR_BGR2LAB).astype("float32")  # pylint: disable=no-member
        target = cv2.cvtColor(  # pylint: disable=no-member
            np.rint(new_face * raw_mask * 255.0).astype("uint8"),
            cv2.COLOR_BGR2LAB).astype("float32")  # pylint: disable=no-member
        # compute color statistics for the source and target images
        (l_mean_src, l_std_src,
         a_mean_src, a_std_src,
         b_mean_src, b_std_src) = self.image_stats(source)
        (l_mean_tar, l_std_tar,
         a_mean_tar, a_std_tar,
         b_mean_tar, b_std_tar) = self.image_stats(target)

        # subtract the means from the target image
        (light, col_a, col_b) = cv2.split(target)  # pylint: disable=no-member
        light -= l_mean_tar
        col_a -= a_mean_tar
        col_b -= b_mean_tar

        if preserve_paper:
            # scale by the standard deviations using paper proposed factor
            light = (l_std_tar / l_std_src) * light
            col_a = (a_std_tar / a_std_src) * col_a
            col_b = (b_std_tar / b_std_src) * col_b
        else:
            # scale by the standard deviations using reciprocal of paper proposed factor
            light = (l_std_src / l_std_tar) * light
            col_a = (a_std_src / a_std_tar) * col_a
            col_b = (b_std_src / b_std_tar) * col_b

        # add in the source mean
        light += l_mean_src
        col_a += a_mean_src
        col_b += b_mean_src

        # clip/scale the pixel intensities to [0, 255] if they fall
        # outside this range
        light = self._scale_array(light, clip=clip)
        col_a = self._scale_array(col_a, clip=clip)
        col_b = self._scale_array(col_b, clip=clip)

        # merge the channels together and convert back to the RGB color
        # space, being sure to utilize the 8-bit unsigned integer data
        # type
        transfer = cv2.merge([light, col_a, col_b])  # pylint: disable=no-member
        transfer = cv2.cvtColor(  # pylint: disable=no-member
            transfer.astype("uint8"),
            cv2.COLOR_LAB2BGR).astype("float32") / 255.0  # pylint: disable=no-member
        background = new_face * (1 - raw_mask)
        merged = transfer + background
        # return the color transferred image
        return merged

    @staticmethod
    def image_stats(image):
        """
        Parameters:
        -------
        image: NumPy array
            OpenCV image in L*a*b* color space

        Returns:
        -------
        Tuple of mean and standard deviations for the L*, a*, and b*
        channels, respectively
        """
        # compute the mean and standard deviation of each channel
        (light, col_a, col_b) = cv2.split(image)  # pylint: disable=no-member
        (l_mean, l_std) = (light.mean(), light.std())
        (a_mean, a_std) = (col_a.mean(), col_a.std())
        (b_mean, b_std) = (col_b.mean(), col_b.std())

        # return the color statistics
        return (l_mean, l_std, a_mean, a_std, b_mean, b_std)

    @staticmethod
    def _min_max_scale(arr, new_range=(0, 255)):
        """
        Perform min-max scaling to a NumPy array

        Parameters:
        -------
        arr: NumPy array to be scaled to [new_min, new_max] range
        new_range: tuple of form (min, max) specifying range of
            transformed array

        Returns:
        -------
        NumPy array that has been scaled to be in
        [new_range[0], new_range[1]] range
        """
        # get array's current min and max
        arr_min = arr.min()
        arr_max = arr.max()

        # check if scaling needs to be done to be in new_range
        if arr_min < new_range[0] or arr_max > new_range[1]:
            # perform min-max scaling
            scaled = (new_range[1] - new_range[0]) * (arr - arr_min) / (arr_max -
                                                                        arr_min) + new_range[0]
        else:
            # return array if already in range
            scaled = arr

        return scaled

    def _scale_array(self, arr, clip=True):
        """
        Trim NumPy array values to be in [0, 255] range with option of
        clipping or scaling.

        Parameters:
        -------
        arr: array to be trimmed to [0, 255] range
        clip: should array be scaled by np.clip? if False then input
            array will be min-max scaled to range
            [max([arr.min(), 0]), min([arr.max(), 255])]

        Returns:
        -------
        NumPy array that has been scaled to be in [0, 255] range
        """
        if clip:
            scaled = np.clip(arr, 0, 255)
        else:
            scale_range = (max([arr.min(), 0]), min([arr.max(), 255]))
            scaled = self._min_max_scale(arr, new_range=scale_range)

        return scaled


class Scaling(Adjustment):
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
        radius = max(1, round(new_face.shape[1] * radius_percent / 100))
        kernel_size = int((radius * 2) + 1)
        kernel_size = (kernel_size, kernel_size)
        logger.trace(kernel_size)
        return kernel_size, radius
