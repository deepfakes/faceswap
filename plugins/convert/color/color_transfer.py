#!/usr/bin/env python3
""" Color Transfer adjustment color matching adjustment plugin for faceswap.py converter
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
    THE SOFTWARE. """

import cv2
import numpy as np
from ._base import Adjustment


class Color(Adjustment):
    """
    Transfers the color distribution from the source to the target
    image using the mean and standard deviations of the L*a*b*
    color space.

    This implementation is (loosely) based on to the "Color Transfer
    between Images" paper by Reinhard et al., 2001.
    """

    def process(self, old_face, new_face, raw_mask):
        """
        Parameters
        ----------
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

        Returns
        -------
        transfer: NumPy array
            OpenCV image (w, h, 3) NumPy array (uint8)
        """
        clip = self.config.get("clip", True)
        preserve_paper = self.config.get("preserve_paper", True)

        # convert the images from the RGB to L*ab* color space, being
        # sure to utilizing the floating point data type (note: OpenCV
        # expects floats to be 32-bit, so use that instead of 64-bit)
        source = cv2.cvtColor(  # pylint:disable=no-member
            np.rint(old_face * raw_mask * 255.0).astype("uint8"),
            cv2.COLOR_BGR2LAB).astype("float32")  # pylint:disable=no-member
        target = cv2.cvtColor(  # pylint:disable=no-member
            np.rint(new_face * raw_mask * 255.0).astype("uint8"),
            cv2.COLOR_BGR2LAB).astype("float32")  # pylint:disable=no-member
        # compute color statistics for the source and target images
        (l_mean_src, l_std_src,
         a_mean_src, a_std_src,
         b_mean_src, b_std_src) = self.image_stats(source)
        (l_mean_tar, l_std_tar,
         a_mean_tar, a_std_tar,
         b_mean_tar, b_std_tar) = self.image_stats(target)

        # subtract the means from the target image
        (light, col_a, col_b) = cv2.split(target)  # pylint:disable=no-member
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
        transfer = cv2.merge([light, col_a, col_b])  # pylint:disable=no-member
        transfer = cv2.cvtColor(  # pylint:disable=no-member
            transfer.astype("uint8"),
            cv2.COLOR_LAB2BGR).astype("float32") / 255.0  # pylint:disable=no-member
        background = new_face * (1 - raw_mask)
        merged = transfer + background
        # return the color transferred image
        return merged

    @staticmethod
    def image_stats(image):
        """
        Parameters
        ----------

        image: NumPy array
            OpenCV image in L*a*b* color space

        Returns
        -------
        Tuple of mean and standard deviations for the L*, a*, and b*
        channels, respectively
        """
        # compute the mean and standard deviation of each channel
        (light, col_a, col_b) = cv2.split(image)  # pylint:disable=no-member
        (l_mean, l_std) = (light.mean(), light.std())
        (a_mean, a_std) = (col_a.mean(), col_a.std())
        (b_mean, b_std) = (col_b.mean(), col_b.std())

        # return the color statistics
        return (l_mean, l_std, a_mean, a_std, b_mean, b_std)

    @staticmethod
    def _min_max_scale(arr, new_range=(0, 255)):
        """
        Perform min-max scaling to a NumPy array

        Parameters
        ----------
        arr: NumPy array to be scaled to [new_min, new_max] range
        new_range: tuple of form (min, max) specifying range of
            transformed array

        Returns
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

        Parameters
        ----------
        arr: array to be trimmed to [0, 255] range
        clip: should array be scaled by np.clip? if False then input
            array will be min-max scaled to range
            [max([arr.min(), 0]), min([arr.max(), 255])]

        Returns
        -------
        NumPy array that has been scaled to be in [0, 255] range
        """
        if clip:
            scaled = np.clip(arr, 0, 255)
        else:
            scale_range = (max([arr.min(), 0]), min([arr.max(), 255]))
            scaled = self._min_max_scale(arr, new_range=scale_range)

        return scaled
