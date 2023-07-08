#!/usr/bin/env python3
""" TF Keras implementation of Perceptual Loss Functions for faceswap.py """

import logging
import typing as T

import numpy as np
import tensorflow as tf

# Ignore linting errors from Tensorflow's thoroughly broken import system
from tensorflow.keras import backend as K  # pylint:disable=import-error

from lib.keras_utils import ColorSpaceConvert, frobenius_norm, replicate_pad

logger = logging.getLogger(__name__)


class DSSIMObjective():  # pylint:disable=too-few-public-methods
    """ DSSIM Loss Functions

    Difference of Structural Similarity (DSSIM loss function).

    Adapted from :func:`tensorflow.image.ssim` for a pure keras implentation.

    Notes
    -----
    Channels last only. Assumes all input images are the same size and square

    Parameters
    ----------
    k_1: float, optional
        Parameter of the SSIM. Default: `0.01`
    k_2: float, optional
        Parameter of the SSIM. Default: `0.03`
    filter_size: int, optional
        size of gaussian filter Default: `11`
    filter_sigma: float, optional
        Width of gaussian filter Default: `1.5`
    max_value: float, optional
        Max value of the output. Default: `1.0`

    Notes
    ------
    You should add a regularization term like a l2 loss in addition to this one.
    """
    def __init__(self,
                 k_1: float = 0.01,
                 k_2: float = 0.03,
                 filter_size: int = 11,
                 filter_sigma: float = 1.5,
                 max_value: float = 1.0) -> None:
        self._filter_size = filter_size
        self._filter_sigma = filter_sigma
        self._kernel = self._get_kernel()

        compensation = 1.0
        self._c1 = (k_1 * max_value) ** 2
        self._c2 = ((k_2 * max_value) ** 2) * compensation

    def _get_kernel(self) -> tf.Tensor:
        """ Obtain the base kernel for performing depthwise convolution.

        Returns
        -------
        :class:`tf.Tensor`
            The gaussian kernel based on selected size and sigma
        """
        coords = np.arange(self._filter_size, dtype="float32")
        coords -= (self._filter_size - 1) / 2.

        kernel = np.square(coords)
        kernel *= -0.5 / np.square(self._filter_sigma)
        kernel = np.reshape(kernel, (1, -1)) + np.reshape(kernel, (-1, 1))
        kernel = K.constant(np.reshape(kernel, (1, -1)))
        kernel = K.softmax(kernel)
        kernel = K.reshape(kernel, (self._filter_size, self._filter_size, 1, 1))
        return kernel

    @classmethod
    def _depthwise_conv2d(cls, image: tf.Tensor, kernel: tf.Tensor) -> tf.Tensor:
        """ Perform a standardized depthwise convolution.

        Parameters
        ----------
        image: :class:`tf.Tensor`
            Batch of images, channels last, to perform depthwise convolution
        kernel: :class:`tf.Tensor`
            convolution kernel

        Returns
        -------
        :class:`tf.Tensor`
            The output from the convolution
        """
        return K.depthwise_conv2d(image, kernel, strides=(1, 1), padding="valid")

    def _get_ssim(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        """ Obtain the structural similarity between a batch of true and predicted images.

        Parameters
        ----------
        y_true: :class:`tf.Tensor`
            The input batch of ground truth images
        y_pred: :class:`tf.Tensor`
            The input batch of predicted images

        Returns
        -------
        :class:`tf.Tensor`
            The SSIM for the given images
        :class:`tf.Tensor`
            The Contrast for the given images
        """
        channels = K.int_shape(y_true)[-1]
        kernel = K.tile(self._kernel, (1, 1, channels, 1))

        # SSIM luminance measure is (2 * mu_x * mu_y + c1) / (mu_x ** 2 + mu_y ** 2 + c1)
        mean_true = self._depthwise_conv2d(y_true, kernel)
        mean_pred = self._depthwise_conv2d(y_pred, kernel)
        num_lum = mean_true * mean_pred * 2.0
        den_lum = K.square(mean_true) + K.square(mean_pred)
        luminance = (num_lum + self._c1) / (den_lum + self._c1)

        # SSIM contrast-structure measure is (2 * cov_{xy} + c2) / (cov_{xx} + cov_{yy} + c2)
        num_con = self._depthwise_conv2d(y_true * y_pred, kernel) * 2.0
        den_con = self._depthwise_conv2d(K.square(y_true) + K.square(y_pred), kernel)

        contrast = (num_con - num_lum + self._c2) / (den_con - den_lum + self._c2)

        # Average over the height x width dimensions
        axes = (-3, -2)
        ssim = K.mean(luminance * contrast, axis=axes)
        contrast = K.mean(contrast, axis=axes)

        return ssim, contrast

    def __call__(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """ Call the DSSIM  or MS-DSSIM Loss Function.

        Parameters
        ----------
        y_true: :class:`tf.Tensor`
            The input batch of ground truth images
        y_pred: :class:`tf.Tensor`
            The input batch of predicted images

        Returns
        -------
        :class:`tf.Tensor`
            The DSSIM or MS-DSSIM for the given images
        """
        ssim = self._get_ssim(y_true, y_pred)[0]
        retval = (1. - ssim) / 2.0
        return K.mean(retval)


class GMSDLoss():  # pylint:disable=too-few-public-methods
    """ Gradient Magnitude Similarity Deviation Loss.

    Improved image quality metric over MS-SSIM with easier calculations

    References
    ----------
    http://www4.comp.polyu.edu.hk/~cslzhang/IQA/GMSD/GMSD.htm
    https://arxiv.org/ftp/arxiv/papers/1308/1308.3052.pdf
    """
    def __call__(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """ Return the Gradient Magnitude Similarity Deviation Loss.

        Parameters
        ----------
        y_true: :class:`tf.Tensor`
            The ground truth value
        y_pred: :class:`tf.Tensor`
            The predicted value

        Returns
        -------
        :class:`tf.Tensor`
            The loss value
        """
        true_edge = self._scharr_edges(y_true, True)
        pred_edge = self._scharr_edges(y_pred, True)
        ephsilon = 0.0025
        upper = 2.0 * true_edge * pred_edge
        lower = K.square(true_edge) + K.square(pred_edge)
        gms = (upper + ephsilon) / (lower + ephsilon)
        gmsd = K.std(gms, axis=(1, 2, 3), keepdims=True)
        gmsd = K.squeeze(gmsd, axis=-1)
        return gmsd

    @classmethod
    def _scharr_edges(cls, image: tf.Tensor, magnitude: bool) -> tf.Tensor:
        """ Returns a tensor holding modified Scharr edge maps.

        Parameters
        ----------
        image: :class:`tf.Tensor`
            Image tensor with shape [batch_size, h, w, d] and type float32. The image(s) must be
            2x2 or larger.
        magnitude: bool
            Boolean to determine if the edge magnitude or edge direction is returned

        Returns
        -------
        :class:`tf.Tensor`
            Tensor holding edge maps for each channel. Returns a tensor with shape `[batch_size, h,
            w, d, 2]` where the last two dimensions hold `[[dy[0], dx[0]], [dy[1], dx[1]], ...,
            [dy[d-1], dx[d-1]]]` calculated using the Scharr filter.
        """

        # Define vertical and horizontal Scharr filters.
        static_image_shape = image.get_shape()
        image_shape = K.shape(image)

        # 5x5 modified Scharr kernel ( reshape to (5,5,1,2) )
        matrix = np.array([[[[0.00070, 0.00070]],
                            [[0.00520, 0.00370]],
                            [[0.03700, 0.00000]],
                            [[0.00520, -0.0037]],
                            [[0.00070, -0.0007]]],
                           [[[0.00370, 0.00520]],
                            [[0.11870, 0.11870]],
                            [[0.25890, 0.00000]],
                            [[0.11870, -0.1187]],
                            [[0.00370, -0.0052]]],
                           [[[0.00000, 0.03700]],
                            [[0.00000, 0.25890]],
                            [[0.00000, 0.00000]],
                            [[0.00000, -0.2589]],
                            [[0.00000, -0.0370]]],
                           [[[-0.0037, 0.00520]],
                            [[-0.1187, 0.11870]],
                            [[-0.2589, 0.00000]],
                            [[-0.1187, -0.1187]],
                            [[-0.0037, -0.0052]]],
                           [[[-0.0007, 0.00070]],
                            [[-0.0052, 0.00370]],
                            [[-0.0370, 0.00000]],
                            [[-0.0052, -0.0037]],
                            [[-0.0007, -0.0007]]]])
        num_kernels = [2]
        kernels = K.constant(matrix, dtype='float32')
        kernels = K.tile(kernels, [1, 1, image_shape[-1], 1])

        # Use depth-wise convolution to calculate edge maps per channel.
        # Output tensor has shape [batch_size, h, w, d * num_kernels].
        pad_sizes = [[0, 0], [2, 2], [2, 2], [0, 0]]
        padded = tf.pad(image,  # pylint:disable=unexpected-keyword-arg,no-value-for-parameter
                        pad_sizes,
                        mode='REFLECT')
        output = K.depthwise_conv2d(padded, kernels)

        if not magnitude:  # direction of edges
            # Reshape to [batch_size, h, w, d, num_kernels].
            shape = K.concatenate([image_shape, num_kernels], axis=0)
            output = K.reshape(output, shape=shape)
            output.set_shape(static_image_shape.concatenate(num_kernels))
            output = tf.atan(K.squeeze(output[:, :, :, :, 0] / output[:, :, :, :, 1], axis=None))
        # magnitude of edges -- unified x & y edges don't work well with Neural Networks
        return output


class LDRFLIPLoss():  # pylint:disable=too-few-public-methods
    """ Computes the LDR-FLIP error map between two LDR images, assuming the images are observed
    at a certain number of pixels per degree of visual angle.

    References
    ----------
    https://research.nvidia.com/sites/default/files/node/3260/FLIP_Paper.pdf
    https://github.com/NVlabs/flip

    License
    -------
    BSD 3-Clause License
    Copyright (c) 2020-2022, NVIDIA Corporation & AFFILIATES. All rights reserved.
    Redistribution and use in source and binary forms, with or without modification, are permitted
    provided that the following conditions are met:
    Redistributions of source code must retain the above copyright notice, this list of conditions
    and the following disclaimer.
    Redistributions in binary form must reproduce the above copyright notice, this list of
    conditions and the following disclaimer in the documentation and/or other materials provided
    with the distribution.
    Neither the name of the copyright holder nor the names of its contributors may be used to
    endorse or promote products derived from this software without specific prior written
    permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
    IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY
    AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
    CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
    CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
    OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
    POSSIBILITY OF SUCH DAMAGE.

    Parameters
    ----------
    computed_distance_exponent: float, Optional
        The computed distance exponent to apply to Hunt adjusted, filtered colors.
        (`qc` in original paper). Default: `0.7`
    feature_exponent: float, Optional
        The feature exponent to apply for increasing the impact of feature difference on the
        final loss value. (`qf` in original paper). Default: `0.5`
    lower_threshold_exponent: float, Optional
        The `pc` exponent for the color pipeline as described in the original paper: Default: `0.4`
    upper_threshold_exponent: float, Optional
        The `pt` exponent  for the color pipeline as described in the original paper.
        Default: `0.95`
    epsilon: float
        A small value to improve training stability. Default: `1e-15`
    pixels_per_degree: float, Optional
        The estimated number of pixels per degree of visual angle of the observer. This effectively
        impacts the tolerance when calculating loss. The default corresponds to viewing images on a
        0.7m wide 4K monitor at 0.7m from the display. Default: ``None``
    color_order: str
        The `"BGR"` or `"RGB"` color order of the incoming images
    """
    def __init__(self,
                 computed_distance_exponent: float = 0.7,
                 feature_exponent: float = 0.5,
                 lower_threshold_exponent: float = 0.4,
                 upper_threshold_exponent: float = 0.95,
                 epsilon: float = 1e-15,
                 pixels_per_degree: float | None = None,
                 color_order: T.Literal["bgr", "rgb"] = "bgr") -> None:
        logger.debug("Initializing: %s (computed_distance_exponent '%s', feature_exponent: %s, "
                     "lower_threshold_exponent: %s, upper_threshold_exponent: %s, epsilon: %s, "
                     "pixels_per_degree: %s, color_order: %s)", self.__class__.__name__,
                     computed_distance_exponent, feature_exponent, lower_threshold_exponent,
                     upper_threshold_exponent, epsilon, pixels_per_degree, color_order)

        self._computed_distance_exponent = computed_distance_exponent
        self._feature_exponent = feature_exponent
        self._pc = lower_threshold_exponent
        self._pt = upper_threshold_exponent
        self._epsilon = epsilon
        self._color_order = color_order.lower()

        if pixels_per_degree is None:
            pixels_per_degree = (0.7 * 3840 / 0.7) * np.pi / 180
        self._pixels_per_degree = pixels_per_degree
        self._spatial_filters = _SpatialFilters(pixels_per_degree)
        self._feature_detector = _FeatureDetection(pixels_per_degree)
        logger.debug("Initialized: %s ", self.__class__.__name__)

    def __call__(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """ Call the LDR Flip Loss Function

        Parameters
        ----------
        y_true: :class:`tensorflow.Tensor`
            The ground truth batch of images
        y_pred: :class:`tensorflow.Tensor`
            The predicted batch of images

        Returns
        -------
        :class::class:`tensorflow.Tensor`
            The calculated Flip loss value
        """
        if self._color_order == "bgr":  # Switch models training in bgr order to rgb
            y_true = y_true[..., 2::-1]
            y_pred = y_pred[..., 2::-1]

        y_true = K.clip(y_true, 0, 1.)
        y_pred = K.clip(y_pred, 0, 1.)

        rgb2ycxcz = ColorSpaceConvert("srgb", "ycxcz")
        true_ycxcz = rgb2ycxcz(y_true)
        pred_ycxcz = rgb2ycxcz(y_pred)

        delta_e_color = self._color_pipeline(true_ycxcz, pred_ycxcz)
        delta_e_features = self._process_features(true_ycxcz, pred_ycxcz)

        loss = K.pow(delta_e_color, 1 - delta_e_features)
        return loss

    def _color_pipeline(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """ Perform the color processing part of the FLIP loss function

        Parameters
        ----------
        y_true: :class:`tensorflow.Tensor`
            The ground truth batch of images in YCxCz color space
        y_pred: :class:`tensorflow.Tensor`
            The predicted batch of images in YCxCz color space

        Returns
        -------
        :class:`tensorflow.Tensor`
            The exponentiated, maximum HyAB difference between two colors in Hunt-adjusted
            L*A*B* space
        """
        filtered_true = self._spatial_filters(y_true)
        filtered_pred = self._spatial_filters(y_pred)

        rgb2lab = ColorSpaceConvert(from_space="rgb", to_space="lab")
        preprocessed_true = self._hunt_adjustment(rgb2lab(filtered_true))
        preprocessed_pred = self._hunt_adjustment(rgb2lab(filtered_pred))
        hunt_adjusted_green = self._hunt_adjustment(rgb2lab(K.constant([[[[0.0, 1.0, 0.0]]]],
                                                                       dtype="float32")))
        hunt_adjusted_blue = self._hunt_adjustment(rgb2lab(K.constant([[[[0.0, 0.0, 1.0]]]],
                                                                      dtype="float32")))

        delta = self._hyab(preprocessed_true, preprocessed_pred)
        power_delta = K.pow(delta, self._computed_distance_exponent)
        cmax = K.pow(self._hyab(hunt_adjusted_green, hunt_adjusted_blue),
                     self._computed_distance_exponent)
        return self._redistribute_errors(power_delta, cmax)

    def _process_features(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """ Perform the color processing part of the FLIP loss function

        Parameters
        ----------
        y_true: :class:`tensorflow.Tensor`
            The ground truth batch of images in YCxCz color space
        y_pred: :class:`tensorflow.Tensor`
            The predicted batch of images in YCxCz color space

        Returns
        -------
        :class:`tensorflow.Tensor`
            The exponentiated features delta
        """
        col_y_true = (y_true[..., 0:1] + 16) / 116.
        col_y_pred = (y_pred[..., 0:1] + 16) / 116.

        edges_true = self._feature_detector(col_y_true, "edge")
        points_true = self._feature_detector(col_y_true, "point")
        edges_pred = self._feature_detector(col_y_pred, "edge")
        points_pred = self._feature_detector(col_y_pred, "point")

        delta = K.maximum(K.abs(frobenius_norm(edges_true) - frobenius_norm(edges_pred)),
                          K.abs(frobenius_norm(points_pred) - frobenius_norm(points_true)))

        delta = K.clip(delta, min_value=self._epsilon, max_value=None)
        return K.pow(((1 / np.sqrt(2)) * delta), self._feature_exponent)

    @classmethod
    def _hunt_adjustment(cls, image: tf.Tensor) -> tf.Tensor:
        """ Apply Hunt-adjustment to an image in L*a*b* color space

        Parameters
        ----------
        image: :class:`tensorflow.Tensor`
            The batch of images in L*a*b* to adjust

        Returns
        -------
        :class:`tensorflow.Tensor`
            The hunt adjusted batch of images in L*a*b color space
        """
        ch_l = image[..., 0:1]
        adjusted = K.concatenate([ch_l, image[..., 1:] * (ch_l * 0.01)], axis=-1)
        return adjusted

    def _hyab(self, y_true, y_pred):
        """ Compute the HyAB distance between true and predicted images.

        Parameters
        ----------
        y_true: :class:`tensorflow.Tensor`
            The ground truth batch of images in standard or Hunt-adjusted L*A*B* color space
        y_pred: :class:`tensorflow.Tensor`
            The predicted batch of images in in standard or Hunt-adjusted L*A*B* color space

        Returns
        -------
        :class:`tensorflow.Tensor`
            image tensor containing the per-pixel HyAB distances between true and predicted images
        """
        delta = y_true - y_pred
        root = K.sqrt(K.clip(K.pow(delta[..., 0:1], 2), min_value=self._epsilon, max_value=None))
        delta_norm = frobenius_norm(delta[..., 1:3])
        return root + delta_norm

    def _redistribute_errors(self, power_delta_e_hyab, cmax):
        """ Redistribute exponentiated HyAB errors to the [0,1] range

        Parameters
        ----------
        power_delta_e_hyab: :class:`tensorflow.Tensor`
            The exponentiated HyAb distance
        cmax: :class:`tensorflow.Tensor`
            The exponentiated, maximum HyAB difference between two colors in Hunt-adjusted
            L*A*B* space

        Returns
        -------
        :class:`tensorflow.Tensor`
            The redistributed per-pixel HyAB distances (in range [0,1])
        """
        pccmax = self._pc * cmax
        delta_e_c = K.switch(
            power_delta_e_hyab < pccmax,
            (self._pt / pccmax) * power_delta_e_hyab,
            self._pt + ((power_delta_e_hyab - pccmax) / (cmax - pccmax)) * (1.0 - self._pt))
        return delta_e_c


class _SpatialFilters():  # pylint:disable=too-few-public-methods
    """ Filters an image with channel specific spatial contrast sensitivity functions and clips
    result to the unit cube in linear RGB.

    For use with LDRFlipLoss.

    Parameters
    ----------
    pixels_per_degree: float
        The estimated number of pixels per degree of visual angle of the observer. This effectively
        impacts the tolerance when calculating loss.
    """
    def __init__(self, pixels_per_degree: float) -> None:
        self._pixels_per_degree = pixels_per_degree
        self._spatial_filters, self._radius = self._generate_spatial_filters()
        self._ycxcz2rgb = ColorSpaceConvert(from_space="ycxcz", to_space="rgb")

    def _generate_spatial_filters(self) -> tuple[tf.Tensor, int]:
        """ Generates spatial contrast sensitivity filters with width depending on the number of
        pixels per degree of visual angle of the observer for channels "A", "RG" and "BY"

        Returns
        -------
        dict
            the channels ("A" (Achromatic CSF), "RG" (Red-Green CSF) or "BY" (Blue-Yellow CSF)) as
            key with the Filter kernel corresponding to the spatial contrast sensitivity function
            of channel and kernel's radius
        """
        mapping = {"A": {"a1": 1, "b1": 0.0047, "a2": 0, "b2": 1e-5},
                   "RG": {"a1": 1, "b1": 0.0053, "a2": 0, "b2": 1e-5},
                   "BY": {"a1": 34.1, "b1": 0.04, "a2": 13.5, "b2": 0.025}}

        domain, radius = self._get_evaluation_domain(mapping["A"]["b1"],
                                                     mapping["A"]["b2"],
                                                     mapping["RG"]["b1"],
                                                     mapping["RG"]["b2"],
                                                     mapping["BY"]["b1"],
                                                     mapping["BY"]["b2"])

        weights = np.array([self._generate_weights(mapping[channel], domain)
                            for channel in ("A", "RG", "BY")])
        weights = K.constant(np.moveaxis(weights, 0, -1), dtype="float32")

        return weights, radius

    def _get_evaluation_domain(self,
                               b1_a: float,
                               b2_a: float,
                               b1_rg: float,
                               b2_rg: float,
                               b1_by: float,
                               b2_by: float) -> tuple[np.ndarray, int]:
        """ TODO docstring """
        max_scale_parameter = max([b1_a, b2_a, b1_rg, b2_rg, b1_by, b2_by])
        delta_x = 1.0 / self._pixels_per_degree
        radius = int(np.ceil(3 * np.sqrt(max_scale_parameter / (2 * np.pi**2))
                             * self._pixels_per_degree))
        ax_x, ax_y = np.meshgrid(range(-radius, radius + 1), range(-radius, radius + 1))
        domain = (ax_x * delta_x) ** 2 + (ax_y * delta_x) ** 2
        return domain, radius

    @classmethod
    def _generate_weights(cls, channel: dict[str, float], domain: np.ndarray) -> tf.Tensor:
        """ TODO docstring """
        a_1, b_1, a_2, b_2 = channel["a1"], channel["b1"], channel["a2"], channel["b2"]
        grad = (a_1 * np.sqrt(np.pi / b_1) * np.exp(-np.pi ** 2 * domain / b_1) +
                a_2 * np.sqrt(np.pi / b_2) * np.exp(-np.pi ** 2 * domain / b_2))
        grad = grad / np.sum(grad)
        grad = np.reshape(grad, (*grad.shape, 1))
        return grad

    def __call__(self, image: tf.Tensor) -> tf.Tensor:
        """ Call the spacial filtering.

        Parameters
        ----------
        image: Tensor
            Image tensor to filter in YCxCz color space

        Returns
        -------
        Tensor
            The input image transformed to linear RGB after filtering with spatial contrast
            sensitivity functions
        """
        padded_image = replicate_pad(image, self._radius)
        image_tilde_opponent = K.conv2d(padded_image,
                                        self._spatial_filters,
                                        strides=1,
                                        padding="valid")
        rgb = K.clip(self._ycxcz2rgb(image_tilde_opponent), 0., 1.)
        return rgb


class _FeatureDetection():  # pylint:disable=too-few-public-methods
    """ Detect features (i.e. edges and points) in an achromatic YCxCz image.

    For use with LDRFlipLoss.

    Parameters
    ----------
    pixels_per_degree: float
        The number of pixels per degree of visual angle of the observer
    """
    def __init__(self, pixels_per_degree: float) -> None:
        width = 0.082
        self._std = 0.5 * width * pixels_per_degree
        self._radius = int(np.ceil(3 * self._std))
        self._grid = np.meshgrid(range(-self._radius, self._radius + 1),
                                 range(-self._radius, self._radius + 1))
        self._gradient = np.exp(-(self._grid[0] ** 2 + self._grid[1] ** 2)
                                / (2 * (self._std ** 2)))

    def __call__(self, image: tf.Tensor, feature_type: str) -> tf.Tensor:
        """ Run the feature detection

        Parameters
        ----------
        image: Tensor
            Batch of images in YCxCz color space with normalized Y values
        feature_type: str
            Type of features to detect (`"edge"` or `"point"`)

        Returns
        -------
        Tensor
            Detected features in the 0-1 range
        """
        feature_type = feature_type.lower()

        if feature_type == 'edge':
            grad_x = np.multiply(-self._grid[0], self._gradient)
        else:
            grad_x = np.multiply(self._grid[0] ** 2 / (self._std ** 2) - 1, self._gradient)

        negative_weights_sum = -np.sum(grad_x[grad_x < 0])
        positive_weights_sum = np.sum(grad_x[grad_x > 0])

        grad_x = K.constant(grad_x)
        grad_x = K.switch(grad_x < 0, grad_x / negative_weights_sum, grad_x / positive_weights_sum)
        kernel = K.expand_dims(K.expand_dims(grad_x, axis=-1), axis=-1)

        features_x = K.conv2d(replicate_pad(image, self._radius),
                              kernel,
                              strides=1,
                              padding="valid")
        kernel = K.permute_dimensions(kernel, (1, 0, 2, 3))
        features_y = K.conv2d(replicate_pad(image, self._radius),
                              kernel,
                              strides=1,
                              padding="valid")
        features = K.concatenate([features_x, features_y], axis=-1)
        return features


class MSSIMLoss():  # pylint:disable=too-few-public-methods
    """ Multiscale Structural Similarity Loss Function

    Parameters
    ----------
    k_1: float, optional
        Parameter of the SSIM. Default: `0.01`
    k_2: float, optional
        Parameter of the SSIM. Default: `0.03`
    filter_size: int, optional
        size of gaussian filter Default: `11`
    filter_sigma: float, optional
        Width of gaussian filter Default: `1.5`
    max_value: float, optional
        Max value of the output. Default: `1.0`
    power_factors: tuple, optional
        Iterable of weights for each of the scales. The number of scales used is the length of the
        list. Index 0 is the unscaled resolution's weight and each increasing scale corresponds to
        the image being downsampled by 2. Defaults to the values obtained in the original paper.
        Default: (0.0448, 0.2856, 0.3001, 0.2363, 0.1333)

    Notes
    ------
    You should add a regularization term like a l2 loss in addition to this one.
    """
    def __init__(self,
                 k_1: float = 0.01,
                 k_2: float = 0.03,
                 filter_size: int = 11,
                 filter_sigma: float = 1.5,
                 max_value: float = 1.0,
                 power_factors: tuple[float, ...] = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333)
                 ) -> None:
        self.filter_size = filter_size
        self.filter_sigma = filter_sigma
        self.k_1 = k_1
        self.k_2 = k_2
        self.max_value = max_value
        self.power_factors = power_factors

    def __call__(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """ Call the MS-SSIM Loss Function.

        Parameters
        ----------
        y_true: :class:`tf.Tensor`
            The ground truth value
        y_pred: :class:`tf.Tensor`
            The predicted value

        Returns
        -------
        :class:`tf.Tensor`
            The MS-SSIM Loss value
        """
        im_size = K.int_shape(y_true)[1]
        # filter size cannot be larger than the smallest scale
        smallest_scale = self._get_smallest_size(im_size, len(self.power_factors) - 1)
        filter_size = min(self.filter_size, smallest_scale)

        ms_ssim = tf.image.ssim_multiscale(y_true,
                                           y_pred,
                                           self.max_value,
                                           power_factors=self.power_factors,
                                           filter_size=filter_size,
                                           filter_sigma=self.filter_sigma,
                                           k1=self.k_1,
                                           k2=self.k_2)
        ms_ssim_loss = 1. - ms_ssim
        return K.mean(ms_ssim_loss)

    def _get_smallest_size(self, size: int, idx: int) -> int:
        """ Recursive function to obtain the smallest size that the image will be scaled to.

        Parameters
        ----------
        size: int
            The current scaled size to iterate through
        idx: int
            The current iteration to be performed. When iteration hits zero the value will
            be returned

        Returns
        -------
        int
            The smallest size the image will be scaled to based on the original image size and
            the amount of scaling factors that will occur
        """
        logger.debug("scale id: %s, size: %s", idx, size)
        if idx > 0:
            size = self._get_smallest_size(size // 2, idx - 1)
        return size
