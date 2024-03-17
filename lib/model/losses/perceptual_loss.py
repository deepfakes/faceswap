#!/usr/bin/env python3
""" TF Keras implementation of Perceptual Loss Functions for faceswap.py """

import logging
import typing as T

import numpy as np
import torch

import keras
from keras import ops, Variable

from lib.keras_utils import ColorSpaceConvert, frobenius_norm, replicate_pad

logger = logging.getLogger(__name__)


class DSSIMObjective(keras.losses.Loss):
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
        super().__init__(name=self.__class__.__name__)
        self._filter_size = filter_size
        self._filter_sigma = filter_sigma
        self._kernel = self._get_kernel()

        compensation = 1.0
        self._c1 = (k_1 * max_value) ** 2
        self._c2 = ((k_2 * max_value) ** 2) * compensation

    def _get_kernel(self) -> torch.Tensor:
        """ Obtain the base kernel for performing depthwise convolution.

        Returns
        -------
        :class:`torch.Tensor`
            The gaussian kernel based on selected size and sigma
        """
        coords = np.arange(self._filter_size, dtype="float32")
        coords -= (self._filter_size - 1) / 2.

        kernel = np.square(coords)
        kernel *= -0.5 / np.square(self._filter_sigma)
        kernel = np.reshape(kernel, (1, -1)) + np.reshape(kernel, (-1, 1))
        kernel = Variable(np.reshape(kernel, (1, -1)), trainable=False)
        kernel = ops.softmax(kernel)
        kernel = ops.reshape(kernel, (self._filter_size, self._filter_size, 1, 1))
        return kernel

    @classmethod
    def _depthwise_conv2d(cls, image: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        """ Perform a standardized depthwise convolution.

        Parameters
        ----------
        image: :class:`torch.Tensor`
            Batch of images, channels last, to perform depthwise convolution
        kernel: :class:`torch.Tensor`
            convolution kernel

        Returns
        -------
        :class:`torch.Tensor`
            The output from the convolution
        """
        return ops.depthwise_conv(image, kernel, strides=(1, 1), padding="valid")

    def _get_ssim(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """ Obtain the structural similarity between a batch of true and predicted images.

        Parameters
        ----------
        y_true: :class:`torch.Tensor`
            The input batch of ground truth images
        y_pred: :class:`torch.Tensor`
            The input batch of predicted images

        Returns
        -------
        :class:`torch.Tensor`
            The SSIM for the given images
        :class:`torch.Tensor`
            The Contrast for the given images
        """
        channels = y_true.shape[-1]
        kernel = ops.tile(self._kernel, (1, 1, channels, 1))

        # SSIM luminance measure is (2 * mu_x * mu_y + c1) / (mu_x ** 2 + mu_y ** 2 + c1)
        mean_true = self._depthwise_conv2d(y_true, kernel)
        mean_pred = self._depthwise_conv2d(y_pred, kernel)
        num_lum = mean_true * mean_pred * 2.0
        den_lum = ops.square(mean_true) + ops.square(mean_pred)
        luminance = (num_lum + self._c1) / (den_lum + self._c1)

        # SSIM contrast-structure measure is (2 * cov_{xy} + c2) / (cov_{xx} + cov_{yy} + c2)
        num_con = self._depthwise_conv2d(y_true * y_pred, kernel) * 2.0
        den_con = self._depthwise_conv2d(ops.square(y_true) + ops.square(y_pred), kernel)

        contrast = (num_con - num_lum + self._c2) / (den_con - den_lum + self._c2)

        # Average over the height x width dimensions
        axes = (-3, -2)
        ssim = ops.mean(luminance * contrast, axis=axes)
        contrast = ops.mean(contrast, axis=axes)

        return ssim, contrast

    def call(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """ Call the DSSIM  or MS-DSSIM Loss Function.

        Parameters
        ----------
        y_true: :class:`torch.Tensor`
            The input batch of ground truth images
        y_pred: :class:`torch.Tensor`
            The input batch of predicted images

        Returns
        -------
        :class:`torch.Tensor`
            The DSSIM or MS-DSSIM for the given images
        """
        ssim = self._get_ssim(y_true, y_pred)[0]
        retval = (1. - ssim) / 2.0
        return ops.mean(retval)


class GMSDLoss(keras.losses.Loss):
    """ Gradient Magnitude Similarity Deviation Loss.

    Improved image quality metric over MS-SSIM with easier calculations

    References
    ----------
    http://www4.comp.polyu.edu.hk/~cslzhang/IQA/GMSD/GMSD.htm
    https://arxiv.org/ftp/arxiv/papers/1308/1308.3052.pdf
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, name=self.__class__.__name__, **kwargs)
        self._scharr_edges = Variable(np.array([[[[0.00070, 0.00070]],
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
                                                 [[-0.0007, -0.0007]]]]),
                                      dtype="float32",
                                      trainable=False)

    def _map_scharr_edges(self, image: torch.Tensor, magnitude: bool) -> torch.Tensor:
        """ Returns a tensor holding modified Scharr edge maps.

        Parameters
        ----------
        image: :class:`torch.Tensor`
            Image tensor with shape [batch_size, h, w, d] and type float32. The image(s) must be
            2x2 or larger.
        magnitude: bool
            Boolean to determine if the edge magnitude or edge direction is returned

        Returns
        -------
        :class:`torch.Tensor`
            Tensor holding edge maps for each channel. Returns a tensor with shape `[batch_size, h,
            w, d, 2]` where the last two dimensions hold `[[dy[0], dx[0]], [dy[1], dx[1]], ...,
            [dy[d-1], dx[d-1]]]` calculated using the Scharr filter.
        """
        # Define vertical and horizontal Scharr filters.
        image_shape = image.shape
        num_kernels = [2]

        kernels = ops.tile(self._scharr_edges, [1, 1, image_shape[-1], 1])

        # Use depth-wise convolution to calculate edge maps per channel.
        # Output tensor has shape [batch_size, h, w, d * num_kernels].
        pad_sizes = [[0, 0], [2, 2], [2, 2], [0, 0]]
        padded = ops.pad(image, pad_sizes, mode="reflect")
        output = ops.depthwise_conv(padded, kernels)

        if not magnitude:  # direction of edges
            # Reshape to [batch_size, h, w, d, num_kernels].
            shape = ops.concatenate([image_shape, num_kernels], axis=0)
            output = ops.reshape(output, shape=shape)
            output = ops.reshape(output, ops.concatenate([image_shape, num_kernels]))
            output = torch.atan(ops.squeeze(output[:, :, :, :, 0] / output[:, :, :, :, 1],
                                            axis=None))
        # magnitude of edges -- unified x & y edges don't work well with Neural Networks
        return output

    def call(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """ Return the Gradient Magnitude Similarity Deviation Loss.

        Parameters
        ----------
        y_true: :class:`torch.Tensor`
            The ground truth value
        y_pred: :class:`torch.Tensor`
            The predicted value

        Returns
        -------
        :class:`torch.Tensor`
            The loss value
        """
        true_edge = self._map_scharr_edges(y_true, True)
        pred_edge = self._map_scharr_edges(y_pred, True)
        ephsilon = 0.0025
        upper = 2.0 * true_edge * pred_edge
        lower = ops.square(true_edge) + ops.square(pred_edge)
        gms = (upper + ephsilon) / (lower + ephsilon)
        gmsd = ops.std(gms, axis=(1, 2, 3), keepdims=True)
        gmsd = ops.squeeze(gmsd, axis=-1)
        return gmsd


class LDRFLIPLoss(keras.losses.Loss):
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
        super().__init__(name=self.__class__.__name__)
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
        self._col_conv = {"rgb2lab": ColorSpaceConvert(from_space="rgb", to_space="lab"),
                          "rgb2ycxcz": ColorSpaceConvert("srgb", "ycxcz")}
        self._hunt = {"green": Variable([[[[0.0, 1.0, 0.0]]]], dtype="float32", trainable=False),
                      "blue": Variable([[[[0.0, 0.0, 1.0]]]], dtype="float32", trainable=False)}

        logger.debug("Initialized: %s ", self.__class__.__name__)

    def call(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
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
            y_true = y_true[..., [2, 1, 0]]
            y_pred = y_pred[..., [2, 1, 0]]

        y_true = ops.clip(y_true, 0, 1.)
        y_pred = ops.clip(y_pred, 0, 1.)

        true_ycxcz = self._col_conv["rgb2ycxcz"](y_true)
        pred_ycxcz = self._col_conv["rgb2ycxcz"](y_pred)

        delta_e_color = self._color_pipeline(true_ycxcz, pred_ycxcz)
        delta_e_features = self._process_features(true_ycxcz, pred_ycxcz)

        loss = ops.power(delta_e_color, 1 - delta_e_features)
        return loss

    def _color_pipeline(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
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

        rgb2lab = self._col_conv["rgb2lab"]
        preprocessed_true = self._hunt_adjustment(rgb2lab(filtered_true))
        preprocessed_pred = self._hunt_adjustment(rgb2lab(filtered_pred))
        hunt_adjusted_green = self._hunt_adjustment(rgb2lab(self._hunt["green"]))
        hunt_adjusted_blue = self._hunt_adjustment(rgb2lab(self._hunt["blue"]))

        delta = self._hyab(preprocessed_true, preprocessed_pred)
        power_delta = ops.power(delta, self._computed_distance_exponent)
        cmax = ops.power(self._hyab(hunt_adjusted_green, hunt_adjusted_blue),
                     self._computed_distance_exponent)
        return self._redistribute_errors(power_delta, cmax)

    def _process_features(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
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

        delta = ops.maximum(ops.abs(frobenius_norm(edges_true) - frobenius_norm(edges_pred)),
                            ops.abs(frobenius_norm(points_pred) - frobenius_norm(points_true)))

        delta = ops.clip(delta, x_min=self._epsilon, x_max=np.inf)
        return ops.power(((1 / np.sqrt(2)) * delta), self._feature_exponent)

    @classmethod
    def _hunt_adjustment(cls, image: torch.Tensor) -> torch.Tensor:
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
        adjusted = ops.concatenate([ch_l, image[..., 1:] * (ch_l * 0.01)], axis=-1)
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
        root = ops.sqrt(ops.clip(ops.power(delta[..., 0:1], 2),
                                 x_min=self._epsilon,
                                 x_max=np.inf))
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
        delta_e_c = ops.where(
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

    def _generate_spatial_filters(self) -> tuple[torch.Tensor, int]:
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
        weights = Variable(np.moveaxis(weights, 0, -1), dtype="float32", trainable=False)

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
    def _generate_weights(cls, channel: dict[str, float], domain: np.ndarray) -> np.ndarray:
        """ TODO docstring """
        a_1, b_1, a_2, b_2 = channel["a1"], channel["b1"], channel["a2"], channel["b2"]
        grad = (a_1 * np.sqrt(np.pi / b_1) * np.exp(-np.pi ** 2 * domain / b_1) +
                a_2 * np.sqrt(np.pi / b_2) * np.exp(-np.pi ** 2 * domain / b_2))
        grad = grad / np.sum(grad)
        grad = np.reshape(grad, (*grad.shape, 1))
        return grad

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
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
        image_tilde_opponent = ops.conv(padded_image,
                                        self._spatial_filters,
                                        strides=1,
                                        padding="valid")
        rgb = ops.clip(self._ycxcz2rgb(image_tilde_opponent), 0., 1.)
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
        grid = np.meshgrid(range(-self._radius, self._radius + 1),
                           range(-self._radius, self._radius + 1))

        gradient = np.exp(-(grid[0] ** 2 + grid[1] ** 2) / (2 * (self._std ** 2)))
        self._grads = {
            "edge": Variable(np.multiply(-grid[0], gradient), trainable=False),
            "point": Variable(np.multiply(grid[0] ** 2 / (self._std ** 2) - 1, gradient),
                                         trainable=False)}

    def __call__(self, image: torch.Tensor, feature_type: str) -> torch.Tensor:
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

        grad_x = self._grads[feature_type]
        negative_weights_sum = -ops.sum(grad_x[grad_x < 0])
        positive_weights_sum = ops.sum(grad_x[grad_x > 0])

        grad_x = ops.where(grad_x < 0,
                           grad_x / negative_weights_sum,
                           grad_x / positive_weights_sum)
        kernel = ops.expand_dims(ops.expand_dims(grad_x, axis=-1), axis=-1)

        features_x = ops.conv(replicate_pad(image, self._radius),
                              kernel,
                              strides=1,
                              padding="valid")
        kernel = ops.transpose(kernel, (1, 0, 2, 3))
        features_y = ops.conv(replicate_pad(image, self._radius),
                              kernel,
                              strides=1,
                              padding="valid")
        features = ops.concatenate([features_x, features_y], axis=-1)
        return features


class MSSIMLoss(keras.losses.Loss):
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
    Adapted from Tehnsorflow's tf.image.ssim_multiscale implementation
    """
    def __init__(self,
                 k_1: float = 0.01,
                 k_2: float = 0.03,
                 filter_size: int = 11,
                 filter_sigma: float = 1.5,
                 max_value: float = 1.0,
                 power_factors: tuple[float, ...] = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333)
                 ) -> None:
        super().__init__(name=self.__class__.__name__)
        self.filter_size = filter_size
        self._filter_sigma = Variable(filter_sigma, dtype="float32", trainable=False)
        self._k_1 = k_1
        self._k_2 = k_2
        self._max_value = max_value
        self._power_factors = power_factors
        self._divisor = [1, 2, 2, 1]
        self._divisor_tensor = Variable(self._divisor[1:], dtype="int32", trainable=False)

    @classmethod
    def _reducer(cls, image: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        """ Computes local averages from a set of images

        Parameters
        ----------
        image: :class:`torch.Tensor`
            The images to be processed
        kernel: :class:`torch.Tensor`
            The kernel to apply

        Returns
        -------
        :class:`torch.Tensor`
            The reduced image
        """
        shape = image.shape
        var_x = ops.reshape(image, (-1, *shape[-3:]))
        var_y = ops.nn.depthwise_conv(var_x, kernel, strides=(1, 1), padding="valid")
        return ops.reshape(var_y, (*shape[:-3], *var_y.shape[1:]))

    def _ssim_helper(self,
                     image1: torch.Tensor,
                     image2: torch.Tensor,
                     kernel: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """ Helper function for computing SSIM

        Parameters
        ----------
        image1: :class:`torch.Tensor`
            The first set of images
        image2: :class:`torch.Tensor`
            The second set of images
        kernel: :class:`torch.Tensor`
            The gaussian kernel

        Returns
        -------
        :class:`torch.Tensor`:
            The channel-wise SSIM
        :class:`torch.Tensor`:
            The channel-wise contrast-structure
        """
        c_1 = (self._k_1 * self._max_value) ** 2
        c_2 = (self._k_2 * self._max_value) ** 2

        mean0 = self._reducer(image1, kernel)
        mean1 = self._reducer(image2, kernel)
        num0 = mean0 * mean1 * 2.0
        den0 = ops.square(mean0) + ops.square(mean1)
        luminance = (num0 + c_1) / (den0 + c_1)

        num1 = self._reducer(image1 * image2, kernel) * 2.0
        den1 = self._reducer(ops.square(image1) + ops.square(image2), kernel)
        cs = (num1 - num0 + c_2) / (den1 - den0 + c_2)

        return luminance, cs

    def _fspecial_gauss(self, size: int) -> torch.Tensor:
        """Function to mimic the 'fspecial' gaussian MATLAB function.

        Parameters
        ----------
        filter_size: int
            size of gaussian filter

        Returns
        -------
        :class:`torch.Tensor`
            The gaussian kernel
        """
        coords = ops.cast(range(size), self._filter_sigma.dtype)
        coords -= ops.cast(size - 1, self._filter_sigma.dtype) / 2.0

        gauss = ops.square(coords)
        gauss *= -0.5 / ops.square(self._filter_sigma)

        gauss = ops.reshape(gauss, [1, -1]) + ops.reshape(gauss, [-1, 1])
        gauss = ops.reshape(gauss, [1, -1])  # For tf.nn.softmax().
        gauss = ops.softmax(gauss)
        return ops.reshape(gauss, [size, size, 1, 1])

    def _ssim_per_channel(self,
                          image1: torch.Tensor,
                          image2: torch.Tensor,
                          filter_size: int) -> tuple[torch.Tensor]:
        """Computes SSIM index between image1 and image2 per color channel.

        This function matches the standard SSIM implementation from:
        Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004). Image
        quality assessment: from error visibility to structural similarity. IEEE
        transactions on image processing.

        Parameters
        ----------
        image1: :class;`torch.Tensor`
            The first image batch
        image2: :class;`torch.Tensor`
            The second image batch.
        filter_size: int
            size of gaussian filter).

        Returns
        -------
        :class:`torch.Tensor`:
            The channel-wise SSIM
        :class:`torch.Tensor`:
            The channel-wise contrast-structure
        """
        shape = image1.shape

        kernel = self._fspecial_gauss(filter_size)
        kernel = ops.tile(kernel, [1, 1, shape[-1], 1])

        luminance, cs = self._ssim_helper(image1, image2, kernel)

        # Average over the second and the third from the last: height, width.
        ssim_val = ops.mean(luminance * cs, [-3, -2])
        cs = ops.mean(cs, [-3, -2])
        return ssim_val, cs

    def _do_pad(cls, images: torch.Tensor, remainder: torch.Tensor) -> list[torch.Tensor]:
        """ Pad images

        Parameters
        ----------
        images: :class:`torch.Tensor`
            Images to pad
        remainder: :class:`torch.Tensor`
            Remainding images to pad

        Returns
        -------
        list[:class:`torch.Tensor`]
            Padded images
        """
        padding = ops.expand_dims(remainder, axis=-1)
        padding = ops.pad(padding, [[1, 0], [1, 0]], mode="constant")
        return [ops.pad(x, padding, mode="symmetric") for x in images]

    def _mssism(self,
                y_true: torch.Tensor,
                y_pred: torch.Tensor,
                filter_size: int) -> torch.Tensor:
        """ Perform the MSSISM calculation.

        Ported from Tensorflow implementation `tf.image.ssim_multiscale`

        Parameters
        ----------
        y_true: :class:`torch.Tensor`
            The ground truth value
        y_pred: :class:`torch.Tensor`
            The predicted value
        filter_size: int
            The filter size to use
        """
        images = [y_true, y_pred]
        shapes = [y_true.shape, y_pred.shape]
        heads = [s[:-3] for s in shapes]
        tails = [s[-3:] for s in shapes]

        mcs = []
        for k in range(len(self._power_factors)):
            if k > 0:
                # Avg pool takes rank 4 tensors. Flatten leading dimensions.
                flat_images = [ops.reshape(x, (-1, *t))
                               for x, t in zip(images, tails)]
                remainder = tails[0] % self._divisor_tensor

                need_padding = ops.any(ops.not_equal(remainder, 0))
                padded = ops.cond(need_padding,
                                  lambda: self._do_pad(flat_images, remainder),
                                  lambda: flat_images)

                downscaled = [ops.average_pool(x,
                                               self._divisor[1:3],
                                               strides=self._divisor[1:3],
                                               padding='valid')
                              for x in padded]

                tails = [x.shape[1:] for x in downscaled]
                images = [ops.reshape(x, (*h, *t))
                          for x, h, t in zip(downscaled, heads, tails)]

            # Overwrite previous ssim value since we only need the last one.
            ssim_per_channel, cs = self._ssim_per_channel(*images, filter_size)
            mcs.append(ops.relu(cs))


        mcs.pop()  # Remove the cs score for the last scale.

        mcs_and_ssim = ops.stack(mcs + [ops.relu(ssim_per_channel)], axis=-1)
        ms_ssim = ops.prod(ops.power(mcs_and_ssim, self._power_factors), [-1])

        return ops.mean(ms_ssim, [-1])  # Avg over color channels.

    def call(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """ Call the MS-SSIM Loss Function.

        Parameters
        ----------
        y_true: :class:`torch.Tensor`
            The ground truth value
        y_pred: :class:`torch.Tensor`
            The predicted value

        Returns
        -------
        :class:`torch.Tensor`
            The MS-SSIM Loss value
        """
        im_size = y_true.shape[1]
        # filter size cannot be larger than the smallest scale
        smallest_scale = self._get_smallest_size(im_size, len(self._power_factors) - 1)
        filter_size = min(self.filter_size, smallest_scale)

        ms_ssim = self._mssism(y_true, y_pred, filter_size)
        ms_ssim_loss = 1. - ms_ssim
        return ops.mean(ms_ssim_loss)

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
