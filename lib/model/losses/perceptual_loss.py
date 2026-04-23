#!/usr/bin/env python3
"""Keras implementation of Perceptual Loss Functions for faceswap.py """
from __future__ import annotations

import logging
import typing as T

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from lib.torch_utils import ColorSpaceConvert
from lib.logger import parse_class_init
from lib.utils import get_module_objects


logger = logging.getLogger(__name__)


class DSSIMObjective(nn.Module):
    """DSSIM Loss Functions

    Difference of Structural Similarity (DSSIM loss function).

    Adapted from :func:`tensorflow.image.ssim` for a pure keras implementation.

    Notes
    -----
    Channels last only. Assumes all input images are the same size and square

    Parameters
    ----------
    k_1
        Parameter of the SSIM. Default: `0.01`
    k_2
        Parameter of the SSIM. Default: `0.03`
    filter_size
        size of gaussian filter Default: `11`
    filter_sigma
        Width of gaussian filter Default: `1.5`
    max_value
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
        logger.debug(parse_class_init(locals()))
        super().__init__()
        self._filter_size = filter_size
        self._filter_sigma = filter_sigma
        self._kernel = self._get_kernel()

        compensation = 1.0
        self._c1 = (k_1 * max_value) ** 2
        self._c2 = ((k_2 * max_value) ** 2) * compensation

    def _get_kernel(self) -> torch.Tensor:
        """Obtain the base kernel for performing depthwise convolution.

        Returns
        -------
        The gaussian kernel based on selected size and sigma
        """
        coords = np.arange(self._filter_size, dtype=np.float32)
        coords -= (self._filter_size - 1) / 2.

        kernel = np.square(coords)
        kernel *= -0.5 / np.square(self._filter_sigma)
        kernel = np.reshape(kernel, (1, -1)) + np.reshape(kernel, (-1, 1))
        kernel_t = torch.from_numpy(np.reshape(kernel, (1, -1)))
        kernel_t = torch.softmax(kernel_t, dim=-1)
        kernel_t = torch.reshape(kernel_t, (1, 1, self._filter_size, self._filter_size))
        return kernel_t

    @classmethod
    def _depthwise_conv2d(cls, image: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        """Perform a standardized depthwise convolution.

        Parameters
        ----------
        image
            Batch of images, channels last, to perform depthwise convolution
        kernel
            convolution kernel

        Returns
        -------
        The output from the convolution
        """
        depth, in_ch, h, w = kernel.shape
        kernel = torch.reshape(kernel, (in_ch * depth, 1, h, w))
        return F.conv2d(image, kernel, groups=in_ch)  # pylint:disable=not-callable

    def _get_ssim(self,
                  y_true: torch.Tensor,
                  y_pred: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Obtain the structural similarity between a batch of true and predicted images.

        Parameters
        ----------
        y_true
            The input batch of ground truth images
        y_pred
            The input batch of predicted images

        Returns
        -------
        ssim
            The SSIM for the given images
        contrast
            The Contrast for the given images
        """
        channels = y_true.shape[1]
        kernel = torch.tile(self._kernel, (1, channels, 1, 1))

        # SSIM luminance measure is (2 * mu_x * mu_y + c1) / (mu_x ** 2 + mu_y ** 2 + c1)
        mean_true = self._depthwise_conv2d(y_true, kernel)
        mean_pred = self._depthwise_conv2d(y_pred, kernel)
        num_lum = mean_true * mean_pred * 2.0
        den_lum = torch.square(mean_true) + torch.square(mean_pred)
        luminance = (num_lum + self._c1) / (den_lum + self._c1)

        # SSIM contrast-structure measure is (2 * cov_{xy} + c2) / (cov_{xx} + cov_{yy} + c2)
        num_con = self._depthwise_conv2d(y_true * y_pred, kernel) * 2.0
        den_con = self._depthwise_conv2d(torch.square(y_true) + torch.square(y_pred), kernel)

        contrast = (num_con - num_lum + self._c2) / (den_con - den_lum + self._c2)

        # Average over the height x width dimensions
        axes = (-3, -2)
        ssim = torch.mean(luminance * contrast, dim=axes)
        contrast = torch.mean(contrast, dim=axes)

        return ssim, contrast

    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """Call the DSSIM or MS-DSSIM Loss Function.

        Parameters
        ----------
        y_true
            The input batch of ground truth images
        y_pred
            The input batch of predicted images

        Returns
        -------
            The final DSSIM or MS-DSSIM for each item in the batch
        """
        # TODO remove once channels first
        y_true = y_true.permute(0, 3, 1, 2)
        y_pred = y_pred.permute(0, 3, 1, 2)

        ssim = self._get_ssim(y_true, y_pred)[0]
        retval = (1. - ssim) / 2.0
        return torch.mean(retval, dim=-1)


class GMSDLoss(nn.Module):
    """Gradient Magnitude Similarity Deviation Loss.

    Improved image quality metric over MS-SSIM with easier calculations

    References
    ----------
    http://www4.comp.polyu.edu.hk/~cslzhang/IQA/GMSD/GMSD.htm
    https://arxiv.org/ftp/arxiv/papers/1308/1308.3052.pdf
    """
    _scharr_edges: torch.Tensor

    def __init__(self) -> None:
        logger.debug(parse_class_init(locals()))
        super().__init__()
        self.register_buffer("_scharr_edges", torch.from_numpy(
            np.array([[[[0.00070, 0.00070]],
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
                       [[-0.0007, -0.0007]]]], dtype=np.float32)))

    def _map_scharr_edges(self, image: torch.Tensor, magnitude: bool) -> torch.Tensor:
        """Returns a tensor holding modified Scharr edge maps.

        Parameters
        ----------
        image
            Image tensor with shape [batch_size, h, w, d] and type float32. The image(s) must be
            2x2 or larger.
        magnitude
            Boolean to determine if the edge magnitude or edge direction is returned

        Returns
        -------
        Tensor holding edge maps for each channel. Returns a tensor with shape `[batch_size, h, w,
        d, 2]` where the last two dimensions hold `[[dy[0], dx[0]], [dy[1], dx[1]], ..., [dy[d-1],
        dx[d-1]]]` calculated using the Scharr filter.
        """
        # Define vertical and horizontal Scharr filters.
        bs, channels, height, width = image.shape

        kernel = self._scharr_edges.repeat(1, 1, channels, 1)
        h, w, _, depth = kernel.shape
        kernel = kernel.permute(3, 2, 0, 1).reshape(channels * depth, 1, h, w)

        # Use depth-wise convolution to calculate edge maps per channel.
        # Output tensor has shape [batch_size, h, w, d * num_kernels].
        padded = F.pad(image, (2, 2, 2, 2), mode="reflect")
        out = F.conv2d(padded, kernel, groups=channels)  # pylint:disable=not-callable

        if not magnitude:  # direction of edges
            # Reshape to [batch_size, h, w, d, num_kernels].
            out = out.reshape(bs, height, width, channels, 2)
            gx = out[..., 0]
            gy = out[..., 1]
            out = torch.atan(gx / gy)
        # magnitude of edges -- unified x & y edges don't work well with Neural Networks
        return out

    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """Return the Gradient Magnitude Similarity Deviation Loss.

        Parameters
        ----------
        y_true
            The ground truth value
        y_pred
            The predicted value

        Returns
        -------
        The final loss value for each item in the batch
        """
        # TODO remove once channels first
        y_true = y_true.permute(0, 3, 1, 2)
        y_pred = y_pred.permute(0, 3, 1, 2)

        true_edge = self._map_scharr_edges(y_true, True)
        pred_edge = self._map_scharr_edges(y_pred, True)
        epsilon = 0.0025
        upper = 2.0 * true_edge * pred_edge
        lower = torch.square(true_edge) + torch.square(pred_edge)
        gms = (upper + epsilon) / (lower + epsilon)
        gmsd = torch.std(gms, dim=(1, 2, 3))
        return gmsd


class LDRFLIPLoss(nn.Module):  # pylint:disable=too-many-instance-attributes
    """Computes the LDR-FLIP error map between two LDR images, assuming the images are observed
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
    computed_distance_exponent
        The computed distance exponent to apply to Hunt adjusted, filtered colors.
        (`qc` in original paper). Default: `0.7`
    feature_exponent
        The feature exponent to apply for increasing the impact of feature difference on the
        final loss value. (`qf` in original paper). Default: `0.5`
    lower_threshold_exponent
        The `pc` exponent for the color pipeline as described in the original paper: Default: `0.4`
    upper_threshold_exponent
        The `pt` exponent  for the color pipeline as described in the original paper.
        Default: `0.95`
    epsilon
        A small value to improve training stability. Default: `1e-15`
    pixels_per_degree
        The estimated number of pixels per degree of visual angle of the observer. This effectively
        impacts the tolerance when calculating loss. The default corresponds to viewing images on a
        0.7m wide 4K monitor at 0.7m from the display. Default: ``None``
    color_order
        The `"bgr"` or `"rgb"` color order of the incoming images
    """
    def __init__(self,
                 computed_distance_exponent: float = 0.7,
                 feature_exponent: float = 0.5,
                 lower_threshold_exponent: float = 0.4,
                 upper_threshold_exponent: float = 0.95,
                 epsilon: float = 1e-15,
                 pixels_per_degree: float | None = None,
                 color_order: T.Literal["bgr", "rgb"] = "bgr") -> None:
        logger.debug(parse_class_init(locals()))
        super().__init__()
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
        self._rgb2lab = ColorSpaceConvert(from_space="rgb", to_space="lab")
        self._rgb2ycxcz = ColorSpaceConvert("srgb", "ycxcz")

    @classmethod
    def _hunt_adjustment(cls, image: torch.Tensor) -> torch.Tensor:
        """Apply Hunt-adjustment to an image in L*a*b* color space

        Parameters
        ----------
        image
            The batch of images in L*a*b* to adjust

        Returns
        -------
        The hunt adjusted batch of images in L*a*b color space
        """
        ch_l = image[:, 0:1]
        return torch.cat([ch_l, image[:, 1:] * (ch_l * 0.01)], dim=1)

    def _hyab(self, y_true: torch.Tensor, y_pred: torch.Tensor | float) -> torch.Tensor:
        """Compute the HyAB distance between true and predicted images.

        Parameters
        ----------
        y_true
            The ground truth batch of images in standard or Hunt-adjusted L*A*B* color space
        y_pred
            The predicted batch of images in in standard or Hunt-adjusted L*A*B* color space

        Returns
        -------
        image tensor containing the per-pixel HyAB distances between true and predicted images
        """
        delta = y_true - y_pred
        root = torch.sqrt(torch.clamp(torch.pow(delta[:, 0:1], 2), min=self._epsilon))
        delta_norm = torch.norm(delta[:, 1:3], dim=1, keepdim=True)
        return root + delta_norm

    def _redistribute_errors(self,
                             power_delta_e_hyab: torch.Tensor,
                             c_max: torch.Tensor) -> torch.Tensor:
        """Redistribute exponentiated HyAB errors to the [0,1] range

        Parameters
        ----------
        power_delta_e_hyab
            The exponentiated HyAb distance
        c_max
            The exponentiated, maximum HyAB difference between two colors in Hunt-adjusted
            L*A*B* space

        Returns
        -------
        The redistributed per-pixel HyAB distances (in range [0,1])
        """
        pcc_max = self._pc * c_max
        return torch.where(power_delta_e_hyab < pcc_max,
                           (self._pt / pcc_max) * power_delta_e_hyab,
                           self._pt + ((power_delta_e_hyab - pcc_max) /
                                       (c_max - pcc_max)) * (1.0 - self._pt))

    def _color_pipeline(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """Perform the color processing part of the FLIP loss function

        Parameters
        ----------
        y_true
            The ground truth batch of images in YCxCz color space
        y_pred
            The predicted batch of images in YCxCz color space

        Returns
        -------
        The exponentiated, maximum HyAB difference between two colors in Hunt-adjusted L*A*B* space
        """
        filtered_true = self._spatial_filters(y_true)
        filtered_pred = self._spatial_filters(y_pred)

        preprocessed_true = self._hunt_adjustment(self._rgb2lab(filtered_true))
        preprocessed_pred = self._hunt_adjustment(self._rgb2lab(filtered_pred))
        hunt_adjusted_green = self._hunt_adjustment(
            self._rgb2lab(torch.Tensor([[[[0.0]], [[1.0]], [[0.0]]]]).float().to(y_pred.device))
            )
        hunt_adjusted_blue = self._hunt_adjustment(
            self._rgb2lab(torch.Tensor([[[[0.0]], [[0.0]], [[1.0]]]]).float().to(y_pred.device))
            )

        delta = self._hyab(preprocessed_true, preprocessed_pred)
        power_delta = delta ** self._computed_distance_exponent
        c_max = self._hyab(hunt_adjusted_green,
                           hunt_adjusted_blue) ** self._computed_distance_exponent
        return self._redistribute_errors(power_delta, c_max)

    def _process_features(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """Perform the color processing part of the FLIP loss function

        Parameters
        ----------
        y_true
            The ground truth batch of images in YCxCz color space
        y_pred
            The predicted batch of images in YCxCz color space

        Returns
        -------
        The exponentiated features delta
        """
        col_y_true = (y_true[:, 0:1] + 16) / 116.
        col_y_pred = (y_pred[:, 0:1] + 16) / 116.

        edges_true = self._feature_detector(col_y_true, "edge")
        points_true = self._feature_detector(col_y_true, "point")
        edges_pred = self._feature_detector(col_y_pred, "edge")
        points_pred = self._feature_detector(col_y_pred, "point")

        delta = torch.maximum(torch.abs(torch.norm(edges_true, dim=1, keepdim=True) -
                                        torch.norm(edges_pred, dim=1, keepdim=True)),
                              torch.abs(torch.norm(points_pred, dim=1, keepdim=True) -
                                        torch.norm(points_true, dim=1, keepdim=True)))

        delta = torch.clamp(delta, min=self._epsilon)
        return ((1 / np.sqrt(2)) * delta) ** self._feature_exponent

    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """Call the LDR Flip Loss Function

        Parameters
        ----------
        y_true
            The ground truth batch of images
        y_pred
            The predicted batch of images

        Returns
        -------
        The calculated Flip loss value
        """
        # TODO remove once channels first
        y_true = y_true.permute(0, 3, 1, 2)
        y_pred = y_pred.permute(0, 3, 1, 2)

        if self._color_order == "bgr":  # Switch models training in bgr order to rgb
            y_true = torch.flip(y_true, dims=[1])
            y_pred = torch.flip(y_pred, dims=[1])

        y_true = torch.clamp(y_true, 0, 1.)
        y_pred = torch.clamp(y_pred, 0, 1.)
        true_ycxcz = self._rgb2ycxcz(y_true)
        pred_ycxcz = self._rgb2ycxcz(y_pred)

        delta_e_color = self._color_pipeline(true_ycxcz, pred_ycxcz)
        delta_e_features = self._process_features(true_ycxcz, pred_ycxcz)
        loss = delta_e_color ** (1 - delta_e_features)
        return loss


class _SpatialFilters(nn.Module):
    """Filters an image with channel specific spatial contrast sensitivity functions and clips
    result to the unit cube in linear RGB.

    For use with LDRFlipLoss.

    Parameters
    ----------
    pixels_per_degree
        The estimated number of pixels per degree of visual angle of the observer. This effectively
        impacts the tolerance when calculating loss.
    """
    _spatial_filters: torch.Tensor

    def __init__(self, pixels_per_degree: float) -> None:
        logger.debug(parse_class_init(locals()))
        super().__init__()
        self._pixels_per_degree = pixels_per_degree
        self._radius: int = 0  # Set when spatial filters are generated
        self.register_buffer("_spatial_filters", self._generate_spatial_filters())
        self._ycxcz2rgb = ColorSpaceConvert(from_space="ycxcz", to_space="rgb")

    def _get_evaluation_domain(self,
                               b1_a: float,
                               b2_a: float,
                               b1_rg: float,
                               b2_rg: float,
                               b1_by: float,
                               b2_by: float) -> tuple[np.ndarray, int]:
        """Get the evaluation domain for the spatial filters"""
        max_scale_parameter = max([b1_a, b2_a, b1_rg, b2_rg, b1_by, b2_by])
        delta_x = 1.0 / self._pixels_per_degree
        radius = int(np.ceil(3 * np.sqrt(max_scale_parameter / (2 * np.pi**2))
                             * self._pixels_per_degree))
        ax_x, ax_y = np.meshgrid(range(-radius, radius + 1), range(-radius, radius + 1))
        domain = (ax_x * delta_x) ** 2 + (ax_y * delta_x) ** 2
        return domain, radius

    @classmethod
    def _generate_weights(cls, channel: dict[str, float], domain: np.ndarray) -> np.ndarray:
        """Generate the weights for the spacial filters"""
        a_1, b_1, a_2, b_2 = channel["a1"], channel["b1"], channel["a2"], channel["b2"]
        grad = (a_1 * np.sqrt(np.pi / b_1) * np.exp(-np.pi ** 2 * domain / b_1) +
                a_2 * np.sqrt(np.pi / b_2) * np.exp(-np.pi ** 2 * domain / b_2))
        grad = grad / np.sum(grad)
        grad = np.reshape(grad, (1, *grad.shape))
        return grad

    def _generate_spatial_filters(self) -> torch.Tensor:
        """Generates spatial contrast sensitivity filters with width depending on the number of
        pixels per degree of visual angle of the observer for channels "A", "RG" and "BY"

        Returns
        -------
        The spatial filter kernel for the channels ("A" (Achromatic CSF), "RG" (Red-Green CSF) or
        "BY" (Blue-Yellow CSF)) corresponding to the spatial contrast sensitivity function
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
        self._radius = radius
        weights = np.array([self._generate_weights(mapping[channel], domain)
                            for channel in ("A", "RG", "BY")])
        return torch.from_numpy(weights).float()

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Call the spacial filtering.

        Parameters
        ----------
        image
            Image tensor to filter in YCxCz color space

        Returns
        -------
        The input image transformed to linear RGB after filtering with spatial contrast sensitivity
        functions
        """
        img_pad = F.pad(image, (self._radius, self._radius, self._radius, self._radius),
                        mode="replicate")
        image_tilde_opponent = F.conv2d(img_pad,  # pylint:disable=not-callable
                                        self._spatial_filters,
                                        groups=3)
        return torch.clamp(self._ycxcz2rgb(image_tilde_opponent), 0., 1.)


class _FeatureDetection(nn.Module):
    """Detect features (i.e. edges and points) in an achromatic YCxCz image.

    For use with LDRFlipLoss.

    Parameters
    ----------
    pixels_per_degree
        The number of pixels per degree of visual angle of the observer
    """
    _grads_edge: torch.Tensor
    _grads_point: torch.Tensor

    def __init__(self, pixels_per_degree: float) -> None:
        logger.debug(parse_class_init(locals()))
        super().__init__()
        width = 0.082
        self._std = 0.5 * width * pixels_per_degree
        self._radius = int(np.ceil(3 * self._std))

        grid = np.meshgrid(range(-self._radius, self._radius + 1),
                           range(-self._radius, self._radius + 1))
        gradient = np.exp(-(grid[0] ** 2 + grid[1] ** 2) / (2 * (self._std ** 2)))
        self.register_buffer("_grads_edge",
                             torch.from_numpy(np.multiply(-grid[0], gradient)).float())
        self.register_buffer("_grads_point",
                             torch.from_numpy(np.multiply(grid[0] ** 2 / (self._std ** 2) - 1,
                                                          gradient)).float())

    def forward(self, image: torch.Tensor, feature_type: str) -> torch.Tensor:
        """Run the feature detection

        Parameters
        ----------
        image
            Batch of images in YCxCz color space with normalized Y values
        feature_type
            Type of features to detect (`"edge"` or `"point"`)

        Returns
        -------
        Detected features in the 0-1 range
        """
        feature_type = feature_type.lower()
        grad_x = self._grads_edge if feature_type == "edge" else self._grads_point
        negative_weights_sum = -grad_x[grad_x < 0].sum()
        positive_weights_sum = grad_x[grad_x > 0].sum()

        grad_x = torch.where(grad_x < 0,
                             grad_x / negative_weights_sum,
                             grad_x / positive_weights_sum)
        kernel = grad_x[None, None]
        pad = (self._radius, self._radius, self._radius, self._radius,)

        features_x = F.conv2d(F.pad(image, pad, mode="replicate"),  # pylint:disable=not-callable
                              kernel)
        features_y = F.conv2d(F.pad(image, pad, mode="replicate"),  # pylint:disable=not-callable
                              kernel.swapaxes(2, 3))
        return torch.cat([features_x, features_y], dim=1)


class MSSIMLoss(nn.Module):
    """Multi-scale Structural Similarity Loss Function

    Parameters
    ----------
    k_1
        Parameter of the SSIM. Default: `0.01`
    k_2
        Parameter of the SSIM. Default: `0.03`
    filter_size
        size of gaussian filter Default: `11`
    filter_sigma
        Width of gaussian filter Default: `1.5`
    max_value
        Max value of the output. Default: `1.0`
    power_factors
        Iterable of weights for each of the scales. The number of scales used is the length of the
        list. Index 0 is the unscaled resolution's weight and each increasing scale corresponds to
        the image being downsampled by 2. Defaults to the values obtained in the original paper.
        Default: (0.0448, 0.2856, 0.3001, 0.2363, 0.1333)

    Notes
    ------
    You should add a regularization term like a l2 loss in addition to this one.
    Adapted from Tensorflow's ssim_multi-scale implementation
    """
    _power_factors: torch.Tensor
    _divisor_tensor: torch.Tensor

    def __init__(self,
                 k_1: float = 0.01,
                 k_2: float = 0.03,
                 filter_size: int = 11,
                 filter_sigma: float = 1.5,
                 max_value: float = 1.0,
                 power_factors: tuple[float, ...] = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333)
                 ) -> None:
        logger.debug(parse_class_init(locals()))
        super().__init__()
        self.filter_size = filter_size
        self._filter_sigma_sq = filter_sigma ** 2
        self._k_1 = k_1
        self._k_2 = k_2
        self._max_value = max_value
        self._divisor = [1, 1, 2, 2]
        self.register_buffer("_power_factors", torch.Tensor(power_factors).float())
        self.register_buffer("_divisor_tensor", torch.Tensor(self._divisor[1:]).int())

    @classmethod
    def _reducer(cls, image: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        """Computes local averages from a set of images

        Parameters
        ----------
        image
            The images to be processed (N,C,H,W)
        kernel
            The kernel to apply in depthwise format (C,1,H,W)

        Returns
        -------
        The reduced image
        """
        shape = image.shape
        channels = shape[-3]
        x = image.reshape(-1, *shape[-3:])
        y = F.conv2d(x, kernel, groups=channels)  # pylint:disable=not-callable
        return y.reshape((*shape[:-3], *y.shape[1:]))

    def _ssim_helper(self,
                     image1: torch.Tensor,
                     image2: torch.Tensor,
                     kernel: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Helper function for computing SSIM

        Parameters
        ----------
        image1
            The first set of images (N,C,H,W)
        image2
            The second set of images (N,C,H,W)
        kernel
            The gaussian kernel in depthwise format (C,1,H,W)

        Returns
        -------
        ssim
            The channel-wise SSIM
        contrast
            The channel-wise contrast-structure
        """
        c_1 = (self._k_1 * self._max_value) ** 2
        c_2 = (self._k_2 * self._max_value) ** 2

        mean0 = self._reducer(image1, kernel)
        mean1 = self._reducer(image2, kernel)
        num0 = mean0 * mean1 * 2.0
        den0 = mean0 ** 2 + mean1 ** 2
        luminance = (num0 + c_1) / (den0 + c_1)

        num1 = self._reducer(image1 * image2, kernel) * 2.0
        den1 = self._reducer(image1 ** 2 + image2 ** 2, kernel)
        cs_ = (num1 - num0 + c_2) / (den1 - den0 + c_2)

        return luminance, cs_

    def _fspecial_gauss(self, size: int) -> torch.Tensor:
        """Function to mimic the 'fspecial' gaussian MATLAB function.

        Parameters
        ----------
        filter_size
            size of gaussian filter

        Returns
        -------
        The gaussian kernel in channels first depthwise format (C,1,H,W)
        """
        coords = torch.arange(0, size, dtype=torch.float32, device=self._divisor_tensor.device)
        coords -= size - 1 / 2.

        gauss = coords ** 2 * (-0.5 / self._filter_sigma_sq)

        gauss = gauss.reshape(1, -1) + gauss.reshape(-1, 1)
        gauss = gauss.reshape(1, -1)  # For ops.softmax().
        gauss = F.softmax(gauss, dim=-1)
        return gauss.reshape(1, 1, size, size)

    def _ssim_per_channel(self,
                          image1: torch.Tensor,
                          image2: torch.Tensor,
                          filter_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Computes SSIM index between image1 and image2 per color channel.

        This function matches the standard SSIM implementation from:
        Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004). Image
        quality assessment: from error visibility to structural similarity. IEEE
        transactions on image processing.

        Parameters
        ----------
        image1
            The first image batch (N,C,H,W)
        image2
            The second image batch. (N,C,H,W)
        filter_size
            size of gaussian filter.

        Returns
        -------
        ssim
            The channel-wise SSIM
        contrast
            The channel-wise contrast-structure
        """
        channels = image1.shape[-3]
        kernel = self._fspecial_gauss(filter_size)
        kernel = kernel.repeat(channels, 1, 1, 1)
        luminance, cs_ = self._ssim_helper(image1, image2, kernel)

        # Average over height, width.
        ssim_val = (luminance * cs_).mean(dim=[-2, -1])
        cs_ = cs_.mean(dim=[-2, -1])
        return ssim_val, cs_

    @classmethod
    def _do_pad(cls, images: list[torch.Tensor], remainder: torch.Tensor) -> list[torch.Tensor]:
        """Pad images

        Parameters
        ----------
        images
            Images to pad (N,C,H,W)
        remainder
            Remaining images to pad (C,H,W)

        Returns
        -------
        Padded images (N,C,H,W)
        """
        height = int(remainder[1])
        width = int(remainder[2])
        return [F.pad(x, (0, width, 0, height), mode="replicate") for x in images]

    def _mssism(self,  # pylint:disable=too-many-locals
                y_true: torch.Tensor,
                y_pred: torch.Tensor,
                filter_size: int) -> torch.Tensor:
        """Perform the MSSISM calculation.

        Ported from Tensorflow implementation `image.ssim_multiscale`

        Parameters
        ----------
        y_true
            The ground truth value
        y_pred
            The predicted value
        filter_size
            The filter size to use
        """
        images = [y_true, y_pred]
        shapes = [y_true.shape, y_pred.shape]
        heads = [s[:-3] for s in shapes]
        tails = [s[-3:] for s in shapes]

        mcs = []
        ssim_per_channel = None
        for k in range(len(self._power_factors)):
            if k > 0:
                # Avg pool takes rank 4 tensors. Flatten leading dimensions.
                flat_images = [(x.reshape(-1, *t)) for x, t in zip(images, tails)]
                remainder = torch.tensor(list(tails[0]),
                                         dtype=torch.int32,
                                         device=y_pred.device) % self._divisor_tensor
                if (remainder != 0).any():
                    flat_images = self._do_pad(flat_images, remainder)

                downscaled = [F.avg_pool2d(x,  # pylint:disable=not-callable
                                           self._divisor[1:3],
                                           stride=self._divisor[1:3],
                                           padding=0)
                              for x in flat_images]

                tails = [x.shape[1:] for x in downscaled]
                images = [x.reshape(*h, *t) for x, h, t in zip(downscaled, heads, tails)]

            # Overwrite previous ssim value since we only need the last one.
            ssim_per_channel, cs_ = self._ssim_per_channel(images[0], images[1], filter_size)
            mcs.append(F.relu(cs_))

        mcs.pop()  # Remove the cs score for the last scale.
        assert ssim_per_channel is not None
        mcs_and_ssim = torch.stack(mcs + [F.relu(ssim_per_channel)], dim=-1)
        ms_ssim = torch.prod(mcs_and_ssim ** self._power_factors, dim=-1)
        return ms_ssim.mean(dim=-1)  # Avg over color channels.

    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """Call the MS-SSIM Loss Function.

        Parameters
        ----------
        y_true
            The ground truth value
        y_pred
            The predicted value

        Returns
        -------
        The MS-SSIM Loss value
        """
        # TODO remove once channels first
        y_true = y_true.permute(0, 3, 1, 2)
        y_pred = y_pred.permute(0, 3, 1, 2)

        im_size = y_true.shape[2]
        # filter size cannot be larger than the smallest scale
        smallest_scale = self._get_smallest_size(im_size, len(self._power_factors) - 1)
        filter_size = min(self.filter_size, smallest_scale)

        ms_ssim = self._mssism(y_true, y_pred, filter_size)
        ms_ssim_loss = 1. - ms_ssim
        return ms_ssim_loss

    def _get_smallest_size(self, size: int, idx: int) -> int:
        """Recursive function to obtain the smallest size that the image will be scaled to.

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
        logger.trace("scale id: %s, size: %s", idx, size)  # type:ignore[attr-defined]
        if idx > 0:
            size = self._get_smallest_size(size // 2, idx - 1)
        return size


__all__ = get_module_objects(__name__)
