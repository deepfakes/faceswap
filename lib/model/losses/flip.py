#! /usr/env/bin/python3
"""LDR FliP loss from Nvidia"""
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
    spatial_output
        ``True`` to output the loss function as a HxWx1 image output. ``False`` to reduce to mean
        for each item in the batch. Default: ``False``
    """
    _c_max: torch.Tensor

    def __init__(self,
                 computed_distance_exponent: float = 0.7,
                 feature_exponent: float = 0.5,
                 lower_threshold_exponent: float = 0.4,
                 upper_threshold_exponent: float = 0.95,
                 epsilon: float = 1e-15,
                 pixels_per_degree: float | None = None,
                 color_order: T.Literal["bgr", "rgb"] = "bgr",
                 spatial_output: bool = True) -> None:
        logger.debug(parse_class_init(locals()))
        super().__init__()
        self._computed_distance_exponent = computed_distance_exponent
        self._feature_exponent = feature_exponent
        self._pc = lower_threshold_exponent
        self._pt = upper_threshold_exponent
        self._epsilon = epsilon
        self._color_order = color_order.lower()
        self._spatial_output = spatial_output

        if pixels_per_degree is None:
            pixels_per_degree = (0.7 * 3840 / 0.7) * np.pi / 180
        self._pixels_per_degree = pixels_per_degree
        self._spatial_filters = _SpatialFilters(pixels_per_degree)
        self._feature_detector = _FeatureDetection(pixels_per_degree)
        self._rgb2lab = ColorSpaceConvert(from_space="rgb", to_space="lab")
        self._rgb2ycxcz = ColorSpaceConvert("srgb", "ycxcz")

        hunt_adjusted_green = self._hunt_adjustment(
            self._rgb2lab(torch.Tensor([[[[0.0]], [[1.0]], [[0.0]]]]).float())
            )
        hunt_adjusted_blue = self._hunt_adjustment(
            self._rgb2lab(torch.Tensor([[[[0.0]], [[0.0]], [[1.0]]]]).float())
            )
        self.register_buffer("_c_max",
                             self._hyab(hunt_adjusted_green,
                                        hunt_adjusted_blue) ** self._computed_distance_exponent)

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

    def _redistribute_errors(self, power_delta_e_hyab: torch.Tensor) -> torch.Tensor:
        """Redistribute exponentiated HyAB errors to the [0,1] range

        Parameters
        ----------
        power_delta_e_hyab
            The exponentiated HyAb distance

        Returns
        -------
        The redistributed per-pixel HyAB distances (in range [0,1])
        """
        pcc_max = self._pc * self._c_max
        return torch.where(power_delta_e_hyab < pcc_max,
                           (self._pt / pcc_max) * power_delta_e_hyab,
                           self._pt + ((power_delta_e_hyab - pcc_max) /
                                       (self._c_max - pcc_max)) * (1.0 - self._pt))

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
        delta = self._hyab(preprocessed_true, preprocessed_pred)
        power_delta = delta ** self._computed_distance_exponent
        return self._redistribute_errors(power_delta)

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
        if not self._spatial_output:
            loss = loss.mean(dim=(1, 2, 3))
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


__all__ = get_module_objects(__name__)
