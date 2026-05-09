#!/usr/bin/env python3
"""Custom Loss Functions for faceswap.py"""

from __future__ import annotations
import logging

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from lib.logger import parse_class_init
from lib.utils import get_module_objects

logger = logging.getLogger(__name__)


class FocalFrequencyLoss(nn.Module):
    """Focal frequency Loss Function.

    Parameters
    ----------
    alpha
        Scaling factor of the spectrum weight matrix for flexibility. Default: ``1.0``
    patch_factor
        Factor to crop image patches for patch-based focal frequency loss.
        Default: ``1``
    ave_spectrum
        ``True`` to use mini-batch average spectrum otherwise ``False``. Default: ``False``
    log_matrix
        ``True`` to adjust the spectrum weight matrix by logarithm otherwise ``False``.
        Default: ``False``
    batch_matrix
        ``True`` to calculate the spectrum weight matrix using batch-based statistics otherwise
        ``False``. Default: ``False``
    epsilon
        Small epsilon for safer weights scaling division. Default: `1e-6`
    spatial_output
        ``True`` to output the loss values spatially. ``False`` as scalar per item.
        Default: ``True``

    References
    ----------
    https://arxiv.org/pdf/2012.12821.pdf
    https://github.com/EndlessSora/focal-frequency-loss
    """
    _epsilon: torch.Tensor

    def __init__(self,
                 alpha: float = 1.0,
                 patch_factor: int = 1,
                 ave_spectrum: bool = False,
                 log_matrix: bool = False,
                 batch_matrix: bool = False,
                 epsilon: float = 1e-6,
                 spatial_output: bool = True) -> None:
        logger.debug(parse_class_init(locals()))
        super().__init__()
        self._alpha = alpha
        self._patch_factor = patch_factor
        self._ave_spectrum = ave_spectrum
        self._log_matrix = log_matrix
        self._batch_matrix = batch_matrix
        self.register_buffer("_epsilon", torch.Tensor([epsilon]).float())
        self._spatial = spatial_output

    def _get_patches(self, inputs: torch.Tensor) -> torch.Tensor:
        """Crop the incoming batch of images into patches as defined by :attr:`_patch_factor.

        Parameters
        ----------
        inputs
            A batch of images to be converted into patches

        Returns
        -------
        The incoming batch converted into patches
        """
        patch_list = []
        rows, cols = inputs.shape[2:4]
        assert cols % self._patch_factor == 0 and rows % self._patch_factor == 0, (
            "Patch factor must be a divisor of the image height and width")
        patch_rows = rows // self._patch_factor
        patch_cols = cols // self._patch_factor
        for i in range(self._patch_factor):
            for j in range(self._patch_factor):
                row_from = i * patch_rows
                row_to = (i + 1) * patch_rows
                col_from = j * patch_cols
                col_to = (j + 1) * patch_cols
                patch_list.append(inputs[:, :, row_from: row_to, col_from:col_to])

        retval = torch.stack(patch_list, dim=1)
        return retval

    def _tensor_to_frequency_spectrum(self, patch: torch.Tensor) -> torch.Tensor:
        """Perform FFT to create the orthonomalized DFT frequencies.

        Parameters
        ----------
        inputs
            The incoming batch of patches to convert to the frequency spectrum

        Returns
        -------
        The DFT frequencies split into real and imaginary numbers as float32
        """
        freq = torch.fft.fft2(patch, norm="ortho")  # pylint:disable=not-callable
        freq = torch.stack([freq.real, freq.imag], dim=-1)
        return freq

    def _get_weight_matrix(self, freq_true: torch.Tensor, freq_pred: torch.Tensor) -> torch.Tensor:
        """Calculate a continuous, dynamic weight matrix based on current Euclidean distance.

        Parameters
        ----------
        freq_true
            The real and imaginary DFT frequencies for the true batch of images
        freq_pred
            The real and imaginary DFT frequencies for the predicted batch of images

        Returns
        -------
        The weights matrix for prioritizing hard frequencies
        """
        weights = torch.square(freq_pred - freq_true)
        weights = torch.sqrt(weights[..., 0] + weights[..., 1])
        weights = torch.pow(weights, self._alpha)

        if self._log_matrix:  # adjust the spectrum weight matrix by logarithm
            weights = torch.log(weights + 1.0)

        if self._batch_matrix:  # calculate the spectrum weight matrix using batch-based statistics
            scale = torch.max(weights)
        else:
            scale = torch.amax(weights, dim=(-1, -2), keepdim=True)
        weights = weights / torch.maximum(scale, self._epsilon)
        return torch.clamp(weights, min=0.0, max=1.0)

    def _calculate_loss(self,
                        freq_true: torch.Tensor,
                        freq_pred: torch.Tensor,
                        weight_matrix: torch.Tensor) -> torch.Tensor:
        """Perform the loss calculation on the DFT spectrum applying the weights matrix.

        Parameters
        ----------
        freq_true
            The real and imaginary DFT frequencies for the true batch of images
        freq_pred
            The real and imaginary DFT frequencies for the predicted batch of images

        Returns
        -------
        The final loss value for each item in the batch
        """

        tmp = torch.square(freq_pred - freq_true)  # freq distance using squared Euclidean distance

        freq_distance = tmp[..., 0] + tmp[..., 1]
        loss = weight_matrix * freq_distance  # dynamic spectrum weighting (Hadamard product)
        return torch.mean(loss, dim=(1, ) if self._spatial else (1, 2, 3, 4))

    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """Call the Focal Frequency Loss Function.

        Parameters
        ----------
        y_true
            The ground truth batch of images
        y_pred
            The predicted batch of images

        Returns
        -------
        The final loss value for each item in the batch
        """
        patches_true = self._get_patches(y_true)
        patches_pred = self._get_patches(y_pred)

        freq_true = self._tensor_to_frequency_spectrum(patches_true)
        freq_pred = self._tensor_to_frequency_spectrum(patches_pred)

        if self._ave_spectrum:  # whether to use mini-batch average spectrum
            freq_true = torch.mean(freq_true, dim=0, keepdim=True)
            freq_pred = torch.mean(freq_pred, dim=0, keepdim=True)

        weight_matrix = self._get_weight_matrix(freq_true, freq_pred)
        return self._calculate_loss(freq_true, freq_pred, weight_matrix)


class GeneralizedLoss(nn.Module):
    """Generalized function used to return a large variety of mathematical loss functions.

    The primary benefit is a smooth, differentiable version of L1 loss.

    References
    ----------
    Barron, J. A General and Adaptive Robust Loss Function - https://arxiv.org/pdf/1701.03077.pdf

    Example
    -------
    >>> a=1.0, x>>c , c=1.0/255.0  # will give a smoothly differentiable version of L1 / MAE loss
    >>> a=1.999999 (limit as a->2), beta=1.0/255.0 # will give L2 / RMSE loss

    Parameters
    ----------
    alpha
        Penalty factor. Larger number give larger weight to large deviations. Default: `1.0`
    beta
        Scale factor used to adjust to the input scale (i.e. inputs of mean `1e-4` or `256`).
        Default: `1.0/255.0`
    spatial_output
        ``True`` to output the loss values spatially. ``False`` as scalar per item.
        Default: ``True``
    """
    def __init__(self,
                 alpha: float = 1.0,
                 beta: float = 1.0 / 255.0,
                 spatial_output: bool = True) -> None:
        logger.debug(parse_class_init(locals()))
        super().__init__()
        self._alpha = alpha
        self._beta = beta
        self._spatial = spatial_output

    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """Call the Generalized Loss Function

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
        diff = y_pred - y_true
        second = (torch.pow(torch.pow(diff/self._beta, 2.) / abs(2. - self._alpha) + 1.,
                            (self._alpha / 2.)) - 1.)
        loss = (abs(2. - self._alpha)/self._alpha) * second
        if not self._spatial:
            loss = torch.mean(loss, dim=(1, 2, 3))
        return loss * self._beta


class GradientLoss(nn.Module):
    """Gradient Loss Function.

    Calculates the first and second order gradient difference between pixels of an image in the x
    and y dimensions. These gradients are then compared between the ground truth and the predicted
    image and the difference is taken. When used as a loss, its minimization will result in
    predicted images approaching the same level of sharpness / blurriness as the ground truth.

    Parameters
    ----------
    spatial_output
        ``True`` to output the loss values spatially. ``False`` as scalar per item.
        Default: ``True``

    References
    ----------
    TV+TV2 Regularization with Non-Convex Sparseness-Inducing Penalty for Image Restoration,
    Chengwu Lu & Hua Huang, 2014 - http://downloads.hindawi.com/journals/mpe/2014/790547.pdf
    """
    def __init__(self,
                 spatial_output: bool = True) -> None:
        logger.debug(parse_class_init(locals()))
        super().__init__()
        self.generalized_loss = GeneralizedLoss(alpha=1.9999)
        self._tv_weight = 1.0
        self._tv2_weight = 1.0
        self._spatial = spatial_output

    @classmethod
    def _diff_x(cls, img: torch.Tensor) -> torch.Tensor:
        """X Difference"""
        x_left = img[:, :, 1:2, :] - img[:, :, 0:1, :]
        x_inner = img[:, :, 2:, :] - img[:, :, :-2, :]
        x_right = img[:, :, -1:, :] - img[:, :, -2:-1, :]
        x_out = torch.concatenate([x_left, x_inner, x_right], dim=2)
        return x_out * 0.5

    @classmethod
    def _diff_y(cls, img: torch.Tensor) -> torch.Tensor:
        """Y Difference"""
        y_top = img[:, 1:2, :, :] - img[:, 0:1, :, :]
        y_inner = img[:, 2:, :, :] - img[:, :-2, :, :]
        y_bot = img[:, -1:, :, :] - img[:, -2:-1, :, :]
        y_out = torch.concatenate([y_top, y_inner, y_bot], dim=1)
        return y_out * 0.5

    @classmethod
    def _diff_xx(cls, img: torch.Tensor) -> torch.Tensor:
        """X-X Difference"""
        x_left = img[:, :, 1:2, :] + img[:, :, 0:1, :]
        x_inner = img[:, :, 2:, :] + img[:, :, :-2, :]
        x_right = img[:, :, -1:, :] + img[:, :, -2:-1, :]
        x_out = torch.concatenate([x_left, x_inner, x_right], dim=2)
        return x_out - 2.0 * img

    @classmethod
    def _diff_yy(cls, img: torch.Tensor) -> torch.Tensor:
        """Y-Y Difference"""
        y_top = img[:, 1:2, :, :] + img[:, 0:1, :, :]
        y_inner = img[:, 2:, :, :] + img[:, :-2, :, :]
        y_bot = img[:, -1:, :, :] + img[:, -2:-1, :, :]
        y_out = torch.concatenate([y_top, y_inner, y_bot], dim=1)
        return y_out - 2.0 * img

    @classmethod
    def _diff_xy(cls, img: torch.Tensor) -> torch.Tensor:
        """X-Y Difference"""
        # x_out1
        # Left
        top = img[:, 1:2, 1:2, :] + img[:, 0:1, 0:1, :]
        inner = img[:, 2:, 1:2, :] + img[:, :-2, 0:1, :]
        bottom = img[:, -1:, 1:2, :] + img[:, -2:-1, 0:1, :]
        xy_left = torch.concatenate([top, inner, bottom], dim=1)
        # Mid
        top = img[:, 1:2, 2:, :] + img[:, 0:1, :-2, :]
        mid = img[:, 2:, 2:, :] + img[:, :-2, :-2, :]
        bottom = img[:, -1:, 2:, :] + img[:, -2:-1, :-2, :]
        xy_mid = torch.concatenate([top, mid, bottom], dim=1)
        # Right
        top = img[:, 1:2, -1:, :] + img[:, 0:1, -2:-1, :]
        inner = img[:, 2:, -1:, :] + img[:, :-2, -2:-1, :]
        bottom = img[:, -1:, -1:, :] + img[:, -2:-1, -2:-1, :]
        xy_right = torch.concatenate([top, inner, bottom], dim=1)

        # X_out2
        # Left
        top = img[:, 0:1, 1:2, :] + img[:, 1:2, 0:1, :]
        inner = img[:, :-2, 1:2, :] + img[:, 2:, 0:1, :]
        bottom = img[:, -2:-1, 1:2, :] + img[:, -1:, 0:1, :]
        xy1_left = torch.concatenate([top, inner, bottom], dim=1)
        # Mid
        top = img[:, 0:1, 2:, :] + img[:, 1:2, :-2, :]
        mid = img[:, :-2, 2:, :] + img[:, 2:, :-2, :]
        bottom = img[:, -2:-1, 2:, :] + img[:, -1:, :-2, :]
        xy1_mid = torch.concatenate([top, mid, bottom], dim=1)
        # Right
        top = img[:, 0:1, -1:, :] + img[:, 1:2, -2:-1, :]
        inner = img[:, :-2, -1:, :] + img[:, 2:, -2:-1, :]
        bottom = img[:, -2:-1, -1:, :] + img[:, -1:, -2:-1, :]
        xy1_right = torch.concatenate([top, inner, bottom], dim=1)

        xy_out1 = torch.concatenate([xy_left, xy_mid, xy_right], dim=2)
        xy_out2 = torch.concatenate([xy1_left, xy1_mid, xy1_right], dim=2)
        return (xy_out1 - xy_out2) * 0.25

    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """Call the gradient loss function.

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
        loss = 0.0
        loss += self._tv_weight * (self.generalized_loss(self._diff_x(y_true),
                                                         self._diff_x(y_pred)) +
                                   self.generalized_loss(self._diff_y(y_true),
                                                         self._diff_y(y_pred)))
        loss += self._tv2_weight * (self.generalized_loss(self._diff_xx(y_true),
                                                          self._diff_xx(y_pred)) +
                                    self.generalized_loss(self._diff_yy(y_true),
                                    self._diff_yy(y_pred)) +
                                    self.generalized_loss(self._diff_xy(y_true),
                                    self._diff_xy(y_pred)) * 2.)
        loss = loss / (self._tv_weight + self._tv2_weight)
        # TODO simplify to use MSE instead
        if not self._spatial:
            loss = loss.mean(dim=(1, 2, 3))
        return loss


class LaplacianPyramidLoss(nn.Module):
    """Laplacian Pyramid Loss Function

    Notes
    -----
    Channels last implementation on square images only.

    Parameters
    ----------
    max_levels
        The max number of laplacian pyramid levels to use. Default: `5`
    gaussian_size
        The size of the gaussian kernel. Default: `5`
    gaussian_sigma
        The gaussian sigma. Default: 2.0
    device
        The device to place the variables onto. Default: `"cpu"`
    spatial_output
        ``True`` to output the loss values spatially. ``False`` as scalar per item.
        Default: ``True``

    References
    ----------
    https://arxiv.org/abs/1707.05776
    https://github.com/nathanaelbosch/generative-latent-optimization/blob/master/utils.py
    """
    _weight: torch.Tensor
    _kernel: torch.Tensor

    def __init__(self,
                 max_levels: int = 5,
                 gaussian_size: int = 5,
                 gaussian_sigma: float = 1.0,
                 spatial_output: bool = True) -> None:
        logger.debug(parse_class_init(locals()))
        super().__init__()
        self._max_levels = max_levels
        self._gaussian_sigma = gaussian_sigma
        self._spatial = spatial_output
        self.register_buffer("_weight",
                             torch.Tensor([np.power(2., -2 * idx)
                                           for idx in range(max_levels + 1)]))
        self.register_buffer("_kernel", self._generate_gaussian_kernel(gaussian_size))

    def _generate_gaussian_kernel(self, size: int) -> torch.Tensor:
        """Obtain the base gaussian kernel for the Laplacian Pyramid

        Parameters
        ----------
        size
            The size of the kernel to create

        Returns
        -------
            The base three channel Gaussian kernel
        """
        assert size % 2 == 1, ("kernel size must be uneven")
        x_1 = np.linspace(- (size // 2), size // 2, size, dtype="float32")
        x_1 /= np.sqrt(2) * self._gaussian_sigma
        x_2 = x_1 ** 2
        kernel = np.exp(- x_2[:, None] - x_2[None, :])
        kernel /= kernel.sum()

        kernel = np.tile(kernel, (3, 1, 1, 1))
        return torch.from_numpy(kernel).float()

    def _conv_gaussian(self, inputs: torch.Tensor) -> torch.Tensor:
        """Perform Gaussian convolution on a batch of images.

        Parameters
        ----------
        inputs
            The input batch of images to perform Gaussian convolution on.

        Returns
        -------
        The convolved images
        """
        gauss_size = self._kernel.shape[2]
        padded_inputs = F.pad(inputs,
                              (gauss_size // 2, gauss_size // 2, gauss_size // 2, gauss_size // 2),
                              mode="replicate")
        return F.conv2d(padded_inputs,  # pylint:disable=not-callable
                        self._kernel,
                        groups=3)

    def _get_laplacian_pyramid(self, inputs: torch.Tensor) -> list[torch.Tensor]:
        """Obtain the Laplacian Pyramid.

        Parameters
        ----------
        inputs
            The input batch of images to run through the Laplacian Pyramid

        Returns
        -------
        The tensors produced from the Laplacian Pyramid
        """
        pyramid = []
        current = inputs
        for _ in range(self._max_levels):
            filtered = self._conv_gaussian(current)
            diff = current - filtered
            pyramid.append(diff)
            current = F.avg_pool2d(filtered, 2)  # pylint:disable=not-callable
        pyramid.append(current)
        return pyramid

    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """Calculate the Laplacian Pyramid Loss.

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
        pyramid_true = self._get_laplacian_pyramid(y_true)
        pyramid_pred = self._get_laplacian_pyramid(y_pred)

        losses = [F.l1_loss(o, t, reduction="none") for o, t in zip(pyramid_true, pyramid_pred)]
        if self._spatial:
            size = y_true.shape[-2:]
            loss = torch.stack(
                [x if x.shape[-2:] == size else (F.interpolate(x,
                                                               size=size,
                                                               mode="bilinear",
                                                               align_corners=False))
                 for x in losses]).swapaxes(0, 1) * self._weight[..., None, None, None]
        else:
            loss = torch.stack([x.mean(dim=(1, 2, 3)) for x in losses]).T * self._weight
        return loss.sum(dim=1)


class LInfNorm(nn.Module):
    """Calculate the L-inf norm as a loss function."""

    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """Call the L-inf norm loss function.

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
        diff = torch.abs(y_true - y_pred)
        loss = diff.amax(dim=(1, 2)).mean(dim=-1)
        return loss


class LogCosh(nn.Module):
    """Logarithm of the hyperbolic cosine of the prediction error. Ported from Keras implementation

    Parameters
    ----------
    spatial_output
        ``True`` to output the loss values spatially. ``False`` as scalar per item.
        Default: ``True``
    """
    def __init__(self, spatial_output: bool = True) -> None:
        logger.debug(parse_class_init(locals()))
        super().__init__()
        self._spatial = spatial_output

    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """Call the LogCosh loss function.

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
        diff = y_true - y_pred
        loss: torch.Tensor = (diff + F.softplus(diff * -2.0) -  # pylint:disable=not-callable
                              np.log(2))
        if not self._spatial:
            loss = loss.mean(dim=(1, 2, 3))
        return loss


__all__ = get_module_objects(__name__)
