#!/usr/bin/env python3
"""Keras implementation of Perceptual Loss Functions for faceswap.py """
from __future__ import annotations

import logging

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from lib.logger import parse_class_init
from lib.utils import FaceswapError, get_module_objects


logger = logging.getLogger(__name__)


class GMSDLoss(nn.Module):
    """Gradient Magnitude Similarity Deviation Loss.

    Improved image quality metric over MS-SSIM with easier calculations

    Parameters
    ----------
    spatial_output
        ``True`` to output the loss values spatially. ``False`` as scalar per item.
        Default: ``True``

    References
    ----------
    http://www4.comp.polyu.edu.hk/~cslzhang/IQA/GMSD/GMSD.htm
    https://arxiv.org/ftp/arxiv/papers/1308/1308.3052.pdf
    """
    _scharr_edges: torch.Tensor

    def __init__(self, spatial_output: bool = True) -> None:
        logger.debug(parse_class_init(locals()))
        super().__init__()
        self._spatial = spatial_output
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
            out = torch.atan2(gx, gy)
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
        true_edge = self._map_scharr_edges(y_true, True)
        pred_edge = self._map_scharr_edges(y_pred, True)
        epsilon = 0.0025
        upper = 2.0 * true_edge * pred_edge
        lower = torch.square(true_edge) + torch.square(pred_edge)
        gms = (upper + epsilon) / (lower + epsilon)
        if self._spatial:
            # per-pixel similarity reasonable proxy for spatial loss
            loss = 1.0 - gms.mean(dim=1)[:, None]
        else:
            loss = torch.std(gms, dim=(1, 2, 3))
        return loss


class _SSIM(nn.Module):  # pylint:disable=abstract-method
    """Parent class for SSIM and MSSIM loss functions

    Parameters
    ----------
    max_val
        The dynamic range of the images (i.e., the difference between the maximum the and minimum
        allowed values). Default `1.0` (0.0 - 1.0)
    filter_size
        Size of gaussian filter. Default: `11`
    filter_sigma:
        Width of gaussian filter. Default: 1.5
    k1
        The K1 value. Default: `0.01`
    k2
        The K2 value. Default: `0.03` (SSIM is less sensitivity to K2 for lower values, so
        it would be better if we took the values in the range of 0 < K2 < 0.4).
    spatial_output
        ``True`` to output the loss values spatially. ``False`` as scalar per item.
        Default: ``True``

    Reference
    ---------
    https://github.com/tensorflow/tensorflow/blob/v2.16.1/tensorflow/python/ops/image_ops_impl.py
    """
    _kernel: torch.Tensor

    def __init__(self,
                 max_val: float = 1.0,
                 filter_size: int = 11,
                 filter_sigma: float = 1.5,
                 k1: float = 0.01,
                 k2: float = 0.03,
                 spatial_output: bool = True) -> None:
        super().__init__()
        self._max_value = max_val
        self._filter_sigma = filter_sigma
        self._k1 = k1
        self._k2 = k2
        self._spatial = spatial_output
        self.register_buffer("_kernel", self._fspecial_gauss(filter_size, filter_sigma))

    def _fspecial_gauss(self, size: int, sigma: float) -> torch.Tensor:
        """Function to mimic the 'fspecial' gaussian MATLAB function.

        Parameters
        ----------
        filter_size
            size of gaussian filter
        sigma
            width of gaussian filter

        Returns
        -------
        The gaussian kernel in channels first depthwise format (1,1,H,W)
        """
        coords = torch.arange(0, size, dtype=torch.float32)
        coords -= (size - 1) / 2.

        gauss = coords ** 2
        gauss *= (-0.5 / (sigma ** 2))

        gauss = gauss.reshape(1, -1) + gauss.reshape(-1, 1)
        gauss = gauss.reshape(1, -1)  # For ops.softmax().
        gauss = F.softmax(gauss, dim=-1)
        return gauss.reshape(1, 1, size, size)

    def _reducer(self, image: torch.Tensor) -> torch.Tensor:
        """Computes local averages from a set of images

        Parameters
        ----------
        image
            The images to be processed (N,C,H,W)

        Returns
        -------
        The reduced image
        """
        shape = image.shape
        channels = shape[-3]
        kernel = self._kernel.repeat(channels, 1, 1, 1)
        x = image.reshape(-1, *shape[-3:])
        pad = self._kernel.shape[-1] // 2
        if self._spatial:
            x = F.pad(x, [pad, pad, pad, pad], mode="reflect")  # preserve spatial dims
        y = F.conv2d(x, kernel, groups=channels)  # pylint:disable=not-callable
        return y.reshape((*shape[:-3], *y.shape[1:]))

    def _ssim_helper(self,
                     image1: torch.Tensor,
                     image2: torch.Tensor,
                     compensation: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
        """Helper function for computing SSIM

        Parameters
        ----------
        image1
            The first set of images (N,C,H,W)
        image2
            The second set of images (N,C,H,W)
        compensation
            Compensation factor. Default: `1.0`

        Returns
        -------
        ssim
            The channel-wise SSIM
        contrast
            The channel-wise contrast-structure
        """
        c_1 = (self._k1 * self._max_value) ** 2
        c_2 = (self._k2 * self._max_value) ** 2

        mean0 = self._reducer(image1)
        mean1 = self._reducer(image2)

        num0 = mean0 * mean1 * 2.0
        den0 = mean0 ** 2 + mean1 ** 2
        luminance = (num0 + c_1) / (den0 + c_1)

        num1 = self._reducer(image1 * image2) * 2.0
        den1 = self._reducer(image1 ** 2 + image2 ** 2)

        c_2 *= compensation
        cs_ = (num1 - num0 + c_2) / ((den1 - den0).clamp(min=0) + c_2)

        return luminance, cs_

    def _ssim_per_channel(self,
                          image1: torch.Tensor,
                          image2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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
        luminance, cs_ = self._ssim_helper(image1, image2)
        ssim_val = luminance * cs_
        if not self._spatial:  # Average over height, width.
            ssim_val = ssim_val.mean(dim=(-2, -1))
            cs_ = cs_.mean(dim=(-2, -1))
        return ssim_val, cs_


class SSIMLoss(_SSIM):
    """Computes SSIM index between img1 and img2.

    This function is based on the standard SSIM implementation from:
    Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004). Image
    quality assessment: from error visibility to structural similarity. IEEE
    transactions on image processing.

    Note: The true SSIM is only defined on grayscale.  This function does not
    perform any color-space transform.  (If the input is already YUV, then it will
    compute YUV SSIM average.)

    Details:
        - 11x11 Gaussian filter of width 1.5 is used.
        - k1 = 0.01, k2 = 0.03 as in the original paper.

    The filter is reduced in size of the image is smaller than 11x11.

    Reference
    ---------
    https://github.com/tensorflow/tensorflow/blob/v2.16.1/tensorflow/python/ops/image_ops_impl.py
    """

    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """Call the SSIM Loss Function.

        Parameters
        ----------
        y_true
            The input batch of ground truth images
        y_pred
            The input batch of predicted images

        Returns
        -------
        The final SSIM for each item in the batch
        """
        ssim_per_channel, _ = self._ssim_per_channel(y_true, y_pred)
        loss = 1.0 - ssim_per_channel
        if not self._spatial:
            loss = loss.mean(dim=-1)
        return loss


class MSSIMLoss(_SSIM):
    """Computes the MS-SSIM between img1 and img2.

    This function assumes that `img1` and `img2` are image batches, i.e. the last
    three dimensions are [height, width, channels].

    Note: The true SSIM is only defined on grayscale.  This function does not
    perform any color-space transform.  (If the input is already YUV, then it will
    compute YUV SSIM average.)

    Original paper: Wang, Zhou, Eero P. Simoncelli, and Alan C. Bovik. "Multiscale
    structural similarity for image quality assessment." Signals, Systems and
    Computers, 2004.

    Details:
        - 11x11 Gaussian filter of width 1.5 is used.
        - k1 = 0.01, k2 = 0.03 as in the original paper.

    The filter is reduced in size if the smallest image is smaller than 11x11.

    Parameters
    ----------
    max_val
        The dynamic range of the images (i.e., the difference between the maximum the and minimum
        allowed values). Default `1.0` (0.0 - 1.0)
    filter_size
        Size of gaussian filter. Default: `11`
    filter_sigma:
        Width of gaussian filter. Default: 1.5
    k1
        The K1 value. Default: `0.01`
    k2
        The K2 value. Default: `0.03` (SSIM is less sensitivity to K2 for lower values, so
        it would be better if we took the values in the range of 0 < K2 < 0.4).
    spatial_output
        ``True`` to output the loss values spatially. ``False`` as scalar per item.
        Default: ``True``
    power_factors
        Iterable of weights for each of the scales. The number of scales used is the length of the
        list. Index 0 is the unscaled resolution's weight and each increasing scale corresponds to
        the image being downsampled by 2. Defaults to the values obtained in the original paper.
        Default: (0.0448, 0.2856, 0.3001, 0.2363, 0.1333)

    Reference
    ---------
    https://github.com/tensorflow/tensorflow/blob/v2.16.1/tensorflow/python/ops/image_ops_impl.py
    """
    _power_factors: torch.Tensor
    _divisor_tensor: torch.Tensor

    def __init__(self,
                 max_val: float = 1.0,
                 filter_size: int = 11,
                 filter_sigma: float = 1.5,
                 k1: float = 0.01,
                 k2: float = 0.03,
                 spatial_output: bool = True,
                 power_factors: tuple[float, ...] = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333)
                 ) -> None:
        super().__init__(max_val, filter_size, filter_sigma, k1, k2, spatial_output)
        self._divisor = [1, 1, 2, 2]
        self.register_buffer("_power_factors", torch.Tensor(power_factors).float())
        self.register_buffer("_divisor_tensor", torch.Tensor(self._divisor[1:]).int())
        self._validated = False

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
        logger.trace("[MSSIM] scale id: %s, size: %s", idx, size)  # type:ignore[attr-defined]
        if idx > 0:
            size = self._get_smallest_size(size // 2, idx - 1)
        return size

    def _validate_kernel(self, image: torch.Tensor) -> None:
        """Validate that the kernel is an appropriate size for the smallest scale image. If not,
        create a new kernel and show warning. Validation is run once on first batch of images seen

        Parameters
        ----------
        image
            A batch of incoming images to perform size validation on
        """
        if self._validated:
            return
        im_size = image.shape[2]
        smallest_scale = self._get_smallest_size(im_size, len(self._power_factors) - 1)
        kernel_size = self._kernel.shape[-1]

        if smallest_scale >= kernel_size:
            logger.info("[MSSIM] Inbound images are valid. smallest_scale: %s, kernel_size: %s",
                        smallest_scale, kernel_size)
            self._validated = True
            return

        logger.warning("[MSSIM] Output size %spx is below 176px. The MS-SSIM kernel must be "
                       "adjusted to accommodate. You will likely get better results using SSIM.",
                       im_size)
        del self._kernel
        flt = smallest_scale - 1 if smallest_scale % 2 == 0 else smallest_scale
        if flt < 3:
            raise FaceswapError("The output size of the selected model is too small for MS-SSIM. "
                                "Use SSIM instead.")
        logger.debug("[MSSIM] Adjusting filter kernel to %s from %s for smallest scale %s.",
                     flt, kernel_size, smallest_scale)
        self._kernel = self._fspecial_gauss(flt, self._filter_sigma).to(image.device)
        self._validated = True

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
                y_pred: torch.Tensor) -> torch.Tensor:
        """Perform the MSSISM calculation.

        Ported from Tensorflow implementation `image.ssim_multiscale`

        Parameters
        ----------
        y_true
            The ground truth value
        y_pred
            The predicted value
        """
        images = [y_true, y_pred]
        shapes = [y_true.shape, y_pred.shape]
        heads = [s[:-3] for s in shapes]  # Batch dimensions
        tails = [s[-3:] for s in shapes]  # Image dimensions
        mcs = []
        ssim_per_channel = None
        size = y_true.shape[-1]
        for k in range(len(self._power_factors)):
            if k > 0:
                # Avg pool takes rank 4 tensors. Flatten leading dimensions.
                flat_images = [(x.reshape(-1, *t)) for x, t in zip(images, tails)]
                remainder = torch.tensor(tails[0], device=y_pred.device) % self._divisor_tensor
                if (remainder != 0).any():
                    flat_images = self._do_pad(flat_images, remainder)

                downscaled = [F.avg_pool2d(x,  # pylint:disable=not-callable
                                           self._divisor[2:],
                                           stride=self._divisor[2:],
                                           padding=0)
                              for x in flat_images]
                tails = [x.shape[1:] for x in downscaled]
                images = [x.reshape(*h, *t) for x, h, t in zip(downscaled, heads, tails)]

            # Overwrite previous ssim value since we only need the last one.
            ssim_per_channel, cs_ = self._ssim_per_channel(images[0], images[1])
            if self._spatial:
                cs_ = F.interpolate(cs_, size=size, mode="bilinear", align_corners=False)
            mcs.append(F.relu(cs_))

        mcs.pop()  # Remove the cs score for the last scale.
        assert ssim_per_channel is not None
        if self._spatial:
            ssim_per_channel = F.interpolate(ssim_per_channel,
                                             size=size,
                                             mode="bilinear",
                                             align_corners=False)
        mcs_and_ssim = torch.stack(mcs + [F.relu(ssim_per_channel)], dim=-1)
        ms_ssim = torch.prod(mcs_and_ssim ** self._power_factors, dim=-1)
        if not self._spatial:
            ms_ssim = ms_ssim.mean(dim=-1)  # Avg over color channels.
        return ms_ssim

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
        self._validate_kernel(y_true)
        ms_ssim = self._mssism(y_true, y_pred)
        ms_ssim_loss = 1. - ms_ssim
        return ms_ssim_loss


__all__ = get_module_objects(__name__)
