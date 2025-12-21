#!/usr/bin/env python3
""" Custom Loss Functions for faceswap.py """

from __future__ import annotations
import logging
import typing as T

import numpy as np
from keras import Loss, backend as K
from keras import ops, Variable

from lib.logger import parse_class_init
from lib.utils import get_module_objects

if K.backend() == "torch":
    import torch  # pylint:disable=import-error
else:
    import tensorflow as tf  # pylint:disable=import-error  # type:ignore

if T.TYPE_CHECKING:
    from collections.abc import Callable
    from keras import KerasTensor

logger = logging.getLogger(__name__)


class FocalFrequencyLoss(Loss):
    """ Focal Frequencey Loss Function.

    A channels last implementation.

    Notes
    -----
    There is a bug in this implementation that will do an incorrect FFT if
    :attr:`patch_factor` >  ``1``, which means incorrect loss will be returned, so keep
    patch factor at 1.

    Parameters
    ----------
    alpha: float, Optional
        Scaling factor of the spectrum weight matrix for flexibility. Default: ``1.0``
    patch_factor: int, Optional
        Factor to crop image patches for patch-based focal frequency loss.
        Default: ``1``
    ave_spectrum: bool, Optional
        ``True`` to use minibatch average spectrum otherwise ``False``. Default: ``False``
    log_matrix: bool, Optional
        ``True`` to adjust the spectrum weight matrix by logarithm otherwise ``False``.
        Default: ``False``
    batch_matrix: bool, Optional
        ``True`` to calculate the spectrum weight matrix using batch-based statistics otherwise
        ``False``. Default: ``False``
    epsilon : float, Optional
        Small epsilon for safer weights scaling division. Default: `1e-6`


    References
    ----------
    https://arxiv.org/pdf/2012.12821.pdf
    https://github.com/EndlessSora/focal-frequency-loss
    """

    def __init__(self,
                 alpha: float = 1.0,
                 patch_factor: int = 1,
                 ave_spectrum: bool = False,
                 log_matrix: bool = False,
                 batch_matrix: bool = False,
                 epsilon: float = 1e-6) -> None:
        logger.debug(parse_class_init(locals()))
        super().__init__(name=self.__class__.__name__)
        self._alpha = alpha
        # TODO Fix bug where FFT will be incorrect if patch_factor > 1 for tensorflow
        self._patch_factor = patch_factor
        self._ave_spectrum = ave_spectrum
        self._log_matrix = log_matrix
        self._batch_matrix = batch_matrix
        self._epsilon = epsilon
        self._dims: tuple[int, int] = (0, 0)
        logger.debug("Initialized: %s", self.__class__.__name__)

    def _get_patches(self, inputs: KerasTensor) -> KerasTensor:
        """ Crop the incoming batch of images into patches as defined by :attr:`_patch_factor.

        Parameters
        ----------
        inputs: :class:`keras.KerasTensor`
            A batch of images to be converted into patches

        Returns
        -------
        :class:`keras.KerasTensor``
            The incoming batch converted into patches
        """
        patch_list = []
        patch_rows = self._dims[0] // self._patch_factor
        patch_cols = self._dims[1] // self._patch_factor
        for i in range(self._patch_factor):
            for j in range(self._patch_factor):
                row_from = i * patch_rows
                row_to = (i + 1) * patch_rows
                col_from = j * patch_cols
                col_to = (j + 1) * patch_cols
                patch_list.append(inputs[:, row_from: row_to, col_from: col_to, :])

        retval = ops.stack(patch_list, axis=1)
        return T.cast("KerasTensor", retval)

    def _tensor_to_frequency_spectrum(self, patch: KerasTensor) -> KerasTensor:
        """ Perform FFT to create the orthonomalized DFT frequencies.

        Parameters
        ----------
        inputs: :class:`keras.KerasTensor`
            The incoming batch of patches to convert to the frequency spectrum

        Returns
        -------
        :class:`keras.KerasTensor`
            The DFT frequencies split into real and imaginary numbers as float32
        """
        patch = T.cast("KerasTensor",
                       ops.transpose(patch, (0, 1, 4, 2, 3)))  # move channels to first

        assert K.backend() in ("torch", "tensorflow"), "Only Torch and Tensorflow are supported"
        if K.backend() == "torch":
            freq = torch.fft.fft2(patch,  # pylint:disable=not-callable  # type:ignore
                                  norm="ortho")
        else:
            patch = patch / np.sqrt(self._dims[0] * self._dims[1])  # Orthonormalization
            patch = T.cast("KerasTensor", ops.cast(patch, "complex64"))
            freq = tf.signal.fft2d(patch)[..., None]  # type:ignore

        freq = ops.stack([freq.real, freq.imag], axis=-1)

        if K.backend() == "tensorflow":
            freq = ops.cast(freq, "float32")

        freq = ops.transpose(freq, (0, 1, 3, 4, 2, 5))  # channels to last
        return T.cast("KerasTensor", freq)

    def _get_weight_matrix(self, freq_true: KerasTensor, freq_pred: KerasTensor) -> KerasTensor:
        """ Calculate a continuous, dynamic weight matrix based on current Euclidean distance.

        Parameters
        ----------
        freq_true: :class:`keras.KerasTensor`
            The real and imaginary DFT frequencies for the true batch of images
        freq_pred: :class:`keras.KerasTensor`
            The real and imaginary DFT frequencies for the predicted batch of images

        Returns
        -------
        :class:`keras.KerasTensor`
            The weights matrix for prioritizing hard frequencies
        """
        weights = ops.square(freq_pred - freq_true)
        weights = ops.sqrt(weights[..., 0] + weights[..., 1])
        weights = ops.power(weights, self._alpha)

        if self._log_matrix:  # adjust the spectrum weight matrix by logarithm
            weights = ops.log(weights + 1.0)

        if self._batch_matrix:  # calculate the spectrum weight matrix using batch-based statistics
            scale = ops.max(weights)
        else:
            scale = ops.max(weights, axis=(-2, -3), keepdims=True)
        weights = weights / ops.maximum(scale, self._epsilon)

        weights = ops.clip(weights, x_min=0.0, x_max=1.0)

        return T.cast("KerasTensor", weights)

    @classmethod
    def _calculate_loss(cls,
                        freq_true: KerasTensor,
                        freq_pred: KerasTensor,
                        weight_matrix: KerasTensor) -> KerasTensor:
        """ Perform the loss calculation on the DFT spectrum applying the weights matrix.

        Parameters
        ----------
        freq_true: :class:`keras.KerasTensor`
            The real and imaginary DFT frequencies for the true batch of images
        freq_pred: :class:`keras.KerasTensor`
            The real and imaginary DFT frequencies for the predicted batch of images

        Returns
        :class:`keras.KerasTensor`
            The final loss matrix
        """

        tmp = ops.square(freq_pred - freq_true)  # freq distance using squared Euclidean distance

        freq_distance = tmp[..., 0] + tmp[..., 1]
        loss = weight_matrix * freq_distance  # dynamic spectrum weighting (Hadamard product)

        return T.cast("KerasTensor", ops.mean(loss))

    def call(self, y_true: KerasTensor, y_pred: KerasTensor) -> KerasTensor:
        """ Call the Focal Frequency Loss Function.

        Parameters
        ----------
        y_true: :class:`keras.KerasTensor`
            The ground truth batch of images
        y_pred: :class:`keras.KerasTensor`
            The predicted batch of images

        Returns
        -------
        :class:`keras.KerasTensor`
            The loss for this batch of images
        """
        if not all(self._dims):
            rows, cols = y_true.shape[1:3]
            assert rows is not None and cols is not None
            assert cols % self._patch_factor == 0 and rows % self._patch_factor == 0, (
                "Patch factor must be a divisor of the image height and width")
            self._dims = (rows, cols)

        patches_true = self._get_patches(y_true)
        patches_pred = self._get_patches(y_pred)

        freq_true = self._tensor_to_frequency_spectrum(patches_true)
        freq_pred = self._tensor_to_frequency_spectrum(patches_pred)

        if self._ave_spectrum:  # whether to use minibatch average spectrum
            freq_true = T.cast("KerasTensor", ops.mean(freq_true, axis=0, keepdims=True))
            freq_pred = T.cast("KerasTensor", ops.mean(freq_pred, axis=0, keepdims=True))

        weight_matrix = self._get_weight_matrix(freq_true, freq_pred)
        return self._calculate_loss(freq_true, freq_pred, weight_matrix)


class GeneralizedLoss(Loss):
    """  Generalized function used to return a large variety of mathematical loss functions.

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
    alpha: float, optional
        Penalty factor. Larger number give larger weight to large deviations. Default: `1.0`
    beta: float, optional
        Scale factor used to adjust to the input scale (i.e. inputs of mean `1e-4` or `256`).
        Default: `1.0/255.0`
    """
    def __init__(self, alpha: float = 1.0, beta: float = 1.0/255.0) -> None:
        logger.debug(parse_class_init(locals()))
        super().__init__(name=self.__class__.__name__)
        self._alpha = alpha
        self._beta = beta
        logger.debug("Initialized: %s", self.__class__.__name__)

    def call(self, y_true: KerasTensor, y_pred: KerasTensor) -> KerasTensor:
        """ Call the Generalized Loss Function

        Parameters
        ----------
        y_true: :class:`keras.KerasTensor`
            The ground truth value
        y_pred: :class:`keras.KerasTensor`
            The predicted value

        Returns
        -------
        :class:`keras.KerasTensor`
            The loss value from the results of function(y_pred - y_true)
        """
        diff = y_pred - y_true
        second = (ops.power(ops.power(diff/self._beta, 2.) / ops.abs(2. - self._alpha) + 1.,
                            (self._alpha / 2.)) - 1.)
        loss = (ops.abs(2. - self._alpha)/self._alpha) * second
        loss = ops.mean(loss, axis=-1) * self._beta
        return T.cast("KerasTensor", loss)


class GradientLoss(Loss):
    """ Gradient Loss Function.

    Calculates the first and second order gradient difference between pixels of an image in the x
    and y dimensions. These gradients are then compared between the ground truth and the predicted
    image and the difference is taken. When used as a loss, its minimization will result in
    predicted images approaching the same level of sharpness / blurriness as the ground truth.

    References
    ----------
    TV+TV2 Regularization with Non-Convex Sparseness-Inducing Penalty for Image Restoration,
    Chengwu Lu & Hua Huang, 2014 - http://downloads.hindawi.com/journals/mpe/2014/790547.pdf
    """
    def __init__(self) -> None:
        logger.debug(parse_class_init(locals()))
        super().__init__(name=self.__class__.__name__)
        self.generalized_loss = GeneralizedLoss(alpha=1.9999)
        self._tv_weight = 1.0
        self._tv2_weight = 1.0
        logger.debug("Initialized: %s", self.__class__.__name__)

    @classmethod
    def _diff_x(cls, img: KerasTensor) -> KerasTensor:
        """ X Difference """
        x_left = img[:, :, 1:2, :] - img[:, :, 0:1, :]
        x_inner = img[:, :, 2:, :] - img[:, :, :-2, :]
        x_right = img[:, :, -1:, :] - img[:, :, -2:-1, :]
        x_out = ops.concatenate([x_left, x_inner, x_right], axis=2)
        return T.cast("KerasTensor", x_out) * 0.5

    @classmethod
    def _diff_y(cls, img: KerasTensor) -> KerasTensor:
        """ Y Difference """
        y_top = img[:, 1:2, :, :] - img[:, 0:1, :, :]
        y_inner = img[:, 2:, :, :] - img[:, :-2, :, :]
        y_bot = img[:, -1:, :, :] - img[:, -2:-1, :, :]
        y_out = ops.concatenate([y_top, y_inner, y_bot], axis=1)
        return T.cast("KerasTensor", y_out) * 0.5

    @classmethod
    def _diff_xx(cls, img: KerasTensor) -> KerasTensor:
        """ X-X Difference """
        x_left = img[:, :, 1:2, :] + img[:, :, 0:1, :]
        x_inner = img[:, :, 2:, :] + img[:, :, :-2, :]
        x_right = img[:, :, -1:, :] + img[:, :, -2:-1, :]
        x_out = ops.concatenate([x_left, x_inner, x_right], axis=2)
        return x_out - 2.0 * img

    @classmethod
    def _diff_yy(cls, img: KerasTensor) -> KerasTensor:
        """ Y-Y Difference """
        y_top = img[:, 1:2, :, :] + img[:, 0:1, :, :]
        y_inner = img[:, 2:, :, :] + img[:, :-2, :, :]
        y_bot = img[:, -1:, :, :] + img[:, -2:-1, :, :]
        y_out = ops.concatenate([y_top, y_inner, y_bot], axis=1)
        return y_out - 2.0 * img

    @classmethod
    def _diff_xy(cls, img: KerasTensor) -> KerasTensor:
        """ X-Y Difference """
        # xout1
        # Left
        top = img[:, 1:2, 1:2, :] + img[:, 0:1, 0:1, :]
        inner = img[:, 2:, 1:2, :] + img[:, :-2, 0:1, :]
        bottom = img[:, -1:, 1:2, :] + img[:, -2:-1, 0:1, :]
        xy_left = ops.concatenate([top, inner, bottom], axis=1)
        # Mid
        top = img[:, 1:2, 2:, :] + img[:, 0:1, :-2, :]
        mid = img[:, 2:, 2:, :] + img[:, :-2, :-2, :]
        bottom = img[:, -1:, 2:, :] + img[:, -2:-1, :-2, :]
        xy_mid = ops.concatenate([top, mid, bottom], axis=1)
        # Right
        top = img[:, 1:2, -1:, :] + img[:, 0:1, -2:-1, :]
        inner = img[:, 2:, -1:, :] + img[:, :-2, -2:-1, :]
        bottom = img[:, -1:, -1:, :] + img[:, -2:-1, -2:-1, :]
        xy_right = ops.concatenate([top, inner, bottom], axis=1)

        # Xout2
        # Left
        top = img[:, 0:1, 1:2, :] + img[:, 1:2, 0:1, :]
        inner = img[:, :-2, 1:2, :] + img[:, 2:, 0:1, :]
        bottom = img[:, -2:-1, 1:2, :] + img[:, -1:, 0:1, :]
        xy_left = ops.concatenate([top, inner, bottom], axis=1)
        # Mid
        top = img[:, 0:1, 2:, :] + img[:, 1:2, :-2, :]
        mid = img[:, :-2, 2:, :] + img[:, 2:, :-2, :]
        bottom = img[:, -2:-1, 2:, :] + img[:, -1:, :-2, :]
        xy_mid = ops.concatenate([top, mid, bottom], axis=1)
        # Right
        top = img[:, 0:1, -1:, :] + img[:, 1:2, -2:-1, :]
        inner = img[:, :-2, -1:, :] + img[:, 2:, -2:-1, :]
        bottom = img[:, -2:-1, -1:, :] + img[:, -1:, -2:-1, :]
        xy_right = ops.concatenate([top, inner, bottom], axis=1)

        xy_out1 = T.cast("KerasTensor", ops.concatenate([xy_left, xy_mid, xy_right], axis=2))
        xy_out2 = T.cast("KerasTensor", ops.concatenate([xy_left, xy_mid, xy_right], axis=2))
        return (xy_out1 - xy_out2) * 0.25

    def call(self, y_true: KerasTensor, y_pred: KerasTensor) -> KerasTensor:
        """ Call the gradient loss function.

        Parameters
        ----------
        y_true: :class:`keras.KerasTensor`
            The ground truth value
        y_pred: :class:`keras.KerasTensor`
            The predicted value

        Returns
        -------
        :class:`keras.KerasTensor`
            The loss value
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
        return T.cast("KerasTensor", loss)


class LaplacianPyramidLoss(Loss):
    """ Laplacian Pyramid Loss Function

    Notes
    -----
    Channels last implementation on square images only.

    Parameters
    ----------
    max_levels: int, Optional
        The max number of laplacian pyramid levels to use. Default: `5`
    gaussian_size: int, Optional
        The size of the gaussian kernel. Default: `5`
    gaussian_sigma: float, optional
        The gaussian sigma. Default: 2.0

    References
    ----------
    https://arxiv.org/abs/1707.05776
    https://github.com/nathanaelbosch/generative-latent-optimization/blob/master/utils.py
    """
    def __init__(self,
                 max_levels: int = 5,
                 gaussian_size: int = 5,
                 gaussian_sigma: float = 1.0) -> None:
        logger.debug(parse_class_init(locals()))
        super().__init__(name=self.__class__.__name__)
        self._max_levels = max_levels
        self._weights = Variable([np.power(2., -2 * idx) for idx in range(max_levels + 1)],
                                 trainable=False)
        self._gaussian_kernel = self._get_gaussian_kernel(gaussian_size, gaussian_sigma)
        logger.debug("Initialized: %s", self.__class__.__name__)

    @classmethod
    def _get_gaussian_kernel(cls, size: int, sigma: float) -> KerasTensor:
        """ Obtain the base gaussian kernel for the Laplacian Pyramid.

        Parameters
        ----------
        size: int, Optional
            The size of the gaussian kernel
        sigma: float
            The gaussian sigma

        Returns
        -------
        :class:`keras.KerasTensor`
            The base single channel Gaussian kernel
        """
        assert size % 2 == 1, ("kernel size must be uneven")
        x_1 = np.linspace(- (size // 2), size // 2, size, dtype="float32")
        x_1 /= np.sqrt(2)*sigma
        x_2 = x_1 ** 2
        kernel = np.exp(- x_2[:, None] - x_2[None, :])
        kernel /= kernel.sum()
        kernel = np.reshape(kernel, (size, size, 1, 1))
        return Variable(kernel, trainable=False)

    def _conv_gaussian(self, inputs: KerasTensor) -> KerasTensor:
        """ Perform Gaussian convolution on a batch of images.

        Parameters
        ----------
        inputs: :class:`keras.KerasTensor`
            The input batch of images to perform Gaussian convolution on.

        Returns
        -------
        :class:`keras.KerasTensor`
            The convolved images
        """
        channels = inputs.shape[-1]
        gauss = ops.tile(self._gaussian_kernel, (1, 1, 1, channels))

        # TF doesn't implement replication padding like pytorch. This is an inefficient way to
        # implement it for a square guassian kernel
        # TODO Make this pure pytorch code
        gauss_shape = self._gaussian_kernel.shape[1]
        assert gauss_shape is not None
        size = gauss_shape // 2
        padded_inputs = inputs
        for _ in range(size):
            padded_inputs = ops.pad(padded_inputs,
                                    ([0, 0], [1, 1], [1, 1], [0, 0]),
                                    mode="symmetric")

        retval = ops.conv(padded_inputs, gauss, strides=1, padding="valid")
        return T.cast("KerasTensor", retval)

    def _get_laplacian_pyramid(self, inputs: KerasTensor) -> list[KerasTensor]:
        """ Obtain the Laplacian Pyramid.

        Parameters
        ----------
        inputs: :class:`keras.KerasTensor`
            The input batch of images to run through the Laplacian Pyramid

        Returns
        -------
        list
            The tensors produced from the Laplacian Pyramid
        """
        pyramid = []
        current = inputs
        for _ in range(self._max_levels):
            gauss = self._conv_gaussian(current)
            diff = current - gauss
            pyramid.append(diff)
            current = ops.average_pool(gauss, (2, 2), strides=(2, 2), padding="valid")
        pyramid.append(current)
        return pyramid

    def call(self, y_true: KerasTensor, y_pred: KerasTensor) -> KerasTensor:
        """ Calculate the Laplacian Pyramid Loss.

        Parameters
        ----------
        y_true: :class:`keras.KerasTensor`
            The ground truth value
        y_pred: :class:`keras.KerasTensor`
            The predicted value

        Returns
        -------
        :class:`keras.KerasTensor`
            The loss value
        """
        pyramid_true = self._get_laplacian_pyramid(y_true)
        pyramid_pred = self._get_laplacian_pyramid(y_pred)

        losses = ops.stack(
            [ops.sum(ops.abs(ppred - ptrue)) / ops.cast(ops.prod(ops.shape(ptrue)), "float32")
             for ptrue, ppred in zip(pyramid_true, pyramid_pred)])
        loss = ops.sum(losses * self._weights)
        return T.cast("KerasTensor", loss)


class LInfNorm(Loss):
    """ Calculate the L-inf norm as a loss function. """
    def __init__(self, *args, **kwargs) -> None:
        logger.debug(parse_class_init(locals()))
        super().__init__(*args, name=self.__class__.__name__, **kwargs)
        logger.debug("Initialized: %s", self.__class__.__name__)

    def call(self, y_true: KerasTensor, y_pred: KerasTensor) -> KerasTensor:
        """ Call the L-inf norm loss function.

        Parameters
        ----------
        y_true: :class:`keras.KerasTensor`
            The ground truth value
        y_pred: :class:`keras.KerasTensor`
            The predicted value

        Returns
        -------
        :class:`keras.KerasTensor`
            The loss value
        """
        diff = ops.abs(y_true - y_pred)
        max_loss = ops.max(diff, axis=(1, 2), keepdims=True)
        loss = ops.mean(max_loss, axis=-1)
        return T.cast("KerasTensor", loss)


class LossWrapper(Loss):
    """ A wrapper class for multiple keras losses to enable multiple masked weighted loss
    functions on a single output.

    Notes
    -----
    Whilst Keras does allow for applying multiple weighted loss functions, it does not allow
    for an easy mechanism to add additional data (in our case masks) that are batch specific
    but are not fed in to the model.

    This wrapper receives this additional mask data for the batch stacked onto the end of the
    color channels of the received :attr:`y_true` batch of images. These masks are then split
    off the batch of images and applied to both the :attr:`y_true` and :attr:`y_pred` tensors
    prior to feeding into the loss functions.

    For example, for an image of shape (4, 128, 128, 3) 3 additional masks may be stacked onto
    the end of y_true, meaning we receive an input of shape (4, 128, 128, 6). This wrapper then
    splits off (4, 128, 128, 3:6) from the end of the tensor, leaving the original y_true of
    shape (4, 128, 128, 3) ready for masking and feeding through the loss functions.
    """
    def __init__(self, name="LossWrapper", reduction="sum_over_batch_size") -> None:
        logger.debug(parse_class_init(locals()))
        super().__init__(name=name, reduction=reduction)
        self._loss_functions: list[Loss | Callable] = []
        self._loss_weights: list[float] = []
        self._mask_channels: list[int] = []
        logger.debug("Initialized: %s", self.__class__.__name__)

    def add_loss(self,
                 function: Callable | Loss,
                 weight: float = 1.0,
                 mask_channel: int = -1) -> None:
        """ Add the given loss function with the given weight to the loss function chain.

        Parameters
        ----------
        function: :class:`keras.losses.Loss`
            The loss function to add to the loss chain
        weight: float, optional
            The weighting to apply to the loss function. Default: `1.0`
        mask_channel: int, optional
            The channel in the `y_true` image that the mask exists in. Set to `-1` if there is no
            mask for the given loss function. Default: `-1`
        """
        logger.debug("Adding loss: (function: %s, weight: %s, mask_channel: %s)",
                     function, weight, mask_channel)
        # Loss must be compiled inside LossContainer for keras to handle distibuted strategies
        self._loss_functions.append(function)
        self._loss_weights.append(weight)
        self._mask_channels.append(mask_channel)

    def call(self, y_true: KerasTensor, y_pred: KerasTensor) -> KerasTensor:
        """ Call the sub loss functions for the loss wrapper.

        Loss is returned as the weighted sum of the chosen losses.

        If masks are being applied to the loss function inputs, then they should be included as
        additional channels at the end of :attr:`y_true`, so that they can be split off and
        applied to the actual inputs to the selected loss function(s).

        Parameters
        ----------
        y_true: :class:`keras.KerasTensor`
            The ground truth batch of images, with any required masks stacked on the end
        y_pred: :class:`keras.KerasTensor`
            The batch of model predictions

        Returns
        -------
        :class:`keras.KerasTensor`
            The final weighted loss
        """
        loss = 0.0
        for func, weight, mask_channel in zip(self._loss_functions,
                                              self._loss_weights,
                                              self._mask_channels):
            logger.trace("Processing loss function: "  # type:ignore[attr-defined]
                         "(func: %s, weight: %s, mask_channel: %s)",
                         func, weight, mask_channel)
            n_true, n_pred = self._apply_mask(y_true, y_pred, mask_channel)
            loss += (func(n_true, n_pred) * weight)
        return T.cast("KerasTensor", loss)

    @classmethod
    def _apply_mask(cls,
                    y_true: KerasTensor,
                    y_pred: KerasTensor,
                    mask_channel: int,
                    mask_prop: float = 1.0) -> tuple[KerasTensor, KerasTensor]:
        """ Apply the mask to the input y_true and y_pred. If a mask is not required then
        return the unmasked inputs.

        Parameters
        ----------
        y_true: :class:`keras.KerasTensor`
            The ground truth value
        y_pred: :class:`keras.KerasTensor`
            The predicted value
        mask_channel: int
            The channel within y_true that the required mask resides in
        mask_prop: float, optional
            The amount of mask propagation. Default: `1.0`

        Returns
        -------
        :class:`keras.KerasTensor`
            The ground truth batch of images, with the required mask applied
        :class:`keras.KerasTensor`
            The predicted batch of images with the required mask applied
        """
        if mask_channel == -1:
            logger.trace("No mask to apply")  # type:ignore[attr-defined]
            return y_true[..., :3], y_pred[..., :3]

        logger.trace("Applying mask from channel %s", mask_channel)  # type:ignore[attr-defined]

        mask = ops.tile(ops.expand_dims(y_true[..., mask_channel], axis=-1), (1, 1, 1, 3))
        mask_as_k_inv_prop = 1 - mask_prop
        mask = (mask * mask_prop) + mask_as_k_inv_prop

        m_true = y_true[..., :3] * mask
        m_pred = y_pred[..., :3] * mask

        return m_true, m_pred


__all__ = get_module_objects(__name__)
