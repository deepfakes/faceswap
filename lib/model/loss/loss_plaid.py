#!/usr/bin/env python3
""" Custom Loss Functions for faceswap.py """

from __future__ import absolute_import

import logging
from typing import Callable, List, Tuple

import numpy as np
import plaidml

from keras import backend as K
from lib.plaidml_utils import pad
from lib.utils import FaceswapError

from .feature_loss_plaid import LPIPSLoss  #pylint:disable=unused-import # noqa
from .perceptual_loss_plaid import DSSIMObjective, GMSDLoss, LDRFLIPLoss, MSSIMLoss  #pylint:disable=unused-import # noqa

logger = logging.getLogger(__name__)  # pylint:disable=invalid-name


class FocalFrequencyLoss():  # pylint:disable=too-few-public-methods
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
                 batch_matrix: bool = False) -> None:
        self._alpha = alpha
        self._patch_factor = patch_factor
        self._ave_spectrum = ave_spectrum
        self._log_matrix = log_matrix
        self._batch_matrix = batch_matrix
        self._dims: Tuple[int, int] = (0, 0)

    def __call__(self,
                 y_true: plaidml.tile.Value,
                 y_pred: plaidml.tile.Value) -> plaidml.tile.Value:
        """ Call the Focal Frequency Loss Function.

        # TODO Not implemented as:
          - We need a PlaidML replacement for tf.signal
          - The dimensions do not appear to be readable for y_pred

        Parameters
        ----------
        y_true: :class:`plaidml.tile.Value`
            The ground truth batch of images
        y_pred: :class:`plaidml.tile.Value`
            The predicted batch of images

        Returns
        -------
        :class:`plaidml.tile.Value`
            The loss for this batch of images
        """
        raise FaceswapError("Focal Frequency Loss is not currently compatible with PlaidML. "
                            "Please select a different Loss method.")


class GeneralizedLoss():  # pylint:disable=too-few-public-methods
    """  Generalized function used to return a large variety of mathematical loss functions.

    The primary benefit is a smooth, differentiable version of L1 loss.

    References
    ----------
    Barron, J. A More General Robust Loss Function - https://arxiv.org/pdf/1701.03077.pdf

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
        self._alpha = alpha
        self._beta = beta

    def __call__(self,
                 y_true: plaidml.tile.Value,
                 y_pred: plaidml.tile.Value) -> plaidml.tile.Value:
        """ Call the Generalized Loss Function

        Parameters
        ----------
        y_true: :class:`plaidml.tile.Value`
            The ground truth value
        y_pred: :class:`plaidml.tile.Value`
            The predicted value

        Returns
        -------
        :class:`plaidml.tile.Value`
            The loss value from the results of function(y_pred - y_true)
        """
        diff = y_pred - y_true
        second = (K.pow(K.pow(diff/self._beta, 2.) / K.abs(2. - self._alpha) + 1.,
                        (self._alpha / 2.)) - 1.)
        loss = (K.abs(2. - self._alpha)/self._alpha) * second
        loss = K.mean(loss, axis=-1) * self._beta
        return loss


class GradientLoss():  # pylint:disable=too-few-public-methods
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
    def __init__(self):
        self.generalized_loss = GeneralizedLoss(alpha=1.9999)

    def __call__(self,
                 y_true: plaidml.tile.Value,
                 y_pred: plaidml.tile.Value) -> plaidml.tile.Value:
        """ Call the gradient loss function.

        Parameters
        ----------
        y_true: :class:`plaidml.tile.Value`
            The ground truth value
        y_pred: tensor or variable
            :class:`plaidml.tile.Value`

        Returns
        -------
        :class:`plaidml.tile.Value`
            The loss value
        """
        tv_weight = 1.0
        tv2_weight = 1.0
        loss = 0.0
        loss += tv_weight * (self.generalized_loss(self._diff_x(y_true), self._diff_x(y_pred)) +
                             self.generalized_loss(self._diff_y(y_true), self._diff_y(y_pred)))
        loss += tv2_weight * (self.generalized_loss(self._diff_xx(y_true), self._diff_xx(y_pred)) +
                              self.generalized_loss(self._diff_yy(y_true), self._diff_yy(y_pred)) +
                              self.generalized_loss(self._diff_xy(y_true), self._diff_xy(y_pred))
                              * 2.)
        loss = loss / (tv_weight + tv2_weight)
        # TODO simplify to use MSE instead
        return loss

    @classmethod
    def _diff_x(cls, img):
        """ X Difference """
        x_left = img[:, :, 1:2, :] - img[:, :, 0:1, :]
        x_inner = img[:, :, 2:, :] - img[:, :, :-2, :]
        x_right = img[:, :, -1:, :] - img[:, :, -2:-1, :]
        x_out = K.concatenate([x_left, x_inner, x_right], axis=2)
        return x_out * 0.5

    @classmethod
    def _diff_y(cls, img):
        """ Y Difference """
        y_top = img[:, 1:2, :, :] - img[:, 0:1, :, :]
        y_inner = img[:, 2:, :, :] - img[:, :-2, :, :]
        y_bot = img[:, -1:, :, :] - img[:, -2:-1, :, :]
        y_out = K.concatenate([y_top, y_inner, y_bot], axis=1)
        return y_out * 0.5

    @classmethod
    def _diff_xx(cls, img):
        """ X-X Difference """
        x_left = img[:, :, 1:2, :] + img[:, :, 0:1, :]
        x_inner = img[:, :, 2:, :] + img[:, :, :-2, :]
        x_right = img[:, :, -1:, :] + img[:, :, -2:-1, :]
        x_out = K.concatenate([x_left, x_inner, x_right], axis=2)
        return x_out - 2.0 * img

    @classmethod
    def _diff_yy(cls, img):
        """ Y-Y Difference """
        y_top = img[:, 1:2, :, :] + img[:, 0:1, :, :]
        y_inner = img[:, 2:, :, :] + img[:, :-2, :, :]
        y_bot = img[:, -1:, :, :] + img[:, -2:-1, :, :]
        y_out = K.concatenate([y_top, y_inner, y_bot], axis=1)
        return y_out - 2.0 * img

    @classmethod
    def _diff_xy(cls, img: plaidml.tile.Value) -> plaidml.tile.Value:
        """ X-Y Difference """
        # xout1
        # Left
        top = img[:, 1:2, 1:2, :] + img[:, 0:1, 0:1, :]
        inner = img[:, 2:, 1:2, :] + img[:, :-2, 0:1, :]
        bottom = img[:, -1:, 1:2, :] + img[:, -2:-1, 0:1, :]
        xy_left = K.concatenate([top, inner, bottom], axis=1)
        # Mid
        top = img[:, 1:2, 2:, :] + img[:, 0:1, :-2, :]
        mid = img[:, 2:, 2:, :] + img[:, :-2, :-2, :]
        bottom = img[:, -1:, 2:, :] + img[:, -2:-1, :-2, :]
        xy_mid = K.concatenate([top, mid, bottom], axis=1)
        # Right
        top = img[:, 1:2, -1:, :] + img[:, 0:1, -2:-1, :]
        inner = img[:, 2:, -1:, :] + img[:, :-2, -2:-1, :]
        bottom = img[:, -1:, -1:, :] + img[:, -2:-1, -2:-1, :]
        xy_right = K.concatenate([top, inner, bottom], axis=1)

        # Xout2
        # Left
        top = img[:, 0:1, 1:2, :] + img[:, 1:2, 0:1, :]
        inner = img[:, :-2, 1:2, :] + img[:, 2:, 0:1, :]
        bottom = img[:, -2:-1, 1:2, :] + img[:, -1:, 0:1, :]
        xy_left = K.concatenate([top, inner, bottom], axis=1)
        # Mid
        top = img[:, 0:1, 2:, :] + img[:, 1:2, :-2, :]
        mid = img[:, :-2, 2:, :] + img[:, 2:, :-2, :]
        bottom = img[:, -2:-1, 2:, :] + img[:, -1:, :-2, :]
        xy_mid = K.concatenate([top, mid, bottom], axis=1)
        # Right
        top = img[:, 0:1, -1:, :] + img[:, 1:2, -2:-1, :]
        inner = img[:, :-2, -1:, :] + img[:, 2:, -2:-1, :]
        bottom = img[:, -2:-1, -1:, :] + img[:, -1:, -2:-1, :]
        xy_right = K.concatenate([top, inner, bottom], axis=1)

        xy_out1 = K.concatenate([xy_left, xy_mid, xy_right], axis=2)
        xy_out2 = K.concatenate([xy_left, xy_mid, xy_right], axis=2)
        return (xy_out1 - xy_out2) * 0.25


class LaplacianPyramidLoss():  # pylint:disable=too-few-public-methods
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
        self._max_levels = max_levels
        self._weights = K.constant([np.power(2., -2 * idx) for idx in range(max_levels + 1)])
        self._gaussian_kernel = self._get_gaussian_kernel(gaussian_size, gaussian_sigma)
        self._shape: Tuple[int, ...] = ()

    @classmethod
    def _get_gaussian_kernel(cls, size: int, sigma: float) -> plaidml.tile.Value:
        """ Obtain the base gaussian kernel for the Laplacian Pyramid.

        Parameters
        ----------
        size: int, Optional
            The size of the gaussian kernel
        sigma: float
            The gaussian sigma

        Returns
        -------
        :class:`plaidml.tile.Value`
            The base single channel Gaussian kernel
        """
        assert size % 2 == 1, ("kernel size must be uneven")
        x_1 = np.linspace(- (size // 2), size // 2, size, dtype="float32")
        x_1 /= np.sqrt(2)*sigma
        x_2 = x_1 ** 2
        kernel = np.exp(- x_2[:, None] - x_2[None, :])
        kernel /= kernel.sum()
        kernel = np.reshape(kernel, (size, size, 1, 1))
        return K.constant(kernel)

    def _conv_gaussian(self, inputs: plaidml.tile.Value) -> plaidml.tile.Value:
        """ Perform Gaussian convolution on a batch of images.

        Parameters
        ----------
        inputs: :class:`plaidml.tile.Value`
            The input batch of images to perform Gaussian convolution on.

        Returns
        -------
        :class:`plaidml.tile.Value`
            The convolved images
        """
        channels = self._shape[-1]
        gauss = K.tile(self._gaussian_kernel, (1, 1, 1, channels))

        # PlaidML doesn't implement replication padding like pytorch. This is an inefficient way to
        # implement it for a square guassian kernel
        size = K.int_shape(self._gaussian_kernel)[1] // 2
        padded_inputs = inputs
        for _ in range(size):
            padded_inputs = pad(padded_inputs,  # noqa,pylint:disable=no-value-for-parameter,unexpected-keyword-arg
                                ([0, 0], [1, 1], [1, 1], [0, 0]),
                                mode="REFLECT")

        retval = K.conv2d(padded_inputs, gauss, strides=(1, 1), padding="valid")
        return retval

    def _get_laplacian_pyramid(self, inputs: plaidml.tile.Value) -> List[plaidml.tile.Value]:
        """ Obtain the Laplacian Pyramid.

        Parameters
        ----------
        inputs: :class:`plaidml.tile.Value`
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
            current = K.pool2d(gauss, (2, 2), strides=(2, 2), padding="valid", pool_mode="avg")
        pyramid.append(current)
        return pyramid

    def __call__(self,
                 y_true: plaidml.tile.Value,
                 y_pred: plaidml.tile.Value) -> plaidml.tile.Value:
        """ Calculate the Laplacian Pyramid Loss.

        Parameters
        ----------
        y_true: :class:`plaidml.tile.Value`
            The ground truth value
        y_pred: :class:`plaidml.tile.Value`
            The predicted value

        Returns
        -------
        :class: `plaidml.tile.Value`
            The loss value
        """
        if not self._shape:
            self._shape = K.int_shape(y_pred)
        pyramid_true = self._get_laplacian_pyramid(y_true)
        pyramid_pred = self._get_laplacian_pyramid(y_pred)

        losses = K.stack([K.sum(K.abs(ppred - ptrue)) / K.cast(K.prod(K.shape(ptrue)), "float32")
                          for ptrue, ppred in zip(pyramid_true, pyramid_pred)])
        loss = K.sum(losses * self._weights)
        return loss


class LInfNorm():  # pylint:disable=too-few-public-methods
    """ Calculate the L-inf norm as a loss function. """

    def __call__(self,
                 y_true: plaidml.tile.Value,
                 y_pred: plaidml.tile.Value) -> plaidml.tile.Value:
        """ Call the L-inf norm loss function.

        Parameters
        ----------
        y_true: :class:`plaidml.tile.Value`
            The ground truth value
        y_pred: :class:`plaidml.tile.Value`
            The predicted value

        Returns
        -------
        :class:`plaidml.tile.Value`
            The loss value
        """
        diff = K.abs(y_true - y_pred)
        max_loss = K.max(diff, axis=(1, 2), keepdims=True)
        loss = K.mean(max_loss, axis=-1)
        return loss


class LogCosh():  # pylint:disable=too-few-public-methods
    """Logarithm of the hyperbolic cosine of the prediction error.

    `log(cosh(x))` is approximately equal to `(x ** 2) / 2` for small `x` and to `abs(x) - log(2)`
    for large `x`. This means that 'logcosh' works mostly like the mean squared error, but will not
    be so strongly affected by the occasional wildly incorrect prediction.
    """
    def __call__(self,
                 y_true: plaidml.tile.Value,
                 y_pred: plaidml.tile.Value) -> plaidml.tile.Value:
        """ Call the LogCosh loss function.
        Parameters
        ----------
        y_true: :class:`plaidml.tile.Value`
            The ground truth value
        y_pred: :class:`plaidml.tile.Value`
            The predicted value

        Returns
        -------
        :class:`plaidml.tile.Value`
            The loss value
        """
        diff = y_pred - y_true
        loss = diff + K.softplus(-2. * diff) - K.log(K.constant(2., dtype="float32"))
        return K.mean(loss, axis=-1)


class LossWrapper():  # pylint:disable=too-few-public-methods
    """ A wrapper class for multiple keras losses to enable multiple weighted loss functions on a
    single output and masking.
    """
    def __init__(self) -> None:
        self.__name__ = "LossWrapper"
        logger.debug("Initializing: %s", self.__class__.__name__)
        self._loss_functions: List[Callable] = []
        self._loss_weights: List[float] = []
        self._mask_channels: List[int] = []
        logger.debug("Initialized: %s", self.__class__.__name__)

    def add_loss(self,
                 function,
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
        self._loss_functions.append(function)
        self._loss_weights.append(weight)
        self._mask_channels.append(mask_channel)

    def __call__(self,
                 y_true: plaidml.tile.Value,
                 y_pred: plaidml.tile.Value) -> plaidml.tile.Value:
        """ Call the sub loss functions for the loss wrapper.

        Weights are returned as the weighted sum of the chosen losses.

        Parameters
        ----------
        y_true: :class:`plaidml.tile.Value`
            The ground truth value
        y_pred: :class:`plaidml.tile.Value`
            The predicted value

        Returns
        -------
        :class:`plaidml.tile.Value`
            The final loss value
        """
        loss = 0.0
        for func, weight, mask_channel in zip(self._loss_functions,
                                              self._loss_weights,
                                              self._mask_channels):
            logger.debug("Processing loss function: (func: %s, weight: %s, mask_channel: %s)",
                         func, weight, mask_channel)
            n_true, n_pred = self._apply_mask(y_true, y_pred, mask_channel)
            # Some loss functions requires that y_pred be of a known shape, so specifically
            # reshape the tensor.
            n_pred = K.reshape(n_pred, K.int_shape(y_pred))
            this_loss = func(n_true, n_pred)
            loss_dims = K.ndim(this_loss)
            loss += (K.mean(this_loss, axis=list(range(1, loss_dims))) * weight)
        return loss

    @classmethod
    def _apply_mask(cls,
                    y_true: plaidml.tile.Value,
                    y_pred: plaidml.tile.Value,
                    mask_channel: int,
                    mask_prop: float = 1.0) -> Tuple[plaidml.tile.Value, plaidml.tile.Value]:
        """ Apply the mask to the input y_true and y_pred. If a mask is not required then
        return the unmasked inputs.

        Parameters
        ----------
        y_true: :class:`plaidml.tile.Value`
            The ground truth value
        y_pred: :class:`plaidml.tile.Value`
            The predicted value
        mask_channel: int
            The channel within y_true that the required mask resides in
        mask_prop: float, optional
            The amount of mask propagation. Default: `1.0`

        Returns
        -------
        tuple
            (n_true, n_pred): The ground truth and predicted value tensors with the mask applied
        """
        if mask_channel == -1:
            logger.debug("No mask to apply")
            return y_true[..., :3], y_pred[..., :3]

        logger.debug("Applying mask from channel %s", mask_channel)

        mask = K.tile(K.expand_dims(y_true[..., mask_channel], axis=-1), (1, 1, 1, 3))
        mask_as_k_inv_prop = 1 - mask_prop
        mask = (mask * mask_prop) + mask_as_k_inv_prop

        m_true = y_true[..., :3] * mask
        m_pred = y_pred[..., :3] * mask

        return m_true, m_pred
