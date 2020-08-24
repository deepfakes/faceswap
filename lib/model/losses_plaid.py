#!/usr/bin/env python3
""" Custom Loss Functions for faceswap.py """

from __future__ import absolute_import

import logging

import numpy as np
import tensorflow as tf

from keras import backend as K
from plaidml.op import extract_image_patches
from lib.plaidml_utils import pad
from lib.utils import FaceswapError

logger = logging.getLogger(__name__)  # pylint:disable=invalid-name


class DSSIMObjective():
    """ DSSIM Loss Function

    Difference of Structural Similarity (DSSIM loss function). Clipped between 0 and 0.5

    Parameters
    ----------
    k_1: float, optional
        Parameter of the SSIM. Default: `0.01`
    k_2: float, optional
        Parameter of the SSIM. Default: `0.03`
    kernel_size: int, optional
        Size of the sliding window Default: `3`
    max_value: float, optional
        Max value of the output. Default: `1.0`

    Notes
    ------
    You should add a regularization term like a l2 loss in addition to this one.

    References
    ----------
    https://github.com/keras-team/keras-contrib/blob/master/keras_contrib/losses/dssim.py

    MIT License

    Copyright (c) 2017 Fariz Rahman

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
    """
    def __init__(self, k_1=0.01, k_2=0.03, kernel_size=3, max_value=1.0):
        self.__name__ = 'DSSIMObjective'
        self.kernel_size = kernel_size
        self.k_1 = k_1
        self.k_2 = k_2
        self.max_value = max_value
        self.c_1 = (self.k_1 * self.max_value) ** 2
        self.c_2 = (self.k_2 * self.max_value) ** 2
        self.dim_ordering = K.image_data_format()

    @staticmethod
    def __int_shape(input_tensor):
        """ Returns the shape of tensor or variable as a tuple of int or None entries.

        Parameters
        ----------
        input_tensor: tensor or variable
            The input to return the shape for

        Returns
        -------
        tuple
            A tuple of integers (or None entries)
        """
        return K.int_shape(input_tensor)

    def __call__(self, y_true, y_pred):
        """ Call the DSSIM Loss Function.

        Parameters
        ----------
        y_true: tensor or variable
            The ground truth value
        y_pred: tensor or variable
            The predicted value

        Returns
        -------
        tensor
            The DSSIM Loss value

        Notes
        -----
        There are additional parameters for this function. some of the 'modes' for edge behavior
        do not yet have a gradient definition in the Theano tree and cannot be used for learning
        """

        kernel = [self.kernel_size, self.kernel_size]
        y_true = K.reshape(y_true, [-1] + list(self.__int_shape(y_pred)[1:]))
        y_pred = K.reshape(y_pred, [-1] + list(self.__int_shape(y_pred)[1:]))
        patches_pred = self.extract_image_patches(y_pred,
                                                  kernel,
                                                  kernel,
                                                  'valid',
                                                  self.dim_ordering)
        patches_true = self.extract_image_patches(y_true,
                                                  kernel,
                                                  kernel,
                                                  'valid',
                                                  self.dim_ordering)

        # Get mean
        u_true = K.mean(patches_true, axis=-1)
        u_pred = K.mean(patches_pred, axis=-1)
        # Get variance
        var_true = K.var(patches_true, axis=-1)
        var_pred = K.var(patches_pred, axis=-1)
        # Get standard deviation
        covar_true_pred = K.mean(
            patches_true * patches_pred, axis=-1) - u_true * u_pred

        ssim = (2 * u_true * u_pred + self.c_1) * (
            2 * covar_true_pred + self.c_2)
        denom = (K.square(u_true) + K.square(u_pred) + self.c_1) * (
            var_pred + var_true + self.c_2)
        ssim /= denom  # no need for clipping, c_1 + c_2 make the denorm non-zero
        return K.mean((1.0 - ssim) / 2.0)

    @staticmethod
    def _preprocess_padding(padding):
        """Convert keras padding to tensorflow padding.

        Parameters
        ----------
        padding: string,
            `"same"` or `"valid"`.

        Returns
        -------
        str
            `"SAME"` or `"VALID"`.

        Raises
        ------
        ValueError
            If `padding` is invalid.
        """
        if padding == 'same':
            padding = 'SAME'
        elif padding == 'valid':
            padding = 'VALID'
        else:
            raise ValueError('Invalid padding:', padding)
        return padding

    def extract_image_patches(self, input_tensor, k_sizes, s_sizes,
                              padding='same', data_format='channels_last'):
        """ Extract the patches from an image.

        Parameters
        ----------
        input_tensor: tensor
            The input image
        k_sizes: tuple
            2-d tuple with the kernel size
        s_sizes: tuple
            2-d tuple with the strides size
        padding: str, optional
            `"same"` or `"valid"`. Default: `"same"`
        data_format: str, optional.
            `"channels_last"` or `"channels_first"`. Default: `"channels_last"`

        Returns
        -------
        The (k_w, k_h) patches extracted
            Tensorflow ==> (batch_size, w, h, k_w, k_h, c)
            Theano ==> (batch_size, w, h, c, k_w, k_h)
        """
        kernel = [1, k_sizes[0], k_sizes[1], 1]
        strides = [1, s_sizes[0], s_sizes[1], 1]
        padding = self._preprocess_padding(padding)
        if data_format == 'channels_first':
            input_tensor = K.permute_dimensions(input_tensor, (0, 2, 3, 1))
        patches = extract_image_patches(input_tensor, kernel, strides, [1, 1, 1, 1], padding)
        return patches


class PenalizedLoss():  # pylint:disable=too-few-public-methods
    """ Penalized Loss function.

    Applies the given loss function just to the masked area of the image.

    Parameters
    ----------
    loss_func: function
        The actual loss function to use
    mask_prop: float, optional
        The amount of mask propagation. Default: `1.0`
    """
    def __init__(self, loss_func, mask_prop=1.0):
        self._loss_func = loss_func
        self._mask_prop = mask_prop

    def __call__(self, y_true, y_pred):
        """ Apply the loss function to the masked area of the image.

        Parameters
        ----------
        y_true: tensor or variable
            The ground truth value. This should contain the mask in the 4th channel that will be
            split off for penalizing.
        y_pred: tensor or variable
            The predicted value

        Returns
        -------
        tensor
            The Loss value
        """
        mask = self._prepare_mask(K.expand_dims(y_true[..., -1], axis=-1))
        y_true = y_true[..., :-1]
        n_true = y_true * mask
        n_pred = y_pred * mask
        if isinstance(self._loss_func, DSSIMObjective):
            # Extract Image Patches in SSIM requires that y_pred be of a known shape, so
            # specifically reshape the tensor.
            n_pred = K.reshape(n_pred, K.int_shape(y_pred))
        return self._loss_func(n_true, n_pred)

    def _prepare_mask(self, mask):
        """ Prepare the masks for calculating loss

        Parameters
        ----------
        mask: :class:`numpy.ndarray`
            The masks for the current batch

        Returns
        -------
        tensor
            The prepared mask for applying to loss
        """
        mask_as_k_inv_prop = 1 - self._mask_prop
        mask = (mask * self._mask_prop) + mask_as_k_inv_prop
        return mask


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
    def __init__(self, alpha=1.0, beta=1.0/255.0):
        self.alpha = alpha
        self.beta = beta

    def __call__(self, y_true, y_pred):
        """ Call the Generalized Loss Function

        Parameters
        ----------
        y_true: tensor or variable
            The ground truth value
        y_pred: tensor or variable
            The predicted value

        Returns
        -------
        tensor
            The loss value from the results of function(y_pred - y_true)
        """
        diff = y_pred - y_true
        second = (K.pow(K.pow(diff/self.beta, 2.) / K.abs(2. - self.alpha) + 1.,
                        (self.alpha / 2.)) - 1.)
        loss = (K.abs(2. - self.alpha)/self.alpha) * second
        loss = K.mean(loss, axis=-1) * self.beta
        return loss


class LInfNorm():  # pylint:disable=too-few-public-methods
    """ Calculate the L-inf norm as a loss function. """

    def __call__(self, y_true, y_pred):
        """ Call the L-inf norm loss function.

        Parameters
        ----------
        y_true: tensor or variable
            The ground truth value
        y_pred: tensor or variable
            The predicted value

        Returns
        -------
        tensor
            The loss value
        """
        diff = K.abs(y_true - y_pred)
        max_loss = K.max(diff, axis=(1, 2), keepdims=True)
        loss = K.mean(max_loss, axis=-1)
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

    def __call__(self, y_true, y_pred):
        """ Call the gradient loss function.

        Parameters
        ----------
        y_true: tensor or variable
            The ground truth value
        y_pred: tensor or variable
            The predicted value

        Returns
        -------
        tensor
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
    def _diff_xy(cls, img):
        """ X-Y Difference """
        # xout1
        top_left = img[:, 1:2, 1:2, :] + img[:, 0:1, 0:1, :]
        inner_left = img[:, 2:, 1:2, :] + img[:, :-2, 0:1, :]
        bot_left = img[:, -1:, 1:2, :] + img[:, -2:-1, 0:1, :]
        xy_left = K.concatenate([top_left, inner_left, bot_left], axis=1)

        top_mid = img[:, 1:2, 2:, :] + img[:, 0:1, :-2, :]
        mid_mid = img[:, 2:, 2:, :] + img[:, :-2, :-2, :]
        bot_mid = img[:, -1:, 2:, :] + img[:, -2:-1, :-2, :]
        xy_mid = K.concatenate([top_mid, mid_mid, bot_mid], axis=1)

        top_right = img[:, 1:2, -1:, :] + img[:, 0:1, -2:-1, :]
        inner_right = img[:, 2:, -1:, :] + img[:, :-2, -2:-1, :]
        bot_right = img[:, -1:, -1:, :] + img[:, -2:-1, -2:-1, :]
        xy_right = K.concatenate([top_right, inner_right, bot_right], axis=1)

        # Xout2
        top_left = img[:, 0:1, 1:2, :] + img[:, 1:2, 0:1, :]
        inner_left = img[:, :-2, 1:2, :] + img[:, 2:, 0:1, :]
        bot_left = img[:, -2:-1, 1:2, :] + img[:, -1:, 0:1, :]
        xy_left = K.concatenate([top_left, inner_left, bot_left], axis=1)

        top_mid = img[:, 0:1, 2:, :] + img[:, 1:2, :-2, :]
        mid_mid = img[:, :-2, 2:, :] + img[:, 2:, :-2, :]
        bot_mid = img[:, -2:-1, 2:, :] + img[:, -1:, :-2, :]
        xy_mid = K.concatenate([top_mid, mid_mid, bot_mid], axis=1)

        top_right = img[:, 0:1, -1:, :] + img[:, 1:2, -2:-1, :]
        inner_right = img[:, :-2, -1:, :] + img[:, 2:, -2:-1, :]
        bot_right = img[:, -2:-1, -1:, :] + img[:, -1:, -2:-1, :]
        xy_right = K.concatenate([top_right, inner_right, bot_right], axis=1)

        xy_out1 = K.concatenate([xy_left, xy_mid, xy_right], axis=2)
        xy_out2 = K.concatenate([xy_left, xy_mid, xy_right], axis=2)
        return (xy_out1 - xy_out2) * 0.25


class GMSDLoss():  # pylint:disable=too-few-public-methods
    """ Gradient Magnitude Similarity Deviation Loss.

    Improved image quality metric over MS-SSIM with easier calculations

    References
    ----------
    http://www4.comp.polyu.edu.hk/~cslzhang/IQA/GMSD/GMSD.htm
    https://arxiv.org/ftp/arxiv/papers/1308/1308.3052.pdf
    """

    def __call__(self, y_true, y_pred):
        """ Return the Gradient Magnitude Similarity Deviation Loss.

        Parameters
        ----------
        y_true: tensor or variable
            The ground truth value
        y_pred: tensor or variable
            The predicted value

        Returns
        -------
        tensor
            The loss value
        """
        raise FaceswapError("GMSD Loss is not currently compatible with PlaidML. Please select a "
                            "different Loss method.")

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
    def _scharr_edges(cls, image, magnitude):
        """ Returns a tensor holding modified Scharr edge maps.

        Parameters
        ----------
        image: tensor
            Image tensor with shape [batch_size, h, w, d] and type float32. The image(s) must be
            2x2 or larger.
        magnitude: bool
            Boolean to determine if the edge magnitude or edge direction is returned

        Returns
        -------
        tensor
            Tensor holding edge maps for each channel. Returns a tensor with shape `[batch_size, h,
            w, d, 2]` where the last two dimensions hold `[[dy[0], dx[0]], [dy[1], dx[1]], ...,
            [dy[d-1], dx[d-1]]]` calculated using the Scharr filter.
        """

        # Define vertical and horizontal Scharr filters.
        # TODO PlaidML: AttributeError: 'Value' object has no attribute 'get_shape'
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
        padded = pad(image, pad_sizes, mode='REFLECT')
        output = K.depthwise_conv2d(padded, kernels)

        if not magnitude:  # direction of edges
            # Reshape to [batch_size, h, w, d, num_kernels].
            shape = K.concatenate([image_shape, num_kernels], axis=0)
            output = K.reshape(output, shape=shape)
            output.set_shape(static_image_shape.concatenate(num_kernels))
            output = tf.atan(K.squeeze(output[:, :, :, :, 0] / output[:, :, :, :, 1], axis=None))
        # magnitude of edges -- unified x & y edges don't work well with Neural Networks
        return output


class LossWrapper():  # pylint:disable=too-few-public-methods
    """ A wrapper class for multiple keras losses to enable multiple weighted loss functions on a
    single output.

    Parameters
    ----------
    loss_functions: list
        A list of either a tuple of (:class:`keras.losses.Loss`, scalar weight) or just a
        :class:`keras.losses.Loss` function. If just the loss function is passed, then the weight
        is assumed to be 1.0 """
    def __init__(self, loss_functions):
        logger.debug("Initializing: %s: (loss_functions: %s)",
                     self.__class__.__name__, loss_functions)
        self._loss_functions = []
        self._loss_weights = []
        self._compile_losses(loss_functions)
        logger.debug("Initialized: %s", self.__class__.__name__)

    def _compile_losses(self, loss_functions):
        """ Splits the given loss_functions into the corresponding :attr:`_loss_functions' and
        :attr:`_loss_weights' lists.

        Loss functions are compiled into :class:`keras.compile_utils.LossesContainer` objects

        Parameters
        ----------
        loss_functions: list
            A list of either a tuple of (:class:`keras.losses.Loss`, scalar weight) or just a
            :class:`keras.losses.Loss` function. If just the loss function is passed, then the
            weight is assumed to be 1.0 """
        for loss_func in loss_functions:
            if isinstance(loss_func, tuple):
                assert len(loss_func) == 2, "Tuple loss functions should contain 2 items"
                assert isinstance(loss_func[1], float), "weight should be a float"
                func, weight = loss_func
            else:
                func = loss_func
                weight = 1.0
            self._loss_functions.append(func)
            self._loss_weights.append(weight)
        logger.debug("Compiled losses: (functions: %s, weights: %s",
                     self._loss_functions, self._loss_weights)

    def __call__(self, y_true, y_pred):
        """ Call the sub loss functions for the loss wrapper.

        Weights are returned as the weighted sum of the chosen losses.

        Parameters
        ----------
        y_true: tensor or variable
            The ground truth value
        y_pred: tensor or variable
            The predicted value

        Returns
        -------
        tensor
            The final loss value
        """
        loss = 0.0
        for func, weight in zip(self._loss_functions, self._loss_weights):
            loss += (K.mean(func(y_true, y_pred)) * weight)
        return loss
