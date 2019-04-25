#!/usr/bin/env python3
""" Custom Loss Functions for faceswap.py
    Losses from:
        keras.contrib
        dfaker: https://github.com/dfaker/df
        shoanlu GAN: https://github.com/shaoanlu/faceswap-GAN"""

from __future__ import absolute_import

import keras.backend as K
from keras.layers import Lambda, concatenate
import numpy as np
import tensorflow as tf
from tensorflow.contrib.distributions import Beta

from .normalization import InstanceNormalization
if K.backend() == "plaidml.keras.backend":
    from lib.plaidml_utils import extract_image_patches
else:
    from tensorflow import extract_image_patches  # pylint: disable=ungrouped-imports


class DSSIMObjective():
    """ DSSIM Loss Function

    Code copy and pasted, with minor ammendments from:
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
    SOFTWARE. """
    # pylint: disable=C0103
    def __init__(self, k1=0.01, k2=0.03, kernel_size=3, max_value=1.0):
        """
        Difference of Structural Similarity (DSSIM loss function). Clipped
        between 0 and 0.5
        Note : You should add a regularization term like a l2 loss in
               addition to this one.
        Note : In theano, the `kernel_size` must be a factor of the output
               size. So 3 could not be the `kernel_size` for an output of 32.
        # Arguments
            k1: Parameter of the SSIM (default 0.01)
            k2: Parameter of the SSIM (default 0.03)
            kernel_size: Size of the sliding window (default 3)
            max_value: Max value of the output (default 1.0)
        """
        self.__name__ = 'DSSIMObjective'
        self.kernel_size = kernel_size
        self.k1 = k1
        self.k2 = k2
        self.max_value = max_value
        self.c_1 = (self.k1 * self.max_value) ** 2
        self.c_2 = (self.k2 * self.max_value) ** 2
        self.dim_ordering = K.image_data_format()
        self.backend = K.backend()

    @staticmethod
    def __int_shape(x):
        return K.int_shape(x)

    def __call__(self, y_true, y_pred):
        # There are additional parameters for this function
        # Note: some of the 'modes' for edge behavior do not yet have a
        # gradient definition in the Theano tree and cannot be used for
        # learning

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
        # Get std dev
        covar_true_pred = K.mean(
            patches_true * patches_pred, axis=-1) - u_true * u_pred

        ssim = (2 * u_true * u_pred + self.c_1) * (
            2 * covar_true_pred + self.c_2)
        denom = (K.square(u_true) + K.square(u_pred) + self.c_1) * (
            var_pred + var_true + self.c_2)
        ssim /= denom  # no need for clipping, c_1 + c_2 make the denom non-zero
        return K.mean((1.0 - ssim) / 2.0)

    @staticmethod
    def _preprocess_padding(padding):
        """Convert keras' padding to tensorflow's padding.
        # Arguments
            padding: string, `"same"` or `"valid"`.
        # Returns
            a string, `"SAME"` or `"VALID"`.
        # Raises
            ValueError: if `padding` is invalid.
        """
        if padding == 'same':
            padding = 'SAME'
        elif padding == 'valid':
            padding = 'VALID'
        else:
            raise ValueError('Invalid padding:', padding)
        return padding

    def extract_image_patches(self, x, ksizes, ssizes, padding='same',
                              data_format='channels_last'):
        """
        Extract the patches from an image
        # Parameters
            x : The input image
            ksizes : 2-d tuple with the kernel size
            ssizes : 2-d tuple with the strides size
            padding : 'same' or 'valid'
            data_format : 'channels_last' or 'channels_first'
        # Returns
            The (k_w, k_h) patches extracted
            TF ==> (batch_size, w, h, k_w, k_h, c)
            TH ==> (batch_size, w, h, c, k_w, k_h)
        """
        kernel = [1, ksizes[0], ksizes[1], 1]
        strides = [1, ssizes[0], ssizes[1], 1]
        padding = self._preprocess_padding(padding)
        if data_format == 'channels_first':
            x = K.permute_dimensions(x, (0, 2, 3, 1))
        patches = extract_image_patches(x, kernel, strides, [1, 1, 1, 1], padding)
        return patches


# <<< START: from Dfaker >>> #
def PenalizedLoss(mask, loss_func, mask_prop=1.0):  # pylint: disable=invalid-name
    """ Plaidml + tf Penalized loss function """
    mask_as_k_inv_prop = 1 - mask_prop
    mask = (mask * mask_prop) + mask_as_k_inv_prop

    def inner_loss(y_true, y_pred):
        # Workaround until https://github.com/plaidml/plaidml/pull/284 is accepted
        if K.backend() == "plaidml.keras.backend":
            y_true = K.reshape(y_true, y_pred.shape.dims)
        n_true = K.concatenate([y_true[:, :, :, i:i+1] * mask for i in range(3)], axis=-1)
        n_pred = K.concatenate([y_pred[:, :, :, i:i+1] * mask for i in range(3)], axis=-1)
        return loss_func(n_true, n_pred)
    return inner_loss
# <<< END: from Dfaker >>> #


# <<< START: from Shoanlu GAN >>> #
def first_order(var_x, axis=1):
    """ First Order Function from Shoanlu GAN """
    img_nrows = var_x.shape[1]
    img_ncols = var_x.shape[2]
    if axis == 1:
        return K.abs(var_x[:, :img_nrows - 1, :img_ncols - 1, :] - var_x[:, 1:, :img_ncols - 1, :])
    if axis == 2:
        return K.abs(var_x[:, :img_nrows - 1, :img_ncols - 1, :] - var_x[:, :img_nrows - 1, 1:, :])
    return None


def calc_loss(pred, target, loss='l2'):
    """ Calculate Loss from Shoanlu GAN """
    if loss.lower() == "l2":
        return K.mean(K.square(pred - target))
    if loss.lower() == "l1":
        return K.mean(K.abs(pred - target))
    if loss.lower() == "cross_entropy":
        return -K.mean(K.log(pred + K.epsilon()) * target +
                       K.log(1 - pred + K.epsilon()) * (1 - target))
    raise ValueError('Recieve an unknown loss type: {}.'.format(loss))


def cyclic_loss(net_g1, net_g2, real1):
    """ Cyclic Loss Function from Shoanlu GAN """
    fake2 = net_g2(real1)[-1]  # fake2 ABGR
    fake2 = Lambda(lambda x: x[:, :, :, 1:])(fake2)  # fake2 BGR
    cyclic1 = net_g1(fake2)[-1]  # cyclic1 ABGR
    cyclic1 = Lambda(lambda x: x[:, :, :, 1:])(cyclic1)  # cyclic1 BGR
    loss = calc_loss(cyclic1, real1, loss='l1')
    return loss


def adversarial_loss(net_d, real, fake_abgr, distorted, gan_training="mixup_LSGAN", **weights):
    """ Adversarial Loss Function from Shoanlu GAN """
    alpha = Lambda(lambda x: x[:, :, :, :1])(fake_abgr)
    fake_bgr = Lambda(lambda x: x[:, :, :, 1:])(fake_abgr)
    fake = alpha * fake_bgr + (1-alpha) * distorted

    if gan_training == "mixup_LSGAN":
        dist = Beta(0.2, 0.2)
        lam = dist.sample()
        mixup = lam * concatenate([real, distorted]) + (1 - lam) * concatenate([fake, distorted])
        pred_fake = net_d(concatenate([fake, distorted]))
        pred_mixup = net_d(mixup)
        loss_d = calc_loss(pred_mixup, lam * K.ones_like(pred_mixup), "l2")
        loss_g = weights['w_D'] * calc_loss(pred_fake, K.ones_like(pred_fake), "l2")
        mixup2 = lam * concatenate([real,
                                    distorted]) + (1 - lam) * concatenate([fake_bgr,
                                                                           distorted])
        pred_fake_bgr = net_d(concatenate([fake_bgr, distorted]))
        pred_mixup2 = net_d(mixup2)
        loss_d += calc_loss(pred_mixup2, lam * K.ones_like(pred_mixup2), "l2")
        loss_g += weights['w_D'] * calc_loss(pred_fake_bgr, K.ones_like(pred_fake_bgr), "l2")
    elif gan_training == "relativistic_avg_LSGAN":
        real_pred = net_d(concatenate([real, distorted]))
        fake_pred = net_d(concatenate([fake, distorted]))
        loss_d = K.mean(K.square(real_pred - K.ones_like(fake_pred)))/2
        loss_d += K.mean(K.square(fake_pred - K.zeros_like(fake_pred)))/2
        loss_g = weights['w_D'] * K.mean(K.square(fake_pred - K.ones_like(fake_pred)))

        fake_pred2 = net_d(concatenate([fake_bgr, distorted]))
        loss_d += K.mean(K.square(real_pred - K.mean(fake_pred2, axis=0) -
                                  K.ones_like(fake_pred2)))/2
        loss_d += K.mean(K.square(fake_pred2 - K.mean(real_pred, axis=0) -
                                  K.zeros_like(fake_pred2)))/2
        loss_g += weights['w_D'] * K.mean(K.square(real_pred - K.mean(fake_pred2, axis=0) -
                                                   K.zeros_like(fake_pred2)))/2
        loss_g += weights['w_D'] * K.mean(K.square(fake_pred2 - K.mean(real_pred, axis=0) -
                                                   K.ones_like(fake_pred2)))/2
    else:
        raise ValueError("Receive an unknown GAN training method: {gan_training}")
    return loss_d, loss_g


def reconstruction_loss(real, fake_abgr, mask_eyes, model_outputs, **weights):
    """ Reconstruction Loss Function from Shoanlu GAN """
    alpha = Lambda(lambda x: x[:, :, :, :1])(fake_abgr)
    fake_bgr = Lambda(lambda x: x[:, :, :, 1:])(fake_abgr)

    loss_g = 0
    loss_g += weights['w_recon'] * calc_loss(fake_bgr, real, "l1")
    loss_g += weights['w_eyes'] * K.mean(K.abs(mask_eyes*(fake_bgr - real)))

    for out in model_outputs[:-1]:
        out_size = out.get_shape().as_list()
        resized_real = tf.image.resize_images(real, out_size[1:3])
        loss_g += weights['w_recon'] * calc_loss(out, resized_real, "l1")
    return loss_g


def edge_loss(real, fake_abgr, mask_eyes, **weights):
    """ Edge Loss Function from Shoanlu GAN """
    alpha = Lambda(lambda x: x[:, :, :, :1])(fake_abgr)
    fake_bgr = Lambda(lambda x: x[:, :, :, 1:])(fake_abgr)

    loss_g = 0
    loss_g += weights['w_edge'] * calc_loss(first_order(fake_bgr, axis=1),
                                            first_order(real, axis=1), "l1")
    loss_g += weights['w_edge'] * calc_loss(first_order(fake_bgr, axis=2),
                                            first_order(real, axis=2), "l1")
    shape_mask_eyes = mask_eyes.get_shape().as_list()
    resized_mask_eyes = tf.image.resize_images(mask_eyes,
                                               [shape_mask_eyes[1]-1, shape_mask_eyes[2]-1])
    loss_g += weights['w_eyes'] * K.mean(K.abs(resized_mask_eyes *
                                               (first_order(fake_bgr, axis=1) -
                                                first_order(real, axis=1))))
    loss_g += weights['w_eyes'] * K.mean(K.abs(resized_mask_eyes *
                                               (first_order(fake_bgr, axis=2) -
                                                first_order(real, axis=2))))
    return loss_g


def perceptual_loss(real, fake_abgr, distorted, mask_eyes, vggface_feats, **weights):
    """ Perceptual Loss Function from Shoanlu GAN """
    alpha = Lambda(lambda x: x[:, :, :, :1])(fake_abgr)
    fake_bgr = Lambda(lambda x: x[:, :, :, 1:])(fake_abgr)
    fake = alpha * fake_bgr + (1-alpha) * distorted

    def preprocess_vggface(var_x):
        var_x = (var_x + 1) / 2 * 255  # channel order: BGR
        var_x -= [91.4953, 103.8827, 131.0912]
        return var_x

    real_sz224 = tf.image.resize_images(real, [224, 224])
    real_sz224 = Lambda(preprocess_vggface)(real_sz224)
    dist = Beta(0.2, 0.2)
    lam = dist.sample()  # use mixup trick here to reduce foward pass from 2 times to 1.
    mixup = lam*fake_bgr + (1-lam)*fake
    fake_sz224 = tf.image.resize_images(mixup, [224, 224])
    fake_sz224 = Lambda(preprocess_vggface)(fake_sz224)
    real_feat112, real_feat55, real_feat28, real_feat7 = vggface_feats(real_sz224)
    fake_feat112, fake_feat55, fake_feat28, fake_feat7 = vggface_feats(fake_sz224)

    # Apply instance norm on VGG(ResNet) features
    # From MUNIT https://github.com/NVlabs/MUNIT
    loss_g = 0

    def instnorm():
        return InstanceNormalization()

    loss_g += weights['w_pl'][0] * calc_loss(instnorm()(fake_feat7),
                                             instnorm()(real_feat7), "l2")
    loss_g += weights['w_pl'][1] * calc_loss(instnorm()(fake_feat28),
                                             instnorm()(real_feat28), "l2")
    loss_g += weights['w_pl'][2] * calc_loss(instnorm()(fake_feat55),
                                             instnorm()(real_feat55), "l2")
    loss_g += weights['w_pl'][3] * calc_loss(instnorm()(fake_feat112),
                                             instnorm()(real_feat112), "l2")
    return loss_g

# <<< END: from Shoanlu GAN >>> #


def generalized_loss_function(y_true, y_pred, var_a=1.0, cnst=1.0/255.0):
    """
    generalized function used to return a large variety of mathematical loss functions
    primary benefit is smooth, differentiable version of L1 loss

    Barron, J. A More General Robust Loss Function
    https://arxiv.org/pdf/1701.03077.pdf

    Parameters:
        a: penalty factor. larger number give larger weight to large deviations
        c: scale factor used to adjust to the input scale (i.e. inputs of mean 1e-4 or 256)

    Return:
        a loss value from the results of function(y_pred - y_true)

    Example:
        a=1.0, x>>c , c=1.0/255.0 will give a smoothly differentiable version of L1 / MAE loss
        a=1.999999 (lim as a->2), c=1.0/255.0 will give L2 / RMSE loss
    """
    var_x = y_pred - y_true
    loss = (K.abs(2.0-var_a)/var_a) * (K.pow(K.pow(var_x/cnst, 2.0)/K.abs(2.0-var_a) + 1.0,
                                             (var_a/2.0)) - 1.0)
    return K.mean(loss, axis=-1) * cnst


def staircase_loss(y_true, y_pred, var_a=16.0, cnst=1.0/255.0):
    """ Keras Staircase Loss """
    height = cnst
    width = cnst
    var_x = K.clip(K.abs(y_true - y_pred) - 0.5 * cnst, 0.0, 1.0)
    loss = height*(K.tanh(var_a*((var_x/width)-tf.floor(var_x/width)-0.5)) /
                   (2.0*K.tanh(var_a/2.0)) + 0.5 + tf.floor(var_x/width))
    loss += 1e-10
    return K.mean(loss, axis=-1)


def gradient_loss(y_true, y_pred):
    """
    Calculates the first and second order gradient difference between pixels of an image in the
    x and y dimensions. These gradients are then compared between the ground truth and the
    predicted image and the difference is taken. The difference used is a smooth L1 norm
    (approximate to MAE but differable at zero) When used as a loss, its minimization will result
    in predicted images approaching the samelevel of sharpness / blurriness as the ground truth.

    TV+TV2 Regularization with Nonconvex Sparseness-Inducing Penalty for Image Restoration,
    Chengwu Lu & Hua Huang, 2014 (http://downloads.hindawi.com/journals/mpe/2014/790547.pdf)

    Parameters:
        y_true: The predicted frames at each scale.
        y_true: The ground truth frames at each scale

    Return:
        The GD loss.
    """

    assert K.ndim(y_true) == 4
    y_true.set_shape([None, 80, 80, 3])
    y_pred.set_shape([None, 80, 80, 3])
    tv_weight = 1.0
    tv2_weight = 1.0
    loss = 0.0

    def diff_x(var_x):
        xleft = var_x[:, :, 1, :] - var_x[:, :, 0, :]
        xinner = tf.unstack(var_x[:, :, 2:, :] - var_x[:, :, :-2, :], axis=2)
        xright = var_x[:, :, -1, :] - var_x[:, :, -2, :]
        xout = [xleft] + xinner + [xright]
        xout = tf.stack(xout, axis=2)
        return xout * 0.5

    def diff_y(var_x):
        xtop = var_x[:, 1, :, :] - var_x[:, 0, :, :]
        xinner = tf.unstack(var_x[:, 2:, :, :] - var_x[:, :-2, :, :], axis=1)
        xbot = var_x[:, -1, :, :] - var_x[:, -2, :, :]
        xout = [xtop] + xinner + [xbot]
        xout = tf.stack(xout, axis=1)
        return xout * 0.5

    def diff_xx(var_x):
        xleft = var_x[:, :, 1, :] + var_x[:, :, 0, :]
        xinner = tf.unstack(var_x[:, :, 2:, :] + var_x[:, :, :-2, :], axis=2)
        xright = var_x[:, :, -1, :] + var_x[:, :, -2, :]
        xout = [xleft] + xinner + [xright]
        xout = tf.stack(xout, axis=2)
        return xout - 2.0 * var_x

    def diff_yy(var_x):
        xtop = var_x[:, 1, :, :] + var_x[:, 0, :, :]
        xinner = tf.unstack(var_x[:, 2:, :, :] + var_x[:, :-2, :, :], axis=1)
        xbot = var_x[:, -1, :, :] + var_x[:, -2, :, :]
        xout = [xtop] + xinner + [xbot]
        xout = tf.stack(xout, axis=1)
        return xout - 2.0 * var_x

    def diff_xy(var_x):
        # xout1
        top_left = var_x[:, 1, 1, :]+var_x[:, 0, 0, :]
        inner_left = tf.unstack(var_x[:, 2:, 1, :]+var_x[:, :-2, 0, :], axis=1)
        bot_left = var_x[:, -1, 1, :]+var_x[:, -2, 0, :]
        x_left = [top_left] + inner_left + [bot_left]
        x_left = tf.stack(x_left, axis=1)

        top_mid = var_x[:, 1, 2:, :]+var_x[:, 0, :-2, :]
        mid_mid = tf.unstack(var_x[:, 2:, 2:, :]+var_x[:, :-2, :-2, :], axis=1)
        bot_mid = var_x[:, -1, 2:, :]+var_x[:, -2, :-2, :]
        x_mid = [top_mid] + mid_mid + [bot_mid]
        x_mid = tf.stack(x_mid, axis=1)

        top_right = var_x[:, 1, -1, :]+var_x[:, 0, -2, :]
        inner_right = tf.unstack(var_x[:, 2:, -1, :]+var_x[:, :-2, -2, :], axis=1)
        bot_right = var_x[:, -1, -1, :]+var_x[:, -2, -2, :]
        x_right = [top_right] + inner_right + [bot_right]
        x_right = tf.stack(x_right, axis=1)

        x_mid = tf.unstack(x_mid, axis=2)
        xout1 = [x_left] + x_mid + [x_right]
        xout1 = tf.stack(xout1, axis=2)

        # xout2
        top_left = var_x[:, 0, 1, :]+var_x[:, 1, 0, :]
        inner_left = tf.unstack(var_x[:, :-2, 1, :]+var_x[:, 2:, 0, :], axis=1)
        bot_left = var_x[:, -2, 1, :]+var_x[:, -1, 0, :]
        x_left = [top_left] + inner_left + [bot_left]
        x_left = tf.stack(x_left, axis=1)

        top_mid = var_x[:, 0, 2:, :]+var_x[:, 1, :-2, :]
        mid_mid = tf.unstack(var_x[:, :-2, 2:, :]+var_x[:, 2:, :-2, :], axis=1)
        bot_mid = var_x[:, -2, 2:, :]+var_x[:, -1, :-2, :]
        x_mid = [top_mid] + mid_mid + [bot_mid]
        x_mid = tf.stack(x_mid, axis=1)

        top_right = var_x[:, 0, -1, :]+var_x[:, 1, -2, :]
        inner_right = tf.unstack(var_x[:, :-2, -1, :]+var_x[:, 2:, -2, :], axis=1)
        bot_right = var_x[:, -2, -1, :]+var_x[:, -1, -2, :]
        x_right = [top_right] + inner_right + [bot_right]
        x_right = tf.stack(x_right, axis=1)

        x_mid = tf.unstack(x_mid, axis=2)
        xout2 = [x_left] + x_mid + [x_right]
        xout2 = tf.stack(xout2, axis=2)

        return (xout1 - xout2) * 0.25

    loss += tv_weight * (generalized_loss_function(diff_x(y_true),
                                                   diff_x(y_pred),
                                                   var_a=1.999999) +
                         generalized_loss_function(diff_y(y_true),
                                                   diff_y(y_pred),
                                                   var_a=1.999999))

    loss += tv2_weight * (generalized_loss_function(diff_xx(y_true),
                                                    diff_xx(y_pred),
                                                    var_a=1.999999) +
                          generalized_loss_function(diff_yy(y_true),
                                                    diff_yy(y_pred),
                                                    var_a=1.999999) +
                          2.0 * generalized_loss_function(diff_xy(y_true),
                                                          diff_xy(y_pred),
                                                          var_a=1.999999))

    return loss / (tv_weight + tv2_weight)


def scharr_edges(image, magnitude):
    """
    Returns a tensor holding modified Scharr edge maps.
    Arguments:
    image: Image tensor with shape [batch_size, h, w, d] and type float32.
    The image(s) must be 2x2 or larger.
    magnitude: Boolean to determine if the edge magnitude or edge direction is returned
    Returns:
    Tensor holding edge maps for each channel. Returns a tensor with shape
    [batch_size, h, w, d, 2] where the last two dimensions hold [[dy[0], dx[0]],
    [dy[1], dx[1]], ..., [dy[d-1], dx[d-1]]] calculated using the Scharr filter.
    """

    # Define vertical and horizontal Scharr filters.
    static_image_shape = image.get_shape()
    image_shape = tf.shape(image)

    # modified 3x3 Scharr
    # kernels = [[[-17.0, -61.0, -17.0], [0.0, 0.0, 0.0], [17.0, 61.0, 17.0]],
    #         [[-17.0, 0.0, 17.0], [-61.0, 0.0, 61.0], [-17.0, 0.0, 17.0]]]

    # 5x5 Scharr
    kernels = [[[-1.0, -2.0, -3.0, -2.0, -1.0],
                [-1.0, -2.0, -6.0, -2.0, -1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 2.0, 6.0, 2.0, 1.0],
                [1.0, 2.0, 3.0, 2.0, 1.0]],
               [[-1.0, -1.0, 0.0, 1.0, 1.0],
                [-2.0, -2.0, 0.0, 2.0, 2.0],
                [-3.0, -6.0, 0.0, 6.0, 3.0],
                [-2.0, -2.0, 0.0, 2.0, 2.0],
                [-1.0, -1.0, 0.0, 1.0, 1.0]]]
    num_kernels = len(kernels)
    kernels = np.transpose(np.asarray(kernels), (1, 2, 0))
    kernels = np.expand_dims(kernels, -2) / np.sum(np.abs(kernels))
    kernels_tf = tf.constant(kernels, dtype=image.dtype)
    kernels_tf = tf.tile(kernels_tf, [1, 1, image_shape[-1], 1], name='scharr_filters')

    # Use depth-wise convolution to calculate edge maps per channel.
    pad_sizes = [[0, 0], [2, 2], [2, 2], [0, 0]]
    padded = tf.pad(image, pad_sizes, mode='REFLECT')

    # Output tensor has shape [batch_size, h, w, d * num_kernels].
    strides = [1, 1, 1, 1]
    output = tf.nn.depthwise_conv2d(padded, kernels_tf, strides, 'VALID')

    # Reshape to [batch_size, h, w, d, num_kernels].
    shape = tf.concat([image_shape, [num_kernels]], 0)
    output = tf.reshape(output, shape=shape)
    output.set_shape(static_image_shape.concatenate([num_kernels]))

    if magnitude:  # magnitude of edges
        output = tf.sqrt(tf.reduce_sum(tf.square(output), axis=-1))
    else:  # direction of edges
        output = tf.atan(tf.squeeze(tf.div(output[:, :, :, :, 0]/output[:, :, :, :, 1])))

    return output


def gmsd_loss(y_true, y_pred):
    """
    Improved image quality metric over MS-SSIM with easier calc
    http://www4.comp.polyu.edu.hk/~cslzhang/IQA/GMSD/GMSD.htm
    https://arxiv.org/ftp/arxiv/papers/1308/1308.3052.pdf
    """
    true_edge_mag = scharr_edges(y_true, True)
    pred_edge_mag = scharr_edges(y_pred, True)
    cnst = 0.002
    upper = 2.0 * tf.multiply(true_edge_mag, pred_edge_mag) + cnst
    lower = tf.square(true_edge_mag) + tf.square(pred_edge_mag) + cnst
    gms = tf.div(upper, lower)
    _mean, _var = tf.nn.moments(gms, axes=[1, 2], keep_dims=True)
    # single metric value per image in tensor [?, 1, 1]
    gmsd = tf.reduce_mean(tf.sqrt(_var), axis=-1)
    # need to expand to [?, height, width] dimensions for Keras ... modify to not be hard-coded
    return K.tile(gmsd, [1, 64, 64])


def ms_ssim(img1, img2, max_val=1.0, power_factors=(0.0517, 0.3295, 0.3462, 0.2726)):
    """
    Computes the MS-SSIM between img1 and img2.
    This function assumes that `img1` and `img2` are image batches, i.e. the last
    three dimensions are [height, width, channels].
    Note: The true SSIM is only defined on grayscale.  This function does not
    perform any colorspace transform.  (If input is already YUV, then it will
    compute YUV SSIM average.)
    Original paper: Wang, Zhou, Eero P. Simoncelli, and Alan C. Bovik. "Multiscale
    structural similarity for image quality assessment." Signals, Systems and
    Computers, 2004.
    Arguments:
    img1: First image batch.
    img2: Second image batch. Must have the same rank as img1.
    max_val: The dynamic range of the images (i.e., the difference between the
      maximum the and minimum allowed values).
    power_factors: Iterable of weights for each of the scales. The number of
      scales used is the length of the list. Index 0 is the unscaled
      resolution's weight and each increasing scale corresponds to the image
      being downsampled by 2.  Defaults to (0.0448, 0.2856, 0.3001, 0.2363,
      0.1333), which are the values obtained in the original paper.
    Returns:
    A tensor containing an MS-SSIM value for each image in batch.  The values
    are in range [0, 1].  Returns a tensor with shape:
    broadcast(img1.shape[:-3], img2.shape[:-3]).
    """

    def _verify_compatible_image_shapes(img1, img2):
        """
        Checks if two image tensors are compatible for applying SSIM or PSNR.
        This function checks if two sets of images have ranks at least 3, and if the
        last three dimensions match.
        Args:
        img1: Tensor containing the first image batch.
        img2: Tensor containing the second image batch.
        Returns:
        A tuple containing: the first tensor shape, the second tensor shape, and a
        list of control_flow_ops.Assert() ops implementing the checks.
        Raises:
        ValueError: When static shape check fails.
        """
        shape1 = img1.get_shape().with_rank_at_least(3)
        shape2 = img2.get_shape().with_rank_at_least(3)
        shape1[-3:].assert_is_compatible_with(shape2[-3:])

        if shape1.ndims is not None and shape2.ndims is not None:
            for dim1, dim2 in zip(reversed(shape1[:-3]), reversed(shape2[:-3])):
                if not (dim1 == 1 or dim2 == 1 or dim1.is_compatible_with(dim2)):
                    raise ValueError('Two images are not compatible: %s and %s' % (shape1, shape2))

        # Now assign shape tensors.
        shape1, shape2 = tf.shape_n([img1, img2])

        # TODO(sjhwang): Check if shape1[:-3] and shape2[:-3] are broadcastable.
        checks = []
        checks.append(tf.Assert(tf.greater_equal(tf.size(shape1), 3),
                                [shape1, shape2], summarize=10))
        checks.append(tf.Assert(tf.reduce_all(tf.equal(shape1[-3:], shape2[-3:])),
                                [shape1, shape2], summarize=10))

        return shape1, shape2, checks

    def _ssim_per_channel(img1, img2, max_val=1.0):
        """
        Computes SSIM index between img1 and img2 per color channel.
        This function matches the standard SSIM implementation from:
        Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004). Image
        quality assessment: from error visibility to structural similarity. IEEE
        transactions on image processing.
        Details:
        - 11x11 Gaussian filter of width 1.5 is used.
        - k1 = 0.01, k2 = 0.03 as in the original paper.
        Args:
        img1: First image batch.
        img2: Second image batch.
        max_val: The dynamic range of the images (i.e., the difference between the
          maximum the and minimum allowed values).
        Returns:
        A pair of tensors containing and channel-wise SSIM and contrast-structure
        values. The shape is [..., channels].
        """

        def _fspecial_gauss(size, sigma):
            """
            Function to mimic the 'fspecial' gaussian MATLAB function.
            """
            size = tf.convert_to_tensor(size, 'int32')
            sigma = tf.convert_to_tensor(sigma)

            coords = tf.cast(tf.range(size), sigma.dtype)
            coords -= tf.cast(size - 1, sigma.dtype) / 2.0

            var_g = tf.square(coords)
            var_g *= -0.5 / tf.square(sigma)

            var_g = tf.reshape(var_g, shape=[1, -1]) + tf.reshape(var_g, shape=[-1, 1])
            var_g = tf.reshape(var_g, shape=[1, -1])  # For tf.nn.softmax().
            var_g = tf.nn.softmax(var_g)
            return tf.reshape(var_g, shape=[size, size, 1, 1])

        def _ssim_helper(var_x, var_y, max_val, kernel, compensation=1.0):
            """
            Helper function for computing SSIM.
            SSIM estimates covariances with weighted sums.  The default parameters
            use a biased estimate of the covariance:
            Suppose `reducer` is a weighted sum, then the mean estimators are
            mu_x = sum_i w_i x_i,
            mu_y = sum_i w_i y_i,
            where w_i's are the weighted-sum weights, and covariance estimator is
            cov_{xy} = sum_i w_i (x_i - mu_x) (y_i - mu_y)
            with assumption sum_i w_i = 1. This covariance estimator is biased, since
            E[cov_{xy}] = (1 - sum_i w_i ^ 2) Cov(X, Y).
            For SSIM measure with unbiased covariance estimators, pass as `compensation`
            argument (1 - sum_i w_i ^ 2).
            Arguments:
            x: First set of images.
            y: Second set of images.
            reducer: Function that computes 'local' averages from set of images.
              For non-covolutional version, this is usually tf.reduce_mean(x, [1, 2]),
              and for convolutional version, this is usually tf.nn.avg_pool or
              tf.nn.conv2d with weighted-sum kernel.
            max_val: The dynamic range (i.e., the difference between the maximum
              possible allowed value and the minimum allowed value).
            compensation: Compensation factor. See above.
            Returns:
            A pair containing the luminance measure, and the contrast-structure measure.
            """

            def reducer(var_x, kernel):
                shape = tf.shape(var_x)
                var_x = tf.reshape(var_x, shape=tf.concat([[-1], shape[-3:]], 0))
                var_y = tf.nn.depthwise_conv2d(var_x, kernel, strides=[1, 1, 1, 1], padding='VALID')
                return tf.reshape(var_y, tf.concat([shape[:-3], tf.shape(var_y)[1:]], 0))

            _ssim_k1 = 0.01
            _ssim_k2 = 0.03

            c_1 = (_ssim_k1 * max_val) ** 2
            c_2 = (_ssim_k2 * max_val) ** 2

            # SSIM luminance measure is
            # (2 * mu_x * mu_y + c_1) / (mu_x ** 2 + mu_y ** 2 + c_1).
            mean0 = reducer(var_x, kernel)
            mean1 = reducer(var_y, kernel)
            num0 = mean0 * mean1 * 2.0
            den0 = tf.square(mean0) + tf.square(mean1)
            luminance = (num0 + c_1) / (den0 + c_1)

            # SSIM contrast-structure measure is
            #   (2 * cov_{xy} + c_2) / (cov_{xx} + cov_{yy} + c_2).
            # Note that `reducer` is a weighted sum with weight w_k, \sum_i w_i = 1, then
            #   cov_{xy} = \sum_i w_i (x_i - \mu_x) (y_i - \mu_y)
            #          = \sum_i w_i x_i y_i - (\sum_i w_i x_i) (\sum_j w_j y_j).
            num1 = reducer(var_x * var_y, kernel) * 2.0
            den1 = reducer(tf.square(var_x) + tf.square(var_y), kernel)
            c_2 *= compensation
            c_s = (num1 - num0 + c_2) / (den1 - den0 + c_2)

            # SSIM score is the product of the luminance and contrast-structure measures.
            return luminance, c_s

        filter_size = tf.constant(9, dtype='int32')  # changed from 11 to 9 due
        filter_sigma = tf.constant(1.5, dtype=img1.dtype)

        shape1, shape2 = tf.shape_n([img1, img2])
        checks = [tf.Assert(tf.reduce_all(tf.greater_equal(shape1[-3:-1], filter_size)),
                            [shape1, filter_size], summarize=8),
                  tf.Assert(tf.reduce_all(tf.greater_equal(shape2[-3:-1], filter_size)),
                            [shape2, filter_size], summarize=8)]

        # Enforce the check to run before computation.
        with tf.control_dependencies(checks):
            img1 = tf.identity(img1)

        # TODO(sjhwang): Try to cache kernels and compensation factor.
        kernel = _fspecial_gauss(filter_size, filter_sigma)
        kernel = tf.tile(kernel, multiples=[1, 1, shape1[-1], 1])

        # The correct compensation factor is `1.0 - tf.reduce_sum(tf.square(kernel))`,
        # but to match MATLAB implementation of MS-SSIM, we use 1.0 instead.
        compensation = 1.0

        # TODO(sjhwang): Try FFT.
        # TODO(sjhwang): Gaussian kernel is separable in space. Consider applying
        #   1-by-n and n-by-1 Gaussain filters instead of an n-by-n filter.

        luminance, c_s = _ssim_helper(img1, img2, max_val, kernel, compensation)

        # Average over the second and the third from the last: height, width.
        axes = tf.constant([-3, -2], dtype='int32')
        ssim_val = tf.reduce_mean(luminance * c_s, axes)
        c_s = tf.reduce_mean(c_s, axes)
        return ssim_val, c_s

    def do_pad(images, remainder):
        padding = tf.expand_dims(remainder, -1)
        padding = tf.pad(padding, [[1, 0], [1, 0]])
        return [tf.pad(x, padding, mode='SYMMETRIC') for x in images]

    # Shape checking.
    shape1 = img1.get_shape().with_rank_at_least(3)
    shape2 = img2.get_shape().with_rank_at_least(3)
    shape1[-3:].merge_with(shape2[-3:])

    with tf.name_scope(None, 'MS-SSIM', [img1, img2]):
        shape1, shape2, checks = _verify_compatible_image_shapes(img1, img2)
    with tf.control_dependencies(checks):
        img1 = tf.identity(img1)

    # Need to convert the images to float32.  Scale max_val accordingly so that
    # SSIM is computed correctly.
    max_val = tf.cast(max_val, img1.dtype)
    max_val = tf.image.convert_image_dtype(max_val, 'float32')
    img1 = tf.image.convert_image_dtype(img1, 'float32')
    img2 = tf.image.convert_image_dtype(img2, 'float32')

    imgs = [img1, img2]
    shapes = [shape1, shape2]

    # img1 and img2 are assumed to be a (multi-dimensional) batch of
    # 3-dimensional images (height, width, channels). `heads` contain the batch
    # dimensions, and `tails` contain the image dimensions.
    heads = [s[:-3] for s in shapes]
    tails = [s[-3:] for s in shapes]

    divisor = [1, 2, 2, 1]
    divisor_tensor = tf.constant(divisor[1:], dtype='int32')

    mc_s = []
    for k in range(len(power_factors)):
        with tf.name_scope(None, 'Scale%d' % k, imgs):
            if k > 0:
                # Avg pool takes rank 4 tensors. Flatten leading dimensions.
                flat_imgs = [tf.reshape(x, tf.concat([[-1], t], 0))
                             for x, t in zip(imgs, tails)]

                remainder = tails[0] % divisor_tensor
                need_padding = tf.reduce_any(tf.not_equal(remainder, 0))
                padded = tf.cond(need_padding,
                                 lambda: do_pad(flat_imgs, remainder),
                                 lambda: flat_imgs)

                downscaled = [tf.nn.avg_pool(x, ksize=divisor, strides=divisor, padding='VALID')
                              for x in padded]
                tails = [x[1:] for x in tf.shape_n(downscaled)]
                imgs = [tf.reshape(x, tf.concat([h, t], 0))
                        for x, h, t in zip(downscaled, heads, tails)]

                # Overwrite previous ssim value since we only need the last one.
                ssim_per_channel, c_s = _ssim_per_channel(*imgs, max_val=max_val)
                mc_s.append(tf.nn.relu(c_s))

    # Remove the c_s score for the last scale. In the MS-SSIM calculation,
    # we use the l(p) at the highest scale. l(p) * c_s(p) is ssim(p).
    mc_s.pop()  # Remove the c_s score for the last scale.
    mc_s_and_ssim = tf.stack(mc_s + [tf.nn.relu(ssim_per_channel)], axis=-1)
    # Take weighted geometric mean across the scale axis.
    ms_ssim = tf.reduce_prod(tf.pow(mc_s_and_ssim, power_factors), [-1])

    return tf.reduce_mean(ms_ssim, [-1])  # Avg over color channels.


def ms_ssim_loss(y_true, y_pred):
    """ Keras ms_ssim loss """
    msssim = K.expand_dims(K.expand_dims(1.0 - ms_ssim(y_true, y_pred), axis=-1), axis=-1)
    # need to expand to [1, height, width] dimensions for Keras ... modify to not be hard-coded
    return K.tile(msssim, [1, 64, 64])
