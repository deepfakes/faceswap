#!/usr/bin/env python3
""" Custom Loss Functions for faceswap.py
    Losses from:
        keras.contrib
        dfaker: https://github.com/dfaker/df
        shoanlu GAN: https://github.com/shaoanlu/faceswap-GAN"""

from __future__ import absolute_import


import keras.backend as K
from keras.layers import Lambda, concatenate
import tensorflow as tf
from tensorflow.contrib.distributions import Beta

from .normalization import InstanceNormalization


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
        self.c1 = (self.k1 * self.max_value) ** 2
        self.c2 = (self.k2 * self.max_value) ** 2
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

        # Reshape to get the var in the cells
        _, w, h, c1, c2, c3 = self.__int_shape(patches_pred)
        patches_pred = K.reshape(patches_pred, [-1, w, h, c1 * c2 * c3])
        patches_true = K.reshape(patches_true, [-1, w, h, c1 * c2 * c3])
        # Get mean
        u_true = K.mean(patches_true, axis=-1)
        u_pred = K.mean(patches_pred, axis=-1)
        # Get variance
        var_true = K.var(patches_true, axis=-1)
        var_pred = K.var(patches_pred, axis=-1)
        # Get std dev
        covar_true_pred = K.mean(
            patches_true * patches_pred, axis=-1) - u_true * u_pred

        ssim = (2 * u_true * u_pred + self.c1) * (
            2 * covar_true_pred + self.c2)
        denom = (K.square(u_true) + K.square(u_pred) + self.c1) * (
            var_pred + var_true + self.c2)
        ssim /= denom  # no need for clipping, c1 + c2 make the denom non-zero
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
        '''
        Extract the patches from an image
        # Parameters
            x : The input image
            ksizes : 2-d tuple with the kernel size
            ssizes : 2-d tuple with the strides size
            padding : 'same' or 'valid'
            data_format : 'channels_last' or 'channels_first'
        # Returns
            The (k_w,k_h) patches extracted
            TF ==> (batch_size,w,h,k_w,k_h,c)
            TH ==> (batch_size,w,h,c,k_w,k_h)
        '''
        kernel = [1, ksizes[0], ksizes[1], 1]
        strides = [1, ssizes[0], ssizes[1], 1]
        padding = self._preprocess_padding(padding)
        if data_format == 'channels_first':
            x = K.permute_dimensions(x, (0, 2, 3, 1))
        _, _, _, ch_i = K.int_shape(x)
        patches = tf.extract_image_patches(x, kernel, strides, [1, 1, 1, 1],
                                           padding)
        # Reshaping to fit Theano
        _, w, h, ch = K.int_shape(patches)
        patches = tf.reshape(tf.transpose(tf.reshape(patches,
                                                     [-1, w, h,
                                                      tf.floordiv(ch, ch_i),
                                                      ch_i]),
                                          [0, 1, 2, 4, 3]),
                             [-1, w, h, ch_i, ksizes[0], ksizes[1]])
        if data_format == 'channels_last':
            patches = K.permute_dimensions(patches, [0, 1, 2, 4, 5, 3])
        return patches


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


class PenalizedLoss():  # pylint: disable=too-few-public-methods
    """ Penalized Loss
        from: https://github.com/dfaker/df """
    def __init__(self, mask, loss_func, mask_prop=1.0):
        self.mask = mask
        self.loss_func = loss_func
        self.mask_prop = mask_prop
        self.mask_as_k_inv_prop = 1-mask_prop

    def __call__(self, y_true, y_pred):
        # pylint: disable=invalid-name
        tro, tgo, tbo = tf.split(y_true, 3, 3)
        pro, pgo, pbo = tf.split(y_pred, 3, 3)

        tr = tro
        tg = tgo
        tb = tbo

        pr = pro
        pg = pgo
        pb = pbo
        m = self.mask

        m = m * self.mask_prop
        m += self.mask_as_k_inv_prop
        tr *= m
        tg *= m
        tb *= m

        pr *= m
        pg *= m
        pb *= m

        y = tf.concat([tr, tg, tb], 3)
        p = tf.concat([pr, pg, pb], 3)

        # yo = tf.stack([tro,tgo,tbo],3)
        # po = tf.stack([pro,pgo,pbo],3)

        return self.loss_func(y, p)
