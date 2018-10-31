#!/usr/bin/env python3
""" Taken from: https://github.com/dfaker """

import tensorflow as tf


class PenalizedLoss():
    """ Penalized Loss """
    def __init__(self, mask, loss_func, mask_prop=1.0):
        self.mask = mask
        self.loss_func = loss_func
        self.mask_prop = mask_prop
        self.mask_as_k_inv_prop = 1-mask_prop

    def __call__(self, y_true, y_pred):
        # pylint: disable=C0103
        tro, tgo, tbo = tf.split(y_true, 3, 3)
        pro, pgo, pbo = tf.split(y_pred, 3, 3)

        tr = tro
        tg = tgo
        tb = tbo

        pr = pro
        pg = pgo
        pb = pbo
        m = self.mask

        m = m*self.mask_prop
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
