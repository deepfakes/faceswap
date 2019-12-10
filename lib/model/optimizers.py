#!/usr/bin/env python3
""" Optimizers for faceswap.py """
# Naming convention inherited from Keras so ignore invalid names
# pylint:disable=invalid-name

import logging

from keras import backend as K
from keras.optimizers import Adam as KerasAdam

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class Adam(KerasAdam):
    """Adapted Keras Adam Optimizer to allow support of calculations
       on CPU for Tensorflow.

       Adapted from https://github.com/iperov/DeepFaceLab
    """

    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999,
                 epsilon=None, decay=0., amsgrad=False, cpu_mode=0, **kwargs):
        super().__init__(lr, beta_1, beta_2, epsilon, decay, **kwargs)
        self.cpu_mode = self.set_cpu_mode(cpu_mode)

    @staticmethod
    def set_cpu_mode(cpu_mode):
        """ Set the CPU mode to 0 if not using tensorflow, else passed in arg """
        retval = False if K.backend() != "tensorflow" else cpu_mode
        logger.debug("Optimizer CPU Mode set to %s", retval)
        return retval

    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr = lr * (1. / (1. + self.decay * K.cast(self.iterations,
                                                      K.dtype(self.decay))))

        t = K.cast(self.iterations, K.floatx()) + 1
        lr_t = lr * (K.sqrt(1. - K.pow(self.beta_2, t)) /
                     (1. - K.pow(self.beta_1, t)))

        # Pass off to CPU if requested
        if self.cpu_mode:
            with K.tf.device("/cpu:0"):
                ms, vs, vhats = self.update_1(params)
        else:
            ms, vs, vhats = self.update_1(params)

        self.weights = [self.iterations] + ms + vs + vhats

        for p, g, m, v, vhat in zip(params, grads, ms, vs, vhats):
            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(g)
            if self.amsgrad:
                vhat_t = K.maximum(vhat, v_t)
                p_t = p - lr_t * m_t / (K.sqrt(vhat_t) + self.epsilon)
                self.updates.append(K.update(vhat, vhat_t))
            else:
                p_t = p - lr_t * m_t / (K.sqrt(v_t) + self.epsilon)

            self.updates.append(K.update(m, m_t))
            self.updates.append(K.update(v, v_t))
            new_p = p_t

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def update_1(self, params):
        """ First update on CPU or GPU """
        ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        if self.amsgrad:
            vhats = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        else:
            vhats = [K.zeros(1) for _ in params]
        return ms, vs, vhats
