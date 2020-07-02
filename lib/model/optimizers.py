#!/usr/bin/env python3
""" Optimizers for faceswap.py """
# Naming convention inherited from Keras so ignore invalid names
# pylint:disable=invalid-name

import logging

from keras import backend as K
from keras.optimizers import Adam as KerasAdam

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class Adam(KerasAdam):
    """Adapted Keras Adam Optimizer to allow support of calculations on CPU for Tensorflow.

    Default parameters follow those provided in the original paper. Adapted from
    https://github.com/iperov/DeepFaceLab

    Parameters
    ----------
    lr: float, optional
        >= `0`. Learning rate. Default: `0.001`
    beta_1: float, optional
        `0` < beta < `1` Generally close to `1`. Default: `0.9`
    beta_2: float, optional
        `0` < beta < `1`. Generally close to `1`. Default: `0.999`
    epsilon: float, optional
        >= `0`. Fuzz factor. If ``None``, defaults to `K.epsilon()`. Default: ``None``
    decay: float, optional
        >= 0. Learning rate decay over each update. Default: `0`
    amsgrad: bool, optional
        ``True`` to apply the AMSGrad variant of this algorithm from the paper "On the Convergence
        of Adam and Beyond" otherwise ``False``. Default: ``False``
    cpu_mode: bool, optional
        Set to ``True`` to perform some of the calculations on CPU for Nvidia backends, otherwise
        ``False``. Default: ``False``
    kwargs: dict
        Any additional standard Keras optimizer keyword arguments

    References
    ----------
    - Adam - A Method for Stochastic Optimization - https://arxiv.org/abs/1412.6980v8

    - On the Convergence of Adam and Beyond - https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self,
                 lr=0.001,
                 beta_1=0.9,
                 beta_2=0.999,
                 epsilon=None,
                 decay=0.,
                 amsgrad=False,
                 cpu_mode=False,
                 **kwargs):
        super().__init__(lr, beta_1, beta_2, epsilon, decay, **kwargs)
        self.cpu_mode = self._set_cpu_mode(cpu_mode)

    @staticmethod
    def _set_cpu_mode(cpu_mode):
        """ Sets the CPU mode to False if not using Tensorflow, otherwise the given value.

        Parameters
        ----------
        cpu_mode: bool
            Set to ``True`` to perform some of the calculations on CPU for Nvidia backends,
            otherwise ``False``.

        Returns
        -------
        bool
            ``True`` if some calculations should be performed on CPU otherwise ``False``
        """
        retval = False if K.backend() != "tensorflow" else cpu_mode
        logger.debug("Optimizer CPU Mode set to %s", retval)
        return retval

    def get_updates(self, loss, params):
        """ Obtain the optimizer loss updates.

        Parameters
        ----------
        loss: list
            List of tensors

        params: list
            List of tensors

        Returns
        -------
        list
            List of tensors
        """
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
                ms, vs, vhats = self._update_1(params)
        else:
            ms, vs, vhats = self._update_1(params)

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

    def _update_1(self, params):
        """ Perform the first update. Run under CPU context if running on Tensorflow and CPU mode
        is enabled, otherwise run on the default device. """
        ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        if self.amsgrad:
            vhats = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        else:
            vhats = [K.zeros(1) for _ in params]
        return ms, vs, vhats
