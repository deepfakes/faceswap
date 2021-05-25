#!/usr/bin/env python3
""" Custom Optimizers for PlaidML/Keras 2.2. """

from keras import backend as K
from keras.optimizers import Optimizer


class AdaBelief(Optimizer):
    """AdaBelief optimizer.

    Default parameters follow those provided in the original paper.

    Parameters
    ----------
    learning_rate: float
        The learning rate.
    beta_1: float
        The exponential decay rate for the 1st moment estimates.
    beta_2: float
        The exponential decay rate for the 2nd moment estimates.
    epsilon: float, optional
        A small constant for numerical stability. Default: `K.epsilon()`.
    amsgrad: bool
        Whether to apply AMSGrad variant of this algorithm from the paper "On the Convergence
        of Adam and beyond".

    References
    ----------
    AdaBelief - A Method for Stochastic Optimization - https://arxiv.org/abs/1412.6980v8
    On the Convergence of AdaBelief and Beyond - https://openreview.net/forum?id=ryQu7f-RZ

    Adapted from https://github.com/liaoxuanzhi/adabelief

    BSD 2-Clause License

    Copyright (c) 2021, Juntang Zhuang
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:

    1. Redistributions of source code must retain the above copyright notice, this
    list of conditions and the following disclaimer.

    2. Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
    FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
    DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
    CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
    OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

    """

    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999,
                 epsilon=None, decay=0., weight_decay=0.0, **kwargs):
        super().__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = K.variable(lr, name='lr')
            self.beta_1 = K.variable(beta_1, name='beta_1')
            self.beta_2 = K.variable(beta_2, name='beta_2')
            self.decay = K.variable(decay, name='decay')
        if epsilon is None:
            epsilon = K.epsilon()
        self.epsilon = float(epsilon)
        self.initial_decay = decay
        self.weight_decay = float(weight_decay)

    def get_updates(self, loss, params):  # pylint:disable=too-many-locals
        """ Get the weight updates

        Parameters
        ----------
        loss: list
            The loss to update
        parans: list
            The variables
        """
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        l_r = self.lr
        if self.initial_decay > 0:
            l_r = l_r * (1. / (1. + self.decay * K.cast(self.iterations,
                                                        K.dtype(self.decay))))

        var_t = K.cast(self.iterations, K.floatx()) + 1
        # bias correction
        bias_correction1 = 1. - K.pow(self.beta_1, var_t)
        bias_correction2 = 1. - K.pow(self.beta_2, var_t)

        m_s = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        v_s = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]

        self.weights = [self.iterations] + m_s + v_s

        for param, grad, var_m, var_v in zip(params, grads, m_s, v_s):
            if self.weight_decay != 0.:
                grad += self.weight_decay * K.stop_gradient(param)

            m_t = (self.beta_1 * var_m) + (1. - self.beta_1) * grad
            m_corr_t = m_t / bias_correction1

            v_t = (self.beta_2 * var_v) + (1. - self.beta_2) * K.square(grad - m_t) + self.epsilon
            v_corr_t = K.sqrt(v_t / bias_correction2)

            p_t = param - l_r * m_corr_t / (v_corr_t + self.epsilon)

            self.updates.append(K.update(var_m, m_t))
            self.updates.append(K.update(var_v, v_t))
            new_param = p_t

            # Apply constraints.
            if getattr(param, 'constraint', None) is not None:
                new_param = param.constraint(new_param)

            self.updates.append(K.update(param, new_param))
        return self.updates

    def get_config(self):
        """ Returns the config of the optimizer.

        An optimizer config is a Python dictionary (serializable) containing the configuration of
        an optimizer. The same optimizer can be reinstantiated later (without any saved state) from
        this configuration.

        Returns
        -------
        dict
            The optimizer configuration.
        """
        config = dict(lr=float(K.get_value(self.lr)),
                      beta_1=float(K.get_value(self.beta_1)),
                      beta_2=float(K.get_value(self.beta_2)),
                      decay=float(K.get_value(self.decay)),
                      epsilon=self.epsilon,
                      weight_decay=self.weight_decay)
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
