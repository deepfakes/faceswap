#!/usr/bin/env python3
""" Custom Optimizers for TensorFlow 2.x/tf.keras """

import inspect
import sys

import tensorflow as tf

# Ignore linting errors from Tensorflow's thoroughly broken import system
from tensorflow.keras.optimizers import Adam, Nadam, RMSprop  # noqa:E501,F401  pylint:disable=import-error,unused-import
keras = tf.keras


class AdaBelief(tf.keras.optimizers.Optimizer):
    """ Implementation of the AdaBelief Optimizer

    Inherits from: tf.keras.optimizers.Optimizer.

    AdaBelief Optimizer is not a placement of the heuristic warmup, the settings should be kept if
    warmup has already been employed and tuned in the baseline method. You can enable warmup by
    setting `total_steps` and `warmup_proportion` (see examples)

    Lookahead (see references) can be integrated with AdaBelief Optimizer, which is announced by
    Less Wright and the new combined optimizer can also be called "Ranger". The mechanism can be
    enabled by using the lookahead wrapper. (See examples)

    Parameters
    ----------
    learning_rate: `Tensor`, float or :class: `tf.keras.optimizers.schedules.LearningRateSchedule`
        The learning rate.
    beta_1: float
        The exponential decay rate for the 1st moment estimates.
    beta_2: float
        The exponential decay rate for the 2nd moment estimates.
    epsilon: float
        A small constant for numerical stability.
    weight_decay: `Tensor`, float or :class: `tf.keras.optimizers.schedules.LearningRateSchedule`
        Weight decay for each parameter.
    rectify: bool
        Whether to enable rectification as in RectifiedAdam
    amsgrad: bool
        Whether to apply AMSGrad variant of this algorithm from the paper "On the Convergence
        of Adam and beyond".
    sma_threshold. float
        The threshold for simple mean average.
    total_steps: int
        Total number of training steps. Enable warmup by setting a positive value.
    warmup_proportion: float
        The proportion of increasing steps.
    min_lr: float
        Minimum learning rate after warmup.
    name: str, optional
        Name for the operations created when applying gradients. Default: ``"AdaBeliefOptimizer"``.
    **kwargs: dict
        Standard Keras Optimizer keyword arguments. Allowed to be {`clipnorm`, `clipvalue`, `lr`,
        `decay`}. `clipnorm` is clip gradients by norm; `clipvalue` is clip gradients by value,
        `decay` is included for backward compatibility to allow time inverse decay of learning
        rate. `lr` is included for backward compatibility, recommended to use `learning_rate`
        instead.

    Examples
    --------
    >>> from adabelief_tf import AdaBelief
    >>> opt = AdaBelief(lr=1e-3)

    Example of serialization:

    >>> optimizer = AdaBelief(learning_rate=lr_scheduler, weight_decay=wd_scheduler)
    >>> config = tf.keras.optimizers.serialize(optimizer)
    >>> new_optimizer = tf.keras.optimizers.deserialize(config,
    ...                                                 custom_objects=dict(AdaBelief=AdaBelief))

    Example of warm up:

    >>> opt = AdaBelief(lr=1e-3, total_steps=10000, warmup_proportion=0.1, min_lr=1e-5)

    In the above example, the learning rate will increase linearly from 0 to `lr` in 1000 steps,
    then decrease linearly from `lr` to `min_lr` in 9000 steps.

    Example of enabling Lookahead:

    >>> adabelief = AdaBelief()
    >>> ranger = tfa.optimizers.Lookahead(adabelief, sync_period=6, slow_step_size=0.5)

    Notes
    -----
    `amsgrad` is not described in the original paper. Use it with caution.

    References
    ----------
    Juntang Zhuang et al. - AdaBelief Optimizer: Adapting stepsizes by the belief in observed
    gradients - https://arxiv.org/abs/2010.07468.

    Original implementation - https://github.com/juntang-zhuang/Adabelief-Optimizer

    Michael R. Zhang et.al - Lookahead Optimizer: k steps forward, 1 step back -
    https://arxiv.org/abs/1907.08610v1

    Adapted from https://github.com/juntang-zhuang/Adabelief-Optimizer

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

    def __init__(self, learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-14,
                 weight_decay=0.0, rectify=True, amsgrad=False, sma_threshold=5.0, total_steps=0,
                 warmup_proportion=0.1, min_lr=0.0, name="AdaBeliefOptimizer", **kwargs):
        # pylint:disable=too-many-arguments
        super().__init__(name, **kwargs)
        self._set_hyper("learning_rate", kwargs.get("lr", learning_rate))
        self._set_hyper("beta_1", beta_1)
        self._set_hyper("beta_2", beta_2)
        self._set_hyper("decay", self._initial_decay)
        self._set_hyper("weight_decay", weight_decay)
        self._set_hyper("sma_threshold", sma_threshold)
        self._set_hyper("total_steps", int(total_steps))
        self._set_hyper("warmup_proportion", warmup_proportion)
        self._set_hyper("min_lr", min_lr)
        self.epsilon = epsilon or tf.keras.backend.epsilon()
        self.amsgrad = amsgrad
        self.rectify = rectify
        self._has_weight_decay = weight_decay != 0.0
        self._initial_total_steps = total_steps

    def _create_slots(self, var_list):
        """ Create slots for the first and second moments

        Parameters
        ----------
        var_list: list
            List of tensorflow variables to create slots for
        """
        for var in var_list:
            self.add_slot(var, "m")
            self.add_slot(var, "v")
            if self.amsgrad:
                self.add_slot(var, "vhat")

    def set_weights(self, weights):
        """ Set the weights of the optimizer.

        The weights of an optimizer are its state (IE, variables). This function takes the weight
        values associated with this optimizer as a list of Numpy arrays. The first value is always
        the iterations count of the optimizer, followed by the optimizers state variables in the
        order they are created. The passed values are used to set the new state of the optimizer.

        Parameters
        ----------
        weights: list
            weight values as a list of numpy arrays.
        """
        params = self.weights
        num_vars = int((len(params) - 1) / 2)
        if len(weights) == 3 * num_vars + 1:
            weights = weights[: len(params)]
        super().set_weights(weights)

    def _decayed_wd(self, var_dtype):
        """ Set the weight decay

        Parameters
        ----------
        var_dtype: str
            The data type to to set up weight decay for

        Returns
        -------
        Tensor
            The weight decay variable
        """
        wd_t = self._get_hyper("weight_decay", var_dtype)
        if isinstance(wd_t, tf.keras.optimizers.schedules.LearningRateSchedule):
            wd_t = tf.cast(wd_t(self.iterations), var_dtype)
        return wd_t

    def _resource_apply_dense(self, grad, handle, apply_state=None):
        # pylint:disable=too-many-locals,unused-argument
        """ Add ops to apply dense gradients to the variable handle.

        Parameters
        ----------
        grad: Tensor
            A tensor representing the gradient.
        handle: Tensor
            a Tensor of dtype resource which points to the variable to be updated.
        apply_state: dict
            A dict which is used across multiple apply calls.

        Returns
        -------
            An Operation which updates the value of the variable.
        """
        var_dtype = handle.dtype.base_dtype
        lr_t = self._decayed_lr(var_dtype)
        wd_t = self._decayed_wd(var_dtype)
        var_m = self.get_slot(handle, "m")
        var_v = self.get_slot(handle, "v")
        beta_1_t = self._get_hyper("beta_1", var_dtype)
        beta_2_t = self._get_hyper("beta_2", var_dtype)
        epsilon_t = tf.convert_to_tensor(self.epsilon, var_dtype)
        local_step = tf.cast(self.iterations + 1, var_dtype)
        beta_1_power = tf.math.pow(beta_1_t, local_step)
        beta_2_power = tf.math.pow(beta_2_t, local_step)

        if self._initial_total_steps > 0:
            total_steps = self._get_hyper("total_steps", var_dtype)
            warmup_steps = total_steps * self._get_hyper("warmup_proportion", var_dtype)
            min_lr = self._get_hyper("min_lr", var_dtype)
            decay_steps = tf.maximum(total_steps - warmup_steps, 1)
            decay_rate = (min_lr - lr_t) / decay_steps
            lr_t = tf.where(local_step <= warmup_steps,
                            lr_t * (local_step / warmup_steps),
                            lr_t + decay_rate * tf.minimum(local_step - warmup_steps, decay_steps))

        m_t = var_m.assign(beta_1_t * var_m + (1.0 - beta_1_t) * grad,
                           use_locking=self._use_locking)
        m_corr_t = m_t / (1.0 - beta_1_power)

        v_t = var_v.assign(
            beta_2_t * var_v + (1.0 - beta_2_t) * tf.math.square(grad - m_t) + epsilon_t,
            use_locking=self._use_locking)

        if self.amsgrad:
            vhat = self.get_slot(handle, "vhat")
            vhat_t = vhat.assign(tf.maximum(vhat, v_t), use_locking=self._use_locking)
            v_corr_t = tf.math.sqrt(vhat_t / (1.0 - beta_2_power))
        else:
            vhat_t = None
            v_corr_t = tf.math.sqrt(v_t / (1.0 - beta_2_power))

        if self.rectify:
            sma_inf = 2.0 / (1.0 - beta_2_t) - 1.0
            sma_t = sma_inf - 2.0 * local_step * beta_2_power / (1.0 - beta_2_power)
            r_t = tf.math.sqrt((sma_t - 4.0) / (sma_inf - 4.0) *
                               (sma_t - 2.0) / (sma_inf - 2.0) *
                               sma_inf / sma_t)
            sma_threshold = self._get_hyper("sma_threshold", var_dtype)
            var_t = tf.where(sma_t >= sma_threshold,
                             r_t * m_corr_t / (v_corr_t + epsilon_t),
                             m_corr_t)
        else:
            var_t = m_corr_t / (v_corr_t + epsilon_t)

        if self._has_weight_decay:
            var_t += wd_t * handle

        var_update = handle.assign_sub(lr_t * var_t, use_locking=self._use_locking)
        updates = [var_update, m_t, v_t]

        if self.amsgrad:
            updates.append(vhat_t)
        return tf.group(*updates)

    def _resource_apply_sparse(self, grad, handle, indices, apply_state=None):
        # pylint:disable=too-many-locals, unused-argument
        """ Add ops to apply sparse gradients to the variable handle.

        Similar to _apply_sparse, the indices argument to this method has been de-duplicated.
        Optimizers which deal correctly with non-unique indices may instead override
        :func:`_resource_apply_sparse_duplicate_indices` to avoid this overhead.

        Parameters
        ----------
        grad: Tensor
            a Tensor representing the gradient for the affected indices.
        handle: Tensor
            a Tensor of dtype resource which points to the variable to be updated.
        indices: Tensor
            a Tensor of integral type representing the indices for which the gradient is nonzero.
            Indices are unique.
        apply_state: dict
            A dict which is used across multiple apply calls.

        Returns
        -------
            An Operation which updates the value of the variable.
        """
        var_dtype = handle.dtype.base_dtype
        lr_t = self._decayed_lr(var_dtype)
        wd_t = self._decayed_wd(var_dtype)
        beta_1_t = self._get_hyper("beta_1", var_dtype)
        beta_2_t = self._get_hyper("beta_2", var_dtype)
        epsilon_t = tf.convert_to_tensor(self.epsilon, var_dtype)
        local_step = tf.cast(self.iterations + 1, var_dtype)
        beta_1_power = tf.math.pow(beta_1_t, local_step)
        beta_2_power = tf.math.pow(beta_2_t, local_step)

        if self._initial_total_steps > 0:
            total_steps = self._get_hyper("total_steps", var_dtype)
            warmup_steps = total_steps * self._get_hyper("warmup_proportion", var_dtype)
            min_lr = self._get_hyper("min_lr", var_dtype)
            decay_steps = tf.maximum(total_steps - warmup_steps, 1)
            decay_rate = (min_lr - lr_t) / decay_steps
            lr_t = tf.where(local_step <= warmup_steps,
                            lr_t * (local_step / warmup_steps),
                            lr_t + decay_rate * tf.minimum(local_step - warmup_steps, decay_steps))

        var_m = self.get_slot(handle, "m")
        m_scaled_g_values = grad * (1 - beta_1_t)
        m_t = var_m.assign(var_m * beta_1_t, use_locking=self._use_locking)
        m_t = self._resource_scatter_add(var_m, indices, m_scaled_g_values)
        m_corr_t = m_t / (1.0 - beta_1_power)

        var_v = self.get_slot(handle, "v")
        m_t_indices = tf.gather(m_t, indices)  # pylint:disable=no-value-for-parameter
        v_scaled_g_values = tf.math.square(grad - m_t_indices) * (1 - beta_2_t)
        v_t = var_v.assign(var_v * beta_2_t + epsilon_t, use_locking=self._use_locking)
        v_t = self._resource_scatter_add(var_v, indices, v_scaled_g_values)

        if self.amsgrad:
            vhat = self.get_slot(handle, "vhat")
            vhat_t = vhat.assign(tf.maximum(vhat, v_t), use_locking=self._use_locking)
            v_corr_t = tf.math.sqrt(vhat_t / (1.0 - beta_2_power))
        else:
            vhat_t = None
            v_corr_t = tf.math.sqrt(v_t / (1.0 - beta_2_power))

        if self.rectify:
            sma_inf = 2.0 / (1.0 - beta_2_t) - 1.0
            sma_t = sma_inf - 2.0 * local_step * beta_2_power / (1.0 - beta_2_power)
            r_t = tf.math.sqrt((sma_t - 4.0) / (sma_inf - 4.0) *
                               (sma_t - 2.0) / (sma_inf - 2.0) *
                               sma_inf / sma_t)
            sma_threshold = self._get_hyper("sma_threshold", var_dtype)
            var_t = tf.where(sma_t >= sma_threshold,
                             r_t * m_corr_t / (v_corr_t + epsilon_t),
                             m_corr_t)
        else:
            var_t = m_corr_t / (v_corr_t + epsilon_t)

        if self._has_weight_decay:
            var_t += wd_t * handle

        var_update = self._resource_scatter_add(handle,
                                                indices,
                                                tf.gather(  # pylint:disable=no-value-for-parameter
                                                    tf.math.negative(lr_t) * var_t,
                                                    indices))

        updates = [var_update, m_t, v_t]
        if self.amsgrad:
            updates.append(vhat_t)
        return tf.group(*updates)

    def get_config(self):
        """ Returns the config of the optimizer.

        An optimizer config is a Python dictionary (serializable) containing the configuration of
        an optimizer. The same optimizer can be re-instantiated later (without any saved state)
        from this configuration.

        Returns
        -------
        dict
            The optimizer configuration.
        """
        config = super().get_config()
        config.update({"learning_rate": self._serialize_hyperparameter("learning_rate"),
                       "beta_1": self._serialize_hyperparameter("beta_1"),
                       "beta_2": self._serialize_hyperparameter("beta_2"),
                       "decay": self._serialize_hyperparameter("decay"),
                       "weight_decay": self._serialize_hyperparameter("weight_decay"),
                       "sma_threshold": self._serialize_hyperparameter("sma_threshold"),
                       "epsilon": self.epsilon,
                       "amsgrad": self.amsgrad,
                       "rectify": self.rectify,
                       "total_steps": self._serialize_hyperparameter("total_steps"),
                       "warmup_proportion": self._serialize_hyperparameter("warmup_proportion"),
                       "min_lr": self._serialize_hyperparameter("min_lr")})
        return config


# Update layers into Keras custom objects
for _name, obj in inspect.getmembers(sys.modules[__name__]):
    if inspect.isclass(obj) and obj.__module__ == __name__:
        keras.utils.get_custom_objects().update({_name: obj})
