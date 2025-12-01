#!/usr/bin/env python3
""" Custom Optimizers for Torch/keras """
from __future__ import annotations
import inspect
import logging
import sys
import typing as T

from keras import ops, Optimizer, saving

from lib.logger import parse_class_init
from lib.utils import get_module_objects

if T.TYPE_CHECKING:
    from keras import KerasTensor, Variable

logger = logging.getLogger(__name__)


class AdaBelief(Optimizer):  # pylint:disable=too-many-instance-attributes,too-many-ancestors
    """ Implementation of the AdaBelief Optimizer

    Inherits from: keras.optimizers.Optimizer.

    AdaBelief Optimizer is not a placement of the heuristic warmup, the settings should be kept if
    warmup has already been employed and tuned in the baseline method. You can enable warmup by
    setting `total_steps` and `warmup_proportion` (see examples)

    Lookahead (see references) can be integrated with AdaBelief Optimizer, which is announced by
    Less Wright and the new combined optimizer can also be called "Ranger". The mechanism can be
    enabled by using the lookahead wrapper. (See examples)

    Parameters
    ----------
    learning_rate: `Tensor`, float or :class: `keras.optimizers.schedules.LearningRateSchedule`
        The learning rate.
    beta_1: float
        The exponential decay rate for the 1st moment estimates.
    beta_2: float
        The exponential decay rate for the 2nd moment estimates.
    epsilon: float
        A small constant for numerical stability.
    amsgrad: bool
        Whether to apply AMSGrad variant of this algorithm from the paper "On the Convergence
        of Adam and beyond".
    rectify: bool
        Whether to enable rectification as in RectifiedAdam
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
        Standard Keras Optimizer keyword arguments. Allowed to be (`weight_decay`, `clipnorm`,
        `clipvalue`, `global_clipnorm`, `use_ema`, `ema_momentum`, `ema_overwrite_frequency`,
        `loss_scale_factor`, `gradient_accumulation_steps`)

    Examples
    --------
    >>> from optimizers import AdaBelief
    >>> opt = AdaBelief(lr=1e-3)

    Example of serialization:

    >>> optimizer = AdaBelief(learning_rate=lr_scheduler, weight_decay=wd_scheduler)
    >>> config = keras.optimizers.serialize(optimizer)
    >>> new_optimizer = keras.optimizers.deserialize(config,
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

    def __init__(self,  # pylint:disable=too-many-arguments,too-many-positional-arguments
                 learning_rate: float = 0.001,
                 beta_1: float = 0.9,
                 beta_2: float = 0.999,
                 epsilon: float = 1e-14,
                 amsgrad: bool = False,
                 rectify: bool = True,
                 sma_threshold: float = 5.0,
                 total_steps: int = 0,
                 warmup_proportion: float = 0.1,
                 min_learning_rate: float = 0.0,
                 name="AdaBeliefOptimizer",
                 **kwargs):
        logger.debug(parse_class_init(locals()))
        super().__init__(learning_rate=learning_rate, name=name, **kwargs)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.amsgrad = amsgrad
        self.rectify = rectify
        self.sma_threshold = sma_threshold
        # TODO change the following 2 to "warm_up_steps"
        # TODO Make learning rate warm up a global option
        # Or these params can be calculated from a user "warm_up_steps" parameter
        self.total_steps = total_steps
        self.warmup_proportion = warmup_proportion
        self.min_learning_rate = min_learning_rate
        logger.debug("Initialized %s", self.__class__.__name__)

        self._momentums: list[Variable] = []
        self._velocities: list[Variable] = []
        self._velocity_hats: list[Variable] = []  # Amsgrad only

    def build(self, variables: list[Variable]) -> None:
        """Initialize optimizer variables.

        AdaBelief optimizer has 3 types of variables: momentums, velocities and
        velocity_hat (only set when amsgrad is applied),

        Parameters
        ----------
        variables: list[:class:`keras.Variable`]
            list of model variables to build AdaBelief variables on.
        """
        if self.built:
            return
        logger.debug("Building AdaBelief. var_list: %s", variables)
        super().build(variables)

        for var in variables:
            self._momentums.append(self.add_variable_from_reference(
                    reference_variable=var, name="momentum"))
            self._velocities.append(self.add_variable_from_reference(
                    reference_variable=var, name="velocity"))
            if self.amsgrad:
                self._velocity_hats.append(self.add_variable_from_reference(
                        reference_variable=var, name="velocity_hat"))
        logger.debug("Built AdaBelief. momentums: %s, velocities: %s, velocity_hats: %s",
                     len(self._momentums), len(self._velocities), len(self._velocity_hats))

    def _maybe_warmup(self, learning_rate: KerasTensor, local_step: KerasTensor) -> KerasTensor:
        """ Do learning rate warm up if requested

        Parameters
        ----------
        learning_rate: :class:`keras.KerasTensor`
            The learning rate
        local_step: :class:`keras.KerasTensor`
            The current training step

        Returns
        -------
        :class:`keras.KerasTensor`
            Either the original learning rate or adjusted learning rate if warmup is requested
        """
        if self.total_steps <= 0:
            return learning_rate

        total_steps = ops.cast(self.total_steps, learning_rate.dtype)
        warmup_steps = total_steps * ops.cast(self.warmup_proportion, learning_rate.dtype)
        min_lr = ops.cast(self.min_learning_rate, learning_rate.dtype)
        decay_steps = ops.maximum(total_steps - warmup_steps, 1)
        decay_rate = ops.divide(min_lr - learning_rate, decay_steps)
        return ops.where(local_step <= warmup_steps,
                         ops.multiply(learning_rate, (ops.divide(local_step, warmup_steps))),
                         ops.multiply(learning_rate + decay_rate,
                                      ops.minimum(local_step - warmup_steps, decay_steps)))

    def _maybe_rectify(self,
                       momentum: KerasTensor,
                       velocity: KerasTensor,
                       local_step: KerasTensor,
                       beta_2_power: KerasTensor) -> KerasTensor:
        """ Apply rectification, if requested

        Parameters
        ----------
        momentum: :class:`keras.KerasTensor`
            The momentum update
        velocity: :class:`keras.KerasTensor`
            The velocity update
        local_step: :class:`keras.KerasTensor`
            The current training step
        beta_2_power
            Adjusted exponential decay rate for the 2nd moment estimates.

        Returns
        -------
        :class:`keras.KerasTensor`
            The standard or rectified update (if rectification enabled)
        """
        if not self.rectify:
            return ops.divide(momentum, ops.add(velocity, self.epsilon))

        sma_inf = 2 / (1 - self.beta_2) - 1
        sma_t = sma_inf - 2 * local_step * beta_2_power / (1 - beta_2_power)
        rect = ops.sqrt((sma_t - 4) / (sma_inf - 4) *
                        (sma_t - 2) / (sma_inf - 2) *
                        sma_inf / sma_t)
        return ops.where(sma_t >= self.sma_threshold,
                         ops.divide(
                            ops.multiply(rect, momentum),
                            (ops.add(velocity, self.epsilon))),
                         momentum)

    def update_step(self,
                    gradient: KerasTensor,
                    variable: Variable,
                    learning_rate: Variable) -> None:
        """Update step given gradient and the associated model variable for AdaBelief.

        Parameters
        ----------
        gradient :class:`keras.KerasTensor`
            The gradient to update
        variable: :class:`keras.Variable`
            The variable to update
        learning_rate: :class:`keras.Variable`
            The learning rate
        """
        local_step = ops.cast(self.iterations + 1, variable.dtype)
        learning_rate = self._maybe_warmup(ops.cast(learning_rate, variable.dtype), local_step)
        gradient = ops.cast(gradient, variable.dtype)
        beta_1_power = ops.power(ops.cast(self.beta_1, variable.dtype), local_step)
        beta_2_power = ops.power(ops.cast(self.beta_2, variable.dtype), local_step)

        #     m_t = b1 * m + (1 - b1) * g
        # =>  m_t = m + (g - m) * (1 - b1)
        momentum = self._momentums[self._get_variable_index(variable)]
        self.assign_add(momentum, ops.multiply(ops.subtract(gradient, momentum), 1 - self.beta_1))
        momentum_corr = ops.divide(momentum, (1 - beta_1_power))

        #    v_t = b2 * v + (1 - b2) * (g - m_t)^2 + e
        # => v_t = v + ((g - m_t)^2 - v) * (1 - b2) + e
        velocity = self._velocities[self._get_variable_index(variable)]
        self.assign_add(velocity,
                        ops.multiply(
                            ops.subtract(ops.square(gradient - momentum), velocity),
                            1 - self.beta_2)
                        + self.epsilon)

        if self.amsgrad:
            velocity_hat = self._velocity_hats[self._get_variable_index(variable)]
            self.assign(velocity_hat, ops.maximum(velocity, velocity_hat))
            velocity_corr = ops.sqrt(ops.divide(velocity_hat, (1 - beta_2_power)))
        else:
            velocity_corr = ops.sqrt(ops.divide(velocity, (1 - beta_2_power)))

        var_t = self._maybe_rectify(momentum_corr, velocity_corr, local_step, beta_2_power)

        self.assign_sub(variable, ops.multiply(learning_rate, var_t))

    def get_config(self) -> dict[str, T.Any]:
        """ Returns the config of the optimizer.

        Optimizer configuration for AdaBelief.

        Returns
        -------
        dict[str, Any]
            The optimizer configuration.
        """
        config = super().get_config()
        config.update({"beta_1": self.beta_1,
                       "beta_2": self.beta_2,
                       "epsilon": self.epsilon,
                       "amsgrad": self.amsgrad,
                       "rectify": self.rectify,
                       "sma_threshold": self.sma_threshold,
                       "total_steps": self.total_steps,
                       "warmup_proportion": self.warmup_proportion,
                       "min_learning_rate": self.min_learning_rate})
        return config


# Update Optimizers into Keras custom objects
for _name, obj in inspect.getmembers(sys.modules[__name__]):
    if inspect.isclass(obj) and obj.__module__ == __name__:
        saving.get_custom_objects().update({_name: obj})


__all__ = get_module_objects(__name__)
