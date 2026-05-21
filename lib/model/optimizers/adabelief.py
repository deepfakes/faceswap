#! /usr/env/bin/python3
"""AdaBelief optimizer for Torch"""
# BSD 2-Clause License
#
# Copyright (c) 2021, Juntang Zhuang
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
import logging
import math
import typing as T

import torch
from torch.optim.optimizer import Optimizer

from lib.logger import parse_class_init
from lib.utils import get_module_objects

logger = logging.getLogger(__name__)


class AdaBelief(Optimizer):
    """Implements AdaBelief algorithm. Modified from Adam in PyTorch

    Parameters
    ----------
    params
        Iterable of parameters to optimize or dicts defining parameter groups
    lr
        Learning rate. Default: 1e-3
    betas
        Coefficients used for computing running averages of gradient and its square.
        Default: (0.9, 0.999)
    eps
        Term added to the denominator to improve numerical stability. Default: 1e-16
    weight_decay
        Weight decay (L2 penalty). Default: 0
    amsgrad
        Whether to use the AMSGrad variant of this algorithm from the paper `On the Convergence
        of Adam and Beyond`. Default: ``False``
    weight_decouple
        If set as True, then the optimizer uses decoupled weight decay as in AdamW.
        Default: ``True``
    fixed_decay
        This is used when weight_decouple is set as True.
        - When fixed_decay == True, the weight decay is performed as W_{new} = W_{old} - W_{old}
        * decay.
        - When fixed_decay == False, the weight decay is performed as W_{new} = W_{old} - W_{old}
        * decay * lr. Note that in this case, the weight decay ratio decreases with learning rate
        (lr).
        Default: ``False``
    rectify
        If set as True, then perform the rectified update similar to RAdam.
        Default: ``True``
    degenerated_to_sgd
        If set as True, then perform SGD update when variance of gradient is high.
        Default: ``True``

    Reference
    ---------
    AdaBelief Optimizer, adapting step sizes by the belief in observed gradients, NeurIPS 2020
    https://github.com/juntang-zhuang/Adabelief-Optimizer
    """
    def __init__(self,  # pylint:disable=too-many-positional-arguments,too-many-arguments  # noqa[C901]
                 params: T.Iterable,
                 lr: float = 1e-3,
                 betas: tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-16,
                 weight_decay: float = 0.0,
                 amsgrad: bool = False,
                 weight_decouple: bool = True,
                 fixed_decay: bool = False,
                 rectify: bool = True,
                 degenerated_to_sgd: bool = True) -> None:
        logger.debug(parse_class_init(locals()))
        if 0.0 > lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if 0.0 > eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")

        self.degenerated_to_sgd = degenerated_to_sgd
        if isinstance(params, (list, tuple)) and len(params) > 0 and isinstance(params[0],
                                                                                dict):
            for param in params:
                if "betas" in param and (param["betas"][0] != betas[0]
                                         or param["betas"][1] != betas[1]):
                    param["buffer"] = [[None, None, None] for _ in range(10)]

        defaults = {"lr": lr,
                    "betas": betas,
                    "eps": eps,
                    "weight_decay": weight_decay,
                    "amsgrad": amsgrad,
                    "buffer": [[None, None, None] for _ in range(10)]}
        super().__init__(params, defaults)

        self.degenerated_to_sgd = degenerated_to_sgd
        self.weight_decouple = weight_decouple
        self.rectify = rectify
        self.fixed_decay = fixed_decay
        if self.weight_decouple:
            logger.debug("[AdaBelief] Weight decoupling enabled in AdaBelief")
            if self.fixed_decay:
                logger.debug("[AdaBelief] Weight decay fixed")
        if self.rectify:
            logger.debug("[AdaBelief] Rectification enabled in AdaBelief")
        if amsgrad:
            logger.debug("[AdaBelief] AMSGrad enabled in AdaBelief")

    def __setstate__(self, state: dict[str, T.Any]) -> None:
        """Set parameter state"""
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("amsgrad", False)

    def reset(self) -> None:
        """Reset parameters"""
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                amsgrad = group["amsgrad"]

                # State initialization
                state["step"] = torch.zeros((), dtype=torch.float32)
                # Exponential moving average of gradient values
                state["exp_avg"] = torch.zeros_like(p.data, memory_format=torch.preserve_format)

                # Exponential moving average of squared gradient values
                state["exp_avg_var"] = torch.zeros_like(p.data,
                                                        memory_format=torch.preserve_format)

                if amsgrad:
                    # Maintains max of all exp. moving avg. of sq. grad. values
                    state["max_exp_avg_var"] = torch.zeros_like(
                        p.data, memory_format=torch.preserve_format)

    def step(self,  # type:ignore[override]  # noqa[C901]
             closure: T.Callable | None = None) -> torch.Tensor:
        """Performs a single optimization step.

        Parameters
        ----------
        closure
            A closure that reevaluates the model and returns the loss. Default: ``None``
        """
        # pylint:disable=duplicate-code,too-many-statements,too-many-branches,too-many-locals
        loss: torch.Tensor | None = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                # cast data type
                half_precision = False
                if p.data.dtype == torch.float16:
                    half_precision = True
                    p.data = p.data.float()
                    p.grad = p.grad.float()

                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        "AdaBelief does not support sparse gradients, please consider SparseAdam "
                        "instead")
                amsgrad = group["amsgrad"]

                state = self.state[p]

                beta1, beta2 = group["betas"]

                # State initialization
                if len(state) == 0:
                    state["step"] = torch.zeros((), dtype=torch.float32)
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p.data,
                                                        memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_var"] = torch.zeros_like(p.data,
                                                            memory_format=torch.preserve_format)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state["max_exp_avg_var"] = torch.zeros_like(
                            p.data, memory_format=torch.preserve_format)

                # perform weight decay, check if decoupled weight decay
                if self.weight_decouple:
                    if not self.fixed_decay:
                        p.data.mul_(1.0 - group["lr"] * group["weight_decay"])
                    else:
                        p.data.mul_(1.0 - group["weight_decay"])
                else:
                    if group["weight_decay"] != 0:
                        grad.add_(p.data, alpha=group["weight_decay"])

                # get current state variable
                exp_avg, exp_avg_var = state["exp_avg"], state["exp_avg_var"]

                state["step"] += 1
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                # Update first and second moment running average
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                grad_residual = grad - exp_avg
                exp_avg_var.mul_(beta2).addcmul_(grad_residual, grad_residual, value=1 - beta2)

                if amsgrad:
                    max_exp_avg_var = state["max_exp_avg_var"]
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_var, exp_avg_var.add_(group["eps"]), out=max_exp_avg_var)

                    # Use the max. for normalizing running avg. of gradient
                    denom = (max_exp_avg_var.sqrt() /
                             math.sqrt(bias_correction2)).add_(group["eps"])
                else:
                    denom = (exp_avg_var.add_(group["eps"]).sqrt() /
                             math.sqrt(bias_correction2)).add_(group["eps"])

                # update
                if not self.rectify:
                    # Default update
                    step_size = group["lr"] / bias_correction1
                    p.data.addcdiv_(exp_avg, denom, value=-step_size)

                else:  # Rectified update, forked from RAdam
                    buffered = group["buffer"][int(state["step"] % 10)]
                    if state["step"] == buffered[0]:
                        n_sma, step_size = buffered[1], buffered[2]
                    else:
                        buffered[0] = state["step"]
                        beta2_t = beta2 ** state["step"]
                        n_sma_max = 2 / (1 - beta2) - 1
                        n_sma = n_sma_max - 2 * state["step"] * beta2_t / (1 - beta2_t)
                        buffered[1] = n_sma

                        # more conservative since it"s an approximated value
                        if n_sma >= 5:
                            step_size = math.sqrt(
                                (1 - beta2_t) * (n_sma - 4) /
                                (n_sma_max - 4) * (n_sma - 2) /
                                n_sma * n_sma_max / (n_sma_max - 2)) / (1 - beta1 ** state["step"])
                        elif self.degenerated_to_sgd:
                            step_size = 1.0 / (1 - beta1 ** state["step"])
                        else:
                            step_size = -1
                        buffered[2] = step_size

                    if n_sma >= 5:
                        denom = exp_avg_var.sqrt().add_(group["eps"])
                        p.data.addcdiv_(exp_avg, denom, value=-step_size * group["lr"])
                    elif step_size > 0:
                        p.data.add_(exp_avg, alpha=-step_size * group["lr"])

                if half_precision:
                    p.data = p.data.half()
                    p.grad = p.grad.half()

        return T.cast(torch.Tensor, loss)


__all__ = get_module_objects(__name__)
