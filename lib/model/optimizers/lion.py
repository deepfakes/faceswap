#! /usr/env/bin/python3
"""PyTorch implementation of the Lion optimizer."""
#  Copyright 2023 Google Research. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import logging
import typing as T

import torch
from torch.optim.optimizer import Optimizer

from lib.logger import parse_class_init
from lib.utils import get_module_objects

logger = logging.getLogger(__name__)


class Lion(Optimizer):
    """Lion optimizer from Google

    Parameters
    ----------
    params
        Iterable of parameters to optimize or dicts defining parameter groups
    lr
        Learning rate. Default: 1e-4
    betas
        Coefficients used for computing running averages of gradient and its square.
        Default: (0.9, 0.99)
    weight_decay
        Weight decay coefficient. Default: 0

    Reference
    ---------
    https://github.com/google/automl/blob/master/lion/lion_pytorch.py
    """
    def __init__(self,
                 params: T.Iterable,
                 lr: float = 1e-4,
                 betas: tuple[float, float] = (0.9, 0.99),
                 weight_decay: float = 0.0) -> None:
        logger.debug(parse_class_init(locals()))
        if 0.0 > lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        defaults = {"lr": lr, "betas": betas, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: T.Callable | None = None) -> torch.Tensor:  # type:ignore[override]
        """Performs a single optimization step.

        Parameters
        ----------
        closure
            A closure that reevaluates the model and returns the loss.

        Returns
        -------
        The loss
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                # Perform step weight decay
                p.data.mul_(1 - group["lr"] * group["weight_decay"])

                grad = p.grad
                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p)

                exp_avg = state["exp_avg"]
                beta1, beta2 = group["betas"]

                # Weight update
                update = exp_avg * beta1 + grad * (1 - beta1)

                p.add_(update.sign_(), alpha=-group["lr"])

                # Decay the momentum running average coefficient
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)

        return T.cast(torch.Tensor, loss)


__all__ = get_module_objects(__name__)
