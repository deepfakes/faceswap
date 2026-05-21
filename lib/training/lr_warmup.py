#! /usr/env/bin/python3
"""Handles Learning Rate Warmup when training a model"""
from __future__ import annotations

import logging
import typing as T

from torch.optim.lr_scheduler import LRScheduler

from lib.logger import parse_class_init
from lib.utils import get_module_objects

if T.TYPE_CHECKING:
    from torch import Tensor
    from torch.optim import Optimizer

logger = logging.getLogger(__name__)


class WarmupScheduler(LRScheduler):
    """Handles the updating of the model's learning rate during Learning Rate Warmup

    Parameters
    ----------
    optimizer
        The torch optimizer in use
    steps
        The number of iterations to warmup the learning rate for
    last_epoch
        The last step that was run (last_epoch is a misnomer inherited from PyTorch and actually
        refers to steps in our use case). Default: -1 (not yet started)
    """
    def __init__(self, optimizer: Optimizer, steps: int, last_epoch: int = -1) -> None:
        logger.debug(parse_class_init(locals()))
        self.steps = steps
        """The total number of steps to warmup the LR for"""
        self._reporting_points = [int(self.steps * i / 10) for i in range(11)]
        super().__init__(optimizer, last_epoch)

    @classmethod
    def _fmt(cls, value: float) -> str:
        """Format a float to scientific notation at 1 decimal place

        Parameters
        ----------
        value
            The value to format

        Returns
        -------
        The formatted float in scientific notation at 1 decimal place
        """
        return f"{value:.1e}"

    def get_lr(self) -> list[float | Tensor]:
        """Get the learning rate for the current step

        Returns
        -------
        The next learning rate for each parameter group for the next step
        """
        if self.last_epoch >= self.steps:
            return self.base_lrs

        factor = self.last_epoch / self.steps
        lrs = [base_lr * factor for base_lr in self.base_lrs]
        logger.trace("Learning rate set to %s for step %s/%s",  # type:ignore[attr-defined]
                     lrs, self.last_epoch, self.steps)
        return lrs

    def _output_status(self) -> None:
        """Output the progress of Learning Rate Warmup at set intervals"""
        step = self.last_epoch
        if step < 1:
            return

        current_lr = T.cast(float, self.get_last_lr()[0])
        target_lr = T.cast(float, self.base_lrs[0])

        if step == 1:
            logger.info("[Learning Rate Warmup] Start: %s, Target: %s, Steps: %s",
                        self._fmt(current_lr), self._fmt(target_lr), self.steps)
            return

        if step == self.steps:
            print()
            logger.info("[Learning Rate Warmup] Final Learning Rate: %s", self._fmt(target_lr))
            return

        if step in self._reporting_points:
            print()
            progress = int(round(100 / (len(self._reporting_points) - 1) *
                           self._reporting_points.index(step), 0))
            logger.info("[Learning Rate Warmup] Step: %s/%s (%s), Current: %s, Target: %s",
                        step,
                        self.steps,
                        f"{progress}%",
                        self._fmt(current_lr),
                        self._fmt(target_lr))

    def step(self, epoch=None) -> None:
        """If a learning rate update is required, update the model's learning rate, otherwise
        do nothing

        Parameters
        ----------
        epoch
            Deprecated argument from PyTorch that should always be ``None``. Default: ``None``
        """
        super().step(epoch)
        self._output_status()


__all__ = get_module_objects(__name__)
