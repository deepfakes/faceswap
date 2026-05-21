"""Auto clipper for clipping gradients."""
from __future__ import annotations

import logging
import math
from collections import deque

import numpy as np
import torch
from torch import nn

from lib.logger import parse_class_init
from lib.utils import get_module_objects

logger = logging.getLogger(__name__)


class AutoClipper():
    """AutoClip: Adaptive Gradient Clipping for Source Separation Networks

    Parameters
    ----------
    clip_percentile
        The percentile to clip the gradients at
    history_size
        The number of iterations of data to use to calculate the norm Default: ``10000``

    References
    ----------
    Adapted from: https://github.com/pseeth/autoclip
    original paper: https://arxiv.org/abs/2007.14469
    """
    def __init__(self, clip_percentile: int, history_size: int = 10000) -> None:
        logger.debug(parse_class_init(locals()))
        self._clip_percentile = clip_percentile
        self._grad_history: deque[float] = deque(maxlen=history_size)

    def __call__(self, parameters: list[nn.Parameter], *args) -> None:
        """Call the AutoClip function.

        Parameters
        ----------
        parameters
            The parameters to clip
        args
            Unused but for compatibility
        """
        with torch.no_grad():
            norms = [p.grad.norm(2).item() for p in parameters if p.grad is not None]

        if not norms:
            return

        global_norm = sum(n ** 2 for n in norms) ** 0.5
        if not math.isfinite(global_norm):
            return

        self._grad_history.append(global_norm)
        clip_value = float(np.percentile(self._grad_history, self._clip_percentile))
        nn.utils.clip_grad_norm_(parameters, clip_value)


__all__ = get_module_objects(__name__)
