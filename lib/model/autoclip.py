""" Auto clipper for clipping gradients. """
from __future__ import annotations

import logging
import typing as T

import numpy as np
import torch

from lib.logger import parse_class_init
from lib.utils import get_module_objects

if T.TYPE_CHECKING:
    from keras import KerasTensor

logger = logging.getLogger(__name__)


class AutoClipper():
    """ AutoClip: Adaptive Gradient Clipping for Source Separation Networks

    Parameters
    ----------
    clip_percentile: int
        The percentile to clip the gradients at
    history_size: int, optional
        The number of iterations of data to use to calculate the norm Default: ``10000``

    References
    ----------
    Adapted from: https://github.com/pseeth/autoclip
    original paper: https://arxiv.org/abs/2007.14469
    """
    def __init__(self, clip_percentile: int, history_size: int = 10000) -> None:
        logger.debug(parse_class_init(locals()))

        self._clip_percentile = clip_percentile
        self._history_size = history_size
        self._grad_history: list[float] = []

        logger.debug("Initialized %s", self.__class__.__name__)

    def __call__(self, gradients: list[KerasTensor]) -> list[KerasTensor]:
        """ Call the AutoClip function.

        Parameters
        ----------
        gradients: list[:class:`keras.KerasTensor`]
            The list of gradient tensors for the optimizer

        Returns
        ----------
        list[:class:`keras.KerasTensor`]
            The autoclipped gradients
        """
        self._grad_history.append(sum(g.data.norm(2).item() ** 2
                                      for g in gradients if g is not None) ** (1. / 2))
        self._grad_history = self._grad_history[-self._history_size:]
        clip_value = np.percentile(self._grad_history, self._clip_percentile)
        torch.nn.utils.clip_grad_norm_(gradients, T.cast(float, clip_value))
        return gradients


__all__ = get_module_objects(__name__)
