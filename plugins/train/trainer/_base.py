#!/usr/bin/env python3
""" Base Class for Faceswap Trainer plugins. All Trainer plugins should be inherited from
this class.

At present there is only the :class:`~plugins.train.trainer.original` plugin, so that entirely
inherits from this class. If further plugins are developed, then common code should be kept here,
with "original" unique code split out to the original plugin.
"""
from __future__ import annotations
import abc
import logging
import typing as T

import torch

if T.TYPE_CHECKING:
    from plugins.train.model._base import ModelBase

logger = logging.getLogger(__name__)


class TrainerBase(abc.ABC):
    """ A trainer plugin interface. It must implement the method "train_batch" which takes an input
    of inputs to the model and target images for model output. It returns loss per side

    Parameters
    ----------
    model : :class:`plugins.train.model.Base.ModelBase`
        The model plugin
    batch_size : int
        The requested batch size for each iteration to be trained through the model.
    """
    def __init__(self, model: ModelBase, batch_size: int) -> None:
        self.model = model
        """:class:`plugins.train.model.Base.ModelBase` : The model plugin to train the batch on"""
        self.batch_size = batch_size
        """int : The batch size for each iteration to be trained through the model."""

    @abc.abstractmethod
    def train_batch(self, inputs: torch.Tensor, targets: list[torch.Tensor]) -> torch.Tensor:
        """Override to run a single forward and backwards pass through the model for a single
        batch

        Parameters
        ----------
        inputs : :class:`torch.Tensor`
            The batch of input image tensors to the model in shape `(side, batch_size,
            *dims)` with `side` 0 being input A and `side` 1 being input B
        targets : list[:class:`torch.Tensor`]
            The corresponding batch of target images for the model for each side's output(s). For
            each model output an array should exist in the order of model outputs in the format `(
            side, batch_size, *dims)` where `side` 0 is "A" and `side` 1 is "B"

        Returns
        -------
        :class:`torch.Tensor`
            The loss for each side of this batch in layout (A1, ..., An, B1, ..., Bn)
        """
