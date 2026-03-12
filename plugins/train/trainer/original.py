#!/usr/bin/env python3
""" Original Trainer """
from __future__ import annotations

import logging
import typing as T

from keras import ops
from keras.src.tree import flatten
import torch

from lib.utils import get_module_objects
from ._base import TrainerBase


logger = logging.getLogger(__name__)


class Trainer(TrainerBase):
    """ Original trainer """

    def _forward(self,
                 inputs: torch.Tensor,
                 targets: list[torch.Tensor]) -> torch.Tensor:
        """ Perform the forward pass on the model

        Parameters
        ----------
        inputs : :class:`torch.Tensor`
            The batch of input image tensors to the model in shape `(side, batch_size,
            *dims)` with `side` 0 being input A and `side` 1 being input B
        targets : list[:class:`torch.Tensor`]
            The corresponding batch of target images for the model for each side's output(s). For
            each model output an array should exist in the order of model outputs in the format `(
            side, batch_size, *dims)` with `side` 0 being input A and `side` 1 being input B

        Returns
        -------
        :class:`torch.Tensor`
            The loss for each side of this batch in layout (A1, ..., An, B1, ..., Bn)
        """
        feed_targets = [[t[i] for t in targets] for i in range(2)]
        preds = self.model.model((inputs[0], inputs[1]), training=True)
        self.model.model.zero_grad()

        losses = torch.stack([loss_fn(y_true, y_pred)
                              for loss_fn, y_true, y_pred in zip(self.model.model.loss,
                                                                 flatten(feed_targets),
                                                                 preds)])
        logger.trace("Losses: %s", losses)  # type:ignore[attr-defined]
        return losses

    def _backwards_and_apply(self, all_loss: torch.Tensor) -> None:
        """ Perform the backwards pass on the model

        Parameters
        ----------
        all_loss : :class:`torch.Tensor`
            The loss for each output from the model
        """
        total_loss = T.cast(torch.Tensor,
                            self.model.model.optimizer.scale_loss(ops.sum(all_loss)))
        total_loss.backward()

        trainable_weights = self.model.model.trainable_weights[:]
        gradients = [v.value.grad for v in trainable_weights]

        # Update weights
        with torch.inference_mode():
            self.model.model.optimizer.apply(gradients, trainable_weights)

    def train_batch(self,
                    inputs: torch.Tensor,
                    targets: list[torch.Tensor]) -> torch.Tensor:
        """Run a single forward and backwards pass through the model for a single batch

        Parameters
        ----------
        inputs : :class:`torch.Tensor`
            The batch of input image tensors to the model in shape `(side, batch_size,
            *dims)` with `side` 0 being input A and `side` 1 being input B
        targets : list[:class:`torch.Tensor`]
            The corresponding batch of target images for the model for each side's output(s). For
            each model output an array should exist in the order of model outputs in the format `(
            side, batch_size, *dims)` with `side` 0 being input A and `side` 1 being input B

        Returns
        -------
        :class:`torch.Tensor`
            The loss for each side of this batch in layout (A1, ..., An, B1, ..., Bn)
        """
        loss_tensor = self._forward(inputs, targets)
        self._backwards_and_apply(loss_tensor)
        return loss_tensor


__all__ = get_module_objects(__name__)
