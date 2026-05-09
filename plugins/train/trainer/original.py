#!/usr/bin/env python3
"""Original Trainer """
from __future__ import annotations

import logging
import typing as T

from keras import ops
import torch

from lib.utils import get_module_objects
from .base import TrainerBase

if T.TYPE_CHECKING:
    from lib.training.data import BatchMeta
    from lib.training.loss import BatchLoss


logger = logging.getLogger(__name__)


class Trainer(TrainerBase):
    """Original trainer"""

    def get_sampler(self) -> type[torch.utils.data.RandomSampler]:
        """Obtain a standard random sampler

        Returns
        -------
        The Random sampler
        """
        return torch.utils.data.RandomSampler

    def _forward(self,
                 inputs: list[torch.Tensor],
                 targets: list[torch.Tensor],
                 meta: BatchMeta) -> list[BatchLoss]:
        """Perform the forward pass on the model

        Parameters
        ----------
        inputs
            The batch of input image tensors to the model of length(num inputs)
        targets
            List of len (num_outputs) of target images in shape (batch_size, num_inputs, height,
            width, 3) at all model output sizes as float32 0.0 - 1.0 range
        meta
            The meta information for the batch

        Returns
        -------
        The loss for each input to the model in order (A, B, ...)
        """
        predictions = self.model.model(inputs, training=True)
        num_sides = len(inputs)
        num_outputs = len(predictions) // num_sides
        losses = [self.loss_func([t[:, i] for t in targets],
                                 predictions[i * num_outputs:i * num_outputs + num_outputs],
                                 meta[i])
                  for i in range(num_sides)]
        logger.trace("Losses: %s", losses)  # type:ignore[attr-defined]
        return losses

    def _backwards_and_apply(self, all_loss: torch.Tensor) -> None:
        """Perform the backwards pass on the model

        Parameters
        ----------
        all_loss
            The loss for each output from the model
        """
        total_loss = T.cast(torch.Tensor,
                            self.model.model.optimizer.scale_loss(ops.sum(all_loss)))
        total_loss.backward()

        trainable_weights = self.model.model.trainable_weights[:]
        gradients = [v.value.grad for v in trainable_weights]

        # Update weights
        with torch.no_grad():
            self.model.model.optimizer.apply(gradients, trainable_weights)

    def train_batch(self,
                    inputs: list[torch.Tensor],
                    targets: list[torch.Tensor],
                    meta: BatchMeta) -> list[BatchLoss]:
        """Run a single forward and backwards pass through the model for a single batch

        Parameters
        ----------
        inputs
            The batch of input image tensors to the model of length(num inputs)
        targets
            List of len (num_outputs) of target images in shape (batch_size, num_inputs, height,
            width, 3) at all model output sizes as float32 0.0 - 1.0 range
        meta
            The meta information for the batch

        Returns
        -------
        The loss for each input to the model in order (A, B, ...)
        """
        self.model.model.zero_grad()  # TODO move this to optimizer
        loss = self._forward(inputs, targets, meta)
        total_loss = T.cast(torch.Tensor, sum(x.total for x in loss))
        self._backwards_and_apply(total_loss)
        return loss


__all__ = get_module_objects(__name__)
