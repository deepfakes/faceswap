#!/usr/bin/env python3
""" Original Trainer """
from __future__ import annotations
import logging
import typing as T
import warnings

from keras import ops
import torch


from lib.utils import get_module_objects
from .original import Trainer as OriginalTrainer

if T.TYPE_CHECKING:
    from plugins.train.model._base import ModelBase
    import keras

logger = logging.getLogger(__name__)


class WrappedModel(torch.nn.Module):
    """ A torch module that wraps a dual input Faceswap model with a single input version that is
    compatible with DataParallel training

    Parameters
    ----------
    model : :class:`keras.Model`
        The original faceswap model that is to be wrapped
    """
    def __init__(self, model: keras.Model):
        logger.debug("Wrapping keras model: %s", model.name)
        super().__init__()
        self._keras_model = model
        logger.debug("Wrapped keras model: %s (%s)", model.name, self)

    def forward(self,
                input_a: torch.Tensor,
                input_b: torch.Tensor,
                targets_a: torch.Tensor,
                targets_b: torch.Tensor,
                *targets: torch.Tensor) -> torch.Tensor:
        """ Run the forward pass per GPU

        Parameters
        ----------
        input_a : :class:`torch.Tensor`
            The A batch of input images for 1 GPU
        input_b : :class:`torch.Tensor`
            The B batch of input images for 1 GPU
        targets_a : :class:`torch.Tensor` | list[torch.Tensor]
            The A batch of target images for 1 GPU. If this is a multi-output model then this list
            will be the target images per output for all items in the current batch, regardless of
            GPU. If we have 1 output, this will be a Tensor for this GPUs current batch output
        targets_b : :class:`torch.Tensor` | list[torch.Tensor]
            The B batch of target images for 1 GPU. If this is a multi-output model then this list
            will be the target images per output for all items in the current batch, regardless of
            GPU. If we have 1 output, this will be a Tensor for this GPUs current batch output
        targets : :class:`torch.Tensor` | list[torch.Tensor], optional
            Used for multi-output models. Any additional outputs can be added here. They should be
            added in A-B order


        Returns
        -------
        :class:`torch.Tensor`
            The loss outputs for each side of the model for 1 GPU
        """
        preds = self._keras_model((input_a, input_b), training=True)
        self._keras_model.zero_grad()

        if targets:  # Go from [A1, B1, A2, B2, A3, B3] to [A1, A2, A3, B1, B2, B3]
            all_targets = [targets_a, targets_b, *targets]
            assert len(all_targets) % 2 == 0
            loss_targets = all_targets[0::2] + all_targets[1::2]
        else:
            loss_targets = [targets_a, targets_b]

        losses = torch.stack([loss_fn(y_true, y_pred)
                              for loss_fn, y_true, y_pred in zip(self._keras_model.loss,
                                                                 loss_targets,
                                                                 preds)])
        logger.trace("Losses: %s", losses)  # type:ignore[attr-defined]
        return losses


class Trainer(OriginalTrainer):
    """ Distributed training with torch.nn.DataParallel

    Parameters
    ----------
    model : plugin from :mod:`plugins.train.model`
        The model that will be running this trainer
    batch_size : int
        The requested batch size for iteration to be trained through the model.
    """
    def __init__(self, model: ModelBase, batch_size: int) -> None:

        self._gpu_count = torch.cuda.device_count()
        batch_size = self._validate_batch_size(batch_size)
        self._is_multi_out: bool | None = None

        super().__init__(model, batch_size)

        self._distributed_model = self._set_distributed()

    def _validate_batch_size(self, batch_size: int) -> int:
        """ Validate that the batch size is suitable for the number of GPUs and update accordingly.

        Parameters
        ----------
        batch_size : int
            The requested training batch size

        Returns
        -------
        int
            A valid batch size for the GPU configuration
        """
        if batch_size < self._gpu_count:
            logger.warning("Batch size (%s) is less than the number of GPUs (%s). Updating batch "
                           "size to: %s", batch_size, self._gpu_count, self._gpu_count)
            batch_size = self._gpu_count
        if batch_size % self._gpu_count:
            new_batch_size = (batch_size // self._gpu_count) * self._gpu_count
            logger.warning("Batch size %s is sub-optimal for %s GPUs. You may want to adjust your "
                           "batch size to %s or %s.",
                           batch_size,
                           self._gpu_count,
                           new_batch_size,
                           new_batch_size + self._gpu_count)
        return batch_size

    def _handle_torch_gpu_mismatch_warning(
            self, warn_messages: list[warnings.WarningMessage] | None) -> None:
        """ Handle the warning generated by Torch when significantly mismatched GPUs are used and
        remove potentially confusing information not relevant for Faceswap

        Parameters
        ----------
        warn_messages : list[:class:`warnings.WarningMessage]
            Any qualifying warning messages that may have been generated when wrapping the model
        """
        if warn_messages is None or not warn_messages:
            return
        warn_msg = warn_messages[0]
        terminate = "You can do so by"
        msg = ""
        for x in str(warn_msg.message).split("\n"):
            x = x.strip()
            if not x:
                continue
            if terminate in msg:
                msg = msg[:msg.find(terminate)]
                break
            msg += f" {x}"
        logger.warning(msg.strip())

    def _set_distributed(self) -> torch.nn.DataParallel:
        """Wrap the loaded model in a torch.nn.DataParallel instance

        Returns
        -------
        :class:`torch.nn.Parallel`
            A wrapped version of the faceswap model compatible with distributed training
        """
        name = self.model.model.name
        logger.debug("Setting distributed training for '%s'", name)

        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings("default",
                                    message="There is an imbalance between your GPUs",
                                    category=UserWarning)
            # We already set CUDA_VISIBLE_DEVICES from -X command line flag, so just need to wrap
            wrapped = torch.nn.DataParallel(WrappedModel(model=self.model.model))
            self._handle_torch_gpu_mismatch_warning(w)

        logger.info("Distributed training enabled. Model: '%s', devices: %s",
                    name, wrapped.device_ids)
        return wrapped

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
        if self._is_multi_out is None:
            self._is_multi_out = len(targets) > 1
            logger.debug("Setting multi-out to: %s", self._is_multi_out)

        if self._is_multi_out:
            multi_targets = tuple(t[i] for t in targets[1:] for i in range(2))
        else:
            multi_targets = ()

        loss: torch.Tensor = self._distributed_model(inputs[0],
                                                     inputs[1],
                                                     targets[0][0],
                                                     targets[0][1],
                                                     *multi_targets)
        scaled = T.cast(torch.Tensor, ops.sum(ops.reshape(loss, (self._gpu_count, 2, -1)),
                                              axis=0) / self._gpu_count)
        return scaled.flatten()


__all__ = get_module_objects(__name__)
