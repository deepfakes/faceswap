#!/usr/bin/env python3
"""Base Class for Faceswap Trainer plugins. All Trainer plugins should be inherited from
this class.

At present there is only the :class:`~plugins.train.trainer.original` plugin, so that entirely
inherits from this class. If further plugins are developed, then common code should be kept here,
with "original" unique code split out to the original plugin.
"""
from __future__ import annotations
import abc
import logging
import typing as T
from dataclasses import dataclass

import torch

if T.TYPE_CHECKING:
    from lib.training.data import BatchMeta
    from lib.training.loss import LossCollator, BatchLoss
    from plugins.train.model._base import ModelBase

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    """Configuration for training a model

    Parameters
    ----------
    image_folders
        List of folders to be used as inputs to the model. Folders are provided in processing order
        (eg: [A, B, ...])
    batch_size
        The batch size to load data from each of the loaders
    augment_color
        ``True`` to perform color augmentation otherwise ``False``
    flip
        ``True`` to perform image flipping otherwise ``False``
    warp
        ``False`` to disable warping ``True`` to enable warping
    cache_landmarks
        ``True`` to cache landmarks from the other side for Warp to landmarks
    use_lr_finder
        ``True`` to use the learning rate finder. Default: ``False``
    snapshot interval
        The number of iterations between snapshots. Default -1 (Disabled)
    """
    folders: list[str]
    """List of folders to be used as inputs to the model. Folders are provided in processing order
    (eg: [A, B, ...])"""
    batch_size: int
    """The batch size to load data from each of the loaders"""
    augment_color: bool
    """``True`` to perform color augmentation otherwise ``False``"""
    flip: bool
    """``False`` to disable warping ``True`` to enable warping"""
    warp: bool
    """``False`` to disable warping ``True`` to enable warping"""
    cache_landmarks: bool
    """``True`` to cache landmarks from the other side for Warp to landmarks"""
    lr_finder: bool = False
    """``True`` to use the learning rate finder"""
    snapshot_interval: int = -1
    """The number of iterations between snapshots"""


class TrainerBase(abc.ABC):
    """A trainer plugin interface. It must implement the method "train_batch" which takes an input
    of inputs to the model and target images for model output. It returns loss per side

    Parameters
    ----------
    model
        The model plugin
    config
        The Training Configuration options
    """
    def __init__(self, model: ModelBase, config: TrainConfig) -> None:
        self.model = model
        """The model plugin to train the batch on"""
        self.batch_size = config.batch_size
        """The batch size for each iteration to be trained through the model."""
        self.config = config
        """Training configuration options"""
        self.sampler = self.get_sampler()
        """The data sampler that the data loader should use"""
        self.loss_func: LossCollator
        """The selected loss functions for the model"""

    def __repr__(self) -> str:
        """Pretty print for logging"""
        params = f"model={repr(self.model)}, config={repr(self.config)}"
        return f"{self.__class__.__name__}({params})"

    def register_loss(self, loss: LossCollator) -> None:
        """Registers the selected loss functions to the underlying model nn.module

        Parameters
        ----------
        loss
            The configured loss functions
        """
        logger.debug("[%s] Registering loss: %s", self.__class__.__name__, loss)
        self.model.model.add_module("loss_func", loss)
        self.loss_func = loss

    @abc.abstractmethod
    def get_sampler(self) -> type[torch.utils.data.RandomSampler |
                                  torch.utils.data.DistributedSampler]:
        """Override to set the sampler that the Torch DataLoader should use

        Returns
        -------
        The sampler that the torch DataLoader should use
        """

    @abc.abstractmethod
    def train_batch(self,
                    inputs: list[torch.Tensor],
                    targets: list[torch.Tensor],
                    meta: BatchMeta) -> list[BatchLoss]:
        """Override to run a single forward and backwards pass through the model for a single batch

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
