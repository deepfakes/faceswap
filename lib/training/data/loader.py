#! /usr/env/bin/python3
"""Handles the loading of data for training and previews for faceswap models"""
from __future__ import annotations

import logging
import os
import typing as T

import torch
from torch.utils import data as tch_data
from torch.utils.data import DataLoader
from lib.logger import parse_class_init
from lib.utils import get_module_objects
from plugins.train import train_config as mod_cfg
from plugins.train.trainer import trainer_config as trn_cfg

from .data_set import get_label, TrainSet, PreviewSet, MultiDataset
from .collate import Collate, LandmarkMatcher

if T.TYPE_CHECKING:
    from lib.align.constants import CenteringType
    from plugins.train.trainer.base import TrainConfig
    from .collate import BatchMeta

logger = logging.getLogger(__name__)


class TrainLoader():  # pylint:disable=too-many-instance-attributes
    """Generator for feeding faceswap models with multiple inputs and outputs. Gets the next items
    from each of the configured loaders and collates them for feeding into a model

    Parameters
    ----------
    input_size
        The input size to the model
    output_sizes
        The output sizes to the model (list as some models have multi-scale outputs)
    color_order
        The color order of the model
    config
        The training configuration for feeding the model
    sampler
        The sampler to use for the data loaders. Default: ``None`` (RandomSampler)
    """
    def __init__(self,
                 input_size: int,
                 output_sizes: tuple[int, ...],
                 color_order: T.Literal["bgr", "rgb"],
                 config: TrainConfig,
                 sampler: None | type[tch_data.RandomSampler |
                                      tch_data.DistributedSampler] = None) -> None:
        logger.debug(parse_class_init(locals()))
        self._learn_mask = mod_cfg.Loss.learn_mask()
        self._output_sizes = output_sizes
        self._config = config
        self._process_size = max(*self._output_sizes, input_size)
        self._landmarks: None | LandmarkMatcher = None

        if config.warp and config.cache_landmarks:
            self._landmarks = LandmarkMatcher(config.folders,
                                              self._process_size,
                                              T.cast("CenteringType", mod_cfg.centering()),
                                              mod_cfg.coverage() / 100.,
                                              mod_cfg.vertical_offset() / 100.)

        self._input_size = input_size
        self._color_order: T.Literal["bgr", "rgb"] = T.cast(T.Literal["bgr", "rgb"],
                                                            color_order.lower())
        self._sampler = tch_data.RandomSampler if sampler is None else sampler
        self._loader = self.get_loader()
        self._iterator = T.cast(T.Iterator[tuple[list[torch.Tensor],
                                                 list[torch.Tensor],
                                                 "BatchMeta"]],
                                iter(self._loader))
        self._epoch = 0

    def __iter__(self) -> T.Self:
        """This is an iterator"""
        return self

    def __repr__(self) -> str:
        """Pretty print for logging"""
        params = {f"{k}"[1:]: v for k, v in self.__dict__.items()
                  if k in ("_input_size", "_output_sizes", "_color_order", "_config", "_sampler")}
        s_params = ", ".join(f"{k}={repr(v)}" for k, v in params.items())
        return f"{self.__class__.__name__}({s_params})"

    def get_loader(self) -> DataLoader:
        """Obtain the dataloaders for each input/output for the model

        Returns
        -------
        The Training data loaders in side order
        """
        num_workers = trn_cfg.Loader.num_processes()
        max_proc = os.cpu_count()
        max_proc = 1 if max_proc is None else max_proc
        if num_workers > max_proc:
            logger.warning("Data Loader processes set to %s but only %s processors available. "
                           "Lowering to %s", num_workers, max_proc, max_proc - 1)
            num_workers = max_proc - 1

        data_sets = tuple(TrainSet(get_label(i, len(self._config.folders)), f, self._process_size)
                          for i, f in enumerate(self._config.folders))
        train_set = MultiDataset(data_sets, is_random=True)
        collate_fn = Collate(self._input_size,
                             self._output_sizes,
                             self._color_order,
                             self._config,
                             landmarks=self._landmarks)
        retval = DataLoader(dataset=train_set,
                            batch_size=self._config.batch_size,
                            sampler=self._sampler(train_set),
                            num_workers=num_workers,
                            prefetch_factor=trn_cfg.Loader.pre_fetch(),
                            collate_fn=collate_fn,
                            pin_memory=True,
                            drop_last=True)
        logger.debug("[TrainLoader] Set loader: %s", retval)
        return retval

    def __next__(self) -> tuple[list[torch.Tensor], list[torch.Tensor], BatchMeta]:
        """Obtain the next outputs from the loader

        Returns
        -------
        inputs
            list of len (num_inputs) tensors of shape(batch_size, H, W, C) inputs for the model
        targets
            List of len (num_outputs) of target images in shape (batch_size, num_inputs, height,
            width, 3) at all model output sizes as float32 0.0 - 1.0 range
        meta
            The meta information for the batch
        """
        try:
            inputs, targets, meta = T.cast(tuple[list[torch.Tensor],
                                                 list[torch.Tensor],
                                                 "BatchMeta"],
                                           next(self._iterator))
        except StopIteration:
            epoch = self._epoch
            logger.debug("[TrainLoader] epoch %s end", epoch)

            if isinstance(self._loader.sampler, tch_data.DistributedSampler):
                self._loader.sampler.set_epoch(epoch + 1)
            T.cast(MultiDataset, self._loader.dataset).shuffle()
            self._iterator = iter(self._loader)
            inputs, targets, meta = next(self._iterator)
            self._epoch += 1

        if self._learn_mask:  # Add the face mask as it's own target
            assert meta.mask_face is not None
            targets += [meta.mask_face[-1].permute(0, 1, 3, 4, 2)]
        logger.trace(  # type:ignore[attr-defined]
            "[TrainLoader] input_shapes: %s, target_shapes: %s, meta: %s",
            [i.shape for i in inputs], [t.shape for t in targets], meta)
        return inputs, targets, meta


class PreviewLoader():
    """Generator for feeding faceswap models input data for generating preview images. Gets the
    next items from each of the configured loaders and collates them for feeding into a model

    Parameters
    ----------
    input_size
        The input size to the model
    output_sizes
        The output sizes to the model (list as some models have multi-scale outputs)
    color_order
        The color order of the model
    input_folders
        list of folders to read images from for each side being trained
    batch_size
        The number of images being displayed in the preview
    sampler
        The sampler to use for the data loaders. Default: ``None`` (RandomSampler)
    num_samples
        Set to 0 for random previews from the image folder. Set to a positive integer for this
        number of images to use for a static timelapse. Default: 0
    """
    def __init__(self,
                 input_size: int,
                 output_size: int,
                 color_order: T.Literal["bgr", "rgb"],
                 input_folders: list[str],
                 batch_size: int,
                 sampler: None | type[tch_data.RandomSampler | tch_data.SequentialSampler] = None,
                 num_samples: int = 0) -> None:
        self._output_size = output_size
        self._input_folders = input_folders
        self._batch_size = batch_size
        self._num_samples = num_samples

        self._input_size = input_size
        self._color_order: T.Literal["bgr", "rgb"] = T.cast(T.Literal["bgr", "rgb"],
                                                            color_order.lower())
        self._sampler = tch_data.RandomSampler if sampler is None else sampler
        self._loader = self.get_loader()
        self._iterator = T.cast(T.Iterator[tuple[torch.Tensor, torch.Tensor]],
                                iter(self._loader))

    def __iter__(self) -> T.Self:
        """This is an iterator"""
        return self

    def __repr__(self) -> str:
        """Pretty print for logging"""
        params = ", ".join(f"{k[1:]}={repr(v)}" for k, v in self.__dict__.items()
                           if k in ("_input_size", "_output_size", "_color_order",
                                    "_input_folders", "_batch_size", "_sampler", "_num_samples"))
        return f"{self.__class__.__name__}({params})"

    def get_loader(self) -> DataLoader:
        """Obtain the dataloaders for each input/output for the model

        Returns
        -------
        The Training data loaders in side order
        """
        data_sets = tuple(PreviewSet(get_label(i, len(self._input_folders)),
                                     f,
                                     self._input_size,
                                     self._output_size,
                                     self._color_order,
                                     num_images=self._num_samples)
                          for i, f in enumerate(self._input_folders))
        preview_set = MultiDataset(data_sets, is_random=self._num_samples == 0)
        retval = DataLoader(dataset=preview_set,
                            batch_size=self._batch_size,
                            sampler=self._sampler(preview_set),
                            num_workers=1,  # Previews don't need speed
                            pin_memory=True,
                            drop_last=True)
        logger.debug("[PreviewLoader] Set loader : %s", retval)
        return retval

    def _items_from_loader(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Obtain the next outputs from the given loader index

        Returns
        -------
        feed
            The batch of feed images for a side
        targets
            A batch of full sized, full coverage input images with mask in the 4th channel
        """
        try:
            inputs, targets = T.cast(tuple[torch.Tensor, torch.Tensor], next(self._iterator))

        except StopIteration:
            logger.debug("[PreviewLoader] end")
            self._iterator = iter(self._loader)
            inputs, targets = next(self._iterator)

        logger.trace(  # type:ignore[attr-defined]
            "[PreviewLoader] input_shapes: %s, target_shape: %s",
            inputs.shape, targets.shape)
        return inputs, targets

    def __next__(self) -> tuple[torch.Tensor, torch.Tensor]:
        """ Obtain the next batch of data for each side for feeding the model

        Returns
        -------
        inputs
            The inputs to the model for each side of the model. The array is returned in `(side,
            batch_size, *dims)` where `side` 0 is "A" and `side` 1 is "B" etc.
        targets
            The full sized source image with mask in 4th channel for each side of the model in
            format `(side, batch_size, *dims, 4) where `side` 0 is "A" and `side` 1 is "B" etc.
        """
        items = self._items_from_loader()
        inputs = items[0].swapaxes(0, 1)
        targets = items[1].swapaxes(0, 1)
        logger.debug("[PreviewLoader] inputs: %s, targets: %s",  # type:ignore[attr-defined]
                     inputs.shape, targets.shape)
        return inputs, targets


get_module_objects(__name__)
