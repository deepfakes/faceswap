#!/usr/bin/env python3
""" Tensorboard call back for PyTorch logging. Hopefully temporary until a native Keras version
is implemented """

import logging
import os
import typing as T

import keras
from torch.utils.tensorboard import SummaryWriter

from lib.logger import parse_class_init
from lib.utils import get_module_objects

logger = logging.getLogger(__name__)


class TorchTensorBoard(keras.callbacks.Callback):
    """Enable visualizations for TensorBoard. Adapted from Keras' Tensorboard Callback keeping
    only the parts we need, and using Torch rather than TensorFlow

    Parameters
    ----------
    log_dir str
        The path of the directory where to save the log files to be parsed by TensorBoard. e.g.,
        `log_dir = os.path.join(working_dir, 'logs')`. This directory should not be reused by any
        other callbacks.
    write_graph: bool  (Not supported at this time)
        Whether to visualize the graph in TensorBoard. Note that the log file can become quite
        large when `write_graph` is set to `True`.
    update_freq: Literal["batch", "epoch"] | int
        When using `"epoch"`, writes the losses and metrics to TensorBoard after every epoch.
        If using an integer, let's say `1000`, all metrics and losses (including custom ones
        added by `Model.compile`) will be logged to TensorBoard every 1000 batches. `"batch"`
        is a synonym for 1, meaning that they will be written every batch. Note however that
        writing too frequently to TensorBoard can slow down your training, especially when used
        with distribution strategies as it will incur additional synchronization overhead. Batch-
        level summary writing is also available via `train_step` override. Please see [TensorBoard
        Scalars
        tutorial](https://www.tensorflow.org/tensorboard/scalars_and_keras#batch-level_logging)
    """

    def __init__(self,
                 log_dir: str = "logs",
                 write_graph: bool = True,
                 update_freq: T.Literal["batch", "epoch"] | int = "epoch") -> None:
        logger.debug(parse_class_init(locals()))
        super().__init__()
        self.log_dir = str(log_dir)
        self.write_graph = write_graph
        self.update_freq = 1 if update_freq == "batch" else update_freq

        self._should_write_train_graph = False
        self._train_dir = os.path.join(self.log_dir, "train")
        self._train_step = 0
        self._global_train_batch = 0
        self._previous_epoch_iterations = 0

        self._model: keras.models.Model | None = None
        self._writers: dict[str, SummaryWriter] = {}
        logger.debug("Initialized %s", self.__class__.__name__)

    @property
    def _train_writer(self) -> SummaryWriter:
        """:class:`torch.utils.tensorboard.SummaryWriter`: The summary writer """
        if "train" not in self._writers:
            self._writers["train"] = SummaryWriter(self._train_dir)
        return self._writers["train"]

    def _write_keras_model_summary(self) -> None:
        """Writes Keras graph network summary to TensorBoard."""
        assert self._model is not None
        summary = self._model.to_json()
        self._train_writer.add_text("keras", summary, global_step=0)

    def _write_keras_model_train_graph(self) -> None:
        """Writes Keras graph to TensorBoard."""
        # TODO implement
        logger.debug("Tensorboard graph logging not yet implemented")

    def set_model(self, model: keras.models.Model) -> None:
        """Sets Keras model and writes graph if specified.

        Parameters
        ----------
        model: :class:`keras.models.Model`
            The model that is being trained
        """
        self._model = model

        if self.write_graph:
            self._write_keras_model_summary()
            self._should_write_train_graph = True

    def on_train_begin(self, logs=None) -> None:
        """ Initialize the call back on train start

        Parameters
        ----------
        logs: None
            Unused
        """
        self._global_train_batch = 0
        self._previous_epoch_iterations = 0

    def on_train_batch_end(self, batch: int, logs: dict[str, float] | None = None) -> None:
        """ Update Tensorboard logs on batch end

        Parameters
        ----------
        batch: int
            The current iteration count
        logs: dict[str, float]
            The logs to write
        """
        assert logs is not None
        if self._should_write_train_graph:
            self._write_keras_model_train_graph()
            self._should_write_train_graph = False

        for key, value in logs.items():
            self._train_writer.add_scalar(f"batch_{key}",
                                          value,
                                          global_step=batch)

    def on_save(self) -> None:
        """ Flush data to disk on save """
        logger.debug("Flushing Tensorboard writer")
        self._train_writer.flush()

    def on_train_end(self, logs=None) -> None:
        """ Close the writer on train completion

        Parameters
        ----------
        logs: None
            Unused
        """
        for writer in self._writers.values():
            writer.flush()
            writer.close()


__all__ = get_module_objects(__name__)
