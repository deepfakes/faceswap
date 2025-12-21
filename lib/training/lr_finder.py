#!/usr/bin/env python3
""" Learning Rate Finder for faceswap.py. """
from __future__ import annotations
import logging
import os
import shutil
import typing as T
from datetime import datetime
from enum import Enum

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from lib.logger import parse_class_init
from lib.utils import get_module_objects
from plugins.train import train_config as cfg

if T.TYPE_CHECKING:
    from keras import optimizers
    from plugins.train import training

logger = logging.getLogger(__name__)


class LRStrength(Enum):
    """ Enum for how aggressively to set the optimal learning rate """
    DEFAULT = 10
    AGGRESSIVE = 5
    EXTREME = 2.5


class LearningRateFinder:  # pylint:disable=too-many-instance-attributes
    """ Learning Rate Finder

    Parameters
    ----------
    trainer : :class:`plugins.train.run_trainer.Trainer`
        The training loop with the loaded training plugin
    stop_factor : int
        When to stop finding the optimal learning rate
    beta : float
        Amount to smooth loss by, for graphing purposes
    """
    def __init__(self,  # pylint:disable=too-many-positional-arguments
                 trainer: training.Trainer,
                 stop_factor: int = 4,
                 beta: float = 0.98) -> None:
        logger.debug(parse_class_init(locals()))
        self._iterations = cfg.lr_finder_iterations()
        self._save_graph = cfg.lr_finder_mode() in ("graph_and_set", "graph_and_exit")
        self._strength = LRStrength[cfg.lr_finder_strength().upper()].value

        self._start_lr = 1e-10
        end_lr = 1e+1

        self._trainer = trainer

        self._model = trainer._plugin.model
        self._optimizer = trainer._plugin.model.model.optimizer

        self._stop_factor = stop_factor
        self._beta = beta
        self._lr_multiplier: float = (end_lr / self._start_lr) ** (1.0 / self._iterations)

        self._metrics: dict[T.Literal["learning_rates", "losses"], list[float]] = {
            "learning_rates": [],
            "losses": []}
        self._loss: dict[T.Literal["avg", "best"], float] = {"avg": 0.0, "best": 1e9}

        logger.debug("Initialized %s", self.__class__.__name__)

    def _on_batch_end(self, iteration: int, loss: float) -> None:
        """ Learning rate actions to perform at the end of a batch

        Parameters
        ----------
        iteration: int
            The current iteration
        loss: float
            The loss value for the current batch
        """
        learning_rate = float(self._optimizer.learning_rate.numpy())
        self._metrics["learning_rates"].append(learning_rate)

        self._loss["avg"] = (self._beta * self._loss["avg"]) + ((1 - self._beta) * loss)
        smoothed = self._loss["avg"] / (1 - (self._beta ** iteration))
        self._metrics["losses"].append(smoothed)

        stop_loss = self._stop_factor * self._loss["best"]

        if iteration > 1 and smoothed > stop_loss:
            self._model.model.stop_training = True
            return

        if iteration == 1 or smoothed < self._loss["best"]:
            self._loss["best"] = smoothed

        learning_rate *= self._lr_multiplier

        self._optimizer.learning_rate.assign(learning_rate)

    def _update_description(self, progress_bar: tqdm) -> None:
        """ Update the description of the progress bar for the current iteration

        Parameters
        ----------
        progress_bar: :class:`tqdm.tqdm`
            The learning rate finder progress bar to update
        """
        current = self._metrics['learning_rates'][-1]
        best_idx = self._metrics["losses"].index(self._loss["best"])
        best = self._metrics["learning_rates"][best_idx] / self._strength
        progress_bar.set_description(f"Current: {current:.1e}  Best: {best:.1e}")

    def _train(self) -> None:
        """ Train the model for the given number of iterations to find the optimal
        learning rate and show progress"""
        logger.info("Finding optimal learning rate...")
        pbar = tqdm(range(1, self._iterations + 1),
                    desc="Current: N/A      Best: N/A    ",
                    leave=False)
        for idx in pbar:
            loss = self._trainer.train_one_batch()

            if any(np.isnan(x) for x in loss):
                logger.warning("NaN detected! Exiting early")
                break
            self._on_batch_end(idx, loss[0])
            self._update_description(pbar)

    def _rebuild_optimizer(self, optimizer: optimizers.Optimizer) -> optimizers.Optimizer:
        """ Pass through nested Optimizers (eg LossScaleOptimizer) and create new nested
        optimizers based on their original config

        Returns
        -------
        :class:`keras.optimizers.Optimizer`
            A new optimizer of the same type as the given one, with the same config
        """
        logger.debug("Processing optimizer: '%s'", optimizer.name)
        config = optimizer.get_config()
        if hasattr(optimizer, "inner_optimizer"):
            config["inner_optimizer"] = self._rebuild_optimizer(optimizer.inner_optimizer)
        retval = optimizer.__class__(**config)
        logger.debug("Created optimizer '%s': (old: %s, new: %s)",
                     optimizer.name, optimizer, retval)
        return retval

    def _reset_model(self, original_lr: float, new_lr: float) -> None:
        """ Reset the model's weights to initial values, reset the model's optimizer and set the
        learning rate

        Parameters
        ----------
        original_lr: float
            The model's original learning rate
        new_lr: float
            The discovered optimal learning rate
        """
        self._model.state.add_lr_finder(new_lr)
        self._model.state.save()

        if cfg.lr_finder_mode() == "graph_and_exit":
            return

        logger.debug("Resetting optimizer")
        optimizer = self._rebuild_optimizer(self._optimizer)
        del self._optimizer
        del self._model.model.optimizer

        logger.info("Loading initial weights")
        self._model.model.load_weights(self._model.io.filename)

        self._model.model.compile(optimizer=optimizer,
                                  loss=self._model.model.loss,
                                  metrics=self._model.model.loss)

        logger.info("Updating Learning Rate from %s to %s", f"{original_lr:.1e}", f"{new_lr:.1e}")
        self._model.model.optimizer.learning_rate.assign(new_lr)
        self._optimizer = self._model.model.optimizer

    def find(self) -> bool:
        """ Find the optimal learning rate

        Returns
        -------
        bool
            ``True`` if the learning rate was succesfully discovered otherwise ``False``
        """
        if not self._model.io.model_exists:
            self._model.io.save()

        original_lr = float(self._model.model.optimizer.learning_rate.numpy())
        self._model.model.optimizer.learning_rate.assign(self._start_lr)

        self._train()
        print("\x1b[2K", end="\r")  # Clear line

        best_idx = self._metrics["losses"].index(self._loss["best"])
        new_lr = self._metrics["learning_rates"][best_idx] / self._strength
        if new_lr < 1e-9:
            logger.error("The optimal learning rate could not be found. This is most likely "
                         "because you did not run the finder for enough iterations.")
            shutil.rmtree(self._model.io.model_dir)
            return False

        self._plot_loss()
        self._reset_model(original_lr, new_lr)
        return True

    def _plot_loss(self, skip_begin: int = 10, skip_end: int = 1) -> None:
        """ Plot a graph of loss vs learning rate and save to the training folder

        Parameters
        ----------
        skip_begin: int, optional
            Number of iterations to skip at the start. Default: `10`
        skip_end: int, optional
            Number of iterations to skip at the end. Default: `1`
        """
        if not self._save_graph:
            return

        matplotlib.use("Agg")
        lrs = self._metrics["learning_rates"][skip_begin:-skip_end]
        losses = self._metrics["losses"][skip_begin:-skip_end]
        plt.plot(lrs, losses, label="Learning Rate")
        best_idx = self._metrics["losses"].index(self._loss["best"])
        best_lr = self._metrics["learning_rates"][best_idx]
        for val, color in zip(LRStrength, ("g", "y", "r")):
            l_r = best_lr / val.value
            idx = lrs.index(next(r for r in lrs if r >= l_r))
            plt.plot(l_r, losses[idx],
                     f"{color}o",
                     label=f"{val.name.title()}: {l_r:.1e}")

        plt.xscale("log")
        plt.xlabel("Learning Rate (Log Scale)")
        plt.ylabel("Loss")
        plt.title("Learning Rate Finder")
        plt.legend()

        now = datetime.now().strftime("%Y-%m-%d_%H.%M.%S")
        output = os.path.join(self._model.io.model_dir, f"learning_rate_finder_{now}.png")
        logger.info("Saving Learning Rate Finder graph to: '%s'", output)
        plt.savefig(output)


__all__ = get_module_objects(__name__)
