#!/usr/bin/env python3
"""Learning Rate Finder for faceswap.py."""
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

if T.TYPE_CHECKING:
    from torch import Tensor
    from torch.optim.lr_scheduler import ExponentialLR
    from . import train

logger = logging.getLogger(__name__)


class LRStrength(Enum):
    """Enum for how aggressively to set the optimal learning rate"""
    DEFAULT = 10
    AGGRESSIVE = 5
    EXTREME = 2.5


class LearningRateFinder:  # pylint:disable=too-many-instance-attributes
    """Learning Rate Finder

    Parameters
    ----------
    trainer
        The training loop with the loaded training plugin
    scheduler
        The LRFinder scheduler
    steps
        The number of steps to run the finder for
    strength
        How aggressively to set the optimal learning rate
    mode
        The mode to run the Learning Rate Finder in
    stop_factor
        When to stop finding the optimal learning rate
    beta
        Amount to smooth loss by, for graphing purposes
    """
    def __init__(self,
                 trainer: train.Trainer,
                 scheduler: ExponentialLR,
                 steps: int,
                 strength: T.Literal["default", "aggressive", "extreme"],
                 mode: T.Literal["set", "graph_and_set", "graph_and_exit"],
                 stop_factor: int = 4,
                 beta: float = 0.98) -> None:
        logger.debug(parse_class_init(locals()))
        self._trainer = trainer
        self._scheduler = scheduler
        self._steps = steps
        self._strength = LRStrength[strength.upper()].value
        self._mode = mode
        self._stop_factor = stop_factor
        self._beta = beta

        self._model = trainer._plugin.model
        self._losses: list[float] = []
        self._learning_rates: list[float] = []
        self._loss: dict[T.Literal["avg", "best"], float] = {"avg": 0.0, "best": 1e9}
        self._best_lr: None | float = None

    @property
    def best_lr(self) -> None | float:
        """The discovered best learning rate or ``None`` if not found"""
        return self._best_lr

    def _on_batch_end(self, iteration: int, loss: float) -> bool:
        """Learning rate actions to perform at the end of a batch

        Parameters
        ----------
        iteration
            The current iteration
        loss
            The loss value for the current batch

        Returns
        -------
        ``True`` if training should cease. ``False`` to continue
        """
        if np.isnan(loss):
            logger.info("Loss has NaN'd. Exiting early")
            return True

        self._learning_rates.append(T.cast(float, self._scheduler.get_last_lr()[0]))
        self._loss["avg"] = (self._beta * self._loss["avg"]) + ((1 - self._beta) * loss)
        smoothed = self._loss["avg"] / (1 - (self._beta ** iteration))
        self._losses.append(smoothed)

        stop_loss = self._stop_factor * self._loss["best"]
        if iteration > 1 and smoothed > stop_loss:
            logger.info("Loss has diverged. Exiting early")
            return True

        if iteration == 1 or smoothed < self._loss["best"]:
            self._loss["best"] = smoothed

        return False

    def _update_description(self, progress_bar: tqdm) -> None:
        """Update the description of the progress bar for the current iteration

        Parameters
        ----------
        progress_bar
            The learning rate finder progress bar to update
        """
        current = self._learning_rates[-1]
        best_idx = self._losses.index(self._loss["best"])
        best = self._learning_rates[best_idx] / self._strength
        progress_bar.set_description(f"Current: {current:.1e}  Best: {best:.1e}")

    def _train(self) -> None:
        """Train the model for the given number of iterations to find the optimal
        learning rate and show progress"""
        logger.info("Finding optimal learning rate...")
        p_bar = tqdm(range(1, self._steps + 1),
                     desc="Current: N/A      Best: N/A    ",
                     leave=False)
        for idx in p_bar:
            loss = self._trainer.train_one_batch()
            total_loss = T.cast("Tensor", sum(x.total for x in loss)).item()

            if self._on_batch_end(idx, total_loss):
                logger.debug("[LearningRateFinder] Exiting early")
                break

            self._update_description(p_bar)

    def _reset_model(self, new_lr: float) -> None:
        """Reset the model's weights to initial values, reset the model's optimizer and set the
        learning rate

        Parameters
        ----------
        new_lr
            The discovered optimal learning rate
        """
        self._model.state.add_lr_finder(new_lr)
        self._model.state.save()

        if self._mode == "graph_and_exit":
            return

        logger.info("Loading initial weights")
        self._model.model.load_weights(self._model.io.filename)

    def _plot_loss(self, skip_begin: int = 10, skip_end: int = 1) -> None:
        """Plot a graph of loss vs learning rate and save to the training folder

        Parameters
        ----------
        skip_begin
            Number of iterations to skip at the start. Default: `10`
        skip_end
            Number of iterations to skip at the end. Default: `1`
        """
        if self._mode not in ("graph_and_set", "graph_and_exit"):
            return

        matplotlib.use("Agg")
        lrs = self._learning_rates[skip_begin:-skip_end]
        losses = self._losses[skip_begin:-skip_end]
        plt.plot(lrs, losses, label="Learning Rate")
        best_idx = self._losses.index(self._loss["best"])
        best_lr = self._learning_rates[best_idx]
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

    def find(self) -> None:
        """Find the optimal learning rate"""
        if not self._model.io.model_exists:
            self._model.io.save()

        self._train()
        print("\x1b[2K", end="\r")  # Clear line

        best_idx = self._losses.index(self._loss["best"])
        new_lr = self._learning_rates[best_idx] / self._strength
        if new_lr < 1e-9:
            logger.error("The optimal learning rate could not be found. This is most likely "
                         "because you did not run the finder for enough iterations.")
            shutil.rmtree(self._model.io.model_dir)
            return

        self._best_lr = new_lr
        self._plot_loss()
        self._reset_model(new_lr)


__all__ = get_module_objects(__name__)
