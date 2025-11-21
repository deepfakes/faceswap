#! /usr/env/bin/python3
""" Handles Learning Rate Warmup when training a model """
from __future__ import annotations

import logging
import typing as T

import keras.backend as K

logger = logging.getLogger(__name__)

if T.TYPE_CHECKING:
    from keras.models import Model


class LearningRateWarmup():
    """ Handles the updating of the model's learning rate during Learning Rate Warmup

    Parameters
    ----------
    model : :class:`keras.models.Model`
        The keras model that is to be trained
    target_learning_rate : float
        The final learning rate at the end of warmup
    steps : int
        The number of iterations to warmup the learning rate for
    """
    def __init__(self, model: Model, target_learning_rate: float, steps: int) -> None:
        self._model = model
        self._target_lr = target_learning_rate
        self._steps = steps
        self._current_lr = 0.0
        self._current_step = 0
        self._reporting_points = [int(self._steps * i / 10) for i in range(11)]
        logger.debug("Initialized %s", self)

    def __repr__(self) -> str:
        """ Pretty string representation for logging """
        params = ", ".join(f"{k}={v}" for k, v in self.__dict__.items())
        return f"{self.__class__.__name__}({params})"

    @classmethod
    def _format_notation(cls, value: float) -> str:
        """ Format a float to scientific notation at 1 decimal place

        Parameters
        ----------
        value : float
            The value to format

        Returns
        -------
        str
            The formatted float in scientific notation at 1 decimal place
        """
        return f"{value:.1e}"

    def _set_learning_rate(self) -> None:
        """ Set the learning rate for the current step """
        self._current_lr = self._current_step / self._steps * self._target_lr
        K.set_value(self._model.optimizer.lr, self._current_lr)
        logger.debug("Learning rate set to %s for step %s/%s",
                     self._current_lr, self._current_step, self._steps)

    def _output_status(self) -> None:
        """ Output the progress of Learning Rate Warmup at set intervals """
        if self._current_step == 1:
            logger.info("[Learning Rate Warmup] Start: %s, Target: %s, Steps: %s",
                        self._format_notation(self._current_lr),
                        self._format_notation(self._target_lr), self._steps)
            return

        if self._current_step == self._steps:
            print()
            logger.info("[Learning Rate Warmup] Final Learning Rate: %s",
                        self._format_notation(self._target_lr))
            return

        if self._current_step in self._reporting_points:
            print()
            progress = int(round(100 / (len(self._reporting_points) - 1) *
                           self._reporting_points.index(self._current_step), 0))
            logger.info("[Learning Rate Warmup] Step: %s/%s (%s), Current: %s, Target: %s",
                        self._current_step,
                        self._steps,
                        f"{progress}%",
                        self._format_notation(self._current_lr),
                        self._format_notation(self._target_lr))

    def __call__(self) -> None:
        """ If a learning rate update is required, update the model's learning rate, otherwise
        do nothing """
        if self._steps == 0 or self._current_step >= self._steps:
            return

        self._current_step += 1
        self._set_learning_rate()
        self._output_status()
