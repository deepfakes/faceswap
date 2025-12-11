#!/usr/bin/env python3
""" Base Class for Faceswap Trainer plugins. All Trainer plugins should be inherited from
this class.

At present there is only the :class:`~plugins.train.trainer.original` plugin, so that entirely
inherits from this class. If further plugins are developed, then common code should be kept here,
with "original" unique code split out to the original plugin.
"""
from __future__ import annotations
import logging
import os
import time
import typing as T

import numpy as np
import torch

from keras import ops
from keras.src.tree import flatten
from torch.cuda import OutOfMemoryError

from lib.training import Feeder, LearningRateFinder, LearningRateWarmup
from lib.utils import get_module_objects, FaceswapError
from plugins.train._config import Config

from ._display import Samples, Timelapse
from ._tensorboard import TorchTensorBoard

if T.TYPE_CHECKING:
    from collections.abc import Callable
    from plugins.train.model._base import ModelBase
    from lib.config import ConfigValueType

logger = logging.getLogger(__name__)


def _get_config(plugin_name: str,
                configfile: str | None = None) -> dict[str, ConfigValueType]:
    """ Return the configuration for the requested trainer.

    Parameters
    ----------
    plugin_name: str
        The name of the plugin to load the configuration for
    configfile: str, optional
        A custom configuration file. If ``None`` then configuration is loaded from the default
        :file:`.config.train.ini` file. Default: ``None``

    Returns
    -------
    dict
        The configuration dictionary for the requested plugin
    """
    return Config(plugin_name, configfile=configfile).config_dict


class TrainerBase():
    """ Handles the feeding of training images to Faceswap models, the generation of Tensorboard
    logs and the creation of sample/time-lapse preview images.

    All Trainer plugins must inherit from this class.

    Parameters
    ----------
    model: plugin from :mod:`plugins.train.model`
        The model that will be running this trainer
    images: dict
        The file paths for the images to be trained on for each side. The dictionary should contain
        2 keys ("a" and "b") with the values being a list of full paths corresponding to each side.
    batch_size: int
        The requested batch size for iteration to be trained through the model.
    configfile: str
        The path to a custom configuration file. If ``None`` is passed then configuration is loaded
        from the default :file:`.config.train.ini` file.
    """

    def __init__(self,
                 model: ModelBase,
                 images: dict[T.Literal["a", "b"], list[str]],
                 batch_size: int,
                 configfile: str | None) -> None:
        logger.debug("Initializing %s: (model: '%s', batch_size: %s)",
                     self.__class__.__name__, model, batch_size)
        self._model = model
        self._config = self._get_config(configfile)

        self._feeder = Feeder(images, model, batch_size, self._config)

        self._exit_early = self._handle_lr_finder()
        if self._exit_early:
            return

        self._warmup = self._get_warmup()
        self._model.state.add_session_batchsize(batch_size)
        self._images = images
        self._sides = sorted(key for key in self._images.keys())

        self._tensorboard = self._set_tensorboard()
        self._samples = Samples(self._model,
                                self._model.coverage_ratio,
                                T.cast(int, self._config["mask_opacity"]),
                                T.cast(str, self._config["mask_color"]))

        num_images = self._config.get("preview_images", 14)
        assert isinstance(num_images, int)
        self._timelapse = Timelapse(self._model,
                                    self._model.coverage_ratio,
                                    num_images,
                                    T.cast(int, self._config["mask_opacity"]),
                                    T.cast(str, self._config["mask_color"]),
                                    self._feeder,
                                    self._images)
        logger.debug("Initialized %s", self.__class__.__name__)

    @property
    def exit_early(self) -> bool:
        """ True if the trainer should exit early, without perfoming any training steps """
        return self._exit_early

    def _get_config(self, configfile: str | None) -> dict[str, ConfigValueType]:
        """ Get the saved training config options. Override any global settings with the setting
        provided from the model's saved config.

        Parameters
        -----------
        configfile: str
            The path to a custom configuration file. If ``None`` is passed then configuration is
            loaded from the default :file:`.config.train.ini` file.

        Returns
        -------
        dict
            The trainer configuration options
        """
        config = _get_config(".".join(self.__module__.split(".")[-2:]),
                             configfile=configfile)
        for key, val in config.items():
            if key in self._model.config and val != self._model.config[key]:
                new_val = self._model.config[key]
                logger.debug("Updating global training config item for '%s' form '%s' to '%s'",
                             key, val, new_val)
                config[key] = new_val
        return config

    def _handle_lr_finder(self) -> bool:
        """ Handle the learning rate finder.

        If this is a new model, then find the optimal learning rate and return ``True`` if user has
        just requested the graph, otherwise return ``False`` to continue training

        If it as existing model, set the learning rate to the value found by the learing rate
        finder and return ``False`` to continue training

        Returns
        -------
        bool
            ``True`` if the learning rate finder options dictate that training should not continue
            after finding the optimal leaning rate
        """
        if not self._model.command_line_arguments.use_lr_finder:
            return False

        if self._model.state.lr_finder > -1:
            learning_rate = self._model.state.lr_finder
            logger.info("Setting learning rate from Learning Rate Finder to %s",
                        f"{learning_rate:.1e}")
            self._model.model.optimizer.learning_rate.assign(learning_rate)
            self._model.state.update_session_config("learning_rate", learning_rate)
            return False

        if self._model.state.iterations == 0 and self._model.state.session_id == 1:
            lrf = LearningRateFinder(self._model, self._config, self._feeder)
            success = lrf.find()
            return self._config["lr_finder_mode"] == "graph_and_exit" or not success

        logger.debug("No learning rate finder rate. Not setting")
        return False

    def _get_warmup(self) -> LearningRateWarmup:
        """ Obtain the learning rate warmup instance

        Returns
        -------
        :class:`plugins.train.lr_warmup.LRWarmup`
            The Learning Rate Warmup object
        """
        target_lr = float(self._model.model.optimizer.learning_rate.value.cpu().numpy())
        return LearningRateWarmup(self._model.model, target_lr, self._model.warmup_steps)

    def _set_tensorboard(self) -> TorchTensorBoard | None:
        """ Set up Tensorboard callback for logging loss.

        Bypassed if command line option "no-logs" has been selected.

        Returns
        -------
        :class:`keras.callbacks.TensorBoard` | None
            Tensorboard object for the the current training session. ``None`` if Tensorboard
            logging is not selected
        """
        if self._model.state.current_session["no_logs"]:
            logger.verbose("TensorBoard logging disabled")  # type: ignore
            return None
        logger.debug("Enabling TensorBoard Logging")

        logger.debug("Setting up TensorBoard Logging")
        log_dir = os.path.join(str(self._model.io.model_dir),
                               f"{self._model.name}_logs",
                               f"session_{self._model.state.session_id}")
        tensorboard = TorchTensorBoard(log_dir=log_dir,
                                       write_graph=True,
                                       update_freq="batch")
        tensorboard.set_model(self._model.model)
        logger.verbose("Enabled TensorBoard Logging")  # type: ignore
        return tensorboard

    def toggle_mask(self) -> None:
        """ Toggle the mask overlay on or off based on user input. """
        self._samples.toggle_mask_display()

    def _forward(self) -> list[torch.Tensor]:
        """ Perform the forward pass on the model

        Returns
        -------
        list[:class:`torch.Tensor`]
            The loss for each side of the model
        """
        model_inputs, model_targets = self._feeder.get_batch()
        preds = self._model.model(model_inputs)
        losses = [loss_fn(y_true, y_pred)
                  for loss_fn, y_true, y_pred in zip(self._model.model.loss,
                                                     flatten(model_targets),
                                                     preds)]
        logger.trace("Losses: %s", losses)  # type:ignore[attr-defined]
        return losses

    def _backwards_and_apply(self, all_loss: list[torch.Tensor]) -> None:
        """ Perform the backwards passes on the model, once for each side and apply the gradient
        update

        Parameters
        ----------
        all_loss: list[:class:`torch.Tensor`]
            The loss for each output from the model
        """
        self._model.model.zero_grad()
        side_loss = [ops.sum(all_loss[:len(all_loss) // 2]),
                     ops.sum(all_loss[len(all_loss) // 2:])]
        for loss in side_loss:
            loss = T.cast(torch.Tensor, self._model.model.optimizer.scale_loss(loss))
            loss.backward()

        trainable_weights = self._model.model.trainable_weights[:]
        gradients = [v.value.grad for v in trainable_weights]

        # Update weights
        with torch.no_grad():
            self._model.model.optimizer.apply(gradients, trainable_weights)

    def train_one_step(self,
                       viewer: Callable[[np.ndarray, str], None] | None,
                       timelapse_kwargs: dict[T.Literal["input_a", "input_b", "output"],
                                              str] | None) -> None:
        """ Running training on a batch of images for each side.

        Triggered from the training cycle in :class:`scripts.train.Train`.

        * Runs a training batch through the model.

        * Outputs the iteration's loss values to the console

        * Logs loss to Tensorboard, if logging is requested.

        * If a preview or time-lapse has been requested, then pushes sample images through the \
        model to generate the previews

        * Creates a snapshot if the total iterations trained so far meet the requested snapshot \
        criteria

        Notes
        -----
        As every iteration is called explicitly, the Parameters defined should always be ``None``
        except on save iterations.

        Parameters
        ----------
        viewer: :func:`scripts.train.Train._show` or ``None``
            The function that will display the preview image
        timelapse_kwargs: dict
            The keyword arguments for generating time-lapse previews. If a time-lapse preview is
            not required then this should be ``None``. Otherwise all values should be full paths
            the keys being `input_a`, `input_b`, `output`.
        """
        self._model.state.increment_iterations()
        logger.trace("Training one step: (iteration: %s)", self._model.iterations)  # type: ignore
        snapshot_interval = self._model.command_line_arguments.snapshot_interval
        do_snapshot = (snapshot_interval != 0 and
                       self._model.iterations - 1 >= snapshot_interval and
                       (self._model.iterations - 1) % snapshot_interval == 0)
        self._warmup()
        try:
            loss_t = self._forward()
            self._backwards_and_apply(loss_t)
            loss = [float(x.detach().cpu().numpy()) for x in loss_t]
            loss = [sum(loss), *loss]
        except OutOfMemoryError as err:
            msg = ("You do not have enough GPU memory available to train the selected model at "
                   "the selected settings. You can try a number of things:"
                   "\n1) Close any other application that is using your GPU (web browsers are "
                   "particularly bad for this)."
                   "\n2) Lower the batchsize (the amount of images fed into the model each "
                   "iteration)."
                   "\n3) Try enabling 'Mixed Precision' training."
                   "\n4) Use a more lightweight model, or select the model's 'LowMem' option "
                   "(in config) if it has one.")
            raise FaceswapError(msg) from err
        self._log_tensorboard(loss)
        loss = self._collate_and_store_loss(loss[1:])
        self._print_loss(loss)
        if do_snapshot:
            self._model.io.snapshot()
        self._update_viewers(viewer, timelapse_kwargs)

    def _log_tensorboard(self, loss: list[float]) -> None:
        """ Log current loss to Tensorboard log files

        Parameters
        ----------
        loss: list
            The list of loss ``floats`` output from the model
        """
        if not self._tensorboard:
            return
        logger.trace("Updating TensorBoard log")  # type: ignore
        logs = {log[0]: log[1]
                for log in zip(self._model.state.loss_names, loss)}

        self._tensorboard.on_train_batch_end(self._model.iterations, logs=logs)

    def _collate_and_store_loss(self, loss: list[float]) -> list[float]:
        """ Collate the loss into totals for each side.

        The losses are summed into a total for each side. Loss totals are added to
        :attr:`model.state._history` to track the loss drop per save iteration for backup purposes.

        If NaN protection is enabled, Checks for NaNs and raises an error if detected.

        Parameters
        ----------
        loss: list
            The list of loss ``floats`` for each side this iteration (excluding total combined
            loss)

        Returns
        -------
        list
            List of 2 ``floats`` which is the total loss for each side (eg sum of face + mask loss)

        Raises
        ------
        FaceswapError
            If a NaN is detected, a :class:`FaceswapError` will be raised
        """
        # NaN protection
        if self._config["nan_protection"] and not all(np.isfinite(val) for val in loss):
            logger.critical("NaN Detected. Loss: %s", loss)
            raise FaceswapError("A NaN was detected and you have NaN protection enabled. Training "
                                "has been terminated.")

        split = len(loss) // 2
        combined_loss = [sum(loss[:split]), sum(loss[split:])]
        self._model.add_history(combined_loss)
        logger.trace("original loss: %s, combined_loss: %s", loss, combined_loss)  # type: ignore
        return combined_loss

    def _print_loss(self, loss: list[float]) -> None:
        """ Outputs the loss for the current iteration to the console.

        Parameters
        ----------
        loss: list
            The loss for each side. List should contain 2 ``floats`` side "a" in position 0 and
            side "b" in position `.
         """
        output = ", ".join([f"Loss {side}: {side_loss:.5f}"
                            for side, side_loss in zip(("A", "B"), loss)])
        timestamp = time.strftime("%H:%M:%S")
        output = f"[{timestamp}] [#{self._model.iterations:05d}] {output}"
        print(f"{output}", end="\r")

    def _update_viewers(self,
                        viewer: Callable[[np.ndarray, str], None] | None,
                        timelapse_kwargs: dict[T.Literal["input_a", "input_b", "output"],
                                               str] | None) -> None:
        """ Update the preview viewer and timelapse output

        Parameters
        ----------
        viewer: :func:`scripts.train.Train._show` or ``None``
            The function that will display the preview image
        timelapse_kwargs: dict
            The keyword arguments for generating time-lapse previews. If a time-lapse preview is
            not required then this should be ``None``. Otherwise all values should be full paths
            the keys being `input_a`, `input_b`, `output`.
        """
        if viewer is not None:
            self._samples.images = self._feeder.generate_preview()
            samples = self._samples.show_sample()
            if samples is not None:
                viewer(samples,
                       "Training - 'S': Save Now. 'R': Refresh Preview. 'M': Toggle Mask. 'F': "
                       "Toggle Screen Fit-Actual Size. 'ENTER': Save and Quit")

        if timelapse_kwargs:
            self._timelapse.output_timelapse(timelapse_kwargs)

    def _clear_tensorboard(self) -> None:
        """ Stop Tensorboard logging.

        Tensorboard logging needs to be explicitly shutdown on training termination. Called from
        :class:`scripts.train.Train` when training is stopped.
         """
        if not self._tensorboard:
            return
        logger.debug("Ending Tensorboard Session: %s", self._tensorboard)
        self._tensorboard.on_train_end()

    def save(self, is_exit: bool = False) -> None:
        """ Save the model

        Parameters
        ----------
        is_exit: bool, optional
            ``True`` if save has been called on model exit. Default: ``False``
        """
        self._model.io.save(is_exit=is_exit)
        assert self._tensorboard is not None
        self._tensorboard.on_save()
        if is_exit:
            self._clear_tensorboard()


__all__ = get_module_objects(__name__)
