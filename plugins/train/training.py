#! /usr/env/bin/python3
""" Run the training loop for a training plugin """
from __future__ import annotations

import logging
import os
import typing as T
import time

import numpy as np
import torch

from torch.cuda import OutOfMemoryError

from lib.training import Feeder, LearningRateFinder, LearningRateWarmup
from lib.training.tensorboard import TorchTensorBoard
from lib.utils import get_module_objects, FaceswapError
from plugins.train import train_config as mod_cfg
from plugins.train.trainer import trainer_config as trn_cfg

from plugins.train.trainer._display import Samples, Timelapse

if T.TYPE_CHECKING:
    from collections.abc import Callable
    from plugins.train.trainer._base import TrainerBase

logger = logging.getLogger(__name__)


class Trainer:
    """ Handles the feeding of training images to Faceswap models, the generation of Tensorboard
    logs and the creation of sample/time-lapse preview images.

    All Trainer plugins must inherit from this class.

    Parameters
    ----------
    plugin : :class:`TrainerBase`
        The plugin that will be processing each batch
    images : dict[literal["a", "b"], list[str]]
        The file paths for the images to be trained on for each side. The dictionary should contain
        2 keys ("a" and "b") with the values being a list of full paths corresponding to each side.
    """

    def __init__(self, plugin: TrainerBase, images: dict[T.Literal["a", "b"], list[str]]) -> None:
        self._batch_size = plugin.batch_size
        self._plugin = plugin
        self._model = plugin.model

        self._feeder = Feeder(images, plugin.model, plugin.batch_size)

        self._exit_early = self._handle_lr_finder()
        if self._exit_early:
            logger.debug("Exiting from LR Finder")
            return

        self._warmup = self._get_warmup()
        self._model.state.add_session_batchsize(plugin.batch_size)
        self._images = images
        self._sides = sorted(key for key in self._images.keys())

        self._tensorboard = self._set_tensorboard()
        self._samples = Samples(self._model,
                                self._model.coverage_ratio,
                                trn_cfg.mask_opacity(),
                                trn_cfg.mask_color())

        num_images = trn_cfg.preview_images()
        assert isinstance(num_images, int)
        self._timelapse = Timelapse(self._model,
                                    self._model.coverage_ratio,
                                    num_images,
                                    trn_cfg.mask_opacity(),
                                    trn_cfg.mask_color(),
                                    self._feeder,
                                    self._images)
        logger.debug("Initialized %s", self.__class__.__name__)

    @property
    def exit_early(self) -> bool:
        """ True if the trainer should exit early, without perfoming any training steps """
        return self._exit_early

    @property
    def batch_size(self) -> int:
        """int : The batch size that the model is set to train at. """
        return self._batch_size

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
            lrf = LearningRateFinder(self)
            success = lrf.find()
            return mod_cfg.lr_finder_mode() == "graph_and_exit" or not success

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

    def train_one_batch(self) -> np.ndarray:
        """ Process a single batch through the model and obtain the loss

        Returns
        -------
        :class:`numpy.ndarray`
            The total loss in the first position then A losses, by output order, then B losses, by
            output order
        """
        try:
            inputs, targets = self._feeder.get_batch()
            loss_t = self._plugin.train_batch(torch.from_numpy(inputs),
                                              [torch.from_numpy(t) for t in targets])
            loss_cpu = loss_t.detach().cpu().numpy()
            retval = np.array([sum(loss_cpu), *loss_cpu])
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
        return retval

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
        loss = self.train_one_batch()
        self._log_tensorboard(loss)
        loss = self._collate_and_store_loss(loss[1:])
        self._print_loss(loss)
        if do_snapshot:
            self._model.io.snapshot()
        self._update_viewers(viewer, timelapse_kwargs)

    def _log_tensorboard(self, loss: np.ndarray) -> None:
        """ Log current loss to Tensorboard log files

        Parameters
        ----------
        loss : :class:`numpy.ndarray`
            The total loss in the first position then A losses, by output order, then B losses, by
            output order
        """
        if not self._tensorboard:
            return
        logger.trace("Updating TensorBoard log")  # type: ignore
        logs = {log[0]: float(log[1])
                for log in zip(self._model.state.loss_names, loss)}

        self._tensorboard.on_train_batch_end(self._model.iterations, logs=logs)

    def _collate_and_store_loss(self, loss: np.ndarray) -> np.ndarray:
        """ Collate the loss into totals for each side.

        The losses are summed into a total for each side. Loss totals are added to
        :attr:`model.state._history` to track the loss drop per save iteration for backup purposes.

        If NaN protection is enabled, Checks for NaNs and raises an error if detected.

        Parameters
        ----------
        loss : :class:`numpy.ndarray`
            The total loss in the first position then A losses, by output order, then B losses, by
            output order

        Returns
        -------
        :class:`numpy.ndarray`
            2 ``floats`` which is the total loss for each side (eg sum of face + mask loss)

        Raises
        ------
        FaceswapError
            If a NaN is detected, a :class:`FaceswapError` will be raised
        """
        # NaN protection
        if mod_cfg.nan_protection() and not all(np.isfinite(val) for val in loss):
            logger.critical("NaN Detected. Loss: %s", loss)
            raise FaceswapError("A NaN was detected and you have NaN protection enabled. Training "
                                "has been terminated.")

        split = len(loss) // 2
        combined_loss = np.array([sum(loss[:split]), sum(loss[split:])])
        self._model.add_history(combined_loss)
        logger.trace("original loss: %s, combined_loss: %s", loss, combined_loss)  # type: ignore
        return combined_loss

    def _print_loss(self, loss: np.ndarray) -> None:
        """ Outputs the loss for the current iteration to the console.

        Parameters
        ----------
        loss : :class`numpy.ndarray`
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
