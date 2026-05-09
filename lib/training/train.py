#! /usr/env/bin/python3
"""Run the training loop for a training plugin """
from __future__ import annotations

import logging
import os
import typing as T
import time
import warnings

import cv2
import numpy as np

import torch
from torch.cuda import OutOfMemoryError

from lib.logger import format_array, parse_class_init
from lib.torch_utils import get_device
from lib.training import LearningRateFinder, LearningRateWarmup
from lib.training.preview import Samples
from lib.training.data import get_label, PreviewLoader, TrainLoader
from lib.training.tensorboard import TorchTensorBoard
from lib.utils import get_module_objects, FaceswapError
from plugins.train import train_config as mod_cfg
from plugins.train.trainer import trainer_config as trn_cfg

from .loss import LossCollator

if T.TYPE_CHECKING:
    import numpy.typing as npt
    from collections.abc import Callable
    from plugins.train.trainer.base import TrainerBase
    from .loss import BatchLoss

logger = logging.getLogger(__name__)


# Suppress non-Faceswap related Keras warning about backend padding mismatches
warnings.filterwarnings("ignore",
                        message="You might experience inconsistencies",
                        category=UserWarning)


class Trainer:  # pylint:disable=too-many-instance-attributes
    """Handles the feeding of training images to Faceswap models, the generation of Tensorboard
    logs and the creation of sample/time-lapse preview images.

    All Trainer plugins must inherit from this class.

    Parameters
    ----------
    plugin
        The plugin that will be processing each batch
    preview
        ``True`` to generate previews
    timelapse_folders
        The input folders to create timelapse images from. Default: ``None`` (no timelapse)
    timelapse_output
        The folder to output timelapse images. Default: "" (no timelapse)
    """

    def __init__(self,
                 plugin: TrainerBase,
                 preview: bool,
                 timelapse_folders: list[str] | None = None,
                 timelapse_output: str = "") -> None:
        logger.debug(parse_class_init(locals()))
        self._plugin = plugin
        self._preview = preview
        self._timelapse_folders = [] if timelapse_folders is None else timelapse_folders
        self._timelapse_output = timelapse_output

        self._device = get_device()
        self._model = plugin.model
        self._out_size = max(x[1] for x in self._model.output_shapes if x[-1] != 1)
        self._configure_model(plugin)

        self._train_loader = self._get_train_loader()
        self._preview_loader = self._get_preview_loader()
        self._timelapse_loader = self._get_timelapse_loader()

        self._exit_early = self._handle_lr_finder()
        if self._exit_early:
            logger.debug("[Trainer] Exiting from LR Finder")
            return

        self._warmup = self._get_warmup()
        self._model.state.add_session_batchsize(plugin.batch_size)

        self._tensorboard = self._set_tensorboard()
        self._samples = Samples(self._model.coverage_ratio,
                                mod_cfg.Loss.learn_mask() or mod_cfg.Loss.penalized_mask_loss(),
                                trn_cfg.Augmentation.mask_opacity(),
                                trn_cfg.Augmentation.mask_color())

    def __repr__(self) -> str:
        """Pretty print for logging"""
        params = ", ".join(f"{k[1:]}={repr(v)}" for k, v in self.__dict__.items()
                           if k in ("_plugin", "_preview", "_timelapse_folders",
                                    "_timelapse_output"))
        return f"{self.__class__.__name__}({params})"

    @property
    def exit_early(self) -> bool:
        """``True`` if the trainer should exit early, without performing any training steps"""
        return self._exit_early

    def _configure_model(self, plugin: TrainerBase):
        """Add the loss functions to the model and move to the correct device

        Parameters
        ----------
        plugin
            The plugin that is training the model
        """
        loss = LossCollator(
            functions=[mod_cfg.Loss.loss_function(),
                       mod_cfg.Loss.loss_function_2(),
                       mod_cfg.Loss.loss_function_3(),
                       mod_cfg.Loss.loss_function_4()],
            weights=[1.0,
                     mod_cfg.Loss.loss_weight_2() / 100.,
                     mod_cfg.Loss.loss_weight_3() / 100.,
                     mod_cfg.Loss.loss_weight_4() / 100.],
            use_mask=mod_cfg.Loss.penalized_mask_loss(),
            eye_multiplier=mod_cfg.Loss.eye_multiplier(),
            mouth_multiplier=mod_cfg.Loss.mouth_multiplier(),
            smallest_output=min(x[1] for x in self._model.output_shapes
                                if x[-1] != 1),
            mask_loss=(None if not mod_cfg.Loss.learn_mask()
                       else mod_cfg.Loss.mask_loss_function()))
        plugin.register_loss(loss)
        plugin.model.model.to(self._device)

    def _get_train_loader(self) -> TrainLoader:
        """Get the loaders for training the model

        Returns
        -------
        The loaders for feeding the model's training loop
        """
        input_sizes = [x[1] for x in self._model.input_shapes]
        assert len(set(input_sizes)) == 1, f"Multiple input sizes not supported. Got {input_sizes}"

        out_sizes = [x[1] for x in self._model.output_shapes if x[-1] != 1]
        num_sides = len(self._plugin.config.folders)
        assert len(out_sizes) % num_sides == 0, (
            f"Output count ({len(out_sizes)}) doesn't match number of inputs ({num_sides})")
        split = len(out_sizes) // num_sides
        split_sizes = [out_sizes[x:x+split] for x in range(0, len(out_sizes), split)]
        assert len(set(out_sizes)) == len(set(split_sizes[0])), "Sizes for each output must match"

        retval = TrainLoader(input_sizes[0],
                             tuple(split_sizes[0]),
                             self._model.color_order,
                             self._plugin.config,
                             self._plugin.sampler)
        logger.debug("[Trainer] data loader: %s", retval)
        return retval

    def _get_preview_loader(self) -> PreviewLoader | None:
        """Get the loader for generating previews whilst training the model

        Returns
        -------
        The loader for generating preview images during training or ``None`` if previews are
        disabled
        """
        if not self._preview:
            return None
        input_size = self._model.input_shapes[0][1]
        retval = PreviewLoader(input_size,
                               self._out_size,
                               self._model.color_order,
                               self._plugin.config.folders,
                               trn_cfg.Augmentation.preview_images(),
                               torch.utils.data.RandomSampler)
        logger.debug("[Trainer] Preview data loader: %s", retval)
        return retval

    def _get_timelapse_loader(self) -> PreviewLoader | None:
        """Get the loader for generating timelapse images whilst training the model

        Returns
        -------
        The loaders for timelapse preview images during training or ``None`` if previews are
        disabled
        """
        if not self._timelapse_folders or not self._timelapse_output:
            return None
        num_images = trn_cfg.Augmentation.preview_images()
        avail_images = min(len([fname for fname in os.listdir(folder)
                                if os.path.splitext(fname)[-1].lower() == ".png"])
                           for folder in self._timelapse_folders)
        num_samples = min(num_images, avail_images)
        logger.debug("[Train] preview count: %s, available_images: %s, timelapse count: %s",
                     num_images, avail_images, num_samples)
        input_size = self._model.input_shapes[0][1]
        retval = PreviewLoader(input_size,
                               self._out_size,
                               self._model.color_order,
                               self._timelapse_folders,
                               trn_cfg.Augmentation.preview_images(),
                               torch.utils.data.SequentialSampler,
                               num_samples=num_samples)
        logger.debug("[Trainer] Preview data loader: %s", retval)
        return retval

    def _handle_lr_finder(self) -> bool:
        """Handle the learning rate finder.

        If this is a new model, then find the optimal learning rate and return ``True`` if user has
        just requested the graph, otherwise return ``False`` to continue training

        If it as existing model, set the learning rate to the value found by the learning rate
        finder and return ``False`` to continue training

        Returns
        -------
        ``True`` if the learning rate finder options dictate that training should not continue
        after finding the optimal leaning rate
        """
        if not self._plugin.config.lr_finder:
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

        logger.debug("[Trainer] No learning rate finder rate. Not setting")
        return False

    def _get_warmup(self) -> LearningRateWarmup:
        """Obtain the learning rate warmup instance

        Returns
        -------
        The Learning Rate Warmup object
        """
        target_lr = float(self._model.model.optimizer.learning_rate.value.cpu().numpy())
        return LearningRateWarmup(self._model.model, target_lr, self._model.warmup_steps)

    def _set_tensorboard(self) -> TorchTensorBoard | None:
        """Set up Tensorboard callback for logging loss.

        Bypassed if command line option "no-logs" has been selected.

        Returns
        -------
        Tensorboard object for the the current training session. ``None`` if Tensorboard logging is
        not selected
        """
        if self._model.state.current_session["no_logs"]:
            logger.verbose("TensorBoard logging disabled")  # type: ignore
            return None
        logger.debug("[Trainer] Enabling TensorBoard Logging")

        logger.debug("[Trainer] Setting up TensorBoard Logging")
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
        """Toggle the mask overlay on or off based on user input."""
        self._samples.toggle_mask_display()

    def train_one_batch(self) -> list[BatchLoss]:
        """Process a single batch through the model and obtain the loss

        Returns
        -------
        The collated loss values detached and moved to CPU in order (A, B, ...)
        """
        try:
            inputs, targets, meta = next(self._train_loader)
            loss = self._plugin.train_batch([i.to(self._device) for i in inputs],
                                            [t.to(self._device) for t in targets],
                                            meta.to(self._device))
            retval = [x.to_cpu() for x in loss]
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

    def _log_tensorboard(self, loss: list[BatchLoss]) -> None:
        """Log current loss to Tensorboard log files

        Parameters
        ----------
        loss
            The loss scalars for the batch detached and moved to cpu in order (A, B, ...)
        """
        if not self._tensorboard:
            return
        logger.trace("[Trainer] Updating TensorBoard log: %s", loss)  # type: ignore
        logs: dict[str, float | dict[str, float]] = {
            "total": T.cast(torch.Tensor, sum(x.total for x in loss)).item()}
        for i, out in enumerate(loss):
            lbl = get_label(i, len(loss))
            for idx, (w, u) in enumerate(zip(out.weighted, out.unweighted)):
                key = lbl if len(out.unweighted) == 1 else f"{lbl}_{idx}"
                weighted = {k: v.mean() for k, v in w.items()}
                unweighted = {k: v.mean() for k, v in u.items()}
                logs[f"face_{key}"] = T.cast(torch.Tensor, sum(weighted.values())).item()
                logs[f"weighted_{key}"] = {k: v.item() for k, v in weighted.items()}
                logs[f"unweighted_{key}"] = {k: v.item() for k, v in unweighted.items()}
            if out.mask is not None:
                logs[f"mask_{lbl}"] = out.mask.mean().item()
        self._tensorboard.on_train_batch_end(self._model.iterations, logs=logs)

    def _collate_and_store_loss(self, loss: list[BatchLoss]) -> np.ndarray:
        """Collate the loss into totals for each side.

        The losses are summed into a total for each side. Loss totals are added to
        :attr:`model.state._history` to track the loss drop per save iteration for backup purposes.

        If NaN protection is enabled, Checks for NaNs and raises an error if detected.

        Parameters
        ----------
        loss
            The list of loss scalars in order (A, B, ...)

        Returns
        -------
        2 ``floats`` which is the total loss for each side (eg sum of face + mask loss)

        Raises
        ------
        FaceswapError
            If a NaN is detected, a :class:`FaceswapError` will be raised
        """
        # NaN protection
        if mod_cfg.nan_protection() and not all(torch.isfinite(val.total).all() for val in loss):
            loss_str = ", ".join(f"Loss {get_label(i, len(loss))}: {round(x.total.item(), 6)}"
                                 for i, x in enumerate(loss))
            msg = f"NaN Detected. {loss_str}"
            failed = ", ".join(f"{key}({get_label(i, len(loss))})"
                               for i, out in enumerate(loss)
                               for unweighted in out.unweighted
                               for key, sub_loss in unweighted.items()
                               if not torch.isfinite(sub_loss).all())
            if failed:
                msg += f". The loss function(s) that NaN'd: {failed}"
            logger.critical(msg)
            raise FaceswapError("A NaN was detected and you have NaN protection enabled. Training "
                                "has been terminated.")

        combined_loss = np.array([x.total.item() for x in loss], dtype=np.float32)
        self._model.add_history(combined_loss)
        logger.trace("[Trainer] original loss: %s, combined_loss: %s",  # type:ignore[attr-defined]
                     loss, combined_loss)
        return combined_loss

    def _print_loss(self, loss: np.ndarray) -> None:
        """Outputs the loss for the current iteration to the console.

        Parameters
        ----------
        The loss for each side. List should contain 2 ``floats`` side "a" in position 0 and side
        "b" in position 1.
         """
        output = ", ".join([f"Loss {side}: {side_loss:.5f}"
                            for side, side_loss in zip(("A", "B"), loss)])
        timestamp = time.strftime("%H:%M:%S")
        output = f"[{timestamp}] [#{self._model.iterations:05d}] {output}"
        print(f"{output}", end="\r")

    def _get_predictions(self, feed: torch.Tensor) -> npt.NDArray[np.float32]:
        """Obtain preview predictions from the model, chunking feeds into the model's batch size

        Parameters
        ----------
        feed
            The input tensor to obtain predictions from the model in shape (num_sides, N, height,
            width, 3)

        Returns
        -------
        The predictions from the model for the given preview feed
        """
        batch_size = self._plugin.batch_size
        ndim = 4 if mod_cfg.Loss.learn_mask() else 3
        retval = np.empty((feed.shape[0], feed.shape[1], self._out_size, self._out_size, ndim),
                          dtype=np.float32)
        for idx in range(0, feed.shape[1], batch_size):
            feed_batch = feed[:, idx:idx + batch_size]
            feed_size = feed_batch.shape[1]
            is_padded = feed_size < batch_size

            if is_padded:
                holder = torch.empty((feed_batch.shape[0], batch_size, *feed_batch.shape[2:]),
                                     dtype=feed.dtype)
                logger.debug("[Trainer] Padding undersized batch of shape %s to %s",
                             feed_batch.shape, holder.shape)
                holder[:, :feed_size] = feed_batch
                feed_batch = holder
            with torch.inference_mode():
                out = [x.cpu().numpy() for x in self._model.model(list(feed_batch))
                       if x.shape[1] == self._out_size]  # Filter multi-scale output
            if mod_cfg.Loss.learn_mask():  # Apply mask to alpha channel
                out = [np.concatenate(out[i:i + 2], axis=-1) for i in range(0, len(out), 2)]
            out_arr = np.stack(out, axis=0)
            if is_padded:
                out_arr = out_arr[:, :feed_size]
            retval[:, idx:idx + feed_size] = out_arr
        return retval

    def _update_viewers(self,  # pylint:disable=too-many-locals
                        viewer: Callable[[np.ndarray, str], None] | None,
                        do_timelapse: bool = False) -> None:
        """Update the preview viewer and timelapse output

        Parameters
        ----------
        viewer
            The function that will display the preview image
        do_timelapse
            ``True`` to generate a timelapse preview image
        """
        if (viewer is None or self._preview_loader is None) and not do_timelapse:
            return

        if do_timelapse:
            assert self._timelapse_loader is not None
            loader = self._timelapse_loader
        else:
            assert self._preview_loader is not None
            loader = self._preview_loader
        feed, target = next(loader)

        num_sides = feed.shape[0]
        ndim = 4 if mod_cfg.Loss.learn_mask() else 3
        predictions: npt.NDArray[np.float32] = np.empty((num_sides,
                                                         num_sides,
                                                         target.shape[1],
                                                         self._out_size,
                                                         self._out_size,
                                                         ndim),
                                                        dtype=np.float32)
        logger.debug("[Trainer] feed: %s, target: %s, predictions_holder: %s",
                     feed.shape, target.shape, predictions.shape)
        for side_idx in range(num_sides):
            rolled_feed = torch.roll(feed, shifts=side_idx, dims=0)
            pred = self._get_predictions(rolled_feed)
            for input_idx in range(num_sides):
                original_idx = (input_idx - side_idx) % num_sides
                predictions[original_idx, side_idx] = pred[input_idx]

        targets = target.cpu().numpy()
        if self._model.color_order == "rgb":
            predictions[..., :3] = predictions[..., 2::-1]
            targets[..., :3] = targets[..., 2::-1]
        logger.debug("[Trainer] Got preview images: predictions: %s, targets: %s",
                     format_array(predictions), format_array(targets))

        samples = self._samples.get_preview(predictions, targets)

        if do_timelapse:
            filename = os.path.join(self._timelapse_output, str(int(time.time())) + ".jpg")
            cv2.imwrite(filename, samples)
            logger.debug("[Trainer] Created time-lapse: '%s'", filename)
            return

        if viewer is not None:
            viewer(samples,
                   "Training - 'S': Save Now. 'R': Refresh Preview. 'M': Toggle Mask. 'F': "
                   "Toggle Screen Fit-Actual Size. 'ENTER': Save and Quit")

    def train_one_step(self,
                       viewer: Callable[[np.ndarray, str], None] | None,
                       do_timelapse: bool = False) -> None:
        """Running training on a batch of images for each side.

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
        viewer
            The function that will display the preview image
        do_timelapse
            ``True`` to generate a timelapse preview image
        """
        self._model.state.increment_iterations()
        logger.trace("[Trainer] Training one step: (iteration: %s)",  # type:ignore[attr-defined]
                     self._model.iterations)
        do_snapshot = (self._plugin.config.snapshot_interval != 0 and
                       self._model.iterations - 1 >= self._plugin.config.snapshot_interval and
                       (self._model.iterations - 1) % self._plugin.config.snapshot_interval == 0)
        self._warmup()
        loss = self.train_one_batch()
        self._log_tensorboard(loss)
        total_loss = self._collate_and_store_loss(loss)
        self._print_loss(total_loss)
        if do_snapshot:
            self._model.io.snapshot()
        self._update_viewers(viewer, do_timelapse)

    def _clear_tensorboard(self) -> None:
        """Stop Tensorboard logging.

        Tensorboard logging needs to be explicitly shutdown on training termination. Called from
        :class:`scripts.train.Train` when training is stopped.
         """
        if not self._tensorboard:
            return
        logger.debug("[Trainer] Ending Tensorboard Session: %s", self._tensorboard)
        self._tensorboard.on_train_end()

    def save(self, is_exit: bool = False) -> None:
        """Save the model

        Parameters
        ----------
        is_exit
            ``True`` if save has been called on model exit. Default: ``False``
        """
        self._model.io.save(is_exit=is_exit)
        assert self._tensorboard is not None
        self._tensorboard.on_save()
        if is_exit:
            self._clear_tensorboard()


__all__ = get_module_objects(__name__)
