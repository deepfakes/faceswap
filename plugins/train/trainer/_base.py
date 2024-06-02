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

import cv2
import numpy as np

import tensorflow as tf
from tensorflow.python.framework import (  # pylint:disable=no-name-in-module
    errors_impl as tf_errors)

from lib.image import hex_to_rgb
from lib.training import Feeder, LearningRateFinder
from lib.utils import FaceswapError, get_folder, get_image_paths
from plugins.train._config import Config

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

        self._model.state.add_session_batchsize(batch_size)
        self._images = images
        self._sides = sorted(key for key in self._images.keys())

        self._tensorboard = self._set_tensorboard()
        self._samples = _Samples(self._model,
                                 self._model.coverage_ratio,
                                 T.cast(int, self._config["mask_opacity"]),
                                 T.cast(str, self._config["mask_color"]))

        num_images = self._config.get("preview_images", 14)
        assert isinstance(num_images, int)
        self._timelapse = _Timelapse(self._model,
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

        if self._model.state.iterations == 0 and self._model.state.session_id == 1:
            lrf = LearningRateFinder(self._model, self._config, self._feeder)
            success = lrf.find()
            return self._config["lr_finder_mode"] == "graph_and_exit" or not success

        learning_rate = self._model.state.sessions[1]["config"]["learning_rate"]
        logger.info("Setting learning rate from Learning Rate Finder to %s",
                    f"{learning_rate:.1e}")
        return False

    def _set_tensorboard(self) -> tf.keras.callbacks.TensorBoard:
        """ Set up Tensorboard callback for logging loss.

        Bypassed if command line option "no-logs" has been selected.

        Returns
        -------
        :class:`tf.keras.callbacks.TensorBoard`
            Tensorboard object for the the current training session.
        """
        if self._model.state.current_session["no_logs"]:
            logger.verbose("TensorBoard logging disabled")  # type: ignore
            return None
        logger.debug("Enabling TensorBoard Logging")

        logger.debug("Setting up TensorBoard Logging")
        log_dir = os.path.join(str(self._model.io.model_dir),
                               f"{self._model.name}_logs",
                               f"session_{self._model.state.session_id}")
        tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                                     histogram_freq=0,  # Must be 0 or hangs
                                                     write_graph=True,
                                                     write_images=False,
                                                     update_freq="batch",
                                                     profile_batch=0,
                                                     embeddings_freq=0,
                                                     embeddings_metadata=None)
        tensorboard.set_model(self._model.model)
        tensorboard.on_train_begin(0)
        logger.verbose("Enabled TensorBoard Logging")  # type: ignore
        return tensorboard

    def toggle_mask(self) -> None:
        """ Toggle the mask overlay on or off based on user input. """
        self._samples.toggle_mask_display()

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

        model_inputs, model_targets = self._feeder.get_batch()

        try:
            loss: list[float] = self._model.model.train_on_batch(model_inputs, y=model_targets)
        except tf_errors.ResourceExhaustedError as err:
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

        # Bug in TF 2.8/2.9/2.10 where batch recording got deleted.
        # ref: https://github.com/keras-team/keras/issues/16173
        with tf.summary.record_if(True), self._tensorboard._train_writer.as_default():  # noqa:E501  pylint:disable=protected-access,not-context-manager
            for name, value in logs.items():
                tf.summary.scalar(
                    "batch_" + name,
                    value,
                    step=self._tensorboard._train_step)  # pylint:disable=protected-access
        # TODO revert this code if fixed in tensorflow
        # self._tensorboard.on_train_batch_end(self._model.iterations, logs=logs)

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
        try:
            print(f"\r{output}", end="")
        except OSError as err:
            logger.warning("Swallowed OS Error caused by Tensorflow distributed training. output "
                           "line: %s, error: %s", output, str(err))

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

    def clear_tensorboard(self) -> None:
        """ Stop Tensorboard logging.

        Tensorboard logging needs to be explicitly shutdown on training termination. Called from
        :class:`scripts.train.Train` when training is stopped.
         """
        if not self._tensorboard:
            return
        logger.debug("Ending Tensorboard Session: %s", self._tensorboard)
        self._tensorboard.on_train_end(None)


class _Samples():  # pylint:disable=too-few-public-methods
    """ Compile samples for display for preview and time-lapse

    Parameters
    ----------
    model: plugin from :mod:`plugins.train.model`
        The selected model that will be running this trainer
    coverage_ratio: float
        Ratio of face to be cropped out of the training image.
    mask_opacity: int
        The opacity (as a percentage) to use for the mask overlay
    mask_color: str
        The hex RGB value to use the mask overlay

    Attributes
    ----------
    images: dict
        The :class:`numpy.ndarray` training images for generating previews on each side. The
        dictionary should contain 2 keys ("a" and "b") with the values being the training images
        for generating samples corresponding to each side.
    """
    def __init__(self,
                 model: ModelBase,
                 coverage_ratio: float,
                 mask_opacity: int,
                 mask_color: str) -> None:
        logger.debug("Initializing %s: model: '%s', coverage_ratio: %s, mask_opacity: %s, "
                     "mask_color: %s)",
                     self.__class__.__name__, model, coverage_ratio, mask_opacity, mask_color)
        self._model = model
        self._display_mask = model.config["learn_mask"] or model.config["penalized_mask_loss"]
        self.images: dict[T.Literal["a", "b"], list[np.ndarray]] = {}
        self._coverage_ratio = coverage_ratio
        self._mask_opacity = mask_opacity / 100.0
        self._mask_color = np.array(hex_to_rgb(mask_color))[..., 2::-1] / 255.
        logger.debug("Initialized %s", self.__class__.__name__)

    def toggle_mask_display(self) -> None:
        """ Toggle the mask overlay on or off depending on user input. """
        if not (self._model.config["learn_mask"] or self._model.config["penalized_mask_loss"]):
            return
        display_mask = not self._display_mask
        print("")  # Break to not garble loss output
        logger.info("Toggling mask display %s...", "on" if display_mask else "off")
        self._display_mask = display_mask

    def show_sample(self) -> np.ndarray:
        """ Compile a preview image.

        Returns
        -------
        :class:`numpy.ndarry`
            A compiled preview image ready for display or saving
        """
        logger.debug("Showing sample")
        feeds: dict[T.Literal["a", "b"], np.ndarray] = {}
        for idx, side in enumerate(T.get_args(T.Literal["a", "b"])):
            feed = self.images[side][0]
            input_shape = self._model.model.input_shape[idx][1:]
            if input_shape[0] / feed.shape[1] != 1.0:
                feeds[side] = self._resize_sample(side, feed, input_shape[0])
            else:
                feeds[side] = feed

        preds = self._get_predictions(feeds["a"], feeds["b"])
        return self._compile_preview(preds)

    @classmethod
    def _resize_sample(cls,
                       side: T.Literal["a", "b"],
                       sample: np.ndarray,
                       target_size: int) -> np.ndarray:
        """ Resize a given image to the target size.

        Parameters
        ----------
        side: str
            The side ("a" or "b") that the samples are being generated for
        sample: :class:`numpy.ndarray`
            The sample to be resized
        target_size: int
            The size that the sample should be resized to

        Returns
        -------
        :class:`numpy.ndarray`
            The sample resized to the target size
        """
        scale = target_size / sample.shape[1]
        if scale == 1.0:
            # cv2 complains if we don't do this :/
            return np.ascontiguousarray(sample)
        logger.debug("Resizing sample: (side: '%s', sample.shape: %s, target_size: %s, scale: %s)",
                     side, sample.shape, target_size, scale)
        interpn = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
        retval = np.array([cv2.resize(img, (target_size, target_size), interpolation=interpn)
                           for img in sample])
        logger.debug("Resized sample: (side: '%s' shape: %s)", side, retval.shape)
        return retval

    def _get_predictions(self, feed_a: np.ndarray, feed_b: np.ndarray) -> dict[str, np.ndarray]:
        """ Feed the samples to the model and return predictions

        Parameters
        ----------
        feed_a: :class:`numpy.ndarray`
            Feed images for the "a" side
        feed_a: :class:`numpy.ndarray`
            Feed images for the "b" side

        Returns
        -------
        list:
            List of :class:`numpy.ndarray` of predictions received from the model
        """
        logger.debug("Getting Predictions")
        preds: dict[str, np.ndarray] = {}

        # Calling model.predict() can lead to both VRAM and system memory leaks, so call model
        # directly
        standard = self._model.model([feed_a, feed_b])
        swapped = self._model.model([feed_b, feed_a])

        if self._model.config["learn_mask"]:  # Add mask to 4th channel of final output
            standard = [np.concatenate(side[-2:], axis=-1)
                        for side in [[s.numpy() for s in t] for t in standard]]
            swapped = [np.concatenate(side[-2:], axis=-1)
                       for side in [[s.numpy() for s in t] for t in swapped]]
        else:  # Retrieve final output
            standard = [side[-1] if isinstance(side, list) else side
                        for side in [t.numpy() for t in standard]]
            swapped = [side[-1] if isinstance(side, list) else side
                       for side in [t.numpy() for t in swapped]]

        preds["a_a"] = standard[0]
        preds["b_b"] = standard[1]
        preds["a_b"] = swapped[0]
        preds["b_a"] = swapped[1]

        logger.debug("Returning predictions: %s", {key: val.shape for key, val in preds.items()})
        return preds

    def _compile_preview(self, predictions: dict[str, np.ndarray]) -> np.ndarray:
        """ Compile predictions and images into the final preview image.

        Parameters
        ----------
        predictions: dict
            The predictions from the model

        Returns
        -------
        :class:`numpy.ndarry`
            A compiled preview image ready for display or saving
        """
        figures: dict[T.Literal["a", "b"], np.ndarray] = {}
        headers: dict[T.Literal["a", "b"], np.ndarray] = {}

        for side, samples in self.images.items():
            other_side = "a" if side == "b" else "b"
            preds = [predictions[f"{side}_{side}"],
                     predictions[f"{other_side}_{side}"]]
            display = self._to_full_frame(side, samples, preds)
            headers[side] = self._get_headers(side, display[0].shape[1])
            figures[side] = np.stack([display[0], display[1], display[2], ], axis=1)
            if self.images[side][1].shape[0] % 2 == 1:
                figures[side] = np.concatenate([figures[side],
                                                np.expand_dims(figures[side][0], 0)])

        width = 4
        if width // 2 != 1:
            headers = self._duplicate_headers(headers, width // 2)

        header = np.concatenate([headers["a"], headers["b"]], axis=1)
        figure = np.concatenate([figures["a"], figures["b"]], axis=0)
        height = int(figure.shape[0] / width)
        figure = figure.reshape((width, height) + figure.shape[1:])
        figure = _stack_images(figure)
        figure = np.concatenate((header, figure), axis=0)

        logger.debug("Compiled sample")
        return np.clip(figure * 255, 0, 255).astype('uint8')

    def _to_full_frame(self,
                       side: T.Literal["a", "b"],
                       samples: list[np.ndarray],
                       predictions: list[np.ndarray]) -> list[np.ndarray]:
        """ Patch targets and prediction images into images of model output size.

        Parameters
        ----------
        side: {"a" or "b"}
            The side that these samples are for
        samples: list
            List of :class:`numpy.ndarray` of feed images and sample images
        predictions: list
            List of :class: `numpy.ndarray` of predictions from the model

        Returns
        -------
        list
            The images resized and collated for display in the preview frame
        """
        logger.debug("side: '%s', number of sample arrays: %s, prediction.shapes: %s)",
                     side, len(samples), [pred.shape for pred in predictions])
        faces, full = samples[:2]

        if self._model.color_order.lower() == "rgb":  # Switch color order for RGB model display
            full = full[..., ::-1]
            faces = faces[..., ::-1]
            predictions = [pred[..., 2::-1] for pred in predictions]

        full = self._process_full(side, full, predictions[0].shape[1], (0., 0., 1.0))
        images = [faces] + predictions

        if self._display_mask:
            images = self._compile_masked(images, samples[-1])
        elif self._model.config["learn_mask"]:
            # Remove masks when learn mask is selected but mask toggle is off
            images = [batch[..., :3] for batch in images]

        images = [self._overlay_foreground(full.copy(), image) for image in images]

        return images

    def _process_full(self,
                      side: T.Literal["a", "b"],
                      images: np.ndarray,
                      prediction_size: int,
                      color: tuple[float, float, float]) -> np.ndarray:
        """ Add a frame overlay to preview images indicating the region of interest.

        This applies the red border that appears in the preview images.

        Parameters
        ----------
        side: {"a" or "b"}
            The side that these samples are for
        images: :class:`numpy.ndarray`
            The input training images to to process
        prediction_size: int
            The size of the predicted output from the model
        color: tuple
            The (Blue, Green, Red) color to use for the frame

        Returns
        -------
        :class:`numpy,ndarray`
            The input training images, sized for output and annotated for coverage
        """
        logger.debug("full_size: %s, prediction_size: %s, color: %s",
                     images.shape[1], prediction_size, color)

        display_size = int((prediction_size / self._coverage_ratio // 2) * 2)
        images = self._resize_sample(side, images, display_size)  # Resize targets to display size
        padding = (display_size - prediction_size) // 2
        if padding == 0:
            logger.debug("Resized background. Shape: %s", images.shape)
            return images

        length = display_size // 4
        t_l, b_r = (padding - 1, display_size - padding)
        for img in images:
            cv2.rectangle(img, (t_l, t_l), (t_l + length, t_l + length), color, 1)
            cv2.rectangle(img, (b_r, t_l), (b_r - length, t_l + length), color, 1)
            cv2.rectangle(img, (b_r, b_r), (b_r - length, b_r - length), color, 1)
            cv2.rectangle(img, (t_l, b_r), (t_l + length, b_r - length), color, 1)
        logger.debug("Overlayed background. Shape: %s", images.shape)
        return images

    def _compile_masked(self, faces: list[np.ndarray], masks: np.ndarray) -> list[np.ndarray]:
        """ Add the mask to the faces for masked preview.

        Places an opaque red layer over areas of the face that are masked out.

        Parameters
        ----------
        faces: list
            The :class:`numpy.ndarray` sample faces and predictions that are to have the mask
            applied
        masks: :class:`numpy.ndarray`
            The masks that are to be applied to the faces

        Returns
        -------
        list
            List of :class:`numpy.ndarray` faces with the opaque mask layer applied
        """
        orig_masks = 1. - masks
        masks3: list[np.ndarray] | np.ndarray = []

        if faces[-1].shape[-1] == 4:  # Mask contained in alpha channel of predictions
            pred_masks = [1. - face[..., -1][..., None] for face in faces[-2:]]
            faces[-2:] = [face[..., :-1] for face in faces[-2:]]
            masks3 = [orig_masks, *pred_masks]
        else:
            masks3 = np.repeat(np.expand_dims(orig_masks, axis=0), 3, axis=0)

        retval: list[np.ndarray] = []
        overlays3 = np.ones_like(faces) * self._mask_color
        for previews, overlays, compiled_masks in zip(faces, overlays3, masks3):
            compiled_masks *= self._mask_opacity
            overlays *= compiled_masks
            previews *= (1. - compiled_masks)
            retval.append(previews + overlays)
        logger.debug("masked shapes: %s", [faces.shape for faces in retval])
        return retval

    @classmethod
    def _overlay_foreground(cls, backgrounds: np.ndarray, foregrounds: np.ndarray) -> np.ndarray:
        """ Overlay the preview images into the center of the background images

        Parameters
        ----------
        backgrounds: :class:`numpy.ndarray`
            Background images for placing the preview images onto
        backgrounds: :class:`numpy.ndarray`
            Preview images for placing onto the background images

        Returns
        -------
        :class:`numpy.ndarray`
            The preview images compiled into the full frame size for each preview
        """
        offset = (backgrounds.shape[1] - foregrounds.shape[1]) // 2
        for foreground, background in zip(foregrounds, backgrounds):
            background[offset:offset + foreground.shape[0],
                       offset:offset + foreground.shape[1], :3] = foreground
        logger.debug("Overlayed foreground. Shape: %s", backgrounds.shape)
        return backgrounds

    @classmethod
    def _get_headers(cls, side: T.Literal["a", "b"], width: int) -> np.ndarray:
        """ Set header row for the final preview frame

        Parameters
        ----------
        side: {"a" or "b"}
            The side that the headers should be generated for
        width: int
            The width of each column in the preview frame

        Returns
        -------
        :class:`numpy.ndarray`
            The column headings for the given side
        """
        logger.debug("side: '%s', width: %s",
                     side, width)
        titles = ("Original", "Swap") if side == "a" else ("Swap", "Original")
        height = int(width / 4.5)
        total_width = width * 3
        logger.debug("height: %s, total_width: %s", height, total_width)
        font = cv2.FONT_HERSHEY_SIMPLEX
        texts = [f"{titles[0]} ({side.upper()})",
                 f"{titles[0]} > {titles[0]}",
                 f"{titles[0]} > {titles[1]}"]
        scaling = (width / 144) * 0.45
        text_sizes = [cv2.getTextSize(texts[idx], font, scaling, 1)[0]
                      for idx in range(len(texts))]
        text_y = int((height + text_sizes[0][1]) / 2)
        text_x = [int((width - text_sizes[idx][0]) / 2) + width * idx
                  for idx in range(len(texts))]
        logger.debug("texts: %s, text_sizes: %s, text_x: %s, text_y: %s",
                     texts, text_sizes, text_x, text_y)
        header_box = np.ones((height, total_width, 3), np.float32)
        for idx, text in enumerate(texts):
            cv2.putText(header_box,
                        text,
                        (text_x[idx], text_y),
                        font,
                        scaling,
                        (0, 0, 0),
                        1,
                        lineType=cv2.LINE_AA)
        logger.debug("header_box.shape: %s", header_box.shape)
        return header_box

    @classmethod
    def _duplicate_headers(cls,
                           headers: dict[T.Literal["a", "b"], np.ndarray],
                           columns: int) -> dict[T.Literal["a", "b"], np.ndarray]:
        """ Duplicate headers for the number of columns displayed for each side.

        Parameters
        ----------
        headers: dict
            The headers to be duplicated for each side
        columns: int
            The number of columns that the header needs to be duplicated for

        Returns
        -------
        :class:dict
            The original headers duplicated by the number of columns for each side
        """
        for side, header in headers.items():
            duped = tuple(header for _ in range(columns))
            headers[side] = np.concatenate(duped, axis=1)
            logger.debug("side: %s header.shape: %s", side, header.shape)
        return headers


class _Timelapse():  # pylint:disable=too-few-public-methods
    """ Create a time-lapse preview image.

    Parameters
    ----------
    model: plugin from :mod:`plugins.train.model`
        The selected model that will be running this trainer
    coverage_ratio: float
        Ratio of face to be cropped out of the training image.
    image_count: int
        The number of preview images to be displayed in the time-lapse
    mask_opacity: int
        The opacity (as a percentage) to use for the mask overlay
    mask_color: str
        The hex RGB value to use the mask overlay
    feeder: :class:`~lib.training.generator.Feeder`
        The feeder for generating the time-lapse images.
    image_paths: dict
        The full paths to the training images for each side of the model
    """
    def __init__(self,
                 model: ModelBase,
                 coverage_ratio: float,
                 image_count: int,
                 mask_opacity: int,
                 mask_color: str,
                 feeder: Feeder,
                 image_paths: dict[T.Literal["a", "b"], list[str]]) -> None:
        logger.debug("Initializing %s: model: %s, coverage_ratio: %s, image_count: %s, "
                     "mask_opacity: %s, mask_color: %s, feeder: %s, image_paths: %s)",
                     self.__class__.__name__, model, coverage_ratio, image_count, mask_opacity,
                     mask_color, feeder, len(image_paths))
        self._num_images = image_count
        self._samples = _Samples(model, coverage_ratio, mask_opacity, mask_color)
        self._model = model
        self._feeder = feeder
        self._image_paths = image_paths
        self._output_file = ""
        logger.debug("Initialized %s", self.__class__.__name__)

    def _setup(self, input_a: str, input_b: str, output: str) -> None:
        """ Setup the time-lapse folder locations and the time-lapse feed.

        Parameters
        ----------
        input_a: str
            The full path to the time-lapse input folder containing faces for the "a" side
        input_b: str
            The full path to the time-lapse input folder containing faces for the "b" side
        output: str, optional
            The full path to the time-lapse output folder. If ``None`` is provided this will
            default to the model folder
        """
        logger.debug("Setting up time-lapse")
        if not output:
            output = get_folder(os.path.join(str(self._model.io.model_dir),
                                             f"{self._model.name}_timelapse"))
        self._output_file = output
        logger.debug("Time-lapse output set to '%s'", self._output_file)

        # Rewrite paths to pull from the training images so mask and face data can be accessed
        images: dict[T.Literal["a", "b"], list[str]] = {}
        for side, input_ in zip(T.get_args(T.Literal["a", "b"]), (input_a, input_b)):
            training_path = os.path.dirname(self._image_paths[side][0])
            images[side] = [os.path.join(training_path, os.path.basename(pth))
                            for pth in get_image_paths(input_)]

        batchsize = min(len(images["a"]),
                        len(images["b"]),
                        self._num_images)
        self._feeder.set_timelapse_feed(images, batchsize)
        logger.debug("Set up time-lapse")

    def output_timelapse(self, timelapse_kwargs: dict[T.Literal["input_a",
                                                                "input_b",
                                                                "output"], str]) -> None:
        """ Generate the time-lapse samples and output the created time-lapse to the specified
        output folder.

        Parameters
        ----------
        timelapse_kwargs: dict:
            The keyword arguments for setting up the time-lapse. All values should be full paths
            the keys being `input_a`, `input_b`, `output`
        """
        logger.debug("Ouputting time-lapse")
        if not self._output_file:
            self._setup(**T.cast(dict[str, str], timelapse_kwargs))

        logger.debug("Getting time-lapse samples")
        self._samples.images = self._feeder.generate_preview(is_timelapse=True)
        logger.debug("Got time-lapse samples: %s",
                     {side: len(images) for side, images in self._samples.images.items()})

        image = self._samples.show_sample()
        if image is None:
            return
        filename = os.path.join(self._output_file, str(int(time.time())) + ".jpg")

        cv2.imwrite(filename, image)
        logger.debug("Created time-lapse: '%s'", filename)


def _stack_images(images: np.ndarray) -> np.ndarray:
    """ Stack images evenly for preview.

    Parameters
    ----------
    images: :class:`numpy.ndarray`
        The preview images to be stacked

    Returns
    -------
    :class:`numpy.ndarray`
        The stacked preview images
    """
    logger.debug("Stack images")

    def get_transpose_axes(num):
        if num % 2 == 0:
            logger.debug("Even number of images to stack")
            y_axes = list(range(1, num - 1, 2))
            x_axes = list(range(0, num - 1, 2))
        else:
            logger.debug("Odd number of images to stack")
            y_axes = list(range(0, num - 1, 2))
            x_axes = list(range(1, num - 1, 2))
        return y_axes, x_axes, [num - 1]

    images_shape = np.array(images.shape)
    new_axes = get_transpose_axes(len(images_shape))
    new_shape = [np.prod(images_shape[x]) for x in new_axes]
    logger.debug("Stacked images")
    return np.transpose(images, axes=np.concatenate(new_axes)).reshape(new_shape)
