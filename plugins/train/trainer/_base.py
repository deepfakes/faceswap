#!/usr/bin/env python3
""" Base Class for Faceswap Trainer plugins. All Trainer plugins should be inherited from
this class.

At present there is only the :class:`~plugins.train.trainer.original` plugin, so that entirely
inherits from this class. If further plugins are developed, then common code should be kept here,
with "original" unique code split out to the original plugin.
"""

# pylint:disable=too-many-lines
import logging
import os
import time

import cv2
import numpy as np

import tensorflow as tf
from tensorflow.python.framework import (  # pylint:disable=no-name-in-module
    errors_impl as tf_errors)

from lib.training import TrainingDataGenerator
from lib.utils import FaceswapError, get_backend, get_folder, get_image_paths, get_tf_version
from plugins.train._config import Config

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def _get_config(plugin_name, configfile=None):
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
    :class:`lib.config.FaceswapConfig`
        The configuration file for the requested plugin
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

    def __init__(self, model, images, batch_size, configfile):
        logger.debug("Initializing %s: (model: '%s', batch_size: %s)",
                     self.__class__.__name__, model, batch_size)
        self._model = model
        self._config = self._get_config(configfile)

        self._model.state.add_session_batchsize(batch_size)
        self._images = images
        self._sides = sorted(key for key in self._images.keys())

        self._feeder = _Feeder(images, self._model, batch_size, self._config)

        self._tensorboard = self._set_tensorboard()
        self._samples = _Samples(self._model,
                                 self._model.coverage_ratio,
                                 self._model.command_line_arguments.preview_scale / 100)
        self._timelapse = _Timelapse(self._model,
                                     self._model.coverage_ratio,
                                     self._config.get("preview_images", 14),
                                     self._feeder,
                                     self._images)
        logger.debug("Initialized %s", self.__class__.__name__)

    def _get_config(self, configfile):
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

    def _set_tensorboard(self):
        """ Set up Tensorboard callback for logging loss.

        Bypassed if command line option "no-logs" has been selected.

        Returns
        -------
        :class:`tf.keras.callbacks.TensorBoard`
            Tensorboard object for the the current training session.
        """
        if self._model.state.current_session["no_logs"]:
            logger.verbose("TensorBoard logging disabled")
            return None
        logger.debug("Enabling TensorBoard Logging")

        logger.debug("Setting up TensorBoard Logging")
        log_dir = os.path.join(str(self._model.model_dir),
                               f"{self._model.name}_logs",
                               f"session_{self._model.state.session_id}")
        tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                                     histogram_freq=0,  # Must be 0 or hangs
                                                     write_graph=get_backend() != "amd",
                                                     write_images=False,
                                                     update_freq="batch",
                                                     profile_batch=0,
                                                     embeddings_freq=0,
                                                     embeddings_metadata=None)
        tensorboard.set_model(self._model.model)
        tensorboard.on_train_begin(0)
        logger.verbose("Enabled TensorBoard Logging")
        return tensorboard

    def toggle_mask(self):
        """ Toggle the mask overlay on or off based on user input. """
        self._samples.toggle_mask_display()

    def train_one_step(self, viewer, timelapse_kwargs):
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
        viewer: :func:`scripts.train.Train._show`
            The function that will display the preview image
        timelapse_kwargs: dict
            The keyword arguments for generating time-lapse previews. If a time-lapse preview is
            not required then this should be ``None``. Otherwise all values should be full paths
            the keys being `input_a`, `input_b`, `output`.
        """
        self._model.state.increment_iterations()
        logger.trace("Training one step: (iteration: %s)", self._model.iterations)
        do_preview = viewer is not None
        snapshot_interval = self._model.command_line_arguments.snapshot_interval
        do_snapshot = (snapshot_interval != 0 and
                       self._model.iterations - 1 >= snapshot_interval and
                       (self._model.iterations - 1) % snapshot_interval == 0)

        model_inputs, model_targets = self._feeder.get_batch()
        try:
            loss = self._model.model.train_on_batch(model_inputs, y=model_targets)
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
        except Exception as err:
            if get_backend() == "amd":
                # pylint:disable=import-outside-toplevel
                from lib.plaidml_utils import is_plaidml_error
                if (is_plaidml_error(err) and (
                        "CL_MEM_OBJECT_ALLOCATION_FAILURE" in str(err).upper() or
                        "enough memory for the current schedule" in str(err).lower())):
                    msg = ("You do not have enough GPU memory available to train the selected "
                           "model at the selected settings. You can try a number of things:"
                           "\n1) Close any other application that is using your GPU (web browsers "
                           "are particularly bad for this)."
                           "\n2) Lower the batchsize (the amount of images fed into the model "
                           "each iteration)."
                           "\n3) Use a more lightweight model, or select the model's 'LowMem' "
                           "option (in config) if it has one.")
                    raise FaceswapError(msg) from err
            raise
        self._log_tensorboard(loss)
        loss = self._collate_and_store_loss(loss[1:])
        self._print_loss(loss)

        if do_snapshot:
            self._model.snapshot()

        if do_preview:
            self._feeder.generate_preview(do_preview)
            self._samples.images = self._feeder.compile_sample(None)
            samples = self._samples.show_sample()
            if samples is not None:
                viewer(samples,
                       "Training - 'S': Save Now. 'R': Refresh Preview. 'M': Toggle Mask. "
                       "'ENTER': Save and Quit")

        if timelapse_kwargs:
            self._timelapse.output_timelapse(timelapse_kwargs)

    def _log_tensorboard(self, loss):
        """ Log current loss to Tensorboard log files

        Parameters
        ----------
        loss: list
            The list of loss ``floats`` output from the model
        """
        if not self._tensorboard:
            return
        logger.trace("Updating TensorBoard log")
        logs = {log[0]: log[1]
                for log in zip(self._model.state.loss_names, loss)}

        self._tensorboard.on_train_batch_end(self._model.iterations, logs=logs)
        if get_tf_version() == 2.8:
            # Bug in TF 2.8 where batch recording got deleted.
            # ref: https://github.com/keras-team/keras/issues/16173
            for name, value in logs.items():
                tf.summary.scalar(
                    "batch_" + name,
                    value,
                    step=self._model._model._train_counter)  # pylint:disable=protected-access

    def _collate_and_store_loss(self, loss):
        """ Collate the loss into totals for each side.

        The losses are summed into a total for each side. Loss totals are added to
        :attr:`model.state._history` to track the loss drop per save iteration for backup purposes.

        If NaN protection is enabled, Checks for NaNs and raises an error if detected.

        Parameters
        ----------
        loss: list
            The list of loss ``floats`` for this iteration.

        Returns
        -------
        list
            List of 2 ``floats`` which is the total loss for each side

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
        logger.trace("original loss: %s, comibed_loss: %s", loss, combined_loss)
        return combined_loss

    def _print_loss(self, loss):
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

    def clear_tensorboard(self):
        """ Stop Tensorboard logging.

        Tensorboard logging needs to be explicitly shutdown on training termination. Called from
        :class:`scripts.train.Train` when training is stopped.
         """
        if not self._tensorboard:
            return
        logger.debug("Ending Tensorboard Session: %s", self._tensorboard)
        self._tensorboard.on_train_end(None)


class _Feeder():
    """ Handles the processing of a Batch for training the model and generating samples.

    Parameters
    ----------
    images: dict
        The list of full paths to the training images for this :class:`_Feeder` for each side
    model: plugin from :mod:`plugins.train.model`
        The selected model that will be running this trainer
    batch_size: int
        The size of the batch to be processed for each side at each iteration
    config: :class:`lib.config.FaceswapConfig`
        The configuration for this trainer
    """
    def __init__(self, images, model, batch_size, config):
        logger.debug("Initializing %s: num_images: %s, batch_size: %s, config: %s)",
                     self.__class__.__name__, len(images), batch_size, config)
        self._model = model
        self._images = images
        self._config = config
        self._target = {}
        self._samples = {}
        self._masks = {}

        self._feeds = {side: self._load_generator(idx).minibatch_ab(images[side], batch_size, side)
                       for idx, side in enumerate(("a", "b"))}

        self._display_feeds = dict(preview=self._set_preview_feed(), timelapse={})
        logger.debug("Initialized %s:", self.__class__.__name__)

    def _load_generator(self, output_index):
        """ Load the :class:`~lib.training_data.TrainingDataGenerator` for this feeder.

        Parameters
        ----------
        output_index: int
            The output index from the model to get output shapes for

        Returns
        -------
        :class:`~lib.training_data.TrainingDataGenerator`
            The training data generator
        """
        logger.debug("Loading generator")
        input_size = self._model.model.input_shape[output_index][1]
        output_shapes = self._model.output_shapes[output_index]
        logger.debug("input_size: %s, output_shapes: %s", input_size, output_shapes)
        generator = TrainingDataGenerator(input_size,
                                          output_shapes,
                                          self._model.coverage_ratio,
                                          self._model.color_order,
                                          not self._model.command_line_arguments.no_augment_color,
                                          self._model.command_line_arguments.no_flip,
                                          self._model.command_line_arguments.no_warp,
                                          self._model.command_line_arguments.warp_to_landmarks,
                                          self._config)
        return generator

    def _set_preview_feed(self):
        """ Set the preview feed for this feeder.

        Creates a generator from :class:`lib.training_data.TrainingDataGenerator` specifically
        for previews for the feeder.

        Returns
        -------
        dict
            The side ("a" or "b") as key, :class:`~lib.training_data.TrainingDataGenerator` as
            value.
        """
        retval = {}
        for idx, side in enumerate(("a", "b")):
            logger.debug("Setting preview feed: (side: '%s')", side)
            preview_images = self._config.get("preview_images", 14)
            preview_images = min(max(preview_images, 2), 16)
            batchsize = min(len(self._images[side]), preview_images)
            retval[side] = self._load_generator(idx).minibatch_ab(self._images[side],
                                                                  batchsize,
                                                                  side,
                                                                  do_shuffle=True,
                                                                  is_preview=True)
        logger.debug("Set preview feed. Batchsize: %s", batchsize)
        return retval

    def get_batch(self):
        """ Get the feed data and the targets for each training side for feeding into the model's
        train function.

        Returns
        -------
        model_inputs: list
            The inputs to the model for each side A and B
        model_targets: list
            The targets for the model for each side A and B
        """
        model_inputs = []
        model_targets = []
        for side in ("a", "b"):
            batch = next(self._feeds[side])
            side_inputs = batch["feed"]
            side_targets = self._compile_mask_targets(batch["targets"],
                                                      batch["masks"],
                                                      batch.get("additional_masks", None))
            if self._model.config["learn_mask"]:
                side_targets = side_targets + [batch["masks"]]
            logger.trace("side: %s, input_shapes: %s, target_shapes: %s",
                         side, [i.shape for i in side_inputs], [i.shape for i in side_targets])
            if get_backend() == "amd":
                model_inputs.extend(side_inputs)
                model_targets.extend(side_targets)
            else:
                model_inputs.append(side_inputs)
                model_targets.append(side_targets)
        return model_inputs, model_targets

    def _compile_mask_targets(self, targets, masks, additional_masks):
        """ Compile the masks into the targets for penalized loss and for targeted learning.

        Penalized loss expects the target mask to be included for all outputs in the 4th channel
        of the targets. Any additional masks are placed into subsequent channels for extraction
        by the relevant loss functions.

        Parameters
        ----------
        targets: list
            The targets for the model, with the mask as the final entry in the list
        masks: list
            The masks for the model
        additional_masks: list or ``None``
            Any additional masks for the model, or ``None`` if no additional masks are required

        Returns
        -------
        list
            The targets for the model with the mask compiled into the 4th channel. The original
            mask is still output as the final item in the list
        """
        if not self._model.config["penalized_mask_loss"] and additional_masks is None:
            logger.trace("No masks to compile. Returning targets")
            return targets

        if not self._model.config["penalized_mask_loss"] and additional_masks is not None:
            masks = additional_masks
        elif additional_masks is not None:
            masks = np.concatenate((masks, additional_masks), axis=-1)

        for idx, tgt in enumerate(targets):
            tgt_dim = tgt.shape[1]
            if tgt_dim == masks.shape[1]:
                add_masks = masks
            else:
                add_masks = np.array([cv2.resize(mask, (tgt_dim, tgt_dim))
                                      for mask in masks])
                if add_masks.ndim == 3:
                    add_masks = add_masks[..., None]
            targets[idx] = np.concatenate((tgt, add_masks), axis=-1)
        logger.trace("masks added to targets: %s", [tgt.shape for tgt in targets])
        return targets

    def generate_preview(self, do_preview):
        """ Generate the preview images.

        Parameters
        ----------
        do_preview: bool
            Whether the previews should be generated. ``True`` if they should ``False`` if they
            should not be generated, in which case currently stored previews should be deleted.
        """
        if not do_preview:
            self._samples = {}
            self._target = {}
            self._masks = {}
            return
        logger.debug("Generating preview")
        for side in ("a", "b"):
            batch = next(self._display_feeds["preview"][side])
            self._samples[side] = batch["samples"]
            self._target[side] = batch["targets"][-1]
            self._masks[side] = batch["masks"]

    def compile_sample(self, batch_size, samples=None, images=None, masks=None):
        """ Compile the preview samples for display.

        Parameters
        ----------
        batch_size: int
            The requested batch size for each training iterations
        samples: dict, optional
            Dictionary for side "a", "b" of :class:`numpy.ndarray`. The sample images that should
            be used for creating the preview. If ``None`` then the samples will be generated from
            the internal random image generator. Default: ``None``
        images: dict, optional
            Dictionary for side "a", "b" of :class:`numpy.ndarray`. The target images that should
            be used for creating the preview. If ``None`` then the targets will be generated from
            the internal random image generator. Default: ``None``
        masks: dict, optional
            Dictionary for side "a", "b" of :class:`numpy.ndarray`. The masks that should be used
            for creating the preview. If ``None`` then the masks will be generated from the
            internal random image generator. Default: ``None``

        Returns
        -------
        list
            The list of samples, targets and masks as :class:`numpy.ndarrays` for creating a
            preview image
         """
        num_images = self._config.get("preview_images", 14)
        num_images = min(batch_size, num_images) if batch_size is not None else num_images
        retval = {}
        for side in ("a", "b"):
            logger.debug("Compiling samples: (side: '%s', samples: %s)", side, num_images)
            side_images = images[side] if images is not None else self._target[side]
            side_masks = masks[side] if masks is not None else self._masks[side]
            side_samples = samples[side] if samples is not None else self._samples[side]
            retval[side] = [side_samples[0:num_images],
                            side_images[0:num_images],
                            side_masks[0:num_images]]
        return retval

    def compile_timelapse_sample(self):
        """ Compile the sample images for creating a time-lapse frame.

        Returns
        -------
        dict
            For sides "a" and "b"; The list of samples, targets and masks as
            :class:`numpy.ndarrays` for creating a time-lapse frame
        """
        batchsizes = []
        samples = {}
        images = {}
        masks = {}
        for side in ("a", "b"):
            batch = next(self._display_feeds["timelapse"][side])
            batchsizes.append(len(batch["samples"]))
            samples[side] = batch["samples"]
            images[side] = batch["targets"][-1]
            masks[side] = batch["masks"]
        batchsize = min(batchsizes)
        sample = self.compile_sample(batchsize, samples=samples, images=images, masks=masks)
        return sample

    def set_timelapse_feed(self, images, batch_size):
        """ Set the time-lapse feed for this feeder.

        Creates a generator from :class:`lib.training_data.TrainingDataGenerator` specifically
        for generating time-lapse previews for the feeder.

        Parameters
        ----------
        images: list
            The list of full paths to the images for creating the time-lapse for this
            :class:`_Feeder`
        batch_size: int
            The number of images to be used to create the time-lapse preview.
        """
        logger.debug("Setting time-lapse feed: (input_images: '%s', batch_size: %s)",
                     images, batch_size)
        for idx, side in enumerate(("a", "b")):
            self._display_feeds["timelapse"][side] = self._load_generator(idx).minibatch_ab(
                images[side][:batch_size],
                batch_size,
                side,
                do_shuffle=False,
                is_timelapse=True)
        logger.debug("Set time-lapse feed: %s", self._display_feeds["timelapse"])


class _Samples():  # pylint:disable=too-few-public-methods
    """ Compile samples for display for preview and time-lapse

    Parameters
    ----------
    model: plugin from :mod:`plugins.train.model`
        The selected model that will be running this trainer
    coverage_ratio: float
        Ratio of face to be cropped out of the training image.
    scaling: float, optional
        The amount to scale the final preview image by. Default: `1.0`

    Attributes
    ----------
    images: dict
        The :class:`numpy.ndarray` training images for generating previews on each side. The
        dictionary should contain 2 keys ("a" and "b") with the values being the training images
        for generating samples corresponding to each side.
    """
    def __init__(self, model, coverage_ratio, scaling=1.0):
        logger.debug("Initializing %s: model: '%s', coverage_ratio: %s)",
                     self.__class__.__name__, model, coverage_ratio)
        self._model = model
        self._display_mask = model.config["learn_mask"] or model.config["penalized_mask_loss"]
        self.images = {}
        self._coverage_ratio = coverage_ratio
        self._scaling = scaling
        logger.debug("Initialized %s", self.__class__.__name__)

    def toggle_mask_display(self):
        """ Toggle the mask overlay on or off depending on user input. """
        if not (self._model.config["learn_mask"] or self._model.config["penalized_mask_loss"]):
            return
        display_mask = not self._display_mask
        print("\n")  # Break to not garble loss output
        logger.info("Toggling mask display %s...", "on" if display_mask else "off")
        self._display_mask = display_mask

    def show_sample(self):
        """ Compile a preview image.

        Returns
        -------
        :class:`numpy.ndarry`
            A compiled preview image ready for display or saving
        """
        logger.debug("Showing sample")
        feeds = {}
        figures = {}
        headers = {}
        for idx, side in enumerate(("a", "b")):
            samples = self.images[side]
            faces = samples[1]
            input_shape = self._model.model.input_shape[idx][1:]
            if input_shape[0] / faces.shape[1] != 1.0:
                feeds[side] = self._resize_sample(side, faces, input_shape[0])
                feeds[side] = feeds[side].reshape((-1, ) + input_shape)
            else:
                feeds[side] = faces

        preds = self._get_predictions(feeds["a"], feeds["b"])

        for side, samples in self.images.items():
            other_side = "a" if side == "b" else "b"
            predictions = [preds[f"{side}_{side}"],
                           preds[f"{other_side}_{side}"]]
            display = self._to_full_frame(side, samples, predictions)
            headers[side] = self._get_headers(side, display[0].shape[1])
            figures[side] = np.stack([display[0], display[1], display[2], ], axis=1)
            if self.images[side][0].shape[0] % 2 == 1:
                figures[side] = np.concatenate([figures[side],
                                                np.expand_dims(figures[side][0], 0)])

        width = 4
        side_cols = width // 2
        if side_cols != 1:
            headers = self._duplicate_headers(headers, side_cols)

        header = np.concatenate([headers["a"], headers["b"]], axis=1)
        figure = np.concatenate([figures["a"], figures["b"]], axis=0)
        height = int(figure.shape[0] / width)
        figure = figure.reshape((width, height) + figure.shape[1:])
        figure = _stack_images(figure)
        figure = np.concatenate((header, figure), axis=0)

        logger.debug("Compiled sample")
        return np.clip(figure * 255, 0, 255).astype('uint8')

    @classmethod
    def _resize_sample(cls, side, sample, target_size):
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
            return sample
        logger.debug("Resizing sample: (side: '%s', sample.shape: %s, target_size: %s, scale: %s)",
                     side, sample.shape, target_size, scale)
        interpn = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
        retval = np.array([cv2.resize(img, (target_size, target_size), interpn)
                           for img in sample])
        logger.debug("Resized sample: (side: '%s' shape: %s)", side, retval.shape)
        return retval

    def _get_predictions(self, feed_a, feed_b):
        """ Feed the samples to the model and return predictions

        Parameters
        ----------
        feed_a: list
            List of :class:`numpy.ndarray` of feed images for the "a" side
        feed_a: list
            List of :class:`numpy.ndarray` of feed images for the "b" side

        Returns
        -------
        list:
            List of :class:`numpy.ndarray` of predictions received from the model
        """
        logger.debug("Getting Predictions")
        preds = {}
        standard = self._model.model.predict([feed_a, feed_b])
        swapped = self._model.model.predict([feed_b, feed_a])

        if self._model.config["learn_mask"] and get_backend() == "amd":
            # Ravel results for plaidml
            split = len(standard) // 2
            standard = [standard[:split], standard[split:]]
            swapped = [swapped[:split], swapped[split:]]

        if self._model.config["learn_mask"]:  # Add mask to 4th channel of final output
            standard = [np.concatenate(side[-2:], axis=-1) for side in standard]
            swapped = [np.concatenate(side[-2:], axis=-1) for side in swapped]
        else:  # Retrieve final output
            standard = [side[-1] if isinstance(side, list) else side for side in standard]
            swapped = [side[-1] if isinstance(side, list) else side for side in swapped]

        preds["a_a"] = standard[0]
        preds["b_b"] = standard[1]
        preds["a_b"] = swapped[0]
        preds["b_a"] = swapped[1]

        logger.debug("Returning predictions: %s", {key: val.shape for key, val in preds.items()})
        return preds

    def _to_full_frame(self, side, samples, predictions):
        """ Patch targets and prediction images into images of model output size.

        Parameters
        ----------
        side: {"a" or "b"}
            The side that these samples are for
        samples: list
            List of :class:`numpy.ndarray` of feed images and target images
        predictions: list
            List of :class: `numpy.ndarray` of predictions from the model

        Returns
        -------
        list
            The images resized and collated for display in the preview frame
        """
        logger.debug("side: '%s', number of sample arrays: %s, prediction.shapes: %s)",
                     side, len(samples), [pred.shape for pred in predictions])
        full, faces = samples[:2]

        if self._model.color_order.lower() == "rgb":  # Switch color order for RGB model display
            full = full[..., ::-1]
            faces = faces[..., ::-1]
            predictions = [pred[..., ::-1] if pred.shape[-1] == 3 else pred
                           for pred in predictions]

        full = self._process_full(side, full, predictions[0].shape[1], (0, 0, 255))
        images = [faces] + predictions
        if self._display_mask:
            images = self._compile_masked(images, samples[-1])
        images = [self._overlay_foreground(full.copy(), image) for image in images]

        if self._scaling != 1.0:
            new_size = int(images[0].shape[1] * self._scaling)
            images = [self._resize_sample(side, image, new_size) for image in images]
        return images

    def _process_full(self, side, images, prediction_size, color):
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

    @classmethod
    def _compile_masked(cls, faces, masks):
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
        orig_masks = np.tile(1 - np.rint(masks), 3)
        orig_masks[np.where((orig_masks == [1., 1., 1.]).all(axis=3))] = [0., 0., 1.]

        if faces[-1].shape[-1] == 4:  # Mask contained in alpha channel of predictions
            pred_masks = [np.tile(1 - np.rint(face[..., -1])[..., None], 3) for face in faces[-2:]]
            for swap_masks in pred_masks:
                swap_masks[np.where((swap_masks == [1., 1., 1.]).all(axis=3))] = [0., 0., 1.]
            faces[-2:] = [face[..., :-1] for face in faces[-2:]]
            masks3 = [orig_masks, *pred_masks]
        else:
            masks3 = np.repeat(np.expand_dims(orig_masks, axis=0), 3, axis=0)

        retval = [np.array([cv2.addWeighted(img, 1.0, mask, 0.3, 0)
                            for img, mask in zip(previews, compiled_masks)])
                  for previews, compiled_masks in zip(faces, masks3)]
        logger.debug("masked shapes: %s", [faces.shape for faces in retval])
        return retval

    @classmethod
    def _overlay_foreground(cls, backgrounds, foregrounds):
        """ Overlay the preview images into the center of the background images

        Parameters
        ----------
        backgrounds: list
            List of :class:`numpy.ndarray` background images for placing the preview images onto
        backgrounds: list
            List of :class:`numpy.ndarray` preview images for placing onto the background images

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
    def _get_headers(cls, side, width):
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
        side = side.upper()
        height = int(width / 4.5)
        total_width = width * 3
        logger.debug("height: %s, total_width: %s", height, total_width)
        font = cv2.FONT_HERSHEY_SIMPLEX
        texts = [f"{titles[0]} ({side})",
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
    def _duplicate_headers(cls, headers, columns):
        """ Duplicate headers for the number of columns displayed for each side.

        Parameters
        ----------
        headers: :class:`numpy.ndarray`
            The header to be duplicated
        columns: int
            The number of columns that the header needs to be duplicated for

        Returns
        -------
        :class:`numpy.ndarray`
            The original headers duplicated by the number of columns
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
    scaling: float, optional
        The amount to scale the final preview image by. Default: `1.0`
    image_count: int
        The number of preview images to be displayed in the time-lapse
    feeder: dict
        The :class:`_Feeder` for generating the time-lapse images.
    image_paths: dict
        The full paths to the training images for each side of the model
    """
    def __init__(self, model, coverage_ratio, image_count, feeder, image_paths):
        logger.debug("Initializing %s: model: %s, coverage_ratio: %s, image_count: %s, "
                     "feeder: '%s', image_paths: %s)", self.__class__.__name__, model,
                     coverage_ratio, image_count, feeder, len(image_paths))
        self._num_images = image_count
        self._samples = _Samples(model, coverage_ratio)
        self._model = model
        self._feeder = feeder
        self._image_paths = image_paths
        self._output_file = None
        logger.debug("Initialized %s", self.__class__.__name__)

    def _setup(self, input_a=None, input_b=None, output=None):
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
        if output is None:
            output = get_folder(os.path.join(str(self._model.model_dir),
                                             f"{self._model.name}_timelapse"))
        self._output_file = str(output)
        logger.debug("Time-lapse output set to '%s'", self._output_file)

        # Rewrite paths to pull from the training images so mask and face data can be accessed
        images = {}
        for side, input_ in zip(("a", "b"), (input_a, input_b)):
            training_path = os.path.dirname(self._image_paths[side][0])
            images[side] = [os.path.join(training_path, os.path.basename(pth))
                            for pth in get_image_paths(input_)]

        batchsize = min(len(images["a"]),
                        len(images["b"]),
                        self._num_images)
        self._feeder.set_timelapse_feed(images, batchsize)
        logger.debug("Set up time-lapse")

    def output_timelapse(self, timelapse_kwargs):
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
            self._setup(**timelapse_kwargs)

        logger.debug("Getting time-lapse samples")
        self._samples.images = self._feeder.compile_timelapse_sample()
        logger.debug("Got time-lapse samples: %s",
                     {side: len(images) for side, images in self._samples.images.items()})

        image = self._samples.show_sample()
        if image is None:
            return
        filename = os.path.join(self._output_file, str(int(time.time())) + ".jpg")

        cv2.imwrite(filename, image)
        logger.debug("Created time-lapse: '%s'", filename)


def _stack_images(images):
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
