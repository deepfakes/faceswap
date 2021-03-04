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

from concurrent import futures
from functools import partial

import cv2
import numpy as np

import tensorflow as tf
from tensorflow.python.framework import errors_impl as tf_errors
from tqdm import tqdm

from lib.align import (Alignments, AlignedFace, DetectedFace, get_centered_size,
                       update_legacy_png_header)
from lib.image import read_image_meta_batch
from lib.training_data import TrainingDataGenerator
from lib.utils import FaceswapError, get_backend, get_folder, get_image_paths
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
        alignment_data = self._get_alignments_data()

        self._feeder = _Feeder(images,
                               self._model,
                               batch_size,
                               self._config,
                               alignment_data)

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

    def _get_alignments_data(self):
        """ Extrapolate alignments and masks from the alignments file into a `dict` for the
        training data generator.

        Removes any images from :attr:`_images` if they do not have alignment data attached.

        Returns
        -------
        dict:
            Includes the key `aligned_faces` holding aligned face information and the key
            `versions` indicating the alignments file versions that the faces have come from.
            In addition, the following optional keys are provided: `masks` if masks are required
            for training, `masks_eye` if eye masks are required and `masks_mouth` if mouth masks
            are required. """
        penalized_loss = self._model.config["penalized_mask_loss"]

        alignments = _TrainingAlignments(self._model, self._images)
        # Update centering if it has been changed by legacy face sets in TrainingAlignments
        self._config["centering"] = self._model.config["centering"]
        retval = dict(aligned_faces=alignments.aligned_faces,
                      versions=alignments.versions)

        if self._model.config["learn_mask"] or penalized_loss:
            logger.debug("Adding masks to training opts dict")
            retval["masks"] = alignments.masks

        if penalized_loss and self._model.config["eye_multiplier"] > 1:
            retval["masks_eye"] = alignments.masks_eye

        if penalized_loss and self._model.config["mouth_multiplier"] > 1:
            retval["masks_mouth"] = alignments.masks_mouth

        logger.debug({key: {k: v if isinstance(v, float) else len(v)
                            for k, v in val.items()}
                      for key, val in retval.items()})

        # Replace _images with list containing valid alignment data
        for side, aligned_faces in alignments.aligned_faces.items():
            if len(aligned_faces) != len(self._images[side]):
                logger.info("Updating training images list with images containing valid metadata "
                            "for side '%s'", side.upper())
                self._images[side] = list(aligned_faces.keys())
        return retval

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
                               "{}_logs".format(self._model.name),
                               "session_{}".format(self._model.state.session_id))
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
        do_timelapse = timelapse_kwargs is not None
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
                       "Training - 'S': Save Now. 'R': Refresh Preview. 'ENTER': Save and Quit")

        if do_timelapse:
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

    def _collate_and_store_loss(self, loss):
        """ Collate the loss into totals for each side.

        The losses are then into a total for each side. Loss totals are added to
        :attr:`model.state._history` to track the loss drop per save iteration for backup purposes.

        Parameters
        ----------
        loss: list
            The list of loss ``floats`` for this iteration.

        Returns
        -------
        list
            List of 2 ``floats`` which is the total loss for each side
        """
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
        output = ", ".join(["Loss {}: {:.5f}".format(side, side_loss)
                            for side, side_loss in zip(("A", "B"), loss)])
        timestamp = time.strftime("%H:%M:%S")
        output = "[{}] [#{:05d}] {}".format(timestamp, self._model.iterations, output)
        print("\r{}".format(output), end="")

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
    alignments: dict
        A dictionary containing aligned face data, extract version information and masks if these
        are required for training for each side
    """
    def __init__(self, images, model, batch_size, config, alignments):
        logger.debug("Initializing %s: num_images: %s, batch_size: %s, config: %s)",
                     self.__class__.__name__, len(images), batch_size, config)
        self._model = model
        self._images = images
        self._config = config
        self._alignments = alignments
        self._target = dict()
        self._samples = dict()
        self._masks = dict()

        self._feeds = {side: self._load_generator(idx).minibatch_ab(images[side], batch_size, side)
                       for idx, side in enumerate(("a", "b"))}

        self._display_feeds = dict(preview=self._set_preview_feed(), timelapse=dict())
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
                                          self._alignments,
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
        retval = dict()
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
            self._samples = dict()
            self._target = dict()
            self._masks = dict()
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
        retval = dict()
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
        samples = dict()
        images = dict()
        masks = dict()
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
        self.images = dict()
        self._coverage_ratio = coverage_ratio
        self._scaling = scaling
        logger.debug("Initialized %s", self.__class__.__name__)

    def show_sample(self):
        """ Compile a preview image.

        Returns
        -------
        :class:`numpy.ndarry`
            A compiled preview image ready for display or saving
        """
        logger.debug("Showing sample")
        feeds = dict()
        figures = dict()
        headers = dict()
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
            predictions = [preds["{0}_{0}".format(side)],
                           preds["{}_{}".format(other_side, side)]]
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
        preds = dict()
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
        texts = ["{} ({})".format(titles[0], side),
                 "{0} > {0}".format(titles[0]),
                 "{} > {}".format(titles[0], titles[1])]
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
            duped = tuple([header for _ in range(columns)])
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
            output = str(get_folder(os.path.join(str(self._model.model_dir),
                                                 "{}_timelapse".format(self._model.name))))
        self._output_file = str(output)
        logger.debug("Time-lapse output set to '%s'", self._output_file)

        # Rewrite paths to pull from the training images so mask and face data can be accessed
        images = dict()
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


class _TrainingAlignments():
    """ Obtain Landmarks and required mask from alignments file.

    Parameters
    ----------
    model: plugin from :mod:`plugins.train.model`
        The model that will be running this trainer
    image_list: dict
        The file paths for the images to be trained on for each side. The dictionary should contain
        2 keys ("a" and "b") with the values being a list of full paths corresponding to each side.
    """
    def __init__(self, model, image_list):
        logger.debug("Initializing %s: (model: %s, image counts: %s)",
                     self.__class__.__name__, model, {k: len(v) for k, v in image_list.items()})
        self._args = model.command_line_arguments
        self._config = model.config

        self._alignments_version = dict()
        self._image_sizes = {key: None for key in image_list}
        self._detected_faces = self._load_detected_faces(image_list)
        self._update_legacy_facesets(image_list)

        self._validity_check()
        self._aligned_faces = self._get_aligned_faces()
        logger.debug("Initialized %s", self.__class__.__name__)

    @property
    def versions(self):
        """ dict: The "a", "b" keys for each side, with value being the alignment file version
        that provided the data. This is used to crop the faces correctly based on whether the
        extracted faces are legacy or full-head extracts. """
        return self._alignments_version

    @property
    def aligned_faces(self):
        """ dict: The "a", "b" keys for each side, containing a sub-dictionary with the
        filename as key and :class:`lib.align.AlignedFace` object as value. """
        return self._aligned_faces

    # <<< LOAD DETECTED FACE INFORMATION FROM PNG HEADER >>>
    def _load_detected_faces(self, image_list):
        """ Obtain the metadata from the png training image headers for all images used for
        training.

        The Faceswap alignments data is returned as a dictionary, whist :attr:`_image_sizes` is
        populated from the training image metadata

        Parameters
        ----------
        image_list: dict
            The file paths for the images to be trained on for each side. The dictionary should
            contain 2 keys ("a" and "b") with the values being a list of full paths corresponding
            to each side.

        Returns
        -------
        dict
            For keys "a" and "b" the values are a ``dict`` with the key being the filename of the
            training image and the value being the :class:`lib.align.DetectedFace` object
        """
        metadata = dict()
        for side, filelist in image_list.items():
            meta_side = dict()
            logger.debug("side: %s, file count: %s", side, len(filelist))
            for filename, meta in tqdm(read_image_meta_batch(filelist),
                                       desc="Reading training images ({})".format(side.upper()),
                                       total=len(filelist),
                                       leave=False):

                self._validate_image_size(side, filename, meta["width"], meta["height"])

                if "itxt" not in meta or "alignments" not in meta["itxt"]:
                    meta_side[filename] = None
                else:
                    alignments_version = meta["itxt"]["source"]["alignments_version"]
                    self._alignments_version.setdefault(side, set()).add(alignments_version)
                    detected_face = DetectedFace()
                    detected_face.from_png_meta(meta["itxt"]["alignments"])
                    meta_side[filename] = detected_face
            metadata[side] = meta_side
        return metadata

    def _validate_image_size(self, side, filename, width, height):
        """ Validate that the images are square and that the sizes for all image in a side are
        the same.

        Parameters
        ----------
        side: ["a" or "b"]
            The training side that is being processed
        filename: str
            The filename of the image that is being validated
        width: int
            The width of the image to be validated
        height: int
            The height of the image to be validated

        Raises
        ------
        FaceswapError
            If the image to be checked is not square or is of a different size of any other image
            for the current side, an error is raised.
        """
        # Add the image size to the sizes dictionary if this is the first image
        if not self._image_sizes[side]:
            self._image_sizes[side] = width

        # Validate image is square
        if width != height:
            msg = ("Training images must be created by the extraction process and must be "
                   "square.\nThe image '{}' has dimensions {}x{} so the process cannot "
                   "continue.\nThere may be more images with these issues. Please double "
                   "check your dataset".format(filename, width, height))
            raise FaceswapError(msg)

        # Validate image is the same size as the other images for the side
        if width != self._image_sizes[side]:
            msg = ("All training images for each side must be of the same size.\nImages "
                   "in side '{}' have mismatched sizes {} and {}.\nPlease double check "
                   "your dataset".format(side.upper(), self._image_sizes[side], width))
            raise FaceswapError(msg)

    def _update_legacy_facesets(self, image_list):
        """ Update the png header data for legacy face sets that do not contain the meta data in
        the exif header.

        Parameters
        ----------
        image_list: dict
            The file paths for the images to be trained on for each side. The dictionary should
            contain 2 keys ("a" and "b") with the values being a list of full paths corresponding
            to each side.
        """
        if self._validate_metadata(output_warning=False):
            logger.debug("All faces contain valid header information")
            return

        for side, png_meta in self._detected_faces.items():
            if all(png_meta.values()):
                continue
            filenames = [filename for filename, meta in png_meta.items() if not meta]
            logger.info("Legacy faces discovered for side '%s'. Updating %s images...",
                        side.upper(), len(filenames))
            alignments = Alignments(*os.path.split(self._get_alignments_path(side)))
            self._alignments_version.setdefault(side, set()).add(alignments.version)

            executor = futures.ThreadPoolExecutor()
            with executor:
                images = {executor.submit(update_legacy_png_header, filename, alignments): filename
                          for filename in filenames}

                for future in tqdm(
                        futures.as_completed(images),
                        desc="Updating legacy training images ({})".format(side.upper()),
                        total=len(filenames),
                        leave=False):
                    result = future.result()
                    if result:
                        filename = images[future]
                        if os.path.splitext(filename)[-1].lower() != ".png":
                            # Update the image list to point at newly created png
                            del png_meta[filename]
                            image_list[side].remove(filename)

                            filename = os.path.splitext(filename)[0] + ".png"
                            image_list[side].append(filename)

                        detected_face = DetectedFace()
                        detected_face.from_png_meta(future.result()["alignments"])
                        png_meta[filename] = detected_face

    def _get_alignments_path(self, side):
        """ Obtain the path to an alignments file for the given training side.

        Used for updating legacy face sets to contain the meta information within the image header

        Parameters
        ----------
        side: ["a" or "b"]
            The training side to obtain the alignments file for.

        Returns
        -------
        str
            The full path to the training alignments file

        Raises
        ------
        FaceswapError
            If an alignments file cannot be located
        """
        alignments_path = getattr(self._args, "alignments_path_{}".format(side))
        if not alignments_path:
            image_path = getattr(self._args, "input_{}".format(side))
            alignments_path = os.path.join(image_path, "alignments.fsa")
        if not os.path.exists(alignments_path):
            msg = ("You are using a legacy faceset that does not contain embedded "
                   "meta-information. An alignments file must be provided so that these files can "
                   "be updated.\n"
                   f"Alignments file does not exist: '{alignments_path}'")
            raise FaceswapError(msg)
        return alignments_path

    # <<< VALIDATE LOADED DETECTED FACE INFORMATION >>>
    def _validity_check(self):
        """ Check the validity of the finally loaded data.

        Ensure that each side contains alignments data that was extracted with the same centering.
        Ensure that each side has a full compliment of metadata.
        """
        invalid = [side.upper()
                   for side, vers in self._alignments_version.items()
                   if len(vers) > 1 and any(v < 2 for v in vers) and any(v > 1 for v in vers)]

        if invalid:
            raise FaceswapError("Mixing legacy and full head extracted facesets is not supported. "
                                "The following side(s) contain a mix of extracted face "
                                "types: {}".format(invalid))
        # Replace check alignments version sets with actual floats
        self._alignments_version = {key: val.pop()
                                    for key, val in self._alignments_version.items()}

        if 1.0 in self._alignments_version.values() and self._config["centering"] != "legacy":
            logger.warning("You are using legacy extracted faces but have selected '%s' "
                           "centering which is incompatible. Switching centering to 'legacy'",
                           self._config["centering"])
            self._config["centering"] = "legacy"

        self._validate_metadata(output_warning=True)
        self._validate_masks()

    def _validate_metadata(self, output_warning=True):
        """ Validate that all images to be trained on have associated alignments data. If not
        generate a warning.

        Parameters
        ----------
        output_warning: bool, optional
            If ``True`` outputs a warning that images are missing alignments data.

        Returns
        -------
        bool
            ``True`` if all images have valid metadata otherwise ``False``
        """
        all_valid = {side: all(val.values()) for side, val in self._detected_faces.items()}
        if all(all_valid.values()):
            return True
        if not output_warning:
            return False

        for side, valid in all_valid.items():
            if valid:
                continue
            if all(val is None for val in self._detected_faces[side].values()):
                raise FaceswapError("There is no valid training data for side '{}'. Re-check your "
                                    "data and try again.".format(side.upper()))
            invalid = [filename
                       for filename, meta in self._detected_faces[side].items() if not meta]

            logger.warning("Data for training side '%s' contains %s faces that do not contain "
                           "valid metadata and will be excluded from training.",
                           side.upper(), len(invalid))
            logger.warning("Run in VERBOSE mode if you wish to see a list of these files.")
            logger.verbose("Side '%s' images missing metadata: %s", side.upper(),
                           sorted(os.path.basename(fname) for fname in invalid))
            # Remove images without metadata
            self._detected_faces[side] = {key: val
                                          for key, val in self._detected_faces[side].items()
                                          if val}
        return False

    def _validate_masks(self):
        """ Validate the the loaded metadata all contain the masks required for training.

        Raises
        ------
        FaceswapError
            If at least one face in the training data does not contain the selected mask type
        """
        mask_type = self._config["mask_type"]
        if mask_type is None:
            logger.debug("No mask selected. Not validating")
            return
        invalid = {side: [filename for filename, detected_face in faces.items()
                          if mask_type not in detected_face.mask]
                   for side, faces in self._detected_faces.items()}
        if any(invalid.values()):
            msg = ("You have selected the Mask Type '{}' in your training configuration options "
                   "but at least one face does not have this mask type stored for it.\nYou should "
                   "select a mask type that exists within your face data, or generate the "
                   "required masks with the Mask Tool.".format(mask_type))
            for side, filenames in invalid.items():
                if not filenames:
                    continue
                available = set(mask
                                for det_face in self._detected_faces[side].values()
                                for mask in det_face.mask)
                msg += ("\n{} faces in side {} do not contain the mask '{}'. Available "
                        "masks: {}".format(len(filenames), side.upper(), mask_type, available))
            raise FaceswapError(msg)

    # <<< LOAD REQUIRED DATA FOR TRAINING >>>
    def _get_aligned_faces(self):
        """ Pre-generate aligned faces as they are needed for all training functions.

        Returns
        -------
        dict
            The "a", "b" keys for each side, containing a sub-dictionary with the
            filename as key and :class:`lib.align.AlignedFace` object as value.
        """
        logger.debug("Loading aligned faces: %s",
                     {k: len(v) for k, v in self._detected_faces.items()})
        retval = dict()
        for side, detected_faces in self._detected_faces.items():
            retval[side] = dict()
            size = get_centered_size("legacy" if self._alignments_version[side] == 1.0 else "head",
                                     self._config["centering"],
                                     self._image_sizes[side])
            for filename, face in detected_faces.items():
                retval[side][filename] = AlignedFace(face.landmarks_xy,
                                                     centering=self._config["centering"],
                                                     size=size,
                                                     is_aligned=True)
        logger.debug("Loaded aligned faces: %s", {k: len(v) for k, v in retval.items()})
        return retval

    # Get masks
    @property
    def masks(self):
        """ dict: The :class:`lib.align.Mask` objects of requested mask type for
        keys a" and "b"
        """
        retval = dict()
        for side, faces in self._detected_faces.items():
            retval[side] = dict()
            for filename, detected_face in faces.items():
                mask = detected_face.mask[self._config["mask_type"]]
                mask.set_blur_and_threshold(blur_kernel=self._config["mask_blur_kernel"],
                                            threshold=self._config["mask_threshold"])
                if self._alignments_version[side] > 1.0 and self._config["centering"] == "legacy":
                    mask.set_sub_crop(self._aligned_faces[side][filename].pose.offset["face"] * -1)
                retval[side][filename] = mask
        logger.trace(retval)
        return retval

    @property
    def masks_eye(self):
        """ dict: filename mapping to zip compressed eye masks for keys "a" and "b" """
        retval = {side: self._get_landmarks_masks(side, detected_faces, "eyes")
                  for side, detected_faces in self._detected_faces.items()}
        return retval

    @property
    def masks_mouth(self):
        """ dict: filename mapping to zip compressed mouth masks for keys "a" and "b" """
        retval = {side: self._get_landmarks_masks(side, detected_faces, "mouth")
                  for side, detected_faces in self._detected_faces.items()}
        return retval

    def _get_landmarks_masks(self, side, detected_faces, area):
        """ Obtain the area landmarks masks for the given area.

        A :func:`functools.partial` is returned rather than the full compressed mask to speed up
        pre-loading. The partials are expanded the first time they are accessed within the training
        loop.

        Parameters
        ----------
        side: {"a" or "b"}
            The side currently being processed
        detected_faces: dict
            Key is the filename of the face, value is the corresponding
            :class:`lib.align.DetectedFace` object
        area: {"eyes" or "mouth"}
            The area of the face to obtain the mask for

        Returns
        -------
        dict
            The face filenames as keys with the :func:`functools.partial` of the mask as value
        """
        logger.trace("side: %s, detected_faces: %s, area: %s", side, detected_faces, area)
        masks = dict()
        size = list(self._aligned_faces[side].values())[0].size
        for filename, face in detected_faces.items():
            masks[filename] = partial(face.get_landmark_mask,
                                      size,
                                      area,
                                      aligned=True,
                                      centering=self._config["centering"],
                                      dilation=size // 32,
                                      blur_kernel=size // 16,
                                      as_zip=True)
        logger.trace("side: %s, area: %s, masks: %s",
                     side, area, {key: type(val) for key, val in masks.items()})
        return masks


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
