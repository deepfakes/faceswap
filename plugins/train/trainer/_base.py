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
from tensorflow.python import errors_impl as tf_errors  # pylint:disable=no-name-in-module
from tqdm import tqdm

from lib.alignments import Alignments
from lib.faces_detect import DetectedFace
from lib.image import read_image_hash_batch
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
        self._config = _get_config(".".join(self.__module__.split(".")[-2:]),
                                   configfile=configfile)
        self._model = model
        self._model.state.add_session_batchsize(batch_size)
        self._images = images
        self._sides = sorted(key for key in self._images.keys())

        self._feeder = _Feeder(images,
                               self._model,
                               batch_size,
                               self._config,
                               self._get_alignments_data())

        self._tensorboard = self._set_tensorboard()
        self._samples = _Samples(self._model,
                                 self._model.coverage_ratio,
                                 self._model.command_line_arguments.preview_scale / 100)
        self._timelapse = _Timelapse(self._model,
                                     self._model.coverage_ratio,
                                     self._config.get("preview_images", 14),
                                     self._feeder)
        logger.debug("Initialized %s", self.__class__.__name__)

    def _get_alignments_data(self):
        """ Extrapolate alignments and masks from the alignments file into a `dict` for the
        training data generator.

        Returns
        -------
        dict:
            Includes the key `landmarks` if landmarks are required for training, `masks` if masks
            are required for training, `masks_eye` if eye masks are required and `masks_mouth` if
            mouth masks are required. """
        retval = dict()

        if not any([self._model.config["learn_mask"],
                    self._model.config["penalized_mask_loss"],
                    self._model.config["eye_multiplier"] > 1,
                    self._model.config["mouth_multiplier"] > 1,
                    self._model.command_line_arguments.warp_to_landmarks]):
            return retval

        alignments = _TrainingAlignments(self._model, self._images)

        if self._model.command_line_arguments.warp_to_landmarks:
            logger.debug("Adding landmarks to training opts dict")
            retval["landmarks"] = alignments.landmarks

        if self._model.config["learn_mask"] or self._model.config["penalized_mask_loss"]:
            logger.debug("Adding masks to training opts dict")
            retval["masks"] = alignments.masks

        if self._model.config["eye_multiplier"] > 1:
            retval["masks_eye"] = alignments.masks_eye

        if self._model.config["mouth_multiplier"] > 1:
            retval["masks_mouth"] = alignments.masks_mouth

        logger.debug({key: {k: len(v) for k, v in val.items()} for key, val in retval.items()})
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
        A dictionary containing landmarks and masks if these are required for training for each
        side
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
                                          not self._model.command_line_arguments.no_augment_color,
                                          self._model.command_line_arguments.no_flip,
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
        """ Patch targets and prediction images into images of training image size.

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
        images = [faces] + predictions
        full_size = full.shape[1]
        target_size = int(full_size * self._coverage_ratio)
        if target_size != full_size:
            frame = self._frame_overlay(full, target_size, (0, 0, 255))

        if self._display_mask:
            images = self._compile_masked(images, samples[-1])
        images = [self._resize_sample(side, image, target_size) for image in images]
        if target_size != full_size:
            images = [self._overlay_foreground(frame, image) for image in images]
        if self._scaling != 1.0:
            new_size = int(full_size * self._scaling)
            images = [self._resize_sample(side, image, new_size) for image in images]
        return images

    @classmethod
    def _frame_overlay(cls, images, target_size, color):
        """ Add a frame overlay to preview images indicating the region of interest.

        This is the red border that appears in the preview images.

        Parameters
        ----------
        images: :class:`numpy.ndarray`
            The samples to apply the frame to
        target_size: int
            The size of the sample within the full size frame
        color: tuple
            The (Blue, Green, Red) color to use for the frame

        Returns
        -------
        :class:`numpy,ndarray`
            The samples with the frame overlay applied
        """
        logger.debug("full_size: %s, target_size: %s, color: %s",
                     images.shape[1], target_size, color)
        new_images = list()
        full_size = images.shape[1]
        padding = (full_size - target_size) // 2
        length = target_size // 4
        t_l, b_r = (padding, full_size - padding)
        for img in images:
            cv2.rectangle(img, (t_l, t_l), (t_l + length, t_l + length), color, 3)
            cv2.rectangle(img, (b_r, t_l), (b_r - length, t_l + length), color, 3)
            cv2.rectangle(img, (b_r, b_r), (b_r - length, b_r - length), color, 3)
            cv2.rectangle(img, (t_l, b_r), (t_l + length, b_r - length), color, 3)
            new_images.append(img)
        retval = np.array(new_images)
        logger.debug("Overlayed background. Shape: %s", retval.shape)
        return retval

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
        new_images = list()
        for idx, img in enumerate(backgrounds):
            img[offset:offset + foregrounds[idx].shape[0],
                offset:offset + foregrounds[idx].shape[1], :3] = foregrounds[idx]
            new_images.append(img)
        retval = np.array(new_images)
        logger.debug("Overlayed foreground. Shape: %s", retval.shape)
        return retval

    def _get_headers(self, side, width):
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
        height = int(64 * self._scaling)
        total_width = width * 3
        logger.debug("height: %s, total_width: %s", height, total_width)
        font = cv2.FONT_HERSHEY_SIMPLEX
        texts = ["{} ({})".format(titles[0], side),
                 "{0} > {0}".format(titles[0]),
                 "{} > {}".format(titles[0], titles[1])]
        text_sizes = [cv2.getTextSize(texts[idx], font, self._scaling * 0.8, 1)[0]
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
                        self._scaling * 0.8,
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
    """
    def __init__(self, model, coverage_ratio, image_count, feeder):
        logger.debug("Initializing %s: model: %s, coverage_ratio: %s, image_count: %s, "
                     "feeder: '%s')", self.__class__.__name__, model, coverage_ratio,
                     image_count, feeder)
        self._num_images = image_count
        self._samples = _Samples(model, coverage_ratio)
        self._model = model
        self._feeder = feeder
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

        images = {"a": get_image_paths(input_a), "b": get_image_paths(input_b)}
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
        self._training_size = model.state.training_size
        self._alignments_paths = self._get_alignments_paths()
        self._hashes = self._get_image_hashes(image_list)
        self._detected_faces = self._load_alignments()
        self._check_all_faces()
        logger.debug("Initialized %s", self.__class__.__name__)

    # Get landmarks
    @property
    def landmarks(self):
        """ dict: The :class:`numpy.ndarray` aligned landmarks for keys "a" and "b" """
        retval = {side: self._transform_landmarks(side, detected_faces)
                  for side, detected_faces in self._detected_faces.items()}
        logger.trace(retval)
        return retval

    def _get_alignments_paths(self):
        """ Obtain the alignments file paths from the command line arguments passed to the model.

        If the argument does not exist or is empty, then scan the input folder for an alignments
        file.

        Returns
        -------
        dict
            The alignments paths for each of the source and destination faces. Key is the
            side, value is the path to the alignments file

        Raises
        ------
        FaceswapError
            If at least one alignments file does not exist
        """
        retval = dict()
        for side in ("a", "b"):
            alignments_path = getattr(self._args, "alignments_path_{}".format(side))
            if not alignments_path:
                image_path = getattr(self._args, "input_{}".format(side))
                alignments_path = os.path.join(image_path, "alignments.fsa")
            if not os.path.exists(alignments_path):
                raise FaceswapError("Alignments file does not exist: `{}`".format(alignments_path))
            retval[side] = alignments_path
        logger.debug("Alignments paths: %s", retval)
        return retval

    def _transform_landmarks(self, side, detected_faces):
        """ Transform frame landmarks to their aligned face variant.

        Parameters
        ----------
        side: {"a" or "b"}
            The side currently being processed
        detected_faces: list
            A list of :class:`lib.faces_detect.DetectedFace` objects

        Returns
        -------
        dict
            The face filenames as keys with the aligned landmarks as value.
        """
        landmarks = dict()
        for face in detected_faces.values():
            face.load_aligned(None, size=self._training_size)
            for filename in self._hash_to_filenames(side, face.hash):
                landmarks[filename] = face.aligned_landmarks
        return landmarks

    # Get masks
    @property
    def masks(self):
        """ dict: The :class:`lib.faces_detect.Mask` objects of requested mask type for
        keys a" and "b"
        """
        retval = {side: self._get_masks(side, detected_faces)
                  for side, detected_faces in self._detected_faces.items()}
        logger.trace(retval)
        return retval

    def _get_masks(self, side, detected_faces):
        """ For each face, obtain the mask and set the requested blurring and threshold level.

        Parameters
        ----------
        side: {"a" or "b"}
            The side currently being processed
        detected_faces: dict
            Key is the hash of the face, value is the corresponding
            :class:`lib.faces_detect.DetectedFace` object

        Returns
        -------
        dict
            The face filenames as keys with the :class:`lib.faces_detect.Mask` as value.
        """
        masks = dict()
        for fhash, face in detected_faces.items():
            mask = face.mask[self._config["mask_type"]]
            mask.set_blur_and_threshold(blur_kernel=self._config["mask_blur_kernel"],
                                        threshold=self._config["mask_threshold"])
            for filename in self._hash_to_filenames(side, fhash):
                masks[filename] = mask
        return masks

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

        Parameters
        ----------
        side: {"a" or "b"}
            The side currently being processed
        detected_faces: dict
            Key is the hash of the face, value is the corresponding
            :class:`lib.faces_detect.DetectedFace` object
        area: {"eyes" or "mouth"}
            The area of the face to obtain the mask for

        Returns
        -------
        dict
            The face filenames as keys with the zip compressed mask as value.
        """
        logger.trace("side: %s, detected_faces: %s, area: %s", side, detected_faces, area)
        masks = dict()
        for fhash, face in detected_faces.items():
            mask = face.get_landmark_mask(self._training_size,
                                          area,
                                          aligned=True,
                                          dilation=self._training_size // 32,
                                          blur_kernel=self._training_size // 16,
                                          as_zip=True)
            for filename in self._hash_to_filenames(side, fhash):
                masks[filename] = mask
        logger.trace("side: %s, area: %s, masks: %s",
                     side, area, {key: type(val) for key, val in masks.items()})
        return masks

    # Hashes for image folders
    @classmethod
    def _get_image_hashes(cls, image_list):
        """ Return the hashes for all images used for training.

        Parameters
        ----------
        image_list: dict
            The file paths for the images to be trained on for each side. The dictionary should
            contain 2 keys ("a" and "b") with the values being a list of full paths corresponding
            to each side.

        Returns
        -------
        dict
            For keys "a" and "b" the values are a ``dict`` with the key being the sha1 hash and
            the value being a list of filenames that correspond to the hash for images that exist
            within the training data folder
        """
        hashes = {key: dict() for key in image_list}
        for side, filelist in image_list.items():
            logger.debug("side: %s, file count: %s", side, len(filelist))
            for filename, hsh in tqdm(read_image_hash_batch(filelist),
                                      desc="Reading training images ({})".format(side.upper()),
                                      total=len(filelist),
                                      leave=False):
                hashes[side].setdefault(hsh, list()).append(filename)
        logger.trace(hashes)
        return hashes

    # Hashes for Detected Faces
    def _load_alignments(self):
        """ Load the alignments and convert to :class:`lib.faces_detect.DetectedFace` objects.

        Returns
        -------
        dict
            For keys "a" and "b" values are a dict with the key being the sha1 hash of the face
            and the value being the corresponding :class:`lib.faces_detect.DetectedFace` object.
        """
        logger.debug("Loading alignments")
        retval = dict()
        for side, fullpath in self._alignments_paths.items():
            logger.debug("side: '%s', path: '%s'", side, fullpath)
            path, filename = os.path.split(fullpath)
            alignments = Alignments(path, filename=filename)
            retval[side] = self._to_detected_faces(alignments, side)
        logger.debug("Returning: %s", {k: len(v) for k, v in retval.items()})
        return retval

    def _to_detected_faces(self, alignments, side):
        """ Convert alignments to DetectedFace objects.

        Filter the detected faces to only those that exist in the training folders.

        Parameters
        ----------
        alignments: :class:`lib.alignments.Alignments`
            The alignments for the current faces
        side: {"a" or "b"}
            The side being processed

        Returns
        -------
        dict
            key is sha1 hash of face, value is the corresponding
            :class:`lib.faces_detect.DetectedFace` object
        """
        skip_count = 0
        dupe_count = 0
        side_hashes = set(self._hashes[side])
        detected_faces = dict()
        for _, faces, _, filename in alignments.yield_faces():
            for idx, face in enumerate(faces):
                if face["hash"] in detected_faces:
                    dupe_count += 1
                    logger.debug("Face already exists, skipping: '%s'", filename)
                if not self._validate_face(face, filename, idx, side, side_hashes):
                    skip_count += 1
                    continue
                detected_face = DetectedFace()
                detected_face.from_alignment(face)
                detected_faces[face["hash"]] = detected_face
        logger.debug("Detected Faces count: %s, Skipped faces count: %s, duplicate faces "
                     "count: %s", len(detected_faces), skip_count, dupe_count)
        if skip_count != 0:
            logger.warning("%s alignments have been removed as their corresponding faces do not "
                           "exist in the input folder for side %s. Run in verbose mode if you "
                           "wish to see which alignments have been excluded.",
                           skip_count, side.upper())
        return detected_faces

    # Validation
    def _validate_face(self, face, filename, idx, side, side_hashes):
        """ Validate that the currently processing face has a corresponding hash entry and the
        requested mask exists

        Parameters
        ----------
        face: dict
            A face retrieved from an alignments file
        filename: str
            The original frame filename that the given face comes from
        idx: int
            The index of the face in the frame
        side: {'A', 'B'}
            The side that this face belongs to
        side_hashes: set
            A set of hashes that exist in the alignments folder for these faces

        Returns
        -------
        bool
            ``True`` if the face is valid otherwise ``False``

        Raises
        ------
        FaceswapError
            If the current face doesn't pass validation
        """
        mask_type = self._config["mask_type"]
        if mask_type is not None and "mask" not in face:
            msg = ("You have selected a Mask Type in your training configuration options but at "
                   "least one face has no mask stored for it.\nYou should generate the required "
                   "masks with the Mask Tool or set the Mask Type configuration option to `none`."
                   "\nThe face that caused this failure was side: `{}`, frame: `{}`, index: {}. "
                   "However there are probably more faces without masks".format(
                       side.upper(), filename, idx))
            raise FaceswapError(msg)

        if mask_type is not None and mask_type not in face["mask"]:
            msg = ("At least one of your faces does not have the mask `{}` stored for it.\nYou "
                   "should run the Mask Tool to generate this mask for your faceset or "
                   "select a different mask in the training configuration options.\n"
                   "The face that caused this failure was [side: `{}`, frame: `{}`, index: {}]. "
                   "The masks that exist for this face are: {}.\nBe aware that there are probably "
                   "more faces without this Mask Type".format(
                       mask_type, side.upper(), filename, idx, list(face["mask"].keys())))
            raise FaceswapError(msg)

        if face["hash"] not in side_hashes:
            logger.verbose("Skipping alignment for non-existant face in frame '%s' index: %s",
                           filename, idx)
            return False
        return True

    def _check_all_faces(self):
        """ Ensure that all faces in the training folder exist in the alignments file.
        If not, output missing filenames

        Raises
        ------
        FaceswapError
            If there are faces in the training folder which do not exist in the alignments file
        """
        logger.debug("Checking faces exist in alignments")
        missing_alignments = dict()
        for side, train_hashes in self._hashes.items():
            align_hashes = set(self._detected_faces[side])
            if not align_hashes.issuperset(set(train_hashes)):
                missing_alignments[side] = [
                    os.path.basename(filename)
                    for hsh, filenames in train_hashes.items()
                    for filename in filenames
                    if hsh not in align_hashes]
        if missing_alignments:
            msg = ("There are faces in your training folder(s) which do not exist in your "
                   "alignments file. Training cannot continue. See above for a full list of "
                   "files missing alignments.")
            for side, filelist in missing_alignments.items():
                logger.error("Faces missing alignments for side %s: %s",
                             side.capitalize(), filelist)
            raise FaceswapError(msg)

    # Utils
    def _hash_to_filenames(self, side, face_hash):
        """ For a given hash return all the filenames that match for the given side.

        Notes
        -----
        Multiple faces can have the same hash, so this makes sure that all filenames are updated
        for all instances of a hash.

        Parameters
        ----------
        side: {"a" or "b"}
            The side currently being processed
        face_hash: str
            The sha1 hash of the face to obtain the filename for

        Returns
        -------
        list
            The filenames that exist for the given hash
        """
        retval = self._hashes[side][face_hash]
        logger.trace("side: %s, hash: %s, filenames: %s", side, face_hash, retval)
        return retval


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
