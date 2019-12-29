#!/usr/bin/env python3
""" Base Class for Faceswap Trainer plugins. All Trainer plugins should be inherited from
this class.

At present there is only the :class:`~plugins.train.trainer.original` plugin, so that entirely
inherits from this class.

This class heavily references the :attr:`plugins.train.model._base.ModelBase.training_opts`
``dict``. The following keys are expected from this ``dict``:

    * **alignments** (`dict`, `optional`) - If training with a mask or the warp to landmarks \
    command line option is selected then this is required, otherwise it can be ``None``. The \
    dictionary should contain 2 keys ("a" and "b") with the values being the path to the \
    alignments file for the corresponding side.

    * **preview_scaling** (`int`) - How much to scale displayed preview image by.

    * **training_size** ('int') - Size of the training images in pixels.

    * **coverage_ratio** ('float') - Ratio of face to be cropped out of the training image.

    * **mask_type** ('str') - The type of mask to select from the alignments file.

    * **mask_blur_kernel** ('int') - The size of the kernel to use for gaussian blurring the mask.

    * **mask_threshold** ('int') - The threshold for min/maxing mask to 0/100.

    * **learn_mask** ('bool') - Whether the mask should be trained in the model.

    * **penalized_mask_loss** ('bool') - Whether the mask should be penalized from loss.

    * **no_logs** ('bool') - Whether Tensorboard logging should be disabled.

    * **snapshot_interval** ('int') - How many iterations between model snapshot saves.

    * **warp_to_landmarks** ('bool') - Whether to use random_warp_landmarks instead of random_warp.

    * **augment_color** ('bool') - Whether to use color augmentation.

    * **no_flip** ('bool') - Whether to turn off random horizontal flipping.

    * **pingpong** ('bool') - Train each side separately per save iteration rather than together.
"""

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
from lib.utils import FaceswapError, get_folder, get_image_paths
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
    """ Trainer plugin base Object.

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

        self._process_training_opts()
        self._pingpong = PingPong(model, self._sides)

        self._batchers = {side: Batcher(side,
                                        images[side],
                                        self._model,
                                        self._use_mask,
                                        batch_size,
                                        self._config)
                          for side in self._sides}

        self._tensorboard = self._set_tensorboard()
        self._samples = Samples(self._model,
                                self._use_mask,
                                self._model.training_opts["coverage_ratio"],
                                self._model.training_opts["preview_scaling"])
        self._timelapse = Timelapse(self._model,
                                    self._use_mask,
                                    self._model.training_opts["coverage_ratio"],
                                    self._config.get("preview_images", 14),
                                    self._batchers)
        logger.debug("Initialized %s", self.__class__.__name__)

    @property
    def pingpong(self):
        """ :class:`pingpong`: Ping-pong object for ping-pong memory saving training. """
        return self._pingpong

    @property
    def _timestamp(self):
        """ str: Current time formatted as HOURS:MINUTES:SECONDS """
        return time.strftime("%H:%M:%S")

    @property
    def _landmarks_required(self):
        """ bool: ``True`` if Landmarks are required otherwise ``False ``"""
        retval = self._model.training_opts["warp_to_landmarks"]
        logger.debug(retval)
        return retval

    @property
    def _use_mask(self):
        """ bool: ``True`` if a mask is required otherwise ``False`` """
        retval = (self._model.training_opts["learn_mask"] or
                  self._model.training_opts["penalized_mask_loss"])
        logger.debug(retval)
        return retval

    def _process_training_opts(self):
        """ Extrapolate alignments and masks from the alignments file into
        :attr:`_model.training_opts`."""
        logger.debug(self._model.training_opts)
        if not self._landmarks_required and not self._use_mask:
            return

        alignments = TrainingAlignments(self._model.training_opts, self._images)
        if self._landmarks_required:
            logger.debug("Adding landmarks to training opts dict")
            self._model.training_opts["landmarks"] = alignments.landmarks

        if self._use_mask:
            logger.debug("Adding masks to training opts dict")
            self._model.training_opts["masks"] = alignments.masks

    def _set_tensorboard(self):
        """ Set up Tensorboard callback for logging loss.

        Bypassed if command line option "no-logs" has been selected.

        Returns
        -------
        dict:
            2 Dictionary keys of "a" and "b" the values of which are the
        :class:`tf.keras.callbacks.TensorBoard` objects for the respective sides.
        """
        if self._model.training_opts["no_logs"]:
            logger.verbose("TensorBoard logging disabled")
            return None
        if self._pingpong.active:
            # Currently TensorBoard uses the tf.session, meaning that VRAM does not
            # get cleared when model switching
            # TODO find a fix for this
            logger.warning("Currently TensorBoard logging is not supported for Ping-Pong "
                           "training. Session stats and graphing will not be available for this "
                           "training session.")
            return None

        logger.debug("Enabling TensorBoard Logging")
        tensorboard = dict()

        for side in self._sides:
            logger.debug("Setting up TensorBoard Logging. Side: %s", side)
            log_dir = os.path.join(str(self._model.model_dir),
                                   "{}_logs".format(self._model.name),
                                   side,
                                   "session_{}".format(self._model.state.session_id))
            tbs = tf.keras.callbacks.TensorBoard(log_dir=log_dir, **self._tensorboard_kwargs)
            tbs.set_model(self._model.predictors[side])
            tensorboard[side] = tbs
        logger.info("Enabled TensorBoard Logging")
        return tensorboard

    @property
    def _tensorboard_kwargs(self):
        """ dict: The keyword arguments to be passed to :class:`tf.keras.callbacks.TensorBoard`.
        NB: Tensorflow 1.13 + needs an additional keyword argument which is not valid for earlier
        versions """
        kwargs = dict(histogram_freq=0,  # Must be 0 or hangs
                      batch_size=64,
                      write_graph=True,
                      write_grads=True)
        tf_version = [int(ver) for ver in tf.__version__.split(".") if ver.isdigit()]
        logger.debug("Tensorflow version: %s", tf_version)
        if tf_version[0] > 1 or (tf_version[0] == 1 and tf_version[1] > 12):
            kwargs["update_freq"] = "batch"
        if tf_version[0] > 1 or (tf_version[0] == 1 and tf_version[1] > 13):
            kwargs["profile_batch"] = 0
        logger.debug(kwargs)
        return kwargs

    def __print_loss(self, loss):
        """ Outputs the loss for the current iteration to the console.

        Parameters
        ----------
        loss: dict
            The loss for each side. The dictionary should contain 2 keys ("a" and "b") with the
            values being a list of loss values for the current iteration corresponding to
            each side.
         """
        logger.trace(loss)
        output = ["Loss {}: {:.5f}".format(side.capitalize(), loss[side][0])
                  for side in sorted(loss.keys())]
        output = ", ".join(output)
        output = "[{}] [#{:05d}] {}".format(self._timestamp, self._model.iterations, output)
        print("\r{}".format(output), end="")

    def train_one_step(self, viewer, timelapse_kwargs):
        """ Running training on a batch of images for each side.

        Triggered from the training cycle in :class:`scripts.train.Train`.

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
        logger.trace("Training one step: (iteration: %s)", self._model.iterations)
        do_preview = viewer is not None
        do_timelapse = timelapse_kwargs is not None
        snapshot_interval = self._model.training_opts.get("snapshot_interval", 0)
        do_snapshot = (snapshot_interval != 0 and
                       self._model.iterations >= snapshot_interval and
                       self._model.iterations % snapshot_interval == 0)

        loss = dict()
        try:
            for side, batcher in self._batchers.items():
                if self._pingpong.active and side != self._pingpong.side:
                    continue
                loss[side] = batcher.train_one_batch()
                if not do_preview and not do_timelapse:
                    continue
                if do_preview:
                    batcher.generate_preview(do_preview)
                    self._samples.images[side] = batcher.compile_sample(None)
                if do_timelapse:
                    self._timelapse.get_sample(side, timelapse_kwargs)

            self._model.state.increment_iterations()

            for side, side_loss in loss.items():
                self._store_history(side, side_loss)
                self._log_tensorboard(side, side_loss)

            if not self._pingpong.active:
                self.__print_loss(loss)
            else:
                for key, val in loss.items():
                    self._pingpong.loss[key] = val
                self.__print_loss(self._pingpong.loss)

            if do_preview:
                samples = self._samples.show_sample()
                if samples is not None:
                    viewer(samples, "Training - 'S': Save Now. 'ENTER': Save and Quit")

            if do_timelapse:
                self._timelapse.output_timelapse()

            if do_snapshot:
                self._model.do_snapshot()
        except Exception as err:
            raise err

    def _store_history(self, side, loss):
        """ Store the loss for this step into :attr:`model.history`.

        Parameters
        ----------
        side: {"a", "b"}
            The side to store the loss for
        loss: list
            The list of loss ``floats`` for this side
        """
        logger.trace("Updating loss history: '%s'", side)
        self._model.history[side].append(loss[0])  # Either only loss or total loss
        logger.trace("Updated loss history: '%s'", side)

    def _log_tensorboard(self, side, loss):
        """ Log current loss to Tensorboard log files

        Parameters
        ----------
        side: {"a", "b"}
            The side to store the loss for
        loss: list
            The list of loss ``floats`` for this side
        """
        if not self._tensorboard:
            return
        logger.trace("Updating TensorBoard log: '%s'", side)
        logs = {log[0]: log[1]
                for log in zip(self._model.state.loss_names[side], loss)}
        self._tensorboard[side].on_batch_end(self._model.state.iterations, logs)
        logger.trace("Updated TensorBoard log: '%s'", side)

    def clear_tensorboard(self):
        """ Stop Tensorboard logging.

        Tensorboard logging needs to be explicitly shutdown on training termination. Called from
        :class:`scripts.train.Train` when training is stopped.
         """
        if not self._tensorboard:
            return
        for side, tensorboard in self._tensorboard.items():
            logger.debug("Ending Tensorboard. Side: '%s'", side)
            tensorboard.on_train_end(None)


class Batcher():
    """ Handles the processing of a Batch for a single side.

    Parameters
    ----------
    side: {"a" or "b"}
        The side that this :class:`Batcher` belongs to
    images: list
        The list of full paths to the training images for this :class:`Batcher`
    model: plugin from :mod:`plugins.train.model`
        The selected model that will be running this trainer
    use_mask: bool
        ``True`` if a mask is required for training otherwise ``False``
    batch_size: int
        The size of the batch to be processed at each iteration
    config: :class:`lib.config.FaceswapConfig`
        The configuration for this trainer
    """
    def __init__(self, side, images, model, use_mask, batch_size, config):
        logger.debug("Initializing %s: side: '%s', num_images: %s, use_mask: %s, batch_size: %s, "
                     "config: %s)",
                     self.__class__.__name__, side, len(images), use_mask, batch_size, config)
        self._model = model
        self._use_mask = use_mask
        self._side = side
        self._images = images
        self._config = config
        self._target = None
        self._samples = None
        self._masks = None

        generator = self._load_generator()
        self._feed = generator.minibatch_ab(images, batch_size, self._side)

        self._preview_feed = None
        self._timelapse_feed = None
        self._set_preview_feed()

    def _load_generator(self):
        """ Load the :class:`lib.training_data.TrainingDataGenerator` for this batcher """
        logger.debug("Loading generator: %s", self._side)
        input_size = self._model.input_shape[0]
        output_shapes = self._model.output_shapes
        logger.debug("input_size: %s, output_shapes: %s", input_size, output_shapes)
        generator = TrainingDataGenerator(input_size,
                                          output_shapes,
                                          self._model.training_opts,
                                          self._config)
        return generator

    def train_one_batch(self):
        """ Train on a single batch of images for this :class:`Batcher`

        Returns
        -------
        list
            The list of loss values (as ``float``) for this batch
        """
        logger.trace("Training one step: (side: %s)", self._side)
        model_inputs, model_targets = self._get_next()
        try:
            loss = self._model.predictors[self._side].train_on_batch(model_inputs, model_targets)
        except tf_errors.ResourceExhaustedError as err:
            msg = ("You do not have enough GPU memory available to train the selected model at "
                   "the selected settings. You can try a number of things:"
                   "\n1) Close any other application that is using your GPU (web browsers are "
                   "particularly bad for this)."
                   "\n2) Lower the batchsize (the amount of images fed into the model each "
                   "iteration)."
                   "\n3) Try 'Memory Saving Gradients' and/or 'Optimizer Savings' and/or 'Ping "
                   "Pong Training'."
                   "\n4) Use a more lightweight model, or select the model's 'LowMem' option "
                   "(in config) if it has one.")
            raise FaceswapError(msg) from err
        loss = loss if isinstance(loss, list) else [loss]
        return loss

    def _get_next(self):
        """ Return the next batch from the :class:`lib.training_data.TrainingDataGenerator` for
        this batcher ready for feeding into the model.

        Returns
        -------
        model_inputs: list
            A list of :class:`numpy.ndarray` for feeding into the model
        model_targets: list
            A list of :class:`numpy.ndarray` for comparing the output of the model
        """
        logger.trace("Generating targets")
        batch = next(self._feed)
        targets_use_mask = self._model.training_opts["learn_mask"]
        model_inputs = batch["feed"] + batch["masks"] if self._use_mask else batch["feed"]
        model_targets = batch["targets"] + batch["masks"] if targets_use_mask else batch["targets"]
        return model_inputs, model_targets

    def generate_preview(self, do_preview):
        """ Generate the preview images.

        Parameters
        ----------
        do_preview: bool
            Whether the previews should be generated. ``True`` if they should ``False`` if they
            should not be generated, in which case currently stored previews should be deleted.
        """
        if not do_preview:
            self._samples = None
            self._target = None
            self._masks = None
            return
        logger.debug("Generating preview")
        batch = next(self._preview_feed)
        self._samples = batch["samples"]
        self._target = batch["targets"][self._model.largest_face_index]
        self._masks = batch["masks"][0]

    def _set_preview_feed(self):
        """ Set the preview feed for this batcher.

        Creates a generator from :class:`lib.training_data.TrainingDataGenerator` specifically
        for previews for the batcher.
        """
        logger.debug("Setting preview feed: (side: '%s')", self._side)
        preview_images = self._config.get("preview_images", 14)
        preview_images = min(max(preview_images, 2), 16)
        batchsize = min(len(self._images), preview_images)
        self._preview_feed = self._load_generator().minibatch_ab(self._images,
                                                                 batchsize,
                                                                 self._side,
                                                                 do_shuffle=True,
                                                                 is_preview=True)
        logger.debug("Set preview feed. Batchsize: %s", batchsize)

    def compile_sample(self, batch_size, samples=None, images=None, masks=None):
        """ Compile the preview samples for display.

        Parameters
        ----------
        batch_size: int
            The requested batch size for each training iterations
        samples: :class:`numpy.ndarray`, optional
            The sample images that should be used for creating the preview. If ``None`` then the
            samples will be generated from the internal random image generator.
            Default: ``None``
        images:  :class:`numpy.ndarray`, optional
            The target images that should be used for creating the preview. If ``None`` then the
            targets will be generated from the internal random image generator.
            Default: ``None``
        masks:  :class:`numpy.ndarray`, optional
            The masks that should be used for creating the preview. If ``None`` then the
            masks will be generated from the internal random image generator.
            Default: ``None``

        Returns
        -------
        list
            The list of samples, targets and masks as :class:`numpy.ndarrays` for creating a
            preview image
         """
        num_images = self._config.get("preview_images", 14)
        num_images = min(batch_size, num_images) if batch_size is not None else num_images
        logger.debug("Compiling samples: (side: '%s', samples: %s)", self._side, num_images)
        images = images if images is not None else self._target
        masks = masks if masks is not None else self._masks
        samples = samples if samples is not None else self._samples
        retval = [samples[0:num_images], images[0:num_images], masks[0:num_images]]
        return retval

    def compile_timelapse_sample(self):
        """ Compile the sample images for creating a time-lapse frame.

        Returns
        -------
        list
            The list of samples, targets and masks as :class:`numpy.ndarrays` for creating a
            time-lapse frame
        """
        batch = next(self._timelapse_feed)
        batchsize = len(batch["samples"])
        images = batch["targets"][self._model.largest_face_index]
        masks = batch["masks"][0]
        sample = self.compile_sample(batchsize,
                                     samples=batch["samples"],
                                     images=images,
                                     masks=masks)
        return sample

    def set_timelapse_feed(self, images, batch_size):
        """ Set the time-lapse feed for this batcher.

        Creates a generator from :class:`lib.training_data.TrainingDataGenerator` specifically
        for generating time-lapse previews for the batcher.

        Parameters
        ----------
        images: list
            The list of full paths to the images for creating the time-lapse for this
            :class:`Batcher`
        batch_size: int
            The number of images to be used to create the time-lapse preview.
        """
        logger.debug("Setting time-lapse feed: (side: '%s', input_images: '%s', batch_size: %s)",
                     self._side, images, batch_size)
        self._timelapse_feed = self._load_generator().minibatch_ab(images[:batch_size],
                                                                   batch_size, self._side,
                                                                   do_shuffle=False,
                                                                   is_timelapse=True)
        logger.debug("Set time-lapse feed")


class Samples():
    """ Compile samples for display for preview and time-lapse

    Parameters
    ----------
    model: plugin from :mod:`plugins.train.model`
        The selected model that will be running this trainer
    use_mask: bool
        ``True`` if a mask should be displayed otherwise ``False``
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
    def __init__(self, model, use_mask, coverage_ratio, scaling=1.0):
        logger.debug("Initializing %s: model: '%s', use_mask: %s, coverage_ratio: %s)",
                     self.__class__.__name__, model, use_mask, coverage_ratio)
        self._model = model
        self._use_mask = use_mask
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
        if len(self.images) != 2:
            logger.debug("Ping Pong training - Only one side trained. Aborting preview")
            return None
        logger.debug("Showing sample")
        feeds = dict()
        figures = dict()
        headers = dict()
        for side, samples in self.images.items():
            faces = samples[1]
            if self._model.input_shape[0] / faces.shape[1] != 1.0:
                feeds[side] = self._resize_sample(side, faces, self._model.input_shape[0])
                feeds[side] = feeds[side].reshape((-1, ) + self._model.input_shape)
            else:
                feeds[side] = faces
            if self._use_mask:
                mask = samples[-1]
                feeds[side] = [feeds[side], mask]

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

    @staticmethod
    def _resize_sample(side, sample, target_size):
        """ Resize a given image to the target size.

        Parameters
        ----------
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
        list:
            List of :class:`numpy.ndarray` of predictions received from the model
        """
        logger.debug("Getting Predictions")
        preds = dict()
        preds["a_a"] = self._model.predictors["a"].predict(feed_a)
        preds["b_a"] = self._model.predictors["b"].predict(feed_a)
        preds["a_b"] = self._model.predictors["a"].predict(feed_b)
        preds["b_b"] = self._model.predictors["b"].predict(feed_b)
        # Get the returned largest image from predictors that emit multiple items
        if not isinstance(preds["a_a"], np.ndarray):
            for key, val in preds.items():
                preds[key] = val[self._model.largest_face_index]
        logger.debug("Returning predictions: %s", {key: val.shape for key, val in preds.items()})
        return preds

    def _to_full_frame(self, side, samples, predictions):
        """ Patch targets and prediction images into images of training image size.

        Parameters
        ----------
        side: {"a" or "b"}
            The side that these samples are for
        samples: list
            List of :class:`numpy.ndarray` of target images and feed images
        predictions: list
            List of :class: `numpy.ndarray` of predictions from the model
        """
        logger.debug("side: '%s', number of sample arrays: %s, prediction.shapes: %s)",
                     side, len(samples), [pred.shape for pred in predictions])
        full, faces = samples[:2]
        images = [faces] + predictions
        full_size = full.shape[1]
        target_size = int(full_size * self._coverage_ratio)
        if target_size != full_size:
            frame = self._frame_overlay(full, target_size, (0, 0, 255))

        if self._use_mask:
            images = self._compile_masked(images, samples[-1])
        images = [self._resize_sample(side, image, target_size) for image in images]
        if target_size != full_size:
            images = [self._overlay_foreground(frame, image) for image in images]
        if self._scaling != 1.0:
            new_size = int(full_size * self._scaling)
            images = [self._resize_sample(side, image, new_size) for image in images]
        return images

    @staticmethod
    def _frame_overlay(images, target_size, color):
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

    @staticmethod
    def _compile_masked(faces, masks):
        """ Add the mask to the faces for masked preview.

        Places an opaque red layer over areas of the face that are masked out.

        Parameters
        ----------
        faces: :class:`numpy.ndarray`
            The sample faces that are to have the mask applied
        masks: :class:`numpy.ndarray`
            The masks that are to be applied to the faces

        Returns
        -------
        list
            List of :class:`numpy.ndarray` faces with the opaque mask layer applied
        """
        retval = list()
        masks3 = np.tile(1 - np.rint(masks), 3)
        for mask in masks3:
            mask[np.where((mask == [1., 1., 1.]).all(axis=2))] = [0., 0., 1.]
        for previews in faces:
            images = np.array([cv2.addWeighted(img, 1.0, masks3[idx], 0.3, 0)
                               for idx, img in enumerate(previews)])
            retval.append(images)
        logger.debug("masked shapes: %s", [faces.shape for faces in retval])
        return retval

    @staticmethod
    def _overlay_foreground(backgrounds, foregrounds):
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

    @staticmethod
    def _duplicate_headers(headers, columns):
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


class Timelapse():
    """ Create a time-lapse preview image.

    Parameters
    ----------
    model: plugin from :mod:`plugins.train.model`
        The selected model that will be running this trainer
    use_mask: bool
        ``True`` if a mask should be displayed otherwise ``False``
    coverage_ratio: float
        Ratio of face to be cropped out of the training image.
    scaling: float, optional
        The amount to scale the final preview image by. Default: `1.0`
    image_count: int
        The number of preview images to be displayed in the time-lapse
    batchers: dict
        The dictionary should contain 2 keys ("a" and "b") with the values being the
        :class:`Batcher` for each side.
    """
    def __init__(self, model, use_mask, coverage_ratio, image_count, batchers):
        logger.debug("Initializing %s: model: %s, use_mask: %s, coverage_ratio: %s, "
                     "image_count: %s, batchers: '%s')", self.__class__.__name__, model,
                     use_mask, coverage_ratio, image_count, batchers)
        self._num_images = image_count
        self._samples = Samples(model, use_mask, coverage_ratio)
        self._model = model
        self._batchers = batchers
        self._output_file = None
        logger.debug("Initialized %s", self.__class__.__name__)

    def get_sample(self, side, timelapse_kwargs):
        """ Compile the time-lapse preview

        Parameters
        ----------
        side: {"a" or "b"}
            The side that the time-lapse is being generated for
        timelapse_kwargs: dict
            The keyword arguments for setting up the time-lapse. All values should be full paths
            the keys being `input_a`, `input_b`, `output`
        """
        logger.debug("Getting time-lapse samples: '%s'", side)
        if not self._output_file:
            self._setup(**timelapse_kwargs)
        self._samples.images[side] = self._batchers[side].compile_timelapse_sample()
        logger.debug("Got time-lapse samples: '%s' - %s", side, len(self._samples.images[side]))

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
        for side, image_files in images.items():
            self._batchers[side].set_timelapse_feed(image_files, batchsize)
        logger.debug("Set up time-lapse")

    def output_timelapse(self):
        """ Write the created time-lapse to the specified output folder. """
        logger.debug("Ouputting time-lapse")
        image = self._samples.show_sample()
        if image is None:
            return
        filename = os.path.join(self._output_file, str(int(time.time())) + ".jpg")

        cv2.imwrite(filename, image)
        logger.debug("Created time-lapse: '%s'", filename)


class PingPong():
    """ Side switcher for ping-pong training (memory saving feature)

    Parameters
    ----------
    model: plugin from :mod:`plugins.train.model`
        The selected model that will be running this trainer
    sides: list
        The sorted sides that are to be trained. Generally ["a", "b"]

    Attributes
    ----------
    side: str
        The side that is currently being trained
    loss: dict
        The loss for each side for ping pong training for the current ping pong session
    """
    def __init__(self, model, sides):
        logger.debug("Initializing %s: (model: '%s')", self.__class__.__name__, model)
        self._model = model
        self._sides = sides
        self.side = sorted(sides)[0]
        self.loss = {side: [0] for side in sides}
        logger.debug("Initialized %s", self.__class__.__name__)

    @property
    def active(self):
        """ bool: ``True`` if Ping Pong training is active otherwise ``False``. """
        return self._model.training_opts.get("pingpong", False)

    def switch(self):
        """ Switch ping-pong training from one side of the model to the other """
        if not self.active:
            return
        retval = [side for side in self._sides if side != self.side][0]
        logger.info("Switching training to side %s", retval.title())
        self.side = retval
        self._reload_model()

    def _reload_model(self):
        """ Clear out the model from VRAM and reload for the next side to be trained with ping-pong
        training """
        logger.verbose("Ping-Pong re-loading model")
        self._model.reset_pingpong()


class TrainingAlignments():
    """ Obtain Landmarks and required mask from alignments file.

    Parameters
    ----------
    training_opts: dict
        The dictionary of model training options (see module doc-string for information about
        contents)
    image_list: dict
        The file paths for the images to be trained on for each side. The dictionary should contain
        2 keys ("a" and "b") with the values being a list of full paths corresponding to each side.
    """
    def __init__(self, training_opts, image_list):
        logger.debug("Initializing %s: (training_opts: '%s', image counts: %s)",
                     self.__class__.__name__, training_opts,
                     {k: len(v) for k, v in image_list.items()})
        self._training_opts = training_opts
        self._check_alignments_exist()
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
            face.load_aligned(None, size=self._training_opts["training_size"])
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
            mask = face.mask[self._training_opts["mask_type"]]
            mask.set_blur_and_threshold(blur_kernel=self._training_opts["mask_blur_kernel"],
                                        threshold=self._training_opts["mask_threshold"])
            for filename in self._hash_to_filenames(side, fhash):
                masks[filename] = mask
        return masks

    # Pre flight checks
    def _check_alignments_exist(self):
        """ Ensure the alignments files exist prior to running any longer running tasks.

        Raises
        ------
        FaceswapError
            If at least one alignments file does not exist
        """
        for fullpath in self._training_opts["alignments"].values():
            if not os.path.exists(fullpath):
                raise FaceswapError("Alignments file does not exist: `{}`".format(fullpath))

    # Hashes for image folders
    @staticmethod
    def _get_image_hashes(image_list):
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
        for side, fullpath in self._training_opts["alignments"].items():
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
        mask_type = self._training_opts["mask_type"]
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
