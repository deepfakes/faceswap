#!/usr/bin/env python3


""" Base Trainer Class for Faceswap

    Trainers should be inherited from this class.

    A training_opts dictionary can be set in the corresponding model.
    Accepted values:
        alignments:             dict containing paths to alignments files for keys 'a' and 'b'
        preview_scaling:        How much to scale the preview out by
        training_size:          Size of the training images
        coverage_ratio:         Ratio of face to be cropped out for training
        mask_type:              The type of mask to select from the alignments file
        mask_blur_kernel:       The size of the kernel to use for gaussian blurring the mask
        mask_threshold:         The threshold for min/maxing mask to 0/100
        learn_mask:             Whether the mask should be trained in the model
        penalized_mask_loss:    Whether the mask should be penalized from loss
        no_logs:                Disable tensorboard logging
        snapshot_interval:      Interval for saving model snapshots
        warp_to_landmarks:      Use random_warp_landmarks instead of random_warp
        augment_color:          Perform random shifting of L*a*b* colors
        no_flip:                Don't perform a random flip on the image
        pingpong:               Train each side separately per save iteration rather than together
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


def get_config(plugin_name, configfile=None):
    """ Return the config for the requested model """
    return Config(plugin_name, configfile=configfile).config_dict


class TrainerBase():
    """ Base Trainer """

    def __init__(self, model, images, batch_size, configfile):
        logger.debug("Initializing %s: (model: '%s', batch_size: %s)",
                     self.__class__.__name__, model, batch_size)
        self.config = get_config(".".join(self.__module__.split(".")[-2:]), configfile=configfile)
        self.batch_size = batch_size
        self.model = model
        self.model.state.add_session_batchsize(batch_size)
        self.images = images
        self.sides = sorted(key for key in self.images.keys())

        self.process_training_opts()
        self.pingpong = PingPong(model, self.sides)

        self.batchers = {side: Batcher(side,
                                       images[side],
                                       self.model,
                                       self.use_mask,
                                       batch_size,
                                       self.config)
                         for side in self.sides}

        self.tensorboard = self.set_tensorboard()
        self.samples = Samples(self.model,
                               self.use_mask,
                               self.model.training_opts["coverage_ratio"],
                               self.model.training_opts["preview_scaling"])
        self.timelapse = Timelapse(self.model,
                                   self.use_mask,
                                   self.model.training_opts["coverage_ratio"],
                                   self.config.get("preview_images", 14),
                                   self.batchers)
        logger.debug("Initialized %s", self.__class__.__name__)

    @property
    def timestamp(self):
        """ Standardized timestamp for loss reporting """
        return time.strftime("%H:%M:%S")

    @property
    def landmarks_required(self):
        """ Return True if Landmarks are required """
        retval = self.model.training_opts["warp_to_landmarks"]
        logger.debug(retval)
        return retval

    @property
    def use_mask(self):
        """ Return True if a mask is requested """
        retval = (self.model.training_opts["learn_mask"] or
                  self.model.training_opts["penalized_mask_loss"])
        logger.debug(retval)
        return retval

    def process_training_opts(self):
        """ Override for processing model specific training options """
        logger.debug(self.model.training_opts)
        if not self.landmarks_required and not self.use_mask:
            return

        alignments = TrainingAlignments(self.model.training_opts, self.images)
        if self.landmarks_required:
            logger.debug("Adding landmarks to training opts dict")
            self.model.training_opts["landmarks"] = alignments.landmarks

        if self.use_mask:
            logger.debug("Adding masks to training opts dict")
            self.model.training_opts["masks"] = alignments.masks

    def set_tensorboard(self):
        """ Set up tensorboard callback """
        if self.model.training_opts["no_logs"]:
            logger.verbose("TensorBoard logging disabled")
            return None
        if self.pingpong.active:
            # Currently TensorBoard uses the tf.session, meaning that VRAM does not
            # get cleared when model switching
            # TODO find a fix for this
            logger.warning("Currently TensorBoard logging is not supported for Ping-Pong "
                           "training. Session stats and graphing will not be available for this "
                           "training session.")
            return None

        logger.debug("Enabling TensorBoard Logging")
        tensorboard = dict()

        for side in self.sides:
            logger.debug("Setting up TensorBoard Logging. Side: %s", side)
            log_dir = os.path.join(str(self.model.model_dir),
                                   "{}_logs".format(self.model.name),
                                   side,
                                   "session_{}".format(self.model.state.session_id))
            tbs = tf.keras.callbacks.TensorBoard(log_dir=log_dir, **self.tensorboard_kwargs)
            tbs.set_model(self.model.predictors[side])
            tensorboard[side] = tbs
        logger.info("Enabled TensorBoard Logging")
        return tensorboard

    @property
    def tensorboard_kwargs(self):
        """ Tensorflow 1.13 + needs an additional keyword argument which is not valid for earlier
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

    def print_loss(self, loss):
        """ Override for specific model loss formatting """
        logger.trace(loss)
        output = ["Loss {}: {:.5f}".format(side.capitalize(), loss[side][0])
                  for side in sorted(loss.keys())]
        output = ", ".join(output)
        print("[{}] [#{:05d}] {}".format(self.timestamp, self.model.iterations, output), end='\r')

    def train_one_step(self, viewer, timelapse_kwargs):
        """ Train a batch """
        logger.trace("Training one step: (iteration: %s)", self.model.iterations)
        do_preview = viewer is not None
        do_timelapse = timelapse_kwargs is not None
        snapshot_interval = self.model.training_opts.get("snapshot_interval", 0)
        do_snapshot = (snapshot_interval != 0 and
                       self.model.iterations >= snapshot_interval and
                       self.model.iterations % snapshot_interval == 0)

        loss = dict()
        try:
            for side, batcher in self.batchers.items():
                if self.pingpong.active and side != self.pingpong.side:
                    continue
                loss[side] = batcher.train_one_batch()
                if not do_preview and not do_timelapse:
                    continue
                if do_preview:
                    batcher.generate_preview(do_preview)
                    self.samples.images[side] = batcher.compile_sample(None)
                if do_timelapse:
                    self.timelapse.get_sample(side, timelapse_kwargs)

            self.model.state.increment_iterations()

            for side, side_loss in loss.items():
                self.store_history(side, side_loss)
                self.log_tensorboard(side, side_loss)

            if not self.pingpong.active:
                self.print_loss(loss)
            else:
                for key, val in loss.items():
                    self.pingpong.loss[key] = val
                self.print_loss(self.pingpong.loss)

            if do_preview:
                samples = self.samples.show_sample()
                if samples is not None:
                    viewer(samples, "Training - 'S': Save Now. 'ENTER': Save and Quit")

            if do_timelapse:
                self.timelapse.output_timelapse()

            if do_snapshot:
                self.model.do_snapshot()
        except Exception as err:
            raise err

    def store_history(self, side, loss):
        """ Store the history of this step """
        logger.trace("Updating loss history: '%s'", side)
        self.model.history[side].append(loss[0])  # Either only loss or total loss
        logger.trace("Updated loss history: '%s'", side)

    def log_tensorboard(self, side, loss):
        """ Log loss to TensorBoard log """
        if not self.tensorboard:
            return
        logger.trace("Updating TensorBoard log: '%s'", side)
        logs = {log[0]: log[1]
                for log in zip(self.model.state.loss_names[side], loss)}
        self.tensorboard[side].on_batch_end(self.model.state.iterations, logs)
        logger.trace("Updated TensorBoard log: '%s'", side)

    def clear_tensorboard(self):
        """ Indicate training end to Tensorboard """
        if not self.tensorboard:
            return
        for side, tensorboard in self.tensorboard.items():
            logger.debug("Ending Tensorboard. Side: '%s'", side)
            tensorboard.on_train_end(None)


class Batcher():
    """ Batch images from a single side """
    def __init__(self, side, images, model, use_mask, batch_size, config):
        logger.debug("Initializing %s: side: '%s', num_images: %s, batch_size: %s, config: %s)",
                     self.__class__.__name__, side, len(images), batch_size, config)
        self.model = model
        self.use_mask = use_mask
        self.side = side
        self.images = images
        self.config = config
        self.target = None
        self.samples = None
        self.masks = None

        generator = self.load_generator()
        self.feed = generator.minibatch_ab(images, batch_size, self.side)

        self.preview_feed = None
        self.timelapse_feed = None
        self.set_preview_feed()

    def load_generator(self):
        """ Pass arguments to TrainingDataGenerator and return object """
        logger.debug("Loading generator: %s", self.side)
        input_size = self.model.input_shape[0]
        output_shapes = self.model.output_shapes
        logger.debug("input_size: %s, output_shapes: %s", input_size, output_shapes)
        generator = TrainingDataGenerator(input_size,
                                          output_shapes,
                                          self.model.training_opts,
                                          self.config)
        return generator

    def train_one_batch(self):
        """ Train a batch """
        logger.trace("Training one step: (side: %s)", self.side)
        model_inputs, model_targets = self.get_next()
        try:
            loss = self.model.predictors[self.side].train_on_batch(model_inputs, model_targets)
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

    def get_next(self):
        """ Return the next batch from the generator
            Items should be returned as as: ([model_inputs], [model_targets]) """
        logger.trace("Generating targets")
        batch = next(self.feed)
        # TODO Move this to model or property
        targets_use_mask = self.model.training_opts["learn_mask"]
        # TODO Check if this migration of 2 lists to 1 list is correct
        model_inputs = batch["feed"] + batch["masks"] if self.use_mask else batch["feed"]
        model_targets = batch["targets"] + batch["masks"] if targets_use_mask else batch["targets"]
        return model_inputs, model_targets

    def generate_preview(self, do_preview):
        """ Generate the preview if a preview iteration """
        if not do_preview:
            self.samples = None
            self.target = None
            self.masks = None
            return
        logger.debug("Generating preview")
        batch = next(self.preview_feed)
        self.samples = batch["samples"]
        self.target = batch["targets"][self.model.largest_face_index]
        self.masks = batch["masks"][0]

    def set_preview_feed(self):
        """ Set the preview dictionary """
        logger.debug("Setting preview feed: (side: '%s')", self.side)
        preview_images = self.config.get("preview_images", 14)
        preview_images = min(max(preview_images, 2), 16)
        batchsize = min(len(self.images), preview_images)
        self.preview_feed = self.load_generator().minibatch_ab(self.images,
                                                               batchsize,
                                                               self.side,
                                                               do_shuffle=True,
                                                               is_preview=True)
        logger.debug("Set preview feed. Batchsize: %s", batchsize)

    def compile_sample(self, batch_size, samples=None, images=None, masks=None):
        """ Training samples to display in the viewer """
        num_images = self.config.get("preview_images", 14)
        num_images = min(batch_size, num_images) if batch_size is not None else num_images
        logger.debug("Compiling samples: (side: '%s', samples: %s)", self.side, num_images)
        images = images if images is not None else self.target
        masks = masks if masks is not None else self.masks
        samples = samples if samples is not None else self.samples
        retval = [samples[0:num_images], images[0:num_images], masks[0:num_images]]
        return retval

    def compile_timelapse_sample(self):
        """ Time-lapse samples """
        batch = next(self.timelapse_feed)
        batchsize = len(batch["samples"])
        images = batch["targets"][self.model.largest_face_index]
        masks = batch["masks"][0]
        sample = self.compile_sample(batchsize,
                                     samples=batch["samples"],
                                     images=images,
                                     masks=masks)
        return sample

    def set_timelapse_feed(self, images, batchsize):
        """ Set the time-lapse dictionary """
        logger.debug("Setting time-lapse feed: (side: '%s', input_images: '%s', batchsize: %s)",
                     self.side, images, batchsize)
        self.timelapse_feed = self.load_generator().minibatch_ab(images[:batchsize],
                                                                 batchsize, self.side,
                                                                 do_shuffle=False,
                                                                 is_timelapse=True)
        logger.debug("Set time-lapse feed")


class Samples():
    """ Display samples for preview and time-lapse """
    def __init__(self, model, use_mask, coverage_ratio, scaling=1.0):
        logger.debug("Initializing %s: model: '%s', use_mask: %s, coverage_ratio: %s)",
                     self.__class__.__name__, model, use_mask, coverage_ratio)
        self.model = model
        self.use_mask = use_mask
        self.images = dict()
        self.coverage_ratio = coverage_ratio
        self.scaling = scaling
        logger.debug("Initialized %s", self.__class__.__name__)

    def show_sample(self):
        """ Display preview data """
        if len(self.images) != 2:
            logger.debug("Ping Pong training - Only one side trained. Aborting preview")
            return None
        logger.debug("Showing sample")
        feeds = dict()
        figures = dict()
        headers = dict()
        for side, samples in self.images.items():
            faces = samples[1]
            if self.model.input_shape[0] / faces.shape[1] != 1.0:
                feeds[side] = self.resize_sample(side, faces, self.model.input_shape[0])
                feeds[side] = feeds[side].reshape((-1, ) + self.model.input_shape)
            else:
                feeds[side] = faces
            if self.use_mask:
                mask = samples[-1]
                feeds[side] = [feeds[side], mask]

        preds = self.get_predictions(feeds["a"], feeds["b"])

        for side, samples in self.images.items():
            other_side = "a" if side == "b" else "b"
            predictions = [preds["{0}_{0}".format(side)],
                           preds["{}_{}".format(other_side, side)]]
            display = self.to_full_frame(side, samples, predictions)
            headers[side] = self.get_headers(side, display[0].shape[1])
            figures[side] = np.stack([display[0], display[1], display[2], ], axis=1)
            if self.images[side][0].shape[0] % 2 == 1:
                figures[side] = np.concatenate([figures[side],
                                                np.expand_dims(figures[side][0], 0)])

        width = 4
        side_cols = width // 2
        if side_cols != 1:
            headers = self.duplicate_headers(headers, side_cols)

        header = np.concatenate([headers["a"], headers["b"]], axis=1)
        figure = np.concatenate([figures["a"], figures["b"]], axis=0)
        height = int(figure.shape[0] / width)
        figure = figure.reshape((width, height) + figure.shape[1:])
        figure = stack_images(figure)
        figure = np.concatenate((header, figure), axis=0)

        logger.debug("Compiled sample")
        return np.clip(figure * 255, 0, 255).astype('uint8')

    @staticmethod
    def resize_sample(side, sample, target_size):
        """ Resize samples where predictor expects different shape from processed image """
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

    def get_predictions(self, feed_a, feed_b):
        """ Return the sample predictions from the model """
        logger.debug("Getting Predictions")
        preds = dict()
        preds["a_a"] = self.model.predictors["a"].predict(feed_a)
        preds["b_a"] = self.model.predictors["b"].predict(feed_a)
        preds["a_b"] = self.model.predictors["a"].predict(feed_b)
        preds["b_b"] = self.model.predictors["b"].predict(feed_b)
        # Get the returned largest image from predictors that emit multiple items
        if not isinstance(preds["a_a"], np.ndarray):
            for key, val in preds.items():
                preds[key] = val[self.model.largest_face_index]
        logger.debug("Returning predictions: %s", {key: val.shape for key, val in preds.items()})
        return preds

    def to_full_frame(self, side, samples, predictions):
        """ Patch the images into the full frame """
        logger.debug("side: '%s', number of sample arrays: %s, prediction.shapes: %s)",
                     side, len(samples), [pred.shape for pred in predictions])
        full, faces = samples[:2]
        images = [faces] + predictions
        full_size = full.shape[1]
        target_size = int(full_size * self.coverage_ratio)
        if target_size != full_size:
            frame = self.frame_overlay(full, target_size, (0, 0, 255))

        if self.use_mask:
            images = self.compile_masked(images, samples[-1])
        images = [self.resize_sample(side, image, target_size) for image in images]
        if target_size != full_size:
            images = [self.overlay_foreground(frame, image) for image in images]
        if self.scaling != 1.0:
            new_size = int(full_size * self.scaling)
            images = [self.resize_sample(side, image, new_size) for image in images]
        return images

    @staticmethod
    def frame_overlay(images, target_size, color):
        """ Add roi frame to a background image """
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
    def compile_masked(faces, masks):
        """ Add the mask to the faces for masked preview """
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
    def overlay_foreground(backgrounds, foregrounds):
        """ Overlay the training images into the center of the background """
        offset = (backgrounds.shape[1] - foregrounds.shape[1]) // 2
        new_images = list()
        for idx, img in enumerate(backgrounds):
            img[offset:offset + foregrounds[idx].shape[0],
                offset:offset + foregrounds[idx].shape[1], :3] = foregrounds[idx]
            new_images.append(img)
        retval = np.array(new_images)
        logger.debug("Overlayed foreground. Shape: %s", retval.shape)
        return retval

    def get_headers(self, side, width):
        """ Set headers for images """
        logger.debug("side: '%s', width: %s",
                     side, width)
        titles = ("Original", "Swap") if side == "a" else ("Swap", "Original")
        side = side.upper()
        height = int(64 * self.scaling)
        total_width = width * 3
        logger.debug("height: %s, total_width: %s", height, total_width)
        font = cv2.FONT_HERSHEY_SIMPLEX
        texts = ["{} ({})".format(titles[0], side),
                 "{0} > {0}".format(titles[0]),
                 "{} > {}".format(titles[0], titles[1])]
        text_sizes = [cv2.getTextSize(texts[idx], font, self.scaling * 0.8, 1)[0]
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
                        self.scaling * 0.8,
                        (0, 0, 0),
                        1,
                        lineType=cv2.LINE_AA)
        logger.debug("header_box.shape: %s", header_box.shape)
        return header_box

    @staticmethod
    def duplicate_headers(headers, columns):
        """ Duplicate headers for the number of columns displayed """
        for side, header in headers.items():
            duped = tuple([header for _ in range(columns)])
            headers[side] = np.concatenate(duped, axis=1)
            logger.debug("side: %s header.shape: %s", side, header.shape)
        return headers


class Timelapse():
    """ Create the time-lapse """
    def __init__(self, model, use_mask, coverage_ratio, preview_images, batchers):
        logger.debug("Initializing %s: model: %s, use_mask: %s, coverage_ratio: %s, "
                     "preview_images: %s, batchers: '%s')", self.__class__.__name__, model,
                     use_mask, coverage_ratio, preview_images, batchers)
        self.preview_images = preview_images
        self.samples = Samples(model, use_mask, coverage_ratio)
        self.model = model
        self.batchers = batchers
        self.output_file = None
        logger.debug("Initialized %s", self.__class__.__name__)

    def get_sample(self, side, timelapse_kwargs):
        """ Perform time-lapse """
        logger.debug("Getting time-lapse samples: '%s'", side)
        if not self.output_file:
            self.setup(**timelapse_kwargs)
        self.samples.images[side] = self.batchers[side].compile_timelapse_sample()
        logger.debug("Got time-lapse samples: '%s' - %s", side, len(self.samples.images[side]))

    def setup(self, input_a=None, input_b=None, output=None):
        """ Set the time-lapse output folder """
        logger.debug("Setting up time-lapse")
        if output is None:
            output = str(get_folder(os.path.join(str(self.model.model_dir),
                                                 "{}_timelapse".format(self.model.name))))
        self.output_file = str(output)
        logger.debug("Time-lapse output set to '%s'", self.output_file)

        images = {"a": get_image_paths(input_a), "b": get_image_paths(input_b)}
        batchsize = min(len(images["a"]),
                        len(images["b"]),
                        self.preview_images)
        for side, image_files in images.items():
            self.batchers[side].set_timelapse_feed(image_files, batchsize)
        logger.debug("Set up time-lapse")

    def output_timelapse(self):
        """ Set the time-lapse dictionary """
        logger.debug("Ouputting time-lapse")
        image = self.samples.show_sample()
        if image is None:
            return
        filename = os.path.join(self.output_file, str(int(time.time())) + ".jpg")

        cv2.imwrite(filename, image)
        logger.debug("Created time-lapse: '%s'", filename)


class PingPong():
    """ Side switcher for ping-pong training """
    def __init__(self, model, sides):
        logger.debug("Initializing %s: (model: '%s')", self.__class__.__name__, model)
        self.active = model.training_opts.get("pingpong", False)
        self.model = model
        self.sides = sides
        self.side = sorted(sides)[0]
        self.loss = {side: [0] for side in sides}
        logger.debug("Initialized %s", self.__class__.__name__)

    def switch(self):
        """ Switch ping-pong side """
        if not self.active:
            return
        retval = [side for side in self.sides if side != self.side][0]
        logger.info("Switching training to side %s", retval.title())
        self.side = retval
        self.reload_model()

    def reload_model(self):
        """ Load the model for just the current side """
        logger.verbose("Ping-Pong re-loading model")
        self.model.reset_pingpong()


class TrainingAlignments():
    """ Get Landmarks and required mask from alignments file """
    def __init__(self, training_opts, image_list):
        logger.debug("Initializing %s: (training_opts: '%s', image counts: %s)",
                     self.__class__.__name__, training_opts,
                     {k: len(v) for k, v in image_list.items()})
        self._training_opts = training_opts
        self._hashes = self._get_image_hashes(image_list)
        self._detected_faces = self._load_alignments()
        self._check_all_faces()
        logger.debug("Initialized %s", self.__class__.__name__)

    @property
    def landmarks(self):
        """dict: The transformed landmarks for each face set """
        retval = {side: self._transform_landmarks(side, detected_faces)
                  for side, detected_faces in self._detected_faces.items()}
        logger.trace(retval)
        return retval

    @property
    def masks(self):
        """dict: The mask objects of requested mask_type for each face set """
        retval = {side: self._get_masks(side, detected_faces)
                  for side, detected_faces in self._detected_faces.items()}
        logger.trace(retval)
        return retval

    # Load alignments
    @staticmethod
    def _get_image_hashes(image_list):
        """ Return the hashes for all images used for training """
        hashes = {key: dict(hashes=[], filenames=[]) for key in image_list}
        pbar = tqdm(desc="Reading training images",
                    total=sum(len(val) for val in image_list.values()))
        for side, filelist in image_list.items():
            logger.debug("side: %s, file count: %s", side, len(filelist))
            for filename, hsh in read_image_hash_batch(filelist):
                hashes[side]["hashes"].append(hsh)
                hashes[side]["filenames"].append(filename)
                pbar.update(1)
        pbar.close()
        logger.trace(hashes)
        return hashes

    def _load_alignments(self):
        """ Load the alignments as list of detected faces for each side in a dict """
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
        """ Return a list of :class:`lib.faces_detect.DetectedFace` objects from a
        :class:'lib.alignments.Alignments' object

        Filter the detected faces to only those that exist in the training folders
        """
        skip_count = 0
        side_hashes = set(self._hashes[side]["hashes"])
        detected_faces = []
        for _, faces, _, filename in alignments.yield_faces():
            for idx, face in enumerate(faces):
                if not self._validate_face(face, filename, idx, side, side_hashes):
                    skip_count += 1
                    continue
                detected_face = DetectedFace()
                detected_face.from_alignment(face)
                detected_faces.append(detected_face)
        logger.debug("Detected Faces count: %s, Skipped faces count: %s",
                     len(detected_faces), skip_count)
        if skip_count != 0:
            logger.warning("%s alignments have been removed as their corresponding faces do not "
                           "exist in the input folder for side %s. Run in verbose mode if you "
                           "wish to see which alignments have been excluded.",
                           skip_count, side.upper())
        return detected_faces

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
                   "The face that caused this failure was side: `{}`, frame: `{}`, index: {}. The "
                   "masks that exist for this face are: {}.\nBe aware that there are probably "
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
            If not, output missing filenames """
        logger.debug("Checking faces exist in alignments")
        missing_alignments = dict()
        for side, train_hashes in self._hashes.items():
            align_hashes = set(face.hash for face in self._detected_faces[side])
            if not align_hashes.issuperset(train_hashes["hashes"]):
                missing_alignments[side] = [
                    os.path.basename(filename)
                    for hsh, filename in zip(train_hashes["hashes"], train_hashes["filenames"])
                    if hsh not in align_hashes]
        if missing_alignments:
            msg = ("There are faces in your training folder(s) which do not exist in your "
                   "alignments file. Training cannot continue. See above for a full list of "
                   "files missing alignments.")
            for side, filelist in missing_alignments.items():
                logger.error("Faces missing alignments for side %s: %s",
                             side.capitalize(), filelist)
            raise FaceswapError(msg)

    # Get landmarks
    def _transform_landmarks(self, side, detected_faces):
        """ For each face transform landmarks and return """
        landmarks = dict()
        for face in detected_faces:
            face.load_aligned(None, size=self._training_opts["training_size"])
            for filename in self._hash_to_filenames(side, face.hash):
                landmarks[filename] = face.aligned_landmarks
        return landmarks

    # Get masks
    def _get_masks(self, side, detected_faces):
        """ For each face, get the mask and set the requested blurring and threshold level """
        masks = dict()
        for face in detected_faces:
            mask = face.mask[self._training_opts["mask_type"]]
            mask.set_blur_kernel_and_threshold(blur_kernel=self._training_opts["mask_blur_kernel"],
                                               threshold=self._training_opts["mask_threshold"])
            for filename in self._hash_to_filenames(side, face.hash):
                masks[filename] = mask
        return masks

    def _hash_to_filenames(self, side, face_hash):
        """ For a given hash return all the filenames that match for the given side
        NB: Multiple faces can have the same hash, so make sure filenames are updated
            for all instances of a hash
        """
        side_hashes = self._hashes[side]
        hash_indices = [idx for idx, hsh in enumerate(side_hashes["hashes"]) if hsh == face_hash]
        retval = [side_hashes["filenames"][idx] for idx in hash_indices]
        logger.trace("side: %s, hash: %s, filenames: %s", side, face_hash, retval)
        return retval


def stack_images(images):
    """ Stack images """
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
