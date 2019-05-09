#!/usr/bin/env python3


""" Base Trainer Class for Faceswap

    Trainers should be inherited from this class.

    A training_opts dictionary can be set in the corresponding model.
    Accepted values:
        alignments:         dict containing paths to alignments files for keys 'a' and 'b'
        preview_scaling:    How much to scale the preview out by
        training_size:      Size of the training images
        coverage_ratio:     Ratio of face to be cropped out for training
        no_logs:            Disable tensorboard logging
        warp_to_landmarks:  Use random_warp_landmarks instead of random_warp
        no_flip:            Don't perform a random flip on the image
        pingpong:           Train each side seperately per save iteration rather than together
"""

import logging
import os
import time
from pathlib import Path
import cv2
import numpy as np

from tensorflow import keras as tf_keras
from lib.alignments import Alignments
from lib.faces_detect import DetectedFace
from lib.training_data import TrainingDataGenerator
from lib.utils import get_folder, get_image_paths

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class TrainerBase():
    """ Base Trainer """

    def __init__(self, model, images, timelapse_kwargs, batch_size):
        logger.debug("Initializing %s: (model: '%s', batch_size: %s)",
                     self.__class__.__name__, model, batch_size)
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
                                       batch_size)
                         for side in self.sides}

        self.tensorboard = self.set_tensorboard()
        self.samples = Samples(self.model,
                               self.model.training_opts["coverage_ratio"],
                               self.model.training_opts["preview_scaling"])
        if timelapse_kwargs:
            self.timelapse = Timelapse(self.model,
                                       self.model.training_opts["coverage_ratio"],
                                       self.batchers,
                                       timelapse_kwargs)
        logger.debug("Initialized %s", self.__class__.__name__)

    @property
    def timestamp(self):
        """ Standardised timestamp for loss reporting """
        return time.strftime("%H:%M:%S")

    @property
    def landmarks_required(self):
        """ Return True if Landmarks are required """
        opts = self.model.training_opts
        retval = bool(opts.get("mask_type", None) or opts["warp_to_landmarks"])
        logger.debug(retval)
        return retval

    def process_training_opts(self):
        """ Override for processing model specific training options """
        logger.debug(self.model.training_opts)
        if self.landmarks_required:
            landmarks = Landmarks(self.model.training_opts).landmarks
            self.model.training_opts["landmarks"] = landmarks

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
            tbs = tf_keras.callbacks.TensorBoard(log_dir=log_dir,
                                                 histogram_freq=0,  # Must be 0 or hangs
                                                 batch_size=self.batch_size,
                                                 write_graph=True,
                                                 write_grads=True)
            tbs.set_model(self.model.predictors[side])
            tensorboard[side] = tbs
        logger.info("Enabled TensorBoard Logging")
        return tensorboard

    def print_loss(self, loss):
        """ Override for specific model loss formatting """
        logger.trace(loss)
        output = list()
        for side in sorted(list(loss.keys())):
            display = ", ".join(["{}_{}: {:.5f}".format(self.model.state.loss_names[side][idx],
                                                        side.capitalize(),
                                                        this_loss)
                                 for idx, this_loss in enumerate(loss[side])])
            output.append(display)
        output = ", ".join(output)
        print("[{}] [#{:05d}] {}".format(self.timestamp, self.model.iterations, output), end='\r')

    def train_one_step(self):
        """ Train a batch """
        logger.trace("Training one step: (iteration: %s)", self.model.iterations)
        loss = dict()
        for side, batcher in self.batchers.items():
            if self.pingpong.active and side != self.pingpong.side:
                continue
            loss[side] = batcher.train_one_batch("training")

        self.model.state.increment_iterations()

        for side, side_loss in loss.items():
            self.store_history(side, side_loss)
            if self.tensorboard:
                self.log_tensorboard(side, side_loss)

        if self.pingpong.active:
            for key, val in loss.items():
                self.pingpong.loss[key] = val
            self.print_loss(self.pingpong.loss)
        else:
            self.print_loss(loss)

    def preview(self, viewer, timelapse):
        """ Preview Samples """
        if viewer:
            for side, batcher in self.batchers.items():
                _, _, self.samples.images[side] = batcher.get_next("preview")
            samples = self.samples.show_sample()
            if samples is not None:
                viewer(samples, "Training - 'S': Save Now. 'ENTER': Save and Quit")

        if timelapse:
            for side, batcher in self.batchers.items():
                _, _, self.timelapse.samples.images[side] = batcher.get_next("timelapse")
            self.timelapse.output_timelapse()

    def store_history(self, side, loss):
        """ Store the history of this step """
        logger.trace("Updating loss history: '%s'", side)
        self.model.history[side].append(loss[0])  # Either only loss or total loss
        logger.trace("Updated loss history: '%s'", side)

    def log_tensorboard(self, side, loss):
        """ Log loss to TensorBoard log """
        logger.trace("Updating TensorBoard log: '%s'", side)
        dual_generator = zip(self.model.state.loss_names[side], loss)
        logs = {log[0]: log[1] for log in dual_generator}
        self.tensorboard[side].on_batch_end(self.model.state.iterations, logs)
        logger.trace("Updated TensorBoard log: '%s'", side)

    def clear_tensorboard(self):
        """ Indicate training end to Tensorboard """
        if self.tensorboard:
            for side, tensorboard in self.tensorboard.items():
                logger.debug("Ending Tensorboard. Side: '%s'", side)
                tensorboard.on_train_end(None)


class Batcher():
    """ Batch images from a single side """
    def __init__(self, side, images, model, batch_size):
        logger.debug("Initializing %s: side: '%s', num_images: %s, batch_size: %s)",
                     self.__class__.__name__, side, len(images), batch_size)
        self.side = side
        self.model = model
        self.batch_size = batch_size
        self.feed = dict()
        preview_size = model.training_opts.get("preview_images", 14)
        self.feed["training"] = self.load_generator().minibatch_ab(images,
                                                                   batch_size,
                                                                   side,
                                                                   do_shuffle=True,
                                                                   augmenting=True)
        self.feed["preview"] = self.load_generator().minibatch_ab(images,
                                                                  preview_size,
                                                                  side,
                                                                  do_shuffle=True,
                                                                  augmenting=False)

    def train_one_batch(self, purpose):
        """ Train a batch """
        logger.trace("Training one step: (side: %s)", self.side)
        inputs, targets, _ = self.get_next(purpose)
        loss = self.model.predictors[self.side].train_on_batch(x=inputs, y=targets)
        loss = loss if isinstance(loss, list) else [loss]
        return loss

    def get_next(self, purpose):
        """ Return the next batch from the generator
            Items should come out as: (full_coverage_img, warped, target, mask]) """
        batch = next(self.feed[purpose])
        inputs = [batch[1], batch[2]]
        targets = [batch[3], batch[4]]
        samples = None
        if purpose != "training":
            samples = [batch[0]] + targets
        return inputs, targets, samples

    def load_generator(self):
        """ Set the feed dataset with TrainingDataGenerator """
        logger.debug("Loading generator: (side: '%s')",self.side)
        input_size = self.model.input_shape[0]
        output_size = self.model.output_shape[0]
        logger.debug("input_size: %s, output_size: %s", input_size, output_size)
        generator = TrainingDataGenerator(input_size,
                                          output_size,
                                          self.model.training_opts)
        logger.debug("'%s' dataset created")
        return generator


class Samples():
    """ Display samples for preview and timelapse """
    def __init__(self, model, coverage_ratio, scaling=1.0):
        logger.debug("Initializing %s: model: '%s', coverage_ratio: %s)",
                     self.__class__.__name__, model, coverage_ratio)
        self.model = model
        self.images = dict()
        self.coverage_ratio = coverage_ratio
        self.scaling = scaling
        logger.debug("Initialized %s", self.__class__.__name__)

    def show_sample(self):
        """ Display preview data """
        if len(self.images) == 2:
            logger.debug("Showing sample")
            display = dict()
            for side, samples in self.images.items():
                other_side = "a" if side == "b" else "b"
                _, faces, masks = samples
                if self.model.input_shape[0] / faces.shape[1] != 1.:
                    faces = self.resize_samples(side, faces, self.model.input_shape[0])
                    faces = faces.reshape((-1, ) + self.model.input_shape)

                logger.debug("Getting Predictions for side %s", side)
                reconstructions = self.model.predictors[side].predict([faces, masks])[0]
                swaps = self.model.predictors[other_side].predict([faces, masks])[0]
                logger.debug("Returning predictions: %s", swaps.shape)

                imgs = self.patch_into_frame(side, samples, [reconstructions, swaps])
                header = self.get_headers(side, other_side, imgs[0].shape[1])
                figures = np.concatenate([imgs[0], imgs[1], imgs[2]], axis=2)
                figures = np.concatenate([img for img in figures], axis=0)
                halves = np.split(figures, 2)
                header = np.concatenate([header, header], axis=1)
                figures = np.concatenate([halves[0], halves[1]], axis=1)
                display[side] = np.concatenate([header, figures], axis=0)

            full_display = np.concatenate([display["a"], display["b"]], axis=1)
            full_display = np.clip(full_display * 255., 0., 255.).astype('uint8')
            logger.debug("Compiled sample")
        else:
            full_display = None
            logger.debug("Ping Pong training - Only one side trained. Aborting preview")

        return full_display

    @staticmethod
    def resize_samples(side, samples, scale):
        """ Resize samples where predictor expects different shape from processed image """
        if scale != 1.:
            logger.debug("Resizing samples: (side: '%s', sample.shape: %s, scale: %s)",
                         side, samples[0].shape, scale)
            interp = cv2.INTER_CUBIC if scale > 1. else cv2.INTER_AREA  # pylint: disable=no-member
            samples = np.stack(np.stack(cv2.resize(img, None, fx=scale, fy=scale,  # pylint: disable=no-member
                                                   interpolation=interp) for img in face_batch) for face_batch in samples)
            logger.debug("Resized sample: (side: '%s' shape: %s)", side, samples[0].shape)
        return samples

    def patch_into_frame(self, side, samples, predictions):
        """ Patch the images into the full frame """
        logger.debug("side: '%s', number of sample arrays: %s, prediction.shapes: %s)",
                     side, len(samples), [pred.shape for pred in predictions])

        frames, original_faces, masks = samples
        images = np.concatenate([original_faces[None, ...], predictions], axis=0)
        unadjusted_scale = frames.shape[1] / original_faces.shape[1]
        target_scale = unadjusted_scale * self.coverage_ratio
        new_scale = unadjusted_scale * self.scaling

        images = self.tint_masked(images, masks)
        images = self.resize_samples(side, images, target_scale)
        if self.coverage_ratio != 1.:
            frames = self.frame_overlay(frames)
            images = self.overlay_foreground(frames, images)
        if self.scaling != 1.:
            images = self.resize_samples(side, images, self.scaling)
        return images

    def frame_overlay(self, frames):
        """ Add roi frame to a backfround image """
        color = (0., 0., 1.)
        logger.debug("full_size: %s, color: %s", frames.shape[1], color)
        full_size = frames.shape[1]
        padding = int(full_size * (1. - self.coverage_ratio)) // 2
        length = int(full_size * self.coverage_ratio) // 4
        t_l = padding - 3
        b_r = full_size - padding + 3

        top_left = slice(t_l, t_l + length), slice(t_l, t_l + length)
        bot_left = slice(b_r - length, b_r), slice(t_l, t_l + length)
        top_right = slice(b_r - length, b_r), slice(b_r - length, b_r)
        bot_right = slice(t_l, t_l + length), slice(b_r - length, b_r)
        for roi in [top_left, bot_left, top_right, bot_right]:
            frames[:, roi[0], roi[1]] = color
        logger.debug("Overlayed background. Shape: %s", frames.shape)
        return frames

    @staticmethod
    def tint_masked(images, masks):
        """ Add the mask to the faces for masked preview """
        red_area = (np.rint(masks) == 0.)
        coloring = np.zeros_like(masks)
        coloring[red_area] = 0.3
        images[..., -1:] += coloring
        logger.debug("masked shapes: %s", [faces.shape for faces in images[0]])
        return images

    @staticmethod
    def overlay_foreground(backgrounds, foregrounds):
        """ Overlay the training images into the center of the background """
        backgrounds[..., -1:] += 0.3
        offset = (backgrounds.shape[1] - foregrounds.shape[2]) // 2
        slice_y = slice(offset, offset + foregrounds.shape[2])
        slice_x = slice(offset, offset + foregrounds.shape[3])
        new_images = np.repeat(backgrounds[None, ...], 3, axis=0)  #TODO why expand dims
        for background, foreground in zip(new_images, foregrounds):
            for fore, back in zip(foreground, background):
                back[slice_y, slice_x, :] = fore
        logger.debug("Overlayed foreground. Shape: %s", new_images.shape)
        return new_images

    def get_headers(self, side, other_side, width):
        """ Set headers for images """
        # pylint: disable=no-member
        logger.debug("side: '%s', other_side: '%s', width: %s", side, other_side, width)

        def text_size(text, font):
            """ Helper function for list comprehension """
            # pylint: disable=no-member
            [text_width, text_height] = cv2.getTextSize(text, font, self.scaling, 1)[0]
            return [text_width, text_height]

        side = side.upper()
        other_side = other_side.upper()
        height = int(64. * self.scaling)
        offsets = [0, width, width * 2]
        header_box = np.ones((height, width * 3, 3), np.float32)
        texts = ["Target {0}".format(side),
                 "{0} > {0}".format(side),
                 "{0} > {1}".format(side, other_side)]
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_sizes = [text_size(text, font) for text in texts]
        text_ys = [int((height + text[1]) / 2) for text in text_sizes]
        text_xs = [int((width - text[0]) / 2 + offset) for offset, text in zip(offsets, text_sizes)]
        for text_x, text_y, text in zip(text_xs, text_ys, texts):
            cv2.putText(header_box,
                        text,
                        (text_x, text_y),
                        font,
                        self.scaling,
                        (0., 0., 0.),
                        1,
                        lineType=cv2.LINE_AA)

        logger.debug("height: %s, total_width: %s", height, width * 3)
        logger.debug("texts: %s, text_sizes: %s, text_x: %s, text_y: %s",
                     texts, text_sizes, text_xs, text_ys)
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
    """ Create the timelapse """
    def __init__(self, model, coverage_ratio, batchers, timelapse_kwargs):
        logger.debug("Initializing %s: model: %s, coverage_ratio: %s, "
                     "batchers: '%s')", self.__class__.__name__, model, coverage_ratio, batchers)
        self.samples = Samples(model, coverage_ratio)
        self.model = model
        self.batchers = batchers
        self.output_file = None
        if timelapse_kwargs:
            self.setup(**timelapse_kwargs)
        logger.debug("Initialized %s", self.__class__.__name__)

    def setup(self, input_a=None, input_b=None, output=None):
        """ Set the timelapse output folder """
        logger.debug("Setting up timelapse")
        if output is None:
            str_dir = "{}".format(self.model.model_dir)
            str_name = "{}_timelapse".format(self.model.name)
            model_path = Path(str_dir) / str_name
            output = get_folder(model_path)
        self.output_file = str(output)
        logger.debug("Timelapse output set to '%s'", self.output_file)

        images = {"a": get_image_paths(input_a), "b": get_image_paths(input_b)}
        batch_size = min(len(images["a"]),
                         len(images["b"]),
                         self.model.training_opts.get("preview_images", 14))
        for side, images in images.items():
            self.batchers[side].feed["timelapse"] = self.batchers[side].load_generator().minibatch_ab(images,
                                                                                                      batch_size,
                                                                                                      side,
                                                                                                      False,
                                                                                                      False)
        logger.debug("Set up timelapse")

    def output_timelapse(self):
        """ Set the timelapse dictionary """
        logger.debug("Ouputting timelapse")
        image = self.samples.show_sample()
        if image is not None:
            #TODO Path(self.output_file) / "{0}.h5".format(str(int(time.time())))
            filename = os.path.join(self.output_file, str(int(time.time())) + ".jpg")
            cv2.imwrite(filename, image)  # pylint: disable=no-member
            logger.debug("Created timelapse: '%s'", filename)


class PingPong():
    """ Side switcher for pingpong training """
    def __init__(self, model, sides):
        logger.debug("Initializing %s: (model: '%s')", self.__class__.__name__, model)
        self.active = model.training_opts.get("pingpong", False)
        self.model = model
        self.sides = sides
        self.side = sorted(sides)[0]
        self.loss = {side: dict() for side in sides}
        logger.debug("Initialized %s", self.__class__.__name__)

    def switch(self):
        """ Switch pingpong side """
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


class Landmarks():
    """ Set Landmarks for training into the model's training options"""
    def __init__(self, training_opts):
        logger.debug("Initializing %s: (training_opts: '%s')",
                     self.__class__.__name__, training_opts)
        self.size = training_opts.get("training_size", 256)
        self.paths = training_opts["alignments"]
        self.landmarks = self.get_alignments()
        logger.debug("Initialized %s", self.__class__.__name__)

    def get_alignments(self):
        """ Obtain the landmarks for each faceset """
        landmarks = dict()
        for side, fullpath in self.paths.items():
            path, filename = os.path.split(fullpath)
            filename, extension = os.path.splitext(filename)
            serializer = extension[1:]
            alignments = Alignments(
                path,
                filename=filename,
                serializer=serializer)
            landmarks[side] = self.transform_landmarks(alignments)
        return landmarks

    def transform_landmarks(self, alignments):
        """ For each face transform landmarks and return """
        landmarks = dict()
        for _, faces, _, _ in alignments.yield_faces():
            for face in faces:
                detected_face = DetectedFace()
                detected_face.from_alignment(face)
                detected_face.load_aligned(None, size=self.size, align_eyes=False)
                landmarks[detected_face.hash] = detected_face.aligned_landmarks
        return landmarks
