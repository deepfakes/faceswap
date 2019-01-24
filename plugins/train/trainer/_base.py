#!/usr/bin/env python3


""" Base Trainer Class for Faceswap

    Trainers should be inherited from this class.

    A training_opts dictionary can be set in the corresponding model.
    Accepted values:
        serializer:     format that alignments data is serialized in
        mask_type:      Type of mask to use. See lib.model.masks for valid mask names
        full_face:      Set to True if training should use full face
        preview_images: Number of preview images to display (default: 14)
"""

import logging
import os
import time

import cv2
import numpy as np

from lib.alignments import Alignments
from lib.faces_detect import DetectedFace
from lib.training_data import TrainingDataGenerator, stack_images
from lib.utils import get_folder, get_image_paths

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class TrainerBase():
    """ Base Trainer """

    def __init__(self, model, images, batch_size):
        logger.debug("Initializing %s: (model: '%s', batch_size: %s)",
                     self.__class__.__name__, model, batch_size)
        self.batch_size = batch_size
        self.model = model
        self.images = images
        use_mask = self.process_training_opts()

        self.batchers = {side: Batcher(side,
                                       images[side],
                                       self.model,
                                       use_mask,
                                       batch_size)
                         for side in images.keys()}
        self.samples = Samples(self.model, use_mask)
        self.timelapse = Timelapse(self.model, use_mask, self.batchers)
        logger.debug("Initialized %s", self.__class__.__name__)

    @property
    def timestamp(self):
        """ Standardised timestamp for loss reporting """
        return time.strftime("%H:%M:%S")

    def process_training_opts(self):
        """ Override for processing model specific training options """
        logger.debug(self.model.training_opts)
        use_mask = False
        if self.model.training_opts.get("mask_type", None):
            use_mask = True
            Landmarks(self.images, self.model).get_alignments()
        return use_mask

    def print_loss(self, loss):
        """ Override for specific model loss formatting """
        output = list()
        for side in sorted(list(loss.keys())):
            display = ", ".join(["{}_{}: {:.5f}".format(self.model.loss_names[side][idx],
                                                        side.capitalize(),
                                                        this_loss)
                                 for idx, this_loss in enumerate(loss[side])])
            output.append(display)
        print("[{}] [#{:05d}] {}, {}".format(
            self.timestamp, self.model.iterations, output[0], output[1]), end='\r')

    def train_one_step(self, viewer, timelapse_kwargs):
        """ Train a batch """
        logger.trace("Training one step: (iteration: %s)", self.model.iterations)
        is_preview_iteration = False if viewer is None else True
        loss = dict()
        for side, batcher in self.batchers.items():
            loss[side] = batcher.train_one_batch(is_preview_iteration)
            if not is_preview_iteration:
                continue
            self.samples.images[side] = batcher.compile_sample(self.batch_size)
            if timelapse_kwargs:
                self.timelapse.get_sample(side, timelapse_kwargs)

        self.model.state.iterations += 1
        self.store_history(loss)
        self.print_loss(loss)

        if viewer is not None:
            viewer(self.samples.show_sample(),
                   "Training - 'S': Save Now. 'ENTER': Save and Quit")

        if timelapse_kwargs is not None:
            self.timelapse.output_timelapse()

    def store_history(self, loss):
        """ Store the history of this step """
        logger.trace("Updating loss history")
        for side in ("a", "b"):
            self.model.history[side].append(loss[side][0])  # Either only loss or total loss
        logger.trace("Updated loss history")


class Batcher():
    """ Batch images from a single side """
    def __init__(self, side, images, model, use_mask, batch_size):
        logger.debug("Initializing %s: side: '%s', num_images: %s, batch_size: %s)",
                     self.__class__.__name__, side, len(images), batch_size)
        self.model = model
        self.use_mask = use_mask
        self.side = side
        self.target = None
        self.mask = None

        self.feed = self.load_generator().minibatch_ab(images, batch_size, self.side)
        self.timelapse_feed = None

    def load_generator(self):
        """ Pass arguments to TrainingDataGenerator and return object """
        logger.debug("Loading generator: %s", self.side)
        input_size = self.model.input_shape[0]
        output_size = self.model.output_shape[0]
        logger.debug("input_size: %s, output_size: %s", input_size, output_size)
        generator = TrainingDataGenerator(input_size, output_size, self.model.training_opts)
        return generator

    def train_one_batch(self, is_preview_iteration):
        """ Train a batch """
        logger.trace("Training one step: (side: %s)", self.side)
        batch = self.get_next()
        self.target = batch[1] if is_preview_iteration else None
        loss = self.model.predictors[self.side].train_on_batch(*batch)
        loss = loss if isinstance(loss, list) else [loss]
        return loss

    def get_next(self):
        """ Return the next batch from the generator
            Items should come out as: (warped, target [, mask]) """
        batch = next(self.feed)
        if self.use_mask:
            batch = self.compile_mask(batch)
        return batch

    def compile_mask(self, batch):
        """ Compile the mask into training data """
        logger.trace("Compiling Mask: (side: '%s')", self.side)
        mask = batch[-1]
        retval = list()
        for idx in range(len(batch) - 1):
            image = batch[idx]
            retval.append([image, mask])
        return retval

    def compile_sample(self, batch_size, images=None):
        """ Training samples to display in the viewer """
        num_images = self.model.training_opts.get("preview_images", 14)
        num_images = min(batch_size, num_images)
        logger.debug("Compiling samples: (side: '%s', samples: %s)", self.side, num_images)
        images = images if images else self.target
        if self.use_mask:
            retval = [tgt[0:num_images] for tgt in images]
        else:
            retval = [images[0:num_images]]
        return retval

    def compile_timelapse_sample(self):
        """ Timelapse samples """
        batch = next(self.timelapse_feed)
        batchsize = len(batch)
        sample = self.compile_sample(batchsize, images=batch[1])
        return sample

    def set_timelapse_feed(self, images, batchsize):
        """ Set the timelapse dictionary """
        logger.debug("Setting timelapse feed: (side: '%s', input_images: '%s', batchsize: %s)",
                     self.side, images, batchsize)
        self.timelapse_feed = self.load_generator().minibatch_ab(images[:batchsize],
                                                                 batchsize, self.side,
                                                                 do_shuffle=False)
        logger.debug("Set timelapse feed")


class Samples():
    """ Display samples for preview and timelapse """
    def __init__(self, model, use_mask):
        logger.debug("Initializing %s: model: '%s', use_mask: %s)",
                     self.__class__.__name__, model, use_mask)
        self.model = model
        self.use_mask = use_mask
        self.images = dict()
        logger.debug("Initialized %s", self.__class__.__name__)

    def show_sample(self):
        """ Display preview data """
        logger.debug("Showing sample")
        feeds = dict()
        figures = dict()
        for side, samples in self.images.items():
            faces = samples[0]
            scale = self.model.input_shape[0] / faces[0].shape[1]
            if scale != 1.0:
                feeds[side] = self.resize_sample(scale, side, faces)
            else:
                feeds[side] = faces
            if self.use_mask:
                feeds[side] = [feeds[side], samples[1]]

        preds = self.get_predictions(feeds["a"], feeds["b"])

        for side, samples in self.images.items():
            other_side = "a" if side == "b" else "b"
            if self.use_mask:
                masked = self.create_masked_preview(samples)
                figures[side] = np.stack([samples[0],
                                          masked,
                                          preds["{}_{}".format(side, side)],
                                          preds["{}_{}".format(other_side, side)], ],
                                         axis=1)
            else:
                figures[side] = np.stack([samples[0],
                                          preds["{}_{}".format(side, side)],
                                          preds["{}_{}".format(other_side, side)], ],
                                         axis=1)

            if self.images[side][0].shape[0] % 2 == 1:
                figures[side] = np.concatenate([figures[side],
                                                np.expand_dims(figures[side][0], 0)])

        figure = np.concatenate([figures["a"], figures["b"]], axis=0)
        width = 4
        height = int(figure.shape[0] / width)
        figure = figure.reshape((width, height) + figure.shape[1:])
        figure = stack_images(figure)

        logger.debug("Compiled sample")
        return np.clip(figure * 255, 0, 255).astype('uint8')

    @staticmethod
    def create_masked_preview(previews):
        """ Add the mask to the faces for masked preview """
        faces, masks = previews
        masked = faces * masks + (1.0 - masks)
        logger.trace("masked.shape: %s", masked.shape)
        return masked

    def resize_sample(self, scale, side, sample):
        """ Resize samples where predictor expects different shape from processed image """
        logger.debug("Resizing sample: (scale: %s, side: '%s', sample: %s",
                     scale, side, sample.shape)
        interpn = cv2.INTER_LINEAR if scale > 1.0 else cv2.INTER_AREA  # pylint: disable=no-member
        resized = [cv2.resize(img,  # pylint: disable=no-member
                              self.model.input_shape[:2],
                              interpn)
                   for img in sample]
        feed = np.array(resized).reshape((-1, ) + self.model.input_shape)
        logger.debug("Resized sample: (side: '%s' shape: %s)", side, feed.shape)
        return feed

    def get_predictions(self, feed_a, feed_b):
        """ Return the sample predictions from the model """
        logger.debug("Getting Predictions")
        preds = dict()
        preds["a_a"] = self.model.predictors["a"].predict(feed_a)
        preds["b_a"] = self.model.predictors["b"].predict(feed_a)
        preds["a_b"] = self.model.predictors["a"].predict(feed_b)
        preds["b_b"] = self.model.predictors["b"].predict(feed_b)

        # Get the returned image from predictors that emit multiple items
        if not isinstance(preds["a_a"], np.ndarray):
            for key, val in preds.items():
                preds[key] = val[0]
        logger.debug("Returning predictions: %s", {key: val.shape for key, val in preds.items()})
        return preds


class Timelapse():
    """ Create the timelapse """
    def __init__(self, model, use_mask, batchers):
        logger.debug("Initializing %s: model: %s, use_mask: %s, batchers: '%s')",
                     self.__class__.__name__, model, use_mask, batchers)
        self.samples = Samples(model, use_mask)
        self.model = model
        self.batchers = batchers
        self.output_file = None
        logger.debug("Initialized %s", self.__class__.__name__)

    def get_sample(self, side, timelapse_kwargs):
        """ Perform timelapse """
        logger.debug("Getting timelapse samples: '%s'", side)
        if not self.output_file:
            self.setup(**timelapse_kwargs)
        self.samples.images[side] = self.batchers[side].compile_timelapse_sample()
        logger.debug("Got timelapse samples: '%s' - %s", side, len(self.samples.images[side]))

    def setup(self, input_a=None, input_b=None, output=None):
        """ Set the timelapse output folder """
        logger.debug("Setting up timelapse")
        if output is None:
            output = get_folder(os.path.join(str(self.model.model_dir), "timelapse"))
        self.output_file = output
        logger.debug("Timelapse output set to '%s'", self.output_file)

        images = {"a": get_image_paths(input_a), "b": get_image_paths(input_b)}
        batchsize = min(len(images["a"]),
                        len(images["b"]),
                        self.model.training_opts.get("preview_images", 14))
        for side, image_files in images.items():
            self.batchers[side].set_timelapse_feed(image_files, batchsize)
        logger.debug("Set up timelapse")

    def output_timelapse(self):
        """ Set the timelapse dictionary """
        logger.debug("Ouputting timelapse")
        image = self.samples.show_sample()
        filename = os.path.join(self.output_file, str(int(time.time())) + ".jpg")

        cv2.imwrite(filename, image)  # pylint: disable=no-member
        logger.debug("Created timelapse: '%s'", filename)


class Landmarks():
    """ Set Landmarks for training into the model's training options"""
    def __init__(self, images, model):
        logger.debug("Initializing %s: (model: '%s')", self.__class__.__name__, model)
        self.images = images
        self.model = model
        logger.debug("Initialized %s", self.__class__.__name__)

    def get_alignments(self):
        """ Obtain the landmarks for each faceset """
        landmarks = dict()
        for side in "a", "b":
            size = cv2.imread(self.images[side][0]).shape[0]  # pylint: disable=no-member
            image_folder = os.path.dirname(self.images[side][0])
            alignments = Alignments(
                image_folder,
                filename="alignments",
                serializer=self.model.training_opts.get("serializer", "json"))
            landmarks[side] = self.transform_landmarks(alignments, size)
        self.model.training_opts["landmarks"] = landmarks

    @staticmethod
    def transform_landmarks(alignments, size):
        """ For each face transform landmarks and return """
        landmarks = dict()
        for _, faces, _, _ in alignments.yield_faces():
            for face in faces:
                detected_face = DetectedFace()
                detected_face.from_alignment(face)
                detected_face.load_aligned(None, size=size, align_eyes=False)
                landmarks[detected_face.hash] = detected_face.aligned_landmarks
        return landmarks
