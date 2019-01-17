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
        self.use_mask = False
        self.process_training_opts()

        generator = self.load_generator()
        self.images_a = generator.minibatch_ab(images["a"], self.batch_size, "a")
        self.images_b = generator.minibatch_ab(images["b"], self.batch_size, "b")
        self.timelapse = None
        logger.debug("Initialized %s", self.__class__.__name__)

    @property
    def timestamp(self):
        """ Standardised timestamp for loss reporting """
        return time.strftime("%H:%M:%S")

    def load_generator(self):
        """ Pass arguments to TrainingDataGenerator and return object """
        input_size = self.model.input_shape[0]
        output_size = self.model.output_shape[0]
        logger.debug("input_size: %s, output_size: %s", input_size, output_size)
        generator = TrainingDataGenerator(input_size, output_size, self.model.training_opts)
        return generator

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

    def process_training_opts(self):
        """ Override for processing model specific training options """
        logger.debug(self.model.training_opts)
        if self.model.training_opts.get("mask_type", None):
            self.use_mask = True
            Landmarks(self.images, self.model).get_alignments()

    def train_one_step(self, viewer, timelapse_kwargs):
        """ Train a batch

            Items should come out as: (warped, target [, mask])
        """
        logger.trace("Training one step: (iteration: %s)", self.model.iterations)
        loss = dict()
        train = dict()
        for side in ("a", "b"):
            images = getattr(self, "images_{}".format(side))
            train[side] = next(images)
            if self.use_mask:
                train[side] = self.compile_mask(train[side])
            side_loss = self.model.predictors[side].train_on_batch(*train[side])
            side_loss = side_loss if isinstance(side_loss, list) else [side_loss]
            loss[side] = side_loss

        self.model.state.iterations += 1
        self.store_history(loss)
        self.print_loss(loss)

        if viewer is not None:
            target_a, target_b = train["a"][1], train["b"][1]
            if self.use_mask:
                target_a, target_b = target_a[0], target_b[0]
            sample_a, sample_b = self.compile_samples(target_a, target_b, self.batch_size)
            viewer(self.show_sample(sample_a, sample_b),
                   "Training - 'S': Save Now. 'ENTER': Save and Quit")

        if timelapse_kwargs is not None:
            self.do_timelapse(timelapse_kwargs)

    def compile_mask(self, train):
        """ Compile the mask into training data """
        logger.trace("Compiling Mask: (iteration: %s)", self.model.iterations)
        mask = train[-1]
        retval = list()
        for idx in range(len(train) - 1):
            image = train[idx]
            retval.append([image, mask])
        return retval

    def store_history(self, loss):
        """ Store the history of this step """
        logger.trace("Updating loss history")
        for side in ("a", "b"):
            self.model.history[side].append(loss[side][0])  # Either only loss or total loss
        logger.trace("Updated loss history")

    def compile_samples(self, target_a, target_b, batch_size):
        """ Training samples to display in the viewer """
        num_images = self.model.training_opts.get("preview_images", 14)
        num_images = min(batch_size, num_images)
        logger.debug("Compiling samples: %s", num_images)
        return target_a[0:num_images], target_b[0:num_images]

    def do_timelapse(self, timelapse_kwargs):
        """ Perform timelapse """
        logger.debug("Creating timelapse")
        if not self.timelapse:
            self.timelapse = self.set_timelapse(**timelapse_kwargs)
        train_a = next(self.timelapse["images_a"])
        train_b = next(self.timelapse["images_b"])

        sample_a, sample_b = self.compile_samples(train_a[1],
                                                  train_b[1],
                                                  self.timelapse["batch_size"])
        image = self.show_sample(sample_a, sample_b)
        filename = os.path.join(self.timelapse["output_dir"], str(int(time.time())) + ".jpg")

        cv2.imwrite(filename, image)  # pylint: disable=no-member
        logger.debug("Created timelapse: '%s'", filename)

    def set_timelapse(self, input_a=None, input_b=None, output=None):
        """ Set the timelapse dictionary """
        logger.debug("Setting timelapse: (input_a: '%s', input_b: '%s', output: '%s')",
                     input_a, input_b, output)
        timelapse = dict()
        files_a = get_image_paths(input_a)
        files_b = get_image_paths(input_b)
        batch_size = min(len(files_a),
                         len(files_b),
                         self.model.training_opts.get("preview_images", 14))
        generator = self.load_generator()
        if output is None:
            output = get_folder(os.path.join(str(self.model.model_dir), "timelapse"))
        timelapse["output_dir"] = str(output)
        timelapse["images_a"] = generator.minibatch_ab(files_a[:batch_size], batch_size, "a",
                                                       do_shuffle=False)
        timelapse["images_b"] = generator.minibatch_ab(files_b[:batch_size], batch_size, "b",
                                                       do_shuffle=False)
        timelapse["batch_size"] = batch_size

        logger.debug("Set timelapse: %s", timelapse)
        return timelapse

    def show_sample(self, test_a, test_b):
        """ Display preview data """
        logger.debug("Compiling sample")
        scale = self.model.input_shape[0] / test_a.shape[1]
        if scale != 1.0:
            feed_a, feed_b = self.resize_sample(scale, test_a, test_b)
        else:
            feed_a, feed_b = test_a, test_b

        if self.use_mask:
            mask = np.zeros(test_a.shape[:3] + (1, ), float)
            feed_a = [feed_a, mask]
            feed_b = [feed_b, mask]

        preds = self.get_predictions(feed_a, feed_b)

        figure_a = np.stack([test_a, preds["a_a"], preds["b_a"], ], axis=1)
        figure_b = np.stack([test_b, preds["b_b"], preds["a_b"], ], axis=1)

        if test_a.shape[0] % 2 == 1:
            figure_a = np.concatenate([figure_a,
                                       np.expand_dims(figure_a[0], 0)])
            figure_b = np.concatenate([figure_b,
                                       np.expand_dims(figure_b[0], 0)])

        figure = np.concatenate([figure_a, figure_b], axis=0)
        width = 4
        height = int(figure.shape[0] / width)
        figure = figure.reshape((width, height) + figure.shape[1:])
        figure = stack_images(figure)

        logger.debug("Compiled sample")
        return np.clip(figure * 255, 0, 255).astype('uint8')

    def resize_sample(self, scale, test_a, test_b):
        """ Resize samples where predictor expects different shape from processed image """
        logger.debug("Resizing sample: (scale: %s, test_a: %s, test_b: %s",
                     scale, test_a.shape, test_b.shape)
        interpn = cv2.INTER_LINEAR if scale > 1.0 else cv2.INTER_AREA  # pylint: disable=no-member
        resized_a = [cv2.resize(img,  # pylint: disable=no-member
                                self.model.input_shape[:2],
                                interpn)
                     for img in test_a]
        resized_b = [cv2.resize(img,  # pylint: disable=no-member
                                self.model.input_shape[:2],
                                interpn)
                     for img in test_b]

        feed_a = np.array(resized_a).reshape((-1, ) + self.model.input_shape)
        feed_b = np.array(resized_b).reshape((-1, ) + self.model.input_shape)
        logger.debug("Resized sample: (test_a: %s test_b: %s)", feed_a.shape, feed_b.shape)
        return feed_a, feed_b

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
                detected_face.load_aligned(None,
                                           size=size,
                                           padding=48,
                                           align_eyes=False)
                landmarks[detected_face.hash] = detected_face.aligned_landmarks
        return landmarks
