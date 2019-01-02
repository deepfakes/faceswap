#!/usr/bin/env python3


# TODO Remove 'remove_alpha'. Hack in there until masks properly implemented
# TODO Remove 'use_alignments'? If using mask then alignments must be provided
""" Base Trainer Class for Faceswap

    Trainers should be inherited from this class.

    A training_opts dictionary can be set in the corresponding model.
    Accepted values:
        use_alignments: Set to true if the model uses an alignments file (default: False)
        serializer:     format that alignments data is serialized in
        use_mask:       Set to true if the model uses a mask (default: False)
        remove_alpha:   Remove the alpha channel from the image before passing to predictor
        preview_images: Number of preview images to display (default: 14)
"""

import logging
import os
import time

import cv2
import numpy as np

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
        self.process_training_opts()
        self.transform_kwargs = self.process_transform_kwargs()

        generator = TrainingDataGenerator(
            transform_kwargs=self.transform_kwargs,
            training_opts=self.model.training_opts)

        self.images_a = generator.minibatch_ab(images["a"], self.batch_size, "a")
        self.images_b = generator.minibatch_ab(images["b"], self.batch_size, "b")
        self.timelapse = None
        logger.debug("Initialized %s", self.__class__.__name__)

    @property
    def timestamp(self):
        """ Standardised timestamp for loss reporting """
        return time.strftime("%H:%M:%S")

    def process_training_opts(self):
        """ Override for processing model specific training options """
        raise NotImplementedError

    def process_transform_kwargs(self):
        """ Override for specific image manipulation kwargs
            See lib.training_data.ImageManipulation() for valid kwargs"""
        transform_kwargs = {"rotation_range": 10,
                            "zoom_range": 0.05,
                            "shift_range": 0.05,
                            "random_flip": 0.4,
                            "zoom": self.model.image_shape[0] // 64,
                            "coverage": 160,
                            "scale": 5}
        logger.debug(transform_kwargs)
        return transform_kwargs

    def print_loss(self, loss_a, loss_b):
        """ Override for specific model loss formatting """
        loss = [loss_a, loss_b]
        for idx, side in enumerate(loss):
            if isinstance(side, list):
                loss[idx] = " | ".join(["{:.5f}".format(loss) for loss in loss_a])
            else:
                loss[idx] = "{:.5f}".format(loss)
        print("[{0}] [#{1:05d}] loss_A: {2}, loss_B: {3}".format(self.timestamp,
                                                                 self.model.iterations,
                                                                 loss[0],
                                                                 loss[1]),
              end='\r')

    def train_one_step(self, viewer, timelapse_kwargs):
        """ Train a batch

            Items should come out as: (warped, target [, mask])
        """
        logger.trace("Training one step: (iteration: %s)", self.model.iterations)
        use_mask = self.model.training_opts.get("use_mask", False)
        train_a = next(self.images_a)
        train_b = next(self.images_b)
        if use_mask:
            train_a, train_b = self.compile_mask(train_a, train_b)

        loss_a = self.model.predictors["a"].train_on_batch(*train_a)
        loss_b = self.model.predictors["b"].train_on_batch(*train_b)

        self.model.state.iterations += 1
        self.print_loss(loss_a, loss_b)

        if viewer is not None:
            target_a, target_b = train_a[1], train_b[1]
            if use_mask:
                target_a, target_b = target_a[0], target_b[0]
            sample_a, sample_b = self.compile_samples(target_a, target_b, self.batch_size)
            viewer(self.show_sample(sample_a, sample_b),
                   "Training - 'S': Save Now. 'ENTER': Save and Quit")

        if timelapse_kwargs is not None:
            self.do_timelapse(timelapse_kwargs)

    def compile_mask(self, train_a, train_b):
        """ Compile the mask into training data """
        logger.trace("Compiling Mask: (iteration: %s)", self.model.iterations)
        sides = list()
        for train in (train_a, train_b):
            mask = train[-1]
            if self.model.training_opts.get("remove_alpha", False):
                mask = np.expand_dims(mask, -1)
            side = list()
            for idx in range(len(train) - 1):
                image = train[idx]
                if self.model.training_opts.get("remove_alpha", False):
                    image = image[..., 0:3]
                side.append([image, mask])
            sides.append(side)
        return sides[0], sides[1]

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
        generator = TrainingDataGenerator(
            transform_kwargs={"rotation_range": 0, "zoom_range": 0, "shift_range": 0,
                              "random_flip": 0, "zoom": self.transform_kwargs["zoom"],
                              "coverage": self.transform_kwargs["coverage"],
                              "scale": self.transform_kwargs["scale"]},
            training_opts=self.model.training_opts)

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
        scale = self.model.image_shape[0] / test_a.shape[1]
        if scale != 1.0:
            feed_a, feed_b = self.resize_sample(scale, test_a, test_b)
        else:
            feed_a, feed_b = test_a, test_b

        if self.model.training_opts.get("use_mask", False):
            mask = np.zeros(test_a.shape[:3] + (1, ), float)
            if self.model.training_opts.get("remove_alpha", False):
                test_a = test_a[..., 0:3]
                test_b = test_b[..., 0:3]
                feed_a = feed_a[..., 0:3]
                feed_b = feed_b[..., 0:3]
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
                                self.model.image_shape[:2],
                                interpn)
                     for img in test_a]
        resized_b = [cv2.resize(img,  # pylint: disable=no-member
                                self.model.image_shape[:2],
                                interpn)
                     for img in test_b]

        feed_a = np.array(resized_a).reshape((-1, ) + self.model.image_shape)
        feed_b = np.array(resized_b).reshape((-1, ) + self.model.image_shape)
        logger.debug("Resized sample: (test_a: %s test_b: %s)", feed_a.shape, feed_b.shape)
        return feed_a, feed_b

    def get_predictions(self, feed_a, feed_b):
        """ Return the sample predictions from the model """
        logger.debug("Getting Predictions")
        preds = dict()
        preds["a_a"] = self.model.predictors["a"].predict(feed_a)
        preds["b_a"] = self.model.predictors["b"].predict(feed_a)
        preds["a_b"] = self.model.predictors["a"].predict(feed_b)
        preds["b_b"] = self.model.predictors["b"].predict(feed_a)

        # Get the returned image from predictors that emit multiple items
        if not isinstance(preds["a_a"], np.ndarray):
            for key, val in preds.items():
                preds[key] = val[0]
        logger.debug("Returning predictions: %s", {key: val.shape for key, val in preds.items()})
        return preds
