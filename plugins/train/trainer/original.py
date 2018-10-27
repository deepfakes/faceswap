#!/usr/bin/env python3
""" Original Trainer """

import time
import numpy as np
from lib.training_data import TrainingDataGenerator, stack_images


class Trainer():
    random_transform_args = {
        'rotation_range': 10,
        'zoom_range': 0.05,
        'shift_range': 0.05,
        'random_flip': 0.4,
    }

    def __init__(self, model, fn_a, fn_b, batch_size, *args):
        self.batch_size = batch_size
        self.model = model

        generator = TrainingDataGenerator(self.random_transform_args, 160,
                                          zoom=self.model.image_shape[0]//64)
        self.images_a = generator.minibatchAB(fn_a, self.batch_size)
        self.images_b = generator.minibatchAB(fn_b, self.batch_size)

    def train_one_step(self, iter, viewer):
        epoch, warped_a, target_a = next(self.images_a)
        epoch, warped_b, target_b = next(self.images_b)

        loss_a = self.model.autoencoder_a.train_on_batch(warped_a, target_a)
        loss_b = self.model.autoencoder_b.train_on_batch(warped_b, target_b)

        self.model._epoch_no += 1

        print("[{0}] [#{1:05d}] loss_A: {2:.5f}, "
              "loss_B: {3:.5f}".format(time.strftime("%H:%M:%S"),
                                       self.model.epoch_no,
                                       loss_a,
                                       loss_b),
              end='\r')

        if viewer is not None:
            viewer(self.show_sample(target_a[0:14], target_b[0:14]),
                   "training")

    def show_sample(self, test_a, test_b):
        figure_a = np.stack([test_a,
                             self.model.autoencoder_a.predict(test_a),
                             self.model.autoencoder_b.predict(test_a), ],
                            axis=1)
        figure_b = np.stack([test_b,
                             self.model.autoencoder_b.predict(test_b),
                             self.model.autoencoder_a.predict(test_b), ],
                            axis=1)

        if test_a.shape[0] % 2 == 1:
            figure_a = np.concatenate([figure_a,
                                       np.expand_dims(figure_a[0], 0)])
            figure_b = np.concatenate([figure_b,
                                       np.expand_dims(figure_b[0], 0)])

        figure = np.concatenate([figure_a, figure_b], axis=0)
        w = 4
        h = int(figure.shape[0] / w)
        figure = figure.reshape((w, h) + figure.shape[1:])
        figure = stack_images(figure)

        return np.clip(figure * 255, 0, 255).astype('uint8')
