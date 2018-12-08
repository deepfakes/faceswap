#!/usr/bin/env python3
""" DFaker Trainer """
import os
import time

import cv2
import numpy as np

from lib.alignments import Alignments
from lib.faces_detect import DetectedFace


from .original import Trainer as OriginalTrainer, stack_images


class Trainer(OriginalTrainer):
    """ Dfaker Trainer """

    def process_training_opts(self):
        """ Process model specific training options """
        self.load_landmarks()

    def load_landmarks(self):
        """ Load landmarks for images A and B """
        landmarks = dict()
        for side in "a", "b":
            image_folder = os.path.dirname(self.images[side][0])
            alignments = Alignments(
                image_folder,
                filename="alignments",
                serializer=self.model.training_opts.get("serializer", "json"))
            landmarks[side] = self.transform_landmarks(alignments, side)
        self.model.training_opts["landmarks"] = landmarks

    @staticmethod
    def transform_landmarks(alignments, side):
        """ For each face transform landmarks and return """
        landmarks = dict()
        for frame, faces, _, _ in alignments.yield_faces():
            for idx, face in enumerate(faces):
                face_name = "{}_{}".format(frame, idx)

                detected_face = DetectedFace()
                detected_face.from_alignment(face)
                # TODO Load size from face
                detected_face.load_aligned(None,
                                           size=256,
                                           padding=48,
                                           align_eyes=False)
                landmarks[face_name] = detected_face.aligned_landmarks
        return landmarks

    def train_one_step(self, iteration, viewer):
        """ Train a batch """
        epoch, warped_a, target_a, mask_a = next(self.images_a)
        epoch, warped_b, target_b, mask_b = next(self.images_b)

        loss_a = self.model.predictors["a"].train_on_batch(
            [warped_a, mask_a],
            [target_a, mask_a])
        loss_b = self.model.predictors["b"].train_on_batch(
            [warped_b, mask_b],
            [target_b, mask_b])

        self.model._epoch_no += 1

        print("[{0}] [#{1:05d}] loss_A: {2}, "
              "loss_B: {3}".format(time.strftime("%H:%M:%S"),
                                   self.model.epoch_no,
                                   " | ".join(["{:.5f}".format(loss)
                                               for loss in loss_a]),
                                   " | ".join(["{:.5f}".format(loss)
                                               for loss in loss_b])),
              end='\r')

        if viewer is not None:
            viewer(self.show_sample(target_a[0:8, :, :, :3],
                                    target_b[0:8, :, :, :3]),
                   "training")

    def show_sample(self, test_a, test_b):
        # TODO Cleanup and standardise
        """ Display preview data """
        test_a_i = list()
        test_b_i = list()

        for i in test_a:
            test_a_i.append(cv2.resize(i, (64, 64), cv2.INTER_AREA))
        test_a_i = np.array(test_a_i).reshape((-1, 64, 64, 3))

        for i in test_b:
            test_b_i.append(cv2.resize(i, (64, 64), cv2.INTER_AREA))
        test_b_i = np.array(test_b_i).reshape((-1, 64, 64, 3))

        zmask = np.zeros((test_a.shape[0], 128, 128, 1), float)

        pred_a_a, _ = self.model.predictors["a"].predict([test_a_i, zmask])
        pred_b_a, _ = self.model.predictors["b"].predict([test_a_i, zmask])

        pred_a_b, _ = self.model.predictors["a"].predict([test_b_i, zmask])
        pred_b_b, _ = self.model.predictors["b"].predict([test_b_i, zmask])

        pred_a_a = pred_a_a[0:18, :, :, :3]
        pred_a_b = pred_a_b[0:18, :, :, :3]
        pred_b_a = pred_b_a[0:18, :, :, :3]
        pred_b_b = pred_b_b[0:18, :, :, :3]

        figure_a = np.stack([test_a,
                             pred_a_a,
                             pred_b_a, ],
                            axis=1)
        figure_b = np.stack([test_b,
                             pred_b_b,
                             pred_a_b, ],
                            axis=1)

        figure = np.concatenate([figure_a, figure_b], axis=0)
        figure = figure.reshape((4, 4) + figure.shape[1:])
        figure = stack_images(figure)

        return np.clip(figure * 255, 0, 255).astype('uint8')
