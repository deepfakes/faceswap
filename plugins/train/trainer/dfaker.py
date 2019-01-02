#!/usr/bin/env python3
""" DFaker Trainer """
import os

from lib.alignments import Alignments
from lib.faces_detect import DetectedFace

from ._base import TrainerBase, logger


class Trainer(TrainerBase):
    """ Dfaker Trainer """

    def process_transform_kwargs(self):
        """ Override for specific image manipulation kwargs
            See lib.training_data.ImageManipulation() for valid kwargs"""
        transform_kwargs = {"rotation_range": 10,
                            "zoom_range": 0.05,
                            "shift_range": 0.05,
                            "random_flip": 0.5,
                            "zoom": 128 // self.model.image_shape[0],
                            "coverage": 160,
                            "scale": 5}
        logger.debug(transform_kwargs)
        return transform_kwargs

    def process_training_opts(self):
        """ Load landmarks for images A and B """
        landmarks = dict()
        for side in "a", "b":
            image_folder = os.path.dirname(self.images[side][0])
            alignments = Alignments(
                image_folder,
                filename="alignments",
                serializer=self.model.training_opts.get("serializer", "json"))
            landmarks[side] = self.transform_landmarks(alignments)
        self.model.training_opts["landmarks"] = landmarks

    @staticmethod
    def transform_landmarks(alignments):
        """ For each face transform landmarks and return """
        landmarks = dict()
        for _, faces, _, _ in alignments.yield_faces():
            for face in faces:
                detected_face = DetectedFace()
                detected_face.from_alignment(face)
                # TODO Load size from face
                detected_face.load_aligned(None,
                                           size=256,
                                           padding=48,
                                           align_eyes=False)
                landmarks[detected_face.hash] = detected_face.aligned_landmarks
        return landmarks

    def print_loss(self, loss_a, loss_b):
        """ Override for DFaker Loss """
        print("[{0}] [#{1:05d}] loss_A: {2}, loss_B: {3}".format(self.timestamp,
                                                                 self.model.state.iterations,
                                                                 " | ".join(["{:.5f}".format(loss)
                                                                             for loss in loss_a]),
                                                                 " | ".join(["{:.5f}".format(loss)
                                                                             for loss in loss_b])),
              end='\r')
