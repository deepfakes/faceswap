#!/usr/bin/env python3
""" DFaker Trainer """
import os

from lib.alignments import Alignments
from lib.faces_detect import DetectedFace

from ._base import TrainerBase, logger


class Trainer(TrainerBase):
    """ Dfaker Trainer """

    def process_training_opts(self):
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

    def print_loss(self, loss_a, loss_b):
        """ Override for DFaker Loss """
        print("[{0}] [#{1:05d}] loss_A: {2}, loss_B: {3}".format(self.timestamp,
                                                                 self.model.iterations,
                                                                 " | ".join(["{:.5f}".format(loss)
                                                                             for loss in loss_a]),
                                                                 " | ".join(["{:.5f}".format(loss)
                                                                             for loss in loss_b])),
              end='\r')
