#!/usr/bin/env python3
""" Base class for Face Aligner plugins
    Plugins should inherit from this class

    See the override methods for which methods are
    required.

    The plugin will receive a dict containing:
    {"filename": <filename of source frame>,
     "image": <source image>,
     "detected_faces": <list of DetectedFaces objects without landmarks>}

    For each source item, the plugin must pass a dict to finalize containing:
    {"filename": <filename of source frame>,
     "image": <source image>,
     "detected_faces": <list of final DetectedFaces objects>}
    """

import os
import cv2
import numpy as np

from lib.aligner import get_align_mat
from lib.align_eyes import FACIAL_LANDMARKS_IDXS

from lib.gpu_stats import GPUStats


class Aligner():
    """ Landmarks Aligner Object """
    def __init__(self, verbose=False, align_eyes=False, size=256):
        self.verbose = verbose
        self.size = size
        self.cachepath = os.path.join(os.path.dirname(__file__), ".cache")
        self.align_eyes = align_eyes
        self.extract = Extract()
        self.init = None

        # The input and output queues for the plugin.
        # See lib.multithreading.QueueManager for getting queues
        self.queues = {"in": None, "out": None}

        #  Path to model if required
        self.model_path = self.set_model_path()

        # Approximate VRAM required for aligner. Used to calculate
        # how many parallel processes / batches can be run.
        # Be conservative to avoid OOM.
        self.vram = None

    # <<< OVERRIDE METHODS >>> #
    # These methods must be overriden when creating a plugin
    @staticmethod
    def set_model_path():
        """ path to data file/models
            override for specific detector """
        raise NotImplementedError()

    def initialize(self, *args, **kwargs):
        """ Inititalize the aligner
            Tasks to be run before any alignments are performed.
            Override for specific detector """
        self.init = kwargs["event"]
        self.queues["in"] = kwargs["in_queue"]
        self.queues["out"] = kwargs["out_queue"]

    def align(self, *args, **kwargs):
        """ Process landmarks
            Override for specific detector
            Must return a list of dlib rects"""
        try:
            if not self.init:
                self.initialize(*args, **kwargs)
        except ValueError as err:
            print("ERROR: {}".format(err))
            exit(1)

    # <<< FINALIZE METHODS>>> #
    def finalize(self, output):
        """ This should be called as the final task of each plugin
            aligns faces and puts to the out queue """
        if output == "EOF":
            self.queues["out"].put("EOF")
            return
        self.align_faces(output)
        self.queues["out"].put((output))

    def align_faces(self, output):
        """ Align the faces """
        detected_faces = output["detected_faces"]
        image = output["image"]

        resized_faces = list()
        t_mats = list()

        for face in detected_faces:
            resized_face, t_mat = self.extract.extract(image,
                                                       face,
                                                       self.size,
                                                       self.align_eyes)
            resized_faces.append(resized_face)
            t_mats.append(t_mat)

        output["resized_faces"] = resized_faces
        output["t_mats"] = t_mats

    # <<< MISC METHODS >>> #
    def get_vram_free(self):
        """ Return free and total VRAM on card with most VRAM free"""
        stats = GPUStats()
        vram = stats.get_card_most_free()
        if self.verbose:
            print("Using device {} with {}MB free of {}MB".format(
                vram["device"],
                int(vram["free"]),
                int(vram["total"])))
        return int(vram["free"]), int(vram["total"])


class Extract():
    """ Based on the original https://www.reddit.com/r/deepfakes/
        code sample + contribs """

    def extract(self, image, face, size, align_eyes):
        """ Extract a face from an image """
        alignment = get_align_mat(face, size, align_eyes)
        extracted = self.transform(image, alignment, size, 48)
        return extracted, alignment

    @staticmethod
    def transform(image, mat, size, padding=0):
        """ Transform Image """
        matrix = mat * (size - 2 * padding)
        matrix[:, 2] += padding
        return cv2.warpAffine(image, matrix, (size, size))

    @staticmethod
    def transform_points(points, mat, size, padding=0):
        """ Transform points along matrix """
        matrix = mat * (size - 2 * padding)
        matrix[:, 2] += padding
        points = np.expand_dims(points, axis=1)
        points = cv2.transform(points, matrix, points.shape)
        points = np.squeeze(points)
        return points

    @staticmethod
    def get_feature_mask(aligned_landmarks_68, size,
                         padding=0, dilation=30):
        """ Return the face feature mask """
        scale = size - 2*padding
        translation = padding
        pad_mat = np.matrix([[scale, 0.0, translation],
                             [0.0, scale, translation]])
        aligned_landmarks_68 = np.expand_dims(aligned_landmarks_68, axis=1)
        aligned_landmarks_68 = cv2.transform(aligned_landmarks_68,
                                             pad_mat,
                                             aligned_landmarks_68.shape)
        aligned_landmarks_68 = np.squeeze(aligned_landmarks_68)

        (l_start, l_end) = FACIAL_LANDMARKS_IDXS["left_eye"]
        (r_start, r_end) = FACIAL_LANDMARKS_IDXS["right_eye"]
        (m_start, m_end) = FACIAL_LANDMARKS_IDXS["mouth"]
        (n_start, n_end) = FACIAL_LANDMARKS_IDXS["nose"]
        (lb_start, lb_end) = FACIAL_LANDMARKS_IDXS["left_eyebrow"]
        (rb_start, rb_end) = FACIAL_LANDMARKS_IDXS["right_eyebrow"]
        (c_start, c_end) = FACIAL_LANDMARKS_IDXS["chin"]

        l_eye_points = aligned_landmarks_68[l_start:l_end].tolist()
        l_brow_points = aligned_landmarks_68[lb_start:lb_end].tolist()
        r_eye_points = aligned_landmarks_68[r_start:r_end].tolist()
        r_brow_points = aligned_landmarks_68[rb_start:rb_end].tolist()
        nose_points = aligned_landmarks_68[n_start:n_end].tolist()
        chin_points = aligned_landmarks_68[c_start:c_end].tolist()
        mouth_points = aligned_landmarks_68[m_start:m_end].tolist()
        l_eye_points = l_eye_points + l_brow_points
        r_eye_points = r_eye_points + r_brow_points
        mouth_points = mouth_points + nose_points + chin_points

        l_eye_hull = cv2.convexHull(np.array(l_eye_points).reshape(
            (-1, 2)).astype(int)).flatten().reshape((-1, 2))
        r_eye_hull = cv2.convexHull(np.array(r_eye_points).reshape(
            (-1, 2)).astype(int)).flatten().reshape((-1, 2))
        mouth_hull = cv2.convexHull(np.array(mouth_points).reshape(
            (-1, 2)).astype(int)).flatten().reshape((-1, 2))

        mask = np.zeros((size, size, 3), dtype=float)
        cv2.fillConvexPoly(mask, l_eye_hull, (1, 1, 1))
        cv2.fillConvexPoly(mask, r_eye_hull, (1, 1, 1))
        cv2.fillConvexPoly(mask, mouth_hull, (1, 1, 1))

        if dilation > 0:
            kernel = np.ones((dilation, dilation), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1)

        return mask
