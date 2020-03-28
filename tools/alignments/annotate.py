#!/usr/bin/env python3
""" Tools for annotating an input image """

from collections import OrderedDict

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class Annotate():
    """ Annotate an input image """

    def __init__(self, image, alignments, original_roi=None):
        logger.debug("Initializing %s: (alignments: %s, original_roi: %s)",
                     self.__class__.__name__, alignments, original_roi)
        self.image = image
        self.alignments = alignments
        self.roi = original_roi
        self.colors = {1: (255, 0, 0),
                       2: (0, 255, 0),
                       3: (0, 0, 255),
                       4: (255, 255, 0),
                       5: (255, 0, 255),
                       6: (0, 255, 255)}
        logger.debug("Initialized %s", self.__class__.__name__)

    def draw_black_image(self):
        """ Change image to black at correct dimensions """
        logger.trace("Drawing black image")
        height, width = self.image.shape[:2]
        self.image = np.zeros((height, width, 3), dtype="uint8")

    def draw_bounding_box(self, color_id=1, thickness=1):
        """ Draw the bounding box around faces """
        color = self.colors[color_id]
        for alignment in self.alignments:
            top_left = (alignment["x"], alignment["y"])
            bottom_right = (alignment["x"] + alignment["w"], alignment["y"] + alignment["h"])
            logger.trace("Drawing bounding box: (top_left: %s, bottom_right: %s, color: %s, "
                         "thickness: %s)", top_left, bottom_right, color, thickness)
            cv2.rectangle(self.image, top_left, bottom_right, color, thickness)

    def draw_extract_box(self, color_id=2, thickness=1):
        """ Draw the extracted face box """
        if not self.roi:
            return
        color = self.colors[color_id]
        for idx, roi in enumerate(self.roi):
            logger.trace("Drawing Extract Box: (idx: %s, roi: %s)", idx, roi)
            top_left = [point for point in roi.squeeze()[0]]
            top_left = (top_left[0], top_left[1] - 10)
            cv2.putText(self.image,
                        str(idx),
                        top_left,
                        cv2.FONT_HERSHEY_DUPLEX,
                        1.0,
                        color,
                        thickness)
            cv2.polylines(self.image, [roi], True, color, thickness)

    def draw_landmarks(self, color_id=3, radius=1):
        """ Draw the facial landmarks """
        color = self.colors[color_id]
        for alignment in self.alignments:
            landmarks = alignment["landmarks_xy"].astype("int32")
            logger.trace("Drawing Landmarks: (landmarks: %s, color: %s, radius: %s)",
                         landmarks, color, radius)
            for (pos_x, pos_y) in landmarks:
                cv2.circle(self.image, (pos_x, pos_y), radius, color, -1)

    def draw_landmarks_mesh(self, color_id=4, thickness=1):
        """ Draw the facial landmarks """
        color = self.colors[color_id]
        facial_landmarks_idxs = OrderedDict([("mouth", (48, 68)),
                                             ("right_eyebrow", (17, 22)),
                                             ("left_eyebrow", (22, 27)),
                                             ("right_eye", (36, 42)),
                                             ("left_eye", (42, 48)),
                                             ("nose", (27, 36)),
                                             ("jaw", (0, 17)),
                                             ("chin", (8, 11))])
        for alignment in self.alignments:
            landmarks = alignment["landmarks_xy"]
            logger.trace("Drawing Landmarks Mesh: (landmarks: %s, color: %s, thickness: %s)",
                         landmarks, color, thickness)
            for key, val in facial_landmarks_idxs.items():
                points = np.array([landmarks[val[0]:val[1]]], np.int32)
                fill_poly = bool(key in ("right_eye", "left_eye", "mouth"))
                cv2.polylines(self.image, points, fill_poly, color, thickness)

    def draw_grey_out_faces(self, live_face):
        """ Grey out all faces except target """
        if not self.roi:
            return
        alpha = 0.6
        overlay = self.image.copy()
        for idx, roi in enumerate(self.roi):
            if idx != int(live_face):
                logger.trace("Greying out face: (idx: %s, roi: %s)", idx, roi)
                cv2.fillPoly(overlay, roi, (0, 0, 0))

        cv2.addWeighted(overlay, alpha, self.image, 1. - alpha, 0., self.image)
