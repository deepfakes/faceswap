#!/usr/bin/env python3
""" Tools for annotating an input image """
# TODO Handle landmark rotation

from cv2 import (rectangle, circle, polylines, putText,
                 FONT_HERSHEY_DUPLEX, fillPoly, addWeighted)
from numpy import array, int32, uint8, zeros

from lib.align_eyes import FACIAL_LANDMARKS_IDXS


class Annotate():
    """ Annotate an input image """

    def __init__(self, image, alignments, original_roi=None):
        self.image = image
        self.alignments = alignments
        self.roi = original_roi
        self.colors = {1: (255, 0, 0),
                       2: (0, 255, 0),
                       3: (0, 0, 255),
                       4: (255, 255, 0),
                       5: (255, 0, 255),
                       6: (0, 255, 255)}

    def draw_black_image(self):
        """ Change image to black at correct dimensions """
        height, width = self.image.shape[:2]
        self.image = zeros((height, width, 3), uint8)

    def draw_bounding_box(self, color_id=1, thickness=1):
        """ Draw the bounding box around faces """
        color = self.colors[color_id]
        for alignment in self.alignments:
            top_left = (alignment["x"], alignment["y"])
            bottom_right = (alignment["x"] + alignment["w"],
                            alignment["y"] + alignment["h"])
            rectangle(self.image, top_left, bottom_right,
                      color, thickness)

    def draw_extract_box(self, color_id=2, thickness=1):
        """ Draw the extracted face box """
        if not self.roi:
            return
        color = self.colors[color_id]
        for idx, roi in enumerate(self.roi):
            top_left = [point for point in roi[0].squeeze()[0]]
            top_left = (top_left[0], top_left[1] - 10)
            putText(self.image, str(idx), top_left, FONT_HERSHEY_DUPLEX, 1.0,
                    color, thickness)
            polylines(self.image, roi, True, color, thickness)

    def draw_landmarks(self, color_id=3, radius=1):
        """ Draw the facial landmarks """
        color = self.colors[color_id]
        for alignment in self.alignments:
            landmarks = alignment["landmarksXY"]
            for (pos_x, pos_y) in landmarks:
                circle(self.image, (pos_x, pos_y), radius, color, -1)

    def draw_landmarks_mesh(self, color_id=4, thickness=1):
        """ Draw the facial landmarks """
        color = self.colors[color_id]
        for alignment in self.alignments:
            landmarks = alignment["landmarksXY"]
            for key, val in FACIAL_LANDMARKS_IDXS.items():
                points = array([landmarks[val[0]:val[1]]], int32)
                fill_poly = bool(key in ("right_eye", "left_eye", "mouth"))
                polylines(self.image, points, fill_poly, color, thickness)

    def draw_grey_out_faces(self, live_face):
        """ Grey out all faces except target """
        if not self.roi:
            return
        alpha = 0.6
        overlay = self.image.copy()
        for idx, roi in enumerate(self.roi):
            if idx != int(live_face):
                fillPoly(overlay, roi, (0, 0, 0))
        addWeighted(overlay, alpha, self.image, 1 - alpha, 0, self.image)
