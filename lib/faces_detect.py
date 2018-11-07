#!/usr/bin python3
""" Face and landmarks detection for faceswap.py """

import cv2
import numpy as np

from dlib import rectangle as d_rectangle  # pylint: disable=no-name-in-module
from lib.aligner import get_align_mat


class DetectedFace():
    """ Detected face and landmark information """
    def __init__(self, image=None, x=None, w=None, y=None, h=None,
                 frame_dims=None, landmarksXY=None):
        self.image = image
        self.x = x
        self.w = w
        self.y = y
        self.h = h
        self.frame_dims = frame_dims
        self.landmarksXY = landmarksXY
        self.matrix = None

    def landmarks_as_xy(self):
        """ Landmarks as XY """
        return self.landmarksXY

    def to_dlib_rect(self):
        """ Return Bounding Box as Dlib Rectangle """
        left = self.x
        top = self.y
        right = self.x + self.w
        bottom = self.y + self.h
        return d_rectangle(left, top, right, bottom)

    def from_dlib_rect(self, d_rect):
        """ Set Bounding Box from a Dlib Rectangle """
        if not isinstance(d_rect, d_rectangle):
            raise ValueError("Supplied Bounding Box is not a dlib.rectangle.")
        self.x = d_rect.left()
        self.w = d_rect.right() - d_rect.left()
        self.y = d_rect.top()
        self.h = d_rect.bottom() - d_rect.top()

    def image_to_face(self, image):
        """ Crop an image around bounding box to the face
            and capture it's dimensions """
        self.image = image[self.y: self.y + self.h,
                           self.x: self.x + self.w]

    def to_alignment(self):
        """ Convert a detected face to alignment dict """
        alignment = dict()
        alignment["x"] = self.x
        alignment["w"] = self.w
        alignment["y"] = self.y
        alignment["h"] = self.h
        alignment["frame_dims"] = self.frame_dims
        alignment["landmarksXY"] = self.landmarksXY
        return alignment

    def from_alignment(self, alignment, image=None):
        """ Convert a face alignment to detected face object """
        self.x = alignment["x"]
        self.w = alignment["w"]
        self.y = alignment["y"]
        self.h = alignment["h"]
        self.frame_dims = alignment["frame_dims"]
        self.landmarksXY = alignment["landmarksXY"]
        if image.any():
            self.image_to_face(image)

    def set_alignment_matrix(self, size):
        """ Set the alignment matrix for this face """
        if self.matrix is not None:
            return
        self.matrix = get_align_mat(self, size, False)

    def original_roi(self, size=256, padding=48):
        """ Return the square aligned box location on the original
            image """
        self.set_alignment_matrix(size)
        points = np.array([[0, 0],
                           [0, size - 1],
                           [size - 1, size - 1],
                           [size - 1, 0]], np.int32)
        points = points.reshape((-1, 1, 2))

        mat = self.matrix * (size - 2 * padding)
        mat[:, 2] += padding
        mat = cv2.invertAffineTransform(mat)  # pylint: disable=no-member
        return [cv2.transform(points, mat)]  # pylint: disable=no-member

    def aligned_landmarks(self, size=256, padding=48):
        """ Return the landmarks location transposed to extracted face """
        self.set_alignment_matrix(size)
        mat = self.matrix * (size - 2 * padding)
        mat[:, 2] += padding
        points = np.expand_dims(self.landmarksXY, axis=1)
        points = cv2.transform(points,    # pylint: disable=no-member
                               mat,
                               points.shape)
        points = np.squeeze(points)
        return points
