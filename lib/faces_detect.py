#!/usr/bin python3
""" Face and landmarks detection for faceswap.py """

from dlib import rectangle as d_rectangle


class DetectedFace():
    """ Detected face and landmark information """
    def __init__(self, image=None, x=None, w=None, y=None, h=None,
                 landmarksXY=None):
        self.image = image
        self.x = x
        self.w = w
        self.y = y
        self.h = h
        self.landmarksXY = landmarksXY

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
        """ Crop an image around bounding box to the face """
        self.image = image[self.y: self.y + self.h,
                           self.x: self.x + self.w]

    def to_alignment(self):
        """ Convert a detected face to alignment dict """
        alignment = dict()
        alignment["x"] = self.x
        alignment["w"] = self.w
        alignment["y"] = self.y
        alignment["h"] = self.h
        alignment["landmarksXY"] = self.landmarksXY
        return alignment

    def from_alignment(self, alignment, image=None):
        """ Convert a face alignment to detected face object """
        self.x = alignment["x"]
        self.w = alignment["w"]
        self.y = alignment["y"]
        self.h = alignment["h"]
        self.landmarksXY = alignment["landmarksXY"]
        if image.any():
            self.image_to_face(image)
