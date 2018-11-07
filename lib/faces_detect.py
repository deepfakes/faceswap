#!/usr/bin python3
""" Face and landmarks detection for faceswap.py """

from dlib import rectangle as d_rectangle  # pylint: disable=no-name-in-module
from lib.aligner import Extract as AlignerExtract, get_align_mat


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

        self.aligned = dict()

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

    # <<< Aligned Face methods and properties >>> #
    def load_aligned(self, image, size=256, padding=48, align_eyes=False):
        """ No need to load aligned information for all uses of this
            class, so only call this to load the information for easy
            reference to aligned properties for this face """
        self.aligned["size"] = size
        self.aligned["padding"] = padding
        self.aligned["align_eyes"] = align_eyes
        self.aligned["matrix"] = get_align_mat(self, size, align_eyes)
        self.aligned["face"] = AlignerExtract().transform(
            image,
            self.aligned["matrix"],
            size,
            padding)

    @property
    def original_roi(self):
        """ Return the square aligned box location on the original
            image """
        return AlignerExtract().get_original_roi(self.aligned["matrix"],
                                                 self.aligned["size"],
                                                 self.aligned["padding"])

    @property
    def aligned_landmarks(self):
        """ Return the landmarks location transposed to extracted face """
        return AlignerExtract().transform_points(self.landmarksXY,
                                                 self.aligned["matrix"],
                                                 self.aligned["size"],
                                                 self.aligned["padding"])

    @property
    def aligned_face(self):
        """ Return aligned detected face """
        return self.aligned["face"]

    @property
    def adjusted_matrix(self):
        """ Return adjusted matrix for size/padding combination """
        return AlignerExtract().transform_matrix(self.aligned["matrix"],
                                                 self.aligned["size"],
                                                 self.aligned["padding"])
