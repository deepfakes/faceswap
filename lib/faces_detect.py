#!/usr/bin python3
""" Face and landmarks detection for faceswap.py """
import logging

from dlib import rectangle as d_rectangle  # pylint: disable=no-name-in-module
from lib.aligner import Extract as AlignerExtract, get_align_mat, get_matrix_scaling

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class DetectedFace():
    """ Detected face and landmark information """
    def __init__(  # pylint: disable=invalid-name
            self, image=None, x=None, w=None, y=None, h=None,
            landmarksXY=None):
        logger.trace("Initializing %s", self.__class__.__name__)
        self.image = image
        self.x = x
        self.w = w
        self.y = y
        self.h = h
        self.landmarksXY = landmarksXY
        self.hash = None  # Hash must be set when the file is saved due to image compression

        self.aligned = dict()
        logger.trace("Initialized %s", self.__class__.__name__)

    @property
    def landmarks_as_xy(self):
        """ Landmarks as XY """
        return self.landmarksXY

    def to_dlib_rect(self):
        """ Return Bounding Box as Dlib Rectangle """
        left = self.x
        top = self.y
        right = self.x + self.w
        bottom = self.y + self.h
        retval = d_rectangle(left, top, right, bottom)
        logger.trace("Returning: %s", retval)
        return retval

    def from_dlib_rect(self, d_rect, image=None):
        """ Set Bounding Box from a Dlib Rectangle """
        logger.trace("Creating from dlib_rectangle: %s", d_rect)
        if not isinstance(d_rect, d_rectangle):
            raise ValueError("Supplied Bounding Box is not a dlib.rectangle.")
        self.x = d_rect.left()
        self.w = d_rect.right() - d_rect.left()
        self.y = d_rect.top()
        self.h = d_rect.bottom() - d_rect.top()
        if image is not None and image.any():
            self.image_to_face(image)
        logger.trace("Created from dlib_rectangle: (x: %s, w: %s, y: %s. h: %s)",
                     self.x, self.w, self.y, self.h)

    def image_to_face(self, image):
        """ Crop an image around bounding box to the face
            and capture it's dimensions """
        logger.trace("Cropping face from image")
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
        alignment["hash"] = self.hash
        logger.trace("Returning: %s", alignment)
        return alignment

    def from_alignment(self, alignment, image=None):
        """ Convert a face alignment to detected face object """
        logger.trace("Creating from alignment: (alignment: %s, has_image: %s)",
                     alignment, bool(image is not None))
        self.x = alignment["x"]
        self.w = alignment["w"]
        self.y = alignment["y"]
        self.h = alignment["h"]
        self.landmarksXY = alignment["landmarksXY"]
        # Manual tool does not know the final hash so default to None
        self.hash = alignment.get("hash", None)
        if image is not None and image.any():
            self.image_to_face(image)
        logger.trace("Created from alignment: (x: %s, w: %s, y: %s. h: %s, "
                     "landmarks: %s)",
                     self.x, self.w, self.y, self.h, self.landmarksXY)

    # <<< Aligned Face methods and properties >>> #
    def load_aligned(self, image, size=256, align_eyes=False):
        """ No need to load aligned information for all uses of this
            class, so only call this to load the information for easy
            reference to aligned properties for this face """
        logger.trace("Loading aligned face: (size: %s, align_eyes: %s)", size, align_eyes)
        padding = int(size * 0.1875)
        self.aligned["size"] = size
        self.aligned["padding"] = padding
        self.aligned["align_eyes"] = align_eyes
        self.aligned["matrix"] = get_align_mat(self, size, align_eyes)
        if image is None:
            self.aligned["face"] = None
        else:
            self.aligned["face"] = AlignerExtract().transform(
                image,
                self.aligned["matrix"],
                size,
                padding)
        logger.trace("Loaded aligned face: %s", {key: val
                                                 for key, val in self.aligned.items()
                                                 if key != "face"})

    @property
    def original_roi(self):
        """ Return the square aligned box location on the original
            image """
        roi = AlignerExtract().get_original_roi(self.aligned["matrix"],
                                                self.aligned["size"],
                                                self.aligned["padding"])
        logger.trace("Returning: %s", roi)
        return roi

    @property
    def aligned_landmarks(self):
        """ Return the landmarks location transposed to extracted face """
        landmarks = AlignerExtract().transform_points(self.landmarksXY,
                                                      self.aligned["matrix"],
                                                      self.aligned["size"],
                                                      self.aligned["padding"])
        logger.trace("Returning: %s", landmarks)
        return landmarks

    @property
    def aligned_face(self):
        """ Return aligned detected face """
        return self.aligned["face"]

    @property
    def adjusted_matrix(self):
        """ Return adjusted matrix for size/padding combination """
        mat = AlignerExtract().transform_matrix(self.aligned["matrix"],
                                                self.aligned["size"],
                                                self.aligned["padding"])
        logger.trace("Returning: %s", mat)
        return mat

    @property
    def adjusted_interpolators(self):
        """ Return the interpolator and reverse interpolator for the adjusted matrix """
        return get_matrix_scaling(self.adjusted_matrix)
