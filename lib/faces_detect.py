#!/usr/bin python3
""" Face and landmarks detection for faceswap.py """
import logging

import numpy as np

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
        self.feed = dict()
        self.reference = dict()
        logger.trace("Initialized %s", self.__class__.__name__)

    @property
    def extract_ratio(self):
        """ The ratio of padding to add for training images """
        return 0.375

    @property
    def landmarks_as_xy(self):
        """ Landmarks as XY """
        return self.landmarksXY

    def to_bounding_box_dict(self):
        """ Return Bounding Box as a bounding box dixt """
        retval = dict(left=self.x, top=self.y, right=self.x + self.w, bottom=self.y + self.h)
        logger.trace("Returning: %s", retval)
        return retval

    def from_bounding_box_dict(self, bounding_box_dict, image=None):
        """ Set Bounding Box from a bounding box dict """
        logger.trace("Creating from bounding box dict: %s", bounding_box_dict)
        if not isinstance(bounding_box_dict, dict):
            raise ValueError("Supplied Bounding Box is not a dictionary.")
        self.x = bounding_box_dict["left"]
        self.w = bounding_box_dict["right"] - bounding_box_dict["left"]
        self.y = bounding_box_dict["top"]
        self.h = bounding_box_dict["bottom"] - bounding_box_dict["top"]
        if image is not None and image.any():
            self.image_to_face(image)
        logger.trace("Created from bounding box dict: (x: %s, w: %s, y: %s. h: %s)",
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
    def load_aligned(self, image, size=256, align_eyes=False, dtype=None):
        """ No need to load aligned information for all uses of this
            class, so only call this to load the information for easy
            reference to aligned properties for this face """
        logger.trace("Loading aligned face: (size: %s, align_eyes: %s, dtype: %s)",
                     size, align_eyes, dtype)
        padding = int(size * self.extract_ratio) // 2
        self.aligned["size"] = size
        self.aligned["padding"] = padding
        self.aligned["align_eyes"] = align_eyes
        self.aligned["matrix"] = get_align_mat(self, size, align_eyes)
        if image is None:
            self.aligned["face"] = None
        else:
            face = AlignerExtract().transform(
                image,
                self.aligned["matrix"],
                size,
                padding)
            self.aligned["face"] = face if dtype is None else face.astype(dtype)

        logger.trace("Loaded aligned face: %s", {key: val
                                                 for key, val in self.aligned.items()
                                                 if key != "face"})

    def padding_from_coverage(self, size, coverage_ratio):
        """ Return the image padding for a face from coverage_ratio set against a
            pre-padded training image """
        adjusted_ratio = coverage_ratio - (1 - self.extract_ratio)
        padding = round((size * adjusted_ratio) / 2)
        logger.trace(padding)
        return padding

    def load_feed_face(self, image, size=64, coverage_ratio=0.625, dtype=None):
        """ Return a face in the correct dimensions for feeding into a NN

            Coverage ratio should be the ratio of the extracted image that was used for
            training """
        logger.trace("Loading feed face: (size: %s, coverage_ratio: %s, dtype: %s)",
                     size, coverage_ratio, dtype)

        self.feed["size"] = size
        self.feed["padding"] = self.padding_from_coverage(size, coverage_ratio)
        self.feed["matrix"] = get_align_mat(self, size, should_align_eyes=False)

        face = np.clip(AlignerExtract().transform(image,
                                                  self.feed["matrix"],
                                                  size,
                                                  self.feed["padding"])[:, :, :3] / 255.0,
                       0.0, 1.0)
        self.feed["face"] = face if dtype is None else face.astype(dtype)

        logger.trace("Loaded feed face. (face_shape: %s, matrix: %s)",
                     self.feed_face.shape, self.feed_matrix)

    def load_reference_face(self, image, size=64, coverage_ratio=0.625, dtype=None):
        """ Return a face in the correct dimensions for reference to the output from a NN

            Coverage ratio should be the ratio of the extracted image that was used for
            training """
        logger.trace("Loading reference face: (size: %s, coverage_ratio: %s, dtype: %s)",
                     size, coverage_ratio, dtype)

        self.reference["size"] = size
        self.reference["padding"] = self.padding_from_coverage(size, coverage_ratio)
        self.reference["matrix"] = get_align_mat(self, size, should_align_eyes=False)

        face = np.clip(AlignerExtract().transform(image,
                                                  self.reference["matrix"],
                                                  size,
                                                  self.reference["padding"])[:, :, :3] / 255.0,
                       0.0, 1.0)
        self.reference["face"] = face if dtype is None else face.astype(dtype)

        logger.trace("Loaded reference face. (face_shape: %s, matrix: %s)",
                     self.reference_face.shape, self.reference_matrix)

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

    @property
    def feed_face(self):
        """ Return face for feeding into NN """
        return self.feed["face"]

    @property
    def feed_matrix(self):
        """ Return matrix for transforming feed face back to image """
        mat = AlignerExtract().transform_matrix(self.feed["matrix"],
                                                self.feed["size"],
                                                self.feed["padding"])
        logger.trace("Returning: %s", mat)
        return mat

    @property
    def feed_interpolators(self):
        """ Return the interpolators for an input face """
        return get_matrix_scaling(self.feed_matrix)

    @property
    def reference_face(self):
        """ Return source face at size of output from NN for reference """
        return self.reference["face"]

    @property
    def reference_landmarks(self):
        """ Return the landmarks location transposed to reference face """
        landmarks = AlignerExtract().transform_points(self.landmarksXY,
                                                      self.reference["matrix"],
                                                      self.reference["size"],
                                                      self.reference["padding"])
        logger.trace("Returning: %s", landmarks)
        return landmarks

    @property
    def reference_matrix(self):
        """ Return matrix for transforming output face back to image """
        mat = AlignerExtract().transform_matrix(self.reference["matrix"],
                                                self.reference["size"],
                                                self.reference["padding"])
        logger.trace("Returning: %s", mat)
        return mat

    @property
    def reference_interpolators(self):
        """ Return the interpolators for an output face """
        return get_matrix_scaling(self.reference_matrix)
