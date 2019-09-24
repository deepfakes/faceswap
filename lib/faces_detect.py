#!/usr/bin python3
""" Face and landmarks detection for faceswap.py """
import logging

import numpy as np

from lib.aligner import Extract as AlignerExtract, get_align_mat, get_matrix_scaling

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class DetectedFace():
    """ Detected face and landmark information

    Holds information about a detected face, it's location in a source image
    and the face's 68 point landmarks.

    Methods for aligning a face are also callable from here.

    Parameters
    ----------
    image: np.ndarray, optional
        This is a generic image placeholder that should not be relied on to be holding a particular
        image. It may hold the source frame that holds the face, a cropped face or a scaled image
        depending on the method using this object.
    x: int
        The left most point (in pixels) of the face's bounding box as discovered in
        :mod:`plugins.extract.detect`
    w: int
        The width (in pixels) of the face's bounding box as discovered in
        :mod:`plugins.extract.detect`
    y: int
        The top most point (in pixels) of the face's bounding box as discovered in
        :mod:`plugins.extract.detect`
    h: int
        The height (in pixels) of the face's bounding box as discovered in
        :mod:`plugins.extract.detect`
    landmarks_xy: list
        The 68 point landmarks as discovered in :mod:`plugins.extract.align`. Should be a ``list``
        of 68 `(x, y)` ``tuples`` with each of the landmark co-ordinates.
    """
    def __init__(self, image=None, x=None, w=None, y=None, h=None, landmarks_xy=None):
        logger.trace("Initializing %s: (image: %s, x: %s, w: %s, y: %s, h:%s, landmarks_xy: %s)",
                     self.__class__.__name__,
                     image.shape if image is not None and image.any() else image,
                     x, w, y, h, landmarks_xy)
        self.image = image
        self.x = x
        self.w = w
        self.y = y
        self.h = h
        self.landmarks_xy = landmarks_xy
        self.hash = None
        """ str: The hash of the face. This cannot be set until the file is saved due to image
        compression, but will be set if loading data from :func:`from_alignment` """

        self.aligned = dict()
        self.feed = dict()
        self.reference = dict()
        logger.trace("Initialized %s", self.__class__.__name__)

    @property
    def left(self):
        """int: Left point (in pixels) of face detection bounding box within the parent image """
        return self.x

    @property
    def top(self):
        """int: Top point (in pixels) of face detection bounding box within the parent image """
        return self.y

    @property
    def right(self):
        """int: Right point (in pixels) of face detection bounding box within the parent image """
        return self.x + self.w

    @property
    def bottom(self):
        """int: Bottom point (in pixels) of face detection bounding box within the parent image """
        return self.y + self.h

    @property
    def _extract_ratio(self):
        """ float: The ratio of padding to add for training images """
        return 0.375

    def to_alignment(self):
        """  Return the detected face formatted for an alignments file

        returns
        -------
        alignment: dict
            The alignment dict will be returned with the keys ``x``, ``w``, ``y``, ``h``,
            ``landmarks_xy``, ``hash``.
        """

        alignment = dict()
        alignment["x"] = self.x
        alignment["w"] = self.w
        alignment["y"] = self.y
        alignment["h"] = self.h
        alignment["landmarks_xy"] = self.landmarks_xy
        alignment["hash"] = self.hash
        logger.trace("Returning: %s", alignment)
        return alignment

    def from_alignment(self, alignment, image=None):
        """ Set the attributes of this class from an alignments file and optionally load the face
        into the ``image`` attribute.

        Parameters
        ----------
        alignment: dict
            A dictionary entry for a face from an alignments file containing the keys
            ``x``, ``w``, ``y``, ``h``, ``landmarks_xy``. Optionally the key ``hash``
            will be provided, but not all use cases will know the face hash at this time.
        image: numpy.ndarray, optional
            If an image is passed in, then the ``image`` attribute will
            be set to the cropped face based on the passed in bounding box co-ordinates
        """

        logger.trace("Creating from alignment: (alignment: %s, has_image: %s)",
                     alignment, bool(image is not None))
        self.x = alignment["x"]
        self.w = alignment["w"]
        self.y = alignment["y"]
        self.h = alignment["h"]
        self.landmarks_xy = alignment["landmarks_xy"]
        # Manual tool does not know the final hash so default to None
        self.hash = alignment.get("hash", None)
        if image is not None and image.any():
            self._image_to_face(image)
        logger.trace("Created from alignment: (x: %s, w: %s, y: %s. h: %s, "
                     "landmarks: %s)",
                     self.x, self.w, self.y, self.h, self.landmarks_xy)

    def _image_to_face(self, image):
        """ set self.image to be the cropped face from detected bounding box """
        logger.trace("Cropping face from image")
        self.image = image[self.top: self.bottom,
                           self.left: self.right]

    # <<< Aligned Face methods and properties >>> #
    def load_aligned(self, image, size=256, dtype=None):
        """ Align a face from a given image.

        Aligning a face is a relatively expensive task and is not required for all uses of
        the :class:`~lib.faces_detect.DetectedFace` object, so call this function explicitly to
        load an aligned face.

        This method plugs into :mod:`lib.aligner` to perform face alignment based on this face's
        ``landmarks_xy``. If the face has already been aligned, then this function will return
        having performed no action.

        Parameters
        ----------
        image: numpy.ndarray
            The image that contains the face to be aligned
        size: int
            The size of the output face in pixels
        align_eyes: bool, optional
            Optionally perform additional alignment to align eyes. Default: `False`
        dtype: str, optional
            Optionally set a ``dtype`` for the final face to be formatted in. Default: ``None``

        Notes
        -----
        This method must be executed to get access to the following `properties`:
            - :func:`original_roi`
            - :func:`aligned_landmarks`
            - :func:`aligned_face`
            - :func:`adjusted_interpolators`
        """
        if self.aligned:
            # Don't reload an already aligned face
            logger.trace("Skipping alignment calculation for already aligned face")
        else:
            logger.trace("Loading aligned face: (size: %s, dtype: %s)", size, dtype)
            padding = int(size * self._extract_ratio) // 2
            self.aligned["size"] = size
            self.aligned["padding"] = padding
            self.aligned["matrix"] = get_align_mat(self)
            self.aligned["face"] = None
        if image is not None and self.aligned["face"] is None:
            logger.trace("Getting aligned face")
            face = AlignerExtract().transform(
                image,
                self.aligned["matrix"],
                size,
                padding)
            self.aligned["face"] = face if dtype is None else face.astype(dtype)

        logger.trace("Loaded aligned face: %s", {key: val
                                                 for key, val in self.aligned.items()
                                                 if key != "face"})

    def _padding_from_coverage(self, size, coverage_ratio):
        """ Return the image padding for a face from coverage_ratio set against a
            pre-padded training image """
        adjusted_ratio = coverage_ratio - (1 - self._extract_ratio)
        padding = round((size * adjusted_ratio) / 2)
        logger.trace(padding)
        return padding

    def load_feed_face(self, image, size=64, coverage_ratio=0.625, dtype=None):
        """ Align a face in the correct dimensions for feeding into a model.

        Parameters
        ----------
        image: numpy.ndarray
            The image that contains the face to be aligned
        size: int
            The size of the face in pixels to be fed into the model
        coverage_ratio: float, optional
            the ratio of the extracted image that was used for training. Default: `0.625`
        dtype: str, optional
            Optionally set a ``dtype`` for the final face to be formatted in. Default: ``None``

        Notes
        -----
        This method must be executed to get access to the following `properties`:
            - :func:`feed_face`
            - :func:`feed_interpolators`
        """
        logger.trace("Loading feed face: (size: %s, coverage_ratio: %s, dtype: %s)",
                     size, coverage_ratio, dtype)

        self.feed["size"] = size
        self.feed["padding"] = self._padding_from_coverage(size, coverage_ratio)
        self.feed["matrix"] = get_align_mat(self)

        face = AlignerExtract().transform(image, self.feed["matrix"], size, self.feed["padding"])
        face = np.clip(face[:, :, :3] / 255., 0., 1.)
        self.feed["face"] = face if dtype is None else face.astype(dtype)

        logger.trace("Loaded feed face. (face_shape: %s, matrix: %s)",
                     self.feed_face.shape, self._feed_matrix)

    def load_reference_face(self, image, size=64, coverage_ratio=0.625, dtype=None):
        """ Align a face in the correct dimensions for reference against the output from a model.

        Parameters
        ----------
        image: numpy.ndarray
            The image that contains the face to be aligned
        size: int
            The size of the face in pixels to be fed into the model
        coverage_ratio: float, optional
            the ratio of the extracted image that was used for training. Default: `0.625`
        dtype: str, optional
            Optionally set a ``dtype`` for the final face to be formatted in. Default: ``None``

        Notes
        -----
        This method must be executed to get access to the following `properties`:
            - :func:`reference_face`
            - :func:`reference_landmarks`
            - :func:`reference_matrix`
            - :func:`reference_interpolators`
        """
        logger.trace("Loading reference face: (size: %s, coverage_ratio: %s, dtype: %s)",
                     size, coverage_ratio, dtype)

        self.reference["size"] = size
        self.reference["padding"] = self._padding_from_coverage(size, coverage_ratio)
        self.reference["matrix"] = get_align_mat(self)

        face = AlignerExtract().transform(image,
                                          self.reference["matrix"],
                                          size,
                                          self.reference["padding"])
        face = np.clip(face[:, :, :3] / 255., 0., 1.)
        self.reference["face"] = face if dtype is None else face.astype(dtype)

        logger.trace("Loaded reference face. (face_shape: %s, matrix: %s)",
                     self.reference_face.shape, self.reference_matrix)

    @property
    def original_roi(self):
        """ numpy.ndarray: The location of the extracted face box within the original frame.
        Only available after :func:`load_aligned` has been called, otherwise returns ``None``"""
        if not self.aligned:
            return None
        roi = AlignerExtract().get_original_roi(self.aligned["matrix"],
                                                self.aligned["size"],
                                                self.aligned["padding"])
        logger.trace("Returning: %s", roi)
        return roi

    @property
    def aligned_landmarks(self):
        """ numpy.ndarray: The 68 point landmarks location transposed to the extracted face box.
        Only available after :func:`load_aligned` has been called, otherwise returns ``None``"""
        if not self.aligned:
            return None
        landmarks = AlignerExtract().transform_points(self.landmarks_xy,
                                                      self.aligned["matrix"],
                                                      self.aligned["size"],
                                                      self.aligned["padding"])
        logger.trace("Returning: %s", landmarks)
        return landmarks

    @property
    def aligned_face(self):
        """ numpy.ndarray: The aligned detected face. Only available after :func:`load_aligned`
        has been called with an image, otherwise returns ``None`` """
        return self.aligned.get("face", None)

    @property
    def _adjusted_matrix(self):
        """ numpy.ndarray: Adjusted matrix for size/padding combination. Only available after
        :func:`load_aligned` has been called, otherwise returns ``None``"""
        if not self.aligned:
            return None
        mat = AlignerExtract().transform_matrix(self.aligned["matrix"],
                                                self.aligned["size"],
                                                self.aligned["padding"])
        logger.trace("Returning: %s", mat)
        return mat

    @property
    def adjusted_interpolators(self):
        """ tuple:  Tuple of (`interpolator` and `reverse interpolator`) for the adjusted matrix.
        Only available after :func:`load_aligned` has been called, otherwise returns ``None``"""
        if not self.aligned:
            return None
        return get_matrix_scaling(self._adjusted_matrix)

    @property
    def feed_face(self):
        """ numpy.ndarray: The aligned face sized for feeding into a model. Only available after
        :func:`load_feed_face` has been called with an image, otherwise returns ``None`` """
        if not self.feed:
            return None
        return self.feed["face"]

    @property
    def _feed_matrix(self):
        """ numpy.ndarray: The adjusted matrix face sized for feeding into a model. Only available
        after :func:`load_feed_face` has been called with an image, otherwise returns ``None`` """
        if not self.feed:
            return None
        mat = AlignerExtract().transform_matrix(self.feed["matrix"],
                                                self.feed["size"],
                                                self.feed["padding"])
        logger.trace("Returning: %s", mat)
        return mat

    @property
    def feed_interpolators(self):
        """ tuple:  Tuple of (`interpolator` and `reverse interpolator`) for the adjusted feed
        matrix. Only available after :func:`load_feed_face` has been called, otherwise returns
        ``None``"""
        if not self.feed:
            return None
        return get_matrix_scaling(self._feed_matrix)

    @property
    def reference_face(self):
        """ numpy.ndarray: The aligned face sized for reference against a face coming out of a
        model. Only available after :func:`load_reference_face` has been called, otherwise
        returns ``None``"""
        if not self.reference:
            return None
        return self.reference["face"]

    @property
    def reference_landmarks(self):
        """ numpy.ndarray: The 68 point landmarks location transposed to the reference face box.
        Only available after :func:`load_reference_face` has been called, otherwise returns
        ``None``"""
        if not self.reference:
            return None
        landmarks = AlignerExtract().transform_points(self.landmarks_xy,
                                                      self.reference["matrix"],
                                                      self.reference["size"],
                                                      self.reference["padding"])
        logger.trace("Returning: %s", landmarks)
        return landmarks

    @property
    def reference_matrix(self):
        """ numpy.ndarray: The adjusted matrix face sized for refence against a face coming out of
         a model. Only available after :func:`load_reference_face` has been called, otherwise
         returns ``None``"""
        if not self.reference:
            return None
        mat = AlignerExtract().transform_matrix(self.reference["matrix"],
                                                self.reference["size"],
                                                self.reference["padding"])
        logger.trace("Returning: %s", mat)
        return mat

    @property
    def reference_interpolators(self):
        """ tuple:  Tuple of (`interpolator` and `reverse interpolator`) for the reference
        matrix. Only available after :func:`load_reference_face` has been called, otherwise
        returns ``None``"""
        if not self.reference:
            return None
        return get_matrix_scaling(self.reference_matrix)
