#!/usr/bin python3
""" Face and landmarks detection for faceswap.py """
import logging

import cv2
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
    mask: dict
        The generated mask(s) for the face as generated in :mod:`plugins.extract.mask`. Must be a
        dict of `{name (str): mask (numpy.ndarray)}
    """
    def __init__(self, image=None, x=None, w=None, y=None, h=None,
                 landmarks_xy=None, mask=None, filename=None):
        logger.trace("Initializing %s: (image: %s, x: %s, w: %s, y: %s, h:%s, "
                     "landmarks_xy: %s, filename: %s)",
                     self.__class__.__name__,
                     image.shape if image is not None and image.any() else image,
                     x, w, y, h, landmarks_xy,
                     {k: v.shape for k, v in mask} if mask is not None else mask,
                     filename)
        self.image = image
        self.x = x  # pylint:disable=invalid-name
        self.w = w  # pylint:disable=invalid-name
        self.y = y  # pylint:disable=invalid-name
        self.h = h  # pylint:disable=invalid-name
        self.landmarks_xy = landmarks_xy
        self.mask = dict() if mask is None else mask
        self.filename = filename
        self.hash = None
        self.face = None
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
    def training_coverage(self):
        """ The coverage ratio to add for training images """
        return 1.0

    def to_alignment(self):
        """  Return the detected face formatted for an alignments file

        returns
        -------
        alignment: dict
            The alignment dict will be returned with the keys ``x``, ``w``, ``y``, ``h``,
            ``landmarks_xy``, ``mask``, ``hash``.
        """

        alignment = dict()
        alignment["x"] = self.x
        alignment["w"] = self.w
        alignment["y"] = self.y
        alignment["h"] = self.h
        alignment["landmarks_xy"] = self.landmarks_xy
        alignment["hash"] = self.hash
        alignment["mask"] = self.mask
        logger.trace("Returning: %s", alignment)
        return alignment

    def from_alignment(self, alignment, image=None):
        """ Set the attributes of this class from an alignments file and optionally load the face
        into the ``image`` attribute.

        Parameters
        ----------
        alignment: dict
            A dictionary entry for a face from an alignments file containing the keys
            ``x``, ``w``, ``y``, ``h``, ``landmarks_xy``.
            Optionally the key ``hash`` will be provided, but not all use cases will know the
            face hash at this time.
            Optionally the key ``mask`` will be provided, but legacy alignments will not have
            this key.
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
        # Manual tool and legacy alignments will not have a mask
        self.mask = alignment.get("mask", None)
        if image is not None and image.any():
            self.image = image
            self._image_to_face(image)
        logger.trace("Created from alignment: (x: %s, w: %s, y: %s. h: %s, "
                     "landmarks: %s)",
                     self.x, self.w, self.y, self.h, self.landmarks_xy)

    def _image_to_face(self, image):
        """ set self.image to be the cropped face from detected bounding box """
        logger.trace("Cropping face from image")
        self.face = image[self.top: self.bottom,
                          self.left: self.right]

    # <<< Aligned Face methods and properties >>> #
    def load_aligned(self, image, size=256, coverage_ratio=1.0, dtype=None):
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
        coverage_ratio: float
            The metric determining the field of view of the returned face
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
            self.aligned["size"] = size
            self.aligned["padding"] = self._padding_from_coverage(size, coverage_ratio)
            self.aligned["matrix"] = get_align_mat(self)
            self.aligned["face"] = None
        if image is not None and self.aligned["face"] is None:
            logger.trace("Getting aligned face")
            face = AlignerExtract().transform(
                image,
                self.aligned["matrix"],
                size,
                self.aligned["padding"])
            self.aligned["face"] = face if dtype is None else face.astype(dtype)

        logger.trace("Loaded aligned face: %s", {k: str(v) if isinstance(v, np.ndarray) else v
                                                 for k, v in self.aligned.items()
                                                 if k != "face"})

    @staticmethod
    def _padding_from_coverage(size, coverage_ratio):
        """ Return the image padding for a face from coverage_ratio set against a
            pre-padded training image """
        padding = int((size * (coverage_ratio - 0.625)) / 2)
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

        face = AlignerExtract().transform(image,
                                          self.feed["matrix"],
                                          size,
                                          self.feed["padding"])
        self.feed["face"] = face if dtype is None else face.astype(dtype)

        logger.trace("Loaded feed face. (face_shape: %s, matrix: %s)",
                     self.feed_face.shape, self.feed_matrix)

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
    def feed_landmarks(self):
        """ numpy.ndarray: The 68 point landmarks location transposed to the feed face box.
        Only available after :func:`load_reference_face` has been called, otherwise returns
        ``None``"""
        if not self.feed:
            return None
        landmarks = AlignerExtract().transform_points(self.landmarks_xy,
                                                      self.feed["matrix"],
                                                      self.feed["size"],
                                                      self.feed["padding"])
        logger.trace("Returning: %s", landmarks)
        return landmarks

    @property
    def feed_matrix(self):
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
        return get_matrix_scaling(self.feed_matrix)

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


def rotate_landmarks(face, rotation_matrix):
    """ Rotates the 68 point landmarks and detection bounding box around the given rotation matrix.

    Paramaters
    ----------
    face: DetectedFace or dict
        A :class:`DetectedFace` or an `alignments file` ``dict`` containing the 68 point landmarks
        and the `x`, `w`, `y`, `h` detection bounding box points.
    rotation_matrix: numpy.ndarray
        The rotation matrix to rotate the given object by.

    Returns
    -------
    DetectedFace or dict
        The rotated :class:`DetectedFace` or `alignments file` ``dict`` with the landmarks and
        detection bounding box points rotated by the given matrix. The return type is the same as
        the input type for ``face``
    """
    logger.trace("Rotating landmarks: (rotation_matrix: %s, type(face): %s",
                 rotation_matrix, type(face))
    rotated_landmarks = None
    # Detected Face Object
    if isinstance(face, DetectedFace):
        bounding_box = [[face.x, face.y],
                        [face.x + face.w, face.y],
                        [face.x + face.w, face.y + face.h],
                        [face.x, face.y + face.h]]
        landmarks = face.landmarks_xy

    # Alignments Dict
    elif isinstance(face, dict) and "x" in face:
        bounding_box = [[face.get("x", 0), face.get("y", 0)],
                        [face.get("x", 0) + face.get("w", 0),
                         face.get("y", 0)],
                        [face.get("x", 0) + face.get("w", 0),
                         face.get("y", 0) + face.get("h", 0)],
                        [face.get("x", 0),
                         face.get("y", 0) + face.get("h", 0)]]
        landmarks = face.get("landmarks_xy", list())

    else:
        raise ValueError("Unsupported face type")

    logger.trace("Original landmarks: %s", landmarks)

    rotation_matrix = cv2.invertAffineTransform(
        rotation_matrix)
    rotated = list()
    for item in (bounding_box, landmarks):
        if not item:
            continue
        points = np.array(item, np.int32)
        points = np.expand_dims(points, axis=0)
        transformed = cv2.transform(points,
                                    rotation_matrix).astype(np.int32)
        rotated.append(transformed.squeeze())

    # Bounding box should follow x, y planes, so get min/max
    # for non-90 degree rotations
    pt_x = min([pnt[0] for pnt in rotated[0]])
    pt_y = min([pnt[1] for pnt in rotated[0]])
    pt_x1 = max([pnt[0] for pnt in rotated[0]])
    pt_y1 = max([pnt[1] for pnt in rotated[0]])
    width = pt_x1 - pt_x
    height = pt_y1 - pt_y

    if isinstance(face, DetectedFace):
        face.x = int(pt_x)
        face.y = int(pt_y)
        face.w = int(width)
        face.h = int(height)
        face.r = 0
        if len(rotated) > 1:
            rotated_landmarks = [tuple(point) for point in rotated[1].tolist()]
            face.landmarks_xy = rotated_landmarks
    else:
        face["left"] = int(pt_x)
        face["top"] = int(pt_y)
        face["right"] = int(pt_x1)
        face["bottom"] = int(pt_y1)
        rotated_landmarks = face

    logger.trace("Rotated landmarks: %s", rotated_landmarks)
    return face
