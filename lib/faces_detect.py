#!/usr/bin python3
""" Face and landmarks detection for faceswap.py """
import logging

from zlib import compress, decompress

import cv2
import numpy as np
from skimage.transform._geometric import _umeyama as umeyama

from lib.aligner import Extract as AlignerExtract, PoseEstimate, get_matrix_scaling

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


_MEAN_FACE = np.array([[0.010086, 0.106454], [0.085135, 0.038915], [0.191003, 0.018748],
                       [0.300643, 0.034489], [0.403270, 0.077391], [0.596729, 0.077391],
                       [0.699356, 0.034489], [0.808997, 0.018748], [0.914864, 0.038915],
                       [0.989913, 0.106454], [0.500000, 0.203352], [0.500000, 0.307009],
                       [0.500000, 0.409805], [0.500000, 0.515625], [0.376753, 0.587326],
                       [0.435909, 0.609345], [0.500000, 0.628106], [0.564090, 0.609345],
                       [0.623246, 0.587326], [0.131610, 0.216423], [0.196995, 0.178758],
                       [0.275698, 0.179852], [0.344479, 0.231733], [0.270791, 0.245099],
                       [0.192616, 0.244077], [0.655520, 0.231733], [0.724301, 0.179852],
                       [0.803005, 0.178758], [0.868389, 0.216423], [0.807383, 0.244077],
                       [0.729208, 0.245099], [0.264022, 0.780233], [0.350858, 0.745405],
                       [0.438731, 0.727388], [0.500000, 0.742578], [0.561268, 0.727388],
                       [0.649141, 0.745405], [0.735977, 0.780233], [0.652032, 0.864805],
                       [0.566594, 0.902192], [0.500000, 0.909281], [0.433405, 0.902192],
                       [0.347967, 0.864805], [0.300252, 0.784792], [0.437969, 0.778746],
                       [0.500000, 0.785343], [0.562030, 0.778746], [0.699747, 0.784792],
                       [0.563237, 0.824182], [0.500000, 0.831803], [0.436763, 0.824182]])


class DetectedFace():
    """ Detected face and landmark information

    Holds information about a detected face, it's location in a source image
    and the face's 68 point landmarks.

    Methods for aligning a face are also callable from here.

    Parameters
    ----------
    image: numpy.ndarray, optional
        Original frame that holds this face. Optional (not required if just storing coordinates)
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
        dict of {**name** (`str`): :class:`Mask`}.

    Attributes
    ----------
    image: numpy.ndarray, optional
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
        The 68 point landmarks as discovered in :mod:`plugins.extract.align`.
    mask: dict
        The generated mask(s) for the face as generated in :mod:`plugins.extract.mask`. Is a
        dict of {**name** (`str`): :class:`Mask`}.
    hash: str
        The hash of the face. This cannot be set until the file is saved due to image compression,
        but will be set if loading data from :func:`from_alignment`
    """
    def __init__(self, image=None, x=None, w=None, y=None, h=None, landmarks_xy=None, mask=None,
                 filename=None):
        logger.trace("Initializing %s: (image: %s, x: %s, w: %s, y: %s, h:%s, landmarks_xy: %s, "
                     "mask: %s, filename: %s)",
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
        self.thumbnail = None
        self.mask = dict() if mask is None else mask
        self.hash = None

        self.aligned = None
        self.feed = None
        self.reference = None
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

    def add_mask(self, name, mask, affine_matrix, interpolator, storage_size=128):
        """ Add a :class:`Mask` to this detected face

        The mask should be the original output from  :mod:`plugins.extract.mask`
        If a mask with this name already exists it will be overwritten by the given
        mask.

        Parameters
        ----------
        name: str
            The name of the mask as defined by the :attr:`plugins.extract.mask._base.name`
            parameter.
        mask: numpy.ndarray
            The mask that is to be added as output from :mod:`plugins.extract.mask`
            It should be in the range 0.0 - 1.0 ideally with a ``dtype`` of ``float32``
        affine_matrix: numpy.ndarray
            The transformation matrix required to transform the mask to the original frame.
        interpolator, int:
            The CV2 interpolator required to transform this mask to it's original frame.
        storage_size, int (optional):
            The size the mask is to be stored at. Default: 128
        """
        logger.trace("name: '%s', mask shape: %s, affine_matrix: %s, interpolator: %s)",
                     name, mask.shape, affine_matrix, interpolator)
        fsmask = Mask(storage_size=storage_size)
        fsmask.add(mask, affine_matrix, interpolator)
        self.mask[name] = fsmask

    def get_landmark_mask(self, size, area, aligned=True, dilation=0, blur_kernel=0, as_zip=False):
        """ Obtain a single channel mask based on the face's landmark points.

        Parameters
        ----------
        size: int or tuple
            The size of the aligned mask to retrieve. Should be an `int` if an aligned face is
            being requested, or a ('height', 'width') shape tuple if a full frame is being
            requested
        area: ["mouth", "eyes"]
            The type of mask to obtain. `face` is a full face mask the others are masks for those
            specific areas
        aligned: bool
            ``True`` if the returned mask should be for an aligned face. ``False`` if a full frame
            mask should be returned
        dilation: int, optional
            The amount of dilation to apply to the mask. `0` for none. Default: `0`
        blur_kernel: int, optional
            The kernel size for applying gaussian blur to apply to the mask. `0` for none.
            Default: `0`
        as_zip: bool, optional
            ``True`` if the mask should be returned zipped otherwise ``False``

        Returns
        -------
        :class:`numpy.ndarray` or zipped array
            The mask as a single channel image of the given :attr:`size` dimension. If
            :attr:`as_zip` is ``True`` then the :class:`numpy.ndarray` will be contained within a
            zipped container
        """
        # TODO Face mask generation from landmarks
        logger.trace("size: %s, area: %s, aligned: %s, dilation: %s, blur_kernel: %s, as_zip: %s",
                     size, area, aligned, dilation, blur_kernel, as_zip)
        areas = dict(mouth=[slice(48, 60)],
                     eyes=[slice(36, 42), slice(42, 48)])
        if aligned and self.aligned is not None and self.aligned.size != size:
            self.load_aligned(None, size=size, force=True)
        size = (size, size) if aligned else size
        landmarks = self.aligned.landmarks if aligned else self.landmarks_xy
        points = [landmarks[zone] for zone in areas[area]]  # pylint:disable=unsubscriptable-object
        mask = _LandmarksMask(size, points, dilation=dilation, blur_kernel=blur_kernel)
        retval = mask.get(as_zip=as_zip)
        return retval

    def to_alignment(self):
        """  Return the detected face formatted for an alignments file

        returns
        -------
        alignment: dict
            The alignment dict will be returned with the keys ``x``, ``w``, ``y``, ``h``,
            ``landmarks_xy``, ``mask``, ``hash``. The additional key ``thumb`` will be provided
            if the detected face object contains a thumbnail.
        """
        alignment = dict()
        alignment["x"] = self.x
        alignment["w"] = self.w
        alignment["y"] = self.y
        alignment["h"] = self.h
        alignment["landmarks_xy"] = self.landmarks_xy
        alignment["hash"] = self.hash
        alignment["mask"] = {name: mask.to_dict() for name, mask in self.mask.items()}
        if self.thumbnail is not None:
            alignment["thumb"] = self.thumbnail
        logger.trace("Returning: %s", alignment)
        return alignment

    def from_alignment(self, alignment, image=None, with_thumb=False):
        """ Set the attributes of this class from an alignments file and optionally load the face
        into the ``image`` attribute.

        Parameters
        ----------
        alignment: dict
            A dictionary entry for a face from an alignments file containing the keys
            ``x``, ``w``, ``y``, ``h``, ``landmarks_xy``.
            Optionally the key ``thumb`` will be provided. This is for use in the manual tool and
            contains the compressed jpg thumbnail of the face to be allocated to :attr:`thumbnail.
            Optionally the key ``hash`` will be provided, but not all use cases will know the
            face hash at this time.
            Optionally the key ``mask`` will be provided, but legacy alignments will not have
            this key.
        image: numpy.ndarray, optional
            If an image is passed in, then the ``image`` attribute will
            be set to the cropped face based on the passed in bounding box co-ordinates
        with_thumb: bool, optional
            Whether to load the jpg thumbnail into the detected face object, if provided.
            Default: ``False``
        """

        logger.trace("Creating from alignment: (alignment: %s, has_image: %s)",
                     alignment, bool(image is not None))
        self.x = alignment["x"]
        self.w = alignment["w"]
        self.y = alignment["y"]
        self.h = alignment["h"]
        landmarks = alignment["landmarks_xy"]
        if not isinstance(landmarks, np.ndarray):
            landmarks = np.array(landmarks, dtype="float32")
        self.landmarks_xy = landmarks.copy()

        if with_thumb:
            # Thumbnails currently only used for manual tool. Default to None
            self.thumbnail = alignment.get("thumb", None)
        # Manual tool does not know the final hash so default to None
        self.hash = alignment.get("hash", None)
        # Manual tool and legacy alignments will not have a mask
        self.aligned = None
        self.feed = None
        self.reference = None

        if alignment.get("mask", None) is not None:
            self.mask = dict()
            for name, mask_dict in alignment["mask"].items():
                self.mask[name] = Mask()
                self.mask[name].from_dict(mask_dict)
        if image is not None and image.any():
            self._image_to_face(image)
        logger.trace("Created from alignment: (x: %s, w: %s, y: %s. h: %s, "
                     "landmarks: %s, mask: %s)",
                     self.x, self.w, self.y, self.h, self.landmarks_xy, self.mask)

    def _image_to_face(self, image):
        """ set self.image to be the cropped face from detected bounding box """
        logger.trace("Cropping face from image")
        self.image = image[self.top: self.bottom,
                           self.left: self.right]

    # <<< Aligned Face methods and properties >>> #
    def load_aligned(self, image, size=256, dtype=None, centering="legacy", force=False):
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
        dtype: str, optional
            Optionally set a ``dtype`` for the final face to be formatted in. Default: ``None``
        centering: ["legacy", "face", "head"], optional
            The type of extracted face that should be loaded. "legacy" places the nose in the
            center of the image (the original method for aligning). "face" aligns for the nose to
            be in the center of the face (top to bottom) but the center of the skull for left to
            right. "head" aligns for the center of the skull (in 3D space) being the center of the
            extracted image, with the crop holding the full head.
            Default: `"legacy"`
        force: bool, optional
            Force an update of the aligned face, even if it is already loaded. Default: ``False``

        Notes
        -----
        This method must be executed to get access to the following an :class:`AlignedFace` object
        """
        if self.aligned and not force:
            # Don't reload an already aligned face
            logger.trace("Skipping alignment calculation for already aligned face")
        else:
            logger.trace("Loading aligned face: (size: %s, dtype: %s)", size, dtype)
            self.aligned = AlignedFace(self.landmarks_xy,
                                       image=image,
                                       centering=centering,
                                       size=size,
                                       coverage_ratio=1.0,
                                       dtype=dtype,
                                       is_aligned=False)

    def load_feed_face(self, image, size=64, coverage_ratio=0.625, dtype=None,
                       centering="legacy", is_aligned_face=False):
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
        centering: ["legacy", "face", "head"], optional
            The type of extracted face that should be loaded. "legacy" places the nose in the
            center of the image (the original method for aligning). "face" aligns for the nose to
            be in the center of the face (top to bottom) but the center of the skull for left to
            right. "head" aligns for the center of the skull (in 3D space) being the center of the
            extracted image, with the crop holding the full head.
            Default: `"legacy"`
        is_aligned_face: bool, optional
            Indicates that the :attr:`image` is an aligned face rather than a frame.
            Default: ``False``

        Notes
        -----
        This method must be executed to get access to the :attr:`feed` attribute.
        """
        logger.trace("Loading feed face: (size: %s, coverage_ratio: %s, dtype: %s, "
                     "centering: %s, is_aligned_face: %s)", size, coverage_ratio, dtype,
                     centering, is_aligned_face)
        self.feed = AlignedFace(self.landmarks_xy,
                                image=image,
                                centering=centering,
                                size=size,
                                coverage_ratio=coverage_ratio,
                                dtype=dtype,
                                is_aligned=is_aligned_face)

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
        This method must be executed to get access to the attribute :attr:`reference`.
        """
        logger.trace("Loading reference face: (size: %s, coverage_ratio: %s, dtype: %s)",
                     size, coverage_ratio, dtype)
        self.reference = AlignedFace(self.landmarks_xy,
                                     image=image,
                                     centering="legacy",
                                     size=size,
                                     coverage_ratio=coverage_ratio,
                                     dtype=dtype,
                                     is_aligned=False)


class AlignedFace():
    """ Class to align a face.

    Holds the aligned landmarks and face image, as well as associated matrices and information
    about an aligned face.

    Parameters
    ----------
    landmarks: :class:`numpy.ndarray`
        The original 68 point landmarks that pertain to the given image for this face
    image: :class:`numpy.ndarray`, optional
        The original frame that contains the face that is to be aligned. Pass `None` if the aligned
        face is not to be generated, and just the co-ordinates should be calculated.
    centering: ["legacy", "face", "head"], optional
        The type of extracted face that should be loaded. "legacy" places the nose in the center of
        the image (the original method for aligning). "face" aligns for the nose to be in the
        center of the face (top to bottom) but the center of the skull for left to right. "head"
        aligns for the center of the skull (in 3D space) being the center of the extracted image,
        with the crop holding the full head. Default: `"legacy"`
    size: int, optional
        The size in pixels, of each edge of the final aligned face. Default: `64`
    coverage_ratio: float, optional
        The amount of the aligned image to return. A ratio of 1.0 will return the full contents of
        the aligned image. A ratio of 0.5 will return an image of the given size, but will crop to
        the central 50%% of the image.
    dtype: str, optional
        Set a data type for the final face to be returned as. Passing ``None`` will return a face
        with the same data type as the original :attr:`image`. Default: ``None``
    is_aligned_face: bool, optional
        Indicates that the :attr:`image` is an aligned face rather than a frame.
        Default: ``False``
    """
    def __init__(self, landmarks, image=None, centering="legacy", size=64, coverage_ratio=1.0,
                 dtype=None, is_aligned=False):
        logger.trace("Initializing: %s (image shape: %s, centering: %s, size: %s, "
                     "coverage_ratio: %s, dtype: %s, is_aligned: %s)", self.__class__.__name__,
                     image if image is None else image.shape, centering, size, coverage_ratio,
                     dtype, is_aligned)
        self._frame_landmarks = landmarks
        self._centering = centering
        self._size = size
        self._dtype = dtype
        self._is_aligned = is_aligned
        self._matrices = dict(legacy=umeyama(landmarks[17:], _MEAN_FACE, True)[0:2],
                              face=None,
                              head=None)
        self._padding = self._padding_from_coverage(size, coverage_ratio)

        self._pose = None
        self._cache = dict(original_roi=None,
                           landmarks=None,
                           adjusted_matrix=None,
                           interpolators=None)

        self._face = self._extract_face(image)
        logger.trace("Initialized: %s (matrix: %s, padding: %s, face shape: %s)",
                     self.__class__.__name__, self._matrices["legacy"], self._padding,
                     self._face if self._face is None else self._face.shape)

    @property
    def size(self):
        """ int: The size (in pixels) of one side of the square extracted face image. """
        return self._size

    @property
    def padding(self):
        """ int: The amount of padding (in pixels) that is applied to each side of the
        extracted face image for the selected extract type. """
        return self._padding[self._centering]

    @property
    def matrix(self):
        """ :class:`numpy.ndarray`: The 3x2 transformation matrix for extracting and aligning the
        core face area out of the original frame, with no padding or sizing applied. The returned
        matrix is offset for the given :attr:`centering`. """
        if self._matrices[self._centering] is None:
            matrix = self._matrices["legacy"].copy()
            matrix[:, 2] -= self.pose.offset[self._centering]
            self._matrices[self._centering] = matrix
            logger.trace("original matrix: %s, new matrix: %s", self._matrices["legacy"], matrix)
        return self._matrices[self._centering]

    @property
    def pose(self):
        """ :class:`lib.aligner.PoseEstimate`: The estimated pose in 3D space. """
        if self._pose is None:
            lms = cv2.transform(np.expand_dims(self._frame_landmarks, axis=1),
                                self._matrices["legacy"]).squeeze()
            self._pose = PoseEstimate(lms)
        return self._pose

    @property
    def adjusted_matrix(self):
        """ :class:`numpy.ndarray`: The 3x2 transformation matrix for extracting and aligning the
        core face area out of the original frame with padding and sizing applied. """
        if self._cache["adjusted_matrix"] is None:
            matrix = self.matrix.copy()
            mat = matrix * (self._size - 2 * self.padding)
            mat[:, 2] += self.padding
            logger.trace("adjusted_matrix: %s", mat)
            self._cache["adjusted_matrix"] = mat
        return self._cache["adjusted_matrix"]

    @property
    def face(self):
        """ :class:`numpy.ndarray`: The aligned face at the given :attr:`size` at the specified
        :attr:`coverage` in the given :attr:`dtype`. If an :attr:`image` has not been provided
        then an the attribute will return ``None``. """
        return self._face

    @property
    def original_roi(self):
        """ :class:`numpy.ndarray`: The location of the extracted face box within the original
        frame. """
        if self._cache["original_roi"] is None:
            roi = np.array([[0, 0],
                            [0, self._size - 1],
                            [self._size - 1, self._size - 1],
                            [self._size - 1, 0]])
            roi = np.rint(self.transform_points(roi, invert=True)).astype("int32")
            logger.trace("original roi: %s", roi)
            self._cache["original_roi"] = roi
        return self._cache["original_roi"]

    @property
    def landmarks(self):
        """ :class:`numpy.ndarray`: The 68 point facial landmarks aligned to the extracted face
        box. """
        if self._cache["landmarks"] is None:
            lms = self.transform_points(self._frame_landmarks)
            logger.trace("aligned landmarks: %s", lms)
            self._cache["landmarks"] = lms
        return self._cache["landmarks"]

    @property
    def interpolators(self):
        """ tuple: (`interpolator` and `reverse interpolator`) for the :attr:`adjusted matrix`. """
        if self._cache["interpolators"] is None:
            interpolators = get_matrix_scaling(self.adjusted_matrix)
            logger.trace("interpolators: %s", interpolators)
            self._cache["interpolators"] = interpolators
        return self._cache["interpolators"]

    def transform_points(self, points, invert=False):
        """ Perform transformation on a series of (x, y) co-ordinates in world space into
        aligned face space.

        Parameters
        ----------
        points: :class:`numpy.ndarray`
            The points to transform
        invert: bool, optional
            ``True`` to reverse the transformation (i.e. transform the points into world space from
            aligned face space). Default: ``False``

        Returns
        -------
        :class:`numpy.ndarray`
            The transformed points
        """
        retval = np.expand_dims(points, axis=1)
        mat = cv2.invertAffineTransform(self.adjusted_matrix) if invert else self.adjusted_matrix
        retval = cv2.transform(retval, mat, retval.shape).squeeze()
        logger.trace("invert: %s, Original points: %s, transformed points: %s",
                     invert, points, retval)
        return retval

    def _extract_face(self, image):
        """ Extract the face from a source image and populate :attr:`face`. If an image is not
        provided then ``None`` is returned.

        Parameters
        ----------
        image: :class:`numpy.ndarray` or ``None``
            The original frame to extract the face from. ``None`` if the face should not be
            extracted

        Returns
        -------
        :class:`numpy.ndarray` or ``None``
            The extracted face at the given size, with the given coverage of the given dtype or
            ``None`` if no image has been provided.
        """
        if image is None:
            logger.debug("_extract_face called without a loaded image. Returning empty face.")
            return None
        if self._is_aligned:  # Resize the given aligned face
            original_size = image.shape[0]
            interp = cv2.INTER_CUBIC if original_size < self._size else cv2.INTER_AREA
            retval = cv2.resize(image, (self._size, self._size), interpolation=interp)
        else:
            retval = AlignerExtract().transform(image, self.matrix, self._size, self.padding)
        retval = retval if self._dtype is None else retval.astype(self._dtype)
        return retval

    @classmethod
    def _padding_from_coverage(cls, size, coverage_ratio):
        """ Return the image padding for a face from coverage_ratio set against a
            pre-padded training image """
        ratios = dict(legacy=0.375, face=0.5625, head=0.625)
        retval = {_type: round((size * (coverage_ratio - (1 - ratios[_type]))) / 2)
                  for _type in ("legacy", "face", "head")}
        logger.trace(retval)
        return retval


class _LandmarksMask():  # pylint:disable=too-few-public-methods
    """ Create a single channel mask from aligned landmark points.

    size: tuple
        The (height, width) shape tuple that the mask should be returned as
    points: list
        A list of landmark points that correspond to the given shape tuple to create
        the mask. Each item in the list should be a :class:`numpy.ndarray` that a filled
        convex polygon will be created from
    dilation: int, optional
        The amount of dilation to apply to the mask. `0` for none. Default: `0`
    blur_kernel: int, optional
        The kernel size for applying gaussian blur to apply to the mask. `0` for none. Default: `0`
    """
    def __init__(self, size, points, dilation=0, blur_kernel=0):
        logger.trace("Initializing: %s: (size: %s, points: %s, dilation: %s, blur_kernel: %s)",
                     self.__class__.__name__, size, points, dilation, blur_kernel)
        self._size = size
        self._points = points
        self._dilation = dilation
        self._blur_kernel = blur_kernel
        self._mask = None
        logger.trace("Initialized: %s", self.__class__.__name__)

    def get(self, as_zip=False):
        """ Obtain the mask.

        Parameters
        ----------
        as_zip: bool, optional
            ``True`` if the mask should be returned zipped otherwise ``False``

        Returns
        -------
        :class:`numpy.ndarray` or zipped array
            The mask as a single channel image of the given :attr:`size` dimension. If
            :attr:`as_zip` is ``True`` then the :class:`numpy.ndarray` will be contained within a
            zipped container
        """
        if not np.any(self._mask):
            self._generate_mask()
        retval = compress(self._mask) if as_zip else self._mask
        logger.trace("as_zip: %s, retval type: %s", as_zip, type(retval))
        return retval

    def _generate_mask(self):
        """ Generate the mask.

        Creates the mask applying any requested dilation and blurring and assigns to
        :attr:`_mask`

        Returns
        -------
        :class:`numpy.ndarray`
            The mask as a single channel image of the given :attr:`size` dimension.
        """
        mask = np.zeros((self._size) + (1, ), dtype="float32")
        for landmarks in self._points:
            lms = np.rint(landmarks).astype("int")
            cv2.fillConvexPoly(mask, cv2.convexHull(lms), 1.0, lineType=cv2.LINE_AA)
        if self._dilation != 0:
            mask = cv2.dilate(mask,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                        (self._dilation, self._dilation)),
                              iterations=1)
        if self._blur_kernel != 0:
            mask = BlurMask("gaussian", mask, self._blur_kernel).blurred
        logger.trace("mask: (shape: %s, dtype: %s)", mask.shape, mask.dtype)
        self._mask = (mask * 255.0).astype("uint8")


class Mask():
    """ Face Mask information and convenience methods

    Holds a Faceswap mask as generated from :mod:`plugins.extract.mask` and the information
    required to transform it to its original frame.

    Holds convenience methods to handle the warping, storing and retrieval of the mask.

    Parameters
    ----------
    storage_size: int, optional
        The size (in pixels) that the mask should be stored at. Default: 128.

    Attributes
    ----------
    stored_size: int
        The size, in pixels, of the stored mask across its height and width.
    """

    def __init__(self, storage_size=128):
        self.stored_size = storage_size

        self._mask = None
        self._affine_matrix = None
        self._interpolator = None

        self._blur = dict()
        self._blur_kernel = 0
        self._threshold = 0.0
        self.set_blur_and_threshold()

    @property
    def mask(self):
        """ numpy.ndarray: The mask at the size of :attr:`stored_size` with any requested blurring
        and threshold amount applied."""
        dims = (self.stored_size, self.stored_size, 1)
        mask = np.frombuffer(decompress(self._mask), dtype="uint8").reshape(dims)
        if self._threshold != 0.0 or self._blur["kernel"] != 0:
            mask = mask.copy()
        if self._threshold != 0.0:
            mask[mask < self._threshold] = 0.0
            mask[mask > 255.0 - self._threshold] = 255.0
        if self._blur["kernel"] != 0:
            mask = BlurMask(self._blur["type"],
                            mask,
                            self._blur["kernel"],
                            passes=self._blur["passes"]).blurred
        logger.trace("mask shape: %s", mask.shape)
        return mask

    @property
    def original_roi(self):
        """ :class: `numpy.ndarray`: The original region of interest of the mask in the
        source frame. """
        points = np.array([[0, 0],
                           [0, self.stored_size - 1],
                           [self.stored_size - 1, self.stored_size - 1],
                           [self.stored_size - 1, 0]], np.int32).reshape((-1, 1, 2))
        matrix = cv2.invertAffineTransform(self._affine_matrix)
        roi = cv2.transform(points, matrix).reshape((4, 2))
        logger.trace("Returning: %s", roi)
        return roi

    @property
    def affine_matrix(self):
        """ :class: `numpy.ndarray`: The affine matrix to transpose the mask to a full frame. """
        return self._affine_matrix

    @property
    def interpolator(self):
        """ int: The cv2 interpolator required to transpose the mask to a full frame. """
        return self._interpolator

    def get_full_frame_mask(self, width, height):
        """ Return the stored mask in a full size frame of the given dimensions

        Parameters
        ----------
        width: int
            The width of the original frame that the mask was extracted from
        height: int
            The height of the original frame that the mask was extracted from

        Returns
        -------
        numpy.ndarray: The mask affined to the original full frame of the given dimensions
        """
        frame = np.zeros((width, height, 1), dtype="uint8")
        mask = cv2.warpAffine(self.mask,
                              self._affine_matrix,
                              (width, height),
                              frame,
                              flags=cv2.WARP_INVERSE_MAP | self._interpolator,
                              borderMode=cv2.BORDER_CONSTANT)
        logger.trace("mask shape: %s, mask dtype: %s, mask min: %s, mask max: %s",
                     mask.shape, mask.dtype, mask.min(), mask.max())
        return mask

    def add(self, mask, affine_matrix, interpolator):
        """ Add a Faceswap mask to this :class:`Mask`.

        The mask should be the original output from  :mod:`plugins.extract.mask`

        Parameters
        ----------
        mask: numpy.ndarray
            The mask that is to be added as output from :mod:`plugins.extract.mask`
            It should be in the range 0.0 - 1.0 ideally with a ``dtype`` of ``float32``
        affine_matrix: numpy.ndarray
            The transformation matrix required to transform the mask to the original frame.
        interpolator, int:
            The CV2 interpolator required to transform this mask to it's original frame
        """
        logger.trace("mask shape: %s, mask dtype: %s, mask min: %s, mask max: %s, "
                     "affine_matrix: %s, interpolator: %s)", mask.shape, mask.dtype, mask.min(),
                     affine_matrix, mask.max(), interpolator)
        self._affine_matrix = self._adjust_affine_matrix(mask.shape[0], affine_matrix)
        self._interpolator = interpolator
        self.replace_mask(mask)

    def replace_mask(self, mask):
        """ Replace the existing :attr:`_mask` with the given mask.

        Parameters
        ----------
        mask: numpy.ndarray
            The mask that is to be added as output from :mod:`plugins.extract.mask`.
            It should be in the range 0.0 - 1.0 ideally with a ``dtype`` of ``float32``
        """
        mask = (cv2.resize(mask,
                           (self.stored_size, self.stored_size),
                           interpolation=cv2.INTER_AREA) * 255.0).astype("uint8")
        self._mask = compress(mask)

    def set_blur_and_threshold(self,
                               blur_kernel=0, blur_type="gaussian", blur_passes=1, threshold=0):
        """ Set the internal blur kernel and threshold amount for returned masks

        Parameters
        ----------
        blur_kernel: int, optional
            The kernel size, in pixels to apply gaussian blurring to the mask. Set to 0 for no
            blurring. Should be odd, if an even number is passed in (outside of 0) then it is
            rounded up to the next odd number. Default: 0
        blur_type: ["gaussian", "normalized"], optional
            The blur type to use. ``gaussian`` or ``normalized`` box filter. Default: ``gaussian``
        blur_passes: int, optional
            The number of passed to perform when blurring. Default: 1
        threshold: int, optional
            The threshold amount to minimize/maximize mask values to 0 and 100. Percentage value.
            Default: 0
        """
        logger.trace("blur_kernel: %s, threshold: %s", blur_kernel, threshold)
        if blur_type is not None:
            blur_kernel += 0 if blur_kernel == 0 or blur_kernel % 2 == 1 else 1
            self._blur["kernel"] = blur_kernel
            self._blur["type"] = blur_type
            self._blur["passes"] = blur_passes
        self._threshold = (threshold / 100.0) * 255.0

    def _adjust_affine_matrix(self, mask_size, affine_matrix):
        """ Adjust the affine matrix for the mask's storage size

        Parameters
        ----------
        mask_size: int
            The original size of the mask.
        affine_matrix: numpy.ndarray
            The affine matrix to transform the mask at original size to the parent frame.

        Returns
        -------
        affine_matrix: numpy,ndarray
            The affine matrix adjusted for the mask at its stored dimensions.
        """
        zoom = self.stored_size / mask_size
        zoom_mat = np.array([[zoom, 0, 0.], [0, zoom, 0.]])
        adjust_mat = np.dot(zoom_mat, np.concatenate((affine_matrix, np.array([[0., 0., 1.]]))))
        logger.trace("storage_size: %s, mask_size: %s, zoom: %s, original matrix: %s, "
                     "adjusted_matrix: %s", self.stored_size, mask_size, zoom, affine_matrix.shape,
                     adjust_mat.shape)
        return adjust_mat

    def to_dict(self):
        """ Convert the mask to a dictionary for saving to an alignments file

        Returns
        -------
        dict:
            The :class:`Mask` for saving to an alignments file. Contains the keys ``mask``,
            ``affine_matrix``, ``interpolator``, ``stored_size``
        """
        retval = dict()
        for key in ("mask", "affine_matrix", "interpolator", "stored_size"):
            retval[key] = getattr(self, self._attr_name(key))
        logger.trace({k: v if k != "mask" else type(v) for k, v in retval.items()})
        return retval

    def from_dict(self, mask_dict):
        """ Populates the :class:`Mask` from a dictionary loaded from an alignments file.

        Parameters
        ----------
        mask_dict: dict
            A dictionary stored in an alignments file containing the keys ``mask``,
            ``affine_matrix``, ``interpolator``, ``stored_size``
        """
        for key in ("mask", "affine_matrix", "interpolator", "stored_size"):
            setattr(self, self._attr_name(key), mask_dict[key])
            logger.trace("%s - %s", key, mask_dict[key] if key != "mask" else type(mask_dict[key]))

    @staticmethod
    def _attr_name(dict_key):
        """ The :class:`Mask` attribute name for the given dictionary key

        Parameters
        ----------
        dict_key: str
            The key name from an alignments dictionary

        Returns
        -------
        attribute_name: str
            The attribute name for the given key for :class:`Mask`
        """
        retval = "_{}".format(dict_key) if dict_key != "stored_size" else dict_key
        logger.trace("dict_key: %s, attribute_name: %s", dict_key, retval)
        return retval


class BlurMask():  # pylint:disable=too-few-public-methods
    """ Factory class to return the correct blur object for requested blur type.

    Works for square images only. Currently supports Gaussian and Normalized Box Filters.

    Parameters
    ----------
    blur_type: ["gaussian", "normalized"]
        The type of blur to use
    mask: :class:`numpy.ndarray`
        The mask to apply the blur to
    kernel: int or float
        Either the kernel size (in pixels) or the size of the kernel as a ratio of mask size
    is_ratio: bool, optional
        Whether the given :attr:`kernel` parameter is a ratio or not. If ``True`` then the
        actual kernel size will be calculated from the given ratio and the mask size. If
        ``False`` then the kernel size will be set directly from the :attr:`kernel` parameter.
        Default: ``False``
    passes: int, optional
        The number of passes to perform when blurring. Default: ``1``

    Example
    -------
    >>> print(mask.shape)
    (128, 128, 1)
    >>> new_mask = BlurMask("gaussian", mask, 3, is_ratio=False, passes=1).blurred
    >>> print(new_mask.shape)
    (128, 128, 1)
    """
    def __init__(self, blur_type, mask, kernel, is_ratio=False, passes=1):
        logger.trace("Initializing %s: (blur_type: '%s', mask_shape: %s, kernel: %s, "
                     "is_ratio: %s, passes: %s)", self.__class__.__name__, blur_type, mask.shape,
                     kernel, is_ratio, passes)
        self._blur_type = blur_type.lower()
        self._mask = mask
        self._passes = passes
        kernel_size = self._get_kernel_size(kernel, is_ratio)
        self._kernel_size = self._get_kernel_tuple(kernel_size)
        logger.trace("Initialized %s", self.__class__.__name__)

    @property
    def blurred(self):
        """ :class:`numpy.ndarray`: The final mask with blurring applied. """
        func = self._func_mapping[self._blur_type]
        kwargs = self._get_kwargs()
        blurred = self._mask
        for i in range(self._passes):
            ksize = int(kwargs["ksize"][0])
            logger.trace("Pass: %s, kernel_size: %s", i + 1, (ksize, ksize))
            blurred = func(blurred, **kwargs)
            ksize = int(round(ksize * self._multipass_factor))
            kwargs["ksize"] = self._get_kernel_tuple(ksize)
        blurred = blurred[..., None]
        logger.trace("Returning blurred mask. Shape: %s", blurred.shape)
        return blurred

    @property
    def _multipass_factor(self):
        """ For multiple passes the kernel must be scaled down. This value is
            different for box filter and gaussian """
        factor = dict(gaussian=0.8, normalized=0.5)
        return factor[self._blur_type]

    @property
    def _sigma(self):
        """ int: The Sigma for Gaussian Blur. Returns 0 to force calculation from kernel size. """
        return 0

    @property
    def _func_mapping(self):
        """ dict: :attr:`_blur_type` mapped to cv2 Function name. """
        return dict(gaussian=cv2.GaussianBlur,  # pylint: disable = no-member
                    normalized=cv2.blur)  # pylint: disable = no-member

    @property
    def _kwarg_requirements(self):
        """ dict: :attr:`_blur_type` mapped to cv2 Function required keyword arguments. """
        return dict(gaussian=["ksize", "sigmaX"],
                    normalized=["ksize"])

    @property
    def _kwarg_mapping(self):
        """ dict: cv2 function keyword arguments mapped to their parameters. """
        return dict(ksize=self._kernel_size,
                    sigmaX=self._sigma)

    def _get_kernel_size(self, kernel, is_ratio):
        """ Set the kernel size to absolute value.

        If :attr:`is_ratio` is ``True`` then the kernel size is calculated from the given ratio and
        the :attr:`_mask` size, otherwise the given kernel size is just returned.

        Parameters
        ----------
        kernel: int or float
            Either the kernel size (in pixels) or the size of the kernel as a ratio of mask size
        is_ratio: bool, optional
            Whether the given :attr:`kernel` parameter is a ratio or not. If ``True`` then the
            actual kernel size will be calculated from the given ratio and the mask size. If
            ``False`` then the kernel size will be set directly from the :attr:`kernel` parameter.

        Returns
        -------
        int
            The size (in pixels) of the blur kernel
        """
        if not is_ratio:
            return kernel

        mask_diameter = np.sqrt(np.sum(self._mask))
        radius = round(max(1., mask_diameter * kernel / 100.))
        kernel_size = int(radius * 2 + 1)
        logger.trace("kernel_size: %s", kernel_size)
        return kernel_size

    @staticmethod
    def _get_kernel_tuple(kernel_size):
        """ Make sure kernel_size is odd and return it as a tuple.

        Parameters
        ----------
        kernel_size: int
            The size in pixels of the blur kernel

        Returns
        -------
        tuple
            The kernel size as a tuple of ('int', 'int')
        """
        kernel_size += 1 if kernel_size % 2 == 0 else 0
        retval = (kernel_size, kernel_size)
        logger.trace(retval)
        return retval

    def _get_kwargs(self):
        """ dict: the valid keyword arguments for the requested :attr:`_blur_type` """
        retval = {kword: self._kwarg_mapping[kword]
                  for kword in self._kwarg_requirements[self._blur_type]}
        logger.trace("BlurMask kwargs: %s", retval)
        return retval
