#!/usr/bin/env python3
""" Holds the data cache for training data generators """
from __future__ import annotations
import logging
import os
import typing as T

from threading import Lock

import cv2
import numpy as np
from tqdm import tqdm

from lib.align import CenteringType, DetectedFace, LandmarkType
from lib.image import read_image_batch, read_image_meta_batch
from lib.utils import FaceswapError

if T.TYPE_CHECKING:
    from lib.align.alignments import PNGHeaderAlignmentsDict, PNGHeaderDict
    from lib.config import ConfigValueType

logger = logging.getLogger(__name__)
_FACE_CACHES: dict[str, "_Cache"] = {}


def get_cache(side: T.Literal["a", "b"],
              filenames: list[str] | None = None,
              config: dict[str, ConfigValueType] | None = None,
              size: int | None = None,
              coverage_ratio: float | None = None) -> "_Cache":
    """ Obtain a :class:`_Cache` object for the given side. If the object does not pre-exist then
    create it.

    Parameters
    ----------
    side: str
        `"a"` or `"b"`. The side of the model to obtain the cache for
    filenames: list
        The filenames of all the images. This can either be the full path or the base name. If the
        full paths are passed in, they are stripped to base name for use as the cache key. Must be
        passed for the first call of this function for each side. For subsequent calls this
        parameter is ignored. Default: ``None``
    config: dict, optional
        The user selected training configuration options. Must be passed for the first call of this
        function for each side. For subsequent calls this parameter is ignored. Default: ``None``
    size: int, optional
        The largest output size of the model. Must be passed for the first call of this function
        for each side. For subsequent calls this parameter is ignored. Default: ``None``
    coverage_ratio: float: optional
        The coverage ratio that the model is using. Must be passed for the first call of this
        function for each side. For subsequent calls this parameter is ignored. Default: ``None``

    Returns
    -------
    :class:`_Cache`
        The face meta information cache for the requested side
    """
    if not _FACE_CACHES.get(side):
        assert config is not None, ("config must be provided for first call to cache")
        assert filenames is not None, ("filenames must be provided for first call to cache")
        assert size is not None, ("size must be provided for first call to cache")
        assert coverage_ratio is not None, ("coverage_ratio must be provided for first call to "
                                            "cache")
        logger.debug("Creating cache. side: %s, size: %s, coverage_ratio: %s",
                     side, size, coverage_ratio)
        _FACE_CACHES[side] = _Cache(filenames, config, size, coverage_ratio)
    return _FACE_CACHES[side]


def _check_reset(face_cache: "_Cache") -> bool:
    """ Check whether a given cache needs to be reset because a face centering change has been
    detected in the other cache.

    Parameters
    ----------
    face_cache: :class:`_Cache`
        The cache object that is checking whether it should reset

    Returns
    -------
    bool
        ``True`` if the given object should reset the cache, otherwise ``False``
    """
    check_cache = next((cache for cache in _FACE_CACHES.values() if cache != face_cache), None)
    retval = False if check_cache is None else check_cache.check_reset()
    return retval


class _Cache():
    """ A thread safe mechanism for collecting and holding face meta information (masks, "
    "alignments data etc.) for multiple :class:`TrainingDataGenerator`s.

    Each side may have up to 3 generators (training, preview and time-lapse). To conserve VRAM
    these need to share access to the same face information for the images they are processing.

    As the cache is populated at run-time, thread safe writes are required for the first epoch.
    Following that, the cache is only used for reads, which is thread safe intrinsically.

    It would probably be quicker to set locks on each individual face, but for code complexity
    reasons, and the fact that the lock is only taken up during cache population, and it should
    only be being read multiple times on save iterations, we lock the whole cache during writes.

    Parameters
    ----------
    filenames: list
        The filenames of all the images. This can either be the full path or the base name. If the
        full paths are passed in, they are stripped to base name for use as the cache key.
    config: dict
        The user selected training configuration options
    size: int
        The largest output size of the model
    coverage_ratio: float
        The coverage ratio that the model is using.
    """
    def __init__(self,
                 filenames: list[str],
                 config: dict[str, ConfigValueType],
                 size: int,
                 coverage_ratio: float) -> None:
        logger.debug("Initializing: %s (filenames: %s, size: %s, coverage_ratio: %s)",
                     self.__class__.__name__, len(filenames), size, coverage_ratio)
        self._lock = Lock()
        self._cache_info = {"cache_full": False, "has_reset": False}
        self._partially_loaded: list[str] = []

        self._image_count = len(filenames)
        self._cache: dict[str, DetectedFace] = {}
        self._aligned_landmarks: dict[str, np.ndarray] = {}
        self._extract_version = 0.0
        self._size = size

        assert config["centering"] in T.get_args(CenteringType)
        self._centering: CenteringType = T.cast(CenteringType, config["centering"])
        self._config = config
        self._coverage_ratio = coverage_ratio

        logger.debug("Initialized: %s", self.__class__.__name__)

    @property
    def cache_full(self) -> bool:
        """bool: ``True`` if the cache has been fully populated. ``False`` if there are items still
        to be cached. """
        if self._cache_info["cache_full"]:
            return self._cache_info["cache_full"]
        with self._lock:
            return self._cache_info["cache_full"]

    @property
    def aligned_landmarks(self) -> dict[str, np.ndarray]:
        """ dict: The filename as key, aligned landmarks as value. """
        # Note: Aligned landmarks are only used for warp-to-landmarks, so this can safely populate
        # all of the aligned landmarks for the entire cache.
        if not self._aligned_landmarks:
            with self._lock:
                # For Warp-To-Landmarks a race condition can occur where this is referenced from
                # the opposite side prior to it being populated, so block on a lock.
                self._aligned_landmarks = {key: face.aligned.landmarks
                                           for key, face in self._cache.items()}
        return self._aligned_landmarks

    @property
    def size(self) -> int:
        """ int: The pixel size of the cropped aligned face """
        return self._size

    def check_reset(self) -> bool:
        """ Check whether this cache has been reset due to a face centering change, and reset the
        flag if it has.

        Returns
        -------
        bool
            ``True`` if the cache has been reset because of a face centering change due to
            legacy alignments, otherwise ``False``. """
        retval = self._cache_info["has_reset"]
        if retval:
            logger.debug("Resetting 'has_reset' flag")
            self._cache_info["has_reset"] = False
        return retval

    def get_items(self, filenames: list[str]) -> list[DetectedFace]:
        """ Obtain the cached items for a list of filenames. The returned list is in the same order
        as the provided filenames.

        Parameters
        ----------
        filenames: list
            A list of image filenames to obtain the cached data for

        Returns
        -------
        list
            List of DetectedFace objects holding the cached metadata. The list returns in the same
            order as the filenames received
        """
        return [self._cache[os.path.basename(filename)] for filename in filenames]

    def cache_metadata(self, filenames: list[str]) -> np.ndarray:
        """ Obtain the batch with metadata for items that need caching and cache DetectedFace
        objects to :attr:`_cache`.

        Parameters
        ----------
        filenames: list
            List of full paths to image file names

        Returns
        -------
        :class:`numpy.ndarray`
            The batch of face images loaded from disk
        """
        keys = [os.path.basename(filename) for filename in filenames]
        with self._lock:
            if _check_reset(self):
                self._reset_cache(False)

            needs_cache = [filename
                           for filename, key in zip(filenames, keys)
                           if key not in self._cache or key in self._partially_loaded]
            logger.trace("Needs cache: %s", needs_cache)  # type: ignore

            if not needs_cache:
                # Don't bother reading the metadata if no images in this batch need caching
                logger.debug("All metadata already cached for: %s", keys)
                return read_image_batch(filenames)

            try:
                batch, metadata = read_image_batch(filenames, with_metadata=True)
            except ValueError as err:
                if "inhomogeneous" in str(err):
                    raise FaceswapError(
                        "There was an error loading a batch of images. This is most likely due to "
                        "non-faceswap extracted faces in your training folder."
                        "\nAll training images should be Faceswap extracted faces."
                        "\nAll training images should be the same size."
                        f"\nThe files that caused this error are: {filenames}") from err
                raise
            if len(batch.shape) == 1:
                folder = os.path.dirname(filenames[0])
                details = [
                    f"{key} ({f'{img.shape[1]}px' if isinstance(img, np.ndarray) else type(img)})"
                    for key, img in zip(keys, batch)]
                msg = (f"There are mismatched image sizes in the folder '{folder}'. All training "
                       "images for each side must have the same dimensions.\nThe batch that "
                       f"failed contains the following files:\n{details}.")
                raise FaceswapError(msg)

            # Populate items into cache
            for filename in needs_cache:
                key = os.path.basename(filename)
                meta = metadata[filenames.index(filename)]

                # Version Check
                self._validate_version(meta, filename)
                if self._partially_loaded:  # Faces already loaded for Warp-to-landmarks
                    self._partially_loaded.remove(key)
                    detected_face = self._cache[key]
                else:
                    detected_face = self._load_detected_face(filename, meta["alignments"])

                self._prepare_masks(filename, detected_face)
                self._cache[key] = detected_face

            # Update the :attr:`cache_full` attribute
            cache_full = not self._partially_loaded and len(self._cache) == self._image_count
            if cache_full:
                logger.verbose("Cache filled: '%s'", os.path.dirname(filenames[0]))  # type: ignore
                self._cache_info["cache_full"] = cache_full

        return batch

    def pre_fill(self, filenames: list[str], side: T.Literal["a", "b"]) -> None:
        """ When warp to landmarks is enabled, the cache must be pre-filled, as each side needs
        access to the other side's alignments.

        Parameters
        ----------
        filenames: list
            The list of full paths to the images to load the metadata from
        side: str
            `"a"` or `"b"`. The side of the model being cached. Used for info output

        Raises
        ------
        FaceSwapError
            If unsupported landmark type exists
        """
        with self._lock:
            for filename, meta in tqdm(read_image_meta_batch(filenames),
                                       desc=f"WTL: Caching Landmarks ({side.upper()})",
                                       total=len(filenames),
                                       leave=False):
                if "itxt" not in meta or "alignments" not in meta["itxt"]:
                    raise FaceswapError(f"Invalid face image found. Aborting: '{filename}'")

                meta = meta["itxt"]
                key = os.path.basename(filename)
                # Version Check
                self._validate_version(meta, filename)
                detected_face = self._load_detected_face(filename, meta["alignments"])

                aligned = detected_face.aligned
                assert aligned is not None
                if aligned.landmark_type != LandmarkType.LM_2D_68:
                    raise FaceswapError("68 Point facial Landmarks are required for Warp-to-"
                                        f"landmarks. The face that failed was: '{filename}'")

                self._cache[key] = detected_face
                self._partially_loaded.append(key)

    def _validate_version(self, png_meta: PNGHeaderDict, filename: str) -> None:
        """ Validate that there are not a mix of v1.0 extracted faces and v2.x faces.

        Parameters
        ----------
        png_meta: dict
            The information held within the Faceswap PNG Header
        filename: str
            The full path to the file being validated

        Raises
        ------
        FaceswapError
            If a version 1.0 face appears in a 2.x set or vice versa
        """
        alignment_version = png_meta["source"]["alignments_version"]

        if not self._extract_version:
            logger.debug("Setting initial extract version: %s", alignment_version)
            self._extract_version = alignment_version
            if alignment_version == 1.0 and self._centering != "legacy":
                self._reset_cache(True)
            return

        if (self._extract_version == 1.0 and alignment_version > 1.0) or (
                alignment_version == 1.0 and self._extract_version > 1.0):
            raise FaceswapError("Mixing legacy and full head extracted facesets is not supported. "
                                "The following folder contains a mix of extracted face types: "
                                f"'{os.path.dirname(filename)}'")

        self._extract_version = min(alignment_version, self._extract_version)

    def _reset_cache(self, set_flag: bool) -> None:
        """ In the event that a legacy extracted face has been seen, and centering is not legacy
        the cache will need to be reset for legacy centering.

        Parameters
        ----------
        set_flag: bool
            ``True`` if the flag should be set to indicate that the cache is being reset because of
            a legacy face set/centering mismatch. ``False`` if the cache is being reset because it
            has detected a reset flag from the opposite cache.
        """
        if set_flag:
            logger.warning("You are using legacy extracted faces but have selected '%s' centering "
                           "which is incompatible. Switching centering to 'legacy'",
                           self._centering)
        self._config["centering"] = "legacy"
        self._centering = "legacy"
        self._cache = {}
        self._cache_info["cache_full"] = False
        if set_flag:
            self._cache_info["has_reset"] = True

    def _load_detected_face(self,
                            filename: str,
                            alignments: PNGHeaderAlignmentsDict) -> DetectedFace:
        """ Load a :class:`DetectedFace` object and load its associated `aligned` property.

        Parameters
        ----------
        filename: str
            The file path for the current image
        alignments: dict
            The alignments for a single face, extracted from a PNG header

        Returns
        -------
        :class:`lib.align.DetectedFace`
            The loaded Detected Face object
        """
        detected_face = DetectedFace()
        detected_face.from_png_meta(alignments)
        detected_face.load_aligned(None,
                                   size=self._size,
                                   centering=self._centering,
                                   coverage_ratio=self._coverage_ratio,
                                   is_aligned=True,
                                   is_legacy=self._extract_version == 1.0)
        logger.trace("Cached aligned face for: %s", filename)  # type: ignore
        return detected_face

    def _prepare_masks(self, filename: str, detected_face: DetectedFace) -> None:
        """ Prepare the masks required from training, and compile into a single compressed array

        Parameters
        ----------
        filename: str
            The file path for the current image
        detected_face: :class:`lib.align.DetectedFace`
            The detected face object that holds the masks
        """
        masks = [(self._get_face_mask(filename, detected_face))]
        for area in T.get_args(T.Literal["eye", "mouth"]):
            masks.append(self._get_localized_mask(filename, detected_face, area))

        detected_face.store_training_masks(masks, delete_masks=True)
        logger.trace("Stored masks for filename: %s)", filename)  # type: ignore

    def _get_face_mask(self, filename: str, detected_face: DetectedFace) -> np.ndarray | None:
        """ Obtain the training sized face mask from the :class:`DetectedFace` for the requested
        mask type.

        Parameters
        ----------
        filename: str
            The file path for the current image
        detected_face: :class:`lib.align.DetectedFace`
            The detected face object that holds the masks

        Raises
        ------
        FaceswapError
            If the requested mask type is not available an error is returned along with a list
            of available masks
        """
        if not self._config["penalized_mask_loss"] and not self._config["learn_mask"]:
            return None

        if not self._config["mask_type"]:
            logger.debug("No mask selected. Not validating")
            return None

        if self._config["mask_type"] not in detected_face.mask:
            exist_masks = list(detected_face.mask)
            msg = "No masks exist for this face"
            if exist_masks:
                msg = f"The masks that exist for this face are: {exist_masks}"
            raise FaceswapError(
                f"You have selected the mask type '{self._config['mask_type']}' but at least one "
                "face does not contain the selected mask.\n"
                f"The face that failed was: '{filename}'\n{msg}")

        mask = detected_face.mask[str(self._config["mask_type"])]
        assert isinstance(self._config["mask_dilation"], float)
        assert isinstance(self._config["mask_blur_kernel"], int)
        assert isinstance(self._config["mask_threshold"], int)
        mask.set_dilation(self._config["mask_dilation"])
        mask.set_blur_and_threshold(blur_kernel=self._config["mask_blur_kernel"],
                                    threshold=self._config["mask_threshold"])

        pose = detected_face.aligned.pose
        mask.set_sub_crop(pose.offset[mask.stored_centering],
                          pose.offset[self._centering],
                          self._centering,
                          self._coverage_ratio)
        face_mask = mask.mask
        if self._size != face_mask.shape[0]:
            interpolator = cv2.INTER_CUBIC if mask.stored_size < self._size else cv2.INTER_AREA
            face_mask = cv2.resize(face_mask,
                                   (self._size, self._size),
                                   interpolation=interpolator)[..., None]

        logger.trace("Obtained face mask for: %s %s", filename, face_mask.shape)  # type: ignore
        return face_mask

    def _get_localized_mask(self,
                            filename: str,
                            detected_face: DetectedFace,
                            area: T.Literal["eye", "mouth"]) -> np.ndarray | None:
        """ Obtain a localized mask for the given area if it is required for training.

        Parameters
        ----------
        filename: str
            The file path for the current image
        detected_face: :class:`lib.align.DetectedFace`
            The detected face object that holds the masks
        area: str
            `"eye"` or `"mouth"`. The area of the face to obtain the mask for
        """
        multiplier = self._config[f"{area}_multiplier"]
        assert isinstance(multiplier, int)
        if not self._config["penalized_mask_loss"] or multiplier <= 1:
            return None
        try:
            mask = detected_face.get_landmark_mask(area, self._size // 16, 2.5)
        except FaceswapError as err:
            logger.error(str(err))
            raise FaceswapError("Eye/Mouth multiplier masks could not be generated due to missing "
                                f"landmark data. The file that failed was: '{filename}'") from err
        logger.trace("Caching localized '%s' mask for: %s %s",  # type: ignore
                     area, filename, mask.shape)
        return mask


class RingBuffer():
    """ Rolling buffer for holding training/preview batches

    Parameters
    ----------
    batch_size: int
        The batch size to create the buffer for
    image_shape: tuple
        The height/width/channels shape of a single image in the batch
    buffer_size: int, optional
        The number of arrays to hold in the rolling buffer. Default: `2`
    dtype: str, optional
        The datatype to create the buffer as. Default: `"uint8"`
    """
    def __init__(self,
                 batch_size: int,
                 image_shape: tuple[int, int, int],
                 buffer_size: int = 2,
                 dtype: str = "uint8") -> None:
        logger.debug("Initializing: %s (batch_size: %s, image_shape: %s, buffer_size: %s, "
                     "dtype: %s", self.__class__.__name__, batch_size, image_shape, buffer_size,
                     dtype)
        self._max_index = buffer_size - 1
        self._index = 0
        self._buffer = [np.empty((batch_size, *image_shape), dtype=dtype)
                        for _ in range(buffer_size)]
        logger.debug("Initialized: %s", self.__class__.__name__)  # type: ignore

    def __call__(self) -> np.ndarray:
        """ Obtain the next array from the ring buffer

        Returns
        -------
        :class:`np.ndarray`
            A pre-allocated numpy array from the buffer
        """
        retval = self._buffer[self._index]
        self._index += 1 if self._index < self._max_index else -self._max_index
        return retval
