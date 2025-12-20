#!/usr/bin/env python3
""" Holds the data cache for training data generators """
from __future__ import annotations
import logging
import os
import typing as T

from dataclasses import dataclass, field
from threading import Lock

import cv2
import numpy as np
from tqdm import tqdm

from lib.align import CenteringType, DetectedFace, LandmarkType
from lib.image import read_image_batch, read_image_meta_batch
from lib.logger import parse_class_init
from lib.utils import FaceswapError, get_module_objects
from plugins.train import train_config as cfg

if T.TYPE_CHECKING:
    from lib.align.alignments import PNGHeaderAlignmentsDict, PNGHeaderDict
    from lib import align

logger = logging.getLogger(__name__)
_FACE_CACHES: dict[str, Cache] = {}


@dataclass
class _MaskConfig:
    """ Holds the constants required for manipulating training masks """
    # pylint:disable=unnecessary-lambda
    penalized: bool = field(default_factory=lambda: cfg.Loss.penalized_mask_loss())
    learn: bool = field(default_factory=lambda: cfg.Loss.learn_mask())
    mask_type: str | None = field(default_factory=lambda: None
                                  if cfg.Loss.mask_type() == "none"
                                  else cfg.Loss.mask_type())
    dilation: float = field(default_factory=lambda: cfg.Loss.mask_dilation())
    kernel: int = field(default_factory=lambda: cfg.Loss.mask_blur_kernel())
    threshold: int = field(default_factory=lambda: cfg.Loss.mask_threshold())
    multiplier_enabled: bool = field(
        default_factory=lambda: ((cfg.Loss.eye_multiplier() > 1 or cfg.Loss.mouth_multiplier() > 1)
                                 and cfg.Loss.penalized_mask_loss()))

    @property
    def mask_enabled(self) -> bool:
        """ bool : ``True`` if any of :attr:`penalized` or :attr:`learn` are true and
        :attr:`mask_type` is not ``None`` """
        return self.mask_type is not None and (self.learn or self.penalized)


class _MaskProcessing:
    """ Handle the extraction and processing of masks from faceswap PNG headers for caching

    Parameters
    ----------
    size : int
        The largest output size of the model
    coverage_ratio : float
        The coverage ratio that the model is using.
    centering : Literal["face", "head", "legacy"]
    """
    def __init__(self,
                 size: int,
                 coverage_ratio: float,
                 centering: CenteringType) -> None:

        assert isinstance(size, int)
        assert isinstance(coverage_ratio, float)
        assert centering in T.get_args(CenteringType)

        self._size = size
        self._coverage = coverage_ratio
        self._centering: CenteringType = centering

        self._config = _MaskConfig()
        logger.debug("Initialized %s", self)

    def __repr__(self) -> str:
        """ Pretty print for logging """
        params = f"coverage_ratio={repr(self._coverage)}, centering={repr(self._centering)}"
        return f"{self.__class__.__name__}({params})"

    def _check_mask_exists(self, filename: str, detected_face: DetectedFace) -> None:
        """ Check that the requested mask exists for the current detected face

        Parameters
        ----------
        filename : str
            The file path for the current image
        detected_face : :class:`~lib.align.detected_face.DetectedFace`
            The detected face object that holds the masks

        Raises
        ------
        FaceswapError
            If the requested mask type is not available an error is returned along with a list
            of available masks
        """
        if self._config.mask_type in detected_face.mask:
            return

        exist_masks = list(detected_face.mask)
        msg = "No masks exist for this face"
        if exist_masks:
            msg = f"The masks that exist for this face are: {exist_masks}"
        raise FaceswapError(
            f"You have selected the mask type '{self._config.mask_type}' but at least one "
            "face does not contain the selected mask.\n"
            f"The face that failed was: '{filename}'\n{msg}")

    def _preprocess(self, detected_face: DetectedFace, mask_type: str) -> align.aligned_mask.Mask:
        """ Apply pre-processing to the mask

        Parameters
        ----------
        detected_face : :class:`~lib.align.detected_face.DetectedFace`
            The detected face object that holds the masks
        mask_type : str
            The stored mask type to use

        Returns
        -------
        :class:`~lib.align.aligned_mask.Mask`
            The pre-processed mask at its stored size and crop
        """
        mask = detected_face.mask[mask_type]
        mask.set_dilation(self._config.dilation)
        mask.set_blur_and_threshold(blur_kernel=self._config.kernel,
                                    threshold=self._config.threshold)
        return mask

    def _crop_and_resize(self,
                         detected_face: DetectedFace,
                         mask: align.aligned_mask.Mask) -> np.ndarray:
        """ Crop and resize the mask to the correct centering and training size

        Parameters
        ----------
        detected_face : :class:`~lib.align.detected_face.DetectedFace`
            The detected face object that holds the masks
        mask : :class:`~lib.align.aligned_mask.Mask`
            The pre-processed mask at its stored size and crop

        Returns
        -------
        :class:`numpy.ndarray`
            The processed, cropped and resized final mask
        """
        pose = detected_face.aligned.pose
        mask.set_sub_crop(pose.offset[mask.stored_centering],
                          pose.offset[self._centering],
                          self._centering,
                          self._coverage,
                          detected_face.aligned.y_offset)
        face_mask = mask.mask
        if self._size != face_mask.shape[0]:
            interpolator = cv2.INTER_CUBIC if mask.stored_size < self._size else cv2.INTER_AREA
            face_mask = cv2.resize(face_mask,
                                   (self._size, self._size),
                                   interpolation=interpolator)[..., None]
        return face_mask

    def _get_face_mask(self, filename: str, detected_face: DetectedFace) -> np.ndarray | None:
        """ Obtain the training sized face mask from the DetectedFace for the requested mask type.

        Parameters
        ----------
        filename : str
            The file path for the current image
        detected_face : :class:`~lib.align.detected_face.DetectedFace`
            The detected face object that holds the masks

        Returns
        -------
        :class:`numpy.ndarray` | None
            The face mask used for training or ``None`` if masks are disabled
        """
        if not self._config.mask_enabled:
            return None

        assert self._config.mask_type is not None
        self._check_mask_exists(filename, detected_face)
        mask = self._preprocess(detected_face, self._config.mask_type)
        retval = self._crop_and_resize(detected_face, mask)
        logger.trace("Obtained face mask for: %s %s",  # type:ignore[attr-defined]
                     filename, retval.shape)
        return retval

    def _get_localized_mask(self,
                            filename: str,
                            detected_face: DetectedFace,
                            area: T.Literal["eye", "mouth"]) -> np.ndarray | None:
        """ Obtain a localized mask for the given area if it is required for training.

        Parameters
        ----------
        filename : str
            The file path for the current image
        detected_face : :class:`~lib.align.detected_face.DetectedFace`
            The detected face object that holds the masks
        area : Literal["eye", "mouth"]
            The area of the face to obtain the mask for

        Raises
        ------
        :class:`~lib.utils.FaceswapError`
            If landmark data is not available to generate the localized mask
        """
        if not self._config.multiplier_enabled:
            return None

        try:
            mask = detected_face.get_landmark_mask(area, self._size // 16, 2.5)
        except FaceswapError as err:
            logger.error(str(err))
            raise FaceswapError("Eye/Mouth multiplier masks could not be generated due to missing "
                                f"landmark data. The file that failed was: '{filename}'") from err
        logger.trace("Caching localized '%s' mask for: %s %s",  # type:ignore[attr-defined]
                     area, filename, mask.shape)
        return mask

    def __call__(self, filename: str, detected_face: DetectedFace) -> None:
        """ Prepare the masks required for training and compile into a single compressed array
        within the given DetectedFaces object

        Parameters
        ----------
        filename : str
            The file path for the image that masks are to be prepared for
        detected_face : :class:`~lib.align.detected_face.DetectedFace`
            The detected face object that holds the masks
        """
        masks = [(self._get_face_mask(filename, detected_face))]
        for area in T.get_args(T.Literal["eye", "mouth"]):
            masks.append(self._get_localized_mask(filename, detected_face, area))

        detected_face.store_training_masks(masks, delete_masks=True)
        logger.trace("Stored masks for filename: %s)", filename)  # type:ignore[attr-defined]


def _check_reset(face_cache: "Cache") -> bool:
    """ Check whether a given cache needs to be reset because a face centering change has been
    detected in the other cache.

    Parameters
    ----------
    face_cache : :class:`Cache`
        The cache object that is checking whether it should reset

    Returns
    -------
    bool
        ``True`` if the given object should reset the cache, otherwise ``False``
    """
    check_cache = next((cache for cache in _FACE_CACHES.values() if cache != face_cache), None)
    retval = False if check_cache is None else check_cache.check_reset()
    return retval


@dataclass
class _CacheConfig:
    """ Holds the configuration options for the cache """
    size: int
    """ int : The size to load images at """
    centering: CenteringType
    """ Literal["face", "head", "legacy"] : The centering type to train at """
    coverage: float
    """ float : The selected coverage ration for training """


class Cache():
    """ A thread safe mechanism for collecting and holding face meta information (masks,
    alignments data etc.) for multiple :class:`~lib.training.generator.TrainingDataGenerator`.

    Each side may have up to 3 generators (training, preview and time-lapse). To conserve RAM
    these need to share access to the same face information for the images they are processing.

    As the cache is populated at run-time, thread safe writes are required for the first epoch.
    Following that, the cache is only used for reads, which is thread safe intrinsically.

    It would probably be quicker to set locks on each individual face, but for code complexity
    reasons, and the fact that the lock is only taken up during cache population, and it should
    only be being read multiple times on save iterations, we lock the whole cache during writes.

    Parameters
    ----------
    filenames : list[str]
        The filenames of all the images. This can either be the full path or the base name. If the
        full paths are passed in, they are stripped to base name for use as the cache key.
    size : int
        The largest output size of the model
    coverage_ratio : float
        The coverage ratio that the model is using.
    """
    def __init__(self,
                 filenames: list[str],
                 size: int,
                 coverage_ratio: float) -> None:
        logger.debug(parse_class_init(locals()))
        self._lock = Lock()
        self._cache_info = {"cache_full": False, "has_reset": False}
        self._partially_loaded: list[str] = []

        self._image_count = len(filenames)
        self._cache: dict[str, DetectedFace] = {}
        self._aligned_landmarks: dict[str, np.ndarray] = {}
        self._extract_version = 0.0

        self._config = _CacheConfig(size=size,
                                    centering=T.cast(CenteringType, cfg.centering()),
                                    coverage=coverage_ratio)
        self._mask_prepare = _MaskProcessing(size, coverage_ratio, self._config.centering)
        logger.debug("Initialized: %s", self.__class__.__name__)

    @property
    def cache_full(self) -> bool:
        """ bool : ``True`` if the cache has been fully populated. ``False`` if there are items
        still to be cached. """
        if self._cache_info["cache_full"]:
            return self._cache_info["cache_full"]
        with self._lock:
            return self._cache_info["cache_full"]

    @property
    def aligned_landmarks(self) -> dict[str, np.ndarray]:
        """ dict[str, :class:`numpy.ndarray`] : filename as key, aligned landmarks as value. """
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
        """ int : The pixel size of the cropped aligned face """
        return self._config.size

    def get_items(self, filenames: list[str]) -> list[DetectedFace]:
        """ Obtain the cached items for a list of filenames. The returned list is in the same order
        as the provided filenames.

        Parameters
        ----------
        filenames : list[str]
            A list of image filenames to obtain the cached data for

        Returns
        -------
        list[:class:`~lib.align.detected_face.DetectedFace`]
            List of DetectedFace objects holding the cached metadata. The list returns in the same
            order as the filenames received
        """
        return [self._cache[os.path.basename(filename)] for filename in filenames]

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
                           self._config.centering)
        cfg.centering.set("legacy")
        self._config.centering = "legacy"
        self._cache = {}
        self._cache_info["cache_full"] = False
        if set_flag:
            self._cache_info["has_reset"] = True

    def _validate_version(self, png_meta: PNGHeaderDict, filename: str) -> None:
        """ Validate that there are not a mix of v1.0 extracted faces and v2.x faces.

        Parameters
        ----------
        png_meta : :class:`~lib.align.alignments.PNGHeaderDict`
            The information held within the Faceswap PNG Header
        filename: str
            The full path to the file being validated

        Raises
        ------
        :class:`~lib.utils.FaceswapError`
            If a version 1.0 face appears in a 2.x set or vice versa
        """
        alignment_version = png_meta["source"]["alignments_version"]

        if not self._extract_version:
            logger.debug("Setting initial extract version: %s", alignment_version)
            self._extract_version = alignment_version
            if alignment_version == 1.0 and self._config.centering != "legacy":
                self._reset_cache(True)
            return

        if (self._extract_version == 1.0 and alignment_version > 1.0) or (
                alignment_version == 1.0 and self._extract_version > 1.0):
            raise FaceswapError("Mixing legacy and full head extracted facesets is not supported. "
                                "The following folder contains a mix of extracted face types: "
                                f"'{os.path.dirname(filename)}'")

        self._extract_version = min(alignment_version, self._extract_version)

    def _load_detected_face(self,
                            filename: str,
                            alignments: PNGHeaderAlignmentsDict) -> DetectedFace:
        """ Load a :class:`~lib.align.detected_face.DetectedFace` object and load its associated
        `aligned` property.

        Parameters
        ----------
        filename : str
            The file path for the current image
        alignments : :class:`~lib.align.alignments.PNGHeaderAlignmentsDict`
            The alignments for a single face, extracted from a PNG header

        Returns
        -------
        :class:`~lib.align.detected_face.DetectedFace`
            The loaded Detected Face object
        """
        y_offset = cfg.vertical_offset()
        detected_face = DetectedFace()
        detected_face.from_png_meta(alignments)
        detected_face.load_aligned(None,
                                   size=self._config.size,
                                   centering=self._config.centering,
                                   coverage_ratio=self._config.coverage,
                                   y_offset=y_offset / 100.,
                                   is_aligned=True,
                                   is_legacy=self._extract_version == 1.0)
        logger.trace("Cached aligned face for: %s", filename)  # type:ignore[attr-defined]
        return detected_face

    def _populate_cache(self,
                        needs_cache: list[str],
                        metadata: list[PNGHeaderDict],
                        filenames: list[str]) -> None:
        """ Populate the given items into the cache

        Parameters
        ----------
        needs_cache : list[str]
            The full path to files within this batch that require caching
        metadata : list[:class:`~lib.align.alignments.PNGHeaderDict`]
            The faceswap metadata loaded from the image png header
        filenames : list[str]
            Full path to the filenames that are being loaded in this batch
        """
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

            self._mask_prepare(filename, detected_face)
            self._cache[key] = detected_face

    def _get_batch_with_metadata(self,
                                 filenames: list[str]) -> tuple[np.ndarray, list[PNGHeaderDict]]:
        """ Load a batch of images along with their faceswap metadata for loading into the cache

        Parameters
        ----------
        filenames : list[str]
            Full path to the images to be loaded

        Returns
        -------
        batch : :class:`numpy.ndarray`
            The batch of images in a single array
        metadata : :class:`~lib.align.alignments.PNGHeaderDict`
            The faceswap metadata corresponding to each image in the batch
        """
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
            keys = [os.path.basename(filename) for filename in filenames]
            details = [
                f"{key} ({f'{img.shape[1]}px' if isinstance(img, np.ndarray) else type(img)})"
                for key, img in zip(keys, batch)]
            msg = (f"There are mismatched image sizes in the folder '{folder}'. All training "
                   "images for each side must have the same dimensions.\nThe batch that "
                   f"failed contains the following files:\n{details}.")
            raise FaceswapError(msg)
        return batch, metadata

    def _update_cache_full(self, filenames: list[str]) -> None:
        """ Check if cache is full and update the "cache_full" flag in :attr:`_cache_info` if so

        Parameters
        ----------
        filenames : list[str]
            Full path to the filenames being processed in the current batch
        """
        cache_full = not self._partially_loaded and len(self._cache) == self._image_count
        if cache_full:
            logger.verbose("Cache filled: '%s'",  # type:ignore[attr-defined]
                           os.path.dirname(filenames[0]))
            self._cache_info["cache_full"] = cache_full

    def cache_metadata(self, filenames: list[str]) -> np.ndarray:
        """ Obtain the batch with metadata for items that need caching and cache DetectedFace
        objects to :attr:`_cache`.

        Parameters
        ----------
        filenames : list[str]
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

            needs_cache = [filename for filename, key in zip(filenames, keys)
                           if key not in self._cache or key in self._partially_loaded]
            logger.trace("Needs cache: %s", needs_cache)  # type:ignore[attr-defined]

            if not needs_cache:  # Metadata already cached. Just get images
                logger.debug("All metadata already cached for: %s", keys)
                return read_image_batch(filenames)

            batch, metadata = self._get_batch_with_metadata(filenames)
            self._populate_cache(needs_cache, metadata, filenames)
            self._update_cache_full(filenames)

        return batch

    def pre_fill(self, filenames: list[str], side: T.Literal["a", "b"]) -> None:
        """ When warp to landmarks is enabled, the cache must be pre-filled, as each side needs
        access to the other side's alignments.

        Parameters
        ----------
        filenames : list[str]
            The list of full paths to the images to load the metadata from
        side : Literal["a", "b"]
            The side of the model being cached. Used for info output

        Raises
        ------
        :class:`~lib.utils.FaceSwapError`
            If unsupported landmark type exists or a non-faceswap image is loaded
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
                self._validate_version(meta, filename)
                detected_face = self._load_detected_face(filename, meta["alignments"])

                aligned = detected_face.aligned
                assert aligned is not None
                if aligned.landmark_type != LandmarkType.LM_2D_68:
                    raise FaceswapError("68 Point facial Landmarks are required for Warp-to-"
                                        f"landmarks. The face that failed was: '{filename}'")

                self._cache[key] = detected_face
                self._partially_loaded.append(key)


def get_cache(side: T.Literal["a", "b"],
              filenames: list[str] | None = None,
              size: int | None = None,
              coverage_ratio: float | None = None) -> Cache:
    """ Obtain a :class:`Cache` object for the given side. If the object does not pre-exist then
    create it.

    Parameters
    ----------
    side : Literal["a", "b"]
        The side of the model to obtain the cache for
    filenames : list[str] | None, optional
        The filenames of all the images. This can either be the full path or the base name. If the
        full paths are passed in, they are stripped to base name for use as the cache key. Must be
        passed for the first call of this function for each side. For subsequent calls this
        parameter is ignored. Default: ``None``
    size: int | None, optional
        The largest output size of the model. Must be passed for the first call of this function
        for each side. For subsequent calls this parameter is ignored. Default: ``None``
    coverage_ratio : float | None, optional
        The coverage ratio that the model is using. Must be passed for the first call of this
        function for each side. For subsequent calls this parameter is ignored. Default: ``None``

    Returns
    -------
    :class:`Cache`
        The face meta information cache for the requested side
    """
    assert side in ("a", "b")
    if not _FACE_CACHES.get(side):
        assert filenames is not None, "filenames must be provided for first call to cache"
        assert size is not None, "size must be provided for first call to cache"
        assert coverage_ratio is not None, ("coverage_ratio must be provided for first call to "
                                            "cache")
        logger.debug("Creating cache. side: %s, size: %s, coverage_ratio: %s",
                     side, size, coverage_ratio)
        _FACE_CACHES[side] = Cache(filenames, size, coverage_ratio)
    return _FACE_CACHES[side]


class RingBuffer():
    """ Rolling buffer for holding training/preview batches

    Parameters
    ----------
    batch_size : int
        The batch size to create the buffer for
    image_shape : tuple[int, int, int]
        The height/width/channels shape of a single image in the batch
    buffer_size : int, optional
        The number of arrays to hold in the rolling buffer. Default: `2`
    dtype : str, optional
        The datatype to create the buffer as. Default: `"uint8"`
    """
    def __init__(self,
                 batch_size: int,
                 image_shape: tuple[int, int, int],
                 buffer_size: int = 2,
                 dtype: str = "uint8") -> None:
        logger.debug(parse_class_init(locals()))
        self._max_index = buffer_size - 1
        self._index = 0
        self._buffer = [np.empty((batch_size, *image_shape), dtype=dtype)
                        for _ in range(buffer_size)]
        logger.debug("Initialized: %s", self)

    def __repr__(self) -> str:
        """ Pretty string representation for logging """
        params = {"batch_size": repr(self._buffer[0].shape[0]),
                  "image_shape": repr(self._buffer[0].shape[1:]),
                  "buffer_size": repr(len(self._buffer)),
                  "dtype": repr(str(self._buffer[0].dtype))}
        str_params = [f"{k}={v}" for k, v in params.items()]
        return f"{self.__class__.__name__}({', '.join(str_params)})"

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


__all__ = get_module_objects(__name__)
