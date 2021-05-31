#!/usr/bin/env python3
""" Handles Data Augmentation for feeding Faceswap Models """

import logging
import os

from random import shuffle, choice
from threading import Lock
from zlib import decompress

import numpy as np
import cv2
from tqdm import tqdm
from lib.align import AlignedFace, DetectedFace, get_centered_size
from lib.image import read_image_batch, read_image_meta_batch
from lib.multithreading import BackgroundGenerator
from lib.utils import FaceswapError

from . import ImageAugmentation

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

_FACE_CACHES = dict()


def _get_cache(side, filenames, config):
    """ Obtain a :class:`_Cache` object for the given side. If the object does not pre-exist then
    create it.

    Parameters
    ----------
    side: str
        `"a"` or `"b"`. The side of the model to obtain the cache for
    filenames: list
        The filenames of all the images. This can either be the full path or the base name. If the
        full paths are passed in, they are stripped to base name for use as the cache key.
    config: dict
        The user selected training configuration options

    Returns
    -------
    :class:`_Cache`
        The face meta information cache for the requested side
    """
    if not _FACE_CACHES.get(side):
        logger.debug("Creating cache. Side: %s", side)
        _FACE_CACHES[side] = _Cache(filenames, config)
    return _FACE_CACHES[side]


def _check_reset(face_cache):
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
    retval = check_cache if check_cache is None else check_cache.check_reset()
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
    """
    def __init__(self, filenames, config):
        self._lock = Lock()
        self._cache = {os.path.basename(filename): dict(cached=False) for filename in filenames}
        self._aligned_landmarks = None
        self._partial_load = False
        self._cache_full = False
        self._extract_version = None
        self._has_reset = False
        self._size = None

        self._centering = config["centering"]
        self._config = config

    @property
    def cache_full(self):
        """bool: ``True`` if the cache has been fully populated. ``False`` if there are items still
        to be cached. """
        if self._cache_full:
            return self._cache_full
        with self._lock:
            return self._cache_full

    @property
    def partially_loaded(self):
        """ bool: ``True`` if the cache has been partially loaded for Warp To Landmarks otherwise
        ``False`` """
        if self._partial_load:
            return self._partial_load
        with self._lock:
            return self._partial_load

    @property
    def extract_version(self):
        """ float: The alignments file version used to extract the faces. """
        return self._extract_version

    @property
    def aligned_landmarks(self):
        """ dict: The filename as key, aligned landmarks as value """
        if self._aligned_landmarks is None:
            with self._lock:
                # For Warp-To-Landmarks a race condition can occur where this is referenced from
                # the opposite side prior to it being populated, so block on a lock.
                self._aligned_landmarks = {key: val["aligned_face"].landmarks
                                           for key, val in self._cache.items()}
        return self._aligned_landmarks

    @property
    def crop_size(self):
        """ int: The pixel size of the cropped aligned face """
        return self._size

    def check_reset(self):
        """ Check whether this cache has been reset due to a face centering change, and reset the
        flag if it has.

        Returns
        -------
        bool
            ``True`` if the cache has been reset because of a face centering change due to
            legacy alignments, otherwise ``False``. """
        retval = self._has_reset
        if retval:
            logger.debug("Resetting 'has_reset' flag")
            self._has_reset = False
        return retval

    def get_items(self, filenames):
        """ Obtain the cached items for a list of filenames. The returned list is in the same order
        as the provided filenames.

        Parameters
        ----------
        filenames: list
            A list of image filenames to obtain the cached data for

        Returns
        -------
        list
            List of dictionaries containing the cached metadata. The list returns in the same order
            as the filenames received
        """
        return [self._cache[os.path.basename(filename)] for filename in filenames]

    def cache_metadata(self, filenames):
        """ Obtain the batch with metadata for items that need caching and cache them to
        :attr:`_cache`.

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
                           if not self._cache[key]["cached"]]
            logger.trace("Needs cache: %s", needs_cache)

            if not needs_cache:
                # Don't bother reading the metadata if no images in this batch need caching
                logger.debug("All metadata already cached for: %s", keys)
                return read_image_batch(filenames)

            batch, metadata = read_image_batch(filenames, with_metadata=True)

            if len(batch.shape) == 1:
                folder = os.path.dirname(filenames[0])
                details = [
                    "{0} ({1})".format(
                        key, f"{img.shape[1]}px" if isinstance(img, np.ndarray) else type(img))
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
                if self._partial_load:  # Faces already loaded for Warp-to-landmarks
                    detected_face = self._cache[key]["detected_face"]
                else:
                    detected_face = self._add_aligned_face(filename,
                                                           meta["alignments"],
                                                           batch.shape[1])

                self._add_mask(filename, detected_face)
                for area in ("eye", "mouth"):
                    self._add_localized_mask(filename, detected_face, area)

                self._cache[key]["cached"] = True
            # Update the :attr:`cache_full` attribute
            cache_full = all(item["cached"] for item in self._cache.values())
            if cache_full:
                logger.verbose("Cache filled: '%s'", os.path.dirname(filenames[0]))
                self._cache_full = cache_full

        return batch

    def pre_fill(self, filenames, side):
        """ When warp to landmarks is enabled, the cache must be pre-filled, as each side needs
        access to the other side's alignments.

        Parameters
        ----------
        filenames: list
            The list of full paths to the images to load the metadata from
        side: str
            `"a"` or `"b"`. The side of the model being cached. Used for info output
        """
        with self._lock:
            for filename, meta in tqdm(read_image_meta_batch(filenames),
                                       desc="WTL: Caching Landmarks ({})".format(side.upper()),
                                       total=len(filenames),
                                       leave=False):
                if "itxt" not in meta or "alignments" not in meta["itxt"]:
                    raise FaceswapError(f"Invalid face image found. Aborting: '{filename}'")

                size = meta["width"]
                meta = meta["itxt"]
                # Version Check
                self._validate_version(meta, filename)
                detected_face = self._add_aligned_face(filename, meta["alignments"], size)
                self._cache[os.path.basename(filename)]["detected_face"] = detected_face
            self._partial_load = True

    def _validate_version(self, png_meta, filename):
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
                                "{}".format(os.path.dirname(filename)))

        self._extract_version = min(alignment_version, self._extract_version)

    def _reset_cache(self, set_flag):
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
        self._cache = {key: dict(cached=False) for key in self._cache}
        self._cache_full = False
        self._size = None
        if set_flag:
            self._has_reset = True

    def _add_aligned_face(self, filename, alignments, image_size):
        """ Add a :class:`lib.align.AlignedFace` object to the cache.

        Parameters
        ----------
        filename: str
            The file path for the current image
        alignments: dict
            The alignments for a single face, extracted from a PNG header
        image_size: int
            The pixel size of the image loaded from disk

        Returns
        -------
        :class:`lib.align.DetectedFace`
            The Detected Face object that was used to create the Aligned Face
        """
        if self._size is None:
            self._size = get_centered_size("legacy" if self._extract_version == 1.0 else "head",
                                           self._centering,
                                           image_size)

        detected_face = DetectedFace()
        detected_face.from_png_meta(alignments)

        aligned_face = AlignedFace(detected_face.landmarks_xy,
                                   centering=self._centering,
                                   size=self._size,
                                   is_aligned=True)
        logger.trace("Caching aligned face for: %s", filename)
        self._cache[os.path.basename(filename)]["aligned_face"] = aligned_face
        return detected_face

    def _add_mask(self, filename, detected_face):
        """ Load the mask to the cache if a mask is required for training.

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
            return

        if not self._config["mask_type"]:
            logger.debug("No mask selected. Not validating")
            return

        if self._config["mask_type"] not in detected_face.mask:
            raise FaceswapError(
                "You have selected the mask type '{}' but at least one face does not contain the "
                "selected mask.\nThe face that failed was: '{}'\nThe masks that exist for this "
                "face are: {}".format(
                    self._config["mask_type"], filename, list(detected_face.mask)))

        key = os.path.basename(filename)
        mask = detected_face.mask[self._config["mask_type"]]
        mask.set_blur_and_threshold(blur_kernel=self._config["mask_blur_kernel"],
                                    threshold=self._config["mask_threshold"])

        pose = self._cache[key]["aligned_face"].pose
        mask.set_sub_crop(pose.offset[self._centering] - pose.offset[mask.stored_centering],
                          self._centering)

        logger.trace("Caching mask for: %s", filename)
        self._cache[key]["mask"] = mask

    def _add_localized_mask(self, filename, detected_face, area):
        """ Load a localized mask to the cache for the given area if it is required for training.

        Parameters
        ----------
        filename: str
            The file path for the current image
        detected_face: :class:`lib.align.DetectedFace`
            The detected face object that holds the masks
        area: str
            `"eye"` or `"mouth"`. The area of the face to obtain the mask for
        """
        if not self._config["penalized_mask_loss"] or self._config[f"{area}_multiplier"] <= 1:
            return
        key = "eyes" if area == "eye" else area

        logger.trace("Caching localized '%s' mask for: %s", key, filename)
        self._cache[os.path.basename(filename)][f"mask_{key}"] = detected_face.get_landmark_mask(
            self._size,
            key,
            aligned=True,
            centering=self._centering,
            dilation=self._size // 32,
            blur_kernel=self._size // 16,
            as_zip=True)


class TrainingDataGenerator():  # pylint:disable=too-few-public-methods
    """ A Training Data Generator for compiling data for feeding to a model.

    This class is called from :mod:`plugins.train.trainer._base` and launches a background
    iterator that compiles augmented data, target data and sample data.

    Parameters
    ----------
    model_input_size: int
        The expected input size for the model. It is assumed that the input to the model is always
        a square image. This is the size, in pixels, of the `width` and the `height` of the input
        to the model.
    model_output_shapes: list
        A list of tuples defining the output shapes from the model, in the order that the outputs
        are returned. The tuples should be in (`height`, `width`, `channels`) format.
    coverage_ratio: float
        The ratio of the training image to be trained on. Dictates how much of the image will be
        cropped out. E.G: a coverage ratio of 0.625 will result in cropping a 160px box from a
        256px image (:math:`256 * 0.625 = 160`).
    color_order: ["rgb", "bgr"]
        The color order that the model expects as input
    augment_color: bool
        ``True`` if color is to be augmented, otherwise ``False``
    no_flip: bool
        ``True`` if the image shouldn't be randomly flipped as part of augmentation, otherwise
        ``False``
    no_warp: bool
        ``True`` if the image shouldn't be warped as part of augmentation, otherwise ``False``
    warp_to_landmarks: bool
        ``True`` if the random warp method should warp to similar landmarks from the other side,
        ``False`` if the standard random warp method should be used.
    face_cache: dict
        A thread safe dictionary containing a cache of information relating to all faces being
        trained on
    config: dict
        The configuration `dict` generated from :file:`config.train.ini` containing the trainer
        plugin configuration options.
    """
    def __init__(self, model_input_size, model_output_shapes, coverage_ratio, color_order,
                 augment_color, no_flip, no_warp, warp_to_landmarks, config):
        logger.debug("Initializing %s: (model_input_size: %s, model_output_shapes: %s, "
                     "coverage_ratio: %s, color_order: %s, augment_color: %s, no_flip: %s, "
                     "no_warp: %s, warp_to_landmarks: %s, config: %s)",
                     self.__class__.__name__, model_input_size, model_output_shapes,
                     coverage_ratio, color_order, augment_color, no_flip, no_warp,
                     warp_to_landmarks, config)
        self._config = config
        self._model_input_size = model_input_size
        self._model_output_shapes = model_output_shapes
        self._coverage_ratio = coverage_ratio
        self._color_order = color_order.lower()
        self._augment_color = augment_color
        self._no_flip = no_flip
        self._warp_to_landmarks = warp_to_landmarks
        self._no_warp = no_warp

        # Batchsize and processing class are set when this class is called by a feeder
        # from lib.training_data
        self._batchsize = 0
        self._face_cache = None
        self._nearest_landmarks = dict()
        self._processing = None
        logger.debug("Initialized %s", self.__class__.__name__)

    def minibatch_ab(self, images, batchsize, side,
                     do_shuffle=True, is_preview=False, is_timelapse=False):
        """ A Background iterator to return augmented images, samples and targets.

        The exit point from this class and the sole attribute that should be referenced. Called
        from :mod:`plugins.train.trainer._base`. Returns an iterator that yields images for
        training, preview and time-lapses.

        Parameters
        ----------
        images: list
            A list of image paths that will be used to compile the final augmented data from.
        batchsize: int
            The batchsize for this iterator. Images will be returned in :class:`numpy.ndarray`
            objects of this size from the iterator.
        side: {'a' or 'b'}
            The side of the model that this iterator is for.
        do_shuffle: bool, optional
            Whether data should be shuffled prior to loading from disk. If true, each time the full
            list of filenames are processed, the data will be reshuffled to make sure they are not
            returned in the same order. Default: ``True``
        is_preview: bool, optional
            Indicates whether this iterator is generating preview images. If ``True`` then certain
            augmentations will not be performed. Default: ``False``
        is_timelapse: bool optional
            Indicates whether this iterator is generating time-lapse images. If ``True``, then
            certain augmentations will not be performed. Default: ``False``

        Yields
        ------
        dict
            The following items are contained in each `dict` yielded from this iterator:

            * **feed** (:class:`numpy.ndarray`) - The feed for the model. The array returned is \
            in the format (`batchsize`, `height`, `width`, `channels`). This is the :attr:`x` \
            parameter for :func:`keras.models.model.train_on_batch`.

            * **targets** (`list`) - A list of 4-dimensional :class:`numpy.ndarray` objects in \
            the order and size of each output of the model as defined in \
            :attr:`model_output_shapes`. the format of these arrays will be (`batchsize`, \
            `height`, `width`, `3`). This is the :attr:`y` parameter for \
            :func:`keras.models.model.train_on_batch` **NB:** masks are not included in the \
            `targets` list. If required for feeding into the Keras model, they will need to be \
            added to this list in :mod:`plugins.train.trainer._base` from the `masks` key.

            * **masks** (:class:`numpy.ndarray`) - A 4-dimensional array containing the target \
            masks in the format (`batchsize`, `height`, `width`, `1`).

            * **samples** (:class:`numpy.ndarray`) - A 4-dimensional array containing the samples \
            for feeding to the model's predict function for generating preview and time-lapse \
            samples. The array will be in the format (`batchsize`, `height`, `width`, \
            `channels`). **NB:** This item will only exist in the `dict` if :attr:`is_preview` \
            or :attr:`is_timelapse` is ``True``
        """
        logger.debug("Queue batches: (image_count: %s, batchsize: %s, side: '%s', do_shuffle: %s, "
                     "is_preview, %s, is_timelapse: %s)", len(images), batchsize, side, do_shuffle,
                     is_preview, is_timelapse)
        self._batchsize = batchsize
        self._face_cache = _get_cache(side, images, self._config)
        self._processing = ImageAugmentation(batchsize,
                                             is_preview or is_timelapse,
                                             self._model_input_size,
                                             self._model_output_shapes,
                                             self._coverage_ratio,
                                             self._config)

        if self._warp_to_landmarks and not self._face_cache.partially_loaded:
            self._face_cache.pre_fill(images, side)

        args = (images, side, do_shuffle, batchsize)
        batcher = BackgroundGenerator(self._minibatch, thread_count=2, args=args)
        return batcher.iterator()

    # << INTERNAL METHODS >> #
    def _validate_samples(self, data):
        """ Ensures that the total number of images within :attr:`images` is greater or equal to
        the selected :attr:`batchsize`. Raises an exception if this is not the case. """
        length = len(data)
        msg = ("Number of images is lower than batch-size (Note that too few "
               "images may lead to bad training). # images: {}, "
               "batch-size: {}".format(length, self._batchsize))
        try:
            assert length >= self._batchsize, msg
        except AssertionError as err:
            msg += ("\nYou should increase the number of images in your training set or lower "
                    "your batch-size.")
            raise FaceswapError(msg) from err

    def _minibatch(self, images, side, do_shuffle, batchsize):
        """ A generator function that yields the augmented, target and sample images.
        see :func:`minibatch_ab` for more details on the output. """
        logger.debug("Loading minibatch generator: (image_count: %s, side: '%s', do_shuffle: %s)",
                     len(images), side, do_shuffle)
        self._validate_samples(images)

        def _img_iter(imgs):
            while True:
                if do_shuffle:
                    shuffle(imgs)
                for img in imgs:
                    yield img

        img_iter = _img_iter(images)
        while True:
            img_paths = [next(img_iter) for _ in range(batchsize)]
            yield self._process_batch(img_paths, side)

        logger.debug("Finished minibatch generator: (side: '%s')", side)

    def _process_batch(self, filenames, side):
        """ Performs the augmentation and compiles target images and samples.

        If this is the first time a face has been loaded, then it's meta data is extracted from the
        png header and added to :attr:`_face_cache`

        See
        :func:`minibatch_ab` for more details on the output.

        Parameters
        ----------
        filenames: list
            List of full paths to image file names
        side: str
            The side of the model being trained on (`a` or `b`)
        """
        logger.trace("Process batch: (filenames: '%s', side: '%s')", filenames, side)

        if not self._face_cache.cache_full:
            batch = self._face_cache.cache_metadata(filenames)
        else:
            batch = read_image_batch(filenames)

        cache = self._face_cache.get_items(filenames)
        batch, landmarks = self._crop_to_center(filenames, cache, batch, side)
        batch = self._apply_mask(filenames, cache, batch, side)
        processed = dict()

        # Initialize processing training size on first image
        if not self._processing.initialized:
            self._processing.initialize(batch.shape[1])

        # Get Landmarks prior to manipulating the image
        if self._warp_to_landmarks:
            batch_dst_pts = self._get_closest_match(filenames, side, landmarks)
            warp_kwargs = dict(batch_src_points=landmarks, batch_dst_points=batch_dst_pts)
        else:
            warp_kwargs = dict()

        # Color Augmentation of the image only
        if self._augment_color:
            batch[..., :3] = self._processing.color_adjust(batch[..., :3])

        # Random Transform and flip
        batch = self._processing.transform(batch)
        if not self._no_flip:
            batch = self._processing.random_flip(batch)

        # Switch color order for RGB models
        if self._color_order == "rgb":
            batch[..., :3] = batch[..., [2, 1, 0]]

        # Add samples to output if this is for display
        if self._processing.is_display:
            processed["samples"] = batch[..., :3].astype("float32") / 255.0

        # Get Targets
        processed.update(self._processing.get_targets(batch))

        # Random Warp # TODO change masks to have a input mask and a warped target mask
        if self._no_warp:
            processed["feed"] = [self._processing.skip_warp(batch[..., :3])]
        else:
            processed["feed"] = [self._processing.warp(batch[..., :3],
                                                       self._warp_to_landmarks,
                                                       **warp_kwargs)]

        logger.trace("Processed batch: (filenames: %s, side: '%s', processed: %s)",
                     filenames,
                     side,
                     {k: v.shape if isinstance(v, np.ndarray) else[i.shape for i in v]
                      for k, v in processed.items()})
        return processed

    def _crop_to_center(self, filenames, cache, batch, side):
        """ Crops the training image out of the full extract image based on the centering used in
        the user's configuration settings.

        If legacy extract images are being used then this just returns the extracted batch with
        their corresponding landmarks.

        Parameters
        ----------
        filenames: list
            The list of filenames that correspond to this batch
        cache: list
            The list of cached items (aligned faces, masks etc.) corresponding to the batch
        batch: :class:`numpy.ndarray`
            The batch of faces that have been loaded from disk
        side: str
            '"a"' or '"b"' the side that is being processed

        Returns
        -------
        batch: :class:`numpy.ndarray`
            The centered faces cropped out of the loaded batch
        landmarks: :class:`numpy.ndarray`
            The aligned landmarks for this batch. NB: The aligned landmarks do not directly
            correspond to the size of the extracted face. They are scaled to the source training
            image, not the sub-image.

        Raises
        ------
        FaceswapError
            If Alignment information is not available for any of the images being loaded in
            the batch
        """
        logger.trace("Cropping training images info: (filenames: %s, side: '%s')", filenames, side)
        aligned = [item["aligned_face"] for item in cache]

        if self._face_cache.extract_version == 1.0:
            # Legacy extract. Don't crop, just return batch with landmarks
            return batch, np.array([face.landmarks for face in aligned])

        landmarks = np.array([face.landmarks for face in aligned])
        cropped = np.array([align.extract_face(img) for align, img in zip(aligned, batch)])
        return cropped, landmarks

    def _apply_mask(self, filenames, cache, batch, side):
        """ Applies the mask to the 4th channel of the image. If masks are not being used
        applies a dummy all ones mask.

        If the configuration options `eye_multiplier` and/or `mouth_multiplier` are greater than 1
        then these masks are applied to the final channels of the batch respectively.

        Parameters
        ----------
        filenames: list
            The list of filenames that correspond to this batch
        cache: list
            The list of cached items (aligned faces, masks etc.) corresponding to the batch
        batch: :class:`numpy.ndarray`
            The batch of faces that have been loaded from disk
        side: str
            '"a"' or '"b"' the side that is being processed

        Returns
        -------
        :class:`numpy.ndarray`
            The batch with masks applied to the final channels
        """
        logger.trace("Input filenames: %s, batch shape: %s, side: %s",
                     filenames, batch.shape, side)
        size = batch.shape[1]

        for key in ("mask", "mask_eyes", "mask_mouth"):
            lookup = cache[0].get(key)
            if lookup is None and key != "mask":
                continue

            if lookup is None and key == "mask":
                logger.trace("Creating dummy masks. side: %s", side)
                masks = np.ones_like(batch[..., :1], dtype=batch.dtype)
            else:
                logger.trace("Obtaining masks for batch. (key: %s side: %s)", key, side)

                masks = np.array([self._get_mask(item[key], size)
                                  for item in cache], dtype=batch.dtype)
                masks = self._resize_masks(size, masks)
            logger.trace("masks: (key: %s, shape: %s)", key, masks.shape)
            batch = np.concatenate((batch, masks), axis=-1)
        logger.trace("Output batch shape: %s, side: %s", batch.shape, side)
        return batch

    @classmethod
    def _get_mask(cls, item, size):
        """ Decompress zipped eye and mouth masks, or return the stored mask

        Parameters
        ----------
        item: :class:`lib.align.Mask` or `bytes`
            Either a stored face mask object or a zipped eye or mouth mask
        size: int
            The size of the stored eye or mouth mask for reshaping

        Returns
        -------
        class:`numpy.ndarray`
            The decompressed mask
        """
        if isinstance(item, bytes):
            retval = np.frombuffer(decompress(item), dtype="uint8").reshape(size, size, 1)
        else:
            retval = item.mask
        return retval

    @classmethod
    def _resize_masks(cls, target_size, masks):
        """ Resize the masks to the target size """
        logger.trace("target size: %s, masks shape: %s", target_size, masks.shape)
        mask_size = masks.shape[1]
        if target_size == mask_size:
            logger.trace("Mask and targets the same size. Not resizing")
            return masks
        interpolator = cv2.INTER_CUBIC if mask_size < target_size else cv2.INTER_AREA
        masks = np.array([cv2.resize(mask,
                                     (target_size, target_size),
                                     interpolation=interpolator)[..., None]
                          for mask in masks])
        logger.trace("Resized masks: %s", masks.shape)
        return masks

    def _get_closest_match(self, filenames, side, batch_src_points):
        """ Only called if the :attr:`_warp_to_landmarks` is ``True``. Gets the closest
        matched 68 point landmarks from the opposite training set. """
        logger.trace("Retrieving closest matched landmarks: (filenames: '%s', src_points: '%s'",
                     filenames, batch_src_points)
        lm_side = "a" if side == "b" else "b"
        landmarks = _FACE_CACHES[lm_side].aligned_landmarks

        closest_matches = [self._nearest_landmarks.get(os.path.basename(filename))
                           for filename in filenames]
        if None in closest_matches:
            # Resize mismatched training image size landmarks
            sizes = {side: cache.crop_size for side, cache in _FACE_CACHES.items()}
            if len(set(sizes.values())) > 1:
                scale = sizes[side] / sizes[lm_side]
                landmarks = {key: lms * scale for key, lms in landmarks.items()}
            closest_matches = self._cache_closest_matches(filenames, batch_src_points, landmarks)

        batch_dst_points = np.array([landmarks[choice(fname)] for fname in closest_matches])
        logger.trace("Returning: (batch_dst_points: %s)", batch_dst_points.shape)
        return batch_dst_points

    def _cache_closest_matches(self, filenames, batch_src_points, landmarks):
        """ Cache the nearest landmarks for this batch """
        logger.trace("Caching closest matches")
        dst_landmarks = list(landmarks.items())
        dst_points = np.array([lm[1] for lm in dst_landmarks])
        batch_closest_matches = list()

        for filename, src_points in zip(filenames, batch_src_points):
            closest = (np.mean(np.square(src_points - dst_points), axis=(1, 2))).argsort()[:10]
            closest_matches = tuple(dst_landmarks[i][0] for i in closest)
            self._nearest_landmarks[os.path.basename(filename)] = closest_matches
            batch_closest_matches.append(closest_matches)
        logger.trace("Cached closest matches")
        return batch_closest_matches
