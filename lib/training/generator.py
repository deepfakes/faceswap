#!/usr/bin/env python3
""" Handles Data Augmentation for feeding Faceswap Models """

import logging
import os
import sys
from concurrent import futures

from random import shuffle, choice
from typing import cast, Dict, Generator, List, Tuple, TYPE_CHECKING, Union

import cv2
import numpy as np
import numexpr as ne
from lib.align import AlignedFace, DetectedFace
from lib.align.aligned_face import CenteringType
from lib.image import read_image_batch
from lib.multithreading import BackgroundGenerator
from lib.utils import FaceswapError

from . import ImageAugmentation
from .cache import get_cache, RingBuffer

if sys.version_info < (3, 8):
    from typing_extensions import get_args, Literal
else:
    from typing import get_args, Literal

if TYPE_CHECKING:
    from plugins.train.model._base import ModelBase
    from .cache import _Cache

logger = logging.getLogger(__name__)
ConfigType = Dict[str, Union[bool, int, float, str]]  # TODO Dataclass
BatchType = Tuple[np.ndarray, List[np.ndarray]]


class DataGenerator():
    """ Parent class for Training and Preview Data Generators.

    This class is called from :mod:`plugins.train.trainer._base` and launches a background
    iterator that compiles augmented data, target data and sample data.

    Parameters
    ----------
    model: :class:`~plugins.train.model.ModelBase`
        The model that this data generator is feeding
    config: dict
        The configuration `dict` generated from :file:`config.train.ini` containing the trainer
        plugin configuration options.
    side: {'a' or 'b'}
        The side of the model that this iterator is for.
    images: list
        A list of image paths that will be used to compile the final augmented data from.
    batch_size: int
        The batch size for this iterator. Images will be returned in :class:`numpy.ndarray`
        objects of this size from the iterator.
    """
    def __init__(self,
                 config: ConfigType,
                 model: "ModelBase",
                 side: Literal["a", "b"],
                 images: List[str],
                 batch_size: int) -> None:
        logger.debug("Initializing %s: (model: %s, side: %s, images: %s , "  # type: ignore
                     "batch_size: %s, config: %s)", self.__class__.__name__, model.name, side,
                     len(images), batch_size, config)
        self._config = config
        self._side = side
        self._images = images
        self._batch_size = batch_size

        self._process_size = max(img[1] for img in model.input_shapes + model.output_shapes)
        self._output_sizes = self._get_output_sizes(model)

        self._coverage_ratio = model.coverage_ratio
        self._color_order = model.color_order.lower()
        self._use_mask = self._config["mask_type"] and (self._config["penalized_mask_loss"] or
                                                        self._config["learn_mask"])

        self._validate_samples()
        self._buffer = RingBuffer(batch_size,
                                  (self._process_size, self._process_size, self._total_channels),
                                  dtype="uint8")
        self._face_cache: "_Cache" = get_cache(side,
                                               filenames=images,
                                               config=self._config,
                                               size=self._process_size,
                                               coverage_ratio=self._coverage_ratio)
        logger.debug("Initialized %s", self.__class__.__name__)

    @property
    def _total_channels(self) -> int:
        """int: The total number of channels, including mask channels that the target image
        should hold. """
        channels = 3
        if self._config["mask_type"] and (self._config["learn_mask"] or
                                          self._config["penalized_mask_loss"]):
            channels += 1

        mults = [area for area in ["eye", "mouth"] if int(self._config[f"{area}_multiplier"]) > 1]
        if self._config["penalized_mask_loss"] and mults:
            channels += len(mults)
        return channels

    def _get_output_sizes(self, model: "ModelBase") -> List[int]:
        """ Obtain the size of each output tensor for the model.

        Parameters
        ----------
        model: :class:`~plugins.train.model.ModelBase`
            The model that this data generator is feeding

        Returns
        -------
        list
            A list of integers for the model output size for the current side
        """
        out_shapes = model.output_shapes
        split = len(out_shapes) // 2
        side_out = out_shapes[:split] if self._side == "a" else out_shapes[split:]
        retval = [shape[1] for shape in side_out if shape[-1] != 1]
        logger.debug("side: %s, model output shapes: %s, output sizes: %s",
                     self._side, model.output_shapes, retval)
        return retval

    def minibatch_ab(self, do_shuffle: bool = True) -> Generator[BatchType, None, None]:
        """ A Background iterator to return augmented images, samples and targets.

        The exit point from this class and the sole attribute that should be referenced. Called
        from :mod:`plugins.train.trainer._base`. Returns an iterator that yields images for
        training, preview and time-lapses.

        Parameters
        ----------
        do_shuffle: bool, optional
            Whether data should be shuffled prior to loading from disk. If true, each time the full
            list of filenames are processed, the data will be reshuffled to make sure they are not
            returned in the same order. Default: ``True``

        Yields
        ------
        feed: list
            4-dimensional array of faces to feed the training the model (:attr:`x` parameter for
            :func:`keras.models.model.train_on_batch`.). The array returned is in the format
            (`batch size`, `height`, `width`, `channels`).
        targets: list
            List of 4-dimensional :class:`numpy.ndarray` objects in the order and size of each
            output of the model. The format of these arrays will be (`batch size`, `height`,
            `width`, `x`). This is the :attr:`y` parameter for
            :func:`keras.models.model.train_on_batch`. The number of channels here will vary.
            The first 3 channels are (rgb/bgr). The 4th channel is the face mask. Any subsequent
            channels are area masks (e.g. eye/mouth masks)
        """
        logger.debug("do_shuffle: %s", do_shuffle)
        args = (do_shuffle, )
        batcher = BackgroundGenerator(self._minibatch, args=args)
        return batcher.iterator()

    # << INTERNAL METHODS >> #
    def _validate_samples(self) -> None:
        """ Ensures that the total number of images within :attr:`images` is greater or equal to
        the selected :attr:`batch_size`.

        Raises
        ------
        :class:`FaceswapError`
            If the number of images loaded is smaller than the selected batch size
        """
        length = len(self._images)
        msg = ("Number of images is lower than batch-size (Note that too few images may lead to "
               f"bad training). # images: {length}, batch-size: {self._batch_size}")
        try:
            assert length >= self._batch_size, msg
        except AssertionError as err:
            msg += ("\nYou should increase the number of images in your training set or lower "
                    "your batch-size.")
            raise FaceswapError(msg) from err

    def _minibatch(self, do_shuffle: bool) -> Generator[BatchType, None, None]:
        """ A generator function that yields the augmented, target and sample images for the
        current batch on the current side.

        Parameters
        ----------
        do_shuffle: bool, optional
            Whether data should be shuffled prior to loading from disk. If true, each time the full
            list of filenames are processed, the data will be reshuffled to make sure they are not
            returned in the same order. Default: ``True``

        Yields
        ------
        feed: list
            4-dimensional array of faces to feed the training the model (:attr:`x` parameter for
            :func:`keras.models.model.train_on_batch`.). The array returned is in the format
            (`batch size`, `height`, `width`, `channels`).
        targets: list
            List of 4-dimensional :class:`numpy.ndarray` objects in the order and size of each
            output of the model. The format of these arrays will be (`batch size`, `height`,
            `width`, `x`). This is the :attr:`y` parameter for
            :func:`keras.models.model.train_on_batch`. The number of channels here will vary.
            The first 3 channels are (rgb/bgr). The 4th channel is the face mask. Any subsequent
            channels are area masks (e.g. eye/mouth masks)
        """
        logger.debug("Loading minibatch generator: (image_count: %s, do_shuffle: %s)",
                     len(self._images), do_shuffle)

        def _img_iter(imgs):
            """ Infinite iterator for recursing through image list and reshuffling at each epoch"""
            while True:
                if do_shuffle:
                    shuffle(imgs)
                for img in imgs:
                    yield img

        img_iter = _img_iter(self._images[:])
        while True:
            img_paths = [next(img_iter)  # pylint:disable=stop-iteration-return
                         for _ in range(self._batch_size)]
            retval = self._process_batch(img_paths)
            yield retval

    def _get_images_with_meta(self, filenames: List[str]) -> Tuple[np.ndarray, List[DetectedFace]]:
        """ Obtain the raw face images with associated :class:`DetectedFace` objects for this
        batch.

        If this is the first time a face has been loaded, then it's meta data is extracted
        from the png header and added to :attr:`_face_cache`.

        Parameters
        ----------
        filenames: list
            List of full paths to image file names

        Returns
        -------
        raw_faces: :class:`numpy.ndarray`
            The full sized batch of training images for the given filenames
        list
            Batch of :class:`~lib.align.DetectedFace` objects for the given filename including the
            aligned face objects for the model output size
        """
        if not self._face_cache.cache_full:
            raw_faces = self._face_cache.cache_metadata(filenames)
        else:
            raw_faces = read_image_batch(filenames)

        detected_faces = self._face_cache.get_items(filenames)
        logger.trace("filenames: %s, raw_faces: '%s', detected_faces: %s",  # type: ignore
                     filenames, raw_faces.shape, len(detected_faces))
        return raw_faces, detected_faces

    def _crop_to_coverage(self,
                          filenames: List[str],
                          images: np.ndarray,
                          detected_faces: List[DetectedFace],
                          batch: np.ndarray) -> None:
        """ Crops the training image out of the full extract image based on the centering and
        coveage used in the user's configuration settings.

        If legacy extract images are being used then this just returns the extracted batch with
        their corresponding landmarks.

        Uses thread pool execution for about a 33% speed increase @ 64 batch size

        Parameters
        ----------
        filenames: list
            The list of filenames that correspond to this batch
        images: :class:`numpy.ndarray`
            The batch of faces that have been loaded from disk
        detected_faces: list
            The list of :class:`lib.align.DetectedFace` items corresponding to the batch
        batch: :class:`np.ndarray`
            The pre-allocated array to hold this batch
        """
        logger.trace("Cropping training images info: (filenames: %s, side: '%s')",  # type: ignore
                     filenames, self._side)

        with futures.ThreadPoolExecutor() as executor:
            proc = {executor.submit(face.aligned.extract_face, img): idx
                    for idx, (face, img) in enumerate(zip(detected_faces, images))}

            for future in futures.as_completed(proc):
                batch[proc[future], ..., :3] = future.result()

    def _apply_mask(self, detected_faces: List[DetectedFace], batch: np.ndarray) -> None:
        """ Applies the masks to the 4th channel of the batch.

        If the configuration options `eye_multiplier` and/or `mouth_multiplier` are greater than 1
        then these masks are applied to the final channels of the batch respectively.

        If masks are not being used then this function returns having done nothing

        Parameters
        ----------
        detected_face: list
            The list of :class:`~lib.align.DetectedFace` objects corresponding to the batch
        batch: :class:`numpy.ndarray`
            The preallocated array to apply masks to
        side: str
            '"a"' or '"b"' the side that is being processed
        """
        if not self._use_mask:
            return

        masks = np.array([face.get_training_masks() for face in detected_faces])
        batch[..., 3:] = masks

        logger.trace("side: %s, masks: %s, batch: %s",  # type: ignore
                     self._side, masks.shape, batch.shape)

    def _process_batch(self, filenames: List[str]) -> BatchType:
        """ Prepares data for feeding through subclassed methods.

        If this is the first time a face has been loaded, then it's meta data is extracted from the
        png header and added to :attr:`_face_cache`

        Parameters
        ----------
        filenames: list
            List of full paths to image file names for a single batch

        Returns
        -------
        list
            4-dimensional array of faces to feed the training the model.
        list
            List of 4-dimensional :class:`numpy.ndarray`. The number of channels here will vary.
            The first 3 channels are (rgb/bgr). The 4th channel is the face mask. Any subsequent
            channels are area masks (e.g. eye/mouth masks)
        """
        raw_faces, detected_faces = self._get_images_with_meta(filenames)
        batch = self._buffer()
        self._crop_to_coverage(filenames, raw_faces, detected_faces, batch)
        self._apply_mask(detected_faces, batch)
        return self.process_batch(filenames, raw_faces, detected_faces, batch)

    def process_batch(self,
                      filenames: List[str],
                      images: np.ndarray,
                      detected_faces: List[DetectedFace],
                      batch: np.ndarray) -> BatchType:
        """ Override for processing the batch for the current generator.

        Parameters
        ----------
        filenames: list
            List of full paths to image file names for a single batch
        images: :class:`numpy.ndarray`
            The batch of faces corresponding to the filenames
        detected_faces: list
            List of :class:`~lib.align.DetectedFace` objects with aligned data and masks loaded for
            the current batch
        batch: :class:`numpy.ndarray`
            The pre-allocated batch with images and masks populated for the selected coverage and
            centering

        Returns
        -------
        list
            4-dimensional array of faces to feed the training the model.
        list
            List of 4-dimensional :class:`numpy.ndarray`. The number of channels here will vary.
            The first 3 channels are (rgb/bgr). The 4th channel is the face mask. Any subsequent
            channels are area masks (e.g. eye/mouth masks)
        """
        raise NotImplementedError()

    def _set_color_order(self, batch) -> None:
        """ Set the color order correctly for the model's input type.

        batch: :class:`numpy.ndarray`
            The pre-allocated batch with images in the first 3 channels in BGR order
        """
        if self._color_order == "rgb":
            batch[..., :3] = batch[..., [2, 1, 0]]

    def _to_float32(self, in_array: np.ndarray) -> np.ndarray:
        """ Cast an UINT8 array in 0-255 range to float32 in 0.0-1.0 range.

        in_array: :class:`numpy.ndarray`
            The input uint8 array
        """
        return ne.evaluate("x / c",
                           local_dict=dict(x=in_array, c=np.float32(255)),
                           casting="unsafe")


class TrainingDataGenerator(DataGenerator):  # pylint:disable=too-few-public-methods
    """ A Training Data Generator for compiling data for feeding to a model.

    This class is called from :mod:`plugins.train.trainer._base` and launches a background
    iterator that compiles augmented data, target data and sample data.

    Parameters
    ----------
    model: :class:`~plugins.train.model.ModelBase`
        The model that this data generator is feeding
    config: dict
        The configuration `dict` generated from :file:`config.train.ini` containing the trainer
        plugin configuration options.
    side: {'a' or 'b'}
        The side of the model that this iterator is for.
    images: list
        A list of image paths that will be used to compile the final augmented data from.
    batch_size: int
        The batch size for this iterator. Images will be returned in :class:`numpy.ndarray`
        objects of this size from the iterator.
    """
    def __init__(self,
                 config: ConfigType,
                 model: "ModelBase",
                 side: Literal["a", "b"],
                 images: List[str],
                 batch_size: int) -> None:
        super().__init__(config, model, side, images, batch_size)
        self._augment_color = not model.command_line_arguments.no_augment_color
        self._no_flip = model.command_line_arguments.no_flip
        self._no_warp = model.command_line_arguments.no_warp
        self._warp_to_landmarks = (not self._no_warp
                                   and model.command_line_arguments.warp_to_landmarks)
        self._model_input_size = max(img[1] for img in model.input_shapes)

        if self._warp_to_landmarks:
            self._face_cache.pre_fill(images, side)
        self._processing = ImageAugmentation(batch_size,
                                             self._process_size,
                                             self._config)
        self._nearest_landmarks: Dict[str, Tuple[str, ...]] = {}
        logger.debug("Initialized %s", self.__class__.__name__)

    def _create_targets(self, batch: np.ndarray) -> List[np.ndarray]:
        """ Compile target images, with masks, for the model output sizes.

        Parameters
        ----------
        batch: :class:`numpy.ndarray`
            This should be a 4-dimensional array of training images in the format (`batch size`,
            `height`, `width`, `channels`). Targets should be requested after performing image
            transformations but prior to performing warps. The 4th channel should be the mask.
            Any channels above the 4th should be any additional area masks (e.g. eye/mouth) that
            are required.

        Returns
        -------
        list
            List of 4-dimensional target images, at all model output sizes, with masks compiled
            into channels 4+ for each output size
        """
        logger.trace("Compiling targets: batch shape: %s", batch.shape)  # type: ignore
        if len(self._output_sizes) == 1 and self._output_sizes[0] == self._process_size:
            # Rolling buffer here makes next to no difference, so just create array on the fly
            retval = [self._to_float32(batch)]
        else:
            retval = [self._to_float32(np.array([cv2.resize(image, (size, size), cv2.INTER_AREA)
                                                 for image in batch]))
                      for size in self._output_sizes]
        logger.trace("Processed targets: %s", [t.shape for t in retval])  # type: ignore
        return retval

    def process_batch(self,
                      filenames: List[str],
                      images: np.ndarray,
                      detected_faces: List[DetectedFace],
                      batch: np.ndarray) -> BatchType:
        """ Performs the augmentation and compiles target images and samples.

        Parameters
        ----------
        filenames: list
            List of full paths to image file names for a single batch
        images: :class:`numpy.ndarray`
            The batch of faces corresponding to the filenames
        detected_faces: list
            List of :class:`~lib.align.DetectedFace` objects with aligned data and masks loaded for
            the current batch
        batch: :class:`numpy.ndarray`
            The pre-allocated batch with images and masks populated for the selected coverage and
            centering

        Returns
        -------
        feed: list
            4-dimensional array of faces to feed the training the model (:attr:`x` parameter for
            :func:`keras.models.model.train_on_batch`.). The array returned is in the format
            (`batch size`, `height`, `width`, `channels`).
        targets: list
            List of 4-dimensional :class:`numpy.ndarray` objects in the order and size of each
            output of the model. The format of these arrays will be (`batch size`, `height`,
            `width`, `x`). This is the :attr:`y` parameter for
            :func:`keras.models.model.train_on_batch`. The number of channels here will vary.
            The first 3 channels are (rgb/bgr). The 4th channel is the face mask. Any subsequent
            channels are area masks (e.g. eye/mouth masks)
        """
        logger.trace("Process training: (side: '%s', filenames: '%s', images: %s, "  # type:ignore
                     "batch: %s, detected_faces: %s)", self._side, filenames, images.shape,
                     batch.shape, len(detected_faces))

        # Color Augmentation of the image only
        if self._augment_color:
            batch[..., :3] = self._processing.color_adjust(batch[..., :3])

        # Random Transform and flip
        self._processing.transform(batch)

        if not self._no_flip:
            self._processing.random_flip(batch)

        # Switch color order for RGB models
        self._set_color_order(batch)

        # Get Targets
        targets = self._create_targets(batch)

        # TODO Look at potential for applying mask on input
        # Random Warp
        if self._warp_to_landmarks:
            landmarks = np.array([face.aligned.landmarks for face in detected_faces])
            batch_dst_pts = self._get_closest_match(filenames, landmarks)
            warp_kwargs = dict(batch_src_points=landmarks, batch_dst_points=batch_dst_pts)
        else:
            warp_kwargs = {}

        warped = batch[..., :3] if self._no_warp else self._processing.warp(
            batch[..., :3],
            self._warp_to_landmarks,
            **warp_kwargs)

        if self._model_input_size != self._process_size:
            feed = self._to_float32(np.array([cv2.resize(image,
                                                         (self._model_input_size,
                                                          self._model_input_size),
                                                         cv2.INTER_AREA)
                                              for image in warped]))
        else:
            feed = self._to_float32(warped)

        logger.trace("Processed batch: (filenames: %s, side: '%s', "  # type: ignore
                     "feed: %s, targets: %s)", filenames, self._side,
                     [f.shape for f in feed], [t.shape for t in targets])

        return feed, targets

    def _get_closest_match(self, filenames: List[str], batch_src_points: np.ndarray) -> np.ndarray:
        """ Only called if the :attr:`_warp_to_landmarks` is ``True``. Gets the closest
        matched 68 point landmarks from the opposite training set.

        Parameters
        ----------
        filenames: list
            Filenames for current batch
        batch_src_points: :class:`np.ndarray`
            The source landmarks for the current batch

        Returns
        -------
        :class:`np.ndarray`
            Randomly selected closest matches from the other side's landmarks
        """
        logger.trace("Retrieving closest matched landmarks: (filenames: '%s', "  # type: ignore
                     "src_points: '%s')", filenames, batch_src_points)
        lm_side: Literal["a", "b"] = "a" if self._side == "b" else "b"
        other_cache = get_cache(lm_side)
        landmarks = other_cache.aligned_landmarks

        try:
            closest_matches = [self._nearest_landmarks[os.path.basename(filename)]
                               for filename in filenames]
        except KeyError:
            # Resize mismatched training image size landmarks
            sizes = {side: cache.size for side, cache in zip((self._side, lm_side),
                                                             (self._face_cache, other_cache))}
            if len(set(sizes.values())) > 1:
                scale = sizes[self._side] / sizes[lm_side]
                landmarks = {key: lms * scale for key, lms in landmarks.items()}
            closest_matches = self._cache_closest_matches(filenames, batch_src_points, landmarks)

        batch_dst_points = np.array([landmarks[choice(fname)] for fname in closest_matches])
        logger.trace("Returning: (batch_dst_points: %s)", batch_dst_points.shape)  # type: ignore
        return batch_dst_points

    def _cache_closest_matches(self,
                               filenames: List[str],
                               batch_src_points: np.ndarray,
                               landmarks: Dict[str, np.ndarray]) -> List[Tuple[str, ...]]:
        """ Cache the nearest landmarks for this batch

        Parameters
        ----------
        filenames: list
            Filenames for current batch
        batch_src_points: :class:`np.ndarray`
            The source landmarks for the current batch
        landmarks: dict
            The destination landmarks with associated filenames

        """
        logger.trace("Caching closest matches")  # type:ignore
        dst_landmarks = list(landmarks.items())
        dst_points = np.array([lm[1] for lm in dst_landmarks])
        batch_closest_matches: List[Tuple[str, ...]] = []

        for filename, src_points in zip(filenames, batch_src_points):
            closest = (np.mean(np.square(src_points - dst_points), axis=(1, 2))).argsort()[:10]
            closest_matches = tuple(dst_landmarks[i][0] for i in closest)
            self._nearest_landmarks[os.path.basename(filename)] = closest_matches
            batch_closest_matches.append(closest_matches)
        logger.trace("Cached closest matches")  # type:ignore
        return batch_closest_matches


class PreviewDataGenerator(DataGenerator):
    """ Generator for compiling images for generating previews.

    This class is called from :mod:`plugins.train.trainer._base` and launches a background
    iterator that compiles sample preview data for feeding the model's predict function and for
    display.

    Parameters
    ----------
    model: :class:`~plugins.train.model.ModelBase`
        The model that this data generator is feeding
    config: dict
        The configuration `dict` generated from :file:`config.train.ini` containing the trainer
        plugin configuration options.
    side: {'a' or 'b'}
        The side of the model that this iterator is for.
    images: list
        A list of image paths that will be used to compile the final images.
    batch_size: int
        The batch size for this iterator. Images will be returned in :class:`numpy.ndarray`
        objects of this size from the iterator.
    """
    def _create_samples(self,
                        images: np.ndarray,
                        detected_faces: List[DetectedFace]) -> List[np.ndarray]:
        """ Compile the 'sample' images. These are the 100% coverage images which hold the model
        output in the preview window.

        Parameters
        ----------
        images: :class:`numpy.ndarray`
            The original batch of images as loaded from disk.
        detected_faces: list
            List of :class:`~lib.align.DetectedFace` for the current batch

        Returns
        -------
        list
            List of 4-dimensional target images, at final model output size
        """
        logger.trace("Compiling samples: images shape: %s, detected_faces: %s ",  # type: ignore
                     images.shape, len(detected_faces))
        output_size = self._output_sizes[-1]
        full_size = 2 * int(np.rint((output_size / self._coverage_ratio) / 2))

        assert self._config["centering"] in get_args(CenteringType)
        retval = np.empty((full_size, full_size, 3), dtype="float32")
        retval = self._to_float32(np.array([AlignedFace(face.landmarks_xy,
                                                        image=images[idx],
                                                        centering=cast(CenteringType,
                                                                       self._config["centering"]),
                                                        size=full_size,
                                                        dtype="uint8",
                                                        is_aligned=True).face
                                            for idx, face in enumerate(detected_faces)]))

        logger.trace("Processed samples: %s", retval.shape)  # type: ignore
        return [retval]

    def process_batch(self,
                      filenames: List[str],
                      images: np.ndarray,
                      detected_faces: List[DetectedFace],
                      batch: np.ndarray) -> BatchType:
        """ Creates the full size preview images and the sub-cropped images for feeding the model's
        predict function.

        Parameters
        ----------
        filenames: list
            List of full paths to image file names for a single batch
        images: :class:`numpy.ndarray`
            The batch of faces corresponding to the filenames
        detected_faces: list
            List of :class:`~lib.align.DetectedFace` objects with aligned data and masks loaded for
            the current batch
        batch: :class:`numpy.ndarray`
            The pre-allocated batch with images and masks populated for the selected coverage and
            centering

        Returns
        -------
        feed: list
            List of 4-dimensional :class:`numpy.ndarray` objects at model input size for feeding
            the model's predict function. The first 3 channels are (rgb/bgr). The 4th channel is
            the face mask.
        samples: list
            4-dimensional array containing the 100% coverage images at the model's centering for
            for generating previews. The array returned is in the format
            (`batch size`, `height`, `width`, `channels`).
        """
        logger.trace("Process preview: (side: '%s', filenames: '%s', images: %s, "  # type:ignore
                     "batch: %s, detected_faces: %s)", self._side, filenames, images.shape,
                     batch.shape, len(detected_faces))

        self._set_color_order(batch)  # Switch color order for RGB models

        if not self._use_mask:
            mask = np.zeros_like(batch[..., 0])[..., None] + 255
            batch = np.concatenate([batch, mask], axis=-1)

        feed = self._to_float32(batch[..., :4])

        samples = self._create_samples(images, detected_faces)

        logger.trace("Processed batch: (filenames: %s, side: '%s', "  # type: ignore
                     "feed: %s, targets: %s)", filenames, self._side,
                     [f.shape for f in feed], [t.shape for t in samples])
        return feed, samples
