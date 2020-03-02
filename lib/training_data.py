#!/usr/bin/env python3
""" Handles Data Augmentation for feeding Faceswap Models """

import logging

from random import shuffle, choice

import numpy as np
import cv2
from scipy.interpolate import griddata

from lib.image import batch_convert_color, read_image_batch
from lib.multithreading import BackgroundGenerator
from lib.utils import FaceswapError

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class TrainingDataGenerator():
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
    training_opts: dict
        This is a dictionary of model training options as defined in
        :mod:`plugins.train.model._base`. These options will be defined by the user from the
        provided cli options or from the model ``config.ini``. At a minimum this ``dict`` should
        contain the following keys:

        * **coverage_ratio** (`float`) - The ratio of the training image to be trained on. \
        Dictates how much of the image will be cropped out. E.G: a coverage ratio of 0.625 \
        will result in cropping a 160px box from a 256px image (256 * 0.625 = 160).

        * **augment_color** (`bool`) - ``True`` if color is to be augmented, otherwise ``False`` \

        * **no_flip** (`bool`) - ``True`` if the image shouldn't be randomly flipped as part of \
        augmentation, otherwise ``False``

        * **warp_to_landmarks** (`bool`) - ``True`` if the random warp method should warp to \
        similar landmarks from the other side, ``False`` if the standard random warp method \
        should be used. If ``True`` then the additional key ``landmarks`` must be provided.

        * **landmarks** (`dict`, `optional`). Required if :attr:`warp_to_landmarks` is \
        ``True``. Returning dictionary has a key of **side** (`str`) the value of which is a \
        `dict` of {**filename** (`str`): **68 point landmarks** (`numpy.ndarray`)}.

        * **masks** (`dict`, `optional`). Required if :attr:`penalized_mask_loss` or \
        :attr:`learn_mask` is ``True``. Returning dictionary has a key of **side** (`str`) the \
        value of which is a `dict` of {**filename** (`str`): :class:`lib.faces_detect.Mask`}.

    config: dict
        The configuration ``dict`` generated from :file:`config.train.ini` containing the trainer \
        plugin configuration options.
    """
    def __init__(self, model_input_size, model_output_shapes, training_opts, config):
        logger.debug("Initializing %s: (model_input_size: %s, model_output_shapes: %s, "
                     "training_opts: %s, landmarks: %s, masks: %s, config: %s)",
                     self.__class__.__name__, model_input_size, model_output_shapes,
                     {key: val
                      for key, val in training_opts.items() if key not in ("landmarks", "masks")},
                     {key: len(val)
                      for key, val in training_opts.get("landmarks", dict()).items()},
                     {key: len(val) for key, val in training_opts.get("masks", dict()).items()},
                     config)
        self._config = config
        self._model_input_size = model_input_size
        self._model_output_shapes = model_output_shapes
        self._training_opts = training_opts
        self._landmarks = self._training_opts.get("landmarks", None)
        self._masks = self._training_opts.get("masks", None)
        self._nearest_landmarks = {}

        # Batchsize and processing class are set when this class is called by a batcher
        # from lib.training_data
        self._batchsize = 0
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
            The batchsize for this iterator. Images will be returned in ``numpy.ndarray`` s of
            this size from the iterator.
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
            The following items are contained in each ``dict`` yielded from this iterator:

            * **feed** (`numpy.ndarray`) - The feed for the model. The array returned is in the \
            format (`batchsize`, `height`, `width`, `channels`). This is the :attr:`x` parameter \
            for :func:`keras.models.model.train_on_batch`.

            * **targets** (`list`) - A list of 4-dimensional ``numpy.ndarray`` s in the order \
            and size of each output of the model as defined in :attr:`model_output_shapes`. the \
            format of these arrays will be (`batchsize`, `height`, `width`, `3`). This is \
            the :attr:`y` parameter for :func:`keras.models.model.train_on_batch` **NB:** \
            masks are not included in the ``targets`` list. If required for feeding into the \
            Keras model, they will need to be added to this list in \
            :mod:`plugins.train.trainer._base` from the ``masks`` key.

            * **masks** (`numpy.ndarray`) - A 4-dimensional array containing the target masks in \
            the format (`batchsize`, `height`, `width`, `1`).

            * **samples** (`numpy.ndarray`) - A 4-dimensional array containing the samples for \
            feeding to the model's predict function for generating preview and time-lapse \
            samples. The array will be in the format (`batchsize`, `height`, `width`, \
            `channels`). **NB:** This item will only exist in the ``dict`` if :attr:`is_preview` \
            or :attr:`is_timelapse` is ``True``
        """
        logger.debug("Queue batches: (image_count: %s, batchsize: %s, side: '%s', do_shuffle: %s, "
                     "is_preview, %s, is_timelapse: %s)", len(images), batchsize, side, do_shuffle,
                     is_preview, is_timelapse)
        self._batchsize = batchsize
        self._processing = ImageAugmentation(batchsize,
                                             is_preview or is_timelapse,
                                             self._model_input_size,
                                             self._model_output_shapes,
                                             self._training_opts.get("coverage_ratio", 0.625),
                                             self._config)
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
        """ Performs the augmentation and compiles target images and samples. See
        :func:`minibatch_ab` for more details on the output. """
        logger.trace("Process batch: (filenames: '%s', side: '%s')", filenames, side)
        batch = read_image_batch(filenames)
        batch = self._apply_mask(filenames, batch, side)
        processed = dict()

        # Initialize processing training size on first image
        if not self._processing.initialized:
            self._processing.initialize(batch.shape[1])

        # Get Landmarks prior to manipulating the image
        if self._training_opts["warp_to_landmarks"]:
            batch_src_pts = self._get_landmarks(filenames, side)
            batch_dst_pts = self._get_closest_match(filenames, side, batch_src_pts)
            warp_kwargs = dict(batch_src_points=batch_src_pts,
                               batch_dst_points=batch_dst_pts)
        else:
            warp_kwargs = dict()

        # Color Augmentation of the image only
        if self._training_opts["augment_color"]:
            batch[..., :3] = self._processing.color_adjust(batch[..., :3])

        # Random Transform and flip
        batch = self._processing.transform(batch)
        if not self._training_opts["no_flip"]:
            batch = self._processing.random_flip(batch)

        # Add samples to output if this is for display
        if self._processing.is_display:
            processed["samples"] = batch[..., :3].astype("float32") / 255.0

        # Get Targets
        processed.update(self._processing.get_targets(batch))

        # Random Warp # TODO change masks to have a input mask and a warped target mask
        processed["feed"] = [self._processing.warp(batch[..., :3],
                                                   self._training_opts["warp_to_landmarks"],
                                                   **warp_kwargs)]

        logger.trace("Processed batch: (filenames: %s, side: '%s', processed: %s)",
                     filenames,
                     side,
                     {k: v.shape if isinstance(v, np.ndarray) else[i.shape for i in v]
                      for k, v in processed.items()})

        return processed

    def _apply_mask(self, filenames, batch, side):
        """ Applies the mask to the 4th channel of the image. If masks are not being used
        applies a dummy all ones mask """
        logger.trace("Input batch shape: %s, side: %s", batch.shape, side)
        if self._masks is None:
            logger.trace("Creating dummy masks. side: %s", side)
            masks = np.ones_like(batch[..., :1], dtype=batch.dtype)
        else:
            logger.trace("Obtaining masks for batch. side: %s", side)
            masks = np.array([self._masks[side][filename].mask
                              for filename, face in zip(filenames, batch)], dtype=batch.dtype)
            masks = self._resize_masks(batch.shape[1], masks)

        logger.trace("masks shape: %s", masks.shape)
        batch = np.concatenate((batch, masks), axis=-1)
        logger.trace("Output batch shape: %s, side: %s", batch.shape, side)
        return batch

    @staticmethod
    def _resize_masks(target_size, masks):
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

    def _get_landmarks(self, filenames, side):
        """ Obtains the 68 Point Landmarks for the images in this batch. This is only called if
        config item ``warp_to_landmarks`` is ``True``. If the landmarks for an image cannot be
        found, then an error is raised. """
        logger.trace("Retrieving landmarks: (filenames: %s, side: '%s')", filenames, side)
        src_points = [self._landmarks[side].get(filename, None) for filename in filenames]
        # Raise error on missing alignments
        if not all(isinstance(pts, np.ndarray) for pts in src_points):
            missing = [filenames[idx] for idx, pts in enumerate(src_points) if pts is None]
            msg = ("Files missing alignments for this batch: {}"
                   "\nAt least one of your images does not have a matching entry in your "
                   "alignments file."
                   "\nIf you are using 'warp to landmarks' then every "
                   "face you intend to train on must exist within the alignments file."
                   "\nThe specific files that caused this failure are listed above."
                   "\nMost likely there will be more than just these files missing from the "
                   "alignments file. You can use the Alignments Tool to help identify missing "
                   "alignments".format(missing))
            raise FaceswapError(msg)

        logger.trace("Returning: (src_points: %s)", [str(src) for src in src_points])
        return np.array(src_points)

    def _get_closest_match(self, filenames, side, batch_src_points):
        """ Only called if the config item ``warp_to_landmarks`` is ``True``. Gets the closest
        matched 68 point landmarks from the opposite training set. """
        logger.trace("Retrieving closest matched landmarks: (filenames: '%s', src_points: '%s'",
                     filenames, batch_src_points)
        landmarks = self._landmarks["a"] if side == "b" else self._landmarks["b"]
        closest_hashes = [self._nearest_landmarks.get(filename) for filename in filenames]
        if None in closest_hashes:
            closest_hashes = self._cache_closest_hashes(filenames, batch_src_points, landmarks)

        batch_dst_points = np.array([landmarks[choice(hsh)] for hsh in closest_hashes])
        logger.trace("Returning: (batch_dst_points: %s)", batch_dst_points.shape)
        return batch_dst_points

    def _cache_closest_hashes(self, filenames, batch_src_points, landmarks):
        """ Cache the nearest landmarks for this batch """
        logger.trace("Caching closest hashes")
        dst_landmarks = list(landmarks.items())
        dst_points = np.array([lm[1] for lm in dst_landmarks])
        batch_closest_hashes = list()

        for filename, src_points in zip(filenames, batch_src_points):
            closest = (np.mean(np.square(src_points - dst_points), axis=(1, 2))).argsort()[:10]
            closest_hashes = tuple(dst_landmarks[i][0] for i in closest)
            self._nearest_landmarks[filename] = closest_hashes
            batch_closest_hashes.append(closest_hashes)
        logger.trace("Cached closest hashes")
        return batch_closest_hashes


class ImageAugmentation():
    """ Performs augmentation on batches of training images.

    Parameters
    ----------
    batchsize: int
        The number of images that will be fed through the augmentation functions at once.
    is_display: bool
        Whether the images being fed through will be used for Preview or Time-lapse. Disables
        the "warp" augmentation for these images.
    input_size: int
        The expected input size for the model. It is assumed that the input to the model is always
        a square image. This is the size, in pixels, of the `width` and the `height` of the input
        to the model.
    output_shapes: list
        A list of tuples defining the output shapes from the model, in the order that the outputs
        are returned. The tuples should be in (`height`, `width`, `channels`) format.
    coverage_ratio: float
        The ratio of the training image to be trained on. Dictates how much of the image will be
        cropped out. E.G: a coverage ratio of 0.625 will result in cropping a 160px box from a "
        "256px image (256 * 0.625 = 160).
    config: dict
        The configuration ``dict`` generated from :file:`config.train.ini` containing the trainer \
        plugin configuration options.

    Attributes
    ----------
    initialized: bool
        Flag to indicate whether :class:`ImageAugmentation` has been initialized with the training
        image size in order to cache certain augmentation operations (see :func:`initialize`)
    is_display: bool
        Flag to indicate whether these augmentations are for time-lapses/preview images (``True``)
        or standard training data (``False)``
    """
    def __init__(self, batchsize, is_display, input_size, output_shapes, coverage_ratio, config):
        logger.debug("Initializing %s: (batchsize: %s, is_display: %s, input_size: %s, "
                     "output_shapes: %s, coverage_ratio: %s, config: %s)",
                     self.__class__.__name__, batchsize, is_display, input_size, output_shapes,
                     coverage_ratio, config)

        self.initialized = False
        self.is_display = is_display

        # Set on first image load from initialize
        self._training_size = 0
        self._constants = None

        self._batchsize = batchsize
        self._config = config
        # Transform and Warp args
        self._input_size = input_size
        self._output_sizes = [shape[1] for shape in output_shapes if shape[2] == 3]
        logger.debug("Output sizes: %s", self._output_sizes)
        # Warp args
        self._coverage_ratio = coverage_ratio
        self._scale = 5  # Normal random variable scale

        logger.debug("Initialized %s", self.__class__.__name__)

    def initialize(self, training_size):
        """ Initializes the caching of constants for use in various image augmentations.

        The training image size is not known prior to loading the images from disk and commencing
        training, so it cannot be set in the ``__init__`` method. When the first training batch is
        loaded this function should be called to initialize the class and perform various
        calculations based on this input size to cache certain constants for image augmentation
        calculations.

        Parameters
        ----------
        training_size: int
             The size of the training images stored on disk that are to be fed into
             :class:`ImageAugmentation`. The training images should always be square and of the
             same size. This is the size, in pixels, of the `width` and the `height` of the
             training images.
         """
        logger.debug("Initializing constants. training_size: %s", training_size)
        self._training_size = training_size
        coverage = int(self._training_size * self._coverage_ratio)

        # Color Aug
        clahe_base_contrast = training_size // 128
        # Target Images
        tgt_slices = slice(self._training_size // 2 - coverage // 2,
                           self._training_size // 2 + coverage // 2)

        # Random Warp
        warp_range_ = np.linspace(self._training_size // 2 - coverage // 2,
                                  self._training_size // 2 + coverage // 2, 5, dtype='float32')
        warp_mapx = np.broadcast_to(warp_range_, (self._batchsize, 5, 5)).astype("float32")
        warp_mapy = np.broadcast_to(warp_mapx[0].T, (self._batchsize, 5, 5)).astype("float32")

        warp_pad = int(1.25 * self._input_size)
        warp_slices = slice(warp_pad // 10, -warp_pad // 10)

        # Random Warp Landmarks
        p_mx = self._training_size - 1
        p_hf = (self._training_size // 2) - 1
        edge_anchors = np.array([(0, 0), (0, p_mx), (p_mx, p_mx), (p_mx, 0),
                                 (p_hf, 0), (p_hf, p_mx), (p_mx, p_hf), (0, p_hf)]).astype("int32")
        edge_anchors = np.broadcast_to(edge_anchors, (self._batchsize, 8, 2))
        grids = np.mgrid[0:p_mx:complex(self._training_size), 0:p_mx:complex(self._training_size)]

        self._constants = dict(clahe_base_contrast=clahe_base_contrast,
                               tgt_slices=tgt_slices,
                               warp_mapx=warp_mapx,
                               warp_mapy=warp_mapy,
                               warp_pad=warp_pad,
                               warp_slices=warp_slices,
                               warp_lm_edge_anchors=edge_anchors,
                               warp_lm_grids=grids)
        self.initialized = True
        logger.debug("Initialized constants: %s", {k: str(v) if isinstance(v, np.ndarray) else v
                                                   for k, v in self._constants.items()})

    # <<< TARGET IMAGES >>> #
    def get_targets(self, batch):
        """ Returns the target images, and masks, if required.

        Parameters
        ----------
        batch: numpy.ndarray
            This should be a 4-dimensional array of training images in the format (`batchsize`,
            `height`, `width`, `channels`). Targets should be requested after performing image
            transformations but prior to performing warps.

        Returns
        -------
        dict
            The following keys will be within the returned dictionary:

            * **targets** (`list`) - A list of 4-dimensional ``numpy.ndarray`` s in the order \
            and size of each output of the model as defined in :attr:`output_shapes`. The \
            format of these arrays will be (`batchsize`, `height`, `width`, `3`). **NB:** \
            masks are not included in the ``targets`` list. If masks are to be included in the \
            output they will be returned as their own item from the ``masks`` key.

            * **masks** (`numpy.ndarray`) - A 4-dimensional array containing the target masks in \
            the format (`batchsize`, `height`, `width`, `1`).
        """
        logger.trace("Compiling targets")
        slices = self._constants["tgt_slices"]
        target_batch = [np.array([cv2.resize(image[slices, slices, :],
                                             (size, size),
                                             cv2.INTER_AREA)
                                  for image in batch], dtype='float32') / 255.
                        for size in self._output_sizes]
        logger.trace("Target image shapes: %s",
                     [tgt_images.shape[1:] for tgt_images in target_batch])

        retval = self._separate_target_mask(target_batch)
        logger.trace("Final targets: %s",
                     {k: v.shape if isinstance(v, np.ndarray) else [img.shape for img in v]
                      for k, v in retval.items()})
        return retval

    @staticmethod
    def _separate_target_mask(target_batch):
        """ Return the batch and the batch of final masks

        Returns the targets as a list of 4-dimensional ``numpy.ndarray`` s of shape (`batchsize`,
        `height`, `width`, 3).

        The target masks are returned as its own item and is the 4th channel of the final target
        output.
        """
        logger.trace("target_batch shapes: %s", [tgt.shape for tgt in target_batch])
        retval = dict(targets=[batch[..., :3] for batch in target_batch],
                      masks=[target_batch[-1][..., 3:]])
        logger.trace("returning: %s", {k: [tgt.shape for tgt in v] for k, v in retval.items()})
        return retval

    # <<< COLOR AUGMENTATION >>> #
    def color_adjust(self, batch):
        """ Perform color augmentation on the passed in batch.

        The color adjustment parameters are set in :file:`config.train.ini`

        Parameters
        ----------
        batch: numpy.ndarray
            The batch should be a 4-dimensional array of shape (`batchsize`, `height`, `width`,
            `3`) and in `BGR` format.

        Returns
        ----------
        numpy.ndarray
            A 4-dimensional array of the same shape as :attr:`batch` with color augmentation
            applied.
        """
        if not self.is_display:
            logger.trace("Augmenting color")
            batch = batch_convert_color(batch, "BGR2LAB")
            batch = self._random_clahe(batch)
            batch = self._random_lab(batch)
            batch = batch_convert_color(batch, "LAB2BGR")
        return batch

    def _random_clahe(self, batch):
        """ Randomly perform Contrast Limited Adaptive Histogram Equalization on
        a batch of images """
        base_contrast = self._constants["clahe_base_contrast"]

        batch_random = np.random.rand(self._batchsize)
        indices = np.where(batch_random > self._config.get("color_clahe_chance", 50) / 100)[0]

        grid_bases = np.rint(np.random.uniform(0,
                                               self._config.get("color_clahe_max_size", 4),
                                               size=indices.shape[0])).astype("uint8")
        contrast_adjustment = (grid_bases * (base_contrast // 2))
        grid_sizes = contrast_adjustment + base_contrast
        logger.trace("Adjusting Contrast. Grid Sizes: %s", grid_sizes)

        clahes = [cv2.createCLAHE(clipLimit=2.0,  # pylint: disable=no-member
                                  tileGridSize=(grid_size, grid_size))
                  for grid_size in grid_sizes]

        for idx, clahe in zip(indices, clahes):
            batch[idx, :, :, 0] = clahe.apply(batch[idx, :, :, 0])
        return batch

    def _random_lab(self, batch):
        """ Perform random color/lightness adjustment in L*a*b* color space on a batch of
        images """
        amount_l = self._config.get("color_lightness", 30) / 100
        amount_ab = self._config.get("color_ab", 8) / 100
        adjust = np.array([amount_l, amount_ab, amount_ab], dtype="float32")
        randoms = (
            (np.random.rand(self._batchsize, 1, 1, 3).astype("float32") * (adjust * 2)) - adjust)
        logger.trace("Random LAB adjustments: %s", randoms)

        for image, rand in zip(batch, randoms):
            for idx in range(rand.shape[-1]):
                adjustment = rand[:, :, idx]
                if adjustment >= 0:
                    image[:, :, idx] = ((255 - image[:, :, idx]) * adjustment) + image[:, :, idx]
                else:
                    image[:, :, idx] = image[:, :, idx] * (1 + adjustment)
        return batch

    # <<< IMAGE AUGMENTATION >>> #
    def transform(self, batch):
        """ Perform random transformation on the passed in batch.

        The transformation parameters are set in :file:`config.train.ini`

        Parameters
        ----------
        batch: numpy.ndarray
            The batch should be a 4-dimensional array of shape (`batchsize`, `height`, `width`,
            `channels`) and in `BGR` format.

        Returns
        ----------
        numpy.ndarray
            A 4-dimensional array of the same shape as :attr:`batch` with transformation applied.
        """
        if self.is_display:
            return batch
        logger.trace("Randomly transforming image")
        rotation_range = self._config.get("rotation_range", 10)
        zoom_range = self._config.get("zoom_range", 5) / 100
        shift_range = self._config.get("shift_range", 5) / 100

        rotation = np.random.uniform(-rotation_range,
                                     rotation_range,
                                     size=self._batchsize).astype("float32")
        scale = np.random.uniform(1 - zoom_range,
                                  1 + zoom_range,
                                  size=self._batchsize).astype("float32")
        tform = np.random.uniform(
            -shift_range,
            shift_range,
            size=(self._batchsize, 2)).astype("float32") * self._training_size

        mats = np.array(
            [cv2.getRotationMatrix2D((self._training_size // 2, self._training_size // 2),
                                     rot,
                                     scl)
             for rot, scl in zip(rotation, scale)]).astype("float32")
        mats[..., 2] += tform

        batch = np.array([cv2.warpAffine(image,
                                         mat,
                                         (self._training_size, self._training_size),
                                         borderMode=cv2.BORDER_REPLICATE)
                          for image, mat in zip(batch, mats)])

        logger.trace("Randomly transformed image")
        return batch

    def random_flip(self, batch):
        """ Perform random horizontal flipping on the passed in batch.

        The probability of flipping an image is set in :file:`config.train.ini`

        Parameters
        ----------
        batch: numpy.ndarray
            The batch should be a 4-dimensional array of shape (`batchsize`, `height`, `width`,
            `channels`) and in `BGR` format.

        Returns
        ----------
        numpy.ndarray
            A 4-dimensional array of the same shape as :attr:`batch` with transformation applied.
        """
        if not self.is_display:
            logger.trace("Randomly flipping image")
            randoms = np.random.rand(self._batchsize)
            indices = np.where(randoms > self._config.get("random_flip", 50) / 100)[0]
            batch[indices] = batch[indices, :, ::-1]
            logger.trace("Randomly flipped %s images of %s", len(indices), self._batchsize)
        return batch

    def warp(self, batch, to_landmarks=False, **kwargs):
        """ Perform random warping on the passed in batch by one of two methods.

        Parameters
        ----------
        batch: numpy.ndarray
            The batch should be a 4-dimensional array of shape (`batchsize`, `height`, `width`,
            `3`) and in `BGR` format.
        to_landmarks: bool, optional
            If ``False`` perform standard random warping of the input image. If ``True`` perform
            warping to semi-random similar corresponding landmarks from the other side. Default:
            ``False``
        kwargs: dict
            If :attr:`to_landmarks` is ``True`` the following additional kwargs must be passed in:

            * **batch_src_points** (`numpy.ndarray`) - A batch of 68 point landmarks for the \
            source faces. This is a 3-dimensional array in the shape (`batchsize`, `68`, `2`).

            * **batch_dst_points** (`numpy.ndarray`) - A batch of randomly chosen closest match \
            destination faces landmarks. This is a 3-dimensional array in the shape (`batchsize`, \
             `68`, `2`).
        Returns
        ----------
        numpy.ndarray
            A 4-dimensional array of the same shape as :attr:`batch` with warping applied.
        """
        if to_landmarks:
            return self._random_warp_landmarks(batch, **kwargs).astype("float32") / 255.0
        return self._random_warp(batch).astype("float32") / 255.0

    def _random_warp(self, batch):
        """ Randomly warp the input batch """
        logger.trace("Randomly warping batch")
        mapx = self._constants["warp_mapx"]
        mapy = self._constants["warp_mapy"]
        pad = self._constants["warp_pad"]
        slices = self._constants["warp_slices"]

        rands = np.random.normal(size=(self._batchsize, 2, 5, 5),
                                 scale=self._scale).astype("float32")
        batch_maps = np.stack((mapx, mapy), axis=1) + rands
        batch_interp = np.array([[cv2.resize(map_, (pad, pad))[slices, slices] for map_ in maps]
                                 for maps in batch_maps])
        warped_batch = np.array([cv2.remap(image, interp[0], interp[1], cv2.INTER_LINEAR)
                                 for image, interp in zip(batch, batch_interp)])

        logger.trace("Warped image shape: %s", warped_batch.shape)
        return warped_batch

    def _random_warp_landmarks(self, batch, batch_src_points, batch_dst_points):
        """ From dfaker. Warp the image to a similar set of landmarks from the opposite side """
        logger.trace("Randomly warping landmarks")
        edge_anchors = self._constants["warp_lm_edge_anchors"]
        grids = self._constants["warp_lm_grids"]
        slices = self._constants["tgt_slices"]

        batch_dst = (batch_dst_points + np.random.normal(size=batch_dst_points.shape,
                                                         scale=2.0))

        face_cores = [cv2.convexHull(np.concatenate([src[17:], dst[17:]], axis=0))
                      for src, dst in zip(batch_src_points.astype("int32"),
                                          batch_dst.astype("int32"))]

        batch_src = np.append(batch_src_points, edge_anchors, axis=1)
        batch_dst = np.append(batch_dst, edge_anchors, axis=1)

        rem_indices = [list(set(idx for fpl in (src, dst)
                                for idx, (pty, ptx) in enumerate(fpl)
                                if cv2.pointPolygonTest(face_core, (pty, ptx), False) >= 0))
                       for src, dst, face_core in zip(batch_src[:, :18, :],
                                                      batch_dst[:, :18, :],
                                                      face_cores)]
        batch_src = [np.delete(src, idxs, axis=0) for idxs, src in zip(rem_indices, batch_src)]
        batch_dst = [np.delete(dst, idxs, axis=0) for idxs, dst in zip(rem_indices, batch_dst)]

        grid_z = np.array([griddata(dst, src, (grids[0], grids[1]), method="linear")
                           for src, dst in zip(batch_src, batch_dst)])
        maps = grid_z.reshape(self._batchsize,
                              self._training_size,
                              self._training_size,
                              2).astype("float32")
        warped_batch = np.array([cv2.remap(image,
                                           map_[..., 1],
                                           map_[..., 0],
                                           cv2.INTER_LINEAR,
                                           cv2.BORDER_TRANSPARENT)
                                 for image, map_ in zip(batch, maps)])
        warped_batch = np.array([cv2.resize(image[slices, slices, :],
                                            (self._input_size, self._input_size),
                                            cv2.INTER_AREA)
                                 for image in warped_batch])
        logger.trace("Warped batch shape: %s", warped_batch.shape)
        return warped_batch
