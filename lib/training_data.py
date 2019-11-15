#!/usr/bin/env python3
""" Handles Data Augmentation for feeding Faceswap Models """

import logging

from hashlib import sha1
from random import shuffle, choice

import numpy as np
import cv2
from scipy.interpolate import griddata

from lib.image import batch_convert_color, read_image_batch
from lib.model import masks
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
        Dictates how much of the image will be cropped out. Eg: a coverage ratio of 0.625 \
        will result in cropping a 160px box from a 256px image (256 * 0.625 = 160).

        * **augment_color** (`bool`) - ``True`` if color is to be augmented, otherwise ``False`` \

        * **no_flip** (`bool`) - ``True`` if the image shouldn't be randomly flipped as part of \
        augmentation, otherwise ``False``

        * **mask_type** (`str`) - The mask type to be used (as defined in \
        :mod:`lib.model.masks`). If not ``None`` then the additional key ``landmarks`` must be \
        provided.

        * **warp_to_landmarks** (`bool`) - ``True`` if the random warp method should warp to \
        similar landmarks from the other side, ``False`` if the standard random warp method \
        should be used. If ``True`` then the additional key ``landmarks`` must be provided.

        * **landmarks** (`numpy.ndarray`, `optional`). Required if using a :attr:`mask_type` is \
        not ``None`` or :attr:`warp_to_landmarks` is ``True``. The 68 point face landmarks from \
        an alignments file.

    config: dict
        The configuration ``dict`` generated from :file:`config.train.ini` containing the trainer \
        plugin configuration options.
    """
    def __init__(self, model_input_size, model_output_shapes, training_opts, config):
        logger.debug("Initializing %s: (model_input_size: %s, model_output_shapes: %s, "
                     "training_opts: %s, landmarks: %s, config: %s)",
                     self.__class__.__name__, model_input_size, model_output_shapes,
                     {key: val for key, val in training_opts.items() if key != "landmarks"},
                     bool(training_opts.get("landmarks", None)), config)
        self._config = config
        self._model_input_size = model_input_size
        self._model_output_shapes = model_output_shapes
        self._training_opts = training_opts
        self._mask_class = self._set__mask_class()
        self._landmarks = self._training_opts.get("landmarks", None)
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
        training, preview and timelapses.

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
            list of filenames are processed, the data will be reshuffled to make sure thay are not
            returned in the same order. Default: ``True``
        is_preview: bool, optional
            Indicates whether this iterator is generating preview images. If ``True`` then certain
            augmentations will not be performed. Default: ``False``
        is_timelapse: bool optional
            Indicates whether this iterator is generating Timelapse images. If ``True``, then
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
            the format (`batchsize`, `height`, `width`, `1`). **NB:** This item will only exist \
            in the ``dict`` if the :attr:`mask_type` is not ``None``

            * **samples** (`numpy.ndarray`) - A 4-dimensional array containg the samples for \
            feeding to the model's predict function for generating preview and timelapse samples. \
            The array will be in the format (`batchsize`, `height`, `width`, `channels`). **NB:** \
            This item will only exist in the ``dict`` if :attr:`is_preview` or \
            :attr:`is_timelapse` is ``True``
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
    def _set__mask_class(self):
        """ Returns the correct mask class from :mod:`lib`.model.masks` as defined in the
        :attr:`mask_type` parameter. """
        mask_type = self._training_opts.get("mask_type", None)
        if mask_type:
            logger.debug("Mask type: '%s'", mask_type)
            _mask_class = getattr(masks, mask_type)
        else:
            _mask_class = None
        logger.debug("Mask class: %s", _mask_class)
        return _mask_class

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
        processed = dict()
        to_landmarks = self._training_opts["warp_to_landmarks"]

        # Initialize processing training size on first image
        if not self._processing.initialized:
            self._processing.initialize(batch.shape[1])

        # Get Landmarks prior to manipulating the image
        if self._mask_class or to_landmarks:
            batch_src_pts = self._get_landmarks(filenames, batch, side)

        # Color augmentation before mask is added
        if self._training_opts["augment_color"]:
            batch = self._processing.color_adjust(batch)

        # Add mask to batch prior to transforms and warps
        if self._mask_class:
            batch = np.array([self._mask_class(src_pts, image, channels=4).mask
                              for src_pts, image in zip(batch_src_pts, batch)])

        # Random Transform and flip
        batch = self._processing.transform(batch)
        if not self._training_opts["no_flip"]:
            batch = self._processing.random_flip(batch)

        # Add samples to output if this is for display
        if self._processing.is_display:
            processed["samples"] = batch[..., :3].astype("float32") / 255.0

        # Get Targets
        processed.update(self._processing.get_targets(batch))

        # Random Warp
        if to_landmarks:
            warp_kwargs = dict(batch_src_points=batch_src_pts,
                               batch_dst_points=self._get_closest_match(filenames,
                                                                        side,
                                                                        batch_src_pts))
        else:
            warp_kwargs = dict()
        processed["feed"] = self._processing.warp(batch[..., :3], to_landmarks, **warp_kwargs)

        logger.trace("Processed batch: (filenames: %s, side: '%s', processed: %s)",
                     filenames,
                     side,
                     {k: v.shape if isinstance(v, np.ndarray) else[i.shape for i in v]
                      for k, v in processed.items()})

        return processed

    def _get_landmarks(self, filenames, batch, side):
        """ Obtains the 68 Point Landmarks for the images in this batch. This is only called if
        config item ``warp_to_landmarks`` is ``True`` or if :attr:`mask_type` is not ``None``. If
        the landmarks for an image cannot be found, then an error is raised. """
        logger.trace("Retrieving landmarks: (filenames: %s, side: '%s')", filenames, side)
        src_points = [self._landmarks[side].get(sha1(face).hexdigest(), None) for face in batch]

        # Raise error on missing alignments
        if not all(isinstance(pts, np.ndarray) for pts in src_points):
            indices = [idx for idx, hsh in enumerate(src_points) if hsh is None]
            missing = [filenames[idx] for idx in indices]
            msg = ("Files missing alignments for this batch: {}"
                   "\nAt least one of your images does not have a matching entry in your "
                   "alignments file."
                   "\nIf you are training with a mask or using 'warp to landmarks' then every "
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
        Whether the images being fed through will be used for Preview or Timelapse. Disables
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
        cropped out. Eg: a coverage ratio of 0.625 will result in cropping a 160px box from a 256px
        image (256 * 0.625 = 160).
    config: dict
        The configuration ``dict`` generated from :file:`config.train.ini` containing the trainer \
        plugin configuration options.

    Attributes
    ----------
    initialized: bool
        Flag to indicate whether :class:`ImageAugmentation` has been initialized with the training
        image size in order to cache certain augmentation operations (see :func:`initialize`)
    is_display: bool
        Flag to indicate whether these augmentations are for timelapses/preview images (``True``)
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
            the format (`batchsize`, `height`, `width`, `1`). **NB:** This item will only exist \
            in the ``dict`` if a batch of 4 channel images has been passed in :attr:`batch`
        """
        logger.trace("Compiling targets")
        slices = self._constants["tgt_slices"]
        target_batch = [np.array([cv2.resize(image[slices, slices, :],
                                             (size, size),
                                             cv2.INTER_AREA)
                                  for image in batch])
                        for size in self._output_sizes]
        logger.trace("Target image shapes: %s",
                     [tgt.shape for tgt_images in target_batch for tgt in tgt_images])

        retval = self._separate_target_mask(target_batch)
        logger.trace("Final targets: %s",
                     {k: v.shape if isinstance(v, np.ndarray) else [img.shape for img in v]
                      for k, v in retval.items()})
        return retval

    @staticmethod
    def _separate_target_mask(batch):
        """ Return the batch and the batch of final masks

        Returns the targets as a list of 4-dimensional ``numpy.ndarray`` s of shape (`batchsize`,
        `height`, `width`, 3). If the :attr:`batch` is 4 channels, then the masks will be split
        from the batch, with the largest output masks being returned in their own item.
        """
        batch = [tgt.astype("float32") / 255.0 for tgt in batch]
        if all(tgt.shape[-1] == 4 for tgt in batch):
            logger.trace("Batch contains mask")
            sizes = [item.shape[1] for item in batch]
            mask_batch = np.expand_dims(batch[sizes.index(max(sizes))][..., -1], axis=-1)
            batch = [item[..., :3] for item in batch]
            logger.trace("batch shapes: %s, mask_batch shape: %s",
                         [tgt.shape for tgt in batch], mask_batch.shape)
            retval = dict(targets=batch, masks=mask_batch)
        else:
            logger.trace("Batch has no mask")
            retval = dict(targets=batch)
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
        """ Randomly perform Contrast Limited Adaptive Histogram Equilization on
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
        """ Perform random color/lightness adjustment in L*a*b* colorspace on a batch of images """
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
