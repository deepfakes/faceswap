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
        The expected input size for the model. This always assumes a square image and is the
        size, in pixels, of the `width` and the `height` of the input to the model.
    model_output_shapes: list
        A list of tuples defining the output shapes from the model, in the order that the outputs
        are returned. The tuples should be in (`height`, `width`, `channels`) format.
    training_opts: dict
        This is a dictionary of model training options as defined in
        :mod:`plugins.train.model._base`. These options will be defined by the user from the
        provided cli options or from the model ``config.ini``. At a minimum this ``dict`` should
        contain the following keys:

        * **coverage_ratio** (`float`) - The ratio of the training image to be trained on. \
        Dictates how much of the image will be cropped out.

        * **augment_color** (`bool`) - ``True`` if color is to be augmented, otherwise ``False`` \

        * **no_flip** (`bool`) - ``True`` if the image shouldn't be randomly flipped as part of \
        augmentation, otherwise ``False``

        * **mask_type** (`str`) - The mask type to be used (as defined in \
        :mod:`lib.model.masks`). If not ``None`` then the additional key ``landmarks`` must be \
        provided.

        * **warp_to_landmarks** (`bool`) - ``True`` if \
        :func:`~lib.training_data.ImageManipulation.random_warp_landmarks` should be used. \
        ``False`` if :func:`~lib.training_data.ImageManipulation.random_warp` should be used. If \
        ``True`` then the additional key ``landmarks`` must be provided.

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
        self.batchsize = 0
        self.processing = None
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
            format of these arrays will be (`batchsize`, `height`, `width`, `channels`). This is \
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
        self.batchsize = batchsize
        self.processing = ImageManipulation(batchsize,
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
               "batch-size: {}".format(length, self.batchsize))
        try:
            assert length >= self.batchsize, msg
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

        # TODO Check Timelapse works
        # TODO Remove this test
        # for idx, image in enumerate(batch):
        #     if side == "a" and not self.processing.is_display:
        #         print("Orig:", image.dtype)
        #         cv2.imwrite("/home/matt/fake/test/testing/{}_orig.png".format(idx), image)

        # Initialize processing training size on first image
        if not self.processing.initialized:
            self.processing.initialize(batch.shape[1])

        # Get Landmarks prior to manipulating the image
        if self._mask_class or self._training_opts["warp_to_landmarks"]:
            batch_src_pts = self._get_landmarks(filenames, batch, side)

        # Color augmentation before mask is added
        if self._training_opts["augment_color"]:
            batch = self.processing.color_adjust(batch)

        # Add mask to batch prior to transforms and warps
        if self._mask_class:
            batch = np.array([self._mask_class(src_pts, image, channels=4).mask
                              for src_pts, image in zip(batch_src_pts, batch)])

        # Random Transform and flip
        batch = self.processing.random_transform(batch)
        if not self._training_opts["no_flip"]:
            batch = self.processing.do_random_flip(batch)

        # TODO Remove this test
        # for idx, image in enumerate(batch):
        #     if side == "a" and not self.processing.is_display:
        #         print("warp:", image.dtype, image.shape, image.min(), image.max())
        #         cv2.imwrite("/home/matt/fake/test/testing/{}_tran.png".format(idx), image)

        # Add samples to output if this is for display
        if self.processing.is_display:
            processed["samples"] = batch[..., :3].astype("float32") / 255.0

        # Get Targets
        processed.update(self.processing.get_targets(batch))

        # TODO Remove this test
        # for idx, (tgt, mask) in enumerate(zip(targets[0][0], targets[1])):
        #     if side == "a" and not self.processing.is_display:
        #         print("tgt:", tgt.dtype, tgt.shape, tgt.min(), tgt.max())
        #         print("mask:", mask.dtype, mask.shape, mask.min(), mask.max())
        #         cv2.imwrite("/home/matt/fake/test/testing/{}_tgt.png".format(idx), tgt)
        #         cv2.imwrite("/home/matt/fake/test/testing/{}_mask.png".format(idx), mask)

        if self._training_opts["warp_to_landmarks"]:
            # TODO
            pass
            # dst_pts = self._get_closest_match(filename, side, src_pts)
            # processed = self.processing.random_warp_landmarks(image, src_pts, dst_pts)
        else:
            processed["feed"] = self.processing.random_warp(batch[..., :3])

        # TODO Remove this test
        # for idx, image in enumerate(batch):
        #     if side == "a" and not self.processing.is_display:
        #         print("warp:", image.dtype, image.shape, image.min(), image.max())
        #         cv2.imwrite("/home/matt/fake/test/testing/{}_warp.png".format(idx), image)
        # exit(0)
        logger.trace("Processed batch: (filenames: %s, side: '%s', processed: %s)",
                     filenames,
                     side,
                     {k: v.shape if isinstance(v, np.ndarray) else[i.shape for i in v]
                      for k, v in processed.items()})
        return processed

    def _get_landmarks(self, filenames, batch, side):
        """ Obtains the 68 Point Landmarks for the images in this batch. This is only called if
        :attr:`warp_to_landmarks` is ``True`` or if :attr:`mask_type` is not ``None``. If the
        landmarks for an image cannot be found, then an error is raised. """
        logger.trace("Retrieving landmarks: (filenames: '%s', side: '%s'", filenames, side)
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

        logger.trace("Returning: (src_points: %s)", src_points)
        return src_points

    def _get_closest_match(self, filename, side, src_points):
        """ Only called if :attr:`warp_to_landmarks` is ``True``. Gets the closest matched 68 point
        landmarks from the opposite training set. """
        logger.trace("Retrieving closest matched landmarks: (filename: '%s', src_points: '%s'",
                     filename, src_points)
        landmarks = self._landmarks["a"] if side == "b" else self._landmarks["b"]
        closest_hashes = self._nearest_landmarks.get(filename)
        if not closest_hashes:
            dst_points_items = list(landmarks.items())
            dst_points = list(x[1] for x in dst_points_items)
            closest = (np.mean(np.square(src_points - dst_points), axis=(1, 2))).argsort()[:10]
            closest_hashes = tuple(dst_points_items[i][0] for i in closest)
            self._nearest_landmarks[filename] = closest_hashes
        dst_points = landmarks[choice(closest_hashes)]
        logger.trace("Returning: (dst_points: %s)", dst_points)
        return dst_points


class ImageManipulation():
    """ Manipulations to be performed on training images """
    def __init__(self, batchsize, is_display, input_size, output_shapes, coverage_ratio, config):
        """ input_size: Size of the face input into the model
            output_shapes: Shapes that come out of the model
            coverage_ratio: Coverage ratio of full image. Eg: 256 * 0.625 = 160
        """
        logger.debug("Initializing %s: (batchsize: %s, is_display: %s, input_size: %s, "
                     "output_shapes: %s, coverage_ratio: %s, config: %s)",
                     self.__class__.__name__, batchsize, is_display, input_size, output_shapes,
                     coverage_ratio, config)
        self.batchsize = batchsize
        self.is_display = is_display
        self._config = config
        # Transform and Warp args
        self.input_size = input_size
        self.output_sizes = [shape[1] for shape in output_shapes if shape[2] == 3]
        logger.debug("Output sizes: %s", self.output_sizes)
        # Warp args
        self.coverage_ratio = coverage_ratio  # Coverage ratio of full image. Eg: 256 * 0.625 = 160
        self.scale = 5  # Normal random variable scale

        # Set on first image load from initialize
        self.initialized = False
        self.training_size = 0
        self.constants = None

        logger.debug("Initialized %s", self.__class__.__name__)

    def initialize(self, training_size):
        """ Initialize the constants once we have the training_size from the
            first batch """
        logger.debug("Initializing constants. training_size: %s", training_size)
        self.training_size = training_size

        coverage = int(self.training_size * self.coverage_ratio)

        # Color Aug
        clahe_base_contrast = training_size // 128
        # Target Images
        tgt_slices = slice(self.training_size // 2 - coverage // 2,
                           self.training_size // 2 + coverage // 2)
        # Warp
        warp_range_ = np.linspace(self.training_size // 2 - coverage // 2,
                                  self.training_size // 2 + coverage // 2, 5, dtype='float32')
        warp_mapx = np.broadcast_to(warp_range_, (self.batchsize, 5, 5)).astype("float32")
        warp_mapy = np.broadcast_to(warp_mapx[0].T, (self.batchsize, 5, 5)).astype("float32")

        warp_pad = int(1.25 * self.input_size)
        warp_slices = slice(warp_pad // 10, -warp_pad // 10)

        self.constants = dict(clahe_base_contrast=clahe_base_contrast,
                              tgt_slices=tgt_slices,
                              warp_mapx=warp_mapx,
                              warp_mapy=warp_mapy,
                              warp_pad=warp_pad,
                              warp_slices=warp_slices)
        self.initialized = True
        logger.debug("Initialized constants: %s", self.constants)

    def color_adjust(self, batch):
        """ Color adjust RGB image """
        if not self.is_display:
            logger.trace("Augmenting color")
            batch = batch_convert_color(batch, "BGR2LAB")
            batch = self.random_clahe(batch)
            batch = self.random_lab(batch)
            batch = batch_convert_color(batch, "LAB2BGR")
        return batch

    def random_clahe(self, batch):
        """ Randomly perform Contrast Limited Adaptive Histogram Equilization on
        a batch of images """
        base_contrast = self.constants["clahe_base_contrast"]

        batch_random = np.random.rand(self.batchsize)
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

    def random_lab(self, batch):
        """ Perform random color/lightness adjustment in L*a*b* colorspace on a batch of images """
        amount_l = self._config.get("color_lightness", 30) / 100
        amount_ab = self._config.get("color_ab", 8) / 100
        adjust = np.array([amount_l, amount_ab, amount_ab], dtype="float32")
        randoms = (
            (np.random.rand(self.batchsize, 1, 1, 3).astype("float32") * (adjust * 2)) - adjust)
        logger.trace("Random LAB adjustments: %s", randoms)

        for image, rand in zip(batch, randoms):
            for idx in range(rand.shape[-1]):
                adjustment = rand[:, :, idx]
                if adjustment >= 0:
                    image[:, :, idx] = ((255 - image[:, :, idx]) * adjustment) + image[:, :, idx]
                else:
                    image[:, :, idx] = image[:, :, idx] * (1 + adjustment)
        return batch

    def get_coverage(self, image):
        """ Return coverage value for given image """
        coverage = int(image.shape[0] * self.coverage_ratio)
        logger.trace("Coverage: %s", coverage)
        return coverage

    def random_transform(self, batch):
        """ Randomly transform a batch """
        if self.is_display:
            return batch
        logger.trace("Randomly transforming image")
        rotation_range = self._config.get("rotation_range", 10)
        zoom_range = self._config.get("zoom_range", 5) / 100
        shift_range = self._config.get("shift_range", 5) / 100

        rotation = np.random.uniform(-rotation_range,
                                     rotation_range,
                                     size=self.batchsize).astype("float32")
        scale = np.random.uniform(1 - zoom_range,
                                  1 + zoom_range,
                                  size=self.batchsize).astype("float32")
        tform = np.random.uniform(-shift_range,
                                  shift_range,
                                  size=(self.batchsize, 2)).astype("float32") * self.training_size

        mats = np.array(
            [cv2.getRotationMatrix2D((self.training_size // 2, self.training_size // 2),
                                     rot,
                                     scl)
             for rot, scl in zip(rotation, scale)]).astype("float32")
        mats[..., 2] += tform

        batch = np.array([cv2.warpAffine(image,
                                         mat,
                                         (self.training_size, self.training_size),
                                         borderMode=cv2.BORDER_REPLICATE)
                          for image, mat in zip(batch, mats)])

        logger.trace("Randomly transformed image")
        return batch

    def do_random_flip(self, batch):
        """ Perform flip on images in batch if random number is within threshold """
        if not self.is_display:
            logger.trace("Randomly flipping image")
            randoms = np.random.rand(self.batchsize)
            indices = np.where(randoms > self._config.get("random_flip", 50) / 100)[0]
            batch[indices] = batch[indices, :, ::-1]
            logger.trace("Randomly flipped %s images of %s", len(indices), self.batchsize)
        return batch

    def get_targets(self, batch):
        """ Get the target batch """
        logger.trace("Compiling targets")
        slices = self.constants["tgt_slices"]
        target_batch = [np.array([cv2.resize(image[slices, slices, :],
                                             (size, size),
                                             cv2.INTER_AREA)
                                  for image in batch])
                        for size in self.output_sizes]
        logger.trace("Target image shapes: %s",
                     [tgt.shape for tgt_images in target_batch for tgt in tgt_images])

        retval = self.separate_target_mask(target_batch)
        logger.trace("Final targets: %s",
                     {k: v.shape if isinstance(v, np.ndarray) else [img.shape for img in v]
                      for k, v in retval.items()})
        return retval

    @staticmethod
    def separate_target_mask(batch):
        """ Return the batch and the batch of final masks from a batch of 4 channel images """
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

    def random_warp(self, batch):
        """ get pair of random warped images from aligned face image """
        logger.trace("Randomly warping batch")
        mapx = self.constants["warp_mapx"]
        mapy = self.constants["warp_mapy"]
        pad = self.constants["warp_pad"]
        slices = self.constants["warp_slices"]

        rands = np.random.normal(size=(self.batchsize, 2, 5, 5),
                                 scale=self.scale).astype("float32")
        batch_maps = np.stack((mapx, mapy), axis=1) + rands
        batch_interp = np.array([[cv2.resize(map_, (pad, pad))[slices, slices] for map_ in maps]
                                 for maps in batch_maps])
        warped_batch = np.array([cv2.remap(image, interp[0], interp[1], cv2.INTER_LINEAR)
                                 for image, interp in zip(batch, batch_interp)])

        logger.trace("Warped image shape: %s", warped_batch.shape)
        return warped_batch.astype("float32") / 255.0

    def random_warp_landmarks(self, image, src_points=None, dst_points=None):
        """ get warped image, target image and target mask
            From DFAKER plugin """
        logger.trace("Randomly warping landmarks")
        size = image.shape[0]
        coverage = self.get_coverage(image) // 2

        p_mx = size - 1
        p_hf = (size // 2) - 1

        edge_anchors = [(0, 0), (0, p_mx), (p_mx, p_mx), (p_mx, 0),
                        (p_hf, 0), (p_hf, p_mx), (p_mx, p_hf), (0, p_hf)]
        grid_x, grid_y = np.mgrid[0:p_mx:complex(size), 0:p_mx:complex(size)]

        source = src_points
        destination = (dst_points.copy().astype('float32') +
                       np.random.normal(size=dst_points.shape, scale=2.0))
        destination = destination.astype('uint8')

        face_core = cv2.convexHull(np.concatenate(
            [source[17:], destination[17:]], axis=0).astype(int))

        source = [(pty, ptx) for ptx, pty in source] + edge_anchors
        destination = [(pty, ptx) for ptx, pty in destination] + edge_anchors

        indicies_to_remove = set()
        for fpl in source, destination:
            for idx, (pty, ptx) in enumerate(fpl):
                if idx > 17:
                    break
                elif cv2.pointPolygonTest(face_core,
                                          (pty, ptx),
                                          False) >= 0:
                    indicies_to_remove.add(idx)

        for idx in sorted(indicies_to_remove, reverse=True):
            source.pop(idx)
            destination.pop(idx)

        grid_z = griddata(destination, source, (grid_x, grid_y), method="linear")
        map_x = np.append([], [ar[:, 1] for ar in grid_z]).reshape(size, size)
        map_y = np.append([], [ar[:, 0] for ar in grid_z]).reshape(size, size)
        map_x_32 = map_x.astype('float32')
        map_y_32 = map_y.astype('float32')

        warped_image = cv2.remap(image,
                                 map_x_32,
                                 map_y_32,
                                 cv2.INTER_LINEAR,
                                 cv2.BORDER_TRANSPARENT)
        target_image = image

        # TODO Make sure this replacement is correct
        slices = slice(size // 2 - coverage, size // 2 + coverage)
#        slices = slice(size // 32, size - size // 32)  # 8px on a 256px image
        warped_image = cv2.resize(
            warped_image[slices, slices, :], (self.input_size, self.input_size),
            cv2.INTER_AREA)
        logger.trace("Warped image shape: %s", warped_image.shape)
        target_images = [cv2.resize(target_image[slices, slices, :],
                                    (size, size),
                                    cv2.INTER_AREA)
                         for size in self.output_sizes]

        logger.trace("Target image shapea: %s", [img.shape for img in target_images])
        return self.compile_images(warped_image, target_images)


def stack_images(images):
    """ Stack images """
    logger.debug("Stack images")

    def get_transpose_axes(num):
        if num % 2 == 0:
            logger.debug("Even number of images to stack")
            y_axes = list(range(1, num - 1, 2))
            x_axes = list(range(0, num - 1, 2))
        else:
            logger.debug("Odd number of images to stack")
            y_axes = list(range(0, num - 1, 2))
            x_axes = list(range(1, num - 1, 2))
        return y_axes, x_axes, [num - 1]

    images_shape = np.array(images.shape)
    new_axes = get_transpose_axes(len(images_shape))
    new_shape = [np.prod(images_shape[x]) for x in new_axes]
    logger.debug("Stacked images")
    return np.transpose(
        images,
        axes=np.concatenate(new_axes)
        ).reshape(new_shape)
