#!/usr/bin/env python3
""" Processes the augmentation of images for feeding into a Faceswap model. """
import logging

import cv2
import numpy as np
from scipy.interpolate import griddata

from lib.image import batch_convert_color

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


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
        cropped out. E.G: a coverage ratio of 0.625 will result in cropping a 160px box from a
        256px image (:math:`256 * 0.625 = 160`)
    config: dict
        The configuration `dict` generated from :file:`config.train.ini` containing the trainer
        plugin configuration options.

    Attributes
    ----------
    initialized: bool
        Flag to indicate whether :class:`ImageAugmentation` has been initialized with the training
        image size in order to cache certain augmentation operations (see :func:`initialize`)
    is_display: bool
        Flag to indicate whether these augmentations are for time-lapses/preview images (``True``)
        or standard training data (``False``)
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
        training, so it cannot be set in the :func:`__init__` method. When the first training batch
        is loaded this function should be called to initialize the class and perform various
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
        coverage = int(self._training_size * self._coverage_ratio // 2) * 2

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
        batch: :class:`numpy.ndarray`
            This should be a 4+-dimensional array of training images in the format (`batchsize`,
            `height`, `width`, `channels`). Targets should be requested after performing image
            transformations but prior to performing warps.

            The 4th channel should be the mask. Any channels above the 4th should be any additional
            masks that are requested.

        Returns
        -------
        dict
            The following keys will be within the returned dictionary:

            * **targets** (`list`) - A list of 4-dimensional :class:`numpy.ndarray` s in the \
            order and size of each output of the model as defined in :attr:`output_shapes`. The \
            format of these arrays will be (`batchsize`, `height`, `width`, `3`). **NB:** \
            masks are not included in the `targets` list. If masks are to be included in the \
            output they will be returned as their own item from the `masks` key.

            * **masks** (:class:`numpy.ndarray`) - A 4-dimensional array containing the target \
            masks in the format (`batchsize`, `height`, `width`, `1`).
        """
        logger.trace("Compiling targets: batch shape: %s", batch.shape)
        slices = self._constants["tgt_slices"]
        target_batch = [np.array([cv2.resize(image[slices, slices, :],
                                             (size, size),
                                             cv2.INTER_AREA)
                                  for image in batch], dtype='float32') / 255.
                        for size in self._output_sizes]
        logger.trace("Target image shapes: %s",
                     [tgt_images.shape for tgt_images in target_batch])

        retval = self._separate_target_mask(target_batch)
        logger.trace("Final targets: %s",
                     {k: v.shape if isinstance(v, np.ndarray) else [img.shape for img in v]
                      for k, v in retval.items()})
        return retval

    @staticmethod
    def _separate_target_mask(target_batch):
        """ Return the batch and the batch of final masks

        Parameters
        ----------
        target_batch: list
            List of 4 dimension :class:`numpy.ndarray` objects resized the model outputs.
            The 4th channel of the array contains the face mask, any additional channels after
            this are additional masks (e.g. eye mask and mouth mask)

        Returns
        -------
        dict:
            The targets and the masks separated into their own items. The targets are a list of
            3 channel, 4 dimensional :class:`numpy.ndarray` objects sized for each output from the
            model. The masks are a :class:`numpy.ndarray` of the final output size. Any additional
            masks(e.g. eye and mouth masks) will be collated together into a :class:`numpy.ndarray`
            of the final output size. The number of channels will be the number of additional
            masks available
        """
        logger.trace("target_batch shapes: %s", [tgt.shape for tgt in target_batch])
        retval = dict(targets=[batch[..., :3] for batch in target_batch],
                      masks=target_batch[-1][..., 3][..., None])
        if target_batch[-1].shape[-1] > 4:
            retval["additional_masks"] = target_batch[-1][..., 4:]
        logger.trace("returning: %s", {k: v.shape if isinstance(v, np.ndarray) else [tgt.shape
                                                                                     for tgt in v]
                                       for k, v in retval.items()})
        return retval

    # <<< COLOR AUGMENTATION >>> #
    def color_adjust(self, batch):
        """ Perform color augmentation on the passed in batch.

        The color adjustment parameters are set in :file:`config.train.ini`

        Parameters
        ----------
        batch: :class:`numpy.ndarray`
            The batch should be a 4-dimensional array of shape (`batchsize`, `height`, `width`,
            `3`) and in `BGR` format.

        Returns
        ----------
        :class:`numpy.ndarray`
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
        batch: :class:`numpy.ndarray`
            The batch should be a 4-dimensional array of shape (`batchsize`, `height`, `width`,
            `channels`) and in `BGR` format.

        Returns
        ----------
        :class:`numpy.ndarray`
            A 4-dimensional array of the same shape as :attr:`batch` with transformation applied.
        """
        if self.is_display:
            return batch
        logger.trace("Randomly transforming image")
        rotation_range = self._config.get("rotation_range", 10)
        zoom_range = self._config.get("zoom_amount", 5) / 100
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
        batch: :class:`numpy.ndarray`
            The batch should be a 4-dimensional array of shape (`batchsize`, `height`, `width`,
            `channels`) and in `BGR` format.

        Returns
        ----------
        :class:`numpy.ndarray`
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
        batch: :class:`numpy.ndarray`
            The batch should be a 4-dimensional array of shape (`batchsize`, `height`, `width`,
            `3`) and in `BGR` format.
        to_landmarks: bool, optional
            If ``False`` perform standard random warping of the input image. If ``True`` perform
            warping to semi-random similar corresponding landmarks from the other side. Default:
            ``False``
        kwargs: dict
            If :attr:`to_landmarks` is ``True`` the following additional kwargs must be passed in:

            * **batch_src_points** (:class:`numpy.ndarray`) - A batch of 68 point landmarks for \
            the source faces. This is a 3-dimensional array in the shape (`batchsize`, `68`, `2`).

            * **batch_dst_points** (:class:`numpy.ndarray`) - A batch of randomly chosen closest \
            match destination faces landmarks. This is a 3-dimensional array in the shape \
            (`batchsize`, `68`, `2`).
        Returns
        ----------
        :class:`numpy.ndarray`
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
        maps = grid_z.reshape((self._batchsize,
                               self._training_size,
                               self._training_size,
                               2)).astype("float32")
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

    def skip_warp(self, batch):
        """ Returns the images resized and cropped for feeding the model, if warping has been
        disabled.

        Parameters
        ----------
        batch: :class:`numpy.ndarray`
            The batch should be a 4-dimensional array of shape (`batchsize`, `height`, `width`,
            `3`) and in `BGR` format.

        Returns
        -------
        :class:`numpy.ndarray`
            The given batch cropped and resized for feeding the model
        """
        logger.trace("Compiling skip warp images: batch shape: %s", batch.shape)
        slices = self._constants["tgt_slices"]
        retval = np.array([cv2.resize(image[slices, slices, :],
                                      (self._input_size, self._input_size),
                                      cv2.INTER_AREA)
                           for image in batch], dtype='float32') / 255.
        logger.trace("feed batch shape: %s", retval.shape)
        return retval
