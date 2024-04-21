#!/usr/bin/env python3
""" Processes the augmentation of images for feeding into a Faceswap model. """
from __future__ import annotations
import logging
import typing as T

import cv2
import numexpr as ne
import numpy as np
from scipy.interpolate import griddata

from lib.image import batch_convert_color
from lib.logger import parse_class_init

if T.TYPE_CHECKING:
    from lib.config import ConfigValueType

logger = logging.getLogger(__name__)


class AugConstants:  # pylint:disable=too-many-instance-attributes,too-few-public-methods
    """ Dataclass for holding constants for Image Augmentation.

    Parameters
    ----------
    config: dict[str, ConfigValueType]
        The user training configuration options
    processing_size: int:
        The size of image to augment the data for
    batch_size: int
        The batch size that augmented data is being prepared for
    """
    def __init__(self,
                 config: dict[str, ConfigValueType],
                 processing_size: int,
                 batch_size: int) -> None:
        logger.debug(parse_class_init(locals()))
        self.clahe_base_contrast: int = 0
        """int: The base number for Contrast Limited Adaptive Histogram Equalization"""
        self.clahe_chance: float = 0.0
        """float: Probability to perform Contrast Limited Adaptive Histogram Equilization"""
        self.clahe_max_size: int = 0
        """int: Maximum clahe window size"""

        self.lab_adjust: np.ndarray
        """:class:`numpy.ndarray`: Adjustment amounts for L*A*B augmentation"""
        self.transform_rotation: int = 0
        """int: Rotation range for transformations"""
        self.transform_zoom: float = 0.0
        """float: Zoom range for transformations"""
        self.transform_shift: float = 0.0
        """float: Shift range for transformations"""
        self.warp_maps: np.ndarray
        """:class:`numpy.ndarray`The stacked (x, y) mappings for image warping"""
        self.warp_pad: tuple[int, int] = (0, 0)
        """:tuple[int, int]: The padding to apply for image warping"""
        self.warp_slices: slice
        """:slice: The slices for extracting a warped image"""
        self.warp_lm_edge_anchors: np.ndarray
        """::class:`numpy.ndarray`: The edge anchors for landmark based warping"""
        self.warp_lm_grids: np.ndarray
        """::class:`numpy.ndarray`: The grids for landmark based warping"""

        self._config = config
        self._size = processing_size
        self._load_config(batch_size)
        logger.debug("Initialized: %s", self.__class__.__name__)

    def _load_clahe(self) -> None:
        """ Load the CLAHE constants from user config """
        color_clahe_chance = self._config.get("color_clahe_chance", 50)
        color_clahe_max_size = self._config.get("color_clahe_max_size", 4)
        assert isinstance(color_clahe_chance, int)
        assert isinstance(color_clahe_max_size, int)

        self.clahe_base_contrast = max(2, self._size // 128)
        self.clahe_chance = color_clahe_chance / 100
        self.clahe_max_size = color_clahe_max_size
        logger.debug("clahe_base_contrast: %s, clahe_chance: %s, clahe_max_size: %s",
                     self.clahe_base_contrast, self.clahe_chance, self.clahe_max_size)

    def _load_lab(self) -> None:
        """ Load the random L*A*B augmentation constants """
        color_lightness = self._config.get("color_lightness", 30)
        color_ab = self._config.get("color_ab", 8)
        assert isinstance(color_lightness, int)
        assert isinstance(color_ab, int)

        amount_l = int(color_lightness) / 100
        amount_ab = int(color_ab) / 100

        self.lab_adjust = np.array([amount_l, amount_ab, amount_ab], dtype="float32")
        logger.debug("lab_adjust: %s", self.lab_adjust)

    def _load_transform(self) -> None:
        """ Load the random transform constants """
        shift_range = self._config.get("shift_range", 5)
        rotation_range = self._config.get("rotation_range", 10)
        zoom_amount = self._config.get("zoom_amount", 5)
        assert isinstance(shift_range, int)
        assert isinstance(rotation_range, int)
        assert isinstance(zoom_amount, int)

        self.transform_shift = (shift_range / 100) * self._size
        self.transform_rotation = rotation_range
        self.transform_zoom = zoom_amount / 100
        logger.debug("transform_shift: %s, transform_rotation: %s, transform_zoom: %s",
                     self.transform_shift, self.transform_rotation, self.transform_zoom)

    def _load_warp(self, batch_size: int) -> None:
        """ Load the warp augmentation constants

        Parameters
        ----------
        batch_size: int
            The batch size that augmented data is being prepared for
        """
        warp_range = np.linspace(0, self._size, 5, dtype='float32')
        warp_mapx = np.broadcast_to(warp_range, (batch_size, 5, 5)).astype("float32")
        warp_mapy = np.broadcast_to(warp_mapx[0].T, (batch_size, 5, 5)).astype("float32")
        warp_pad = int(1.25 * self._size)

        self.warp_maps = np.stack((warp_mapx, warp_mapy), axis=1)
        self.warp_pad = (warp_pad, warp_pad)
        self.warp_slices = slice(warp_pad // 10, -warp_pad // 10)
        logger.debug("warp_maps: (%s, %s), warp_pad: %s, warp_slices: %s",
                     self.warp_maps.shape, self.warp_maps.dtype,
                     self.warp_pad, self.warp_slices)

    def _load_warp_to_landmarks(self, batch_size: int) -> None:
        """ Load the warp-to-landmarks augmentation constants

        Parameters
        ----------
        batch_size: int
            The batch size that augmented data is being prepared for
        """
        p_mx = self._size - 1
        p_hf = (self._size // 2) - 1
        edge_anchors = np.array([(0, 0), (0, p_mx), (p_mx, p_mx), (p_mx, 0),
                                 (p_hf, 0), (p_hf, p_mx), (p_mx, p_hf), (0, p_hf)]).astype("int32")
        edge_anchors = np.broadcast_to(edge_anchors, (batch_size, 8, 2))
        grids = np.mgrid[0: p_mx: complex(self._size),  # type:ignore[misc]
                         0: p_mx: complex(self._size)]  # type:ignore[misc]

        self.warp_lm_edge_anchors = edge_anchors
        self.warp_lm_grids = grids
        logger.debug("warp_lm_edge_anchors: (%s, %s), warp_lm_grids: (%s, %s)",
                     self.warp_lm_edge_anchors.shape, self.warp_lm_edge_anchors.dtype,
                     self.warp_lm_grids.shape, self.warp_lm_grids.dtype)

    def _load_config(self, batch_size: int) -> None:
        """ Load the constants into the class from user config

        Parameters
        ----------
        batch_size: int
            The batch size that augmented data is being prepared for
        """
        logger.debug("Loading augmentation constants")
        self._load_clahe()
        self._load_lab()
        self._load_transform()
        self._load_warp(batch_size)
        self._load_warp_to_landmarks(batch_size)
        logger.debug("Loaded augmentation constants")


class ImageAugmentation():
    """ Performs augmentation on batches of training images.

    Parameters
    ----------
    batch_size: int
        The number of images that will be fed through the augmentation functions at once.
    processing_size: int
        The largest input or output size of the model. This is the size that images are processed
        at.
    config: dict
        The configuration `dict` generated from :file:`config.train.ini` containing the trainer
        plugin configuration options.
    """
    def __init__(self,
                 batch_size: int,
                 processing_size: int,
                 config: dict[str, ConfigValueType]) -> None:
        logger.debug(parse_class_init(locals()))
        self._processing_size = processing_size
        self._batch_size = batch_size

        # flip_args
        flip_chance = config.get("random_flip", 50)
        assert isinstance(flip_chance, int)
        self._flip_chance = flip_chance

        # Warp args
        self._warp_scale = 5 / 256 * self._processing_size  # Normal random variable scale
        self._warp_lm_scale = 2 / 256 * self._processing_size  # Normal random variable scale

        self._constants = AugConstants(config, processing_size, batch_size)
        logger.debug("Initialized %s", self.__class__.__name__)

    # <<< COLOR AUGMENTATION >>> #
    def color_adjust(self, batch: np.ndarray) -> np.ndarray:
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
        logger.trace("Augmenting color")  # type:ignore[attr-defined]
        batch = batch_convert_color(batch, "BGR2LAB")
        self._random_lab(batch)
        self._random_clahe(batch)
        batch = batch_convert_color(batch, "LAB2BGR")
        return batch

    def _random_clahe(self, batch: np.ndarray) -> None:
        """ Randomly perform Contrast Limited Adaptive Histogram Equalization on
        a batch of images """
        base_contrast = self._constants.clahe_base_contrast

        batch_random = np.random.rand(self._batch_size)
        indices = np.where(batch_random < self._constants.clahe_chance)[0]
        if not np.any(indices):
            return
        grid_bases = np.random.randint(self._constants.clahe_max_size + 1,
                                       size=indices.shape[0],
                                       dtype="uint8")
        grid_sizes = (grid_bases * (base_contrast // 2)) + base_contrast
        logger.trace("Adjusting Contrast. Grid Sizes: %s", grid_sizes)  # type:ignore[attr-defined]

        clahes = [cv2.createCLAHE(clipLimit=2.0,
                                  tileGridSize=(grid_size, grid_size))
                  for grid_size in grid_sizes]

        for idx, clahe in zip(indices, clahes):
            batch[idx, :, :, 0] = clahe.apply(batch[idx, :, :, 0], )

    def _random_lab(self, batch: np.ndarray) -> None:
        """ Perform random color/lightness adjustment in L*a*b* color space on a batch of
        images """
        randoms = np.random.uniform(-self._constants.lab_adjust,
                                    self._constants.lab_adjust,
                                    size=(self._batch_size, 1, 1, 3)).astype("float32")
        logger.trace("Random LAB adjustments: %s", randoms)  # type:ignore[attr-defined]
        # Iterating through the images and channels is much faster than numpy.where and slightly
        # faster than numexpr.where.
        for image, rand in zip(batch, randoms):
            for idx in range(rand.shape[-1]):
                adjustment = rand[:, :, idx]
                if adjustment >= 0:
                    image[:, :, idx] = ((255 - image[:, :, idx]) * adjustment) + image[:, :, idx]
                else:
                    image[:, :, idx] = image[:, :, idx] * (1 + adjustment)

    # <<< IMAGE AUGMENTATION >>> #
    def transform(self, batch: np.ndarray):
        """ Perform random transformation on the passed in batch.

        The transformation parameters are set in :file:`config.train.ini`

        Parameters
        ----------
        batch: :class:`numpy.ndarray`
            The batch should be a 4-dimensional array of shape (`batchsize`, `height`, `width`,
            `channels`) and in `BGR` format.
        """
        logger.trace("Randomly transforming image")  # type:ignore[attr-defined]

        rotation = np.random.uniform(-self._constants.transform_rotation,
                                     self._constants.transform_rotation,
                                     size=self._batch_size).astype("float32")
        scale = np.random.uniform(1 - self._constants.transform_zoom,
                                  1 + self._constants.transform_zoom,
                                  size=self._batch_size).astype("float32")

        tform = np.random.uniform(-self._constants.transform_shift,
                                  self._constants.transform_shift,
                                  size=(self._batch_size, 2)).astype("float32")
        mats = np.array(
            [cv2.getRotationMatrix2D((self._processing_size // 2, self._processing_size // 2),
                                     rot,
                                     scl)
             for rot, scl in zip(rotation, scale)]).astype("float32")
        mats[..., 2] += tform

        for image, mat in zip(batch, mats):
            cv2.warpAffine(image,
                           mat,
                           (self._processing_size, self._processing_size),
                           dst=image,
                           borderMode=cv2.BORDER_REPLICATE)

        logger.trace("Randomly transformed image")  # type:ignore[attr-defined]

    def random_flip(self, batch: np.ndarray):
        """ Perform random horizontal flipping on the passed in batch.

        The probability of flipping an image is set in :file:`config.train.ini`

        Parameters
        ----------
        batch: :class:`numpy.ndarray`
            The batch should be a 4-dimensional array of shape (`batchsize`, `height`, `width`,
            `channels`) and in `BGR` format.
        """
        logger.trace("Randomly flipping image")  # type:ignore[attr-defined]
        randoms = np.random.rand(self._batch_size)
        indices = np.where(randoms <= self._flip_chance / 100)[0]
        batch[indices] = batch[indices, :, ::-1]
        logger.trace("Randomly flipped %s images of %s",  # type:ignore[attr-defined]
                     len(indices), self._batch_size)

    def warp(self, batch: np.ndarray, to_landmarks: bool = False, **kwargs) -> np.ndarray:
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
            return self._random_warp_landmarks(batch, **kwargs)
        return self._random_warp(batch)

    def _random_warp(self, batch: np.ndarray) -> np.ndarray:
        """ Randomly warp the input batch

        Parameters
        ----------
        batch: :class:`numpy.ndarray`
            The batch should be a 4-dimensional array of shape (`batchsize`, `height`, `width`,
            `3`) and in `BGR` format.

        Returns
        ----------
        :class:`numpy.ndarray`
            A 4-dimensional array of the same shape as :attr:`batch` with warping applied.
        """
        logger.trace("Randomly warping batch")  # type:ignore[attr-defined]
        slices = self._constants.warp_slices
        rands = np.random.normal(size=(self._batch_size, 2, 5, 5),
                                 scale=self._warp_scale).astype("float32")
        batch_maps = ne.evaluate("m + r", local_dict={"m": self._constants.warp_maps, "r": rands})
        batch_interp = np.array([[cv2.resize(map_, self._constants.warp_pad)[slices, slices]
                                  for map_ in maps]
                                 for maps in batch_maps])
        warped_batch = np.array([cv2.remap(image, interp[0], interp[1], cv2.INTER_LINEAR)
                                 for image, interp in zip(batch, batch_interp)])

        logger.trace("Warped image shape: %s", warped_batch.shape)  # type:ignore[attr-defined]
        return warped_batch

    def _random_warp_landmarks(self,
                               batch: np.ndarray,
                               batch_src_points: np.ndarray,
                               batch_dst_points: np.ndarray) -> np.ndarray:
        """ From dfaker. Warp the image to a similar set of landmarks from the opposite side

        batch: :class:`numpy.ndarray`
            The batch should be a 4-dimensional array of shape (`batchsize`, `height`, `width`,
            `3`) and in `BGR` format.
        batch_src_points :class:`numpy.ndarray`
            A batch of 68 point landmarks for the source faces. This is a 3-dimensional array in
            the shape (`batchsize`, `68`, `2`).
        batch_dst_points :class:`numpy.ndarray`
            A batch of randomly chosen closest match destination faces landmarks. This is a
            3-dimensional array in the shape (`batchsize`, `68`, `2`).

        Returns
        ----------
        :class:`numpy.ndarray`
            A 4-dimensional array of the same shape as :attr:`batch` with warping applied.
        """
        logger.trace("Randomly warping landmarks")  # type:ignore[attr-defined]
        edge_anchors = self._constants.warp_lm_edge_anchors
        grids = self._constants.warp_lm_grids

        batch_dst = (batch_dst_points + np.random.normal(size=batch_dst_points.shape,
                                                         scale=self._warp_lm_scale))

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
        lbatch_src = [np.delete(src, idxs, axis=0) for idxs, src in zip(rem_indices, batch_src)]
        lbatch_dst = [np.delete(dst, idxs, axis=0) for idxs, dst in zip(rem_indices, batch_dst)]

        grid_z = np.array([griddata(dst, src, (grids[0], grids[1]), method="linear")
                           for src, dst in zip(lbatch_src, lbatch_dst)])
        maps = grid_z.reshape((self._batch_size,
                               self._processing_size,
                               self._processing_size,
                               2)).astype("float32")

        warped_batch = np.array([cv2.remap(image,
                                           map_[..., 1],
                                           map_[..., 0],
                                           cv2.INTER_LINEAR,
                                           borderMode=cv2.BORDER_TRANSPARENT)
                                 for image, map_ in zip(batch, maps)])
        logger.trace("Warped batch shape: %s", warped_batch.shape)  # type:ignore[attr-defined]
        return warped_batch
