#!/usr/bin/env python3
""" Processes the augmentation of images for feeding into a Faceswap model. """
from __future__ import annotations
from dataclasses import dataclass
import logging
import typing as T

import cv2
import numexpr as ne
import numpy as np
from scipy.interpolate import griddata

from lib.image import batch_convert_color

if T.TYPE_CHECKING:
    from lib.config import ConfigValueType

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@dataclass
class AugConstants:
    """ Dataclass for holding constants for Image Augmentation.

    Parameters
    ----------
    clahe_base_contrast: int
        The base number for Contrast Limited Adaptive Histogram Equalization
    clahe_chance: float
        Probability to perform Contrast Limited Adaptive Histogram Equilization
    clahe_max_size: int
        Maximum clahe window size
    lab_adjust: np.ndarray
        Adjustment amounts for L*A*B augmentation
    transform_rotation: int
        Rotation range for transformations
    transform_zoom: float
        Zoom range for transformations
    transform_shift: float
        Shift range for transformations
    warp_maps: :class:`numpy.ndarray`
        The stacked (x, y) mappings for image warping
    warp_pads: tuple
        The padding to apply for image warping
    warp_slices: slice
        The slices for extracting a warped image
    warp_lm_edge_anchors: :class:`numpy.ndarray`
        The edge anchors for landmark based warping
    warp_lm_grids: :class:`numpy.ndarray`
        The grids for landmark based warping
    """
    clahe_base_contrast: int
    clahe_chance: float
    clahe_max_size: int
    lab_adjust: np.ndarray
    transform_rotation: int
    transform_zoom: float
    transform_shift: float
    warp_maps: np.ndarray
    warp_pad: tuple[int, int]
    warp_slices: slice
    warp_lm_edge_anchors: np.ndarray
    warp_lm_grids: np.ndarray


class ImageAugmentation():
    """ Performs augmentation on batches of training images.

    Parameters
    ----------
    batchsize: int
        The number of images that will be fed through the augmentation functions at once.
    processing_size: int
        The largest input or output size of the model. This is the size that images are processed
        at.
    config: dict
        The configuration `dict` generated from :file:`config.train.ini` containing the trainer
        plugin configuration options.
    """
    def __init__(self,
                 batchsize: int,
                 processing_size: int,
                 config: dict[str, ConfigValueType]) -> None:
        logger.debug("Initializing %s: (batchsize: %s, processing_size: %s, "
                     "config: %s)",
                     self.__class__.__name__, batchsize, processing_size, config)

        self._processing_size = processing_size
        self._batchsize = batchsize
        self._config = config

        # Warp args
        self._warp_scale = 5 / 256 * self._processing_size  # Normal random variable scale
        self._warp_lm_scale = 2 / 256 * self._processing_size  # Normal random variable scale

        self._constants = self._get_constants()
        logger.debug("Initialized %s", self.__class__.__name__)

    def _get_constants(self) -> AugConstants:
        """ Initializes the caching of constants for use in various image augmentations.

        Returns
        -------
        dict
            Cached constants that are used for various augmentations
        """
        logger.debug("Initializing constants.")

        # Config variables typing check
        shift_range = self._config.get("shift_range", 5)
        color_lightness = self._config.get("color_lightness", 30)
        color_ab = self._config.get("color_ab", 8)
        color_clahe_chance = self._config.get("color_clahe_chance", 50)
        color_clahe_max_size = self._config.get("color_clahe_max_size", 4)
        rotation_range = self._config.get("rotation_range", 10)
        zoom_amount = self._config.get("zoom_amount", 5)

        assert isinstance(shift_range, int)
        assert isinstance(color_lightness, int)
        assert isinstance(color_ab, int)
        assert isinstance(color_clahe_chance, int)
        assert isinstance(color_clahe_max_size, int)
        assert isinstance(rotation_range, int)
        assert isinstance(zoom_amount, int)

        # Transform
        tform_shift = (shift_range / 100) * self._processing_size

        # Color Aug
        amount_l = int(color_lightness) / 100
        amount_ab = int(color_ab) / 100
        lab_adjust = np.array([amount_l, amount_ab, amount_ab], dtype="float32")

        # Random Warp
        warp_range = np.linspace(0, self._processing_size, 5, dtype='float32')
        warp_mapx = np.broadcast_to(warp_range, (self._batchsize, 5, 5)).astype("float32")
        warp_mapy = np.broadcast_to(warp_mapx[0].T, (self._batchsize, 5, 5)).astype("float32")
        warp_pad = int(1.25 * self._processing_size)

        # Random Warp Landmarks
        p_mx = self._processing_size - 1
        p_hf = (self._processing_size // 2) - 1
        edge_anchors = np.array([(0, 0), (0, p_mx), (p_mx, p_mx), (p_mx, 0),
                                 (p_hf, 0), (p_hf, p_mx), (p_mx, p_hf), (0, p_hf)]).astype("int32")
        edge_anchors = np.broadcast_to(edge_anchors, (self._batchsize, 8, 2))
        grids = np.mgrid[0: p_mx: complex(self._processing_size),  # type: ignore
                         0: p_mx: complex(self._processing_size)]  # type: ignore
        retval = AugConstants(clahe_base_contrast=max(2, self._processing_size // 128),
                              clahe_chance=color_clahe_chance / 100,
                              clahe_max_size=color_clahe_max_size,
                              lab_adjust=lab_adjust,
                              transform_rotation=rotation_range,
                              transform_zoom=zoom_amount / 100,
                              transform_shift=tform_shift,
                              warp_maps=np.stack((warp_mapx, warp_mapy), axis=1),
                              warp_pad=(warp_pad, warp_pad),
                              warp_slices=slice(warp_pad // 10, -warp_pad // 10),
                              warp_lm_edge_anchors=edge_anchors,
                              warp_lm_grids=grids)
        logger.debug("Initialized constants: %s", retval)
        return retval

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
        logger.trace("Augmenting color")  # type: ignore
        batch = batch_convert_color(batch, "BGR2LAB")
        self._random_lab(batch)
        self._random_clahe(batch)
        batch = batch_convert_color(batch, "LAB2BGR")
        return batch

    def _random_clahe(self, batch: np.ndarray) -> None:
        """ Randomly perform Contrast Limited Adaptive Histogram Equalization on
        a batch of images """
        base_contrast = self._constants.clahe_base_contrast

        batch_random = np.random.rand(self._batchsize)
        indices = np.where(batch_random < self._constants.clahe_chance)[0]
        if not np.any(indices):
            return
        grid_bases = np.random.randint(self._constants.clahe_max_size + 1,
                                       size=indices.shape[0],
                                       dtype="uint8")
        grid_sizes = (grid_bases * (base_contrast // 2)) + base_contrast
        logger.trace("Adjusting Contrast. Grid Sizes: %s", grid_sizes)  # type: ignore

        clahes = [cv2.createCLAHE(clipLimit=2.0,  # pylint: disable=no-member
                                  tileGridSize=(grid_size, grid_size))
                  for grid_size in grid_sizes]

        for idx, clahe in zip(indices, clahes):
            batch[idx, :, :, 0] = clahe.apply(batch[idx, :, :, 0], )

    def _random_lab(self, batch: np.ndarray) -> None:
        """ Perform random color/lightness adjustment in L*a*b* color space on a batch of
        images """
        randoms = np.random.uniform(-self._constants.lab_adjust,
                                    self._constants.lab_adjust,
                                    size=(self._batchsize, 1, 1, 3)).astype("float32")
        logger.trace("Random LAB adjustments: %s", randoms)  # type: ignore
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
        logger.trace("Randomly transforming image")  # type: ignore

        rotation = np.random.uniform(-self._constants.transform_rotation,
                                     self._constants.transform_rotation,
                                     size=self._batchsize).astype("float32")
        scale = np.random.uniform(1 - self._constants.transform_zoom,
                                  1 + self._constants.transform_zoom,
                                  size=self._batchsize).astype("float32")

        tform = np.random.uniform(-self._constants.transform_shift,
                                  self._constants.transform_shift,
                                  size=(self._batchsize, 2)).astype("float32")
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

        logger.trace("Randomly transformed image")  # type: ignore

    def random_flip(self, batch: np.ndarray):
        """ Perform random horizontal flipping on the passed in batch.

        The probability of flipping an image is set in :file:`config.train.ini`

        Parameters
        ----------
        batch: :class:`numpy.ndarray`
            The batch should be a 4-dimensional array of shape (`batchsize`, `height`, `width`,
            `channels`) and in `BGR` format.
        """
        logger.trace("Randomly flipping image")  # type: ignore
        randoms = np.random.rand(self._batchsize)
        flip_chance = self._config.get("random_flip", 50)
        assert isinstance(flip_chance, int)
        indices = np.where(randoms > flip_chance / 100)[0]
        batch[indices] = batch[indices, :, ::-1]
        logger.trace("Randomly flipped %s images of %s",  # type: ignore
                     len(indices), self._batchsize)

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
        logger.trace("Randomly warping batch")  # type: ignore
        slices = self._constants.warp_slices
        rands = np.random.normal(size=(self._batchsize, 2, 5, 5),
                                 scale=self._warp_scale).astype("float32")
        batch_maps = ne.evaluate("m + r", local_dict={"m": self._constants.warp_maps, "r": rands})
        batch_interp = np.array([[cv2.resize(map_, self._constants.warp_pad)[slices, slices]
                                  for map_ in maps]
                                 for maps in batch_maps])
        warped_batch = np.array([cv2.remap(image, interp[0], interp[1], cv2.INTER_LINEAR)
                                 for image, interp in zip(batch, batch_interp)])

        logger.trace("Warped image shape: %s", warped_batch.shape)  # type: ignore
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
        logger.trace("Randomly warping landmarks")  # type: ignore
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
        maps = grid_z.reshape((self._batchsize,
                               self._processing_size,
                               self._processing_size,
                               2)).astype("float32")
        warped_batch = np.array([cv2.remap(image,
                                           map_[..., 1],
                                           map_[..., 0],
                                           cv2.INTER_LINEAR,
                                           cv2.BORDER_TRANSPARENT)
                                 for image, map_ in zip(batch, maps)])
        logger.trace("Warped batch shape: %s", warped_batch.shape)  # type: ignore
        return warped_batch
