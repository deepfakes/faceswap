#!/usr/bin/env python3
""" Processes the augmentation of images for feeding into a Faceswap model. """
from __future__ import annotations
import logging
from dataclasses import dataclass

import cv2
import numexpr as ne
import numpy as np
from scipy.interpolate import griddata

from lib.image import batch_convert_color
from lib.logger import parse_class_init
from lib.utils import get_module_objects
from plugins.train.trainer import trainer_config as cfg


logger = logging.getLogger(__name__)


@dataclass
class ConstantsColor:
    """ Dataclass for holding constants for enhancing an image (ie contrast/color adjustment)

    Parameters
    ----------
    clahe_base_contrast : int
        The base number for Contrast Limited Adaptive Histogram Equalization
    clahe_chance : float
        Probability to perform Contrast Limited Adaptive Histogram Equilization
    clahe_max_size : int
        Maximum clahe window size
    lab_adjust : :class:`numpy.ndarray`
        Adjustment amounts for L*A*B augmentation
    """
    clahe_base_contrast: int
    """ int : The base number for Contrast Limited Adaptive Histogram Equalization """
    clahe_chance: float
    """ float : Probability to perform Contrast Limited Adaptive Histogram Equilization """
    clahe_max_size: int
    """ int : Maximum clahe window size"""
    lab_adjust: np.ndarray
    """ :class:`numpy.ndarray` : Adjustment amounts for L*A*B augmentation """


@dataclass
class ConstantsTransform:
    """ Dataclass for holding constants for transforming an image

    Parameters
    ----------
    rotation : int
        Rotation range for transformations
    zoom : float
        Zoom range for transformations
    shift : float
        Shift range for transformations
    """
    rotation: int
    """ int : Rotation range for transformations """
    zoom: float
    """ float : Zoom range for transformations """
    shift: float
    """ float : Shift range for transformations """
    flip: float
    """ float : The chance to flip an image """


@dataclass
class ConstantsWarp:
    """ Dataclass for holding constants for warping an image

    Parameters
    ----------
    maps : :class:`numpy.ndarray`
        The stacked (x, y) mappings for image warping
    pad : tuple[int, int]
        The padding to apply for image warping
    slices : slice
        The slices for extracting a warped image
    lm_edge_anchors : :class:`numpy.ndarray`
        The edge anchors for landmark based warping
    lm_grids : :class:`numpy.ndarray`
        The grids for landmark based warping
    """
    maps: np.ndarray
    """ :class:`numpy.ndarray` : The stacked (x, y) mappings for image warping """
    pad: tuple[int, int]
    """ :tuple[int, int] : The padding to apply for image warping """
    slices: slice
    """ slice : The slices for extracting a warped image """
    scale: float
    """ float : The scaling to apply to standard warping """
    lm_edge_anchors: np.ndarray
    """ :class:`numpy.ndarray` : The edge anchors for landmark based warping """
    lm_grids: np.ndarray
    """ :class:`numpy.ndarray` : The grids for landmark based warping """
    lm_scale: float
    """ float : The scaling to apply to landmark based warping """

    def __repr__(self) -> str:
        """ Display shape/type information for arrays in __repr__ """
        params = {k: f"array[shape: {v.shape}, dtype: {v.dtype}]"
                  if isinstance(v, np.ndarray) else v
                  for k, v in self.__dict__.items()}
        str_params = ", ".join(f"{k}={v}" for k, v in params.items())
        return f"{self.__class__.__name__}({str_params})"


@dataclass
class ConstantsAugmentation:
    """ Dataclass for holding constants for Image Augmentation.

    Attributes
    ----------
    color : :class:`ConstantsColor`
        The constants for adjusting color/contrast in an image
    transform : :class:`ConstantsTransform`
        The constants for image transformation
    warp : :class:`ConstantsTransform`
        The constants for image warping

    Dataclass should be initialized using its :func:`from_config` method:

    Example
    -------
    >>> constants = ConstantsAugmentation.from_config(processing_size=256,
    ...                                               batch_size=16)
    """
    color: ConstantsColor
    """ :class:`ConstantsColor` : The constants for adjusting color/contrast in an image """
    transform: ConstantsTransform
    """ :class:`ConstantsTransform` : The constants for image transformation """
    warp: ConstantsWarp
    """ :class:`ConstantsTransform` : The constants for image warping """

    @classmethod
    def _get_clahe(cls, size: int) -> tuple[int, float, int]:
        """ Get the CLAHE constants from user config

        Parameters
        ----------
        size : int
            The size of image to augment the data for

        Returns
        -------
        clahe_base_contrast : int
            The base number for Contrast Limited Adaptive Histogram Equalization
        clahe_chance : float
            Probability to perform Contrast Limited Adaptive Histogram Equilization
        clahe_max_size : int
            Maximum clahe window size
        """
        clahe_base_contrast = max(2, size // 128)
        clahe_chance = cfg.color_clahe_chance() / 100
        clahe_max_size = cfg.color_clahe_max_size()
        logger.debug("clahe_base_contrast: %s, clahe_chance: %s, clahe_max_size: %s",
                     clahe_base_contrast, clahe_chance, clahe_max_size)
        return clahe_base_contrast, clahe_chance, clahe_max_size

    @classmethod
    def _get_lab(cls) -> np.ndarray:
        """ Load the random L*A*B augmentation constants

        Returns
        -------
        :class:`numpy.ndarray`
            Adjustment amounts for L*A*B augmentation
        """
        amount_l = cfg.color_lightness() / 100.
        amount_ab = cfg.color_ab() / 100.

        lab_adjust = np.array([amount_l, amount_ab, amount_ab], dtype="float32")
        logger.debug("lab_adjust: %s", lab_adjust)
        return lab_adjust

    @classmethod
    def _get_color(cls, size: int) -> ConstantsColor:
        """ Get the image enhancements constants from user config

        Parameters
        ----------
        size : int
            The size of image to augment the data for

        Returns
        -------
        :class:`ConstantsColor`
            The constants for image enhancement
        """
        clahe_base_contrast, clahe_chance, clahe_max_size = cls._get_clahe(size)
        retval = ConstantsColor(clahe_base_contrast=clahe_base_contrast,
                                clahe_chance=clahe_chance,
                                clahe_max_size=clahe_max_size,
                                lab_adjust=cls._get_lab())
        logger.debug(retval)
        return retval

    @classmethod
    def _get_transform(cls, size: int) -> ConstantsTransform:
        """ Load the random transform constants

        Parameters
        ----------
        size : int
            The size of image to augment the data for

        Returns
        -------
        :class:`ConstantsTransform`
            The constants for image transformation
        """
        retval = ConstantsTransform(rotation=cfg.rotation_range(),
                                    zoom=cfg.zoom_amount() / 100.,
                                    shift=(cfg.shift_range() / 100.) * size,
                                    flip=cfg.flip_chance() / 100.)
        logger.debug(retval)
        return retval

    @classmethod
    def _get_warp_to_landmarks(cls, size: int, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
        """ Load the warp-to-landmarks augmentation constants

        Parameters
        ----------
        size : int
            The size of image to augment the data for
        batch_size : int
            The batch size that augmented data is being prepared for

        Returns
        -------
        edge_anchors : :class:`numpy.ndarray`
            The edge anchors for landmark based warping
        grids : :class:`numpy.ndarray`
            The grids for landmark based warping
        """
        p_mx = size - 1
        p_hf = (size // 2) - 1
        edge_anchors = np.array([(0, 0), (0, p_mx), (p_mx, p_mx), (p_mx, 0),
                                 (p_hf, 0), (p_hf, p_mx), (p_mx, p_hf), (0, p_hf)]).astype("int32")
        edge_anchors = np.broadcast_to(edge_anchors, (batch_size, 8, 2))
        grids = np.mgrid[0: p_mx: complex(size),  # type:ignore[misc]  # pylint:disable=no-member
                         0: p_mx: complex(size)].astype("float32")  # type:ignore[misc]

        logger.debug("edge_anchors: (%s, %s), grids: (%s, %s)",
                     edge_anchors.shape, edge_anchors.dtype,
                     grids.shape, grids.dtype)  # pylint:disable=no-member
        return edge_anchors, grids

    @classmethod
    def _get_warp(cls, size: int, batch_size: int) -> ConstantsWarp:
        """ Load the warp augmentation constants

        Parameters
        ----------
        size: int
            The size of image to augment the data for
        batch_size : int
            The batch size that augmented data is being prepared for

        Returns
        -------
        :class:`ConstantsTransform`
            The constants for image warping
        """
        lm_edge_anchors, lm_grids = cls._get_warp_to_landmarks(size, batch_size)

        warp_range = np.linspace(0, size, 5, dtype='float32')
        warp_mapx = np.broadcast_to(warp_range, (batch_size, 5, 5)).astype("float32")
        warp_mapy = np.broadcast_to(warp_mapx[0].T, (batch_size, 5, 5)).astype("float32")
        warp_pad = int(1.25 * size)

        retval = ConstantsWarp(maps=np.stack((warp_mapx, warp_mapy), axis=1),
                               pad=(warp_pad, warp_pad),
                               slices=slice(warp_pad // 10, -warp_pad // 10),
                               scale=5 / 256 * size,  # Normal random variable scale
                               lm_edge_anchors=lm_edge_anchors,
                               lm_grids=lm_grids,
                               lm_scale=2 / 256 * size)  # Normal random variable scale
        logger.debug(retval)
        return retval

    @classmethod
    def from_config(cls,
                    processing_size: int,
                    batch_size: int) -> ConstantsAugmentation:
        """ Create a new dataclass instance from user config

        Parameters
        ----------
        processing_size : int:
            The size of image to augment the data for
        batch_size : int
            The batch size that augmented data is being prepared for
        """
        logger.debug("Initializing %s(processing_size=%s, batch_size=%s)",
                     cls.__name__, processing_size, batch_size)
        retval = cls(color=cls._get_color(processing_size),
                     transform=cls._get_transform(processing_size),
                     warp=cls._get_warp(processing_size, batch_size))
        logger.debug(retval)
        return retval


class ImageAugmentation():
    """ Performs augmentation on batches of training images.

    Parameters
    ----------
    batch_size : int
        The number of images that will be fed through the augmentation functions at once.
    processing_size: int
        The largest input or output size of the model. This is the size that images are processed
        at.
    """
    def __init__(self, batch_size: int, processing_size: int) -> None:
        logger.debug(parse_class_init(locals()))
        self._processing_size = processing_size
        self._batch_size = batch_size
        self._constants = ConstantsAugmentation.from_config(processing_size, batch_size)
        logger.debug("Initialized %s", self.__class__.__name__)

    def __repr__(self) -> str:
        """ Pretty print this object """
        return (f"{self.__class__.__name__}(batch_size={self._batch_size}, "
                f"processing_size={self._processing_size})")

    # <<< COLOR AUGMENTATION >>> #
    def _random_lab(self, batch: np.ndarray) -> None:
        """ Perform random color/lightness adjustment in L*a*b* color space on a batch of
        images

        Parameters
        ----------
        batch : :class:`numpy.ndarray`
            The batch should be a 4-dimensional array of shape (`batchsize`, `height`, `width`,
            `3`) and in `BGR` format of uint8 dtype.
        """
        randoms = np.random.uniform(-self._constants.color.lab_adjust,
                                    self._constants.color.lab_adjust,
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

    def _random_clahe(self, batch: np.ndarray) -> None:
        """ Randomly perform Contrast Limited Adaptive Histogram Equalization on
        a batch of images

        Parameters
        ----------
        batch : :class:`numpy.ndarray`
            The batch should be a 4-dimensional array of shape (`batchsize`, `height`, `width`,
            `3`) and in `BGR` format of uint8 dtype.
        """
        base_contrast = self._constants.color.clahe_base_contrast

        batch_random = np.random.rand(self._batch_size)
        indices = np.where(batch_random < self._constants.color.clahe_chance)[0]
        if not np.any(indices):
            return
        grid_bases = np.random.randint(self._constants.color.clahe_max_size + 1,
                                       size=indices.shape[0],
                                       dtype="uint8")
        grid_sizes = (grid_bases * (base_contrast // 2)) + base_contrast
        logger.trace("Adjusting Contrast. Grid Sizes: %s", grid_sizes)  # type:ignore[attr-defined]

        clahes = [cv2.createCLAHE(clipLimit=2.0,
                                  tileGridSize=(grid_size, grid_size))
                  for grid_size in grid_sizes]

        for idx, clahe in zip(indices, clahes):
            batch[idx, :, :, 0] = clahe.apply(batch[idx, :, :, 0], )

    def color_adjust(self, batch: np.ndarray) -> np.ndarray:
        """ Perform color augmentation on the passed in batch.

        The color adjustment parameters are set in :file:`config.train.ini`

        Parameters
        ----------
        batch : :class:`numpy.ndarray`
            The batch should be a 4-dimensional array of shape (`batchsize`, `height`, `width`,
            `3`) and in `BGR` format of uint8 dtype.

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

    # <<< IMAGE AUGMENTATION >>> #
    def transform(self, batch: np.ndarray):
        """ Perform random transformation on the passed in batch.

        The transformation parameters are set in :file:`config.train.ini`

        Parameters
        ----------
        batch : :class:`numpy.ndarray`
            The batch should be a 4-dimensional array of shape (`batchsize`, `height`, `width`,
            `channels`) and in `BGR` format.
        """
        logger.trace("Randomly transforming image")  # type:ignore[attr-defined]
        rotation = np.random.uniform(-self._constants.transform.rotation,
                                     self._constants.transform.rotation,
                                     size=self._batch_size).astype("float32")
        scale = np.random.uniform(1 - self._constants.transform.zoom,
                                  1 + self._constants.transform.zoom,
                                  size=self._batch_size).astype("float32")

        tform = np.random.uniform(-self._constants.transform.shift,
                                  self._constants.transform.shift,
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
        batch : :class:`numpy.ndarray`
            The batch should be a 4-dimensional array of shape (`batchsize`, `height`, `width`,
            `channels`) and in `BGR` format.
        """
        logger.trace("Randomly flipping image")  # type:ignore[attr-defined]
        randoms = np.random.rand(self._batch_size)
        indices = np.where(randoms <= self._constants.transform.flip)[0]
        batch[indices] = batch[indices, :, ::-1]
        logger.trace("Randomly flipped %s images of %s",  # type:ignore[attr-defined]
                     len(indices), self._batch_size)

    def _random_warp(self, batch: np.ndarray) -> np.ndarray:
        """ Randomly warp the input batch

        Parameters
        ----------
        batch : :class:`numpy.ndarray`
            The batch should be a 4-dimensional array of shape (`batchsize`, `height`, `width`,
            `3`) and in `BGR` format.

        Returns
        ----------
        :class:`numpy.ndarray`
            A 4-dimensional array of the same shape as :attr:`batch` with warping applied.
        """
        logger.trace("Randomly warping batch")  # type:ignore[attr-defined]
        slices = self._constants.warp.slices
        rands = np.random.normal(size=(self._batch_size, 2, 5, 5),
                                 scale=self._constants.warp.scale).astype("float32")
        batch_maps = ne.evaluate("m + r", local_dict={"m": self._constants.warp.maps, "r": rands})

        batch_interp = np.array([[cv2.resize(map_, self._constants.warp.pad)[slices, slices]
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

        batch : :class:`numpy.ndarray`
            The batch should be a 4-dimensional array of shape (`batchsize`, `height`, `width`,
            `3`) and in `BGR` format.
        batch_src_points : :class:`numpy.ndarray`
            A batch of 68 point landmarks for the source faces. This is a 3-dimensional array in
            the shape (`batchsize`, `68`, `2`).
        batch_dst_points : :class:`numpy.ndarray`
            A batch of randomly chosen closest match destination faces landmarks. This is a
            3-dimensional array in the shape (`batchsize`, `68`, `2`).

        Returns
        ----------
        :class:`numpy.ndarray`
            A 4-dimensional array of the same shape as :attr:`batch` with warping applied.
        """
        logger.trace("Randomly warping landmarks")  # type:ignore[attr-defined]
        edge_anchors = self._constants.warp.lm_edge_anchors
        grids = self._constants.warp.lm_grids

        batch_dst = batch_dst_points + np.random.normal(size=batch_dst_points.shape,
                                                        scale=self._constants.warp.lm_scale)

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

    def warp(self,
             batch: np.ndarray,
             to_landmarks: bool = False,
             batch_src_points: np.ndarray | None = None,
             batch_dst_points: np.ndarray | None = None
             ) -> np.ndarray:

        """ Perform random warping on the passed in batch by one of two methods.

        Parameters
        ----------
        batch : :class:`numpy.ndarray`
            The batch should be a 4-dimensional array of shape (`batchsize`, `height`, `width`,
            `3`) and in `BGR` format.
        to_landmarks : bool, optional
            If ``False`` perform standard random warping of the input image. If ``True`` perform
            warping to semi-random similar corresponding landmarks from the other side. Default:
            ``False``
        batch_src_points : :class:`numpy.ndarray`, optional
            Only used when :attr:`to_landmarks` is ``True``. A batch of 68 point landmarks for the
            source faces. This is a 3-dimensional array in the shape (`batchsize`, `68`, `2`).
            Default: ``None``
        batch_dst_points : :class:`numpy.ndarray`, optional
            Only used when :attr:`to_landmarks` is ``True``. A batch of randomly chosen closest
            match destination faces landmarks. This is a 3-dimensional array in the shape
            (`batchsize`, `68`, `2`). Default ``None``

        Returns
        ----------
        :class:`numpy.ndarray`
            A 4-dimensional array of the same shape as :attr:`batch` with warping applied.
        """
        if to_landmarks:
            assert batch_src_points is not None
            assert batch_dst_points is not None
            return self._random_warp_landmarks(batch, batch_src_points, batch_dst_points)
        return self._random_warp(batch)


__all__ = get_module_objects(__name__)
