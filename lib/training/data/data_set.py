#!/usr/bin/env python3
"""Handles Data loading and augmentation for feeding Faceswap Models"""
from __future__ import annotations

import abc
import logging
import os
import typing as T

import cv2
import numexpr as ne

import numpy as np
import torch
from torch.utils.data import Dataset

from lib.align import AlignedFace, Mask
from lib.logger import format_array, parse_class_init
from lib.image import read_image
from lib.utils import FaceswapError, get_module_objects
from plugins.train import train_config as cfg


if T.TYPE_CHECKING:
    import numpy.typing as npt
    from lib.align import CenteringType
    from lib.align.objects import MaskAlignmentsFile, PNGAlignments, PNGHeader
    from lib.align.pose import PoseEstimate

logger = logging.getLogger(__name__)


def to_float32(in_array: npt.NDArray[np.uint8]) -> npt.NDArray[np.float32]:
    """ Cast an UINT8 array in 0-255 range to float32 in 0.0-1.0 range.

    Parameters
    ----------
    in_array
        The input uint8 array

    Returns
    -------
    The array cast to 0.0 - 1.0 float32
    """
    return ne.evaluate("x / c",
                       local_dict={"x": in_array, "c": np.float32(255)},
                       casting="unsafe")


def get_label(index: int, num_identities: int, next_identity: bool = False) -> str:
    """Obtain the label for the given current index. Labels start at A at index 0. Values roll.

    Parameters
    ----------
    index
        The index of the current label
    num_identities
        The number of identities that belong to the label set
    next_identity
        ``True`` to return the next identity for the given index. Default: ``False``

    Returns
    -------
    The current or next label. Labels go A-Z,0-9,a-z
    """
    identities = [chr(i) for i in range(65, 65 + 26)]
    if num_identities > len(identities):
        identities += [chr(i) for i in range(48, 48 + 10)]
    if num_identities > len(identities):
        identities += [chr(i) for i in range(97, 97 + 26)]
    if num_identities > len(identities):
        raise FaceswapError(f"Too many identities: {num_identities}. Max: {len(identities)}")
    identities = identities[:num_identities]
    index = index % num_identities
    if not next_identity:
        return identities[index]
    index += 1 if index + 1 < num_identities else -index
    return identities[index]


def get_sorted_images(folder: str) -> list[str]:
    """For the given folder return the sorted list of potential training images

    Parameters
    ----------
    folder
        The folder containing faceswap training images

    Returns
    -------
    The sorted list of full paths to the training images within the folder
    """
    return list(sorted(os.path.join(folder, f) for f in os.listdir(folder)
                       if os.path.splitext(f)[-1] == ".png"))


class _MaskProcessing:  # pylint:disable=too-many-instance-attributes
    """ Handle the extraction and processing of masks from faceswap PNG headers

    Parameters
    ----------
    side
        The side of the model ("A", "B" etc.)
    size
        The size to return the mask at
    coverage_ratio
        The coverage ratio that the model is using.
    centering
        The centering that the model is trained at
    y_offset
        The amount of vertical offset applied to the training images
    """
    def __init__(self,
                 side: str,
                 size: int,
                 coverage_ratio: float,
                 centering: CenteringType,
                 y_offset: float) -> None:
        logger.debug(parse_class_init(locals()))
        self._side = side.upper()
        self._name = f"{self.__class__.__name__}.{self._side}"
        self._coverage = coverage_ratio
        self._centering: CenteringType = centering
        self._y_offset = y_offset
        self._dims = (size, size)
        self._dilation = cfg.Loss.mask_dilation()
        self._kernel = cfg.Loss.mask_blur_kernel()
        self._threshold = cfg.Loss.mask_threshold()
        self._lm_masks: dict[T.Literal["components", "extended", "eye", "mouth"],
                             T.Literal["face", "face_extended", "eye", "mouth"]] = {
                                 "components": "face",
                                 "extended": "face_extended",
                                 "eye": "eye",
                                 "mouth": "mouth"
                                 }
        self._area_dilatation = 2.5
        self._area_kernel = size // 16

    def __repr__(self) -> str:
        """ Pretty print for logging """
        params = (f"side={repr(self._side)}, size={repr(self._dims[0])}, coverage_ratio="
                  f"{repr(self._coverage)}, centering={repr(self._centering)}, "
                  f"y_offset={repr(self._y_offset)}")
        return f"{self.__class__.__name__}({params})"

    def _check_mask_exists(self, masks: list[str], mask_type: str, filename: str) -> None:
        """ Check that the requested mask exists in the given masks dictionary

        Parameters
        ----------
        masks
            The list of mask keys that exist for the currently processing face
        mask_type
            The requested mask type
        filename
            The name of the extracted face file currently being processed

        Raises
        ------
        FaceswapError
            If the requested mask type is not available an error is returned along with a list
            of available masks
        """
        exist_masks = masks + list(self._lm_masks)
        if mask_type in exist_masks:
            return
        msg = (f"The masks that exist for this face are: {exist_masks}" if exist_masks
               else "No masks exist for this face")
        raise FaceswapError(
            f"You have selected the mask type '{mask_type}' but at least one "
            "face does not contain the selected mask.\n"
            f"The face that failed was: '{filename}'\n{msg}")

    def _get_landmarks_mask(self,
                            mask_type: T.Literal["face", "face_extended", "eye", "mouth"],
                            aligned: AlignedFace) -> npt.NDArray[np.uint8]:
        """Obtain a landmarks based mask directly from the aligned face object

        Parameters
        ----------
        mask_type
            The type of landmarks based mask to obtain
        aligned
            The aligned face object to obtain the mask from

        Returns
        -------
        The requested landmarks based mask
        """
        if mask_type in ("face", "face_extended"):
            dilation = self._dilation
            kernel = self._kernel
            blur_type: T.Literal["gaussian"] | None = None
        else:
            dilation = self._area_dilatation
            kernel = self._area_kernel
            blur_type = "gaussian"
        mask = aligned.get_landmark_mask(mask_type,
                                         dilation=dilation,
                                         blur_kernel=kernel,
                                         blur_type=blur_type)
        return mask

    def _get_face_mask(self, mask_header: MaskAlignmentsFile, pose: PoseEstimate
                       ) -> npt.NDArray[np.uint8]:
        """Obtain a stored face mask from the PNG image header

        Parameters
        ----------
        mask_header
            The stored mask information from the PNG Header
        pose
            The pose estimate for the face

        Returns
        -------
        The requested face mask from the PNG Header
        """
        mask = Mask().from_dict(mask_header)
        mask.set_dilation(self._dilation)
        mask.set_blur_and_threshold(blur_kernel=self._kernel, threshold=self._threshold)
        mask.set_sub_crop(pose.offset[mask.stored_centering],
                          pose.offset[self._centering],
                          self._centering,
                          self._coverage,
                          self._y_offset)
        face_mask = mask.mask
        if face_mask.shape[0] == self._dims[0]:
            retval = face_mask
        else:
            retval = np.empty((*self._dims, 1), dtype=face_mask.dtype)
            interpolator = cv2.INTER_CUBIC if mask.stored_size < self._dims[0] else cv2.INTER_AREA
            cv2.resize(face_mask, self._dims, interpolation=interpolator, dst=retval)
        return retval

    def __call__(self,
                 masks: dict[str, MaskAlignmentsFile],
                 mask_type: str,
                 filename: str,
                 aligned: AlignedFace) -> npt.NDArray[np.uint8]:
        """Obtain the training mask cropped to coverage at maximum model input/output size

        Parameters
        ----------
        masks
            The masks that exist for the extracted face patch
        mask_type
            The type of mask to return
        filename
            The name of the extracted face file currently being processed
        aligned
            The aligned face object for the current face patch

        Returns
        -------
        The mask ready for augmentation
        """
        logger.trace(  # type:ignore[attr-defined]
            "[%s] filename: '%s', mask_type: '%s', masks: %s, aligned: %s",
            self._name, filename, mask_type, masks, aligned)
        self._check_mask_exists(list(masks), mask_type, filename)
        if mask_type in self._lm_masks:
            retval = self._get_landmarks_mask(self._lm_masks[
                T.cast(T.Literal["components", "extended", "eye", "mouth"], mask_type)], aligned)
        else:
            retval = self._get_face_mask(masks[mask_type], aligned.pose)
        logger.trace("[%s] Got mask '%s': %s",  # type:ignore[attr-defined]
                     self._name, mask_type, format_array(retval))
        return retval[..., 0]


class _BaseSet(Dataset, abc.ABC):
    """Base class for Training and Preview dataset loaders to inherit from

    Parameters
    ----------
    side
        The side of the model ("A", "B" etc.)
    image_folder
        Full path to a folder containing training images
    """
    def __init__(self, side: str, image_folder: str) -> None:
        self._image_list = get_sorted_images(image_folder)
        self._side = side.upper()
        self._image_folder = image_folder
        self._name = f"{self.__class__.__name__}.{self._side}"
        self._centering: CenteringType = T.cast("CenteringType", cfg.centering())
        self._coverage = cfg.coverage() / 100.
        self._y_offset = cfg.vertical_offset() / 100.
        self._mask_types = self._get_configured_masks()

    def __repr__(self) -> str:
        """ Pretty print for logging """
        params = f"side={repr(self._side)}, image_folder={repr(self._image_folder)}"
        return f"{self.__class__.__name__}({params})"

    def __len__(self) -> int:
        """Number of items within this dataset"""
        return len(self._image_list)

    @abc.abstractmethod
    def _get_configured_masks(self) -> list[str]:
        """Override to get the required masks

        Returns
        -------
        list of configured masks types in the order [<face mask type>, <eye>, <mouth>]
        """

    def _get_face(self,
                  image: npt.NDArray[np.uint8],
                  alignments: PNGAlignments,
                  size: int,
                  coverage: float) -> AlignedFace:
        """Obtain the face patch cropped to coverage at maximum model input/output size

        Parameters
        ----------
        image
            The original extracted head centered face patch
        alignments
            The alignments meta data for the extracted face patch
        size
            The size to obtain the face object at
        coverage
            The coverage to obtain the face patch for

        Returns
        -------
        The face patch ready for augmentation
        """
        logger.trace("[%s] image: %s alignments: %s",  # type:ignore[attr-defined]
                     self._name, format_array(image), alignments)
        retval = AlignedFace(alignments.landmarks_xy,
                             image=image,
                             centering=self._centering,
                             size=size,
                             coverage_ratio=coverage,
                             y_offset=self._y_offset,
                             dtype="uint8",
                             is_aligned=True)
        logger.trace("[%s] face: %s", self._name, retval)  # type:ignore[attr-defined]
        return retval


class TrainSet(_BaseSet):
    """Base class for Training and Preview dataset loaders to inherit from

    Parameters
    ----------
    side
        The side of the model ("A", "B" etc.)
    image_folder
        Full path to a folder containing training images
    size
        The size to return samples at. This should be the maximum of the model input/output
        size for train sets or the model input size for preview sets
    """
    def __init__(self,
                 side: str,
                 image_folder: str,
                 size: int) -> None:
        logger.debug(parse_class_init(locals()))
        super().__init__(side, image_folder)
        self._size = size
        self._out_shape = (self._size, self._size, 3 + len(self._mask_types))
        self._mask = _MaskProcessing(self._side,
                                     self._size,
                                     self._coverage,
                                     self._centering,
                                     self._y_offset)

    def __repr__(self) -> str:
        """ Pretty print for logging """
        return (f"{super().__repr__()[:-1]}, size={repr(self._size)})")

    def _get_configured_masks(self) -> list[str]:
        """Obtain a list of configured training masks

        Returns
        -------
        list of configured masks types in the order [<face mask type>, <eye>, <mouth>]
        """
        retval = []
        if cfg.Loss.mask_type() is not None and (cfg.Loss.learn_mask() or
                                                 cfg.Loss.penalized_mask_loss()):
            retval.append(cfg.Loss.mask_type())
        if cfg.Loss.penalized_mask_loss() and cfg.Loss.eye_multiplier() > 1:
            retval.append("eye")
        if cfg.Loss.penalized_mask_loss() and cfg.Loss.mouth_multiplier() > 1:
            retval.append("mouth")
        logger.debug("[%s] Configured masks: %s", self._name, retval)
        return retval

    def __getitem__(self, index: int) -> tuple[npt.NDArray[np.uint8], int]:
        """Obtain the next item from the data loader

        Parameters
        ----------
        index
            The image index to return the data for

        Returns
        -------
        image
            The training image and masks for the given index at maximum model input/output size
            stacked into a single array
        index
            The image file index
        """
        filename = self._image_list[index]
        logger.trace("[%s] Loading image %s: %s",  # type:ignore[attr-defined]
                     self._name, index, filename)
        meta: PNGHeader
        image, meta = read_image(filename,
                                 raise_error=False,
                                 with_metadata=True)
        face = self._get_face(image, meta.alignments, self._size, self._coverage)
        img = T.cast("npt.NDArray[np.uint8]", face.face)
        retval = np.empty(self._out_shape, dtype=img.dtype)
        retval[..., :3] = img
        for i, mask_type in enumerate(self._mask_types):
            retval[..., 3 + i] = self._mask(meta.alignments.mask, mask_type, filename, face)

        logger.trace("[%s] images and masks: %s",  # type:ignore[attr-defined]
                     self._name, format_array(retval))
        return retval, index


class PreviewSet(_BaseSet):
    """Preview dataset loader. The dataset loader is responsible for loading images from disk
    and preparing them for inference and display in the model preview

    Parameters
    ----------
    side
        The side of the model ("A", "B" etc.)
    image_folder
        Full path to a folder containing training images
    input_size
        The input size to the model
    output_size
        The largest output size of the model
    color_order
        The color order the model expects data in
    num_images
        Set to 0 for random previews from the image folder. Set to a positive integer for this
        number of images to use for a static timelapse. Default: 0
    """
    def __init__(self,
                 side: str,
                 image_folder: str,
                 input_size: int,
                 output_size: int,
                 color_order: T.Literal["bgr", "rgb"],
                 num_images: int = 0) -> None:
        logger.debug(parse_class_init(locals()))
        super().__init__(side, image_folder)
        self._input_size = input_size
        self._output_size = output_size
        self._color_order = color_order
        self._num_images = num_images
        if num_images and num_images != len(self._image_list):
            logger.debug("[%s] Filtering image list of %s for timelapse: %s",
                         self._name, len(self._image_list), num_images)
            self._image_list = self._image_list[:num_images]

        self._full_size = 2 * int(np.rint((self._output_size / self._coverage) / 2))
        self._mask = _MaskProcessing(self._side,
                                     self._full_size,
                                     1.0,
                                     self._centering,
                                     self._y_offset)

    def __repr__(self) -> str:
        """ Pretty print for logging """
        params = (f"input_size={self._input_size}, output_size={self._output_size}, "
                  f"color_order={repr(self._color_order)}, num_images={self._num_images}")
        return f"{super().__repr__()[:-1]}, {params})"

    def _get_configured_masks(self) -> list[str]:
        """Obtain the preview mask type if it has been selected

        Returns
        -------
        list of configured masks types in the order [<face mask type>, <eye>, <mouth>]
        """
        retval = []
        if cfg.Loss.mask_type() is not None and (cfg.Loss.learn_mask() or
                                                 cfg.Loss.penalized_mask_loss()):
            retval.append(cfg.Loss.mask_type())
        logger.debug("[%s] Configured masks: %s", self._name, retval)
        return retval

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Obtain the next item from the preview data loader

        Parameters
        ----------
        index
            The image index to return the data for

        Returns
        -------
        feed
            A feed image for preview
        target
            An output face at full coverage with the mask in the 4th channel
        """
        filename = self._image_list[index]
        logger.trace("[%s] Loading image %s: %s",  # type:ignore[attr-defined]
                     self._name, index, filename)
        meta: PNGHeader
        image, meta = read_image(filename,
                                 raise_error=False,
                                 with_metadata=True)

        in_face = self._get_face(image, meta.alignments, self._input_size, self._coverage)
        in_img = T.cast("npt.NDArray[np.uint8]", in_face.face)
        out_face = self._get_face(image, meta.alignments, self._full_size, 1.0)
        out_img = np.empty((self._full_size, self._full_size, 4), dtype=np.uint8)
        out_img[..., :3] = T.cast("npt.NDArray[np.uint8]", out_face.face)

        if self._mask_types:
            out_img[..., 3] = self._mask(meta.alignments.mask,
                                         self._mask_types[0],
                                         filename,
                                         out_face)
        else:
            out_img[..., 3] = np.zeros_like(out_img[..., 0])[..., None] + 255

        if self._color_order == "rgb":
            in_img[..., :3] = in_img[..., [2, 1, 0]]
            out_img[..., :3] = out_img[..., [2, 1, 0]]

        feed = torch.from_numpy(to_float32(in_img))
        target = torch.from_numpy(to_float32(out_img))
        logger.trace("[%s] feed: %s (%s), target: %s (%s)",  # type:ignore[attr-defined]
                     self._name, feed.shape, feed.dtype, target.shape, target.dtype)
        return feed, target


class MultiDataset(Dataset):
    """Handles processing data for models with multiple inputs. The length is set as the largest
    dataset. Shuffling all datasets is handled internally at the end of each

    Parameters
    ----------
    datasets
        The input specific datasets for feeding the model
    is_random
        ``True`` if data from each of the datasets should be read randomly. ``False`` if all
        datasets should return the item for the given index
    """
    def __init__(self, datasets: tuple[_BaseSet, ...], is_random: bool = True) -> None:
        super().__init__()
        self._datasets = datasets
        self._len = max(len(d) for d in datasets)

        self._remainder = [np.empty(0, dtype=np.int64)] * len(self._datasets)
        self._indices = self._shuffle_indices()
        self._is_random = is_random

    def __repr__(self) -> str:
        """ Pretty print for logging """
        params = f"datasets={self._datasets}, is_random={self._is_random}"
        return f"{self.__class__.__name__}({params})"

    def __len__(self):
        """Number of items within the largest dataset"""
        return self._len

    def _shuffle_indices(self) -> npt.NDArray[np.int64]:
        """At the end of each epoch build a new indices array for each input. The permutations
        for each input are calculated for it's own data length, and random indices are rolled at
        the end of each largest epoch to ensure that all data sources have their full list
        processed prior to reshuffling

        Returns
        -------
        An array of indices of shape (num_datasets, len(self)) of random indices that can be looked
        up for each value given to __get_item__
        """
        retval = np.empty((len(self._datasets), self._len), dtype=np.int64)
        for idx, ds in enumerate(self._datasets):
            ds_len = len(ds)
            filled = 0
            remainder = self._remainder[idx]
            if len(remainder):
                take = min(len(remainder), self._len)
                retval[idx, :take] = remainder[:take]
                filled = take
                self._remainder[idx] = remainder[take:]

            while filled < self._len:
                perm = np.random.permutation(ds_len)
                take = min(ds_len, self._len - filled)
                retval[idx, filled:filled + take] = perm[:take]
                filled += take
                if take < ds_len:
                    self._remainder[idx] = perm[take:]

        logger.debug("[MultiDataset] Shuffled dataset indices: %s", format_array(retval))
        return retval

    def shuffle(self) -> None:
        """Shuffle all of the contained dataset's data"""
        self._indices = self._shuffle_indices()

    def __getitem__(self, index: int) -> tuple[np.ndarray, ...]:
        """Obtain the next item from each of the contained datasets

        Returns
        -------
        tuple of arrays of shape (num_inputs, ...) for each input dataset's output
        """
        if self._is_random:
            results: list[tuple[np.ndarray, ...]] = [dataset[self._indices[i][index]]
                                                     for i, dataset in enumerate(self._datasets)]
        else:
            results = [dataset[index] for dataset in self._datasets]

        retval = tuple(np.stack([res[i] for res in results])
                       for i in range(len(results[0])))
        return retval


__all__ = get_module_objects(__name__)
