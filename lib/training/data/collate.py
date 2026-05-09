#!/usr/bin/env python3
"""Handles collation of data for training faceswap models"""
from __future__ import annotations

import logging
import typing as T
from dataclasses import dataclass

import cv2
import numpy as np
import torch
from tqdm import tqdm

from lib.align.constants import EXTRACT_RATIOS, LandmarkType, MEAN_FACE
from lib.align.aligned_face import batch_umeyama
from lib.align.aligned_utils import batch_transform
from lib.align.pose import Batch3D
from lib.image import read_image_meta_batch
from lib.logger import format_array, parse_class_init
from lib.utils import FaceswapError, get_module_objects

from .augmentation import ImageAugmentation
from .data_set import get_label, get_sorted_images, to_float32

if T.TYPE_CHECKING:
    import numpy.typing as npt
    from lib.align import CenteringType
    from plugins.train.trainer.base import TrainConfig

logger = logging.getLogger(__name__)


@dataclass
class BatchMeta:
    """Dataclass that holds meta information required for training a batch of images

    All lists are of len(number model outputs per side) with tensors in shape (batch_size,
    num_inputs, 1, H, W)
    """
    mask_face: list[torch.Tensor] | None = None
    """The selected face mask for penalized loss/learn mask for each output in NCHW order"""
    mask_eye: list[torch.Tensor] | None = None
    """The eye mask if eye loss multipliers > 1 for each output in NCHW order"""
    mask_mouth: list[torch.Tensor] | None = None
    """The mouth mask if mouth loss multipliers > 1 for each output in NCHW order"""

    def __repr__(self) -> str:
        """Pretty print for logging"""
        params = ", ".join(f"{k}={None if v is None else [(x.shape, x.dtype) for x in v]}"
                           for k, v in self.__dict__.items())
        return f"{self.__class__.__name__}({params})"

    def __getitem__(self, key: int) -> BatchMeta:
        """Obtain a copy of the BatchMeta object for a specific model input index

        Parameters
        ----------
        key
            The input id to obtain data for

        Returns
        -------
        The meta data for a specific model input. Data will be populated in lists of
        length num_outputs in shape (batch_size, 1, H, W)
        """
        return BatchMeta(**{k: None if v is None else [x[:, key] for x in v]
                            for k, v in self.__dict__.items()})

    def to(self, device: str | torch.Device) -> T.Self:
        """Place all contained tensors onto the given device

        Parameters
        ----------
        device
            The device to place the tensors on to

        Returns
        -------
        This object with the tensors placed on the requested device
        """
        for k in list(self.__dict__):
            v = self.__dict__[k]
            if v is None:
                continue
            self.__dict__[k] = [x.to(device) for x in v]
        return self


class LandmarkMatcher:
    """Prepares landmarks when Warp-to-Landmarks is enabled.

    2 sides (A/B) only.

    For each side, stores the aligned landmarks for each side and collates the 10 nearest matches
    on the other side for random warping

    Parameters
    ----------
    folders
        Two training folders for sides A and B
    size
        The aligned face size to transform the landmarks to
    centering
        The aligned centering to transform the landmarks to
    coverage
        Additional coverage ratio to be applied
    y_offset
        Additional vertical offset to be applied
    num_choices
        Number of choices from the opposite side to cache for each landmark. Default: 10
    """
    def __init__(self,
                 folders: list[str],
                 size: int,
                 centering: CenteringType,
                 coverage: float,
                 y_offset: float,
                 num_choices: int = 10) -> None:
        logger.debug(parse_class_init(locals()))
        assert len(folders) == 2, (
            f"Warp to landmarks is only compatible with 2 inputs. Got {len(folders)}")
        self._folders = folders
        self._size = size
        self._centering: CenteringType = centering
        self._coverage = coverage
        self._y_offset = y_offset
        self._num_choices = num_choices

        self._padding = round(size * (EXTRACT_RATIOS[centering] + coverage - 1) / (2 * coverage))
        self._scale = self._size - (2 * self._padding)
        self._landmarks = self._load_landmarks()

        min_file_count = min([self._landmarks[0].shape[0], self._landmarks[1].shape[0]])
        if self._num_choices > min_file_count:
            self._num_choices = min_file_count - 1
        self._closest_indices = self._get_closest_indices()

    def __repr__(self) -> str:
        """Pretty print for logging"""
        params = {f"{k}"[1:]: v for k, v in self.__dict__.items()
                  if k in ("_folders", "_size", "_centering", "_coverage", "_y_offset",
                           "_num_choices")}
        s_params = ", ".join(f"{k}={repr(v)}" for k, v in params.items())
        return f"{self.__class__.__name__}({s_params})"

    def _landmarks_from_header(self, meta: dict[str, T.Any], filename: str
                               ) -> npt.NDArray[np.float32]:
        """Extract the landmarks from the PNG metadata.

        Returns
        -------
        landmarks
            The frame space landmarks for a face
        filename
            The name of the face image that we are loading landmarks for

        Raises
        ------
        FaceswapError
            If an invalid image is loaded or 68 point landmarks are not used
        """
        if "itxt" not in meta or "alignments" not in meta["itxt"]:
            raise FaceswapError(f"Invalid face image found. Aborting: '{filename}'")

        retval = np.array(meta["itxt"]["alignments"]["landmarks_xy"], dtype=np.float32)
        if LandmarkType.from_shape(retval.shape) != LandmarkType.LM_2D_68:
            raise FaceswapError("68 Point facial Landmarks are required for Warp-to-"
                                f"landmarks. The face that failed was: '{filename}'")
        return retval

    def _align_points(self, points: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Normalize and align the landmarks to model input size/coverage/offset

        points
        ------
        The (N, 68, 2) landmark points to align

        Returns
        -------
        The landmark points aligned to model input
        """
        mats = batch_umeyama(points[:, 17:], MEAN_FACE[LandmarkType.LM_2D_51], True)
        norm_lms = batch_transform(mats, points)

        rotation, translation = Batch3D.solve_pnp(norm_lms)
        offsets = Batch3D.get_offsets(self._centering, rotation, translation)
        if self._y_offset:
            offsets[:, 1] -= self._y_offset
        norm_lms -= offsets[:, None, :]
        norm_lms *= self._scale
        norm_lms += self._padding
        return norm_lms

    def _load_landmarks(self) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        """For each input folder load and align the landmarks for each face

        Returns
        -------
        landmarks_a
            The aligned landmarks for side A in shape (N, 68, 2)
        landmarks_b
            The aligned landmarks for side B in shape (N, 68, 2)
        """
        landmarks: list[npt.NDArray[np.float32]] = []
        for i, folder in enumerate(self._folders):
            side = get_label(i, len(self._folders))
            file_list = get_sorted_images(folder)
            lms = np.empty((len(file_list), 68, 2), dtype=np.float32)
            for filename, meta in tqdm(read_image_meta_batch(file_list),
                                       desc=f"WTL: Caching Landmarks ({side.upper()})",
                                       total=len(file_list),
                                       leave=False):
                lms[file_list.index(filename)] = self._landmarks_from_header(meta, filename)
            landmarks.append(self._align_points(lms))
            logger.debug("[LandmarkMatcher] Got landmarks for side %s: %s",
                         side, format_array(landmarks[-1]))
        return landmarks[0], landmarks[1]

    def _get_closest_indices(self) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
        """Obtain the closest x number of landmarks from the opposite side

        Returns
        -------
        indices_a
            Array of size (len(landmarks_a), x) closest B landmarks for each A landmarks
        indices_b
            Array of size (len(landmarks_b), x) closest A landmarks for each B landmarks
        """
        a_count = self._landmarks[0].shape[0]
        b_count = self._landmarks[1].shape[0]
        lms_a = self._landmarks[0].reshape(a_count, -1)
        lms_b = self._landmarks[1].reshape(b_count, -1)

        a_sq = (lms_a ** 2).sum(axis=1, keepdims=True)
        b_sq = (lms_b ** 2).sum(axis=1, keepdims=True)
        dist2 = a_sq + b_sq.T - 2.0 * (lms_a @ lms_b.T)
        np.clip(dist2, 0, None, out=dist2)
        matches_a = np.argpartition(dist2, self._num_choices, axis=1)[:, :self._num_choices]
        matches_b = np.argpartition(dist2.T, self._num_choices, axis=1)[:, :self._num_choices]

        logger.debug("[TrainLoader] Closest matches. A: %s, B: %s",
                     format_array(matches_a), format_array(matches_b))
        return matches_a, matches_b

    def get_close_landmarks(self, indices: npt.NDArray[np.int64]) -> npt.NDArray[np.float32]:
        """For the given image indices, obtain a randomly selected close match landmarks from the
        other side

        Parameters
        ----------
        indices
            The (num_inputs, landmark_indices) image file indices to obtain the matches for

        Returns
        -------
        2 sets of landmarks in shape (num_sides * batch_size, num_sides, 68, 2) stacked to a batch
        of landmark points for augmentation
        """
        matches = np.zeros((*indices.shape, 2, 68, 2), dtype=np.float32)
        for side_id, ind in enumerate(indices):
            src_lms = self._landmarks[side_id][ind]
            dst_choices = self._closest_indices[side_id][ind]
            idx = np.random.randint(0, dst_choices.shape[1], size=dst_choices.shape[0])
            dst_indices = np.take_along_axis(dst_choices, idx[:, None], axis=1).squeeze(1)
            dst_lms = self._landmarks[1 - side_id][dst_indices]
            matches[side_id, :, 0] = src_lms
            matches[side_id, :, 1] = dst_lms

        retval = matches.reshape((-1, 2, 68, 2)).copy()
        logger.trace("[LandmarkMatcher] matched_points: %s",  # type:ignore[attr-defined]
                     format_array(retval))
        return retval


class Collate:  # pylint:disable=too-many-instance-attributes
    """Collation function for processing a batch of samples into input and output tensors applying
    augmentation

    Parameters
    ----------
    input_size
        The pixel size of the model input
    output_sizes
        The pixel sizes of the model output
    color_order
        The color order that the model expects
    config
        The training configuration for the model
    landmarks
        The landmark matching object for the (A and B) sides of the model if warp_to_landmarks is
        enabled otherwise ``None``
    """
    _mask_types = ("mask_face", "mask_eye", "mask_mouth")
    """The masks that are stacked to the end of the targets in the order they are stacked"""

    def __init__(self,
                 input_size: int,
                 output_sizes: tuple[int, ...],
                 color_order: T.Literal["bgr", "rgb"],
                 config: TrainConfig,
                 landmarks: LandmarkMatcher | None) -> None:
        logger.debug(parse_class_init(locals()))
        self._name = f"{self.__class__.__name__}"
        self._input_size = input_size
        self._output_sizes = output_sizes
        self._color_order = color_order.lower()
        self._config = config

        self._num_inputs = len(config.folders)
        self._batch_size = config.batch_size

        # For Warp to Landmarks
        self._landmarks = landmarks

        self._process_size = max(*output_sizes, input_size)
        self._resize_targets = any(x != self._process_size for x in self._output_sizes)
        self._resize_inputs = self._process_size != self._input_size
        self._aug = ImageAugmentation(batch_size=self._batch_size * self._num_inputs,
                                      processing_size=self._process_size)

    def __repr__(self) -> str:
        """Pretty print for logging"""
        params = {f"{k}"[1:]: format_array(v) if isinstance(v, np.ndarray) else v
                  for k, v in self.__dict__.items()
                  if k in ("_input_size", "_output_sizes", "_color_order",
                           "_config", "_landmarks")}
        s_params = ", ".join(f"{k}={repr(v)}" for k, v in params.items())
        return f"{self.__class__.__name__}({s_params})"

    def _create_targets(self, batch: npt.NDArray[np.uint8]
                        ) -> tuple[list[torch.Tensor], BatchMeta]:
        """ Compile target images, with masks, for the model output sizes.

        Parameters
        ----------
        batch
            This should be a 4-dimensional array of training images in the format (`batch size`,
            `height`, `width`, `channels`). Targets should be requested after performing image
            transformations but prior to performing warps. The 4th channel should be the mask.
            Any channels above the 4th should be any additional area masks (e.g. eye/mouth) that
            are required.

        Returns
        -------
        targets
            List of len (num_outputs) of target images in shape (batch_size, num_inputs, height,
            width, 3) at all model output sizes as float32 0.0 - 1.0 range
        meta
            Any additional Meta information relating to the batch required for training the model
        """
        logger.trace("[%s] Compiling targets: batch shape: %s",  # type:ignore[attr-defined]
                     self._name, batch.shape)
        if self._resize_targets:
            reshaped = [to_float32(batch if batch.shape[1] == size else
                                   np.array([
                                       cv2.resize(image,
                                                  (size, size),
                                                  interpolation=cv2.INTER_AREA)
                                       for image in batch
                                      ])).reshape(self._num_inputs,
                                                  self._batch_size,
                                                  size,
                                                  size,
                                                  -1).swapaxes(0, 1)
                        for size in self._output_sizes]
        else:
            reshaped = [to_float32(batch).reshape(self._num_inputs,
                                                  self._batch_size,
                                                  *batch.shape[1:]).swapaxes(0, 1)
                        for _ in self._output_sizes]

        targets = [torch.from_numpy(out[..., :3]) for out in reshaped]
        masks = BatchMeta(
            **{self._mask_types[idx]: [torch.from_numpy(out[..., 3 + idx][:, :, None, :, :])
                                       for out in reshaped]
               for idx in range(reshaped[0].shape[-1] - 3)})
        logger.trace("[%s] Processed targets: %s, masks: %s",  # type:ignore[attr-defined]
                     self._name, [t.shape for t in targets], masks)
        return targets, masks

    def _get_landmarks_pairs(self, indices: npt.NDArray[np.int64]
                             ) -> npt.NDArray[np.float32] | None:
        """Get a pair of matching source landmarks and closely selected destination landmarks
        for Warp to Landmarks for each of the inputs

        Parameters
        ----------
        indices
            The (num_inputs, batch_size) face file image indices to obtain the landmarks pairs for

        Returns
        -------
        2 sets of landmarks in shape (num_inputs * batch_size, 2, 68, 2). On the 3rd dimension,
        position 0 are the source points. position 1 the randomly selected closest match points.
        ``None`` if Warp to Landmarks is disabled
        """
        if not self._config.warp or self._landmarks is None:
            return None
        assert indices.shape[0] == 2, "Only 2 inputs allowed for WTL"
        return self._landmarks.get_close_landmarks(indices)

    def __call__(self, data: list[tuple[tuple[npt.NDArray[np.uint8], int], ...]]
                 ) -> tuple[list[torch.Tensor], list[torch.Tensor], BatchMeta]:
        """Prepare the loaded samples for feeding the model, creating targets and applying
        augmentation

        Parameters
        ----------
        data
            Batch of data tuples with the loaded stacked image and masks from each loader in the
            first position and the image file index for each item in the batch in the 2nd

        Returns
        -------
        feed
            list of len (num_inputs) tensors of shape(batch_size, H, W, C) inputs for the model
        targets
            List of len (num_outputs) of target images in shape (batch_size, num_inputs, height,
            width, 3) at all model output sizes as float32 0.0 - 1.0 range
        meta
            The meta information for the batch
        """
        shape = data[0][0][0].shape
        batch = np.empty((self._num_inputs, self._batch_size, *shape), dtype=np.uint8)
        indices = np.empty((self._num_inputs, self._batch_size), dtype=np.int64)
        for idx in range(self._num_inputs):
            batch[idx] = [d[0][idx] for d in data]
            indices[idx] = [d[1][idx] for d in data]

        batch = batch.reshape(-1, *shape)
        landmarks = self._get_landmarks_pairs(indices)

        if self._config.augment_color:
            batch[..., :3] = self._aug.color_adjust(batch[..., :3])

        self._aug.transform(batch, landmarks)

        if self._config.flip:
            self._aug.random_flip(batch, landmarks)
        if self._color_order == "rgb":
            batch[..., :3] = batch[..., [2, 1, 0]]

        targets, masks = self._create_targets(batch)

        feed = batch[..., :3]
        if self._config.warp and landmarks is not None and self._landmarks is not None:
            feed = self._aug.warp(feed,
                                  to_landmarks=True,
                                  batch_src_points=landmarks[:, 0],
                                  batch_dst_points=landmarks[:, 1])
        elif self._config.warp:
            feed = self._aug.warp(feed, to_landmarks=False)

        if self._resize_inputs:
            feed = to_float32(np.array([cv2.resize(image,
                                                   (self._input_size, self._input_size),
                                                   interpolation=cv2.INTER_AREA)
                                        for image in feed]))
        else:
            feed = to_float32(feed)

        feed = feed.reshape(self._num_inputs, self._batch_size, *feed.shape[1:])
        inputs = [torch.from_numpy(x) for x in feed]
        return inputs, targets, masks


__all__ = get_module_objects(__name__)
