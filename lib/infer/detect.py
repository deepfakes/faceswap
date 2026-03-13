#! /usr/env/bin/python3
"""Handles face detection plugins and runners """
from __future__ import annotations

import logging
import typing as T

import cv2
import numpy as np


from lib.logger import format_array, parse_class_init
from lib.utils import get_module_objects

from .objects import ExtractBatch
from .handler import ExtractHandler

if T.TYPE_CHECKING:
    import numpy.typing as npt

logger = logging.getLogger(__name__)


class Detect(ExtractHandler):
    """Responsible for handling Detection plugins within the extract pipeline

    Parameters
    ----------
    plugin
        The plugin that this runner is to use
    rotation | None
        The rotation arguments. Either a list of angles between 0 and 360 to rotate at or a single
        step size. Default: ``None``, no rotations
    min_size
        Minimum percentage of the frame's shortest edge to accept as a successful detection along
        the detection's longest edge Default: `0` (accept all detections)
    max_size
        Maximum percentage of the frame's shortest edge to accept as a successful detection along
        the detection's longest edge Default: `0` (accept all detections)
    compile_model
        ``True`` to compile any PyTorch models
    config_file
        Full path to a custom config file to load. ``None`` for default config
    """
    def __init__(self,
                 plugin: str,
                 rotation: str | None = None,
                 min_size: int = 0,
                 max_size: int = 0,
                 compile_model: bool = False,
                 config_file: str | None = None) -> None:
        logger.debug(parse_class_init(locals()))
        super().__init__(plugin, compile_model=compile_model, config_file=config_file)
        self._rotation = rotation
        self._rotator = Rotator(rotation, self.plugin.input_size)
        """Responsible for rotating feed images for the model"""
        self._empty_bbox = np.empty((0, 4), dtype="float32")
        """An empty detection result, that will never be used so only needs to be created once"""
        self._min_size = min_size / 100.
        """The user selected shortest frame dim multiplier to accept for minimum size"""
        self._max_size = max_size / 100.
        """The user selected shortest frame dim multiplier to accept for maximum size"""
        self._filter_counts = 0

    def __repr__(self) -> str:
        """Pretty print for logging"""
        retval = super().__repr__()[:-1]
        retval += (f", rotation={repr(self._rotation)}, min_size={int(self._min_size * 100)}, "
                   f"max_size={int(self._max_size * 100)})")
        return retval

    # Pre-processing
    def _get_matrices(self,
                      images: list[npt.NDArray[np.uint8]],
                      filenames: list[str]) -> npt.NDArray[np.float32]:
        """Calculate the scales and padding required to take each image in this batch to model
        input size and store the matrices in the batch object

        Parameters
        ----------
        images
            The images to obtain the matrices for
        filenames : list[str]
            The corresponding file names of the images

        Returns
        -------
        The transformation matrices for taking the images to model input size
        """
        orig_wh = np.array([x.shape[:2] for x in images])[:, ::-1]
        scales = self.plugin.input_size / orig_wh.max(axis=1)
        new_wh = np.rint(orig_wh * scales[:, None]).astype(np.int32)
        pad_xy = (self.plugin.input_size - new_wh) // 2

        retval = np.zeros((len(scales), 3, 3), dtype="float32")
        retval[:, 0, 0] = scales
        retval[:, 1, 1] = scales
        retval[:, 0, 2] = pad_xy[:, 0]
        retval[:, 1, 2] = pad_xy[:, 1]
        retval[:, 2, 2] = 1.

        logger.trace(  # type:ignore[attr-defined]
            "[%s_pre_process] filenames: %s, matrices: %s",
            self.plugin.name, filenames, format_array(retval))
        return retval

    def _scale_images(self,
                      images: list[npt.NDArray[np.uint8]],
                      matrices: npt.NDArray[np.float32]) -> npt.NDArray[np.uint8]:
        """Scale the image and pad to given size

        Parameters
        ----------
        images
            The images to scale
        matrices
            The corresponding warp matrices for scaling the images

        Returns
        -------
        The scaled images
        """
        retval = np.zeros((len(images), self.plugin.input_size, self.plugin.input_size, 3),
                          dtype=images[0].dtype)
        interpolators = np.where(matrices[:, 0, 0] < 1.0, cv2.INTER_AREA, cv2.INTER_CUBIC)
        dims = (self.plugin.input_size, self.plugin.input_size)
        warp_mats = matrices[:, :2]
        for idx, (image, mat, interpolator) in enumerate(zip(images, warp_mats, interpolators)):
            image = image[..., 2::-1] if self.plugin.is_rgb else image
            cv2.warpAffine(image, mat, dims, dst=retval[idx], flags=interpolator)
        logger.trace("Resized batch shape: %s", retval.shape)  # type:ignore[attr-defined]
        return retval

    def pre_process(self, batch: ExtractBatch) -> None:
        """Perform pre-processing for detection plugins.

        - Gets the scale and padding to take the batch of images to model input size
        - Formats the image to the correct color order, dtype and scale for the plugin
        - Performs any plugin specific pre-processing

        Parameters
        ----------
        batch
            The incoming ExtractBatch to use for pre-processing
        """
        batch.matrices = self._get_matrices(batch.images, batch.filenames)
        images = self._scale_images(batch.images, batch.matrices)
        images = self._format_images(images)
        batch.data = self.plugin.pre_process(images)

    # Processing
    def _process_rotations(self,
                           predictions: npt.NDArray[np.float32],
                           mask_requires: npt.NDArray[np.bool_],
                           indices_angle: npt.NDArray[np.int32],
                           box_list: list[npt.NDArray[np.float32] | None],
                           rotation_index: int) -> None:
        """Process the output after a rotation, and store the discovered boxes and the angle index
        they were discovered at

        Parameters
        ----------
        predictions
            The predictions from the model
        mask_requires
            The mask indicating which frames can still be allocated bounding boxes
        indices_angle
            The array that stores the angle index that each frame's faces was found at
        box_list
            The list of final bounding boxes to be output
        rotation_index
            The current angle index we are iterating
        """
        bboxes = (self.plugin.post_process(predictions) if self._overridden["post_process"]
                  else predictions)
        mask_found = np.array([np.any(n) for n in bboxes], dtype="bool")
        indices_requires = np.flatnonzero(mask_requires)
        indices_angle[indices_requires[mask_found]] = rotation_index
        mask_requires[indices_requires[mask_found]] = False
        for i, box in zip(indices_requires[mask_found], bboxes):
            box_list[i] = box

    def process(self, batch: ExtractBatch) -> None:
        """Obtain the output from the plugin's model.

        Executes the plugin's predict function and stores the output prior to post-processing.

        If rotations have been selected, plugin post-processing is done as part of this process as
        the computed bounding boxes are required for re-feeding the model future rotations

        Parameters
        ----------
        batch
            The incoming ExtractBatch to use for processing
        """
        process = "process"
        input_images = batch.data
        batch_size = input_images.shape[0]
        box_list: list[None | np.ndarray] = [None for _ in range(batch_size)]
        boxes: np.ndarray | None = None
        indices_angle = np.zeros((batch_size, ), dtype="int32")

        idx = 0
        mask_requires = np.array([True for _ in range(batch_size)])
        while True:
            feed = self._rotator.rotate(idx, input_images[mask_requires])
            if feed is None:
                logger.trace(  # type:ignore[attr-defined]
                    "[%s.%s] No faces found in %s image(s) of %s after %s rotations: %s",
                    self.plugin.name,
                    process,
                    mask_requires.sum(),
                    batch_size,
                    idx,
                    batch.filenames)

                break
            result = self._predict(feed)
            if not self._rotator.enabled:
                # Not rotating. Do post-processing in next thread
                boxes = result
                break

            # We are rotating, so we have to do post-processing here, to re-feed model
            self._process_rotations(result, mask_requires, indices_angle, box_list, idx)
            if not np.any(mask_requires):
                logger.trace(  # type:ignore[attr-defined]
                    "[%s.%s] Found faces for all %s images after %s rotations: %s",
                    self.plugin.name,
                    process,
                    batch_size,
                    idx + 1,
                    batch.filenames)
                break
            idx += 1

        boxes = (np.array([self._empty_bbox if b is None else b for b in box_list],
                          dtype="object")
                 if boxes is None else boxes)
        batch.data = np.empty(2, dtype="object")
        batch.data[0] = indices_angle
        batch.data[1] = boxes

    # Post-Processing
    def _stack_boxes(self,
                     batch: ExtractBatch,
                     predictions: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Stack the detected boxes into a single array, remove any zero sized boxes, collate the
        indexing information and add to batch

        Parameters
        ----------
        batch
            The detector batch being processed
        predictions
            The face detection bounding boxes received from the plugin

        Returns
        -------
        The stacked detection boxes from all frames in the batch.
        """
        valid = np.fromiter((i for i, p in enumerate(predictions) if np.any(p)), dtype=np.int32)
        if not valid.size:
            batch.frame_ids = valid
            return self._empty_bbox

        result = [predictions[i] for i in valid]
        lengths = np.fromiter((a.shape[0] for a in result), dtype=np.int32)
        batch.frame_ids = np.repeat(valid, lengths)
        return np.vstack(result).astype(np.float32)

    def _scale_boxes(self, batch: ExtractBatch, predictions: npt.NDArray[np.float32]) -> None:
        """Scale the detected faces back out to original image size, round to int and add to the
        batch object

        Parameters
        ----------
        batch
            The detector batch being processed
        predictions
            The stacked face detection predictions at model input size
        """
        if not batch.frame_ids.size:
            return
        mats = batch.matrices[batch.frame_ids]

        predictions[:, [0, 2]] -= mats[:, 0, 2][:, None]
        predictions[:, [1, 3]] -= mats[:, 1, 2][:, None]
        predictions /= mats[:, 0, 0][:, None]
        np.rint(predictions, out=predictions)
        batch.bboxes = predictions.astype("int32")
        logger.trace("[%s.out] Finalized batch: %s",  # type:ignore[attr-defined]
                     self.plugin.name,
                     batch)

    def _filter_boxes(self, batch: ExtractBatch) -> None:
        """Filter out any detections that are smaller or larger than :attr:`_min_size` and
        :attr:`_max_size` along their longest edge

        Parameters
        ----------
        batch
            The detector batch being processed with fully scaled bounding boxes
        """
        if not self._min_size and not self._max_size:
            return

        frames = np.array([i.shape[:2] for i in batch.images]).min(axis=1)
        sizes = np.maximum(batch.bboxes[:, 2] - batch.bboxes[:, 0],
                           batch.bboxes[:, 3] - batch.bboxes[:, 1])

        mins = (frames * self._min_size).astype("int32")[batch.frame_ids]
        if self._max_size:
            maxes = (frames * self._max_size).astype("int32")[batch.frame_ids]
        else:
            maxes = sizes

        keep = np.nonzero(np.logical_and(mins <= sizes, maxes >= sizes))[0]
        if len(keep) == len(sizes):
            return

        logger.debug(
            "[%s.out] Removing %s face(s) from %s detections as outside size thresholds (min: %s, "
            "max: %s): %s",
            self.plugin.name,
            len(sizes) - len(keep),
            len(sizes),
            int(self._min_size * 100),
            int(self._max_size * 100),
            batch.filenames)

        batch.bboxes = batch.bboxes[keep]
        batch.frame_ids = batch.frame_ids[keep]
        self._filter_counts += len(sizes) - len(keep)

    def post_process(self, batch: ExtractBatch) -> None:
        """Perform detection post processing.

        If no rotations were requested, any plugin post-processing will be done here.

        Detection boxes are:
          - stacked into a single array
          - scaled back to frame dimensions,
          - filtered for faces which fall outside min/max thresholds
          - Added to the batch object along with frame to face mapping information.

        Parameters
        ----------
        batch
            The incoming ExtractBatch to use for post-processing
        """
        indices_angle, result = batch.data
        if self._overridden["post_process"] and not self._rotator.enabled:
            result = self.plugin.post_process(result)
        else:
            self._rotator.un_rotate(indices_angle, result)
        result = self._stack_boxes(batch, result)
        self._scale_boxes(batch, result)
        self._filter_boxes(batch)

    def output_info(self) -> None:
        """Output the counts of filtered items """
        if not self._filter_counts:
            return
        logger.info("[Detect filter] Scale (min: %s, max: %s): %s",
                    f"{int(self._min_size * 100)}%",
                    f"{int(self._max_size * 100)}%",
                    self._filter_counts)


class Rotator:
    """Handles pre-calculation of rotation matrices when rotation angles are requested and
    rotating images for feeding the detector. Handles reversing the rotation for any found
    detection bounding boxes.

    Parameters
    ----------
    rotation
        List of requested rotation angles in degrees provided in command line arguments
    image_size
        The size of the square image to obtain rotation matrices for
    """
    def __init__(self, angles: str | None, image_size: int) -> None:
        logger.debug(parse_class_init(locals()))
        self._size = image_size
        self._angles = self._get_angles(angles)
        self._matrices = self._pre_compute_matrices()
        self._matrices_inverse = self._pre_compute_inverse_matrices()
        self._channels_first: bool | None = None
        self.enabled = len(self._angles) > 1
        """``True`` if rotations are to be performed """

    @classmethod
    def _angles_from_step(cls, step_size: int) -> npt.NDArray[np.float32]:
        """Obtain the required rotation angles when the cli argument has been passed in as a step
        size

        Parameters
        ----------
        step_size
            The requested step size

        Returns
        -------
        The rotation angles between 0 and 360 for the given step size
        """
        retval = np.arange(0, 360, step_size, dtype="float32")
        logger.debug("Setting rotation angles to %s from step size: %s", retval, step_size)
        return retval

    def _get_angles(self, rotation: str | None) -> npt.NDArray[np.float32]:
        """Set the rotation angles.

        Parameters
        ----------
        rotation
            List of requested rotation angles in degrees provided in command line arguments

        Returns
        -------
        The complete list of rotation angles to apply in degrees
        """
        if not rotation:
            logger.debug("Not setting rotation angles")
            return np.array([0], dtype=np.float32)

        passed_angles = [int(angle) for angle in rotation.split(",") if int(angle) != 0]
        if len(passed_angles) == 1:
            return self._angles_from_step(passed_angles[0])

        retval = np.array([0] + passed_angles, dtype=np.float32)
        logger.debug("Setting rotation angles to %s from given: %s", retval, rotation)
        return retval

    def _pre_compute_matrices(self) -> npt.NDArray[np.float32]:
        """Pre-compute the rotation matrices required to perform the requested rotations for the
        given square image size

        Returns
        -------
        The rotation matrices for the requested rotation angles
        """
        theta = np.deg2rad(self._angles)
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        cx = (self._size - 1) / 2.0
        cy = (self._size - 1) / 2.0

        matrices = np.zeros((len(self._angles), 2, 3), dtype=np.float32)
        matrices[:, 0, 0] = cos_t
        matrices[:, 0, 1] = -sin_t
        matrices[:, 1, 0] = sin_t
        matrices[:, 1, 1] = cos_t
        matrices[:, 0, 2] = (1 - cos_t) * cx + sin_t * cy
        matrices[:, 1, 2] = (1 - cos_t) * cy - sin_t * cx
        logger.debug("Precomputed rotation matrices: %s", matrices.tolist())
        return matrices

    def _pre_compute_inverse_matrices(self) -> npt.NDArray[np.float32]:
        """Pre-compute the inverse rotation matrices required to perform translation from rotated
        bounding boxes back to original frame

        Returns
        -------
        The rotation matrices for the requested rotation angles
        """
        rot = self._matrices[:, :, :2]
        trans = self._matrices[:, :, 2]
        rot_inv = np.transpose(rot, (0, 2, 1))
        trans_inv = -np.einsum('nij, nj->ni', rot_inv, trans)
        retval = np.concatenate([rot_inv, trans_inv[..., None]], axis=2)
        logger.debug("Precomputed inverse rotation matrices: %s", retval.tolist())
        return retval

    def rotate(self, rotation_index: int, images: np.ndarray) -> np.ndarray | None:
        """Rotate a batch of images by the matrix provided by the given rotation index. Attempts
        to detect and handle channels first images as well as channels last

        Parameters
        ----------
        rotation_index
            The matrix to use. This will be an incrementing index from an enumerated loop that
            selects through the matrices stored for each angle
        images
            The original, correctly orientated, batch of images to rotate

        Returns
        -------
        The batch of image rotated by the angle identified by the given rotation index.
        ``None`` if the given rotation index is invalid
        """
        if rotation_index == 0:
            return images
        if rotation_index >= len(self._angles):
            return None

        if self._channels_first is None:
            self._channels_first = images.shape[1] in (1, 3, 4)
            logger.debug("Set channels_first to %s", self._channels_first)

        if self._channels_first:
            images = images.transpose(0, 2, 3, 1)

        retval = np.empty(images.shape, images.dtype)
        mat = self._matrices[rotation_index]
        size = (self._size, self._size)

        for i, img in enumerate(images):
            cv2.warpAffine(img,
                           mat,
                           size,
                           dst=retval[i],
                           borderMode=cv2.BORDER_REPLICATE)

        if self._channels_first:
            retval = retval.transpose(0, 3, 1, 2)

        return retval

    def un_rotate(self,
                  indices_angle: npt.NDArray[np.int32],
                  roi: npt.NDArray[np.float32]) -> None:
        """Un-rotate the given bounding boxes for the given angle indices and update in place

        Parameters
        ----------
        indices_angle
            The angle indices that correlate to the angle each roi was rotated to to obtain the
            result
        roi
            Ragged array of (B, N, 4) detected bounding discovered at the corresponding angle
            index
        """
        mask_needs_rotate = indices_angle > 0
        if not np.any(mask_needs_rotate):
            return

        indices_needs_rotate = np.flatnonzero(mask_needs_rotate)
        matrices = self._matrices_inverse[indices_angle[mask_needs_rotate]]

        for pred_idx, mat in zip(indices_needs_rotate, matrices):
            bboxes = roi[pred_idx]
            pts = np.empty((bboxes.shape[0], 4, 2), dtype="float32")
            pts[:, 0] = bboxes[:, [0, 1]]  # lt
            pts[:, 1] = bboxes[:, [2, 1]]  # rt
            pts[:, 2] = bboxes[:, [2, 3]]  # rb
            pts[:, 3] = bboxes[:, [0, 3]]  # lb

            pts = pts @ mat[:, :2].T + mat[:, 2]

            # boxes must align on (x, y) planes
            bboxes[:, 0] = pts[..., 0].min(axis=1)
            bboxes[:, 1] = pts[..., 1].min(axis=1)
            bboxes[:, 2] = pts[..., 0].max(axis=1)
            bboxes[:, 3] = pts[..., 1].max(axis=1)


__all__ = get_module_objects(__name__)
