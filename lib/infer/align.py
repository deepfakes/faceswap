#! /usr/env/bin/python3
"""Handles face landmark detection plugins and runners """
from __future__ import annotations

import logging
import typing as T

import cv2
import numpy as np

from lib.align.aligned_face import batch_umeyama
from lib.align.aligned_utils import batch_transform
from lib.align.constants import EXTRACT_RATIOS, LandmarkType, MEAN_FACE
from lib.align.pose import Batch3D
from lib.utils import get_module_objects
from lib.logger import format_array, parse_class_init
from plugins.extract import extract_config as cfg
from plugins.extract.base import ExtractPlugin
from .handler import ExtractHandler
from .objects import ExtractBatch

if T.TYPE_CHECKING:
    import numpy.typing as npt

logger = logging.getLogger(__name__)


class Align(ExtractHandler):
    """Responsible for handling align plugins within the extract pipeline

    Parameters
    ----------
    plugin
        The plugin that this runner is to use
    re_feeds
        Number of times to jitter detection bounding box and average the result. Default: `0`
    re_align
        ``True`` to re-align faces based on their first-pass results. Default: ``False``
    normalization
        The normalization to perform on aligner input images. Default: ``None`` (no normalization)
    filters
        ``True`` to enable aligner filters to filter out faces. Default: ``False``
    compile_model
        ``True`` to compile any PyTorch models
    config_file
        Full path to a custom config file to load. ``None`` for default config
    """
    def __init__(self,
                 plugin: str,
                 re_feeds: int = 0,
                 re_align: bool = False,
                 normalization: T.Literal["none", "clahe", "hist", "mean"] | None = None,
                 filters: bool = False,
                 compile_model: bool = False,
                 config_file: str | None = None) -> None:
        logger.debug(parse_class_init(locals()))
        super().__init__(plugin, compile_model=compile_model, config_file=config_file)
        self._landmark_type: LandmarkType | None = None  # Populate on first plugin output received
        self._re_feed = ReFeed(re_feeds)
        self._normalize = Normalize("none" if normalization is None else normalization)
        self._re_align = ReAlign(re_align, self.plugin, self._re_feed.beta)
        self._filters = AlignedFilter(filters)

    def __repr__(self) -> str:
        """Pretty print for logging"""
        retval = super().__repr__()[:-1]
        retval += (f", re_feeds={self._re_feed._re_feeds}, re_align={self._re_align.enabled}, "
                   f"normalization={repr(self._normalize.name)}, filters={self._filters.enabled})")
        return retval

    # Pre-Processing
    def _clamp_roi(self,
                   batch: ExtractBatch,
                   roi: npt.NDArray[np.int32]) -> npt.NDArray[np.int32]:
        """Adjust the provided ROIs to within frame boundaries

        Parameters
        ----------
        batch
            The batch object that holds the images and ROI co-ordinates for extracting face
            patches for alignments
        roi
            The ROI co-ordinates for extracting face patches for alignments

        Returns
        -------
        The batch ROIs adjusted to fit within the frame's dimensions
        """
        imgs_h_w = np.array([batch.images[i].shape[:2] for i in batch.frame_ids])
        if imgs_h_w.shape[0] != roi.shape[0]:  # Re-feeds
            imgs_h_w = np.repeat(imgs_h_w, self._re_feed.total_feeds, axis=0)
        retval = np.empty_like(roi)
        retval[:, 0] = np.clip(roi[:, 0], 0, imgs_h_w[:, 1] - 1)
        retval[:, 1] = np.clip(roi[:, 1], 0, imgs_h_w[:, 0] - 1)
        retval[:, 2] = np.clip(roi[:, 2], 0, imgs_h_w[:, 1] - 1)
        retval[:, 3] = np.clip(roi[:, 3], 0, imgs_h_w[:, 0] - 1)
        return retval

    @classmethod
    def _get_destinations(cls,
                          original_roi: npt.NDArray[np.int32],
                          clamped_roi: npt.NDArray[np.int32],
                          scales: npt.NDArray[np.float64]) -> npt.NDArray[np.int32]:
        """Provide the destination ROI for resizing the face patch in to the model input

        Parameters
        ----------
        original_roi
            The original square ROIs calculated from a detection bounding box
        clamped_roi
            The same ROIs but with out of bound co-ordinates clamped to frame boundaries
        scales
            The scaling required to take the original ROIs to model input size

        Returns
        -------
        The destination co-ordinates for re-sizing the face box to model input size
        """
        retval = np.empty_like(clamped_roi, dtype=np.int32)
        retval[:, [0, 2]] = (clamped_roi[:, [0, 2]] - original_roi[:, 0, None]) * scales[:, None]
        retval[:, [1, 3]] = (clamped_roi[:, [1, 3]] - original_roi[:, 1, None]) * scales[:, None]
        return retval

    def _crop_and_resize(self,  # pylint:disable=too-many-locals
                         images: list[npt.NDArray[np.uint8]],
                         image_ids: npt.NDArray[np.int32],
                         roi: npt.NDArray[np.int32],
                         destinations: npt.NDArray[np.int32],
                         scales: npt.NDArray[np.float64],
                         is_final: bool) -> np.ndarray:
        """Crop and resize the face images from the ROIs and return as batch at model input size

        Parameters
        ----------
        images
            The images for the batch
        image_ids
            The image indexes that correspond to the batch's ROIs
        roi
            The ROIs required to extract a face from an image
        destinations
            The ROIs that the resized image should be placed on the destination patch
        scales
            The scaling required to take each frame ROI to model input size
        is_final
            ``True`` if this is the final pass through the aligner

        Returns
        -------
        A batch of face patches ready for feeding to an aligner
        """
        num_imgs = len(image_ids)
        total_feeds = self._re_feed.total_feeds if is_final else 1
        batch: np.ndarray = np.zeros((num_imgs,
                                      total_feeds,
                                      self.plugin.input_size,
                                      self.plugin.input_size, 3),
                                     dtype=images[image_ids[0]].dtype)
        roi_reshaped = roi.reshape(num_imgs, -1, 4)
        dest_reshaped = destinations.reshape(num_imgs, -1, 4)
        scales_reshaped = scales.reshape(num_imgs, -1)
        interpolations = np.where(scales_reshaped > 1.0, cv2.INTER_CUBIC, cv2.INTER_AREA)

        for batch_id, (image_id, bboxes, dst) in enumerate(zip(image_ids,
                                                               roi_reshaped,
                                                               dest_reshaped)):
            img = images[image_id]
            img = img[..., 2::-1] if self.plugin.is_rgb else img
            for i, (box, dst) in enumerate(zip(bboxes, dst)):
                out = batch[batch_id, i]
                cv2.resize(img[box[1]:box[3], box[0]:box[2]],
                           (dst[2] - dst[0], dst[3] - dst[1]),
                           dst=out[dst[1]:dst[3], dst[0]:dst[2]],
                           interpolation=interpolations[batch_id, i])
        retval = batch.reshape((-1, self.plugin.input_size, self.plugin.input_size, 3))
        return retval

    def _prepare_images(self,
                        batch: ExtractBatch,
                        roi: npt.NDArray[np.int32],
                        is_final: bool) -> npt.NDArray[np.float32]:
        """Prepare the images from the ROI bounding boxes and model input size for feeding the
        model and populate to the batch's data attribute

        Parameters
        ----------
        batch
            The batch to be fed to the aligner
        roi
            The square ROI from the original image that plugin's face patch should be created from
        is_final
            ``True`` if this is the final pass through the aligner

        Returns
        -------
        The formatted and resized feed images for the plugin
        """
        scale = self.plugin.input_size / batch.matrices[:, 0, 0]
        clamped_roi = self._clamp_roi(batch, roi)
        destinations = self._get_destinations(roi, clamped_roi, scale)
        images = self._crop_and_resize(batch.images,
                                       batch.frame_ids,
                                       clamped_roi,
                                       destinations,
                                       scale,
                                       is_final)
        images = self._normalize(images)
        return self._format_images(images)

    def _matrices_from_roi(self, roi: npt.NDArray[np.int32]) -> npt.NDArray[np.float32]:
        """Convert the ROIs to transformation matrices for mapping predictions back to frame space

        Parameters
        ----------
        roi
            The square (B, left, top, right, bottom) region of interest in the original frame for
            feeding the plugin

        Returns
        -------
        The (B, 3, 3) transformation matrices for taking the ROIs back to frame space
        """
        assert np.all(roi[:, 3] - roi[:, 1] == roi[:, 2] - roi[:, 0]), (
            f"[{self.plugin.name}.pre_process] All ROI bounding boxes for aligner input must "
            "be square")
        retval = np.zeros((roi.shape[0], 3, 3), dtype="float32")
        retval[:, 0, 0] = roi[:, 2] - roi[:, 0]
        retval[:, 1, 1] = roi[:, 3] - roi[:, 1]
        retval[:, 0, 2] = roi[:, 0]
        retval[:, 1, 2] = roi[:, 1]
        retval[:, 2, 2] = 1.0
        return retval

    def _prepare_data(self, batch: ExtractBatch, iteration: int = 1) -> None:
        """Prepare the data, in place, for feeding through the model.

        Parameters
        ----------
        batch
            The aligner batch containing the information required to pre-process data
        iteration
            The iteration that we are on passing through the model. If re-align is not enabled this
            will always be 1. If re-align is enabled this will represent the first or second pass
            through the model
        """
        is_final = iteration == self._re_align.iterations
        # ROIs are adjusted by plugin on first/only pass, otherwise by re-align
        # square crop from frame on first pass. Square Affine from aligned data on 2nd pass
        if iteration == 1:
            # Re-feeds are performed during 2nd pass on aligned bounding box for re-aligns
            boxes = batch.bboxes.copy()
            roi = self.plugin.pre_process(boxes)
            mats = self._matrices_from_roi(roi)
            if is_final and self._re_feed.total_feeds > 1:
                mats, roi = self._re_feed(mats, with_roi=True)
            batch.matrices = mats
            batch.data = self._prepare_images(batch, roi, is_final)
        else:  # If we are here we are re-aligning
            if self._re_feed.total_feeds > 1:
                mats = self._re_feed(self._re_align.default_crop_matrices,
                                     with_roi=False,
                                     size=self.plugin.input_size)
            else:
                mats = self._re_align.default_crop_matrices
            batch.data = self._re_align.get_images(mats, self._re_feed.total_feeds)

    def pre_process(self, batch: ExtractBatch) -> None:
        """Obtain the adjusted square ROIs from the plugin based off the provided detection
        bounding boxes. Crop and size the input face images ready for inference from these ROIs

        Parameters
        ----------
        batch
            The incoming ExtractBatch to use for pre-processing
        """
        self._prepare_data(batch, iteration=1)

    # Processing
    def _get_predictions(self, is_final: bool, feed: np.ndarray) -> np.ndarray:
        """Obtain the predictions from the model. Handles collating any re-feeds

        Parameters
        ----------
        is_final
            ``True`` if this is the final iteration through the plugin
        feed
            The input to the model for the batch.

        Returns
        -------
        The predictions from the model for the provided feed
        """
        batch_size = feed.shape[0]
        if is_final:  # Re-feeds performed on final pass only
            batch_size //= self._re_feed.total_feeds
        results = []
        chunks = self._re_feed.total_feeds if is_final else 1
        for idx in range(chunks):
            start = idx * batch_size
            results.append(self._predict(feed[start: start + batch_size]))

        retval = np.array(results)
        return retval.reshape((feed.shape[0], *retval.shape[2:]))

    def process(self, batch: ExtractBatch) -> None:
        """Perform inference to get results from the aligner

        Parameters
        ----------
        batch
            The incoming ExtractBatch to use for processing
        """
        result = None
        for iteration in range(1, self._re_align.iterations + 1):
            is_final = iteration == self._re_align.iterations

            if is_final and self._re_align.enabled:
                # Need to get prepared aligned images from first-pass output
                self._prepare_data(batch, iteration=iteration)

            assert batch.data is not None
            result = self._get_predictions(is_final, batch.data)

            if is_final and not self._re_align.enabled:  # Nothing left to do. Just the 1 pass
                break

            if self._overridden["post_process"]:  # Must make sure we are final (B, 68, 2) lms
                result = self.plugin.post_process(result)

            self._re_align(batch, result, iteration)  # 1st or 2nd pass re-align op

        assert result is not None
        batch.data = result  # Final pass predictions

    # Post-Processing
    def post_process(self, batch: ExtractBatch) -> None:
        """Post-process the landmark predictions from the model: average any re-feeds, scale back
        to original frame dimensions, apply any filters

        Parameters
        ----------
        batch
            The incoming ExtractBatch to use for post-processing
        """
        result = batch.data
        if self._overridden["post_process"] and not self._re_align.enabled:
            result = self.plugin.post_process(result)
        assert result.dtype == np.float32, (
            f"[{self.plugin.name}.post_process] Landmarks should be a numpy float32 array")

        batch_transform(batch.matrices, result, in_place=True)  # Scale to image space
        landmarks = self._re_feed.merge(result)
        if self._landmark_type is None:
            self._landmark_type = LandmarkType.from_shape(T.cast(tuple[int, int],
                                                                 landmarks.shape[1:]))
            logger.debug("[%s.post_process] Set landmark type to: %s",
                         self.plugin.name, repr(self._landmark_type.name))

        batch.landmarks = landmarks
        batch.landmark_type = self._landmark_type
        self._filters(batch)

    def output_info(self) -> None:
        """Output the counts from the aligner filter"""
        self._filters.output_counts()

    def set_normalize_method(self, method: T.Literal["none", "clahe", "hist", "mean"] | None
                             ) -> None:
        """Update the normalization method with the given method

        Parameters
        ----------
        method
            The normalization method to use
        """
        self._normalize.set_method(method)


class Normalize():
    """Handles the normalization of feed images prior to feeding the model"""
    def __init__(self, method: T.Literal["none", "clahe", "hist", "mean"]) -> None:
        logger.debug(parse_class_init(locals()))
        self.name = method.lower()
        assert self.name in ("none", "clahe", "hist", "mean")
        self._method = None if self.name == "none" else self.name
        self._methods = {"clahe": self._clahe,
                         "hist": self._hist,
                         "mean": self._mean}
        self._clahe_object = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))

    def _clahe(self, images: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        """Perform Contrast Limited Adaptive Histogram Equalization

        Parameters
        ----------
        images
            The images to perform CLAHE normalization on

        Returns
        -------
        The normalized images
        """
        n, h, w, c = images.shape
        reshaped = images.reshape((-1, h, w))  # (N*3, H, W)
        retval = np.empty_like(reshaped)
        for i in range(reshaped.shape[0]):
            retval[i] = self._clahe_object.apply(reshaped[i])
        return retval.reshape((n, h, w, c))

    def _hist(self, images: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        """Perform RGB Histogram Equalization

        Parameters
        ----------
        images
            The images to perform Histogram Equalization on

        Returns
        -------
        The normalized images
        """
        n, h, w, c = images.shape
        reshaped = images.reshape((-1, h, w))  # (N*3, H, W)
        retval = np.empty_like(reshaped)
        for i in range(reshaped.shape[0]):
            retval[i] = cv2.equalizeHist(reshaped[i])
        return retval.reshape((n, h, w, c))

    def _mean(self, images: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        """Normalize each channel to its min/max

        Parameters
        ----------
        images
            The images to mean normalization on

        Returns
        -------
        The normalized images
        """
        imgs = images.astype("float32")
        mins = imgs.min(axis=(1, 2))[:, None, None, :]
        maxes = imgs.max(axis=(1, 2))[:, None, None, :]
        den = np.maximum(maxes - mins, 1e-6)
        out = (imgs - mins) / den * 255.
        return out.astype("uint8")

    def set_method(self, method: T.Literal["none", "clahe", "hist", "mean"] | None) -> None:
        """Update the normalization method with the given method

        Parameters
        ----------
        method
            The normalization method to use
        """
        self.name = "none" if method is None else method.lower()
        assert self.name in ("none", "clahe", "hist", "mean")
        logger.debug("[Align.normalization] Set method to %s", self.name)
        self._method = None if self.name == "none" else self.name

    def __call__(self, images: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        """Perform the selected normalization method on the batch of model input images

        Parameters
        ----------
        images
            The batch of model input images to be normalized

        Returns
        -------
        The given images normalized by the chosen method, or the input batch if no method selected
        """
        if self._method is None:
            return images
        return self._methods[self._method](images)


class ReAlign:
    """Handles re-aligning faces based on first-pass results

    Parameters
    ----------
    enabled
        ``True`` if realigns are to be performed
    plugin
        The plugin that will be processing re-aligns
    margin
        The % amount that re-feed allows bounding box points to drift
    """
    def __init__(self, enabled: bool, plugin: ExtractPlugin, margin: float) -> None:
        logger.debug(parse_class_init(locals()))
        self.enabled = enabled
        """``True`` if re-aligns are enabled"""
        self.iterations = 2 if enabled else 1
        """The total number of iterations through the align process required for the
        selected re-align configuration"""
        self._size = plugin.input_size
        self._expanded_size = int(round(self._size * (1 + 2 * margin)))  # Additional re-feed space
        self._image_scale = plugin.scale
        self._mean_face = MEAN_FACE[LandmarkType.LM_2D_51]

        self._adjust_matrix = self._get_adjust_matrix()
        """Padding and offset for normalized aligned matrix to better represent a face detection
        box"""
        self._default_crop_matrices = self._get_default_matrix()
        """A transform matrix that crops the default (center) image patch out of the expanded image
        patch"""
        self._matrices = np.empty((0, 3, 3), dtype="float32")
        self._images = np.zeros((plugin.batch_size, self._expanded_size, self._expanded_size, 3),
                                dtype=plugin.dtype)

    @property
    def default_crop_matrices(self) -> npt.NDArray[np.float32]:
        """The default crop matrices used for calculating re-feeds"""
        return np.broadcast_to(self._default_crop_matrices, (self._matrices.shape[0], 3, 3))

    def _get_adjust_matrix(self) -> npt.NDArray[np.float32]:
        """Obtain a transformation matrix that applies padding to better represent a face
        detection bounding box location in normalized aligned space for applying to patch space

        Returns
        -------
        The (1, 3, 3) transformation matrix for transforming points from normalized aligned
        space to image patch space
        """
        pad = 0.3  # 30% padding
        retval = np.array([[[1.0 - pad, 0, pad / 2],
                            [0, 1.0 - pad, pad / 2],
                            [0, 0, 1]]], dtype="float32")
        logger.debug("Obtained normalized to image patch matrix: %s", format_array(retval))
        return retval

    def _get_default_matrix(self) -> npt.NDArray[np.float32]:
        """Create the default unit-square to patch-space centered sub-crop from the expanded,
        aligned matrix

        Returns
        -------
        The (N, 3, 3) transformation matrix that takes the central crop in patch space
        """
        offset = (self._expanded_size - self._size) / 2
        retval = np.array([[[1.0, 0, offset],
                            [0, 1.0, offset],
                            [0., 0., 1.]]],
                          dtype="float32")
        logger.debug("Default bounding box: %s", retval)
        return retval

    def get_images(self,  # pylint:disable=too-many-locals
                   matrices: npt.NDArray[np.float32],
                   feeds: int) -> npt.NDArray[np.float32]:
        """Obtain the sub-crops from the main image patches based on the roi stored in the batch
        and populate them to the batch's data attribute

        Parameters
        ----------
        matrices
            The matrices that define the crops to extract from the expanded patch in shape
            (N x total_feeds, 3, 3)
        feeds
            The number of feeds that are to be made through the model for this batch

        Returns
        -------
        The aligned images that are to be used for 2nd pass re-align
        """
        mats = matrices.reshape(-1, feeds, 3, 3)
        all_offsets = np.rint(mats[..., :2, 2]).astype("int32")
        all_scales = mats[..., 0, 0]  # Always same x/y scaling, always aligned
        all_interpolations = np.where(all_scales < 1.0, cv2.INTER_CUBIC, cv2.INTER_AREA)
        all_dims = np.rint(self._size / all_scales).astype(np.int32)  # Always square

        size = (self._size, self._size)
        retval = np.empty((*mats.shape[:2], *size, 3), dtype=self._images.dtype)

        for batch_id, (offsets, scales, interpolations, dims) in enumerate(zip(all_offsets,
                                                                               all_scales,
                                                                               all_interpolations,
                                                                               all_dims)):
            img = self._images[batch_id]
            for feed_id, offset in enumerate(offsets):
                scale = scales[feed_id]
                interpolation = interpolations[feed_id]
                src_dim = dims[feed_id]
                crop = img[offset[1]:offset[1] + src_dim, offset[0]:offset[0] + src_dim]
                if scale != 1.:
                    crop = cv2.resize(crop, size, interpolation=interpolation)
                retval[batch_id, feed_id] = crop

        # Add the adjusted matrices to :attr:`_matrices` for warping back to frame downstream
        base_mats = self._matrices.reshape(self._matrices.shape[0], -1, 3, 3)
        base_mats = base_mats @ mats @ np.diag([self._size, self._size, 1]).astype("float32")
        self._matrices = base_mats.reshape(matrices.shape[0], *base_mats.shape[2:])

        return retval.reshape(matrices.shape[0], *retval.shape[2:])

    def _get_matrix(self,
                    landmarks: npt.NDArray[np.float32],
                    bboxes: npt.NDArray[np.int32],
                    roi_matrices: npt.NDArray[np.float32]) -> np.ndarray:
        """Obtain the (N, 3, 3) transformation matrix to align the landmarks in normalized space
        and add to :attr:`_matrices`

        The matrix:
          - takes the standard matrix that aligns the face/image via umeyama
          - Pads it to better line up with a detection bounding box
          - Adjusts with further padding/offsetting based on the plugin's generated ROI output

        Parameters
        ----------
        landmarks
            The first pass detected landmarks in normalized space
        bboxes
            The original face detection bounding boxes
        roi_matrices
            The original matrices used to map the original square ROIs generated by the plugin back
            to frame space

        Returns
        The (N, 3, 3) transformation matrix that will create an image patch for re-alignment
        """
        # Frame space -> Normalized Space -> Aligned space -> Patch Space
        # normalized -> aligned
        mats = batch_umeyama(landmarks[:, 17:], self._mean_face, True).astype("float32")

        # normalized -> patch
        # Get plugin adjustments
        roi_sizes = roi_matrices[:, 0, 0, None]
        box_sizes = (bboxes[:, 2:] - bboxes[:, :2]).max(axis=1)[..., None]
        bb_to_roi_scales = box_sizes / roi_sizes   # (N, 1)

        roi_center = roi_matrices[:, :2, 2] + (0.5 * roi_sizes)
        bbox_center = (bboxes[:, :2] + bboxes[:, 2:]) / 2.
        bb_to_roi_shifts = (bbox_center - roi_center) / box_sizes

        # Convert plugin adjustment to matrix
        adj_mat = np.repeat(np.eye(3, dtype="float32")[None, :, :], mats.shape[0], axis=0)
        adj_mat[:, 0, 0] = bb_to_roi_scales[:, 0]
        adj_mat[:, 1, 1] = bb_to_roi_scales[:, 0]
        adj_mat[:, :2, 2] = (1 - bb_to_roi_scales) / 2 + bb_to_roi_shifts

        # Combine plugin and default adjustments + scale
        patch_mat = adj_mat @ self._adjust_matrix
        patch_mat[:, :2] *= self._expanded_size

        # Store the matrix that takes expanded space to frame space for updating in get_images
        self._matrices = (roi_matrices @
                          np.linalg.inv(mats) @
                          np.linalg.inv(patch_mat)).astype("float32")
        # Return the matrix that creates the expanded image sub-crop
        return patch_mat @ mats @ np.linalg.inv(roi_matrices)

    def _scale_images(self) -> None:
        """Scale all of the images stored in :attr:`_images` to the correct numeric range """
        if self._image_scale == (0, 255):
            return
        low, high = self._image_scale
        im_range = high - low
        self._images /= (255. / im_range)
        self._images += low

    def _first_pass(self, landmarks: npt.NDArray[np.float32], batch: ExtractBatch) -> None:
        """Process the outputs from the model after the first pass.

        We want to adjust the matrix for any padding and offsets added by the plugin to the
        original detection box. We then store these padded image in :attr:`_images` for sub-
        cropping

        Assumptions:
            - The "default" ROI is a square box along the bbox's longest edge at the same center
            - Padding is how much wider the actual ROI is than this "default" ROI
            - offset is how much the centre of the actual ROI deviates from the "default" ROI
            - A dummy padding 'constant' is added to the matrix to cater for detection box
            'looseness'

        The aim is to end up with a face patch which is about similarly framed to the original
        bbox. A bit of extra padding is added to match with the amount of offset applied by
        re-feed The original 'ROI' will be the square around the center of the image patch that is
        of plugin input size

        Parameters
        ----------
        landmarks
            The (x, y) detected landmarks for a batch in frame space
        batch
            The batch object being processed for re-aligns
        """
        warp_mats = self._get_matrix(landmarks, batch.bboxes, batch.matrices)[:, :2]
        scales = np.sqrt(np.abs(np.linalg.det(warp_mats[:, :, :2])))
        interpolations = np.where(scales < 1.0, cv2.INTER_CUBIC, cv2.INTER_AREA)
        size = (self._expanded_size, self._expanded_size)
        for idx, (frame_id, mat, interpolation) in enumerate(zip(batch.frame_ids,
                                                                 warp_mats,
                                                                 interpolations)):
            img = batch.images[frame_id]
            cv2.warpAffine(img.astype(self._images.dtype),
                           mat,
                           size,
                           dst=self._images[idx],
                           flags=interpolation,
                           borderMode=cv2.BORDER_REPLICATE)
        self._scale_images()

    def _second_pass(self, batch: ExtractBatch) -> None:
        """Add the adjustment matrices to the batch object so downstream can transpose back to
        frame space

        Parameters
        ----------
        batch
            The batch object being processed for re-aligns
        """
        batch.matrices = self._matrices

    def __call__(self,
                 batch: ExtractBatch,
                 landmarks: npt.NDArray[np.float32],
                 iteration: int) -> None:
        """Process the outputs from the plugin when re-aligning data

        Is called twice.
          - First pass: aligns the image based on the first pass landmarks, stores image patches
          that next pass' feed will be generated from and creates ROI boxes for this aligned patch
          - 2nd pass: Rotates detections back to frame alignment and updates the ROI to correctly
          scale and shift the alignments back to frame space downstream

        Parameters
        ----------
        batch
            The batch object being processed for re-aligns
        landmarks
            The (x, y) detected landmarks for a batch in mean-space
        iteration
            The re-align iteration that is being request. Either `1` or `2`
        """
        if not self.enabled:
            return
        assert iteration in (1, 2)
        if iteration == 1:
            self._first_pass(landmarks, batch)
            return
        self._second_pass(batch)


class ReFeed:
    """Handles preparation of images for re-feeding the aligner with minor adjustments to
    detection bounding boxes, and averaging the result at the end.

    Parameters
    ----------
    re_feeds
        The number of re-feeds to be performed.
    """
    def __init__(self, re_feeds: int) -> None:
        logger.debug(parse_class_init(locals()))
        self._re_feeds = re_feeds
        self.beta = 0.05
        """The amount each corner point can move relative to the boxes shortest side"""
        self.total_feeds = re_feeds + 1
        """The total number of feeds through the model for original boxes plus re-feeds"""
        self._corners = np.array([[[0, 0, 1], [1, 1, 1]]], dtype="float32").swapaxes(1, 2)

    @T.overload
    def __call__(self,
                 matrices: npt.NDArray[np.float32],
                 with_roi: T.Literal[False],
                 size: int = 0,) -> npt.NDArray[np.float32]: ...

    @T.overload
    def __call__(self,
                 matrices: npt.NDArray[np.float32],
                 with_roi: T.Literal[True] = True,
                 size: int = 0) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int32]]: ...

    def __call__(self,
                 matrices: npt.NDArray[np.float32],
                 with_roi: bool = False,
                 size: int = 0
                 ) -> npt.NDArray[np.float32] | tuple[npt.NDArray[np.float32],
                                                      npt.NDArray[np.int32]]:
        """Obtain an array of adjusted norm to frame matrices based on the number of re-feed
        iterations that have been selected and the size of the original ROI.

        Parameters
        ----------
        matrices
            A batch of norm to frame transformation matrices to be randomly adjust for re-feeding
            the model in shape (N, 3, 3)
        with_roi
            ``True`` to also return the adjusted ROIs. Default: ``False``
        size
            The size of the image patch that the matrix creates if it cannot be derived from the
            matrices. Default: `0` (derive from matrices)

        Returns
        -------
        matrices
            The adjusted matrices for taking points from normalized to frame space in shape
            ((Num re_feeds * N) + 1, 3, 3), in frame contiguous order (Na, Nb, Nc, Na1, Nb1,
            Nc1...)
        roi
            The ((Num re_feeds * N) + 1, 4) roi for each adjusted feed. Returned if `with_roi` is
            ``True``
        """
        if self._re_feeds == 0:
            raise NotImplementedError
        size_mat = (np.array([size],
                             dtype="float32") if size != 0 else matrices[:, 0, 0])[:, None, None]

        batch_size = matrices.shape[0]
        d_scales = np.random.uniform(1.0 - self.beta,
                                     1.0 + self.beta,
                                     size=(batch_size, self._re_feeds))
        d_shift = size_mat - np.random.uniform(1.0 - self.beta,
                                               1.0 + self.beta,
                                               size=(batch_size, self._re_feeds, 2)) * size_mat

        mats = np.broadcast_to(matrices[:, None], (batch_size, self.total_feeds, 3, 3)).copy()
        mats[:, 1:, (0, 1), (0, 1)] *= d_scales[:, :, None]
        mats[:, 1:, :2, 2] += d_shift
        mats = mats.reshape(-1, 3, 3)
        if not with_roi:
            logger.trace("re-feed. matrices: %s",  # type: ignore[attr-defined]
                         format_array(mats))
            return mats

        tl_br = np.rint((mats @ self._corners).swapaxes(1, 2))
        roi = np.stack([tl_br[:, 0, 0], tl_br[:, 0, 1], tl_br[:, 1, 0], tl_br[:, 1, 1]],
                       axis=1).astype(np.int32)
        logger.trace("re-feed. matrices: %s, roi: %s",  # type: ignore[attr-defined]
                     format_array(mats), format_array(roi))
        return mats, roi

    def merge(self, landmarks: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """If re-feeds enabled return the average result from the re-feeds, otherwise the original
        array

        Parameters
        ----------
        landmarks
            The (N x total_feeds, 68, 2) landmarks from the plugin

        Returns
        -------
        The final (N, 68, 2) landmarks with any re-feeds merged
        """
        if self.total_feeds == 1:
            return landmarks
        lm_shape = landmarks.shape
        rf_shape = (lm_shape[0] // self.total_feeds, self.total_feeds, *lm_shape[1:])
        return landmarks.reshape(rf_shape).mean(axis=1)


class AlignedFilter:  # pylint:disable=too-many-instance-attributes
    """Applies filters to the output of the aligner

    Parameters
    ----------
    enabled
        ``True`` to enable filters. ``False`` to disable
    """
    def __init__(self, enabled: bool) -> None:
        logger.debug(parse_class_init(locals()))
        self._counts: dict[str, int] = {"features": 0, "scale": 0, "distance": 0, "roll": 0}
        self._features = cfg.aligner_features()
        self._min_scale = cfg.aligner_min_scale()
        self._max_scale = cfg.aligner_max_scale()
        self._distance = cfg.aligner_distance() / 100.
        self._roll = cfg.aligner_roll()
        self.enabled = enabled or (not self._features and
                                   self._min_scale <= 0.0 and
                                   self._max_scale <= 0.0 and
                                   self._distance <= 0.0 and
                                   self._roll <= 0.0)
        self._mean_face = MEAN_FACE[LandmarkType.LM_2D_51][None]
        self._expansion = 1.0 - EXTRACT_RATIOS["face"]

    def output_counts(self) -> None:
        """If filters are enabled info log the number of faces filtered"""
        if not self.enabled:
            return
        counts = []
        for key, count in self._counts.items():
            if not count:
                continue
            txt = key.title()
            if key in ("distance", "roll"):
                txt += f" ({getattr(self, f'_{key}')})"
            if key == "scale":
                txt += f" (min: {self._min_scale}, max: {self._max_scale})"
            counts.append(txt + f": {count}")
        if counts:
            logger.info("[Align filter] %s", ", ".join(counts))

    def _handle_filtered(self,
                         key: str,
                         batch: ExtractBatch,
                         mask: npt.NDArray[np.bool]) -> None:
        """Add the filtered item to the filter counts and update the batch object to remove
        filtered faces

        Parameters
        ----------
        key: str
            The key to use for the filter counts dictionary and the sub_folder name
        batch
            The batch object to perform filtering on
        mask
            The mask to apply to filter the faces

        Returns
        -------
        The filtered normalized landmarks
        """
        if np.all(mask):
            return
        self._counts[key] += int(sum(~mask))
        batch.apply_mask(mask)

    def _filter_features(self, landmarks: npt.NDArray[np.float32]) -> npt.NDArray[np.bool]:
        """Filter faces based on the location of relative eye and mouth features

        Parameters
        ----------
        landmarks
            The aligned landmarks in normalized (0. - 1.) space

        Returns
        -------
        Boolean mask indicating faces to keep
        """
        lowest_eyes = np.max(landmarks[:, np.r_[17:27, 36:48], 1], axis=1)
        highest_mouth = np.min(landmarks[:, 48:68, 1], axis=1)
        return (highest_mouth - lowest_eyes) > 0

    def _filter_scale(self, batch: ExtractBatch) -> npt.NDArray[np.bool]:
        """Filter faces based on the scale of the face relative to min/max thresholds.

        Parameters
        ----------
        batch
            The batch object to perform filtering on

        Returns
        -------
        Boolean mask indicating faces to keep
        """
        frames = np.array([i.shape[:2] for i in batch.images]).min(axis=1)
        frame_ids = batch.frame_ids

        linear = batch.aligned.matrices[:, :2, 0]
        sizes = 1.0 / (self._expansion * np.hypot(linear[:, 0], linear[:, 1]))
        mins = (frames * self._min_scale)[frame_ids]
        if self._max_scale:
            maxes = (frames * self._max_scale)[frame_ids]
        else:
            maxes = sizes
        return (mins <= sizes) & (maxes >= sizes)

    def __call__(self, batch: ExtractBatch) -> None:
        """Apply aligner filters to the given batch

        Parameters
        ----------
        batch
            The batch object to perform filtering on with the landmarks populated
        """
        if not self.enabled or batch.landmarks is None:
            return
        if batch.landmark_type not in (LandmarkType.LM_2D_68, LandmarkType.LM_2D_98):
            logger.warning("[Align filter] Filters are not supported for %s landmarks",
                           batch.landmark_type)
            self.enabled = False
            return
        if self._features:
            self._handle_filtered("features",
                                  batch,
                                  self._filter_features(batch.aligned.landmarks_normalized))
        if self._min_scale > 0.0 or self._max_scale > 0.0:
            self._handle_filtered("scale", batch, self._filter_scale(batch))
        if self._distance > 0.0:
            d_msk = np.abs(batch.aligned.landmarks_normalized[:, 17:] -
                           self._mean_face).mean(axis=(1, 2)) <= self._distance
            self._handle_filtered("distance", batch, d_msk)
        if self._roll > 0.0:
            r_msk = np.abs(Batch3D.roll(batch.aligned.rotation)) <= self._roll
            self._handle_filtered("roll", batch, r_msk)


__all__ = get_module_objects(__name__)
