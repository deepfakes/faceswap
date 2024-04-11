#!/usr/bin/env python3
""" Base class for Face Masker plugins

Plugins should inherit from this class

See the override methods for which methods are required.

The plugin will receive a :class:`~plugins.extract.extract_media.ExtractMedia` object.

For each source item, the plugin must pass a dict to finalize containing:

>>> {"filename": <filename of source frame>,
>>>  "detected_faces": <list of bounding box dicts from lib/plugins/extract/detect/_base>}
"""
from __future__ import annotations
import logging
import typing as T

from dataclasses import dataclass, field

import cv2
import numpy as np

from tensorflow.python.framework import errors_impl as tf_errors  # pylint:disable=no-name-in-module  # noqa

from lib.align import AlignedFace, LandmarkType, transform_image
from lib.utils import FaceswapError
from plugins.extract import ExtractMedia
from plugins.extract._base import BatchType, ExtractorBatch, Extractor

if T.TYPE_CHECKING:
    from collections.abc import Generator
    from queue import Queue
    from lib.align import DetectedFace
    from lib.align.aligned_face import CenteringType

logger = logging.getLogger(__name__)


@dataclass
class MaskerBatch(ExtractorBatch):
    """ Dataclass for holding items flowing through the aligner.

    Inherits from :class:`~plugins.extract._base.ExtractorBatch`

    Parameters
    ----------
    roi_masks: list
        The region of interest masks for the batch
    """
    detected_faces: list[DetectedFace] = field(default_factory=list)
    roi_masks: list[np.ndarray] = field(default_factory=list)
    feed_faces: list[AlignedFace] = field(default_factory=list)


class Masker(Extractor):  # pylint:disable=abstract-method
    """ Masker plugin _base Object

    All Masker plugins must inherit from this class

    Parameters
    ----------
    git_model_id: int
        The second digit in the github tag that identifies this model. See
        https://github.com/deepfakes-models/faceswap-models for more information
    model_filename: str
        The name of the model file to be loaded

    Other Parameters
    ----------------
    configfile: str, optional
        Path to a custom configuration ``ini`` file. Default: Use system configfile

    See Also
    --------
    plugins.extract.pipeline : The extraction pipeline for calling plugins
    plugins.extract.align : Aligner plugins
    plugins.extract._base : Parent class for all extraction plugins
    plugins.extract.detect._base : Detector parent class for extraction plugins.
    plugins.extract.align._base : Aligner parent class for extraction plugins.
    """

    _logged_lm_count_once = False

    def __init__(self,
                 git_model_id: int | None = None,
                 model_filename: str | None = None,
                 configfile: str | None = None,
                 instance: int = 0,
                 **kwargs) -> None:
        logger.debug("Initializing %s: (configfile: %s)", self.__class__.__name__, configfile)
        super().__init__(git_model_id,
                         model_filename,
                         configfile=configfile,
                         instance=instance,
                         **kwargs)
        self.input_size = 256  # Override for model specific input_size
        self.coverage_ratio = 1.0  # Override for model specific coverage_ratio

        # Override if a specific type of landmark data is required:
        self.landmark_type: LandmarkType | None = None

        self._plugin_type = "mask"
        self._storage_name = self.__module__.rsplit(".", maxsplit=1)[-1].replace("_", "-")
        self._storage_centering: CenteringType = "face"  # Centering to store the mask at
        self._storage_size = 128  # Size to store masks at. Leave this at default
        logger.debug("Initialized %s", self.__class__.__name__)

    def _maybe_log_warning(self, face: AlignedFace) -> None:
        """ Log a warning, once, if we do not have full facial landmarks

        Parameters
        ----------
        face: :class:`~lib.align.aligned_face.AlignedFace`
            The aligned face object to test the landmark type for
        """
        if face.landmark_type != LandmarkType.LM_2D_4 or self._logged_lm_count_once:
            return

        msg = "are likely to be sub-standard"
        msg = "can not be be generated" if self.name in ("Components", "Extended") else msg

        logger.warning("Extracted faces do not contain facial landmark data. '%s' masks %s.",
                       self.name, msg)
        self._logged_lm_count_once = True

    def get_batch(self, queue: Queue) -> tuple[bool, MaskerBatch]:
        """ Get items for inputting into the masker from the queue in batches

        Items are returned from the ``queue`` in batches of
        :attr:`~plugins.extract._base.Extractor.batchsize`

        Items are received as :class:`~plugins.extract.extract_media.ExtractMedia` objects and
        converted to ``dict`` for internal processing.

        To ensure consistent batch sizes for masker the items are split into separate items for
        each :class:`~lib.align.DetectedFace` object.

        Remember to put ``'EOF'`` to the out queue after processing
        the final batch

        Outputs items in the following format. All lists are of length
        :attr:`~plugins.extract._base.Extractor.batchsize`:

        >>> {'filename': [<filenames of source frames>],
        >>>  'detected_faces': [[<lib.align.DetectedFace objects]]}

        Parameters
        ----------
        queue : queue.Queue()
            The ``queue`` that the plugin will be fed from.

        Returns
        -------
        exhausted, bool
            ``True`` if queue is exhausted, ``False`` if not
        batch, :class:`~plugins.extract._base.ExtractorBatch`
            The batch object for the current batch
        """
        exhausted = False
        batch = MaskerBatch()
        idx = 0
        while idx < self.batchsize:
            item = self.rollover_collector(queue)
            if item == "EOF":
                logger.trace("EOF received")  # type: ignore
                exhausted = True
                break
            # Put frames with no faces into the out queue to keep TQDM consistent
            if not item.detected_faces:
                self._queues["out"].put(item)
                continue
            for f_idx, face in enumerate(item.detected_faces):

                image = item.get_image_copy(self.color_format)
                roi = np.ones((*item.image_size[:2], 1), dtype="float32")

                if not item.is_aligned:
                    # Add the ROI mask to image so we can get the ROI mask with a single warp
                    image = np.concatenate([image, roi], axis=-1)

                feed_face = AlignedFace(face.landmarks_xy,
                                        image=image,
                                        centering=self._storage_centering,
                                        size=self.input_size,
                                        coverage_ratio=self.coverage_ratio,
                                        dtype="float32",
                                        is_aligned=item.is_aligned)

                self._maybe_log_warning(feed_face)

                assert feed_face.face is not None
                if not item.is_aligned:
                    # Split roi mask from feed face alpha channel
                    roi_mask = feed_face.split_mask()
                else:
                    # We have to do the warp here as AlignedFace did not perform it
                    roi_mask = transform_image(roi,
                                               feed_face.matrix,
                                               feed_face.size,
                                               padding=feed_face.padding)

                batch.roi_masks.append(roi_mask)
                batch.detected_faces.append(face)
                batch.feed_faces.append(feed_face)
                batch.filename.append(item.filename)
                idx += 1
                if idx == self.batchsize:
                    frame_faces = len(item.detected_faces)
                    if f_idx + 1 != frame_faces:
                        self._rollover = ExtractMedia(
                            item.filename,
                            item.image,
                            detected_faces=item.detected_faces[f_idx + 1:],
                            is_aligned=item.is_aligned)
                        logger.trace("Rolled over %s faces of %s to next batch "  # type:ignore
                                     "for '%s'", len(self._rollover.detected_faces), frame_faces,
                                     item.filename)
                    break
        if batch:
            logger.trace("Returning batch: %s",  # type:ignore
                         {k: len(v) if isinstance(v, (list, np.ndarray)) else v
                          for k, v in batch.__dict__.items()})
        else:
            logger.trace(item)  # type:ignore
        return exhausted, batch

    def _predict(self, batch: BatchType) -> MaskerBatch:
        """ Just return the masker's predict function """
        assert isinstance(batch, MaskerBatch)
        assert self.name is not None
        try:
            # slightly hacky workaround to deal with landmarks based masks:
            if self.name.lower() in ("components", "extended"):
                feed = np.empty(2, dtype="object")
                feed[0] = batch.feed
                feed[1] = batch.feed_faces
            else:
                feed = batch.feed

            batch.prediction = self.predict(feed)
            return batch
        except tf_errors.ResourceExhaustedError as err:
            msg = ("You do not have enough GPU memory available to run detection at the "
                   "selected batch size. You can try a number of things:"
                   "\n1) Close any other application that is using your GPU (web browsers are "
                   "particularly bad for this)."
                   "\n2) Lower the batchsize (the amount of images fed into the model) by "
                   "editing the plugin settings (GUI: Settings > Configure extract settings, "
                   "CLI: Edit the file faceswap/config/extract.ini)."
                   "\n3) Enable 'Single Process' mode.")
            raise FaceswapError(msg) from err

    def finalize(self, batch: BatchType) -> Generator[ExtractMedia, None, None]:
        """ Finalize the output from Masker

        This should be called as the final task of each `plugin`.

        Pairs the detected faces back up with their original frame before yielding each frame.

        Parameters
        ----------
        batch : dict
            The final ``dict`` from the `plugin` process. It must contain the `keys`:
            ``detected_faces``, ``filename``, ``feed_faces``, ``roi_masks``

        Yields
        ------
        :class:`~plugins.extract.extract_media.ExtractMedia`
            The :attr:`DetectedFaces` list will be populated for this class with the bounding
            boxes, landmarks and masks for the detected faces found in the frame.
        """
        assert isinstance(batch, MaskerBatch)
        for mask, face, feed_face, roi_mask in zip(batch.prediction,
                                                   batch.detected_faces,
                                                   batch.feed_faces,
                                                   batch.roi_masks):
            if self.name in ("Components", "Extended") and not np.any(mask):
                # Components/Extended masks can return empty when called from the manual tool with
                # 4 Point ROI landmarks
                continue
            self._crop_out_of_bounds(mask, roi_mask)
            face.add_mask(self._storage_name,
                          mask,
                          feed_face.adjusted_matrix,
                          feed_face.interpolators[1],
                          storage_size=self._storage_size,
                          storage_centering=self._storage_centering)
        del batch.feed

        logger.trace("Item out: %s",  # type: ignore
                     {key: val.shape if isinstance(val, np.ndarray) else val
                                      for key, val in batch.__dict__.items()})
        for filename, face in zip(batch.filename, batch.detected_faces):
            self._output_faces.append(face)
            if len(self._output_faces) != self._faces_per_filename[filename]:
                continue

            output = self._extract_media.pop(filename)
            output.add_detected_faces(self._output_faces)
            self._output_faces = []
            logger.trace("Yielding: (filename: '%s', image: %s, "  # type:ignore
                         "detected_faces: %s)", output.filename, output.image_shape,
                         len(output.detected_faces))
            yield output

    # <<< PROTECTED ACCESS METHODS >>> #
    @classmethod
    def _resize(cls, image: np.ndarray, target_size: int) -> np.ndarray:
        """ resize input and output of mask models appropriately """
        height, width, channels = image.shape
        image_size = max(height, width)
        scale = target_size / image_size
        if scale == 1.:
            return image
        method = cv2.INTER_CUBIC if scale > 1. else cv2.INTER_AREA  # pylint:disable=no-member
        resized = cv2.resize(image, (0, 0), fx=scale, fy=scale, interpolation=method)
        resized = resized if channels > 1 else resized[..., None]
        return resized

    @classmethod
    def _crop_out_of_bounds(cls, mask: np.ndarray, roi_mask: np.ndarray) -> None:
        """ Un-mask any area of the predicted mask that falls outside of the original frame.

        Parameters
        ----------
        masks: :class:`numpy.ndarray`
            The predicted masks from the plugin
        roi_mask: :class:`numpy.ndarray`
            The roi mask. In frame is white, out of frame is black
        """
        if np.all(roi_mask):
            return  # The whole of the face is within the frame
        roi_mask = roi_mask[..., None] if mask.ndim == 3 else roi_mask
        mask *= roi_mask
