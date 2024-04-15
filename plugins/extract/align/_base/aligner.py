#!/usr/bin/env python3
""" Base class for Face Aligner plugins

All Aligner Plugins should inherit from this class.
See the override methods for which methods are required.

The plugin will receive a :class:`~plugins.extract.extract_media.ExtractMedia` object.

For each source item, the plugin must pass a dict to finalize containing:

>>> {"filename": [<filename of source frame>],
>>>  "landmarks": [list of 68 point face landmarks]
>>>  "detected_faces": [<list of DetectedFace objects>]}
"""
from __future__ import annotations
import logging
import typing as T

from dataclasses import dataclass, field
from time import sleep

import cv2
import numpy as np

from tensorflow.python.framework import errors_impl as tf_errors  # pylint:disable=no-name-in-module # noqa

from lib.align import LandmarkType
from lib.utils import FaceswapError
from plugins.extract import ExtractMedia
from plugins.extract._base import BatchType, ExtractorBatch, Extractor
from .processing import AlignedFilter, ReAlign

if T.TYPE_CHECKING:
    from collections.abc import Generator
    from queue import Queue
    from lib.align import DetectedFace
    from lib.align.aligned_face import CenteringType

logger = logging.getLogger(__name__)
_BATCH_IDX: int = 0


def _get_new_batch_id() -> int:
    """ Obtain the next available batch index

    Returns
    -------
    int
        The next available unique batch id
    """
    global _BATCH_IDX  # pylint:disable=global-statement
    _BATCH_IDX += 1
    return _BATCH_IDX


@dataclass
class AlignerBatch(ExtractorBatch):
    """ Dataclass for holding items flowing through the aligner.

    Inherits from :class:`~plugins.extract._base.ExtractorBatch`

    Parameters
    ----------
    batch_id: int
        A unique integer for tracking this batch
    landmarks: list
        List of 68 point :class:`numpy.ndarray` landmark points returned from the aligner
    refeeds: list
        List of :class:`numpy.ndarrays` for holding each of the feeds that will be put through the
        model for each refeed
    second_pass: bool, optional
        ``True`` if this batch is passing through the aligner for a second time as re-align has
        been selected otherwise ``False``. Default: ``False``
    second_pass_masks: :class:`numpy.ndarray`, optional
        The masks used to filter out re-feed values for passing to the re-aligner.
    """
    batch_id: int = 0
    detected_faces: list[DetectedFace] = field(default_factory=list)
    landmarks: np.ndarray = np.array([])
    refeeds: list[np.ndarray] = field(default_factory=list)
    second_pass: bool = False
    second_pass_masks: np.ndarray = np.array([])

    def __repr__(self):
        """ Prettier repr for debug printing """
        retval = super().__repr__()
        retval += (f", batch_id={self.batch_id}, "
                   f"landmarks=[({self.landmarks.shape}, {self.landmarks.dtype})], "
                   f"refeeds={[(f.shape, f.dtype) for f in self.refeeds]}, "
                   f"second_pass={self.second_pass}, "
                   f"second_pass_masks={self.second_pass_masks})")
        return retval

    def __post_init__(self):
        """ Make sure that we have been given a non-zero ID """
        assert self.batch_id != 0, ("A batch ID must be specified for Aligner Batches")


class Aligner(Extractor):  # pylint:disable=abstract-method
    """ Aligner plugin _base Object

    All Aligner plugins must inherit from this class

    Parameters
    ----------
    git_model_id: int
        The second digit in the github tag that identifies this model. See
        https://github.com/deepfakes-models/faceswap-models for more information
    model_filename: str
        The name of the model file to be loaded
    normalize_method: {`None`, 'clahe', 'hist', 'mean'}, optional
        Normalize the images fed to the aligner. Default: ``None``
    re_feed: int, optional
        The number of times to re-feed a slightly adjusted bounding box into the aligner.
        Default: `0`
    re_align: bool, optional
        ``True`` to obtain landmarks by passing the initially aligned face back through the
        aligner. Default ``False``
    disable_filter: bool, optional
        Disable all aligner filters regardless of config option. Default: ``False``
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
    plugins.extract.mask._base : Masker parent class for extraction plugins.
    """

    def __init__(self,
                 git_model_id: int | None = None,
                 model_filename: str | None = None,
                 configfile: str | None = None,
                 instance: int = 0,
                 normalize_method: T.Literal["none", "clahe", "hist", "mean"] | None = None,
                 re_feed: int = 0,
                 re_align: bool = False,
                 disable_filter: bool = False,
                 **kwargs) -> None:
        logger.debug("Initializing %s: (normalize_method: %s, re_feed: %s, re_align: %s, "
                     "disable_filter: %s)", self.__class__.__name__, normalize_method, re_feed,
                     re_align, disable_filter)
        super().__init__(git_model_id,
                         model_filename,
                         configfile=configfile,
                         instance=instance,
                         **kwargs)
        self._plugin_type = "align"
        self.realign_centering: CenteringType = "face"  # overide for plugin specific centering

        # Override for specific landmark type:
        self.landmark_type = LandmarkType.LM_2D_68

        self._eof_seen = False
        self._normalize_method: T.Literal["clahe", "hist", "mean"] | None = None
        self._re_feed = re_feed
        self._filter = AlignedFilter(feature_filter=self.config["aligner_features"],
                                     min_scale=self.config["aligner_min_scale"],
                                     max_scale=self.config["aligner_max_scale"],
                                     distance=self.config["aligner_distance"],
                                     roll=self.config["aligner_roll"],
                                     save_output=self.config["save_filtered"],
                                     disable=disable_filter)
        self._re_align = ReAlign(re_align,
                                 self.config["realign_refeeds"],
                                 self.config["filter_realign"])
        self._needs_refeed_masks: bool = self._re_feed > 0 and (
            self.config["filter_refeed"] or (self._re_align.do_refeeds and
                                             self._re_align.do_filter))
        self.set_normalize_method(normalize_method)

        logger.debug("Initialized %s", self.__class__.__name__)

    def set_normalize_method(self, method: T.Literal["none", "clahe", "hist", "mean"] | None
                             ) -> None:
        """ Set the normalization method for feeding faces into the aligner.

        Parameters
        ----------
        method: {"none", "clahe", "hist", "mean"}
            The normalization method to apply to faces prior to feeding into the model
        """
        method = None if method is None or method.lower() == "none" else method
        self._normalize_method = T.cast(T.Literal["clahe", "hist", "mean"] | None, method)

    def initialize(self, *args, **kwargs) -> None:
        """ Add a call to add model input size to the re-aligner """
        self._re_align.set_input_size_and_centering(self.input_size, self.realign_centering)
        super().initialize(*args, **kwargs)

    def _handle_realigns(self, queue: Queue) -> tuple[bool, AlignerBatch] | None:
        """ Handle any items waiting for a second pass through the aligner.

        If EOF has been recieved and items are still being processed through the first pass
        then wait for a short time and try again to collect them.

        On EOF return exhausted flag with an empty batch

         Parameters
        ----------
        queue : queue.Queue()
            The ``queue`` that the plugin will be fed from.

        Returns
        -------
        ``None`` or tuple
            If items are processed then returns (`bool`, :class:`AlignerBatch`) containing the
            exhausted flag and the batch to be processed. If no items are processed returns
            ``None``
        """
        if not self._re_align.active:
            return None

        exhausted = False
        if self._re_align.items_queued:
            batch = self._re_align.get_batch()
            logger.trace("Re-align batch: %s", batch)  # type: ignore[attr-defined]
            return exhausted, batch

        if self._eof_seen and self._re_align.items_tracked:
            # EOF seen and items still being processed on first pass
            logger.debug("Tracked re-align items waiting to be flushed, retrying...")
            sleep(0.25)
            return self.get_batch(queue)

        if self._eof_seen:
            exhausted = True
            logger.debug("All items processed. Returning empty batch")
            self._filter.output_counts()
            self._eof_seen = False  # Reset for plugin re-use
            return exhausted, AlignerBatch(batch_id=-1)

        return None

    def get_batch(self, queue: Queue) -> tuple[bool, AlignerBatch]:
        """ Get items for inputting into the aligner from the queue in batches

        Items are returned from the ``queue`` in batches of
        :attr:`~plugins.extract._base.Extractor.batchsize`

        Items are received as :class:`~plugins.extract.extract_media.ExtractMedia` objects and
        converted to ``dict`` for internal processing.

        To ensure consistent batch sizes for aligner the items are split into separate items for
        each :class:`~lib.align.DetectedFace` object.

        Remember to put ``'EOF'`` to the out queue after processing
        the final batch

        Outputs items in the following format. All lists are of length
        :attr:`~plugins.extract._base.Extractor.batchsize`:

        >>> {'filename': [<filenames of source frames>],
        >>>  'image': [<source images>],
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

        realign_batch = self._handle_realigns(queue)
        if realign_batch is not None:
            return realign_batch

        batch = AlignerBatch(batch_id=_get_new_batch_id())
        idx = 0
        while idx < self.batchsize:
            item = self.rollover_collector(queue)
            if item == "EOF":
                logger.debug("EOF received")
                self._eof_seen = True
                exhausted = not self._re_align.active
                break

            # Put frames with no faces or are already aligned into the out queue
            if not item.detected_faces or item.is_aligned:
                self._queues["out"].put(item)
                continue

            converted_image = item.get_image_copy(self.color_format)
            for f_idx, face in enumerate(item.detected_faces):
                batch.image.append(converted_image)
                batch.detected_faces.append(face)
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
                        logger.trace("Rolled over %s faces of %s to "  # type: ignore[attr-defined]
                                     "next batch for '%s'", len(self._rollover.detected_faces),
                                     frame_faces, item.filename)
                    break
        if batch.filename:
            logger.trace("Returning batch: %s", batch)  # type: ignore[attr-defined]
            self._re_align.track_batch(batch.batch_id)
        else:
            logger.debug(item)

        return exhausted, batch

    def faces_to_feed(self, faces: np.ndarray) -> np.ndarray:
        """ Overide for specific plugin processing to convert a batch of face images from UINT8
        (0-255) into the correct format for the plugin's inference

        Parameters
        ----------
        faces: :class:`numpy.ndarray`
            The batch of faces in UINT8 format

        Returns
        -------
        class: `numpy.ndarray`
            The batch of faces in the format to feed through the plugin
        """
        raise NotImplementedError()

    # <<< FINALIZE METHODS >>> #
    def finalize(self, batch: BatchType) -> Generator[ExtractMedia, None, None]:
        """ Finalize the output from Aligner

        This should be called as the final task of each `plugin`.

        Pairs the detected faces back up with their original frame before yielding each frame.

        Parameters
        ----------
        batch : :class:`AlignerBatch`
            The final batch item from the `plugin` process.

        Yields
        ------
        :class:`~plugins.extract.extract_media.ExtractMedia`
            The :attr:`DetectedFaces` list will be populated for this class with the bounding boxes
            and landmarks for the detected faces found in the frame.
        """
        assert isinstance(batch, AlignerBatch)
        if not batch.second_pass and self._re_align.active:
            # Add the batch for second pass re-alignment and return
            self._re_align.add_batch(batch)
            return
        for face, landmarks in zip(batch.detected_faces, batch.landmarks):
            if not isinstance(landmarks, np.ndarray):
                landmarks = np.array(landmarks)
            face.add_landmarks_xy(landmarks)

        logger.trace("Item out: %s", batch)  # type: ignore[attr-defined]

        for frame, filename, face in zip(batch.image, batch.filename, batch.detected_faces):
            self._output_faces.append(face)
            if len(self._output_faces) != self._faces_per_filename[filename]:
                continue

            self._output_faces, folders = self._filter(self._output_faces, min(frame.shape[:2]))

            output = self._extract_media.pop(filename)
            output.add_detected_faces(self._output_faces)
            output.add_sub_folders(folders)
            self._output_faces = []

            logger.trace("Final Output: (filename: '%s', image "  # type: ignore[attr-defined]
                         "shape: %s, detected_faces: %s, item: %s)", output.filename,
                         output.image_shape, output.detected_faces, output)
            yield output
        self._re_align.untrack_batch(batch.batch_id)

    def on_completion(self) -> None:
        """ Output the filter counts when process has completed """
        self._filter.output_counts()

    # <<< PROTECTED METHODS >>> #
    # << PROCESS_INPUT WRAPPER >>
    def _get_adjusted_boxes(self, original_boxes: np.ndarray) -> np.ndarray:
        """ Obtain an array of adjusted bounding boxes based on the number of re-feed iterations
        that have been selected and the minimum dimension of the original bounding box.

        Parameters
        ----------
        original_boxes: :class:`numpy.ndarray`
            The original ('x', 'y', 'w', 'h') detected face boxes corresponding to the incoming
            detected face objects

        Returns
        -------
        :class:`numpy.ndarray`
            The original boxes (in position 0) and the randomly adjusted bounding boxes
        """
        if self._re_feed == 0:
            return original_boxes[None, ...]
        beta = 0.05
        max_shift = np.min(original_boxes[..., 2:], axis=1) * beta
        rands = np.random.rand(self._re_feed, *original_boxes.shape) * 2 - 1
        new_boxes = np.rint(original_boxes + (rands * max_shift[None, :, None])).astype("int32")
        retval = np.concatenate((original_boxes[None, ...], new_boxes))
        logger.trace(retval)  # type: ignore[attr-defined]
        return retval

    def _process_input_first_pass(self, batch: AlignerBatch) -> None:
        """ Standard pre-processing for aligners for first pass (if re-align selected) or the
        only pass.

        Process the input to the aligner model multiple times based on the user selected
        `re-feed` command line option. This adjusts the bounding box for the face to be fed
        into the model by a random amount within 0.05 pixels of the detected face's shortest axis.

        References
        ----------
        https://studios.disneyresearch.com/2020/06/29/high-resolution-neural-face-swapping-for-visual-effects/

        Parameters
        ----------
        batch: :class:`AlignerBatch`
            Contains the batch that is currently being passed through the plugin process
        """
        original_boxes = np.array([(face.left, face.top, face.width, face.height)
                                   for face in batch.detected_faces])
        adjusted_boxes = self._get_adjusted_boxes(original_boxes)

        # Put in random re-feed data to the bounding boxes
        for bounding_boxes in adjusted_boxes:
            for face, box in zip(batch.detected_faces, bounding_boxes):
                face.left, face.top, face.width, face.height = box

            self.process_input(batch)
            batch.feed = self.faces_to_feed(self._normalize_faces(batch.feed))
            # Move the populated feed into the batch refeed list. It will be overwritten at next
            # iteration
            batch.refeeds.append(batch.feed)

        # Place the original bounding box back to detected face objects
        for face, box in zip(batch.detected_faces, original_boxes):
            face.left, face.top, face.width, face.height = box

    def _get_realign_masks(self, batch: AlignerBatch) -> np.ndarray:
        """ Obtain the masks required for processing re-aligns

        Parameters
        ----------
        batch: :class:`AlignerBatch`
            Contains the batch that is currently being passed through the plugin process

        Returns
        -------
        :class:`numpy.ndarray`
            The filter masks required for masking the re-aligns
        """
        if self._re_align.do_refeeds:
            retval = batch.second_pass_masks  # Masks already calculated during re-feed
        elif self._re_align.do_filter:
            retval = self._filter.filtered_mask(batch)[None, ...]
        else:
            retval = np.zeros((batch.landmarks.shape[0], ), dtype="bool")[None, ...]
        return retval

    def _process_input_second_pass(self, batch: AlignerBatch) -> None:
        """ Process the input for 2nd-pass re-alignment

        Parameters
        ----------
        batch: :class:`AlignerBatch`
            Contains the batch that is currently being passed through the plugin process
        """
        batch.second_pass_masks = self._get_realign_masks(batch)

        if not self._re_align.do_refeeds:
            # Expand the dimensions for re-aligns for consistent handling of code
            batch.landmarks = batch.landmarks[None, ...]

        refeeds = self._re_align.process_batch(batch)
        batch.refeeds = [self.faces_to_feed(self._normalize_faces(faces)) for faces in refeeds]

    def _process_input(self, batch: BatchType) -> AlignerBatch:
        """ Perform pre-processing depending on whether this is the first/only pass through the
        aligner or the 2nd pass when re-align has been selected

        Parameters
        ----------
        batch: :class:`AlignerBatch`
            Contains the batch that is currently being passed through the plugin process

        Returns
        -------
        :class:`AlignerBatch`
            The batch with input processed
        """
        assert isinstance(batch, AlignerBatch)
        if batch.second_pass:
            self._process_input_second_pass(batch)
        else:
            self._process_input_first_pass(batch)
        return batch

    # <<< PREDICT WRAPPER >>> #
    def _predict(self, batch: BatchType) -> AlignerBatch:
        """ Just return the aligner's predict function

        Parameters
        ----------
        batch: :class:`AlignerBatch`
            The current batch to find alignments for

        Returns
        -------
        :class:`AlignerBatch`
            The batch item with the :attr:`prediction` populated

        Raises
        ------
        FaceswapError
            If GPU resources are exhausted
        """
        assert isinstance(batch, AlignerBatch)
        try:
            preds = [self.predict(feed) for feed in batch.refeeds]
            try:
                batch.prediction = np.array(preds)
            except ValueError as err:
                # If refeed batches are different sizes, Numpy will error, so we need to explicitly
                # set the dtype to 'object' rather than let it infer
                # numpy error:
                # ValueError: setting an array element with a sequence. The requested array has an
                # inhomogeneous shape after 1 dimensions. The detected shape was (9,) +
                # inhomogeneous part
                if "inhomogeneous" in str(err):
                    logger.trace(  # type:ignore[attr-defined]
                        "Mismatched array sizes, setting dtype to object: %s",
                        [p.shape for p in preds])
                    batch.prediction = np.array(preds, dtype="object")
                else:
                    raise

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

    def _process_refeeds(self, batch: AlignerBatch) -> list[AlignerBatch]:
        """ Process the output for each selected re-feed

        Parameters
        ----------
        batch: :class:`AlignerBatch`
            The batch object passing through the aligner

        Returns
        -------
        list
            List of :class:`AlignerBatch` objects. Each object in the list contains the
            results for each selected re-feed
        """
        retval: list[AlignerBatch] = []
        if batch.second_pass:
            # Re-insert empty sub-patches for re-population in ReAlign for filtered out batches
            selected_idx = 0
            for mask in batch.second_pass_masks:
                all_filtered = np.all(mask)
                if not all_filtered:
                    feed = batch.refeeds[selected_idx]
                    pred = batch.prediction[selected_idx]
                    data = batch.data[selected_idx] if batch.data else {}
                    selected_idx += 1
                else:  # All resuts have been filtered out
                    feed = pred = np.array([])
                    data = {}

                subbatch = AlignerBatch(batch_id=batch.batch_id,
                                        image=batch.image,
                                        detected_faces=batch.detected_faces,
                                        filename=batch.filename,
                                        feed=feed,
                                        prediction=pred,
                                        data=[data],
                                        second_pass=batch.second_pass)

                if not all_filtered:
                    self.process_output(subbatch)

                retval.append(subbatch)
        else:
            b_data = batch.data if batch.data else [{}]
            for feed, pred, dat in zip(batch.refeeds, batch.prediction, b_data):
                subbatch = AlignerBatch(batch_id=batch.batch_id,
                                        image=batch.image,
                                        detected_faces=batch.detected_faces,
                                        filename=batch.filename,
                                        feed=feed,
                                        prediction=pred,
                                        data=[dat],
                                        second_pass=batch.second_pass)
                self.process_output(subbatch)
                retval.append(subbatch)
        return retval

    def _get_refeed_filter_masks(self,
                                 subbatches: list[AlignerBatch],
                                 original_masks: np.ndarray | None = None) -> np.ndarray:
        """ Obtain the boolean mask array for masking out failed re-feed results if filter refeed
        has been selected

        Parameters
        ----------
        subbatches: list
            List of sub-batch results for each re-feed performed
        original_masks: :class:`numpy.ndarray`, Optional
            If passing in the second pass landmarks, these should be the original filter masks so
            that we don't calculate the mask again for already filtered faces. Default: ``None``

        Returns
        -------
        :class:`numpy.ndarray`
            boolean values for every detected face indicating whether the interim landmarks have
            passed the filter test
        """
        retval = np.zeros((len(subbatches), subbatches[0].landmarks.shape[0]), dtype="bool")

        if not self._needs_refeed_masks:
            return retval

        retval = retval if original_masks is None else original_masks
        for subbatch, masks in zip(subbatches, retval):
            masks[:] = self._filter.filtered_mask(subbatch, np.flatnonzero(masks))
        return retval

    def _get_mean_landmarks(self, landmarks: np.ndarray, masks: np.ndarray) -> np.ndarray:
        """ Obtain the averaged landmarks from the re-fed alignments. If config option
        'filter_refeed' is enabled, then average those results which have not been filtered out
        otherwise average all results

        Parameters
        ----------
        landmarks: :class:`numpy.ndarray`
            The batch of re-fed alignments
        masks: :class:`numpy.ndarray`
            List of boolean values indicating whether each re-fed alignments passed or failed
            the filter test

        Returns
        -------
        :class:`numpy.ndarray`
            The final averaged landmarks
        """
        if any(np.all(masked) for masked in masks.T):
            # hacky fix for faces which entirely failed the filter
            # We just unmask one value as it is junk anyway and will be discarded on output
            for idx, masked in enumerate(masks.T):
                if np.all(masked):
                    masks[0, idx] = False

        masks = np.broadcast_to(np.reshape(masks, (*landmarks.shape[:2], 1, 1)),
                                landmarks.shape)
        return np.ma.array(landmarks, mask=masks).mean(axis=0).data.astype("float32")

    def _process_output_first_pass(self, subbatches: list[AlignerBatch]) -> tuple[np.ndarray,
                                                                                  np.ndarray]:
        """ Process the output from the aligner if this is the first or only pass.

        Parameters
        ----------
        subbatches: list
            List of sub-batch results for each re-feed performed

        Returns
        -------
        landmarks: :class:`numpy.ndarray`
            If re-align is not selected or if re-align has been selected but only on the final
            output (ie: realign_reefeeds is ``False``) then the averaged batch of landmarks for all
            re-feeds is returned.
            If re-align_refeeds has been selected, then this will output each batch of re-feed
            landmarks.
        masks: :class:`numpy.ndarray`
            Boolean mask corresponding to the re-fed landmarks output indicating any values which
            should be filtered out prior to further processing
        """
        masks = self._get_refeed_filter_masks(subbatches)
        all_landmarks = np.array([sub.landmarks for sub in subbatches])

        # re-align not selected or not filtering the re-feeds
        if not self._re_align.do_refeeds:
            retval = self._get_mean_landmarks(all_landmarks, masks)
            return retval, masks

        # Re-align selected with filter re-feeds
        return all_landmarks, masks

    def _process_output_second_pass(self,
                                    subbatches: list[AlignerBatch],
                                    masks: np.ndarray) -> np.ndarray:
        """ Process the output from the aligner if this is the first or only pass.

        Parameters
        ----------
        subbatches: list
            List of sub-batch results for each re-aligned re-feed performed
        masks: :class:`numpy.ndarray`
            The original re-feed filter masks from the first pass
        """
        self._re_align.process_output(subbatches, masks)
        masks = self._get_refeed_filter_masks(subbatches, original_masks=masks)
        all_landmarks = np.array([sub.landmarks for sub in subbatches])
        return self._get_mean_landmarks(all_landmarks, masks)

    def _process_output(self, batch: BatchType) -> AlignerBatch:
        """ Process the output from the aligner model multiple times based on the user selected
        `re-feed amount` configuration option, then average the results for final prediction.

        If the config option 'filter_refeed' is enabled, then mask out any returned alignments
        that fail a filter test

        Parameters
        ----------
        batch : :class:`AlignerBatch`
            Contains the batch that is currently being passed through the plugin process

        Returns
        -------
        :class:`AlignerBatch`
            The batch item with :attr:`landmarks` populated
        """
        assert isinstance(batch, AlignerBatch)
        subbatches = self._process_refeeds(batch)
        if batch.second_pass:
            batch.landmarks = self._process_output_second_pass(subbatches, batch.second_pass_masks)
        else:
            landmarks, masks = self._process_output_first_pass(subbatches)
            batch.landmarks = landmarks
            batch.second_pass_masks = masks
        return batch

    # <<< FACE NORMALIZATION METHODS >>> #
    def _normalize_faces(self, faces: np.ndarray) -> np.ndarray:
        """ Normalizes the face for feeding into model
        The normalization method is dictated by the normalization command line argument

        Parameters
        ----------
        faces: :class:`numpy.ndarray`
            The batch of faces to normalize

        Returns
        -------
        :class:`numpy.ndarray`
            The normalized faces
        """
        if self._normalize_method is None:
            return faces
        logger.trace("Normalizing faces")  # type: ignore[attr-defined]
        meth = getattr(self, f"_normalize_{self._normalize_method.lower()}")
        faces = np.array([meth(face) for face in faces])
        logger.trace("Normalized faces")  # type: ignore[attr-defined]
        return faces

    @classmethod
    def _normalize_mean(cls, face: np.ndarray) -> np.ndarray:
        """ Normalize Face to the Mean

        Parameters
        ----------
        face: :class:`numpy.ndarray`
            The face to normalize

        Returns
        -------
        :class:`numpy.ndarray`
            The normalized face
        """
        face = face / 255.0
        for chan in range(3):
            layer = face[:, :, chan]
            layer = (layer - layer.min()) / (layer.max() - layer.min())
            face[:, :, chan] = layer
        return face * 255.0

    @classmethod
    def _normalize_hist(cls, face: np.ndarray) -> np.ndarray:
        """ Equalize the RGB histogram channels

        Parameters
        ----------
        face: :class:`numpy.ndarray`
            The face to normalize

        Returns
        -------
        :class:`numpy.ndarray`
            The normalized face
        """
        for chan in range(3):
            face[:, :, chan] = cv2.equalizeHist(face[:, :, chan])
        return face

    @classmethod
    def _normalize_clahe(cls, face: np.ndarray) -> np.ndarray:
        """ Perform Contrast Limited Adaptive Histogram Equalization

        Parameters
        ----------
        face: :class:`numpy.ndarray`
            The face to normalize

        Returns
        -------
        :class:`numpy.ndarray`
            The normalized face
        """
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        for chan in range(3):
            face[:, :, chan] = clahe.apply(face[:, :, chan])
        return face
