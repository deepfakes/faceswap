#!/usr/bin/env python3
""" Processing methods for aligner plugins """
from __future__ import annotations
import logging
import typing as T

from threading import Lock

import numpy as np

from lib.align import AlignedFace

if T.TYPE_CHECKING:
    from lib.align import DetectedFace
    from .aligner import AlignerBatch
    from lib.align.aligned_face import CenteringType

logger = logging.getLogger(__name__)


class AlignedFilter():
    """ Applies filters on the output of the aligner

    Parameters
    ----------
    feature_filter: bool
        ``True`` to enable filter to check relative position of eyes/eyebrows and mouth. ``False``
        to disable.
    min_scale: float
        Filters out faces that have been aligned at below this value as a multiplier of the
        minimum frame dimension. Set to ``0`` for off.
    max_scale: float
        Filters out faces that have been aligned at above this value as a multiplier of the
        minimum frame dimension. Set to ``0`` for off.
    distance: float
        Filters out faces that are further than this distance from an "average" face. Set to
        ``0`` for off.
    roll: float
        Filters out faces with a roll value outside of 0 +/- the value given here. Set to ``0``
        for off.
    save_output: bool
        ``True`` if the filtered faces should be kept as they are being saved. ``False`` if they
        should be deleted
    disable: bool, Optional
        ``True`` to disable the filter regardless of config options. Default: ``False``
    """
    def __init__(self,
                 feature_filter: bool,
                 min_scale: float,
                 max_scale: float,
                 distance: float,
                 roll: float,
                 save_output: bool,
                 disable: bool = False) -> None:
        logger.debug("Initializing %s: (feature_filter: %s, min_scale: %s, max_scale: %s, "
                     "distance: %s, roll, %s, save_output: %s, disable: %s)",
                     self.__class__.__name__, feature_filter, min_scale, max_scale, distance, roll,
                     save_output, disable)
        self._features = feature_filter
        self._min_scale = min_scale
        self._max_scale = max_scale
        self._distance = distance / 100.
        self._roll = roll
        self._save_output = save_output
        self._active = not disable and (feature_filter or
                                        max_scale > 0.0 or
                                        min_scale > 0.0 or
                                        distance > 0.0 or
                                        roll > 0.0)
        self._counts: dict[str, int] = {"features": 0,
                                        "min_scale": 0,
                                        "max_scale": 0,
                                        "distance": 0,
                                        "roll": 0}
        logger.debug("Initialized %s: ", self.__class__.__name__)

    def _scale_test(self,
                    face: AlignedFace,
                    minimum_dimension: int) -> T.Literal["min", "max"] | None:
        """ Test if a face is below or above the min/max size thresholds. Returns as soon as a test
        fails.

        Parameters
        ----------
        face: :class:`~lib.aligned.AlignedFace`
            The aligned face to test the original size of.

        minimum_dimension: int
            The minimum (height, width) of the original frame

        Returns
        -------
        "min", "max" or ``None``
            Returns min or max if the face failed the minimum or maximum test respectively.
            ``None`` if all tests passed
        """

        if self._min_scale <= 0.0 and self._max_scale <= 0.0:
            return None

        roi = face.original_roi.astype("int64")
        size = ((roi[1][0] - roi[0][0]) ** 2 + (roi[1][1] - roi[0][1]) ** 2) ** 0.5

        if self._min_scale > 0.0 and size < minimum_dimension * self._min_scale:
            return "min"

        if self._max_scale > 0.0 and size > minimum_dimension * self._max_scale:
            return "max"

        return None

    def _handle_filtered(self,
                         key: str,
                         face: DetectedFace,
                         faces: list[DetectedFace],
                         sub_folders: list[str | None],
                         sub_folder_index: int) -> None:
        """ Add the filtered item to the filter counts.

        If config option `save_filtered` has been enabled then add the face to the output faces
        list and update the sub_folder list with the correct name for this face.

        Parameters
        ----------
        key: str
            The key to use for the filter counts dictionary and the sub_folder name
        face: :class:`~lib.align.detected_face.DetectedFace`
            The detected face object to be filtered out
        faces: list
            The list of faces that will be returned from the filter
        sub_folders: list
            List of sub folder names corresponding to the list of detected face objects
        sub_folder_index: int
            The index within the sub-folder list that the filtered face belongs to
        """
        self._counts[key] += 1
        if not self._save_output:
            return

        faces.append(face)
        sub_folders[sub_folder_index] = f"_align_filt_{key}"

    def __call__(self, faces: list[DetectedFace], minimum_dimension: int
                 ) -> tuple[list[DetectedFace], list[str | None]]:
        """ Apply the filter to the incoming batch

        Parameters
        ----------
        faces: list
            List of detected face objects to filter out on size
        minimum_dimension: int
            The minimum (height, width) of the original frame

        Returns
        -------
        detected_faces: list
            The filtered list of detected face objects, if saving filtered faces has not been
            selected or the full list of detected faces
        sub_folders: list
            List of ``Nones`` if saving filtered faces has not been selected or list of ``Nones``
            and sub folder names corresponding the filtered face location
        """
        sub_folders: list[str | None] = [None for _ in range(len(faces))]
        if not self._active:
            return faces, sub_folders

        retval: list[DetectedFace] = []
        for idx, face in enumerate(faces):
            aligned = AlignedFace(landmarks=face.landmarks_xy, centering="face")

            if self._features and aligned.relative_eye_mouth_position < 0.0:
                self._handle_filtered("features", face, retval, sub_folders, idx)
                continue

            min_max = self._scale_test(aligned, minimum_dimension)
            if min_max in ("min", "max"):
                self._handle_filtered(f"{min_max}_scale", face, retval, sub_folders, idx)
                continue

            if 0.0 < self._distance < aligned.average_distance:
                self._handle_filtered("distance", face, retval, sub_folders, idx)
                continue

            if self._roll != 0.0 and not 0.0 < abs(aligned.pose.roll) < self._roll:
                self._handle_filtered("roll", face, retval, sub_folders, idx)
                continue

            retval.append(face)
        return retval, sub_folders

    def filtered_mask(self,
                      batch: AlignerBatch,
                      skip: np.ndarray | list[int] | None = None) -> np.ndarray:
        """ Obtain a list of boolean values for the given batch indicating whether they pass the
        filter test.

        Parameters
        ----------
        batch: :class:`AlignerBatch`
            The batch of face to obtain masks for
        skip: list or :class:`numpy.ndarray`, optional
            List or 1D numpy array of indices indicating faces that have already been filter
            masked and so should not be filtered again. Values in these index positions will be
            returned as ``True``

        Returns
        -------
        :class:`numpy.ndarray`
            Boolean mask array corresponding to any of the input DetectedFace objects that passed a
            test. ``False`` the face passed the test. ``True`` it failed
        """
        skip = [] if skip is None else skip
        retval = np.ones((len(batch.detected_faces), ), dtype="bool")
        for idx, (landmarks, image) in enumerate(zip(batch.landmarks, batch.image)):
            if idx in skip:
                continue
            face = AlignedFace(landmarks)
            if self._features and face.relative_eye_mouth_position < 0.0:
                continue
            if self._scale_test(face, min(image.shape[:2])) is not None:
                continue
            if 0.0 < self._distance < face.average_distance:
                continue
            if self._roll != 0.0 and not 0.0 < abs(face.pose.roll) < self._roll:
                continue
            retval[idx] = False
        return retval

    def output_counts(self):
        """ Output the counts of filtered items """
        if not self._active:
            return
        counts = [f"{key} ({getattr(self, f'_{key}'):.2f}): {count}"
                  for key, count in self._counts.items()
                  if count > 0]
        if counts:
            logger.info("Aligner filtered: (%s)", ", ".join(counts))


class ReAlign():
    """ Holds data and methods for 2nd pass re-aligns

    Parameters
    ----------
    active: bool
        ``True`` if re-alignment has been requested otherwise ``False``
    do_refeeds: bool
        ``True`` if re-feeds should be re-aligned, ``False`` if just the final output of the
        re-feeds should be aligned
    do_filter: bool
        ``True`` if aligner filtered out faces should not be re-aligned. ``False`` if all faces
        should be re-aligned
    """
    def __init__(self, active: bool, do_refeeds: bool, do_filter: bool) -> None:
        logger.debug("Initializing %s: (active: %s, do_refeeds: %s, do_filter: %s)",
                     self.__class__.__name__, active, do_refeeds, do_filter)
        self._active = active
        self._do_refeeds = do_refeeds
        self._do_filter = do_filter
        self._centering: CenteringType = "face"
        self._size = 0
        self._tracked_lock = Lock()
        self._tracked_batchs: dict[int,
                                   dict[T.Literal["filtered_landmarks"], list[np.ndarray]]] = {}
        # TODO. Probably does not need to be a list, just alignerbatch
        self._queue_lock = Lock()
        self._queued: list[AlignerBatch] = []
        logger.debug("Initialized %s", self.__class__.__name__)

    @property
    def active(self) -> bool:
        """bool: ``True`` if re_aligns have been selected otherwise ``False``"""
        return self._active

    @property
    def do_refeeds(self) -> bool:
        """bool: ``True`` if re-aligning is active and re-aligning re-feeds has been selected
        otherwise ``False``"""
        return self._active and self._do_refeeds

    @property
    def do_filter(self) -> bool:
        """bool: ``True`` if re-aligning is active and faces which failed the aligner filter test
        should not be re-aligned otherwise ``False``"""
        return self._active and self._do_filter

    @property
    def items_queued(self) -> bool:
        """bool: ``True`` if re-align is active and items are queued for a 2nd pass otherwise
        ``False`` """
        with self._queue_lock:
            return self._active and bool(self._queued)

    @property
    def items_tracked(self) -> bool:
        """bool: ``True`` if items exist in the tracker so still need to be processed """
        with self._tracked_lock:
            return bool(self._tracked_batchs)

    def set_input_size_and_centering(self, input_size: int, centering: CenteringType) -> None:
        """ Set the input size of the loaded plugin once the model has been loaded

        Parameters
        ----------
        input_size: int
            The input size, in pixels, of the aligner plugin
        centering: ["face", "head" or "legacy"]
            The centering to align the image at for re-aligning
        """
        logger.debug("input_size: %s, centering: %s", input_size, centering)
        self._size = input_size
        self._centering = centering

    def track_batch(self, batch_id: int) -> None:
        """ Add newly seen batch id from the aligner to the batch tracker, so that we can keep
        track of whether there are still batches to be processed when the aligner hits 'EOF'

        Parameters
        ----------
        batch_id: int
            The batch id to add to batch tracking
        """
        if not self._active:
            return
        logger.trace("Tracking batch id: %s", batch_id)  # type: ignore[attr-defined]
        with self._tracked_lock:
            self._tracked_batchs[batch_id] = {}

    def untrack_batch(self, batch_id: int) -> None:
        """ Remove the tracked batch from the tracker once the batch has been fully processed

        Parameters
        ----------
        batch_id: int
            The batch id to remove from batch tracking
        """
        if not self._active:
            return
        logger.trace("Removing batch id from tracking: %s", batch_id)  # type: ignore[attr-defined]
        with self._tracked_lock:
            del self._tracked_batchs[batch_id]

    def add_batch(self, batch: AlignerBatch) -> None:
        """ Add first pass alignments to the queue for picking up for re-alignment, update their
        :attr:`second_pass` attribute to ``True`` and clear attributes not required.

        Parameters
        ----------
        batch: :class:`AlignerBatch`
            aligner batch to perform re-alignment on
        """
        with self._queue_lock:
            logger.trace("Queueing for second pass: %s", batch)  # type: ignore[attr-defined]
            batch.second_pass = True
            batch.feed = np.array([])
            batch.prediction = np.array([])
            batch.refeeds = []
            batch.data = []
            self._queued.append(batch)

    def get_batch(self) -> AlignerBatch:
        """ Retrieve the next batch currently queued for re-alignment

        Returns
        -------
        :class:`AlignerBatch`
            The next :class:`AlignerBatch` for re-alignment
        """
        with self._queue_lock:
            retval = self._queued.pop(0)
            logger.trace("Retrieving for second pass: %s",  # type: ignore[attr-defined]
                         retval.filename)
        return retval

    def process_batch(self, batch: AlignerBatch) -> list[np.ndarray]:
        """ Pre process a batch object for re-aligning through the aligner.

        Parameters
        ----------
        batch: :class:`AlignerBatch`
            aligner batch to perform pre-processing on

        Returns
        -------
        list
            List of UINT8 aligned faces batch for each selected refeed
        """
        logger.trace("Processing batch: %s, landmarks: %s",  # type: ignore[attr-defined]
                     batch.filename, [b.shape for b in batch.landmarks])
        retval: list[np.ndarray] = []
        filtered_landmarks: list[np.ndarray] = []
        for landmarks, masks in zip(batch.landmarks, batch.second_pass_masks):
            if not np.all(masks):  # At least one face has not already been filtered
                aligned_faces = [AlignedFace(lms,
                                             image=image,
                                             size=self._size,
                                             centering=self._centering)
                                 for image, lms, msk in zip(batch.image, landmarks, masks)
                                 if not msk]
                faces = np.array([aligned.face for aligned in aligned_faces
                                 if aligned.face is not None])
                retval.append(faces)
                batch.data.append({"aligned_faces": aligned_faces})

            if np.any(masks):
                # Track the original landmarks for re-insertion on the other side
                filtered_landmarks.append(landmarks[masks])

        with self._tracked_lock:
            self._tracked_batchs[batch.batch_id] = {"filtered_landmarks": filtered_landmarks}
        batch.landmarks = np.array([])  # Clear the old landmarks
        return retval

    def _transform_to_frame(self, batch: AlignerBatch) -> np.ndarray:
        """ Transform the predicted landmarks from the aligned face image back into frame
        co-ordinates

        Parameters
        ----------
        batch: :class:`AlignerBatch`
            An aligner batch containing the aligned faces in the data field and the face
            co-ordinate landmarks in the landmarks field

        Returns
        -------
        :class:`numpy.ndarray`
            The landmarks transformed to frame space
        """
        faces: list[AlignedFace] = batch.data[0]["aligned_faces"]
        retval = np.array([aligned.transform_points(landmarks, invert=True)
                           for landmarks, aligned in zip(batch.landmarks, faces)])
        logger.trace("Transformed points: original max: %s, "  # type: ignore[attr-defined]
                     "new max: %s", batch.landmarks.max(), retval.max())
        return retval

    def _re_insert_filtered(self, batch: AlignerBatch, masks: np.ndarray) -> np.ndarray:
        """ Re-insert landmarks that were filtered out from the re-align process back into the
        landmark results

        Parameters
        ----------
        batch: :class:`AlignerBatch`
            An aligner batch containing the aligned faces in the data field and the landmarks in
            frame space in the landmarks field
        masks: np.ndarray
            The original filter masks for this batch

        Returns
        -------
        :class:`numpy.ndarray`
            The full batch of landmarks with filtered out values re-inserted
        """
        if not np.any(masks):
            logger.trace("No landmarks to re-insert: %s", masks)  # type: ignore[attr-defined]
            return batch.landmarks

        with self._tracked_lock:
            filtered = self._tracked_batchs[batch.batch_id]["filtered_landmarks"].pop(0)

        if np.all(masks):
            retval = filtered
        else:
            retval = np.empty((masks.shape[0], *filtered.shape[1:]), dtype=filtered.dtype)
            retval[~masks] = batch.landmarks
            retval[masks] = filtered

        logger.trace("Filtered re-inserted: old shape: %s, "  # type: ignore[attr-defined]
                     "new shape: %s)", batch.landmarks.shape, retval.shape)

        return retval

    def process_output(self, subbatches: list[AlignerBatch], batch_masks: np.ndarray) -> None:
        """ Process the output from the re-align pass.

        - Transform landmarks from aligned face space to face space
        - Re-insert faces that were filtered out from the re-align process back into the
          landmarks list

        Parameters
        ----------
        subbatches: list
            List of sub-batch results for each re-aligned re-feed performed
        batch_masks: :class:`numpy.ndarray`
            The original re-feed filter masks from the first pass
        """
        for batch, masks in zip(subbatches, batch_masks):
            if not np.all(masks):
                batch.landmarks = self._transform_to_frame(batch)
            batch.landmarks = self._re_insert_filtered(batch, masks)
