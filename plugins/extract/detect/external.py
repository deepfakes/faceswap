#!/usr/bin/env python3
""" Import face detection ROI boxes from a json file """
from __future__ import annotations

import logging
import os
import typing as T

import numpy as np

from lib.align import AlignedFace
from lib.serializer import get_serializer
from lib.utils import FaceswapError

from ._base import Detector

if T.TYPE_CHECKING:
    from lib.align import DetectedFace
    from plugins.extract.pipeline import ExtractMedia
    from ._base import BatchType

logger = logging.getLogger(__name__)


class Detect(Detector):
    """ Import face detection bounding boxes from an external json file """
    def __init__(self, **kwargs) -> None:
        kwargs["rotation"] = None  # Disable rotation
        kwargs["min_size"] = 0  # Disable min_size
        super().__init__(git_model_id=None, model_filename=None, **kwargs)

        self.input_size = 256
        self.name = "External"
        self.vram = 0  # Doesn't use GPU
        self.vram_per_batch = 0
        self.batchsize = 16

        self._serializer = get_serializer("json")
        self._origin: T.Literal["top-left",
                                "bottom-left",
                                "top-right",
                                "bottom-right"] = self.config["origin"]
        self._missing = 0
        self._log_once = True
        self._imported: dict[str, np.ndarray] = {}
        """dict: The imported data from external .json file"""

    def init_model(self) -> None:
        """ No initialization to perform """
        logger.debug("No detector model to initialize")

    def _compile_detection_image(self, item: ExtractMedia
                                 ) -> tuple[np.ndarray, float, tuple[int, int]]:
        """ Override _compile_detection_image method, as no compilation needs performing

        Parameters
        ----------
        item: :class:`plugins.extract.pipeline.ExtractMedia`
            The input item from the pipeline

        Returns
        -------
        image: :class:`numpy.ndarray`
            dummy empty array
        scale: float
            The scaling factor for the image (1.0)
        pad: int
            The amount of padding applied to the image (0, 0)
        """
        return np.empty([0], dtype="uint8"), 1.0, (0, 0)

    @classmethod
    def _bbox_from_detected(cls, bounding_box: list[int]) -> np.ndarray:
        """ Import the detected face roi from a `detected` item in the import file

        Parameters
        ----------
        bounding_box: list[int]
            a bounding box contained within the import file

        Returns
        -------
        :class:`numpy.ndarray`
            The "left", "top", "right", "bottom" bounding box for the face

        Raises
        ------
        FaceSwapError
            If the number of bounding box co-ordinates is incorrect
        """
        if len(bounding_box) != 4:
            raise FaceswapError("Imported 'detected' bounding boxes should be a list of 4 numbers "
                                "representing the 'left', 'top', 'right', `bottom` of a face.")
        return np.rint(bounding_box)

    def _validate_landmarks(self, landmarks: list[list[float]]) -> np.ndarray:
        """ Validate that the there are 4 or 68 landmarks and are a complete list of (x, y)
        co-ordinates

        Parameters
        ----------
        landmarks: list[float]
            The 4 point ROI or 68 point 2D landmarks that are being imported

        Returns
        -------
        :class:`numpy.ndarray`
            The original landmarks as a numpy array

        Raises
        ------
        FaceSwapError
            If the landmarks being imported are not correct
        """
        if len(landmarks) not in (4, 68):
            raise FaceswapError("Imported 'landmarks_2d' should be either 68 facial feature "
                                "landmarks or 4 ROI corner locations")
        retval = np.array(landmarks, dtype="float32")
        if retval.shape[-1] != 2:
            raise FaceswapError("Imported 'landmarks_2d' should be formatted as a list of (x, y) "
                                "co-ordinates")
        return retval

    def _bbox_from_landmarks2d(self, landmarks: list[list[float]]) -> np.ndarray:
        """ Import the detected face roi by estimating from imported landmarks

        Parameters
        ----------
        landmarks: list[float]
            The 4 point ROI or 68 point 2D landmarks that are being imported

        Returns
        -------
        :class:`numpy.ndarray`
            The "left", "top", "right", "bottom" bounding box for the face

        """
        # TODO we need to know the co-ordinate origin of the landmarks here
        # TODO 4 point landmarks
        n_landmarks = self._validate_landmarks(landmarks)
        if n_landmarks.shape[0] == 68:  # fairly tight crop of legacy original ROI
            face = AlignedFace(n_landmarks, centering="legacy", coverage_ratio=0.75)
            return np.concatenate([np.min(face.original_roi, axis=0),
                                   np.max(face.original_roi, axis=0)])
        raise NotImplementedError

    def _import_frame_face(self,
                           face: dict[str, list[int] | list[list[float]]],
                           align_origin: T.Literal["top-left",
                                                   "bottom-left",
                                                   "top-right",
                                                   "bottom-right"] | None) -> np.ndarray:
        """ Import a detected face ROI from the import file

        Parameters
        ----------
        face: dict[str, list[int] | list[list[float]]]
            The data that exists within the import file for the frame
        align_origin: Literal["top-left", "bottom-left", "top-right", "bottom-right"] | None
            The origin of the imported aligner data. Used if the detected ROI is being estimated
            from imported aligner data

        Returns
        -------
        :class:`numpy.ndarray`
            The "left", "top", "right", "bottom" bounding box for the face

        Raises
        ------
        FaceSwapError
            If the required keys for the bounding boxes are not present for the face
        """
        # TODO test calculated
        if "detected" in face:
            return self._bbox_from_detected(T.cast(list[int], face["detected"]))
        if "landmarks_2d" in face:
            if self._log_once and align_origin is None:
                logger.warning("You are importing Detection data, but have only provided "
                               "Alignment data. This is most likely incorrect and will lead "
                               "to poor results")
                self._log_once = False

            if self._log_once and align_origin is not None and align_origin != self._origin:
                logger.info("Updating Detect origin from Aligner config to '%s'", align_origin)
                self._origin = align_origin
                self._log_once = False

            return self._bbox_from_landmarks2d(T.cast(list[list[float]], face["landmarks_2d"]))

        raise FaceswapError("The provided import file is missing both of the required keys "
                            "'detected' and 'landmarks_2d")

    def import_data(self,
                    data: dict[str, list[dict[str, list[int] | list[list[float]]]]],
                    align_origin: T.Literal["top-left",
                                            "bottom-left",
                                            "top-right",
                                            "bottom-right"] | None) -> None:
        """ Import the detection data from the json import file and set to :attr:`_imported`

        Parameters
        ----------
        data: dict[str, list[dict[str, list[int] | list[list[float]]]]]
            The data to be imported
        align_origin: Literal["top-left", "bottom-left", "top-right", "bottom-right"] | None
            The origin of the imported aligner data. Used if the detected ROI is being estimated
            from imported aligner data
        """
        logger.debug("Data length: %s, align_origin: %s", len(data), align_origin)
        for key, faces in data.items():
            try:
                self._imported[key] = np.array([self._import_frame_face(face, align_origin)
                                                for face in faces], dtype="int32")
            except FaceswapError as err:
                logger.error(str(err))
                msg = f"The imported frame key that failed was '{key}'"
                raise FaceswapError(msg) from err

    def process_input(self, batch: BatchType) -> None:
        """ Put the filenames into `batch.feed` so they can be collected for mapping in `.predict`

        Parameters
        ----------
        batch: :class:`~plugins.extract.detect._base.DetectorBatch`
            The batch to be processed by the plugin
        """
        batch.feed = np.array([os.path.basename(f) for f in batch.filename], dtype="object")

    def predict(self, feed: np.ndarray) -> list[np.ndarray]:  # type:ignore[override]
        """ Pair the input filenames to the import file

        Parameters
        ----------
        feed: :class:`numpy.ndarray`
            The filenames to obtain the imported bounding boxes for

        Returns
        -------
        list[]:class:`numpy.ndarray`]
            The bounding boxes for the given filenames
        """
        self._missing += len([f for f in feed if f not in self._imported])
        return [self._imported.pop(f, np.array([], dtype="int32")) for f in feed]

    def process_output(self, batch: BatchType) -> None:
        """ No output processing required for import plugin

        Parameters
        ----------
        batch: :class:`~plugins.extract.detect._base.DetectorBatch`
            The batch to be processed by the plugin
        """
        logger.trace("No output processing for import plugin")  # type:ignore[attr-defined]

    def _remove_zero_sized_faces(self, batch_faces: list[list[DetectedFace]]
                                 ) -> list[list[DetectedFace]]:
        """ Override _remove_zero_sized_faces to just return the faces that have been imported

        Parameters
        ----------
        batch_faces: list[list[DetectedFace]
            List of detected face objects

        Returns
        -------
        list[list[DetectedFace]
            Original list of detected face objects
        """
        return batch_faces
