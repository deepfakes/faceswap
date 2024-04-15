#!/usr/bin/env python3
""" Import face detection ROI boxes from a json file """
from __future__ import annotations

import logging
import os
import re
import typing as T

import numpy as np

from lib.align import AlignedFace
from lib.utils import FaceswapError, IMAGE_EXTENSIONS

from ._base import Detector

if T.TYPE_CHECKING:
    from lib.align import DetectedFace
    from plugins.extract import ExtractMedia
    from ._base import BatchType

logger = logging.getLogger(__name__)


class Detect(Detector):
    """ Import face detection bounding boxes from an external json file """
    def __init__(self, **kwargs) -> None:
        kwargs["rotation"] = None  # Disable rotation
        kwargs["min_size"] = 0  # Disable min_size
        super().__init__(git_model_id=None, model_filename=None, **kwargs)

        self.name = "External"
        self.batchsize = 16

        self._origin: T.Literal["top-left",
                                "bottom-left",
                                "top-right",
                                "bottom-right"] = self.config["origin"]

        self._re_frame_no: re.Pattern = re.compile(r"\d+$")
        self._missing: list[str] = []
        self._log_once = True
        self._is_video = False
        self._imported: dict[str | int, np.ndarray] = {}
        """dict[str | int, np.ndarray]: The imported data from external .json file"""

    def init_model(self) -> None:
        """ No initialization to perform """
        logger.debug("No detector model to initialize")

    def _compile_detection_image(self, item: ExtractMedia
                                 ) -> tuple[np.ndarray, float, tuple[int, int]]:
        """ Override _compile_detection_image method, to obtain the source frame dimensions

        Parameters
        ----------
        item: :class:`~plugins.extract.extract_media.ExtractMedia`
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
        return np.array(item.image_shape[:2], dtype="int64"), 1.0, (0, 0)

    def _check_for_video(self, filename: str) -> None:
        """ Check a sample filename from the import file for a file extension to set
        :attr:`_is_video`

        Parameters
        ----------
        filename: str
            A sample file name from the imported data
        """
        logger.debug("Checking for video from '%s'", filename)
        ext = os.path.splitext(filename)[-1]
        if ext.lower() not in IMAGE_EXTENSIONS:
            self._is_video = True
        logger.debug("Set is_video to %s from extension '%s'", self._is_video, ext)

    def _get_key(self, key: str) -> str | int:
        """ Obtain the key for the item in the lookup table. If the input are images, the key will
        be the image filename. If the input is a video, the key will be the frame number

        Parameters
        ----------
        key: str
            The initial key value from import data or an import image/frame

        Returns
        -------
        str | int
            The filename is the input data is images, otherwise the frame number of a video
        """
        if not self._is_video:
            return key
        original_name = os.path.splitext(key)[0]
        matches = self._re_frame_no.findall(original_name)
        if not matches or len(matches) > 1:
            raise FaceswapError(f"Invalid import name: '{key}'. For video files, the key should "
                                "end with the frame number.")
        retval = int(matches[0])
        logger.trace("Obtained frame number %s from key '%s'",  # type:ignore[attr-defined]
                     retval, key)
        return retval

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
        n_landmarks = self._validate_landmarks(landmarks)
        face = AlignedFace(n_landmarks, centering="legacy", coverage_ratio=0.75)
        return np.concatenate([np.min(face.original_roi, axis=0),
                               np.max(face.original_roi, axis=0)])

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
        self._check_for_video(list(data)[0])
        for key, faces in data.items():
            try:
                store_key = self._get_key(key)
                self._imported[store_key] = np.array([self._import_frame_face(face, align_origin)
                                                      for face in faces], dtype="int32")
            except FaceswapError as err:
                logger.error(str(err))
                msg = f"The imported frame key that failed was '{key}'"
                raise FaceswapError(msg) from err

    def process_input(self, batch: BatchType) -> None:
        """ Put the lookup key into `batch.feed` so they can be collected for mapping in `.predict`

        Parameters
        ----------
        batch: :class:`~plugins.extract.detect._base.DetectorBatch`
            The batch to be processed by the plugin
        """
        batch.feed = np.array([(self._get_key(os.path.basename(f)), i)
                               for f, i in zip(batch.filename, batch.image)], dtype="object")

    def _adjust_for_origin(self, box: np.ndarray, frame_dims: tuple[int, int]) -> np.ndarray:
        """ Adjust the bounding box to be top-left orientated based on the selected import origin

        Parameters
        ----------
        box: :class:`np.ndarray`
            The imported bounding box at original (0, 0) origin
        frame_dims: tuple[int, int]
            The (rows, columns) dimensions of the original frame

        Returns
        -------
        :class:`numpy.ndarray`
            The adjusted bounding box for a top-left origin
        """
        if not np.any(box) or self._origin == "top-left":
            return box
        if self._origin.startswith("bottom"):
            box[:, [1, 3]] = frame_dims[0] - box[:, [1, 3]]
        if self._origin.endswith("right"):
            box[:, [0, 2]] = frame_dims[1] - box[:, [0, 2]]

        return box

    def predict(self, feed: np.ndarray) -> list[np.ndarray]:  # type:ignore[override]
        """ Pair the input filenames to the import file

        Parameters
        ----------
        feed: :class:`numpy.ndarray`
            The filenames with original frame dimensions to obtain the imported bounding boxes for

        Returns
        -------
        list[]:class:`numpy.ndarray`]
            The bounding boxes for the given filenames
        """
        self._missing.extend(f[0] for f in feed if f[0] not in self._imported)
        return [self._adjust_for_origin(self._imported.pop(f[0], np.array([], dtype="int32")),
                                        f[1])
                for f in feed]

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

    def on_completion(self) -> None:
        """ Output information if:
        - Imported items were not matched in input data
        - Input data was not matched in imported items
        """
        super().on_completion()

        if self._missing:
            logger.warning("[DETECT] %s input frames could not be matched in the import file "
                           "'%s'. Run in verbose mode for a list of frames.",
                           len(self._missing), self.config["file_name"])
            logger.verbose(  # type:ignore[attr-defined]
                "[DETECT] Input frames not in import file: %s", self._missing)

        if self._imported:
            logger.warning("[DETECT] %s items in the import file '%s' could not be matched to any "
                           "input frames. Run in verbose mode for a list of items.",
                           len(self._imported), self.config["file_name"])
            logger.verbose(  # type:ignore[attr-defined]
                "[DETECT] import file items not in input frames: %s", list(self._imported))
