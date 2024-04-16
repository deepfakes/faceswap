#!/usr/bin/env python3
""" Import 68 point landmarks or ROI boxes from a json file """
import logging
import typing as T
import os
import re

import numpy as np

from lib.align import EXTRACT_RATIOS, LandmarkType
from lib.utils import FaceswapError, IMAGE_EXTENSIONS

from ._base import BatchType, Aligner, AlignerBatch

logger = logging.getLogger(__name__)


class Align(Aligner):
    """ Import face detection bounding boxes from an external json file """
    def __init__(self, **kwargs) -> None:
        kwargs["normalize_method"] = None  # Disable normalization
        kwargs["re_feed"] = 0  # Disable re-feed
        kwargs["re_align"] = False  # Disablle re-align
        kwargs["disable_filter"] = True  # Disable aligner filters
        super().__init__(git_model_id=None, model_filename=None, **kwargs)

        self.name = "External"
        self.batchsize = 16

        self._origin: T.Literal["top-left",
                                "bottom-left",
                                "top-right",
                                "bottom-right"] = self.config["origin"]

        self._re_frame_no: re.Pattern = re.compile(r"\d+$")
        self._is_video: bool = False
        self._imported: dict[str | int, tuple[int, np.ndarray]] = {}
        """dict[str | int, tuple[int, np.ndarray]]: filename as key, value of [number of faces
        remaining for the frame, all landmarks in the frame] """

        self._missing: list[str] = []
        self._roll: dict[T.Literal["bottom-left", "top-right", "bottom-right"], int] = {
            "bottom-left": 3, "top-right": 1, "bottom-right": 2}
        """dict[Literal["bottom-left", "top-right", "bottom-right"], int]: Amount to roll the
        points by for different origins when 4 Point ROI landmarks are provided """

        centering = self.config["4_point_centering"]
        self._adjustment: float = 1. if centering is None else 1. - EXTRACT_RATIOS[centering]
        """float: The amount to adjust 4 point ROI landmarks to standardize the points for a
        'head' sized extracted face """

    def init_model(self) -> None:
        """ No initialization to perform """
        logger.debug("No aligner model to initialize")

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

    def _import_face(self, face: dict[str, list[int] | list[list[float]]]) -> np.ndarray:
        """ Import the landmarks from a single face

        Parameters
        ----------
        face: dict[str, list[int] | list[list[float]]]
            An import dictionary item for a face

        Returns
        -------
        :class:`numpy.ndarray`
            The landmark data imported from the json file

        Raises
        ------
        FaceSwapError
            If the landmarks_2d key does not exist or the landmarks are in an incorrect format
        """
        landmarks = face.get("landmarks_2d")
        if landmarks is None:
            raise FaceswapError("The provided import file is the required key 'landmarks_2d")
        if len(landmarks) not in (4, 68):
            raise FaceswapError("Imported 'landmarks_2d' should be either 68 facial feature "
                                "landmarks or 4 ROI corner locations")
        retval = np.array(landmarks, dtype="float32")
        if retval.shape[-1] != 2:
            raise FaceswapError("Imported 'landmarks_2d' should be formatted as a list of (x, y) "
                                "co-ordinates")
        if retval.shape[0] == 4:  # Adjust ROI landmarks based on centering selected
            center = np.mean(retval, axis=0)
            retval = (retval - center) * self._adjustment + center

        return retval

    def import_data(self, data: dict[str, list[dict[str, list[int] | list[list[float]]]]]) -> None:
        """ Import the aligner data from the json import file and set to :attr:`_imported`

        Parameters
        ----------
        data: dict[str, list[dict[str, list[int] | list[list[float]]]]]
            The data to be imported
        """
        logger.debug("Data length: %s", len(data))
        self._check_for_video(list(data)[0])
        for key, faces in data.items():
            try:
                lms = np.array([self._import_face(face) for face in faces], dtype="float32")
                if not np.any(lms):
                    logger.trace("Skipping frame '%s' with no faces")  # type:ignore[attr-defined]
                    continue

                store_key = self._get_key(key)
                self._imported[store_key] = (lms.shape[0], lms)
            except FaceswapError as err:
                logger.error(str(err))
                msg = f"The imported frame key that failed was '{key}'"
                raise FaceswapError(msg) from err
        lm_shape = set(v[1].shape[1:] for v in self._imported.values() if v[0] > 0)
        if len(lm_shape) > 1:
            raise FaceswapError("All external data should have the same number of landmarks. "
                                f"Found landmarks of shape: {lm_shape}")
        if (4, 2) in lm_shape:
            self.landmark_type = LandmarkType.LM_2D_4

    def process_input(self, batch: BatchType) -> None:
        """ Put the filenames and original frame dimensions into `batch.feed` so they can be
        collected for mapping in `.predict`

        Parameters
        ----------
        batch: :class:`~plugins.extract.detect._base.AlignerBatch`
            The batch to be processed by the plugin
        """
        batch.feed = np.array([(self._get_key(os.path.basename(f)), i.shape[:2])
                               for f, i in zip(batch.filename, batch.image)], dtype="object")

    def faces_to_feed(self, faces: np.ndarray) -> np.ndarray:
        """ No action required for import plugin

        Parameters
        ----------
        faces: :class:`numpy.ndarray`
            The batch of faces in UINT8 format

        Returns
        -------
        class: `numpy.ndarray`
            the original batch of faces
        """
        return faces

    def _adjust_for_origin(self, landmarks: np.ndarray, frame_dims: tuple[int, int]) -> np.ndarray:
        """ Adjust the landmarks to be top-left orientated based on the selected import origin

        Parameters
        ----------
        landmarks: :class:`np.ndarray`
            The imported facial landmarks box at original (0, 0) origin
        frame_dims: tuple[int, int]
            The (rows, columns) dimensions of the original frame

        Returns
        -------
        :class:`numpy.ndarray`
            The adjusted landmarks box for a top-left origin
        """
        if not np.any(landmarks) or self._origin == "top-left":
            return landmarks

        if LandmarkType.from_shape(landmarks.shape) == LandmarkType.LM_2D_4:
            landmarks = np.roll(landmarks, self._roll[self._origin], axis=0)

        if self._origin.startswith("bottom"):
            landmarks[:, 1] = frame_dims[0] - landmarks[:, 1]
        if self._origin.endswith("right"):
            landmarks[:, 0] = frame_dims[1] - landmarks[:, 0]

        return landmarks

    def predict(self, feed: np.ndarray) -> np.ndarray:
        """ Pair the input filenames to the import file

        Parameters
        ----------
        feed: :class:`numpy.ndarray`
            The filenames in the batch to return imported alignments for

        Returns
        -------
        :class:`numpy.ndarray`
            The predictions for the given filenames
        """
        preds = []
        for key, frame_dims in feed:
            if key not in self._imported:
                self._missing.append(key)
                continue

            remaining, all_lms = self._imported[key]
            preds.append(self._adjust_for_origin(all_lms[all_lms.shape[0] - remaining],
                                                 frame_dims))

            if remaining == 1:
                del self._imported[key]
            else:
                self._imported[key] = (remaining - 1, all_lms)

        return np.array(preds, dtype="float32")

    def process_output(self, batch: BatchType) -> None:
        """ Process the imported data to the landmarks attribute

        Parameters
        ----------
        batch: :class:`AlignerBatch`
            The current batch from the model with :attr:`predictions` populated
        """
        assert isinstance(batch, AlignerBatch)
        batch.landmarks = batch.prediction
        logger.trace("Imported landmarks: %s", batch.landmarks)  # type:ignore[attr-defined]

    def on_completion(self) -> None:
        """ Output information if:
        - Imported items were not matched in input data
        - Input data was not matched in imported items
        """
        super().on_completion()

        if self._missing:
            logger.warning("[ALIGN] %s input frames could not be matched in the import file "
                           "'%s'. Run in verbose mode for a list of frames.",
                           len(self._missing), self.config["file_name"])
            logger.verbose(  # type:ignore[attr-defined]
                "[ALIGN] Input frames not in import file: %s", self._missing)

        if self._imported:
            logger.warning("[ALIGN] %s items in the import file '%s' could not be matched to any "
                           "input frames. Run in verbose mode for a list of items.",
                           len(self._imported), self.config["file_name"])
            logger.verbose(  # type:ignore[attr-defined]
                "[ALIGN] import file items not in input frames: %s", list(self._imported))
