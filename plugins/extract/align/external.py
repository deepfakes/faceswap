#!/usr/bin/env python3
""" Import 68 point landmarks or ROI boxes from a json file """
import logging
import os

import numpy as np

from lib.align import EXTRACT_RATIOS, LandmarkType
from lib.utils import FaceswapError

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

        self.input_size = 256
        self.name = "External"
        self.vram = 0  # Doesn't use GPU
        self.batchsize = 16

        self._imported: dict[str, tuple[int, np.ndarray]] = {}
        """dict[str, tuple[int, np.ndarray]]: filename as key, value of [number of faces remaining
        for the frame, all landmarks in the frame] """

        self._missing = 0

        centering = self.config["4_point_centering"]
        self._adjustment: float = 1. if centering is None else 1. - EXTRACT_RATIOS[centering]
        """float: The amount to adjust 4 point ROI landmarks to standardize the points for a
        'head' sized extracted face """

    def init_model(self) -> None:
        """ No initialization to perform """
        logger.debug("No aligner model to initialize")

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
        for key, faces in data.items():
            try:
                lms = np.array([self._import_face(face) for face in faces], dtype="float32")
                self._imported[key] = (lms.shape[0], lms)
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
        """ Put the filenames into `batch.feed` so they can be collected for mapping in `.predict`

        Parameters
        ----------
        batch: :class:`~plugins.extract.detect._base.AlignerBatch`
            The batch to be processed by the plugin
        """
        batch.feed = np.array([os.path.basename(f) for f in batch.filename], dtype="object")

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
        for key in feed:
            if key not in self._imported:
                # TODO Handle filename missing in imported data
                # As this is will almost definitely be problematic as num detected_faces != preds
                self._missing += 1
                continue

            remaining, all_lms = self._imported[key]
            preds.append(all_lms[all_lms.shape[0] - remaining])

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
