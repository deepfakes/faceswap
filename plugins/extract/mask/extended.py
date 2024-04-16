#!/usr/bin/env python3
""" Extended Mask for faceswap.py """
from __future__ import annotations
import logging
import typing as T

import cv2
import numpy as np

from lib.align import LandmarkType

from ._base import BatchType, Masker

logger = logging.getLogger(__name__)

if T.TYPE_CHECKING:
    from lib.align.aligned_face import AlignedFace


class Mask(Masker):
    """ Perform transformation to align and get landmarks """
    def __init__(self, **kwargs):
        git_model_id = None
        model_filename = None
        super().__init__(git_model_id=git_model_id, model_filename=model_filename, **kwargs)
        self.input_size = 256
        self.name = "Extended"
        self.vram = 0  # Doesn't use GPU
        self.vram_per_batch = 0
        self.batchsize = 1
        self.landmark_type = LandmarkType.LM_2D_68

    def init_model(self) -> None:
        logger.debug("No mask model to initialize")

    def process_input(self, batch: BatchType) -> None:
        """ Compile the detected faces for prediction """
        batch.feed = np.zeros((self.batchsize, self.input_size, self.input_size, 1),
                              dtype="float32")

    def predict(self, feed: np.ndarray) -> np.ndarray:
        """ Run model to get predictions """
        faces: list[AlignedFace] = feed[1]
        feed = feed[0]
        for mask, face in zip(feed, faces):
            if LandmarkType.from_shape(face.landmarks.shape) != self.landmark_type:
                # Called from the manual tool. # TODO This will only work with BS1
                feed = np.zeros_like(feed)
                continue
            parts = self.parse_parts(np.array(face.landmarks))
            for item in parts:
                a_item = np.rint(np.concatenate(item)).astype("int32")
                hull = cv2.convexHull(a_item)
                cv2.fillConvexPoly(mask, hull, [1.0], lineType=cv2.LINE_AA)
        return feed

    def process_output(self, batch: BatchType) -> None:
        """ Compile found faces for output """
        return

    @classmethod
    def _adjust_mask_top(cls, landmarks: np.ndarray) -> None:
        """ Adjust the top of the mask to extend above eyebrows

        Parameters
        ----------
        landmarks: :class:`numpy.ndarray`
            The 68 point landmarks to be adjusted
        """
        # mid points between the side of face and eye point
        ml_pnt = (landmarks[36] + landmarks[0]) // 2
        mr_pnt = (landmarks[16] + landmarks[45]) // 2

        # mid points between the mid points and eye
        ql_pnt = (landmarks[36] + ml_pnt) // 2
        qr_pnt = (landmarks[45] + mr_pnt) // 2

        # Top of the eye arrays
        bot_l = np.array((ql_pnt, landmarks[36], landmarks[37], landmarks[38], landmarks[39]))
        bot_r = np.array((landmarks[42], landmarks[43], landmarks[44], landmarks[45], qr_pnt))

        # Eyebrow arrays
        top_l = landmarks[17:22]
        top_r = landmarks[22:27]

        # Adjust eyebrow arrays
        landmarks[17:22] = top_l + ((top_l - bot_l) // 2)
        landmarks[22:27] = top_r + ((top_r - bot_r) // 2)

    def parse_parts(self, landmarks: np.ndarray) -> list[tuple[np.ndarray, ...]]:
        """ Extended face hull mask """
        self._adjust_mask_top(landmarks)

        r_jaw = (landmarks[0:9], landmarks[17:18])
        l_jaw = (landmarks[8:17], landmarks[26:27])
        r_cheek = (landmarks[17:20], landmarks[8:9])
        l_cheek = (landmarks[24:27], landmarks[8:9])
        nose_ridge = (landmarks[19:25], landmarks[8:9],)
        r_eye = (landmarks[17:22],
                 landmarks[27:28],
                 landmarks[31:36],
                 landmarks[8:9])
        l_eye = (landmarks[22:27],
                 landmarks[27:28],
                 landmarks[31:36],
                 landmarks[8:9])
        nose = (landmarks[27:31], landmarks[31:36])
        parts = [r_jaw, l_jaw, r_cheek, l_cheek, nose_ridge, r_eye, l_eye, nose]
        return parts
