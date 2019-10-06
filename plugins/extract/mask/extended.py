#!/usr/bin/env python3

import cv2
import numpy as np
from ._base import Masker, logger


class Mask(Masker):
    """ Perform transformation to align and get landmarks """
    def __init__(self, **kwargs):
        git_model_id = None
        model_filename = None
        super().__init__(git_model_id=git_model_id, model_filename=model_filename, **kwargs)
        self.name = "Extended"
        self.colorformat = "BGR"
        self.vram = 0
        self.vram_warnings = 0
        self.vram_per_batch = 30
        self.batchsize = self.config["batch-size"]

    def init_model(self):
        logger.debug("No mask model to initialize")

    def process_input(self, batch):
        """ Compile the detected faces for prediction """
        batch["feed"] = np.array([face.image for face in batch["detected_faces"]])
        return batch

    def predict(self, batch):
        """ Run model to get predictions """
        masks = np.zeros(batch["feed"].shape[:-1] + (1,), dtype='uint8')
        for mask, face in zip(masks, batch["detected_faces"]):
            parts = self.parse_parts(np.array(face.landmarks_xy))
            for item in parts:
                item = np.concatenate(item)
                hull = cv2.convexHull(item).astype('int32')  # pylint: disable=no-member
                cv2.fillConvexPoly(mask, hull, 255, lineType=cv2.LINE_AA)
        batch["prediction"] = masks
        return batch

    def process_output(self, batch):
        """ Compile found faces for output """
        generator = zip(batch["feed"], batch["detected_faces"], batch["prediction"])
        for feed, face, prediction in generator:
            face.image = np.concatenate((feed, prediction), axis=-1)
            face.load_feed_face(face.image,
                                size=self.input_size,
                                coverage_ratio=self.coverage_ratio)
            face.load_reference_face(face.image,
                                     size=self.output_size,
                                     coverage_ratio=self.coverage_ratio)
        return batch

    @staticmethod
    def parse_parts(landmarks):
        """ Extended facehull mask """
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
