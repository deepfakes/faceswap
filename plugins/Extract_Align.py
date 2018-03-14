# Based on the original https://www.reddit.com/r/deepfakes/ code sample + contribs

import cv2
import numpy as np

from lib.aligner import get_align_mat
from lib.align_eyes import FACIAL_LANDMARKS_IDXS

class Extract(object):
    def extract(self, image, face, size, align_eyes):
        alignment = get_align_mat(face, size, align_eyes)
        extracted = self.transform(image, alignment, size, 48)
        return extracted, alignment

    def transform(self, image, mat, size, padding=0):
        matrix = mat * (size - 2 * padding)
        matrix[:,2] += padding
        return cv2.warpAffine(image, matrix, (size, size))

    def transform_points(self, points, mat, size, padding=0):
        matrix = mat * (size - 2 * padding)
        matrix[:,2] += padding
        points = np.expand_dims(points, axis=1)
        points = cv2.transform(points, matrix, points.shape)
        points = np.squeeze(points)
        return points

    def get_feature_mask(self, aligned_landmarks_68, size, padding=0, dilation=30):
        scale = size - 2*padding
        translation = padding
        pad_mat = np.matrix([[scale, 0.0, translation], [0.0, scale, translation]])
        aligned_landmarks_68 = np.expand_dims(aligned_landmarks_68, axis=1)
        aligned_landmarks_68 = cv2.transform(aligned_landmarks_68, pad_mat, aligned_landmarks_68.shape)
        aligned_landmarks_68 = np.squeeze(aligned_landmarks_68)

        (lStart, lEnd) = FACIAL_LANDMARKS_IDXS["left_eye"]
        (rStart, rEnd) = FACIAL_LANDMARKS_IDXS["right_eye"]
        (mStart, mEnd) = FACIAL_LANDMARKS_IDXS["mouth"]
        (nStart, nEnd) = FACIAL_LANDMARKS_IDXS["nose"]
        (lbStart, lbEnd) = FACIAL_LANDMARKS_IDXS["left_eyebrow"]
        (rbStart, rbEnd) = FACIAL_LANDMARKS_IDXS["right_eyebrow"]
        (cStart, cEnd) = FACIAL_LANDMARKS_IDXS["chin"]

        l_eye_points = aligned_landmarks_68[lStart:lEnd].tolist()
        l_brow_points = aligned_landmarks_68[lbStart:lbEnd].tolist()
        r_eye_points = aligned_landmarks_68[rStart:rEnd].tolist()
        r_brow_points = aligned_landmarks_68[rbStart:rbEnd].tolist()
        nose_points = aligned_landmarks_68[nStart:nEnd].tolist()
        chin_points = aligned_landmarks_68[cStart:cEnd].tolist()
        mouth_points = aligned_landmarks_68[mStart:mEnd].tolist()
        l_eye_points = l_eye_points + l_brow_points
        r_eye_points = r_eye_points + r_brow_points
        mouth_points = mouth_points + nose_points + chin_points

        l_eye_hull = cv2.convexHull(np.array(l_eye_points).reshape((-1, 2)).astype(int)).flatten().reshape((-1, 2))
        r_eye_hull = cv2.convexHull(np.array(r_eye_points).reshape((-1, 2)).astype(int)).flatten().reshape((-1, 2))
        mouth_hull = cv2.convexHull(np.array(mouth_points).reshape((-1, 2)).astype(int)).flatten().reshape((-1, 2))

        mask = np.zeros((size, size, 3), dtype=float)
        cv2.fillConvexPoly(mask, l_eye_hull, (1,1,1))
        cv2.fillConvexPoly(mask, r_eye_hull, (1,1,1))
        cv2.fillConvexPoly(mask, mouth_hull, (1,1,1))

        if dilation > 0:
            kernel = np.ones((dilation, dilation), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1)

        return mask
