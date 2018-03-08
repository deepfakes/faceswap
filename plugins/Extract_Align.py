# Based on the original https://www.reddit.com/r/deepfakes/ code sample + contribs

import cv2
import numpy as np

from lib.aligner import get_align_mat

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
