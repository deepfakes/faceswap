# Face Mesh. From https://github.com/juniorxsound/Face-Align/blob/master/facemesh.py
# Written by Or Fleisher for Data Art class taught in ITP, NYU during fall 2017 by Genevieve Hoffman.
# Based on Leon Eckerts code from the facemesh workshop - https://github.com/leoneckert/facemash-workshop

import cv2
import dlib
import numpy as np


class Aligner:
    def __init__(self, pred, detect):
        self.detector = dlib.cnn_face_detection_model_v1(detect)
        self.predictor = dlib.shape_predictor(pred)

    def get_rects(self, img):
        rects = self.detector(img, 1)
        # print("[+] Number of faces found:", len(rects))
        return rects

    def get_first_rect(self, img):
        rects = self.get_rects(img)
        if len(rects) > 0:
            return rects[0].rect
        else:
            return None

    def get_landmarks(self, img, rect):
        return np.matrix([[p.x, p.y] for p in self.predictor(img, rect).parts()])

    # https://matthewearl.github.io/2015/07/28/switching-eds-with-python/
    def transformation_from_points(self, points1, points2):
        points1 = points1.astype(np.float64)
        points2 = points2.astype(np.float64)

        c1 = np.mean(points1, axis=0)
        c2 = np.mean(points2, axis=0)
        points1 -= c1
        points2 -= c2

        s1 = np.std(points1)
        s2 = np.std(points2)
        points1 /= s1
        points2 /= s2

        U, S, Vt = np.linalg.svd(points1.T * points2)
        R = (U * Vt).T

        return np.vstack([np.hstack(((s2 / s1) * R,
                                     c2.T - (s2 / s1) * R * c1.T)),
                          np.matrix([0., 0., 1.])])

    def warp_im(self, im, ref, M):
        dshape = ref.shape
        output_im = ref.copy()
        translationMatrix = np.matrix([0, 0])
        cv2.warpAffine(im,
                       M[:2],
                       (dshape[1], dshape[0]),
                       dst=output_im,
                       borderMode=cv2.BORDER_TRANSPARENT,
                       flags=cv2.WARP_INVERSE_MAP)

        return output_im

    def align(self, ref_img, img):
        # TODO optimize with a one step detection for both images
        ref_rect = self.get_first_rect(ref_img)
        if ref_rect is None:
            return None
        ref_landmarks = self.get_landmarks(ref_img, ref_rect)

        rect = self.get_first_rect(img)
        if rect is None:
            return None
        landmarks = self.get_landmarks(img, rect)

        transformation_matrix = self.transformation_from_points(
            ref_landmarks, landmarks)
        warped_img = self.warp_im(img, ref_img, transformation_matrix)

        #cv2.imwrite( 'modified/_aligned.png', warped_img )
        #cv2.imwrite( 'modified/_align_ref.png', ref_img )
        #cv2.imwrite( 'modified/_align_generated.png', img )
        return warped_img
