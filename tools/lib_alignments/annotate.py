#!/usr/bin/env python3
""" Tools for annotating an input image """

from cv2 import (rectangle, circle, polylines, putText,
                 FONT_HERSHEY_DUPLEX, invertAffineTransform, transform,
                 getRotationMatrix2D, warpAffine, fillPoly, addWeighted)
from numpy import array, int32

from lib.align_eyes import FACIAL_LANDMARKS_IDXS
from plugins.PluginLoader import PluginLoader
from . import DetectedFace

EXTRACTOR = None


def load_extractor():
    """ Extractor has to global otherwise it spams
        messages for every face. This is a hacky fix :/ """
    global EXTRACTOR
    if not EXTRACTOR:
        EXTRACTOR = PluginLoader.get_extractor("Align")()
    return EXTRACTOR


class Annotate():
    """ Copy and annotate an input image """

    def __init__(self, image, alignments, align_eyes=False,
                 adjusted_roi=None, are_faces=False):
        self.is_face = are_faces
        self.image = self.set_image(image)
        self.alignments = self.set_alignments(alignments, image)
        self.align_eyes = align_eyes
        self.roi = self.get_roi(adjusted_roi)
        self.colors = {1: (255, 0, 0),
                       2: (0, 255, 0),
                       3: (0, 0, 255),
                       4: (255, 255, 0),
                       5: (255, 0, 255),
                       6: (0, 255, 255)}

    def set_image(self, image):
        """ Set image or list of images """
        if self.is_face:
            return [face.face for face in image]
        return image.copy()

    def set_alignments(self, alignments, image):
        """ Set image or face alignments """
        if self.is_face:
            return [{"landmarksXY": face.aligned_landmarks}
                    for face in image]
        return alignments

    def get_roi(self, adjusted_roi):
        """ Extract ROI if not provided """
        if self.is_face:
            return None
        if adjusted_roi:
            return adjusted_roi

        roi = [ExtractedFace(self.image,
                             alignment,
                             align_eyes=self.align_eyes).original_roi
               for alignment in self.alignments]
        return roi

    def draw_bounding_box(self, color_id, thickness):
        """ Draw the bounding box around faces """
        if self.is_face:
            return
        color = self.colors[color_id]
        for alignment in self.alignments:
            top_left = (alignment["x"], alignment["y"])
            bottom_right = (alignment["x"] + alignment["w"],
                            alignment["y"] + alignment["h"])
            rectangle(self.image, top_left, bottom_right,
                      color, thickness)

    def draw_extract_box(self, color_id, thickness):
        """ Draw the extracted face box """
        if self.is_face:
            return
        color = self.colors[color_id]
        for idx in range(len(self.alignments)):
            roi = self.roi[idx]
            top_left = [point for point in roi[0].squeeze()[0]]
            top_left = (top_left[0], top_left[1] - 10)
            putText(self.image, str(idx), top_left, FONT_HERSHEY_DUPLEX, 1.0,
                    color, thickness)
            polylines(self.image, roi, True, color, thickness)

    def draw_landmarks(self, color_id, radius):
        """ Draw the facial landmarks """
        color = self.colors[color_id]
        for idx, alignment in enumerate(self.alignments):
            image = self.image[idx] if self.is_face else self.image
            landmarks = alignment["landmarksXY"]
            for (pos_x, pos_y) in landmarks:
                circle(image, (pos_x, pos_y), radius, color, -1)

    def draw_landmarks_mesh(self, color_id, thickness):
        """ Draw the facial landmarks """
        color = self.colors[color_id]
        for idx, alignment in enumerate(self.alignments):
            image = self.image[idx] if self.is_face else self.image
            landmarks = alignment["landmarksXY"]
            for key, val in FACIAL_LANDMARKS_IDXS.items():
                points = array([landmarks[val[0]:val[1]]], int32)
                fill_poly = bool(key in ("right_eye", "left_eye", "mouth"))
                polylines(image, points, fill_poly, color, thickness)

    def draw_grey_out_faces(self, live_face):
        """ Grey out all faces except target """
        alpha = 0.6
        if not self.is_face:
            overlay = self.image.copy()
            for idx in range(len(self.alignments)):
                roi = self.roi[idx]
                if idx != int(live_face):
                    fillPoly(overlay, roi, (0, 0, 0))
            addWeighted(overlay, alpha, self.image, 1 - alpha, 0, self.image)
        else:
            for idx, face in enumerate(self.image):
                overlay = face.copy()
                size = overlay.shape[0] - 1
                if idx != int(live_face):
                    rectangle(overlay, (0, 0), (size, size), (0, 0, 0), -1)
                    addWeighted(overlay, alpha, face, 1 - alpha, 0, face)
                    print("execute")


class ExtractedFace():
    """ Holds the extracted face and matrix for
        a single face alignment """
    def __init__(self, image, alignments, size=256,
                 padding=48, align_eyes=False):
        self.size = size
        self.face = None
        self.matrix = None
        self.extractor = load_extractor()
        self.rotation_size = None

        self.get_face(image, alignments, align_eyes)

        self.original_roi = self.get_original_roi(padding)
        self.aligned_landmarks = self.transpose_landmarks(alignments, padding)

    def get_face(self, image, alignments, align_eyes):
        """ Return the transformed face thumbnail matrices
            for this alignment """
        face = DetectedFace(image,
                            alignments["r"],
                            alignments["x"],
                            alignments["w"],
                            alignments["y"],
                            alignments["h"],
                            alignments["landmarksXY"])

        image = self.rotate_image(image, face.r)

        resized_face, matrix = self.extractor.extract(image,
                                                      face,
                                                      self.size,
                                                      align_eyes)

        image = self.rotate_image(image, face.r, reverse=True)

        self.face = resized_face
        self.matrix = matrix

    def rotate_image(self, image, rotation, reverse=False):
        """ Rotate the image forwards or backwards """
        if rotation == 0:
            return image
        if not reverse:
            self.rotation_size = image.shape[:2]
            image = self.rotate_image_by_angle(image, rotation)
        else:
            image = self.rotate_image_by_angle(
                image,
                rotation * -1,
                rotated_width=self.rotation_size[0],
                rotated_height=self.rotation_size[1])
        return image

    @staticmethod
    def rotate_image_by_angle(image, angle,
                              rotated_width=None, rotated_height=None):
        """ Rotate an image by a given angle.
            From: https://stackoverflow.com/questions/22041699 """
        height, width = image.shape[:2]
        image_center = (width/2, height/2)
        rotation_matrix = getRotationMatrix2D(image_center, -1.*angle, 1.)
        if rotated_width is None or rotated_height is None:
            abs_cos = abs(rotation_matrix[0, 0])
            abs_sin = abs(rotation_matrix[0, 1])
            if rotated_width is None:
                rotated_width = int(height*abs_sin + width*abs_cos)
            if rotated_height is None:
                rotated_height = int(height*abs_cos + width*abs_sin)
        rotation_matrix[0, 2] += rotated_width/2 - image_center[0]
        rotation_matrix[1, 2] += rotated_height/2 - image_center[1]
        return warpAffine(image, rotation_matrix,
                          (rotated_width, rotated_height))

    def get_original_roi(self, padding):
        """ Return the original ROI of an extracted face """
        points = array([[0, 0], [0, self.size - 1],
                        [self.size - 1, self.size - 1],
                        [self.size - 1, 0]], int32)
        points = points.reshape((-1, 1, 2))

        matrix = self.matrix * (self.size - 2 * padding)
        matrix[:, 2] += padding
        matrix = invertAffineTransform(matrix)
        return [transform(points, matrix)]

    def transpose_landmarks(self, alignments, padding):
        """ Transpose original landmarks to thumbnail image """
        landmarks_xy = alignments["landmarksXY"]
        return self.extractor.transform_points(
            landmarks_xy, self.matrix, self.size, padding)
