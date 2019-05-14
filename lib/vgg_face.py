#!/usr/bin python3
""" VGG_Face inference using OpenCV-DNN
Model from: https://www.robots.ox.ac.uk/~vgg/software/vgg_face/

Licensed under Creative Commons Attribution License.
https://creativecommons.org/licenses/by-nc/4.0/
"""

import logging
import sys
import os

import cv2
import numpy as np

from lib.utils import GetModel

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class VGGFace():
    """ VGG Face feature extraction.
        Input images should be in BGR Order """

    def __init__(self):
        logger.debug("Initializing %s", self.__class__.__name__)
        git_model_id = 7
        model_filename = ["vgg_face_v1.caffemodel", "vgg_face_v1.prototxt"]
        self.input_size = 224
        # Average image provided in http://www.robots.ox.ac.uk/~vgg/software/vgg_face/
        self.average_img = [129.1863, 104.7624, 93.5940]

        self.model = self.get_model(git_model_id, model_filename)
        logger.debug("Initialized %s", self.__class__.__name__)

    # <<< GET MODEL >>> #
    @staticmethod
    def get_model(git_model_id, model_filename):
        """ Check if model is available, if not, download and unzip it """
        root_path = os.path.abspath(os.path.dirname(sys.argv[0]))
        cache_path = os.path.join(root_path, "plugins", "extract", ".cache")
        model = GetModel(model_filename, cache_path, git_model_id).model_path
        model = cv2.dnn.readNetFromCaffe(model[1], model[0])  # pylint: disable=no-member
        model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)  # pylint: disable=no-member
        return model

    def predict(self, face):
        """ Return encodings for given image from vgg_face """
        if face.shape[0] != self.input_size:
            face = self.resize_face(face)
        blob = cv2.dnn.blobFromImage(face,  # pylint: disable=no-member
                                     1.0,
                                     (self.input_size, self.input_size),
                                     self.average_img,
                                     False,
                                     False)
        self.model.setInput(blob)
        preds = self.model.forward("fc7")[0, :]
        return preds

    def resize_face(self, face):
        """ Resize incoming face to model_input_size """
        if face.shape[0] < self.input_size:
            interpolation = cv2.INTER_CUBIC  # pylint:disable=no-member
        else:
            interpolation = cv2.INTER_AREA  # pylint:disable=no-member

        face = cv2.resize(face,  # pylint:disable=no-member
                          dsize=(self.input_size, self.input_size),
                          interpolation=interpolation)
        return face

    @staticmethod
    def find_cosine_similiarity(source_face, test_face):
        """ Find the cosine similarity between a source face and a test face """
        var_a = np.matmul(np.transpose(source_face), test_face)
        var_b = np.sum(np.multiply(source_face, source_face))
        var_c = np.sum(np.multiply(test_face, test_face))
        return 1 - (var_a / (np.sqrt(var_b) * np.sqrt(var_c)))
