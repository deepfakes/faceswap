#!/usr/bin python3
""" VGG_Face inference using OpenCV-DNN
Model from: https://www.robots.ox.ac.uk/~vgg/software/vgg_face/

Licensed under Creative Commons Attribution License.
https://creativecommons.org/licenses/by-nc/4.0/
"""

import logging

import cv2
import numpy as np
from fastcluster import linkage

from lib.utils import GetModel

logger = logging.getLogger(__name__)


class VGGFace():
    """ VGG Face feature extraction.
        Input images should be in BGR Order """

    def __init__(self, backend="CPU"):
        logger.debug("Initializing %s: (backend: %s)", self.__class__.__name__, backend)
        git_model_id = 7
        model_filename = ["vgg_face_v1.caffemodel", "vgg_face_v1.prototxt"]
        self.input_size = 224
        # Average image provided in http://www.robots.ox.ac.uk/~vgg/software/vgg_face/
        self.average_img = [129.1863, 104.7624, 93.5940]

        self.model = self.get_model(git_model_id, model_filename, backend)
        logger.debug("Initialized %s", self.__class__.__name__)

    # <<< GET MODEL >>> #
    def get_model(self, git_model_id, model_filename, backend):
        """ Check if model is available, if not, download and unzip it """
        model = GetModel(model_filename, git_model_id).model_path
        model = cv2.dnn.readNetFromCaffe(model[1], model[0])
        model.setPreferableTarget(self.get_backend(backend))
        return model

    @staticmethod
    def get_backend(backend):
        """ Return the cv2 DNN backend """
        if backend == "OPENCL":
            logger.info("Using OpenCL backend. If the process runs, you can safely ignore any of "
                        "the failure messages.")
        retval = getattr(cv2.dnn, f"DNN_TARGET_{backend}")
        return retval

    def predict(self, face):
        """ Return encodings for given image from vgg_face """
        if face.shape[0] != self.input_size:
            face = self.resize_face(face)
        blob = cv2.dnn.blobFromImage(face[..., :3],
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
        sizes = (self.input_size, self.input_size)
        interpolation = cv2.INTER_CUBIC if face.shape[0] < self.input_size else cv2.INTER_AREA
        face = cv2.resize(face, dsize=sizes, interpolation=interpolation)
        return face

    @staticmethod
    def find_cosine_similiarity(source_face, test_face):
        """ Find the cosine similarity between a source face and a test face """
        var_a = np.matmul(np.transpose(source_face), test_face)
        var_b = np.sum(np.multiply(source_face, source_face))
        var_c = np.sum(np.multiply(test_face, test_face))
        return 1 - (var_a / (np.sqrt(var_b) * np.sqrt(var_c)))

    def sorted_similarity(self, predictions, method="ward"):
        """ Sort a matrix of predictions by similarity Adapted from:
            https://gmarti.gitlab.io/ml/2017/09/07/how-to-sort-distance-matrix.html
        input:
            - predictions is a stacked matrix of vgg_face predictions shape: (x, 4096)
            - method = ["ward","single","average","complete"]
        output:
            - result_order is a list of indices with the order implied by the hierarhical tree

        sorted_similarity transforms a distance matrix into a sorted distance matrix according to
        the order implied by the hierarchical tree (dendrogram)
        """
        logger.info("Sorting face distances. Depending on your dataset this may take some time...")
        num_predictions = predictions.shape[0]
        result_linkage = linkage(predictions, method=method, preserve_input=False)
        result_order = self.seriation(result_linkage,
                                      num_predictions,
                                      num_predictions + num_predictions - 2)

        return result_order

    def seriation(self, tree, points, current_index):
        """ Seriation method for sorted similarity
            input:
                - tree is a hierarchical tree (dendrogram)
                - points is the number of points given to the clustering process
                - current_index is the position in the tree for the recursive traversal
            output:
                - order implied by the hierarchical tree

            seriation computes the order implied by a hierarchical tree (dendrogram)
        """
        if current_index < points:
            return [current_index]
        left = int(tree[current_index-points, 0])
        right = int(tree[current_index-points, 1])
        return self.seriation(tree, points, left) + self.seriation(tree, points, right)
