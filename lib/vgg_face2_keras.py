#!/usr/bin python3
""" VGG_Face2 inference
Model exported from: https://github.com/WeidiXie/Keras-VGGFace2-ResNet50
which is based on: https://www.robots.ox.ac.uk/~vgg/software/vgg_face/

Licensed under Creative Commons Attribution License.
https://creativecommons.org/licenses/by-nc/4.0/
"""

import logging
import sys
import os

import cv2
import numpy as np
from fastcluster import linkage
from lib.utils import GetModel, set_system_verbosity

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class VGGFace2():
    """ VGG Face feature extraction.
        Input images should be in BGR Order """

    def __init__(self, backend="GPU", loglevel="INFO"):
        logger.debug("Initializing %s: (backend: %s, loglevel: %s)",
                     self.__class__.__name__, backend, loglevel)
        set_system_verbosity(loglevel)
        backend = backend.upper()
        git_model_id = 10
        model_filename = ["vggface2_resnet50_v2.h5"]
        self.input_size = 224
        # Average image provided in https://github.com/ox-vgg/vgg_face2
        self.average_img = np.array([91.4953, 103.8827, 131.0912])

        self.model = self.get_model(git_model_id, model_filename, backend)
        logger.debug("Initialized %s", self.__class__.__name__)

    # <<< GET MODEL >>> #
    def get_model(self, git_model_id, model_filename, backend):
        """ Check if model is available, if not, download and unzip it """
        root_path = os.path.abspath(os.path.dirname(sys.argv[0]))
        cache_path = os.path.join(root_path, "plugins", "extract", ".cache")
        model = GetModel(model_filename, cache_path, git_model_id).model_path
        if backend == "CPU":
            if os.environ.get("KERAS_BACKEND", "") == "plaidml.keras.backend":
                logger.info("Switching to tensorflow backend.")
                os.environ["KERAS_BACKEND"] = "tensorflow"
        import keras
        from lib.model.layers import L2_normalize
        if backend == "CPU":
            with keras.backend.tf.device("/cpu:0"):
                return keras.models.load_model(model, {
                    "L2_normalize":  L2_normalize
                })
        else:
            return keras.models.load_model(model, {
                "L2_normalize":  L2_normalize
            })

    def predict(self, face):
        """ Return encodings for given image from vgg_face """
        if face.shape[0] != self.input_size:
            face = self.resize_face(face)
        face = np.expand_dims(face - self.average_img, axis=0)
        preds = self.model.predict(face)
        return preds[0, :]

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
