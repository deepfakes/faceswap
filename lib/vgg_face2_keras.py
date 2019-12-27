#!/usr/bin python3
""" VGG_Face2 inference and sorting """

import logging
import sys
import os
import psutil

import cv2
import numpy as np
from fastcluster import linkage, linkage_vector
from lib.utils import GetModel, FaceswapError

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class VGGFace2():
    """ VGG Face feature extraction.

    Extracts feature vectors from faces in order to compare similarity.

    Parameters
    ----------
    backend: ['GPU', 'CPU']
        Whether to run inference on a GPU or on the CPU
    loglevel: ['INFO', 'VERBODE', 'DEBUG', 'TRACE']
        The system log level

    Notes
    -----
    Input images should be in BGR Order

    Model exported from: https://github.com/WeidiXie/Keras-VGGFace2-ResNet50 which is based on:
    https://www.robots.ox.ac.uk/~vgg/software/vgg_face/


    Licensed under Creative Commons Attribution License.
    https://creativecommons.org/licenses/by-nc/4.0/
    """

    def __init__(self, backend="GPU", loglevel="INFO"):
        logger.debug("Initializing %s: (backend: %s, loglevel: %s)",
                     self.__class__.__name__, backend, loglevel)
        backend = backend.upper()
        git_model_id = 10
        model_filename = ["vggface2_resnet50_v2.h5"]
        self.input_size = 224
        # Average image provided in https://github.com/ox-vgg/vgg_face2
        self.average_img = np.array([91.4953, 103.8827, 131.0912])

        self.model = self._get_model(git_model_id, model_filename, backend)
        logger.debug("Initialized %s", self.__class__.__name__)

    # <<< GET MODEL >>> #
    @staticmethod
    def _get_model(git_model_id, model_filename, backend):
        """ Check if model is available, if not, download and unzip it

        Parameters
        ----------
        git_model_id: int
            The second digit in the github tag that identifies this model. See
            https://github.com/deepfakes-models/faceswap-models for more information
        model_filename: str
            The name of the model to be loaded (see :class:`lib.utils.GetModel` for more
            information)
        backend: ['GPU', 'CPU']
            Whether to run inference on a GPU or on the CPU

        See Also
        --------
        lib.utils.GetModel: The model downloading and allocation class.
        """
        root_path = os.path.abspath(os.path.dirname(sys.argv[0]))
        cache_path = os.path.join(root_path, "plugins", "extract", "recognition", ".cache")
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
        """ Return encodings for given image from vgg_face2.

        Parameters
        ----------
        face: numpy.ndarray
            The face to be fed through the predictor. Should be in BGR channel order

        Returns
        -------
        numpy.ndarray
            The encodings for the face
        """
        if face.shape[0] != self.input_size:
            face = self._resize_face(face)
        face = face[None, :, :, :3] - self.average_img
        preds = self.model.predict(face)
        return preds[0, :]

    def _resize_face(self, face):
        """ Resize incoming face to model_input_size.

        Parameters
        ----------
        face: numpy.ndarray
            The face to be fed through the predictor. Should be in BGR channel order

        Returns
        -------
        numpy.ndarray
            The face resized to model input size
        """
        sizes = (self.input_size, self.input_size)
        interpolation = cv2.INTER_CUBIC if face.shape[0] < self.input_size else cv2.INTER_AREA
        face = cv2.resize(face, dsize=sizes, interpolation=interpolation)
        return face

    @staticmethod
    def find_cosine_similiarity(source_face, test_face):
        """ Find the cosine similarity between two faces.

        Parameters
        ----------
        source_face: numpy.ndarray
            The first face to test against :attr:`test_face`
        test_face: numpy.ndarray
            The second face to test against :attr:`source_face`

        Returns
        -------
        float:
            The cosine similarity between the two faces
        """
        var_a = np.matmul(np.transpose(source_face), test_face)
        var_b = np.sum(np.multiply(source_face, source_face))
        var_c = np.sum(np.multiply(test_face, test_face))
        return 1 - (var_a / (np.sqrt(var_b) * np.sqrt(var_c)))

    def sorted_similarity(self, predictions, method="ward"):
        """ Sort a matrix of predictions by similarity.

        Transforms a distance matrix into a sorted distance matrix according to the order implied
        by the hierarchical tree (dendrogram).

        Parameters
        ----------
        predictions: numpy.ndarray
            A stacked matrix of vgg_face2 predictions of the shape (`N`, `D`) where `N` is the
            number of observations and `D` are the number of dimensions.  NB: The given
            :attr:`predictions` will be overwritten to save memory. If you still require the
            original values you should take a copy prior to running this method
        method: ['single','centroid','median','ward']
            The clustering method to use.

        Returns
        -------
        list:
            List of indices with the order implied by the hierarchical tree
        """
        logger.info("Sorting face distances. Depending on your dataset this may take some time...")
        num_predictions, dims = predictions.shape

        kwargs = dict(method=method)
        if self._use_vector_linkage(num_predictions, dims):
            func = linkage_vector
        else:
            kwargs["preserve_input"] = False
            func = linkage

        result_linkage = func(predictions, **kwargs)
        result_order = self._seriation(result_linkage,
                                       num_predictions,
                                       num_predictions + num_predictions - 2)
        return result_order

    @staticmethod
    def _use_vector_linkage(item_count, dims):
        """ Calculate the RAM that will be required to sort these images and select the appropriate
        clustering method.

        From fastcluster documentation:
            "While the linkage method requires Θ(N:sup:`2`) memory for clustering of N points, this
            [vector] method needs Θ(N D)for N points in RD, which is usually much smaller."
            also:
            "half the memory can be saved by specifying :attr:`preserve_input`=``False``"

        To avoid under calculating we divide the memory calculation by 1.8 instead of 2

        Parameters
        ----------
        item_count: int
            The number of images that are to be processed
        dims: int
            The number of dimensions in the vgg_face output

        Returns
        -------
            bool:
                ``True`` if vector_linkage should be used. ``False`` if linkage should be used
        """
        np_float = 24  # bytes size of a numpy float
        divider = 1024 * 1024  # bytes to MB

        free_ram = psutil.virtual_memory().available / divider
        linkage_required = (((item_count ** 2) * np_float) / 1.8) / divider
        vector_required = ((item_count * dims) * np_float) / divider
        logger.debug("free_ram: %sMB, linkage_required: %sMB, vector_required: %sMB",
                     int(free_ram), int(linkage_required), int(vector_required))

        if linkage_required < free_ram:
            logger.verbose("Using linkage method")
            retval = False
        elif vector_required < free_ram:
            logger.warning("Not enough RAM to perform linkage clustering. Using vector "
                           "clustering. This will be significantly slower. Free RAM: %sMB. "
                           "Required for linkage method: %sMB",
                           int(free_ram), int(linkage_required))
            retval = True
        else:
            raise FaceswapError("Not enough RAM available to sort faces. Try reducing "
                                "the size of  your dataset. Free RAM: {}MB. "
                                "Required RAM: {}MB".format(int(free_ram), int(vector_required)))
        logger.debug(retval)
        return retval

    def _seriation(self, tree, points, current_index):
        """ Seriation method for sorted similarity.

        Seriation computes the order implied by a hierarchical tree (dendrogram).

        Parameters
        ----------
        tree: numpy.ndarray
           A hierarchical tree (dendrogram)
        points: int
            The number of points given to the clustering process
        current_index: int
            The position in the tree for the recursive traversal

        Returns
        -------
        list:
            The indices in the order implied by the hierarchical tree
        """
        if current_index < points:
            return [current_index]
        left = int(tree[current_index-points, 0])
        right = int(tree[current_index-points, 1])
        return self._seriation(tree, points, left) + self._seriation(tree, points, right)
