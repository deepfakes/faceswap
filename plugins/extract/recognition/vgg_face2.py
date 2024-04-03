#!/usr/bin python3
""" VGG_Face2 inference and sorting """

from __future__ import annotations
import logging
import typing as T

import numpy as np
import psutil
from fastcluster import linkage, linkage_vector

from lib.model.layers import L2_normalize
from lib.model.session import KSession
from lib.utils import FaceswapError
from ._base import BatchType, RecogBatch, Identity

if T.TYPE_CHECKING:
    from collections.abc import Generator

logger = logging.getLogger(__name__)


class Recognition(Identity):
    """ VGG Face feature extraction.

    Extracts feature vectors from faces in order to compare similarity.

    Notes
    -----
    Input images should be in BGR Order

    Model exported from: https://github.com/WeidiXie/Keras-VGGFace2-ResNet50 which is based on:
    https://www.robots.ox.ac.uk/~vgg/software/vgg_face/


    Licensed under Creative Commons Attribution License.
    https://creativecommons.org/licenses/by-nc/4.0/
    """

    def __init__(self, *args, **kwargs) -> None:  # pylint:disable=unused-argument
        logger.debug("Initializing %s", self.__class__.__name__)
        git_model_id = 10
        model_filename = "vggface2_resnet50_v2.h5"
        super().__init__(git_model_id=git_model_id, model_filename=model_filename, **kwargs)
        self.model: KSession
        self.name: str = "VGGFace2"
        self.input_size = 224
        self.color_format = "BGR"

        self.vram = 2468 if not self.config["cpu"] else 0
        self.vram_warnings = 192 if not self.config["cpu"] else 0
        self.vram_per_batch = 32 if not self.config["cpu"] else 0
        self.batchsize = self.config["batch-size"]

        # Average image provided in https://github.com/ox-vgg/vgg_face2
        self._average_img = np.array([91.4953, 103.8827, 131.0912])
        logger.debug("Initialized %s", self.__class__.__name__)

    # <<< GET MODEL >>> #
    def init_model(self) -> None:
        """ Initialize VGG Face 2 Model. """
        assert isinstance(self.model_path, str)
        model_kwargs = {"custom_objects": {"L2_normalize": L2_normalize}}
        self.model = KSession(self.name,
                              self.model_path,
                              model_kwargs=model_kwargs,
                              allow_growth=self.config["allow_growth"],
                              exclude_gpus=self._exclude_gpus,
                              cpu_mode=self.config["cpu"])
        self.model.load_model()

    def process_input(self, batch: BatchType) -> None:
        """ Compile the detected faces for prediction """
        assert isinstance(batch, RecogBatch)
        batch.feed = np.array([T.cast(np.ndarray, feed.face)[..., :3]
                               for feed in batch.feed_faces],
                              dtype="float32") - self._average_img
        logger.trace("feed shape: %s", batch.feed.shape)  # type:ignore

    def predict(self, feed: np.ndarray) -> np.ndarray:
        """ Return encodings for given image from vgg_face2.

        Parameters
        ----------
        batch: numpy.ndarray
            The face to be fed through the predictor. Should be in BGR channel order

        Returns
        -------
        numpy.ndarray
            The encodings for the face
        """
        retval = self.model.predict(feed)
        assert isinstance(retval, np.ndarray)
        return retval

    def process_output(self, batch: BatchType) -> None:
        """ No output processing for  vgg_face2 """
        return


class Cluster():  # pylint:disable=too-few-public-methods
    """ Cluster the outputs from a VGG-Face 2 Model

    Parameters
    ----------
    predictions: numpy.ndarray
        A stacked matrix of vgg_face2 predictions of the shape (`N`, `D`) where `N` is the
        number of observations and `D` are the number of dimensions.  NB: The given
        :attr:`predictions` will be overwritten to save memory. If you still require the
        original values you should take a copy prior to running this method
    method: ['single','centroid','median','ward']
        The clustering method to use.
    threshold: float, optional
        The threshold to start creating bins for. Set to ``None`` to disable binning
    """

    def __init__(self,
                 predictions: np.ndarray,
                 method: T.Literal["single", "centroid", "median", "ward"],
                 threshold: float | None = None) -> None:
        logger.debug("Initializing: %s (predictions: %s, method: %s, threshold: %s)",
                     self.__class__.__name__, predictions.shape, method, threshold)
        self._num_predictions = predictions.shape[0]

        self._should_output_bins = threshold is not None
        self._threshold = 0.0 if threshold is None else threshold
        self._bins: dict[int, int] = {}
        self._iterator = self._integer_iterator()

        self._result_linkage = self._do_linkage(predictions, method)
        logger.debug("Initialized %s", self.__class__.__name__)

    @classmethod
    def _integer_iterator(cls) -> Generator[int, None, None]:
        """ Iterator that just yields consecutive integers """
        i = -1
        while True:
            i += 1
            yield i

    def _use_vector_linkage(self, dims: int) -> bool:
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
        linkage_required = (((self._num_predictions ** 2) * np_float) / 1.8) / divider
        vector_required = ((self._num_predictions * dims) * np_float) / divider
        logger.debug("free_ram: %sMB, linkage_required: %sMB, vector_required: %sMB",
                     int(free_ram), int(linkage_required), int(vector_required))

        if linkage_required < free_ram:
            logger.verbose("Using linkage method")  # type:ignore
            retval = False
        elif vector_required < free_ram:
            logger.warning("Not enough RAM to perform linkage clustering. Using vector "
                           "clustering. This will be significantly slower. Free RAM: %sMB. "
                           "Required for linkage method: %sMB",
                           int(free_ram), int(linkage_required))
            retval = True
        else:
            raise FaceswapError("Not enough RAM available to sort faces. Try reducing "
                                f"the size of  your dataset. Free RAM: {int(free_ram)}MB. "
                                f"Required RAM: {int(vector_required)}MB")
        logger.debug(retval)
        return retval

    def _do_linkage(self,
                    predictions: np.ndarray,
                    method: T.Literal["single", "centroid", "median", "ward"]) -> np.ndarray:
        """ Use FastCluster to perform vector or standard linkage

        Parameters
        ----------
        predictions: :class:`numpy.ndarray`
            A stacked matrix of vgg_face2 predictions of the shape (`N`, `D`) where `N` is the
            number of observations and `D` are the number of dimensions.
        method: ['single','centroid','median','ward']
            The clustering method to use.

        Returns
        -------
        :class:`numpy.ndarray`
            The [`num_predictions`, 4] linkage vector
        """
        dims = predictions.shape[-1]
        if self._use_vector_linkage(dims):
            retval = linkage_vector(predictions, method=method)
        else:
            retval = linkage(predictions, method=method, preserve_input=False)
        logger.debug("Linkage shape: %s", retval.shape)
        return retval

    def _process_leaf_node(self,
                           current_index: int,
                           current_bin: int) -> list[tuple[int, int]]:
        """ Process the output when we have hit a leaf node """
        if not self._should_output_bins:
            return [(current_index, 0)]

        if current_bin not in self._bins:
            next_val = 0 if not self._bins else max(self._bins.values()) + 1
            self._bins[current_bin] = next_val
        return [(current_index, self._bins[current_bin])]

    def _get_bin(self,
                 tree: np.ndarray,
                 points: int,
                 current_index: int,
                 current_bin: int) -> int:
        """ Obtain the bin that we are currently in.

        If we are not currently below the threshold for binning, get a new bin ID from the integer
        iterator.

        Parameters
        ----------
        tree: numpy.ndarray
           A hierarchical tree (dendrogram)
        points: int
            The number of points given to the clustering process
        current_index: int
            The position in the tree for the recursive traversal
        current_bin int, optional
            The ID for the bin we are currently in. Only used when binning is enabled

        Returns
        -------
        int
            The current bin ID for the node
        """
        if tree[current_index - points, 2] >= self._threshold:
            current_bin = next(self._iterator)
            logger.debug("Creating new bin ID: %s", current_bin)
        return current_bin

    def _seriation(self,
                   tree: np.ndarray,
                   points: int,
                   current_index: int,
                   current_bin: int = 0) -> list[tuple[int, int]]:
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
        current_bin int, optional
            The ID for the bin we are currently in. Only used when binning is enabled

        Returns
        -------
        list:
            The indices in the order implied by the hierarchical tree
        """
        if current_index < points:  # Output the leaf node
            return self._process_leaf_node(current_index, current_bin)

        if self._should_output_bins:
            current_bin = self._get_bin(tree, points, current_index, current_bin)

        left = int(tree[current_index-points, 0])
        right = int(tree[current_index-points, 1])

        serate_left = self._seriation(tree, points, left, current_bin=current_bin)
        serate_right = self._seriation(tree, points, right, current_bin=current_bin)

        return serate_left + serate_right  # type: ignore

    def __call__(self) -> list[tuple[int, int]]:
        """ Process the linkages.

        Transforms a distance matrix into a sorted distance matrix according to the order implied
        by the hierarchical tree (dendrogram).

        Returns
        -------
        list:
            List of indices with the order implied by the hierarchical tree or list of tuples of
            (`index`, `bin`) if a binning threshold was provided
        """
        logger.info("Sorting face distances. Depending on your dataset this may take some time...")
        if self._threshold:
            self._threshold = self._result_linkage[:, 2].max() * self._threshold
        result_order = self._seriation(self._result_linkage,
                                       self._num_predictions,
                                       self._num_predictions + self._num_predictions - 2)
        return result_order
