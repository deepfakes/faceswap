#!/usr/bin python3
""" VGG_Face2 inference and sorting """

from __future__ import annotations
import logging
import typing as T

import numpy as np
import psutil
from fastcluster import linkage, linkage_vector
from keras.layers import (Activation, add, AveragePooling2D, BatchNormalization, Conv2D, Dense,
                          Flatten, Input, MaxPooling2D)
from keras.models import Model
from keras.regularizers import L2

from lib.logger import parse_class_init
from lib.model.layers import L2Normalize
from lib.utils import get_module_objects, FaceswapError
from ._base import BatchType, RecogBatch, Identity
from . import vgg_face2_defaults as cfg

if T.TYPE_CHECKING:
    from keras import KerasTensor
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

    def __init__(self, **kwargs) -> None:
        logger.debug("Initializing %s", self.__class__.__name__)
        git_model_id = 10
        model_filename = "vggface2_resnet50_v2.h5"
        super().__init__(git_model_id=git_model_id, model_filename=model_filename, **kwargs)
        self.model: Model
        self.name: str = "VGGFace2"
        self.input_size = 224
        self.color_format = "BGR"

        self.vram = 384 if not cfg.cpu() else 0  # 334 in testing
        self.vram_per_batch = 192 if not cfg.cpu() else 0  # ~155 in testing
        self.batchsize = cfg.batch_size()

        # Average image provided in https://github.com/ox-vgg/vgg_face2
        self._average_img = np.array([91.4953, 103.8827, 131.0912])
        logger.debug("Initialized %s", self.__class__.__name__)

    # <<< GET MODEL >>> #
    def init_model(self) -> None:
        """ Initialize VGG Face 2 Model. """
        assert isinstance(self.model_path, str)
        placeholder = np.zeros((self.batchsize, self.input_size, self.input_size, 3),
                               dtype="float32")

        with self.get_device_context(cfg.cpu()):
            self.model = VGGFace2(self.input_size, self.model_path, self.batchsize)
            self.model(placeholder)

    def process_input(self, batch: BatchType) -> None:
        """ Compile the detected faces for prediction """
        assert isinstance(batch, RecogBatch)
        batch.feed = np.array([T.cast(np.ndarray, feed.face)[..., :3]
                               for feed in batch.feed_faces],
                              dtype="float32") - self._average_img
        logger.trace("feed shape: %s", batch.feed.shape)  # type:ignore[attr-defined]

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
        with self.get_device_context(cfg.cpu()):
            retval = self.model(feed)
        assert isinstance(retval, np.ndarray)
        return retval

    def process_output(self, batch: BatchType) -> None:
        """ No output processing for  vgg_face2 """
        return


class ResNet50:
    """ ResNet50 imported for VGG-Face2 adapted from
    https://github.com/WeidiXie/Keras-VGGFace2-ResNet50

    Parameters
    ----------
    input_shape, Tuple[int, int, int] | None, optional
        The input shape for the model. Default: ``None``
    use_truncated: bool, optional
        ``True`` to use a truncated version of resnet. Default ``False``
    weight_decay: float
        L2 Regularizer weight decay. Default: 1e-4
    trainable: bool, optional
        ``True`` if the block should be trainable. Default: ``True``
    """
    def __init__(self,
                 input_shape: tuple[int, int, int] | None = None,
                 use_truncated: bool = False,
                 weight_decay: float = 1e-4,
                 trainable: bool = True) -> None:
        logger.debug("Initializing %s: input_shape: %s, use_truncated: %s, weight_decay: %s, "
                     "trainable: %s", self.__class__.__name__, input_shape, use_truncated,
                     weight_decay, trainable)

        self._input_shape = (None, None, 3) if input_shape is None else input_shape
        self._weight_decay = weight_decay
        self._trainable = trainable

        self._kernel_initializer = "orthogonal"
        self._use_bias = False
        self._bn_axis = 3
        self._block_suffix = {0: "_reduce", 1: "", 2: "_increase"}

        self._identity_calls = [2, 3, 5, 2]
        self._filters = [(64, 64, 256), (128, 128, 512), (256, 256, 1024), (512, 512, 2048)]
        if use_truncated:
            self._identity_calls = self._identity_calls[:-1]
            self._filters = self._filters[:-1]

        logger.debug("Initialized %s", self.__class__.__name__)

    def _identity_block(self,
                        inputs: KerasTensor,
                        kernel_size: int,
                        filters: tuple[int, int, int],
                        stage: int,
                        block: int) -> KerasTensor:
        """ The identity block is the block that has no conv layer at shortcut.

        Parameters
        ----------
        inputs: :class:`keras.KerasTensor`
            Input tensor
        kernel_size: int
            The kernel size of middle conv layer of the block
        filters: tuple[int, int, int[
            The filterss of 3 conv layers in the main path
        stage: int
            The current stage label, used for generating layer names
        block: int
            The current block label, used for generating layer names

        Returns
        -------
        :class:`keras.KerasTensor`
            Output tensor for the block
        """
        assert len(filters) == 3
        var_x = inputs

        for idx, filts in enumerate(filters):
            k_size = kernel_size if idx == 1 else 1
            conv_name = f"conv{stage}_{block}_{k_size}x{k_size}{self._block_suffix[idx]}"
            bn_name = f"{conv_name}_bn"

            var_x = Conv2D(filts,
                           k_size,
                           padding="same" if idx == 1 else "valid",
                           kernel_initializer=self._kernel_initializer,
                           use_bias=self._use_bias,
                           kernel_regularizer=L2(self._weight_decay),
                           trainable=self._trainable,
                           name=conv_name)(var_x)
            var_x = BatchNormalization(axis=self._bn_axis, name=bn_name)(var_x)
            if idx < 2:
                var_x = Activation("relu")(var_x)

        var_x = add([var_x, inputs])
        var_x = Activation("relu")(var_x)
        return var_x

    def _conv_block(self,
                    inputs: KerasTensor,
                    kernel_size: int,
                    filters: tuple[int, int, int],
                    stage: int,
                    block: int,
                    strides: tuple[int, int] = (2, 2)) -> KerasTensor:
        """ A block that has a conv layer at shortcut.

        Parameters
        ----------
        inputs: :class:`keras.KerasTensor`
            Input tensor
        kernel_size: int
            The kernel size of middle conv layer of the block
        filters: tuple[int, int, int[
            The filterss of 3 conv layers in the main path
        stage: int
            The current stage label, used for generating layer names
        block: int
            The current block label, used for generating layer names
        strides: tuple[int, int], optional
            The stride length for the first and last convolution. Default: (2, 2)

        Returns
        -------
        :class:`keras.KerasTensor`
            Output tensor for the block

        Notes
        -----
        From stage 3, the first conv layer at main path is with `strides = (2,2)` and the shortcut
        should have `strides = (2,2)` as well
        """
        assert len(filters) == 3
        var_x = inputs

        for idx, filts in enumerate(filters):
            k_size = kernel_size if idx == 1 else 1
            conv_name = f"conv{stage}_{block}_{k_size}x{k_size}{self._block_suffix[idx]}"
            bn_name = f"{conv_name}_bn"

            var_x = Conv2D(filts,
                           k_size,
                           strides=strides if idx == 0 else (1, 1),
                           padding="same" if idx == 1 else "valid",
                           kernel_initializer=self._kernel_initializer,
                           use_bias=self._use_bias,
                           kernel_regularizer=L2(self._weight_decay),
                           trainable=self._trainable,
                           name=conv_name)(var_x)
            var_x = BatchNormalization(axis=self._bn_axis, name=bn_name)(var_x)
            if idx < 2:
                var_x = Activation("relu")(var_x)

        conv_name = f"conv{stage}_{block}_1x1_proj"
        bn_name = f"{conv_name}_bn"

        shortcut = Conv2D(filters[-1],
                          (1, 1),
                          strides=strides,
                          kernel_initializer=self._kernel_initializer,
                          use_bias=self._use_bias,
                          kernel_regularizer=L2(self._weight_decay),
                          trainable=self._trainable,
                          name=conv_name)(inputs)
        shortcut = BatchNormalization(axis=self._bn_axis, name=bn_name)(shortcut)

        var_x = add([var_x, shortcut])
        var_x = Activation("relu")(var_x)
        return var_x

    def __call__(self, inputs: KerasTensor) -> KerasTensor:
        """ Call the resnet50 Network

        Parameters
        ----------
        inputs: :class:`keras.KerasTensor`
            Input tensor

        Returns
        -------
        :class::class:`keras.KerasTensor`
            Output tensor from resnet50
        """
        var_x = Conv2D(64,
                       (7, 7),
                       strides=(2, 2),
                       padding="same",
                       use_bias=self._use_bias,
                       kernel_initializer=self._kernel_initializer,
                       kernel_regularizer=L2(self._weight_decay),
                       trainable=self._trainable,
                       name="conv1_7x7_s2")(inputs)

        var_x = BatchNormalization(axis=self._bn_axis, name="conv1_7x7_s2_bn")(var_x)
        var_x = Activation("relu")(var_x)
        var_x = MaxPooling2D((3, 3), strides=(2, 2))(var_x)

        for idx, (recursuions, filters) in enumerate(zip(self._identity_calls, self._filters)):
            stage = idx + 2
            strides = (1, 1) if stage == 2 else (2, 2)
            var_x = self._conv_block(var_x, 3, filters, stage=stage, block=1, strides=strides)

            for recursion in range(recursuions):
                block = recursion + 2
                var_x = self._identity_block(var_x, 3, filters, stage=stage, block=block)

        return var_x


class VGGFace2():
    """ VGG-Face 2 model with resnet 50 backbone. Adapted from
    https://github.com/WeidiXie/Keras-VGGFace2-ResNet50

    Parameters
    ----------
    input_size, int
        The input size for the model.
    weights_path: str
        The path to the keras weights file
    batch_size: int
        The batch size to feed the model
    num_class: int, optional
        Number of classes to train the model on
    weight_decay: float
        L2 Regularizer weight decay. Default: 1e-4
    """
    def __init__(self,
                 input_size: int,
                 weights_path: str,
                 batch_size: int,
                 num_classes: int = 8631,
                 weight_decay: float = 1e-4) -> None:
        logger.debug(parse_class_init(locals()))
        self._input_shape = (input_size, input_size, 3)
        self._batch_size = batch_size
        self._weight_decay = weight_decay
        self._num_classes = num_classes
        self._resnet = ResNet50(input_shape=self._input_shape, weight_decay=self._weight_decay)
        self._model = self._load_model(weights_path)
        logger.debug("Initialized %s", self.__class__.__name__)

    def _load_model(self, weights_path: str) -> Model:
        """ load the vgg-face2 model

        Parameters
        ----------
        weights_path: str
            Full path to the model's weights

        Returns
        -------
        :class:`keras.models.Model`
            The VGG-Obstructed model
        """
        inputs = Input(self._input_shape)
        var_x = self._resnet(inputs)

        var_x = AveragePooling2D((7, 7), name="avg_pool")(var_x)
        var_x = Flatten()(var_x)
        var_x = Dense(512, activation="relu", name="dim_proj")(var_x)
        var_x = L2Normalize(axis=1)(var_x)

        retval = Model(inputs, var_x)
        retval.load_weights(weights_path)
        retval.make_predict_function()
        return retval

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        """ Get output from the vgg-face2 model

        Parameters
        ----------
        inputs: :class:`numpy.ndarray`
            The input to vgg-face2

        Returns
        -------
        :class:`numpy.ndarray`
            The output from vgg-face2
        """
        return self._model.predict(inputs, verbose=0, batch_size=self._batch_size)


class Cluster():
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
            logger.verbose("Using linkage method")  # type:ignore[attr-defined]
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


__all__ = get_module_objects(__name__)
