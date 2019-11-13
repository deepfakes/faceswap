#!/usr/bin python3
""" Settings manager for Keras Backend """

import logging

import tensorflow as tf
from keras.layers import Activation
from tensorflow.python import errors_impl as tf_error  # pylint:disable=no-name-in-module
from keras.models import load_model as k_load_model, Model
import numpy as np

from lib.utils import get_backend, FaceswapError

logger = logging.getLogger(__name__)  # pylint:disable=invalid-name


class KSession():
    """ Handles the settings of backend sessions.

    This class acts as a wrapper for various :class:`keras.Model()` functions, ensuring that
    actions performed on a model are handled consistently within the correct graph.

    This is an early implementation of this class, and should be expanded out over time
    with relevant `AMD`, `CPU` and `NVIDIA` backend methods.

    Parameters
    ----------
    name: str
        The name of the model that is to be loaded
    model_path: str
        The path to the keras model file
    model_kwargs: dict, optional
        Any kwargs that need to be passed to :func:`keras.models.load_models()`. Default: None
    allow_growth: bool, optional
        Enable the Tensorflow GPU allow_growth configuration option. This option prevents "
        Tensorflow from allocating all of the GPU VRAM, but can lead to higher fragmentation and "
        slower performance. Default: False
    """
    def __init__(self, name, model_path, model_kwargs=None, allow_growth=False):
        logger.trace("Initializing: %s (name: %s, model_path: %s, model_kwargs: %s, "
                     "allow_growth: %s)",
                     self.__class__.__name__, name, model_path, model_kwargs, allow_growth)
        self._name = name
        self._session = self._set_session(allow_growth)
        self._model_path = model_path
        self._model_kwargs = model_kwargs
        self._model = None
        logger.trace("Initialized: %s", self.__class__.__name__,)

    def predict(self, feed, batch_size=None):
        """ Get predictions from the model in the correct session.

        This method is a wrapper for :func:`keras.predict()` function.

        Parameters
        ----------
        feed: numpy.ndarray or list
            The feed to be provided to the model as input. This should be a ``numpy.ndarray``
            for single inputs or a ``list`` of ``numpy.ndarrays`` for multiple inputs.
        """
        if self._session is None:
            if batch_size is None:
                return self._model.predict(feed)
            return self._amd_predict_with_optimized_batchsizes(feed, batch_size)

        with self._session.as_default():  # pylint: disable=not-context-manager
            with self._session.graph.as_default():
                return self._model.predict(feed, batch_size=batch_size)

    def _amd_predict_with_optimized_batchsizes(self, feed, batch_size):
        """ Minimizes the amount of kernels to be compiled when using
        the ``Amd`` backend with varying batchsizes while trying to keep
        the batchsize as high as possible.

        Parameters
        ----------
        feed: numpy.ndarray or list
            The feed to be provided to the model as input. This should be a ``numpy.ndarray``
            for single inputs or a ``list`` of ``numpy.ndarrays`` for multiple inputs.
        batch_size: int
            The upper batchsize to use.
        """
        if isinstance(feed, np.ndarray):
            feed = [feed]
        items = feed[0].shape[0]
        done_items = 0
        results = list()
        while done_items < items:
            if batch_size < 4:  # Not much difference in BS < 4
                batch_size = 1
            batch_items = ((items - done_items) // batch_size) * batch_size
            if batch_items:
                pred_data = [x[done_items:done_items + batch_items] for x in feed]
                pred = self._model.predict(pred_data, batch_size=batch_size)
                done_items += batch_items
                results.append(pred)
            batch_size //= 2
        if isinstance(results[0], np.ndarray):
            return np.concatenate(results)
        return [np.concatenate(x) for x in zip(*results)]

    def _set_session(self, allow_growth):
        """ Sets the session and graph.

        If the backend is AMD then this does nothing and the global ``Keras`` ``Session``
        is used
        """
        if get_backend() == "amd":
            return None

        self.graph = tf.Graph()
        config = tf.ConfigProto()
        if allow_growth and get_backend() == "nvidia":
            config.gpu_options.allow_growth = True
        try:
            session = tf.Session(graph=tf.Graph(), config=config)
        except tf_error.InternalError as err:
            if "driver version is insufficient" in str(err):
                msg = ("Your Nvidia Graphics Driver is insufficient for running Faceswap. "
                       "Please upgrade to the latest version.")
                raise FaceswapError(msg) from err
            raise err
        logger.debug("Created tf.session: (graph: %s, session: %s, config: %s)",
                     session.graph, session, config)
        return session

    def load_model(self):
        """ Loads a model within the correct session.

        This method is a wrapper for :func:`keras.models.load_model()`. Loads a model and its
        weights from :attr:`model_path`. Any additional ``kwargs`` to be passed to
        :func:`keras.models.load_model()` should also be defined during initialization of the
        class.
        """
        logger.verbose("Initializing plugin model: %s", self._name)
        if self._session is None:
            self._model = k_load_model(self._model_path, **self._model_kwargs)
        else:
            with self._session.as_default():  # pylint: disable=not-context-manager
                with self._session.graph.as_default():
                    self._model = k_load_model(self._model_path, **self._model_kwargs)

    def define_model(self, function):
        """ Defines a given model in the correct session.

        This method acts as a wrapper for :class:`keras.models.Model()` to ensure that the model
        is defined within it's own graph.

        Parameters
        ----------
        function: function
            A function that defines a :class:`keras.Model` and returns it's ``inputs`` and
            ``outputs``. The function that generates these results should be passed in, NOT the
            results themselves, as the function needs to be executed within the correct context.
        """
        if self._session is None:
            self._model = Model(*function())
        else:
            with self._session.as_default():  # pylint: disable=not-context-manager
                with self._session.graph.as_default():
                    self._model = Model(*function())

    def load_model_weights(self):
        """ Load model weights for a defined model inside the correct session.

        This method is a wrapper for :class:`keras.load_weights()`. Once a model has been defined
        in :func:`define_model()` this method can be called to load its weights in the correct
        graph from the :attr:`model_path` defined during initialization of this class.
        """
        logger.verbose("Initializing plugin model: %s", self._name)
        if self._session is None:
            self._model.load_weights(self._model_path)
        else:
            with self._session.as_default():  # pylint: disable=not-context-manager
                with self._session.graph.as_default():
                    self._model.load_weights(self._model_path)

    def append_softmax_activation(self, layer_index=-1):
        """ Append a softmax activation layer to a model

        Occasionally a softmax activation layer needs to be added to a model's output.
        This is a convenience fuction to append this layer to the loaded model.

        Parameters
        ----------
        layer_index: int, optional
            The layer index of the model to select the output from to use as an input to the
            softmax activation layer. Default: -1 (The final layer of the model)
        """
        logger.debug("Appending Softmax Activation to model: (layer_index: %s)", layer_index)
        softmax = Activation("softmax", name="softmax")(self._model.layers[layer_index].output)
        self._model = Model(inputs=self._model.input, outputs=[softmax])
