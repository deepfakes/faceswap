#!/usr/bin python3
""" Settings manager for Keras Backend """

import logging

import tensorflow as tf
from keras.models import load_model as k_load_model, Model

from lib.utils import get_backend

logger = logging.getLogger(__name__)  # pylint:disable=invalid-name


class KSession():
    """ Handles the settings of backend sessions.

    This class acts as a wrapper for various :class:`keras.Model()` functions, ensuring that
    actions performed on a model are handled consistently within the correct graph.

    Currently this only does anything for Nvidia users, making sure a unique graph and session is
    provided for the given model.

    Parameters
    ----------
    name: str
        The name of the model that is to be loaded
    model_path: str
        The path to the keras model file
    model_kwargs: dict
        Any kwargs that need to be passed to :func:`keras.models.load_models()`
    """
    def __init__(self, name, model_path, model_kwargs=None):
        logger.trace("Initializing: %s (name: %s, model_path: %s, model_kwargs: %s)",
                     self.__class__.__name__, name, model_path, model_kwargs)
        self._name = name
        self._session = self._set_session()
        self._model_path = model_path
        self._model_kwargs = model_kwargs
        self._model = None
        logger.trace("Initialized: %s", self.__class__.__name__,)

    def predict(self, feed):
        """ Get predictions from the model in the correct session.

        This method is a wrapper for :func:`keras.predict()` function.

        Parameters
        ----------
        feed: numpy.ndarray or list
            The feed to be provided to the model as input. This should be a ``numpy.ndarray``
            for single inputs or a ``list`` of ``numpy.ndarrays`` for multiple inputs.
        """
        if self._session is None:
            return self._model.predict(feed)

        with self._session.as_default():  # pylint: disable=not-context-manager
            with self._session.graph.as_default():
                return self._model.predict(feed)

    def _set_session(self):
        """ Sets the session and graph.

        If the backend is AMD then this does nothing and the global ``Keras`` ``Session``
        is used
        """
        if get_backend() == "amd":
            return None

        self.graph = tf.Graph()
        config = tf.ConfigProto()
        session = tf.Session(graph=tf.Graph(), config=config)
        logger.debug("Creating tf.session: (graph: %s, session: %s, config: %s)",
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
