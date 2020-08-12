#!/usr/bin python3
""" Settings manager for Keras Backend """

import logging

import numpy as np
import tensorflow as tf
# pylint:disable=no-name-in-module,import-error
from keras.layers import Activation
from keras.models import load_model as k_load_model, Model

from lib.utils import get_backend

logger = logging.getLogger(__name__)  # pylint:disable=invalid-name


class KSession():
    """ Handles the settings of backend sessions for inference models.

    This class acts as a wrapper for various :class:`keras.Model()` functions, ensuring that
    actions performed on a model are handled consistently and can be performed in parallel in
    separate threads.

    This is an early implementation of this class, and should be expanded out over time
    with relevant `AMD`, `CPU` and `NVIDIA` backend methods.

    Notes
    -----
    The documentation refers to :mod:`keras`. This is a pseudonym for either :mod:`keras` or
    :mod:`tensorflow.keras` depending on the backend in use.

    Parameters
    ----------
    name: str
        The name of the model that is to be loaded
    model_path: str
        The path to the keras model file
    model_kwargs: dict, optional
        Any kwargs that need to be passed to :func:`keras.models.load_models()`. Default: ``None``
    allow_growth: bool, optional
        Enable the Tensorflow GPU allow_growth configuration option. This option prevents
        Tensorflow from allocating all of the GPU VRAM, but can lead to higher fragmentation and
        slower performance. Default: ``False``
    exclude_gpus: list, optional
        A list of indices correlating to connected GPUs that Tensorflow should not use. Pass
        ``None`` to not exclude any GPUs. Default: ``None``

    """
    def __init__(self, name, model_path, model_kwargs=None, allow_growth=False, exclude_gpus=None):
        logger.trace("Initializing: %s (name: %s, model_path: %s, model_kwargs: %s, "
                     "allow_growth: %s, exclude_gpus)", self.__class__.__name__, name, model_path,
                     model_kwargs, allow_growth, exclude_gpus)
        self._name = name
        self._backend = get_backend()
        self._set_session(allow_growth, exclude_gpus)
        self._model_path = model_path
        self._model_kwargs = dict() if not model_kwargs else model_kwargs
        self._model = None
        logger.trace("Initialized: %s", self.__class__.__name__,)

    def predict(self, feed, batch_size=None):
        """ Get predictions from the model.

        This method is a wrapper for :func:`keras.predict()` function. For Tensorflow backends
        this is a straight call to the predict function. For PlaidML backends, this attempts
        to optimize the inference batch sizes to reduce the number of kernels that need to be
        compiled.

        Parameters
        ----------
        feed: numpy.ndarray or list
            The feed to be provided to the model as input. This should be a :class:`numpy.ndarray`
            for single inputs or a `list` of :class:`numpy.ndarray` objects for multiple inputs.
        """
        if self._backend == "amd" and batch_size is not None:
            return self._amd_predict_with_optimized_batchsizes(feed, batch_size)
        return self._model.predict(feed, batch_size=batch_size)

    def _amd_predict_with_optimized_batchsizes(self, feed, batch_size):
        """ Minimizes the amount of kernels to be compiled when using the ``amd`` backend with
        varying batch sizes while trying to keep the batchsize as high as possible.

        Parameters
        ----------
        feed: numpy.ndarray or list
            The feed to be provided to the model as input. This should be a ``numpy.ndarray``
            for single inputs or a ``list`` of ``numpy.ndarray`` objects for multiple inputs.
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

    def _set_session(self, allow_growth, exclude_gpus):
        """ Sets the backend session options.

        For AMD backend this does nothing.

        For CPU backends, this hides any GPUs from Tensorflow.

        For Nvidia backends, this hides any GPUs that Tensorflow should not use and applies
        any allow growth settings

        Parameters
        ----------
        allow_growth: bool, optional
            Enable the Tensorflow GPU allow_growth configuration option. This option prevents
            Tensorflow from allocating all of the GPU VRAM, but can lead to higher fragmentation
            and slower performance. Default: False
        exclude_gpus: list, optional
            A list of indices correlating to connected GPUs that Tensorflow should not use. Pass
            ``None`` to not exclude any GPUs. Default: ``None``
        """
        if self._backend == "amd":
            return
        if self._backend == "cpu":
            logger.verbose("Hiding GPUs from Tensorflow")
            tf.config.set_visible_devices([], "GPU")
            return

        gpus = tf.config.list_physical_devices('GPU')
        if exclude_gpus:
            gpus = [gpu for idx, gpu in enumerate(gpus) if idx not in exclude_gpus]
            logger.debug("Filtering devices to: %s", gpus)
            tf.config.set_visible_devices(gpus, "GPU")

        if allow_growth:
            for gpu in gpus:
                logger.info("Setting allow growth for GPU: %s", gpu)
                tf.config.experimental.set_memory_growth(gpu, True)

    def load_model(self):
        """ Loads a model.

        This method is a wrapper for :func:`keras.models.load_model()`. Loads a model and its
        weights from :attr:`model_path` defined during initialization of this class. Any additional
        ``kwargs`` to be passed to :func:`keras.models.load_model()` should also be defined during
        initialization of the class.

        For Tensorflow backends, the `make_predict_function` method is called on the model to make
        it thread safe.
        """
        logger.verbose("Initializing plugin model: %s", self._name)
        self._model = k_load_model(self._model_path, compile=False, **self._model_kwargs)
        if self._backend != "amd":
            self._model.make_predict_function()

    def define_model(self, function):
        """ Defines a model from the given function.

        This method acts as a wrapper for :class:`keras.models.Model()`.

        Parameters
        ----------
        function: function
            A function that defines a :class:`keras.Model` and returns it's ``inputs`` and
            ``outputs``. The function that generates these results should be passed in, NOT the
            results themselves, as the function needs to be executed within the correct context.
        """
        self._model = Model(*function())

    def load_model_weights(self):
        """ Load model weights for a defined model inside the correct session.

        This method is a wrapper for :class:`keras.load_weights()`. Once a model has been defined
        in :func:`define_model()` this method can be called to load its weights from the
        :attr:`model_path` defined during initialization of this class.

        For Tensorflow backends, the `make_predict_function` method is called on the model to make
        it thread safe.
        """
        logger.verbose("Initializing plugin model: %s", self._name)
        self._model.load_weights(self._model_path)
        if self._backend != "amd":
            self._model.make_predict_function()

    def append_softmax_activation(self, layer_index=-1):
        """ Append a softmax activation layer to a model

        Occasionally a softmax activation layer needs to be added to a model's output.
        This is a convenience function to append this layer to the loaded model.

        Parameters
        ----------
        layer_index: int, optional
            The layer index of the model to select the output from to use as an input to the
            softmax activation layer. Default: `-1` (The final layer of the model)
        """
        logger.debug("Appending Softmax Activation to model: (layer_index: %s)", layer_index)
        softmax = Activation("softmax", name="softmax")(self._model.layers[layer_index].output)
        self._model = Model(inputs=self._model.input, outputs=[softmax])
