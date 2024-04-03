#!/usr/bin python3
""" Settings manager for Keras Backend """
from __future__ import annotations
from contextlib import nullcontext
import logging
import typing as T

import numpy as np
import tensorflow as tf

# Ignore linting errors from Tensorflow's thoroughly broken import system
from tensorflow.keras.layers import Activation  # pylint:disable=import-error
from tensorflow.keras.models import load_model as k_load_model, Model  # noqa:E501  # pylint:disable=import-error

from lib.utils import get_backend

if T.TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)


class KSession():
    """ Handles the settings of backend sessions for inference models.

    This class acts as a wrapper for various :class:`keras.Model()` functions, ensuring that
    actions performed on a model are handled consistently and can be performed in parallel in
    separate threads.

    This is an early implementation of this class, and should be expanded out over time.

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
    cpu_mode: bool, optional
        ``True`` run the model on CPU. Default: ``False``
    """
    def __init__(self,
                 name: str,
                 model_path: str,
                 model_kwargs: dict | None = None,
                 allow_growth: bool = False,
                 exclude_gpus: list[int] | None = None,
                 cpu_mode: bool = False) -> None:
        logger.trace("Initializing: %s (name: %s, model_path: %s, "  # type:ignore
                     "model_kwargs: %s,  allow_growth: %s, exclude_gpus: %s, cpu_mode: %s)",
                     self.__class__.__name__, name, model_path, model_kwargs, allow_growth,
                     exclude_gpus, cpu_mode)
        self._name = name
        self._backend = get_backend()
        self._context = self._set_session(allow_growth,
                                          [] if exclude_gpus is None else exclude_gpus,
                                          cpu_mode)
        self._model_path = model_path
        self._model_kwargs = {} if not model_kwargs else model_kwargs
        self._model: Model | None = None
        logger.trace("Initialized: %s", self.__class__.__name__,)  # type:ignore

    def predict(self,
                feed: list[np.ndarray] | np.ndarray,
                batch_size: int | None = None) -> list[np.ndarray] | np.ndarray:
        """ Get predictions from the model.

        This method is a wrapper for :func:`keras.predict()` function. For Tensorflow backends
        this is a straight call to the predict function.

        Parameters
        ----------
        feed: numpy.ndarray or list
            The feed to be provided to the model as input. This should be a :class:`numpy.ndarray`
            for single inputs or a `list` of :class:`numpy.ndarray` objects for multiple inputs.
        batchsize: int, optional
            The batch size to run prediction at. Default ``None``

        Returns
        -------
        :class:`numpy.ndarray`
            The predictions from the model
        """
        assert self._model is not None
        with self._context:
            return self._model.predict(feed, verbose=0, batch_size=batch_size)

    def _set_session(self,
                     allow_growth: bool,
                     exclude_gpus: list,
                     cpu_mode: bool) -> T.ContextManager:
        """ Sets the backend session options.

        For CPU backends, this hides any GPUs from Tensorflow.

        For Nvidia backends, this hides any GPUs that Tensorflow should not use and applies
        any allow growth settings

        Parameters
        ----------
        allow_growth: bool
            Enable the Tensorflow GPU allow_growth configuration option. This option prevents
            Tensorflow from allocating all of the GPU VRAM, but can lead to higher fragmentation
            and slower performance
        exclude_gpus: list
            A list of indices correlating to connected GPUs that Tensorflow should not use. Pass
            ``None`` to not exclude any GPUs
        cpu_mode: bool
            ``True`` run the model on CPU. Default: ``False``
        """
        retval = nullcontext()
        if self._backend == "cpu":
            logger.verbose("Hiding GPUs from Tensorflow")  # type:ignore
            tf.config.set_visible_devices([], "GPU")
            return retval

        gpus = tf.config.list_physical_devices('GPU')
        if exclude_gpus:
            gpus = [gpu for idx, gpu in enumerate(gpus) if idx not in exclude_gpus]
            logger.debug("Filtering devices to: %s", gpus)
            tf.config.set_visible_devices(gpus, "GPU")

        if allow_growth and self._backend == "nvidia":
            for gpu in gpus:
                logger.info("Setting allow growth for GPU: %s", gpu)
                tf.config.experimental.set_memory_growth(gpu, True)

        if cpu_mode:
            retval = tf.device("/device:cpu:0")
        return retval

    def load_model(self) -> None:
        """ Loads a model.

        This method is a wrapper for :func:`keras.models.load_model()`. Loads a model and its
        weights from :attr:`model_path` defined during initialization of this class. Any additional
        ``kwargs`` to be passed to :func:`keras.models.load_model()` should also be defined during
        initialization of the class.

        For Tensorflow backends, the `make_predict_function` method is called on the model to make
        it thread safe.
        """
        logger.verbose("Initializing plugin model: %s", self._name)  # type:ignore
        with self._context:
            self._model = k_load_model(self._model_path, compile=False, **self._model_kwargs)
            self._model.make_predict_function()

    def define_model(self, function: Callable) -> None:
        """ Defines a model from the given function.

        This method acts as a wrapper for :class:`keras.models.Model()`.

        Parameters
        ----------
        function: function
            A function that defines a :class:`keras.Model` and returns it's ``inputs`` and
            ``outputs``. The function that generates these results should be passed in, NOT the
            results themselves, as the function needs to be executed within the correct context.
        """
        with self._context:
            self._model = Model(*function())

    def load_model_weights(self) -> None:
        """ Load model weights for a defined model inside the correct session.

        This method is a wrapper for :class:`keras.load_weights()`. Once a model has been defined
        in :func:`define_model()` this method can be called to load its weights from the
        :attr:`model_path` defined during initialization of this class.

        For Tensorflow backends, the `make_predict_function` method is called on the model to make
        it thread safe.
        """
        logger.verbose("Initializing plugin model: %s", self._name)  # type:ignore
        assert self._model is not None
        with self._context:
            self._model.load_weights(self._model_path)
            self._model.make_predict_function()

    def append_softmax_activation(self, layer_index: int = -1) -> None:
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
        assert self._model is not None
        with self._context:
            softmax = Activation("softmax", name="softmax")(self._model.layers[layer_index].output)
            self._model = Model(inputs=self._model.input, outputs=[softmax])
