#!/usr/bin/env python3
""" Optimizers for faceswap.py """
# Naming convention inherited from Keras so ignore invalid names
# pylint:disable=invalid-name

import logging

import six
from keras.optimizers import Adam as KerasAdam
import tensorflow as tf
from tensorflow.python.framework import dtypes, ops
from tensorflow.python.keras import backend, initializers
from tensorflow.python.keras.engine import base_layer_utils
from tensorflow.python.ops import variables as tf_variables

from tensorflow.keras.optimizers import Optimizer as tf_optimizer

from lib.utils import get_backend

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

tf.config.optimizer.set_experimental_options({"pin_to_host_optimization": True})

# <<< START: MONKEY PATCH TENSORFLOW.KERAS OPTIMIZER >>> #
_OLD_INIT = tf_optimizer.__init__


def _patched_init(self, name, **kwargs):
    """ Extract `cpu_mode` from the given key word arguments and set to :attr:`_cpu_mode` and
    call the parent `__init__` with the remaining key word arguments.

    Parameters
    ----------
    name: str
        The name to use for accumulators created for the optimizer.
    **kwargs: dict
        keyword arguments. Allowed to be {`clipnorm`, `clipvalue`, `lr`, `decay`, `cpu_mode`}.
        `clipnorm` is clip gradients by norm; `clipvalue` is clip gradients by value, `decay` is
        included for backward compatibility to allow time inverse decay of learning rate. `lr` is
        included for backward compatibility, recommended to use `learning_rate` instead. `cpu_mode`
        is to perform some updates on CPU rather than GPU
    """
    # pylint:disable=protected-access
    logger.debug("initial kwargs: %s", kwargs)
    cpu_mode = False
    if "cpu_mode" in kwargs:
        cpu_mode = kwargs.pop("cpu_mode")
    self._cpu_mode = False if get_backend() != "nvidia" else cpu_mode
    logger.info("Optimizer CPU Mode set to %s", self._cpu_mode)
    _OLD_INIT(self, name, **kwargs)


def _patched_add_weight(self, name, shape,
                        dtype=None,
                        initializer="zeros",
                        trainable=None,
                        synchronization=tf_variables.VariableSynchronization.AUTO,
                        aggregation=tf_variables.VariableAggregation.NONE):
    """ Override of the original :func:`add_weight` method to allow explicit location of
    variables on to the CPU.

    See tensorflow documentation for details on this method
    """
    # pylint:disable=protected-access
    if dtype is None:
        dtype = dtypes.float32
    if isinstance(initializer, six.string_types) or callable(initializer):
        initializer = initializers.get(initializer)

    if synchronization == tf_variables.VariableSynchronization.ON_READ:
        if trainable:
            raise ValueError(
                "Synchronization value can be set to "
                "VariableSynchronization.ON_READ only for non-trainable variables. "
                "You have specified trainable=True and "
                "synchronization=VariableSynchronization.ON_READ.")
        # Set trainable to be false when variable is to be synced on read.
        trainable = False
    elif trainable is None:
        trainable = True

    if self._cpu_mode:
        with ops.device("/CPU:0"):
            variable = self._add_variable_with_custom_getter(
                name=name,
                shape=shape,
                getter=base_layer_utils.make_variable,
                overwrite=True,
                initializer=initializer,
                dtype=dtype,
                trainable=trainable,
                use_resource=True,
                synchronization=synchronization,
                aggregation=aggregation)
    else:
        variable = self._add_variable_with_custom_getter(
            name=name,
            shape=shape,
            getter=base_layer_utils.make_variable,
            overwrite=True,
            initializer=initializer,
            dtype=dtype,
            trainable=trainable,
            use_resource=True,
            synchronization=synchronization,
            aggregation=aggregation)
    backend.track_variable(variable)
    logger.info("Created variable for name '%s' on device '%s'", name, variable.device)
    return variable


tf_optimizer.__init__ = _patched_init
tf_optimizer.add_weight = _patched_add_weight

# <<< END: MONKEY PATCH TENSORFLOW.KERAS OPTIMIZER >>> #


class Adam(KerasAdam):
    """Adapted Keras Adam Optimizer to allow support of calculations on CPU for Tensorflow.

    Default parameters follow those provided in the original paper.

    Parameters
    ----------
    learning_rate: float, optional
        >= `0`. Learning rate. Default: `0.001`
    beta_1: float, optional
        `0` < beta < `1` Generally close to `1`. Default: `0.9`
    beta_2: float, optional
        `0` < beta < `1`. Generally close to `1`. Default: `0.999`
    clipnorm, bool, optional
        ``True`` if the gradient should be clipped if the L2 norm exceeds 1.0 otherwise ``False``.
        Default: ``False``
    cpu_mode: bool, optional
        Set to ``True`` to perform some of the calculations on CPU for Nvidia backends, otherwise
        ``False``. Default: ``False``
    kwargs: dict
        Any additional standard Keras optimizer keyword arguments

    References
    ----------
    - Adam - A Method for Stochastic Optimization - https://arxiv.org/abs/1412.6980v8

    - On the Convergence of Adam and Beyond - https://openreview.net/forum?id=ryQu7f-RZ
    """
    def __init__(self,
                 learning_rate=0.001,
                 beta_1=0.9,
                 beta_2=0.999,
                 clipnorm=False,
                 cpu_mode=False,
                 **kwargs):
        kwargs = self._rewrite_kwargs(learning_rate, clipnorm, cpu_mode, kwargs)
        super().__init__(beta_1=beta_1, beta_2=beta_2, **kwargs)

    @staticmethod
    def _rewrite_kwargs(learning_rate, clipnorm, cpu_mode, kwargs):
        """ Tensorflow Keras and Keras have diverged a little on key word argument naming
        so set the correct key word names for the backend.

        Also extracts the cpu_mode flag from the key word arguments and only passes it along
        if the backend is nvidia

        Notes
        -----
        Clip-norm is ballooning VRAM usage, which is not expected behavior and may be a bug in
        Keras/Tensorflow.

        PlaidML has a bug regarding the clip-norm parameter See:
        https://github.com/plaidml/plaidml/issues/228. We workaround by simply not adding this
        parameter for AMD backend users.
        """
        logger.info("Initial key word arguments: (learning_rate: %s. clipnorm: %s, kwargs: %s",
                    learning_rate, clipnorm, kwargs)
        if get_backend() == "amd":
            kwargs["lr"] = learning_rate
            # TODO add clipnorm in for plaidML when it is fixed in the main repository
        else:
            kwargs["learning_rate"] = learning_rate
            if clipnorm:
                kwargs["clipnorm"] = 1.0
        if get_backend() == "nvidia":
            kwargs["cpu_mode"] = cpu_mode
        logger.info("Returning key word arguments: %s", kwargs)
        return kwargs
