#!/usr/bin/env python3
""" Optimizers for faceswap.py """
# Naming convention inherited from Keras so ignore invalid names
# pylint:disable=invalid-name

import logging

from keras.optimizers import Adam as KerasAdam
from tensorflow.python.framework import ops
from tensorflow.python.ops import variables as tf_variables

from lib.utils import get_backend

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


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
        kwargs = self._rewrite_kwargs(learning_rate, clipnorm, kwargs)
        super().__init__(beta_1=beta_1, beta_2=beta_2, **kwargs)
        self._cpu_mode = self._set_cpu_mode(cpu_mode)

    @staticmethod
    def _rewrite_kwargs(learning_rate, clipnorm, kwargs):
        """ Tensorflow Keras and Keras have diverged a little on key word argument naming
        so set the correct key word names for the backend.

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
        logger.info("Returning key word arguments: %s", kwargs)
        return kwargs

    @staticmethod
    def _set_cpu_mode(cpu_mode):
        """ Sets the CPU mode to False if not using Nvidia backend, otherwise the given value.

        Parameters
        ----------
        cpu_mode: bool
            Set to ``True`` to perform some of the calculations on CPU for Nvidia backends,
            otherwise ``False``.

        Returns
        -------
        bool
            ``True`` if some calculations should be performed on CPU otherwise ``False``
        """
        retval = False if get_backend() != "nvidia" else cpu_mode
        logger.info("Optimizer CPU Mode set to %s", retval)
        return retval

    def _create_hypers(self):
        """ Override the default tensorflow keras optimizer to allow placement on the CPU.

        NB: Default keras does not have this method, so this should not run if running on plaidML.
        Assertion is placed here in case this changes in future
        """
        assert get_backend() != "amd", "Keras backend has changed for plaidML"
        if self._hypers_created:
            return
        # Iterate hyper values deterministically.
        for name, value in sorted(self._hyper.items()):
            if isinstance(value, (ops.Tensor, tf_variables.Variable)) or callable(value):
                continue

            if self._cpu_mode and get_backend() == "nvidia":
                ops.device.__enter__()
            self._hyper[name] = self.add_weight(
                name,
                shape=[],
                trainable=False,
                initializer=value,
                aggregation=tf_variables.VariableAggregation.ONLY_FIRST_REPLICA)
            if self._cpu_mode and get_backend() == "nvidia":
                ops.device.__exit__(None, None, None)
        for key, value in self._hyper.items():
            logger.info("Created variable for name '%s' on device '%s'", key, value.device)
        self._hypers_created = True
