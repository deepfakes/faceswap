""" Auto clipper for clipping gradients.

Non AMD Only
"""
from typing import List

import tensorflow as tf
import tensorflow_probability as tfp


class AutoClipper():  # pylint:disable=too-few-public-methods
    """ AutoClip: Adaptive Gradient Clipping for Source Separation Networks

    Parameters
    ----------
    clip_percentile: int
        The percentile to clip the gradients at
    history_size: int, optional
        The number of iterations of data to use to calculate the norm
    Default: ``10000``

    References
    ----------
    tf implementation: https://github.com/pseeth/autoclip
    original paper: https://arxiv.org/abs/2007.14469
    """
    def __init__(self, clip_percentile: int, history_size: int = 10000):
        self._clip_percentile = clip_percentile
        self._grad_history = tf.Variable(tf.zeros(history_size), trainable=False)
        self._index = tf.Variable(0, trainable=False)
        self._history_size = history_size

    def __call__(self, grads_and_vars: List[tf.Tensor]) -> List[tf.Tensor]:
        """ Call the AutoClip function.

        Parameters
        ----------
        grads_and_vars: list
            The list of gradient tensors and variables for the optimizer
        """
        grad_norms = [self._get_grad_norm(g) for g, _ in grads_and_vars]
        total_norm = tf.norm(grad_norms)
        assign_idx = tf.math.mod(self._index, self._history_size)
        self._grad_history = self._grad_history[assign_idx].assign(total_norm)
        self._index = self._index.assign_add(1)
        clip_value = tfp.stats.percentile(self._grad_history[: self._index],
                                          q=self._clip_percentile)
        return [(tf.clip_by_norm(g, clip_value), v) for g, v in grads_and_vars]

    @classmethod
    def _get_grad_norm(cls, gradients: tf.Tensor) -> tf.Tensor:
        """ Obtain the L2 Norm for the gradients

        Parameters
        ----------
        gradients: :class:`tensorflow.Tensor`
            The gradients to calculate the L2 norm for

        Returns
        -------
        :class:`tensorflow.Tensor`
            The L2 Norm of the given gradients
        """
        values = tf.convert_to_tensor(gradients.values
                                      if isinstance(gradients, tf.IndexedSlices)
                                      else gradients, name="t")

        # Calculate L2-norm, clip elements by ratio of clip_norm to L2-norm
        l2sum = tf.math.reduce_sum(values * values, axis=None, keepdims=True)
        pred = l2sum > 0
        # Two-tap tf.where trick to bypass NaN gradients
        l2sum_safe = tf.where(pred, l2sum, tf.ones_like(l2sum))
        return tf.squeeze(tf.where(pred, tf.math.sqrt(l2sum_safe), l2sum))
