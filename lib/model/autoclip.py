""" Auto clipper for clipping gradients. """
import numpy as np
import tensorflow as tf


class AutoClipper():
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
        self._clip_percentile = tf.cast(clip_percentile, tf.float64)
        self._grad_history = tf.Variable(tf.zeros(history_size), trainable=False)
        self._index = tf.Variable(0, trainable=False)
        self._history_size = history_size

    def _percentile(self, grad_history: tf.Tensor) -> tf.Tensor:
        """ Compute the clip percentile of the gradient history

        Parameters
        ----------
        grad_history: :class:`tensorflow.Tensor`
            Tge gradient history to calculate the clip percentile for

        Returns
        -------
        :class:`tensorflow.Tensor`
            A rank(:attr:`clip_percentile`) `Tensor`

        Notes
        -----
        Adapted from
        https://github.com/tensorflow/probability/blob/r0.14/tensorflow_probability/python/stats/quantiles.py
        to remove reliance on full tensorflow_probability libraray
        """
        with tf.name_scope("percentile"):
            frac_at_q_or_below = self._clip_percentile / 100.
            sorted_hist = tf.sort(grad_history, axis=-1, direction="ASCENDING")

            num = tf.cast(tf.shape(grad_history)[-1], tf.float64)

            # get indices
            indices = tf.round((num - 1) * frac_at_q_or_below)
            indices = tf.clip_by_value(tf.cast(indices, tf.int32),
                                       0,
                                       tf.shape(grad_history)[-1] - 1)
            gathered_hist = tf.gather(sorted_hist, indices, axis=-1)

            # Propagate NaNs. Apparently tf.is_nan doesn't like other dtypes
            nan_batch_members = tf.reduce_any(tf.math.is_nan(grad_history), axis=None)
            right_rank_matched_shape = tf.pad(tf.shape(nan_batch_members),
                                              paddings=[[0, tf.rank(self._clip_percentile)]],
                                              constant_values=1)
            nan_batch_members = tf.reshape(nan_batch_members, shape=right_rank_matched_shape)

            nan = np.array(np.nan, gathered_hist.dtype.as_numpy_dtype)
            gathered_hist = tf.where(nan_batch_members, nan, gathered_hist)

            return gathered_hist

    def __call__(self, grads_and_vars: list[tf.Tensor]) -> list[tf.Tensor]:
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
        clip_value = self._percentile(self._grad_history[: self._index])
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
