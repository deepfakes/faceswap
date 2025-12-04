#!/usr/bin python3
""" Calculate Exponential Moving Average for faceswap GUI Stats. """

import logging

import numpy as np

from lib.logger import parse_class_init
from lib.utils import get_module_objects


logger = logging.getLogger(__name__)


class ExponentialMovingAverage:
    """ Reshapes data before calculating exponential moving average, then iterates once over the
    rows to calculate the offset without precision issues.

    Parameters
    ----------
    data : :class:`numpy.ndarray`
        A 1 dimensional numpy array to obtain smoothed data for
    amount : float
        in the range (0.0, 1.0) The alpha parameter (smoothing amount) for the moving average.

    Notes
    -----
    Adapted from: https://stackoverflow.com/questions/42869495
    """
    def __init__(self, data: np.ndarray, amount: float) -> None:
        logger.debug(parse_class_init(locals()))
        assert data.ndim == 1
        amount = min(max(amount, 0.001), 0.999)

        self._data = np.nan_to_num(data)
        self._alpha = 1. - amount
        self._dtype = "float32" if data.dtype == np.float32 else "float64"
        self._row_size = self._get_max_row_size()
        self._out = np.empty_like(data, dtype=self._dtype)
        logger.debug("Initialized %s", self.__class__.__name__)

    def __call__(self) -> np.ndarray:
        """ Perform the exponential moving average calculation.

        Returns
        -------
        :class:`numpy.ndarray`
            The smoothed data
        """
        if self._data.size <= self._row_size:
            self._ewma_vectorized(self._data, self._out)  # Normal function can handle this input
        else:
            self._ewma_vectorized_safe()  # Use the safe version
        return self._out

    def _get_max_row_size(self) -> int:
        """ Calculate the maximum row size for the running platform for the given dtype.

        Returns
        -------
        int
            The maximum row size possible on the running platform for the given :attr:`_dtype`

        Notes
        -----
        Might not be the optimal value for speed, which is hard to predict due to numpy
        optimizations.
        """
        # Use :func:`np.finfo(dtype).eps` if you are worried about accuracy and want to be safe.
        epsilon = np.finfo(self._dtype).tiny
        # If this produces an OverflowError, make epsilon larger:
        retval = int(np.log(epsilon) / np.log(1 - self._alpha)) + 1
        logger.debug("row_size: %s", retval)
        return retval

    def _ewma_vectorized_safe(self) -> None:
        """ Perform the vectorized exponential moving average in a safe way. """
        num_rows = int(self._data.size // self._row_size)  # the number of rows to use
        leftover = int(self._data.size % self._row_size)  # the amount of data leftover
        first_offset = self._data[0]

        if leftover > 0:
            # set temporary results to slice view of out parameter
            out_main_view = np.reshape(self._out[:-leftover], (num_rows, self._row_size))
            data_main_view = np.reshape(self._data[:-leftover], (num_rows, self._row_size))
        else:
            out_main_view = self._out.reshape(-1, self._row_size)
            data_main_view = self._data.reshape(-1, self._row_size)

        self._ewma_vectorized_2d(data_main_view, out_main_view)  # get the scaled cumulative sums

        scaling_factors = (1 - self._alpha) ** np.arange(1, self._row_size + 1)
        last_scaling_factor = scaling_factors[-1]

        # create offset array
        offsets = np.empty(out_main_view.shape[0], dtype=self._dtype)
        offsets[0] = first_offset
        # iteratively calculate offset for each row

        for i in range(1, out_main_view.shape[0]):
            offsets[i] = offsets[i - 1] * last_scaling_factor + out_main_view[i - 1, -1]

        # add the offsets to the result
        out_main_view += offsets[:, np.newaxis] * scaling_factors[np.newaxis, :]

        if leftover > 0:
            # process trailing data in the 2nd slice of the out parameter
            self._ewma_vectorized(self._data[-leftover:],
                                  self._out[-leftover:],
                                  offset=out_main_view[-1, -1])

    def _ewma_vectorized(self,
                         data: np.ndarray,
                         out: np.ndarray,
                         offset: float | None = None) -> None:
        """ Calculates the exponential moving average over a vector. Will fail for large inputs.

        The result is processed in place into the array passed to the `out` parameter

        Parameters
        ----------
        data : :class:`numpy.ndarray`
            A 1 dimensional numpy array to obtain smoothed data for
        out : :class:`numpy.ndarray`
            A location into which the result is stored. It must have the same shape and dtype as
            the input data
        offset : float, optional
            The offset for the moving average, scalar. Default: the value held in data[0].
        """
        if data.size < 1:  # empty input, return empty array
            return

        offset = data[0] if offset is None else offset

        # scaling_factors -> 0 as len(data) gets large. This leads to divide-by-zeros below
        scaling_factors = np.power(1. - self._alpha, np.arange(data.size + 1, dtype=self._dtype),
                                   dtype=self._dtype)
        # create cumulative sum array
        np.multiply(data, (self._alpha * scaling_factors[-2]) / scaling_factors[:-1],
                    dtype=self._dtype, out=out)
        np.cumsum(out, dtype=self._dtype, out=out)

        out /= scaling_factors[-2::-1]  # cumulative sums / scaling

        if offset != 0:
            noffset = np.asarray(offset).astype(self._dtype, copy=False)
            out += noffset * scaling_factors[1:]

    def _ewma_vectorized_2d(self, data: np.ndarray, out: np.ndarray) -> None:
        """ Calculates the exponential moving average over the last axis.

        The result is processed in place into the array passed to the `out` parameter

        Parameters
        ----------
        data : :class:`numpy.ndarray`
            A 1 or 2 dimensional numpy array to obtain smoothed data for.
        out : :class:`numpy.ndarray`
            A location into which the result is stored. It must have the same shape and dtype as
            the input data
        """
        if data.size < 1:  # empty input, return empty array
            return

        # calculate the moving average
        scaling_factors = np.power(1. - self._alpha, np.arange(data.shape[1] + 1,
                                                               dtype=self._dtype),
                                   dtype=self._dtype)
        # create a scaled cumulative sum array
        np.multiply(data,
                    np.multiply(self._alpha * scaling_factors[-2],
                                np.ones((data.shape[0], 1), dtype=self._dtype),
                                dtype=self._dtype) / scaling_factors[np.newaxis, :-1],
                    dtype=self._dtype, out=out)
        np.cumsum(out, axis=1, dtype=self._dtype, out=out)
        out /= scaling_factors[np.newaxis, -2::-1]


__all__ = get_module_objects(__name__)
