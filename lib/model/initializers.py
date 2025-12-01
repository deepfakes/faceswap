#!/usr/bin/env python3
""" Custom Initializers for faceswap.py """
from __future__ import annotations

import logging
import sys
import inspect
import typing as T

from keras import backend as K, initializers, ops
from keras import saving, Variable
from keras.src.initializers.random_initializers import compute_fans

import numpy as np

from lib.logger import parse_class_init
from lib.utils import get_module_objects

if T.TYPE_CHECKING:
    from keras import KerasTensor

logger = logging.getLogger(__name__)


class ICNR(initializers.Initializer):
    """ ICNR initializer for checkerboard artifact free sub pixel convolution

    Parameters
    ----------
    initializer: :class:`keras.initializers.Initializer`
        The initializer used for sub kernels (orthogonal, glorot uniform, etc.)
    scale: int, optional
        scaling factor of sub pixel convolution (up sampling from 8x8 to 16x16 is scale 2).
        Default: `2`

    Returns
    -------
    :class:`keras.KerasTensor`
        The modified kernel weights

    Example
    -------
    >>> x = conv2d(... weights_initializer=ICNR(initializer=he_uniform(), scale=2))

    References
    ----------
    Andrew Aitken et al. Checkerboard artifact free sub-pixel convolution
    https://arxiv.org/pdf/1707.02937.pdf,  https://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self,
                 initializer: dict[str, T.Any] | initializers.Initializer,
                 scale: int = 2) -> None:
        logger.debug(parse_class_init(locals()))

        self._scale = scale
        self._initializer = initializer

        logger.debug("Initialized %s", self.__class__.__name__)

    def __call__(self,
                 shape: list[int] | tuple[int, ...],
                 dtype: str | None = "float32") -> KerasTensor:
        """ Call function for the ICNR initializer.

        Parameters
        ----------
        shape: list[int] | tuple[int, ...]
            The required resized shape for the output tensor
        dtype: str
            The data type for the tensor
        kwargs: dict[str, Any]
            Standard keras initializer keyword arguments

        Returns
        -------
        :class:`keras.KerasTensor`
            The modified kernel weights
        """
        shape = list(shape)

        if self._scale == 1:
            if isinstance(self._initializer, dict):
                return next(i for i in self._initializer.values())
            return self._initializer(shape)

        new_shape = shape[:3] + [shape[3] // (self._scale ** 2)]
        size = [s * self._scale for s in new_shape[:2]]

        if isinstance(self._initializer, dict):
            self._initializer = initializers.deserialize(self._initializer)

        var_x = self._initializer(new_shape, dtype)
        var_x = ops.transpose(var_x, [2, 0, 1, 3])
        var_x = ops.image.resize(var_x,
                                 size,
                                 interpolation="nearest",
                                 data_format="channels_last")
        var_x = self._space_to_depth(T.cast("KerasTensor", var_x))
        var_x = ops.transpose(var_x, [1, 2, 0, 3])

        logger.debug("ICNR Output shape: %s", var_x.shape)
        return T.cast("KerasTensor", var_x)

    def _space_to_depth(self, input_tensor: KerasTensor) -> KerasTensor:
        """ Space to depth Keras implementation.

        Parameters
        ----------
        input_tensor: :class:`keras.KerasTensor`
            The tensor to be manipulated

        Returns
        -------
        :class:`keras.KerasTensor`
            The manipulated input tensor
        """
        batch, height, width, depth = input_tensor.shape
        assert height is not None and width is not None
        new_height, new_width = height // 2, width // 2
        inter_shape = (batch, new_height, self._scale, new_width, self._scale, depth)

        var_x = ops.reshape(input_tensor, inter_shape)
        var_x = ops.transpose(var_x, (0, 1, 3, 2, 4, 5))
        retval = ops.reshape(var_x, (batch, new_height, new_width, -1))

        logger.debug("Space to depth - Input shape: %s, Output shape: %s",
                     input_tensor.shape, retval.shape)
        return T.cast("KerasTensor", retval)

    def get_config(self) -> dict[str, T.Any]:
        """ Return the ICNR Initializer configuration.

        Returns
        -------
        dict[str, Any]
            The configuration for ICNR Initialization
        """
        config = {"scale": self._scale, "initializer": self._initializer}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ConvolutionAware(initializers.Initializer):
    """
    Initializer that generates orthogonal convolution filters in the Fourier space. If this
    initializer is passed a shape that is not 3D or 4D, orthogonal initialization will be used.

    Adapted, fixed and optimized from:
    https://github.com/keras-team/keras-contrib/blob/master/keras_contrib/initializers/convaware.py

    Parameters
    ----------
    eps_std: float, optional
        The Standard deviation for the random normal noise used to break symmetry in the inverse
        Fourier transform. Default: 0.05
    seed: int | None, optional
        Used to seed the random generator. Default: ``None``
    initialized: bool, optional
        This should always be set to ``False``. To avoid Keras re-calculating the values every time
        the model is loaded, this parameter is internally set on first time initialization.
        Default:``False``

    Returns
    -------
    :class:`keras.Variable`
        The modified kernel weights

    References
    ----------
    Armen Aghajanyan, https://arxiv.org/abs/1702.06295
    """

    def __init__(self,
                 eps_std: float = 0.05,
                 seed: int | None = None,
                 initialized: bool = False) -> None:
        logger.debug(parse_class_init(locals()))

        self._eps_std = eps_std
        self._seed = seed
        self._orthogonal = initializers.OrthogonalInitializer()
        self._he_uniform = initializers.HeUniform()
        self._initialized = initialized

        logger.debug("Initialized %s", self.__class__.__name__)

    @classmethod
    def _symmetrize(cls, inputs: np.ndarray) -> np.ndarray:
        """ Make the given tensor symmetrical.

        Parameters
        ----------
        inputs: :class:`numpy.ndarray`
            The input tensor to make symmetrical

        Returns
        -------
        :class:`numpy.ndarray`
            The symmetrical output
        """
        var_a = np.transpose(inputs, axes=(0, 1, 3, 2))
        diag = var_a.diagonal(axis1=2, axis2=3)
        var_b = np.array([[np.diag(arr) for arr in batch] for batch in diag])
        retval = inputs + var_a - var_b
        logger.debug("Input shape: %s. Output shape: %s", inputs.shape, retval.shape)
        return retval

    def _create_basis(self, filters_size: int, filters: int, size: int, dtype: str) -> np.ndarray:
        """ Create the basis for convolutional aware initialization

        Parameters
        ----------
        filters_size: int
            The size of the filter
        filters: int
            The number of filters
        dtype: str
            The data type

        Returns
        -------
        :class:`numpy.ndarray`
            The output array
        """
        if size == 1:
            return np.random.normal(0.0, self._eps_std, (filters_size, filters, size))
        nbb = filters // size + 1
        var_a = np.random.normal(0.0, 1.0, (filters_size, nbb, size, size))
        var_a = self._symmetrize(var_a)
        var_u = np.linalg.svd(var_a)[0].transpose(0, 1, 3, 2)
        retval = np.reshape(var_u, (filters_size, nbb * size, size))[:, :filters, :].astype(dtype)
        logger.debug("filters_size: %s, filters: %s, size: %s, dtype: %s, output: %s",
                     filters_size, filters, size, dtype, retval.shape)
        return retval

    @classmethod
    def _scale_filters(cls, filters: np.ndarray, variance: float) -> np.ndarray:
        """ Scale the given filters.

        Parameters
        ----------
        filters: :class:`numpy.ndarray`
            The filters to scale
        variance: float
            The amount of variance

        Returns
        -------
        :class:`numpy.ndarray`
            The scaled filters
        """
        c_var = np.var(filters)
        var_p = np.sqrt(variance / c_var)
        retval = filters * var_p
        logger.debug("Scaled filters (filters: %s, variance: %s, output: %s)",
                     filters.shape, variance, retval.shape)
        return retval

    def __call__(self,  # pylint: disable=too-many-locals
                 shape: list[int] | tuple[int, ...],
                 dtype: str | None = None) -> Variable:
        """ Call function for the ICNR initializer.

        Parameters
        ----------
        shape: list[int] | tuple[int, ...]
            The required shape for the output tensor
        dtype: str
            The data type for the tensor

        Returns
        -------
        :class:`keras.Variable`
            The modified kernel weights
        """
        if self._initialized:   # Avoid re-calculating initializer when loading a saved model
            return T.cast("Variable", self._he_uniform(shape, dtype=dtype))
        dtype = K.floatx() if dtype is None else dtype
        logger.info("Calculating Convolution Aware Initializer for shape: %s", shape)
        rank = len(shape)
        if self._seed is not None:
            np.random.seed(self._seed)

        fan_in, _ = compute_fans(shape)
        variance = 2 / fan_in

        kernel_shape: tuple[int, ...]
        transpose_dimensions: tuple[int, ...]
        correct_ifft: T.Callable
        correct_fft: T.Callable

        if rank == 3:
            row, stack_size, filters_size = shape

            transpose_dimensions = (2, 1, 0)
            kernel_shape = (row,)
            correct_ifft = lambda shape, s=[None]: np.fft.irfft(shape, s[0])  # noqa:E731,E501 pylint:disable=unnecessary-lambda-assignment

            correct_fft = np.fft.rfft

        elif rank == 4:
            row, column, stack_size, filters_size = shape

            transpose_dimensions = (2, 3, 1, 0)
            kernel_shape = (row, column)
            correct_ifft = np.fft.irfft2
            correct_fft = np.fft.rfft2

        elif rank == 5:
            var_x, var_y, var_z, stack_size, filters_size = shape

            transpose_dimensions = (3, 4, 0, 1, 2)
            kernel_shape = (var_x, var_y, var_z)
            correct_fft = np.fft.rfftn
            correct_ifft = np.fft.irfftn

        else:
            self._initialized = True
            return Variable(self._orthogonal(shape), dtype=dtype)

        kernel_fourier_shape = correct_fft(np.zeros(kernel_shape)).shape

        basis = self._create_basis(filters_size,
                                   stack_size,
                                   T.cast(int, np.prod(kernel_fourier_shape)),
                                   dtype)
        basis = basis.reshape((filters_size, stack_size,) + kernel_fourier_shape)
        randoms = np.random.normal(0, self._eps_std, basis.shape[:-2] + kernel_shape)
        init = correct_ifft(basis, kernel_shape) + randoms
        init = self._scale_filters(init, variance)
        self._initialized = True
        retval = Variable(init.transpose(transpose_dimensions), dtype=dtype, name="conv_aware")
        logger.debug("ConvAware output: %s", retval)
        return retval

    def get_config(self) -> dict[str, T.Any]:
        """ Return the Convolutional Aware Initializer configuration.

        Returns
        -------
        dict[str, Any]
            The configuration for Convolutional Aware Initialization
        """
        config = {"eps_std": self._eps_std,
                  "seed": self._seed,
                  "initialized": self._initialized}
        # pylint:disable=duplicate-code
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


# pylint:disable=duplicate-code
# Update initializers into Keras custom objects
for name, obj in inspect.getmembers(sys.modules[__name__]):
    if inspect.isclass(obj) and obj.__module__ == __name__:
        saving.get_custom_objects().update({name: obj})


__all__ = get_module_objects(__name__)
