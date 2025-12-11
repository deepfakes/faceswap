#!/usr/bin/env python3
""" Custom Layers for faceswap.py. """
from __future__ import annotations

import inspect
import logging
import operator
import sys
import typing as T

from keras import InputSpec, Layer, ops, saving

from lib.logger import parse_class_init
from lib.utils import get_module_objects

if T.TYPE_CHECKING:
    from keras import KerasTensor


logger = logging.getLogger(__name__)


class _GlobalPooling2D(Layer):  # pylint:disable=too-many-ancestors
    """Abstract class for different global pooling 2D layers. """
    def __init__(self, data_format: str | None = None, **kwargs) -> None:
        logger.debug(parse_class_init(locals()))

        super().__init__(**kwargs)
        self.data_format = "channels_last" if data_format is None else data_format
        self.input_spec = InputSpec(ndim=4)
        logger.debug("Initialized %s", self.__class__.__name__)

    def compute_output_shape(self, input_shape: tuple[int, ...]  # pylint:disable=arguments-differ
                             ) -> tuple[int, ...]:
        """ Compute the output shape based on the input shape.

        Parameters
        ----------
        input_shape: tuple
            The input shape to the layer
        """
        if self.data_format == "channels_last":
            return (input_shape[0], input_shape[3])
        return (input_shape[0], input_shape[1])

    def call(self, inputs: KerasTensor, *args, **kwargs  # pylint:disable=arguments-differ
             ) -> KerasTensor:
        """ Override to call the layer.

        Parameters
        ----------
        inputs: :class:`keras.KerasTensor`
            The input to the layer

        Returns
        -------
        :class:`keras.KerasTensor`
            The output from the layer

        """
        raise NotImplementedError

    def get_config(self) -> dict[str, T.Any]:
        """ Set the Keras config """
        config = {"data_format": self.data_format}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class GlobalMinPooling2D(_GlobalPooling2D):  # pylint:disable=too-many-ancestors,abstract-method
    """Global minimum pooling operation for spatial data. """

    def call(self, inputs: KerasTensor, *args, **kwargs  # pylint:disable=arguments-differ
             ) -> KerasTensor:
        """This is where the layer's logic lives.

        Parameters
        ----------
        inputs: :class:`keras.KerasTensor`
            Input tensor, or list/tuple of input tensors

        Returns
        -------
        :class:`keras.KerasTensor`
            A tensor or list/tuple of tensors
        """
        if self.data_format == "channels_last":
            pooled = ops.min(inputs, axis=[1, 2])
        else:
            pooled = ops.min(inputs, axis=[2, 3])
        return pooled


class GlobalStdDevPooling2D(_GlobalPooling2D):  # pylint:disable=too-many-ancestors,abstract-method
    """Global standard deviation pooling operation for spatial data. """

    def call(self, inputs: KerasTensor, *args, **kwargs  # pylint:disable=arguments-differ
             ) -> KerasTensor:
        """This is where the layer's logic lives.

        Parameters
        ----------
        inputs: :class:`keras.KerasTensor`
            Input tensor, or list/tuple of input tensors

        Returns
        -------
        :class:`keras.KerasTensor`
            A tensor or list/tuple of tensors
        """
        if self.data_format == "channels_last":
            pooled = ops.std(inputs, axis=[1, 2])
        else:
            pooled = ops.std(inputs, axis=[2, 3])
        return pooled


class KResizeImages(Layer):  # pylint:disable=too-many-ancestors,abstract-method
    """ A custom upscale function that uses :class:`keras.backend.resize_images` to upsample.

    Parameters
    ----------
    size: int or float, optional
        The scale to upsample to. Default: `2`
    interpolation: ["nearest", "bilinear"], optional
        The interpolation to use. Default: `"nearest"`
    kwargs: dict
        The standard Keras Layer keyword arguments (if any)
    """
    def __init__(self,
                 size: int = 2,
                 interpolation: T.Literal["nearest", "bilinear"] = "nearest",
                 **kwargs) -> None:
        logger.debug(parse_class_init(locals()))
        super().__init__(**kwargs)
        self.size = size
        self.interpolation = interpolation
        logger.debug("Initialized %s", self.__class__.__name__)

    def call(self, inputs: KerasTensor, *args, **kwargs  # pylint:disable=arguments-differ
             ) -> KerasTensor:
        """ Call the upsample layer

        Parameters
        ----------
        inputs: :class:`keras.KerasTensor`
            Input tensor, or list/tuple of input tensors

        Returns
        -------
        :class:`keras.KerasTensor`
            A tensor or list/tuple of tensors
        """
        height, width = inputs.shape[1:3]
        assert height is not None and width is not None
        size = int(round(width * self.size)), int(round(height * self.size))
        retval = ops.image.resize(inputs,
                                  size,
                                  interpolation=self.interpolation,
                                  data_format="channels_last")
        return retval

    def compute_output_shape(self, input_shape: tuple[int, ...]  # pylint:disable=arguments-differ
                             ) -> tuple[int, ...]:
        """Computes the output shape of the layer.

        This is the input shape with size dimensions multiplied by :attr:`size`

        Parameters
        ----------
        input_shape: tuple or list of tuples
            Shape tuple (tuple of integers) or list of shape tuples (one per output tensor of the
            layer).  Shape tuples can include None for free dimensions, instead of an integer.

        Returns
        -------
        tuple
            An input shape tuple
        """
        batch, height, width, channels = input_shape
        return (batch, int(round(height * self.size)), int(round(width * self.size)), channels)

    def get_config(self) -> dict[str, T.Any]:
        """Returns the config of the layer.

        Returns
        --------
        dict
            A python dictionary containing the layer configuration
        """
        config = {"size": self.size, "interpolation": self.interpolation}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class L2Normalize(Layer):  # pylint:disable=too-many-ancestors,abstract-method
    """ Normalizes a tensor w.r.t. the L2 norm alongside the specified axis.

    Parameters
    ----------
    axis: int
        The axis to perform normalization across
    kwargs: dict
        The standard Keras Layer keyword arguments (if any)
    """
    def __init__(self, axis: int, **kwargs) -> None:
        logger.debug(parse_class_init(locals()))
        self.axis = axis
        super().__init__(**kwargs)
        logger.debug("Initialized %s", self.__class__.__name__)

    def compute_output_shape(self, input_shape: tuple[int, ...]  # pylint:disable=arguments-differ
                             ) -> tuple[int, ...]:
        """ Compute the output shape based on the input shape.

        Parameters
        ----------
        input_shape: tuple
            The input shape to the layer
        """
        return input_shape

    def call(self, inputs: KerasTensor, *args, **kwargs  # pylint:disable=arguments-differ
             ) -> KerasTensor:
        """This is where the layer's logic lives.

        Parameters
        ----------
        inputs: :class:`keras.KerasTensor`
            Input tensor, or list/tuple of input tensors

        Returns
        -------
        :class:`keras.KerasTensor`
            A tensor or list/tuple of tensors
        """
        return ops.normalize(inputs, self.axis, order=2)

    def get_config(self) -> dict[str, T.Any]:
        """Returns the config of the layer.

        A layer config is a Python dictionary (serializable) containing the configuration of a
        layer. The same layer can be reinstated later (without its trained weights) from this
        configuration.

        The configuration of a layer does not include connectivity information, nor the layer
        class name. These are handled by `Network` (one layer of abstraction above).

        Returns
        --------
        dict
            A python dictionary containing the layer configuration
        """
        config = super().get_config()
        config["axis"] = self.axis
        return config


class PixelShuffler(Layer):  # pylint:disable=too-many-ancestors,abstract-method
    """ PixelShuffler layer for Keras.

    This layer requires a Convolution2D prior to it, having output filters computed according to
    the formula :math:`filters = k * (scale_factor * scale_factor)` where `k` is a user defined
    number of filters (generally larger than 32) and `scale_factor` is the up-scaling factor
    (generally 2).

    This layer performs the depth to space operation on the convolution filters, and returns a
    tensor with the size as defined below.

    Notes
    -----
    In practice, it is useful to have a second convolution layer after the
    :class:`PixelShuffler` layer to speed up the learning process. However, if you are stacking
    multiple :class:`PixelShuffler` blocks, it may increase the number of parameters greatly,
    so the Convolution layer after :class:`PixelShuffler` layer can be removed.

    Example
    -------
    >>> # A standard sub-pixel up-scaling block
    >>> x = Convolution2D(256, 3, 3, padding="same", activation="relu")(...)
    >>> u = PixelShuffler(size=(2, 2))(x)
    [Optional]
    >>> x = Convolution2D(256, 3, 3, padding="same", activation="relu")(u)

    Parameters
    ----------
    size: tuple, optional
        The (`h`, `w`) scaling factor for up-scaling. Default: `(2, 2)`
    data_format: ["channels_first", "channels_last", ``None``], optional
        The data format for the input. Default: ``None``
    kwargs: dict
        The standard Keras Layer keyword arguments (if any)

    References
    ----------
    https://gist.github.com/t-ae/6e1016cc188104d123676ccef3264981
    """
    def __init__(self,
                 size: int | tuple[int, int] = (2, 2),
                 data_format: str | None = None,
                 **kwargs) -> None:
        logger.debug(parse_class_init(locals()))
        super().__init__(**kwargs)
        self.data_format = "channels_last" if data_format is None else data_format
        self.size = (size, size) if isinstance(size, int) else tuple(size)
        logger.debug("Initialized %s", self.__class__.__name__)

    def call(self, inputs: KerasTensor, *args, **kwargs  # pylint:disable=arguments-differ
             ) -> KerasTensor:
        """This is where the layer's logic lives.

        Parameters
        ----------
        inputs: :class:`keras.KerasTensor`
            Input tensor, or list/tuple of input tensors

        Returns
        -------
        :class:`keras.KerasTensor`
            A tensor or list/tuple of tensors
        """
        input_shape = inputs.shape
        if len(input_shape) != 4:
            raise ValueError("Inputs should have rank " +
                             str(4) +
                             "; Received input shape:", str(input_shape))

        out = None
        if self.data_format == "channels_first":
            batch_size, channels, height, width = input_shape
            assert height is not None and width is not None and channels is not None
            if batch_size is None:
                batch_size = -1
            r_height, r_width = self.size
            o_height, o_width = height * r_height, width * r_width
            o_channels = channels // (r_height * r_width)

            out = ops.reshape(inputs, (batch_size, r_height, r_width, o_channels, height, width))
            out = ops.transpose(out, (0, 3, 4, 1, 5, 2))
            out = ops.reshape(out, (batch_size, o_channels, o_height, o_width))
        elif self.data_format == "channels_last":
            batch_size, height, width, channels = input_shape
            assert height is not None and width is not None and channels is not None
            if batch_size is None:
                batch_size = -1
            r_height, r_width = self.size
            o_height, o_width = height * r_height, width * r_width
            o_channels = channels // (r_height * r_width)

            out = ops.reshape(inputs, (batch_size, height, width, r_height, r_width, o_channels))
            out = ops.transpose(out, (0, 1, 3, 2, 4, 5))
            out = ops.reshape(out, (batch_size, o_height, o_width, o_channels))
        assert out is not None
        return T.cast("KerasTensor", out)

    def compute_output_shape(self,    # pylint:disable=arguments-differ
                             input_shape: tuple[int | None, ...]) -> tuple[int | None, ...]:
        """Computes the output shape of the layer.

        Assumes that the layer will be built to match that input shape provided.

        Parameters
        ----------
        input_shape: tuple or list of tuples
            Shape tuple (tuple of integers) or list of shape tuples (one per output tensor of the
            layer).  Shape tuples can include None for free dimensions, instead of an integer.

        Returns
        -------
        tuple
            An input shape tuple
        """
        if len(input_shape) != 4:
            raise ValueError("Inputs should have rank " +
                             str(4) +
                             "; Received input shape:", str(input_shape))

        retval: tuple[int | None, ...]
        if self.data_format == "channels_first":
            height = None
            width = None
            if input_shape[2] is not None:
                height = input_shape[2] * self.size[0]
            if input_shape[3] is not None:
                width = input_shape[3] * self.size[1]
            chs = input_shape[1]
            assert chs is not None
            channels = chs // self.size[0] // self.size[1]

            if channels * self.size[0] * self.size[1] != input_shape[1]:
                raise ValueError("channels of input and size are incompatible")

            retval = (input_shape[0],
                      channels,
                      height,
                      width)
        else:
            height = None
            width = None
            if input_shape[1] is not None:
                height = input_shape[1] * self.size[0]
            if input_shape[2] is not None:
                width = input_shape[2] * self.size[1]
            chs = input_shape[3]
            assert chs is not None
            channels = chs // self.size[0] // self.size[1]

            if channels * self.size[0] * self.size[1] != input_shape[3]:
                raise ValueError("channels of input and size are incompatible")

            retval = (input_shape[0],
                      height,
                      width,
                      channels)
        return retval

    def get_config(self) -> dict[str, T.Any]:
        """Returns the config of the layer.

        A layer config is a Python dictionary (serializable) containing the configuration of a
        layer. The same layer can be reinstated later (without its trained weights) from this
        configuration.

        The configuration of a layer does not include connectivity information, nor the layer
        class name. These are handled by `Network` (one layer of abstraction above).

        Returns
        --------
        dict
            A python dictionary containing the layer configuration
        """
        config = {"size": self.size,
                  "data_format": self.data_format}
        base_config = super().get_config()

        return dict(list(base_config.items()) + list(config.items()))


class QuickGELU(Layer):  # pylint:disable=too-many-ancestors,abstract-method
    """ Applies GELU approximation that is fast but somewhat inaccurate.

    Parameters
    ----------
    name: str, optional
        The name for the layer. Default: "QuickGELU"
    kwargs: dict
        The standard Keras Layer keyword arguments (if any)
    """
    def __init__(self, name: str = "QuickGELU", **kwargs) -> None:
        logger.debug(parse_class_init(locals()))
        super().__init__(name=name, **kwargs)
        logger.debug("Initialized %s", self.__class__.__name__)

    def compute_output_shape(self, input_shape: tuple[int, ...]  # pylint:disable=arguments-differ
                             ) -> tuple[int, ...]:
        """ Compute the output shape based on the input shape.

        Parameters
        ----------
        input_shape: tuple
            The input shape to the layer
        """
        return input_shape

    def call(self, inputs: KerasTensor, *args, **kwargs  # pylint:disable=arguments-differ
             ) -> KerasTensor:
        """ Call the QuickGELU layerr

        Parameters
        ----------
        inputs : :class:`keras.KerasTensor`
            The input Tensor

        Returns
        -------
        :class:`keras.KerasTensor`
            The output Tensor
        """
        return inputs * ops.sigmoid(1.702 * inputs)


class ReflectionPadding2D(Layer):  # pylint:disable=too-many-ancestors,abstract-method
    """Reflection-padding layer for 2D input (e.g. picture).

    This layer can add rows and columns at the top, bottom, left and right side of an image tensor.

    Parameters
    ----------
    stride: int, optional
        The stride of the following convolution. Default: `2`
    kernel_size: int, optional
        The kernel size of the following convolution. Default: `5`
    kwargs: dict
        The standard Keras Layer keyword arguments (if any)
    """
    def __init__(self, stride: int = 2, kernel_size: int = 5, **kwargs) -> None:
        logger.debug(parse_class_init(locals()))

        if isinstance(stride, (tuple, list)):
            assert len(stride) == 2 and stride[0] == stride[1]
            stride = stride[0]
        self.stride = stride
        self.kernel_size = kernel_size
        self.input_spec: list[InputSpec] | None = None
        super().__init__(**kwargs)

        logger.debug("Initialized %s", self.__class__.__name__)

    def build(self, input_shape: KerasTensor) -> None:
        """Creates the layer weights.

        Must be implemented on all layers that have weights.

        Parameters
        ----------
        input_shape: :class:`keras.KerasTensor`
            Keras tensor (future input to layer) or ``list``/``tuple`` of Keras tensors to
            reference for weight shape computations.
        """
        self.input_spec = [InputSpec(shape=input_shape)]
        super().build(input_shape)

    def compute_output_shape(self, *args, **kwargs) -> tuple[int | None, ...]:
        """Computes the output shape of the layer.

        Assumes that the layer will be built to match that input shape provided.

        Returns
        -------
        tuple
            An input shape tuple
        """
        assert self.input_spec is not None
        input_shape = self.input_spec[0].shape
        assert input_shape is not None
        assert input_shape[1] is not None and input_shape[2] is not None
        in_width, in_height = input_shape[2], input_shape[1]
        kernel_width, kernel_height = self.kernel_size, self.kernel_size

        if (in_height % self.stride) == 0:
            padding_height = max(kernel_height - self.stride, 0)
        else:
            padding_height = max(kernel_height - (in_height % self.stride), 0)
        if (in_width % self.stride) == 0:
            padding_width = max(kernel_width - self.stride, 0)
        else:
            padding_width = max(kernel_width - (in_width % self.stride), 0)

        return (input_shape[0],
                input_shape[1] + padding_height,
                input_shape[2] + padding_width,
                input_shape[3])

    def call(self, inputs: KerasTensor, *args, **kwargs  # pylint:disable=arguments-differ
             ) -> KerasTensor:
        """This is where the layer's logic lives.

        Parameters
        ----------
        inputs: :class:`keras.KerasTensor`
            Input tensor, or list/tuple of input tensors

        Returns
        -------
        :class:`keras.KerasTensor`
            A tensor or list/tuple of tensors
        """
        assert self.input_spec is not None
        input_shape = self.input_spec[0].shape
        assert input_shape is not None
        assert input_shape[1] is not None and input_shape[2] is not None
        in_width, in_height = input_shape[2], input_shape[1]
        kernel_width, kernel_height = self.kernel_size, self.kernel_size

        if (in_height % self.stride) == 0:
            padding_height = max(kernel_height - self.stride, 0)
        else:
            padding_height = max(kernel_height - (in_height % self.stride), 0)
        if (in_width % self.stride) == 0:
            padding_width = max(kernel_width - self.stride, 0)
        else:
            padding_width = max(kernel_width - (in_width % self.stride), 0)

        padding_top = padding_height // 2
        padding_bot = padding_height - padding_top
        padding_left = padding_width // 2
        padding_right = padding_width - padding_left

        return ops.pad(inputs,
                       [[0, 0], [padding_top, padding_bot], [padding_left, padding_right], [0, 0]],
                       mode="reflect")

    def get_config(self) -> dict[str, T.Any]:
        """Returns the config of the layer.

        A layer config is a Python dictionary (serializable) containing the configuration of a
        layer. The same layer can be reinstated later (without its trained weights) from this
        configuration.

        The configuration of a layer does not include connectivity information, nor the layer
        class name. These are handled by `Network` (one layer of abstraction above).

        Returns
        --------
        dict
            A python dictionary containing the layer configuration
        """
        config = {"stride": self.stride,
                  "kernel_size": self.kernel_size}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Swish(Layer):  # pylint:disable=too-many-ancestors,abstract-method
    """ Swish Activation Layer implementation for Keras.

    Parameters
    ----------
    beta: float, optional
        The beta value to apply to the activation function. Default: `1.0`
    kwargs: dict
        The standard Keras Layer keyword arguments (if any)

    References
    -----------
    Swish: a Self-Gated Activation Function: https://arxiv.org/abs/1710.05941v1
    """
    def __init__(self, beta: float = 1.0, **kwargs) -> None:
        logger.debug(parse_class_init(locals()))
        super().__init__(**kwargs)
        self.beta = beta
        logger.debug("Initialized %s", self.__class__.__name__)

    def compute_output_shape(self, input_shape: tuple[int, ...]  # pylint:disable=arguments-differ
                             ) -> tuple[int, ...]:
        """ Compute the output shape based on the input shape.

        Parameters
        ----------
        input_shape: tuple
            The input shape to the layer
        """
        return input_shape

    def call(self, inputs: KerasTensor, *args, **kwargs  # pylint:disable=arguments-differ
             ) -> KerasTensor:
        """ Call the Swish Activation function.

        Parameters
        ----------
        inputs: tensor
            Input tensor, or list/tuple of input tensors

        Returns
        -------
        :class:`keras.KerasTensor`
            A tensor or list/tuple of tensors
        """
        return ops.nn.swish(inputs * self.beta)

    def get_config(self):
        """Returns the config of the layer.

        Adds the :attr:`beta` to config.

        Returns
        --------
        dict
            A python dictionary containing the layer configuration
        """
        config = super().get_config()
        config["beta"] = self.beta
        return config


class ScalarOp(Layer):  # pylint:disable=too-many-ancestors,abstract-method
    """ A layer for scalar operations for migrating TFLambdaOps in Keras 2 models to Keras 3. This
    layer should not be used directly

    Parameters
    ----------
    operation: Literal["multiply", "truediv", "add", "subtract"]
        The scalar operation to perform
    value: float
        The scalar value to use
    """
    def __init__(self,
                 operation: T.Literal["multiply", "truediv", "add", "subtract"],
                 value: float,
                 **kwargs) -> None:
        logger.debug(parse_class_init(locals()))
        assert operation in ("multiply", "truediv", "add", "subtract")
        self._operation = operation
        self._operator = {"multiply": operator.mul,
                          "truediv": operator.truediv,
                          "add": operator.add,
                          "subtract": operator.sub}[operation]
        self._value = value

        if "name" not in kwargs:
            kwargs["name"] = f"ScalarOp_{operation}"
        super().__init__(**kwargs)

        logger.debug("Initialized %s", self.__class__.__name__)

    def compute_output_shape(self, input_shape: tuple[int, ...]  # pylint:disable=arguments-differ
                             ) -> tuple[int, ...]:
        """ Output shape is the same as the input shape.

        Parameters
        ----------
        input_shape: tuple
            The input shape to the layer
        """
        return input_shape

    def call(self, inputs: KerasTensor, *args, **kwargs  # pylint:disable=arguments-differ
             ) -> KerasTensor:
        """ Call the Scalar operation function.

        Parameters
        ----------
        inputs: tensor
            Input tensor, or list/tuple of input tensors

        Returns
        -------
        :class:`keras.KerasTensor`
            A tensor or list/tuple of tensors
        """
        return self._operator(inputs, self._value)

    def get_config(self):
        """Returns the config of the layer.
        Returns
        --------
        dict
            A python dictionary containing the layer configuration
        """
        config = super().get_config()
        config["operation"] = self._operation
        config["value"] = self._value
        return config


# Update layers into Keras custom objects
for name_, obj in inspect.getmembers(sys.modules[__name__]):
    if inspect.isclass(obj) and obj.__module__ == __name__:
        saving.get_custom_objects().update({name_: obj})


__all__ = get_module_objects(__name__)
