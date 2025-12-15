#!/usr/bin/env python3
""" Neural Network Blocks for faceswap.py. """
from __future__ import annotations
import logging
import typing as T

from keras import initializers, layers

from lib.logger import parse_class_init
from lib.utils import get_module_objects
from plugins.train import train_config as cfg

from .initializers import ICNR, ConvolutionAware
from .layers import PixelShuffler, ReflectionPadding2D, Swish, KResizeImages
from .normalization import InstanceNormalization

if T.TYPE_CHECKING:
    from keras import KerasTensor

logger = logging.getLogger(__name__)


_names: dict[str, int] = {}


def _get_name(name: str) -> str:
    """ Return unique layer name for requested block.

    As blocks can be used multiple times, auto appends an integer to the end of the requested
    name to keep all block names unique

    Parameters
    ----------
    name: str
        The requested name for the layer

    Returns
    -------
    str
        The unique name for this layer
    """
    _names[name] = _names.setdefault(name, -1) + 1
    name = f"{name}_{_names[name]}"
    logger.debug("Generating block name: %s", name)
    return name


def reset_naming() -> None:
    """ Reset the naming convention for nn_block layers to start from 0

    Used when a model needs to be rebuilt and the names for each build should be identical
    """
    logger.debug("Resetting nn_block layer naming")
    global _names  # pylint:disable=global-statement
    _names = {}


#  << CONVOLUTIONS >>
def _get_default_initializer(
        initializer: initializers.Initializer) -> initializers.Initializer:
    """ Returns a default initializer of Convolutional Aware or HeUniform for convolutional
    layers.

    Parameters
    ----------
    initializer: :class:`keras.initializers.Initializer` or None
        The initializer that has been passed into the model. If this value is ``None`` then a
        default initializer will be set to 'HeUniform'. If Convolutional Aware initialization
        has been enabled, then any passed through initializer will be replaced with the
        Convolutional Aware initializer.

    Returns
    -------
    :class:`keras.initializers.Initializer`
        The kernel initializer to use for this convolutional layer. Either the original given
        initializer, HeUniform or convolutional aware (if selected in config options)
    """
    if isinstance(initializer, dict) and initializer.get("class_name", "") == "ConvolutionAware":
        logger.debug("Returning serialized initialized ConvAware initializer: %s", initializer)
        return initializer

    if cfg.conv_aware_init():
        retval = ConvolutionAware()
    elif initializer is None:
        retval = initializers.HeUniform()
    else:
        retval = initializer
        logger.debug("Using model supplied initializer: %s", retval)
    logger.debug("Set default kernel_initializer: (original: %s current: %s)", initializer, retval)

    return retval


class Conv2D():  # pylint:disable=too-many-ancestors,abstract-method
    """ A standard Keras Convolution 2D layer with parameters updated to be more appropriate for
    Faceswap architecture.

    Parameters are the same, with the same defaults, as a standard :class:`keras.layers.Conv2D`
    except where listed below. The default initializer is updated to `HeUniform` or `convolutional
    aware` based on user configuration settings.

    Parameters
    ----------
    padding: str, optional
        One of `"valid"` or `"same"` (case-insensitive). Default: `"same"`. Note that `"same"` is
        slightly inconsistent across backends with `strides` != 1, as described
        `here <https://github.com/keras-team/keras/pull/9473#issuecomment-372166860/>`_.
    is_upscale: `bool`, optional
        ``True`` if the convolution is being called from an upscale layer. This causes the instance
        to check the user configuration options to see if ICNR initialization has been selected and
        should be applied. This should only be passed in as ``True`` from :class:`UpscaleBlock`
        layers. Default: ``False``
    """
    def __init__(self, *args, padding: str = "same", is_upscale: bool = False, **kwargs) -> None:
        logger.debug(parse_class_init(locals()))
        if kwargs.get("name", None) is None:
            filters = kwargs["filters"] if "filters" in kwargs else args[0]
            kwargs["name"] = _get_name(f"conv2d_{filters}")
        initializer = _get_default_initializer(kwargs.pop("kernel_initializer", None))
        if is_upscale and cfg.icnr_init():
            initializer = ICNR(initializer=initializer)
            logger.debug("Using ICNR Initializer: %s", initializer)
        self._conv2d = layers.Conv2D(
            *args,
            padding=padding,
            kernel_initializer=initializer,  # pyright:ignore[reportArgumentType]
            **kwargs)
        logger.debug("Initialized %s", self.__class__.__name__)

    def __call__(self, *args, **kwargs) -> KerasTensor:
        """ Call the Conv2D layer

        Parameters
        ----------
        args : tuple
            Standard Conv2D layer call arguments
        kwargs : dict[str, Any]
            Standard Conv2D layer call keyword arguments

        Returns
        -------
        :class: `keras.KerasTensor`
            The Tensor from the Conv2D layer
        """
        return self._conv2d(*args, **kwargs)

class DepthwiseConv2D():  # noqa,pylint:disable=too-many-ancestors,abstract-method
    """ A standard Keras Depthwise Convolution 2D layer with parameters updated to be more
    appropriate for Faceswap architecture.

    Parameters are the same, with the same defaults, as a standard
    :class:`keras.layers.DepthwiseConv2D` except where listed below. The default initializer is
    updated to `HeUniform` or `convolutional aware` based on user configuration settings.

    Parameters
    ----------
    padding: str, optional
        One of `"valid"` or `"same"` (case-insensitive). Default: `"same"`. Note that `"same"` is
        slightly inconsistent across backends with `strides` != 1, as described
        `here <https://github.com/keras-team/keras/pull/9473#issuecomment-372166860/>`_.
    is_upscale: `bool`, optional
        ``True`` if the convolution is being called from an upscale layer. This causes the instance
        to check the user configuration options to see if ICNR initialization has been selected and
        should be applied. This should only be passed in as ``True`` from :class:`UpscaleBlock`
        layers. Default: ``False``
    """
    def __init__(self, *args, padding: str = "same", is_upscale: bool = False, **kwargs) -> None:
        logger.debug(parse_class_init(locals()))
        if kwargs.get("name", None) is None:
            kwargs["name"] = _get_name("dwconv2d")
        initializer = _get_default_initializer(kwargs.pop("depthwise_initializer", None))
        if is_upscale and cfg.icnr_init():
            initializer = ICNR(initializer=initializer)
            logger.debug("Using ICNR Initializer: %s", initializer)
        self._deptwiseconv2d = layers.DepthwiseConv2D(
            *args,
            padding=padding,
            depthwise_initializer=initializer,  # pyright:ignore[reportArgumentType]
            **kwargs)
        logger.debug("Initialized %s", self.__class__.__name__)

    def __call__(self, *args, **kwargs) -> KerasTensor:
        """ Call the DepthwiseConv2D layer

        Parameters
        ----------
        args : tuple
            Standard DepthwiseConv2D layer call arguments
        kwargs : dict[str, Any]
            Standard DepthwiseConv2D layer call keyword arguments

        Returns
        -------
        :class: `keras.KerasTensor`
            The Tensor from the DepthwiseConv2D layer
        """
        return self._deptwiseconv2d(*args, **kwargs)


class Conv2DOutput():
    """ A Convolution 2D layer that separates out the activation layer to explicitly set the data
    type on the activation to float 32 to fully support mixed precision training.

    The Convolution 2D layer uses default parameters to be more appropriate for Faceswap
    architecture.

    Parameters are the same, with the same defaults, as a standard :class:`keras.layers.Conv2D`
    except where listed below. The default initializer is updated to HeUniform or convolutional
    aware based on user config settings.

    Parameters
    ----------
    filters: int
        The dimensionality of the output space (i.e. the number of output filters in the
        convolution)
    kernel_size: int or tuple/list of 2 ints
        The height and width of the 2D convolution window. Can be a single integer to specify the
        same value for all spatial dimensions.
    activation: str, optional
        The activation function to apply to the output. Default: `"sigmoid"`
    padding: str, optional
        One of `"valid"` or `"same"` (case-insensitive). Default: `"same"`. Note that `"same"` is
        slightly inconsistent across backends with `strides` != 1, as described
        `here <https://github.com/keras-team/keras/pull/9473#issuecomment-372166860/>`_.
    kwargs: dict
        Any additional Keras standard layer keyword arguments to pass to the Convolutional 2D layer
    """
    def __init__(self,
                 filters: int,
                 kernel_size: int | tuple[int],
                 activation: str = "sigmoid",
                 padding: str = "same", **kwargs) -> None:
        logger.debug(parse_class_init(locals()))
        name = _get_name(kwargs.pop("name")) if "name" in kwargs else _get_name(
                         f"conv_output_{filters}")
        self._conv = Conv2D(filters,
                            kernel_size,
                            padding=padding,
                            name=f"{name}_conv2d",
                            **kwargs)
        self._activation = layers.Activation(activation, dtype="float32", name=name)
        logger.debug("Initialized %s", self.__class__.__name__)

    def __call__(self, inputs: KerasTensor) -> KerasTensor:
        """ Call the Faceswap Convolutional Output Layer.

        Parameters
        ----------
        inputs: :class:`keras.KerasTensor`
            The input to the layer

        Returns
        -------
        :class:`keras.KerasTensor`
            The output tensor from the Convolution 2D Layer
        """
        var_x = self._conv(inputs)
        return self._activation(var_x)


class Conv2DBlock():  # pylint:disable=too-many-instance-attributes
    """ A standard Convolution 2D layer which applies user specified configuration to the
    layer.

    Adds reflection padding if it has been selected by the user, and other post-processing
    if requested by the plugin.

    Adds instance normalization if requested. Adds a LeakyReLU if a residual block follows.

    Parameters
    ----------
    filters: int
        The dimensionality of the output space (i.e. the number of output filters in the
        convolution)
    kernel_size: int, optional
        An integer or tuple/list of 2 integers, specifying the height and width of the 2D
        convolution window. Can be a single integer to specify the same value for all spatial
        dimensions. NB: If `use_depthwise` is ``True`` then a value must still be provided here,
        but it will be ignored. Default: 5
    strides: tuple or int, optional
        An integer or tuple/list of 2 integers, specifying the strides of the convolution along the
        height and width. Can be a single integer to specify the same value for all spatial
        dimensions. Default: `2`
    padding: ["valid", "same"], optional
        The padding to use. NB: If reflect padding has been selected in the user configuration
        options, then this argument will be ignored in favor of reflect padding. Default: `"same"`
    normalization: str or ``None``, optional
        Normalization to apply after the Convolution Layer. Select one of "batch" or "instance".
        Set to ``None`` to not apply normalization. Default: ``None``
    activation: str or ``None``, optional
        The activation function to use. This is applied at the end of the convolution block. Select
        one of `"leakyrelu"`, `"prelu"` or `"swish"`. Set to ``None`` to not apply an activation
        function. Default: `"leakyrelu"`
    use_depthwise: bool, optional
        Set to ``True`` to use a Depthwise Convolution 2D layer rather than a standard Convolution
        2D layer. Default: ``False``
    relu_alpha: float
        The alpha to use for LeakyRelu Activation. Default=`0.1`
    kwargs: dict
        Any additional Keras standard layer keyword arguments to pass to the Convolutional 2D layer
    """
    def __init__(self,
                 filters: int,
                 kernel_size: int | tuple[int, int] = 5,
                 strides: int | tuple[int, int] = 2,
                 padding: str = "same",
                 normalization: str | None = None,
                 activation: str | None = "leakyrelu",
                 use_depthwise: bool = False,
                 relu_alpha: float = 0.1,
                 **kwargs) -> None:
        logger.debug(parse_class_init(locals()))

        self._name = kwargs.pop("name") if "name" in kwargs else _get_name(f"conv_{filters}")
        self._use_reflect_padding = cfg.reflect_padding()

        kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self._args = (kernel_size, ) if use_depthwise else (filters, kernel_size)
        self._strides = (strides, strides) if isinstance(strides, int) else strides
        self._padding = "valid" if self._use_reflect_padding else padding
        self._kwargs = kwargs
        self._normalization = None if not normalization else normalization.lower()
        self._activation = None if not activation else activation.lower()
        self._use_depthwise = use_depthwise
        self._relu_alpha = relu_alpha

        self._assert_arguments()
        self._layers = self._get_layers()
        logger.debug("Initialized %s", self.__class__.__name__)

    def _assert_arguments(self) -> None:
        """ Validate the given arguments. """
        assert self._normalization in ("batch", "instance", None), (
            "normalization should be 'batch', 'instance' or None")
        assert self._activation in ("leakyrelu", "swish", "prelu", None), (
            "activation should be 'leakyrelu', 'prelu', 'swish' or None")

    def _get_layers(self) -> list[layers.Layer]:
        """ Obtain the layer chain for the block

        Returns
        -------
        list[:class:`keras.layers.Layer]
            The layers, in the correct order, to pass the tensor through
        """
        retval = []
        if self._use_reflect_padding:
            retval.append(ReflectionPadding2D(stride=self._strides[0],
                                              kernel_size=self._args[-1][0],  # type:ignore[index]
                                              name=f"{self._name}_reflectionpadding2d"))

        conv: layers.Layer = (
            DepthwiseConv2D if self._use_depthwise
            else Conv2D)  # pyright:ignore[reportAssignmentType]

        retval.append(conv(*self._args,
                           strides=self._strides,
                           padding=self._padding,
                           name=f"{self._name}_{'dw' if self._use_depthwise else ''}conv2d",
                           **self._kwargs))

        # normalization
        if self._normalization == "instance":
            retval.append(InstanceNormalization(name=f"{self._name}_instancenorm"))

        if self._normalization == "batch":
            retval.append(layers.BatchNormalization(axis=3, name=f"{self._name}_batchnorm"))

        # activation
        if self._activation == "leakyrelu":
            retval.append(layers.LeakyReLU(self._relu_alpha, name=f"{self._name}_leakyrelu"))
        if self._activation == "swish":
            retval.append(Swish(name=f"{self._name}_swish"))
        if self._activation == "prelu":
            retval.append(layers.PReLU(name=f"{self._name}_prelu"))

        logger.debug("%s layers: %s", self.__class__.__name__, retval)
        return retval

    def __call__(self, inputs: KerasTensor) -> KerasTensor:
        """ Call the Faceswap Convolutional Layer.

        Parameters
        ----------
        inputs: :class:`keras.KerasTensor`
            The input to the layer

        Returns
        -------
        :class:`keras.KerasTensor`
            The output tensor from the Convolution 2D Layer
        """
        var_x = inputs
        for layer in self._layers:
            var_x = layer(var_x)
        return var_x


class SeparableConv2DBlock():
    """ Seperable Convolution Block.

    Parameters
    ----------
    filters: int
        The dimensionality of the output space (i.e. the number of output filters in the
        convolution)
    kernel_size: int, optional
        An integer or tuple/list of 2 integers, specifying the height and width of the 2D
        convolution window. Can be a single integer to specify the same value for all spatial
        dimensions. Default: 5
    strides: tuple or int, optional
        An integer or tuple/list of 2 integers, specifying the strides of the convolution along
        the height and width. Can be a single integer to specify the same value for all spatial
        dimensions. Default: `2`
    kwargs: dict
        Any additional Keras standard layer keyword arguments to pass to the Separable
        Convolutional 2D layer
    """
    def __init__(self,
                 filters: int,
                 kernel_size: int | tuple[int, int] = 5,
                 strides: int | tuple[int, int] = 2, **kwargs) -> None:
        logger.debug(parse_class_init(locals()))

        initializer = _get_default_initializer(kwargs.pop("kernel_initializer", None))

        name = _get_name(f"separableconv2d_{filters}")
        self._conv = layers.SeparableConv2D(
            filters,
            kernel_size=kernel_size,
            strides=strides,
            padding="same",
            depthwise_initializer=initializer,  # pyright:ignore[reportArgumentType]
            pointwise_initializer=initializer,  # pyright:ignore[reportArgumentType]
            name=f"{name}_seperableconv2d",
            **kwargs)
        self._activation = layers.Activation("relu", name=f"{name}_relu")
        logger.debug("Initialized %s", self.__class__.__name__)

    def __call__(self, inputs: KerasTensor) -> KerasTensor:
        """ Call the Faceswap Separable Convolutional 2D Block.

        Parameters
        ----------
        inputs: :class:`keras.KerasTensor`
            The input to the layer

        Returns
        -------
        :class:`keras.KerasTensor`
            The output tensor from the Upscale Layer
        """
        var_x = self._conv(inputs)
        return self._activation(var_x)


#  << UPSCALING >>

class UpscaleBlock():
    """ An upscale layer for sub-pixel up-scaling.

    Adds reflection padding if it has been selected by the user, and other post-processing
    if requested by the plugin.

    Parameters
    ----------
    filters: int
        The dimensionality of the output space (i.e. the number of output filters in the
        convolution)
    kernel_size: int, optional
        An integer or tuple/list of 2 integers, specifying the height and width of the 2D
        convolution window. Can be a single integer to specify the same value for all spatial
        dimensions. Default: 3
    padding: ["valid", "same"], optional
        The padding to use. NB: If reflect padding has been selected in the user configuration
        options, then this argument will be ignored in favor of reflect padding. Default: `"same"`
    scale_factor: int, optional
        The amount to upscale the image. Default: `2`
    normalization: str or ``None``, optional
        Normalization to apply after the Convolution Layer. Select one of "batch" or "instance".
        Set to ``None`` to not apply normalization. Default: ``None``
    activation: str or ``None``, optional
        The activation function to use. This is applied at the end of the convolution block. Select
        one of `"leakyrelu"`, `"prelu"` or `"swish"`. Set to ``None`` to not apply an activation
        function. Default: `"leakyrelu"`
    kwargs: dict
        Any additional Keras standard layer keyword arguments to pass to the Convolutional 2D layer
    """

    def __init__(self,
                 filters: int,
                 kernel_size: int | tuple[int, int] = 3,
                 padding: str = "same",
                 scale_factor: int = 2,
                 normalization: str | None = None,
                 activation: str | None = "leakyrelu",
                 **kwargs) -> None:
        logger.debug(parse_class_init(locals()))
        name = _get_name(f"upscale_{filters}")
        self._conv = Conv2DBlock(filters * scale_factor * scale_factor,
                                 kernel_size,
                                 strides=(1, 1),
                                 padding=padding,
                                 normalization=normalization,
                                 activation=activation,
                                 name=f"{name}_conv2d",
                                 is_upscale=True,
                                 **kwargs)
        self._shuffle = PixelShuffler(name=f"{name}_pixelshuffler", size=scale_factor)
        logger.debug("Initialized %s", self.__class__.__name__)

    def __call__(self, inputs: KerasTensor) -> KerasTensor:
        """ Call the Faceswap Convolutional Layer.

        Parameters
        ----------
        inputs: :class:`keras.KerasTensor`
            The input to the layer

        Returns
        -------
        :class:`keras.KerasTensor`
            The output tensor from the Upscale Layer
        """
        var_x = self._conv(inputs)
        return self._shuffle(var_x)


class Upscale2xBlock():
    """ Custom hybrid upscale layer for sub-pixel up-scaling.

    Most of up-scaling is approximating lighting gradients which can be accurately achieved
    using linear fitting. This layer attempts to improve memory consumption by splitting
    with bilinear and convolutional layers so that the sub-pixel update will get details
    whilst the bilinear filter will get lighting.

    Adds reflection padding if it has been selected by the user, and other post-processing
    if requested by the plugin.

    Parameters
    ----------
    filters: int
        The dimensionality of the output space (i.e. the number of output filters in the
        convolution)
    kernel_size: int, optional
        An integer or tuple/list of 2 integers, specifying the height and width of the 2D
        convolution window. Can be a single integer to specify the same value for all spatial
        dimensions. Default: 3
    padding: ["valid", "same"], optional
        The padding to use. Default: `"same"`
    activation: str or ``None``, optional
        The activation function to use. This is applied at the end of the convolution block. Select
        one of `"leakyrelu"`, `"prelu"` or `"swish"`. Set to ``None`` to not apply an activation
        function. Default: `"leakyrelu"`
    interpolation: ["nearest", "bilinear"], optional
        Interpolation to use for up-sampling. Default: `"bilinear"`
    scale_factor: int, optional
        The amount to upscale the image. Default: `2`
    sr_ratio: float, optional
        The proportion of super resolution (pixel shuffler) filters to use. Non-fast mode only.
        Default: `0.5`
    fast: bool, optional
        Use a faster up-scaling method that may appear more rugged. Default: ``False``
    kwargs: dict
        Any additional Keras standard layer keyword arguments to pass to the Convolutional 2D layer
    """
    # TODO Class function this
    def __init__(self,
                 filters: int,
                 kernel_size: int | tuple[int, int] = 3,
                 padding: str = "same",
                 activation: str | None = "leakyrelu",
                 interpolation: str = "bilinear",
                 sr_ratio: float = 0.5,
                 scale_factor: int = 2,
                 fast: bool = False, **kwargs) -> None:
        logger.debug(parse_class_init(locals()))

        self._fast = fast
        self._filters = filters if fast else filters - int(filters * sr_ratio)

        name = _get_name(f"upscale2x_{filters}_{'fast' if fast else 'hyb'}")

        self._upscale = UpscaleBlock(self._filters,
                                     kernel_size=kernel_size,
                                     padding=padding,
                                     scale_factor=scale_factor,
                                     activation=activation,
                                     **kwargs)

        if self._fast or (not self._fast and self._filters > 0):
            self._conv = Conv2D(self._filters,
                                3,
                                padding=padding,
                                is_upscale=True,
                                name=f"{name}_conv2d",
                                **kwargs)
            self._upsample = layers.UpSampling2D(size=(scale_factor, scale_factor),
                                                 interpolation=interpolation,
                                                 name=f"{name}_upsampling2D")

        self._joiner = layers.Add() if self._fast else layers.Concatenate(
            name=f"{name}_concatenate")

        logger.debug("Initialized %s", self.__class__.__name__)

    def __call__(self, inputs: KerasTensor) -> KerasTensor:
        """ Call the Faceswap Upscale 2x Layer.

        Parameters
        ----------
        inputs: :class:`keras.KerasTensor`
            The input to the layer

        Returns
        -------
        :class:`keras.KerasTensor`
            The output tensor from the Upscale Layer
        """
        var_x = inputs
        var_x_sr = None
        if not self._fast:
            var_x_sr = self._upscale(var_x)
        if self._fast or (not self._fast and self._filters > 0):

            var_x2 = self._conv(var_x)
            var_x2 = self._upsample(var_x2)

            if self._fast:
                var_x1 = self._upscale(var_x)
                var_x = self._joiner([var_x2, var_x1])
            else:
                var_x = self._joiner([var_x_sr, var_x2])

        else:
            assert var_x_sr is not None
            var_x = var_x_sr

        return var_x


class UpscaleResizeImagesBlock():
    """ Upscale block that uses the Keras Backend function resize_images to perform the up scaling
    Similar in methodology to the :class:`Upscale2xBlock`

    Adds reflection padding if it has been selected by the user, and other post-processing
    if requested by the plugin.

    Parameters
    ----------
    filters: int
        The dimensionality of the output space (i.e. the number of output filters in the
        convolution)
    kernel_size: int, optional
        An integer or tuple/list of 2 integers, specifying the height and width of the 2D
        convolution window. Can be a single integer to specify the same value for all spatial
        dimensions. Default: 3
    padding: ["valid", "same"], optional
        The padding to use. Default: `"same"`
    activation: str or ``None``, optional
        The activation function to use. This is applied at the end of the convolution block. Select
        one of `"leakyrelu"`, `"prelu"` or `"swish"`. Set to ``None`` to not apply an activation
        function. Default: `"leakyrelu"`
    scale_factor: int, optional
        The amount to upscale the image. Default: `2`
    interpolation: ["nearest", "bilinear"], optional
        Interpolation to use for up-sampling. Default: `"bilinear"`
    kwargs: dict
        Any additional Keras standard layer keyword arguments to pass to the Convolutional 2D layer
    """
    def __init__(self,
                 filters: int,
                 kernel_size: int | tuple[int, int] = 3,
                 padding: str = "same",
                 activation: str | None = "leakyrelu",
                 scale_factor: int = 2,
                 interpolation: T.Literal["nearest", "bilinear"] = "bilinear") -> None:
        logger.debug(parse_class_init(locals()))
        name = _get_name(f"upscale_ri_{filters}")

        self._resize = KResizeImages(size=scale_factor,
                                     interpolation=interpolation,
                                     name=f"{name}_resize")
        self._conv = Conv2D(filters,
                            kernel_size,
                            strides=1,
                            padding=padding,
                            is_upscale=True,
                            name=f"{name}_conv")
        self._conv_trans = layers.Conv2DTranspose(filters,
                                                  3,
                                                  strides=2,
                                                  padding=padding,
                                                  name=f"{name}_convtrans")
        self._add = layers.Add()

        if activation == "leakyrelu":
            self._acivation = layers.LeakyReLU(0.2, name=f"{name}_leakyrelu")
        if activation == "swish":
            self._acivation = Swish(name=f"{name}_swish")
        if activation == "prelu":
            self._acivation = layers.PReLU(name=f"{name}_prelu")
        logger.debug("Initialized %s", self.__class__.__name__)

    def __call__(self, inputs: KerasTensor) -> KerasTensor:
        """ Call the Faceswap Resize Images Layer.

        Parameters
        ----------
        inputs: :class:`keras.KerasTensor`
            The input to the layer

        Returns
        -------
        :class:`keras.KerasTensor`
            The output tensor from the Upscale Layer
        """
        var_x = inputs

        var_x_sr = self._resize(var_x)
        var_x_sr = self._conv(var_x_sr)

        var_x_us = self._conv_trans(var_x)

        var_x = self._add([var_x_sr, var_x_us])

        return self._acivation(var_x)


class UpscaleDNYBlock():
    """ Upscale block that implements methodology similar to the Disney Research Paper using an
    upsampling2D block and 2 x convolutions

    Adds reflection padding if it has been selected by the user, and other post-processing
    if requested by the plugin.

    References
    ----------
    https://studios.disneyresearch.com/2020/06/29/high-resolution-neural-face-swapping-for-visual-effects/

    Parameters
    ----------
    filters: int
        The dimensionality of the output space (i.e. the number of output filters in the
        convolution)
    kernel_size: int, optional
        An integer or tuple/list of 2 integers, specifying the height and width of the 2D
        convolution window. Can be a single integer to specify the same value for all spatial
        dimensions. Default: 3
    activation: str or ``None``, optional
        The activation function to use. This is applied at the end of the convolution block. Select
        one of `"leakyrelu"`, `"prelu"` or `"swish"`. Set to ``None`` to not apply an activation
        function. Default: `"leakyrelu"`
    size: int, optional
        The amount to upscale the image. Default: `2`
    interpolation: ["nearest", "bilinear"], optional
        Interpolation to use for up-sampling. Default: `"bilinear"`
    kwargs: dict
        Any additional Keras standard layer keyword arguments to pass to the Convolutional 2D
        layers
    """
    def __init__(self,
                 filters: int,
                 kernel_size: int | tuple[int, int] = 3,
                 padding: str = "same",
                 activation: str | None = "leakyrelu",
                 size: int = 2,
                 interpolation: str = "bilinear",
                 **kwargs) -> None:
        logger.debug(parse_class_init(locals()))
        name = _get_name(f"upscale_dny_{filters}")
        self._upsample = layers.UpSampling2D(size=size,
                                             interpolation=interpolation,
                                             name=f"{name}_upsample2d")
        self._convs = [Conv2DBlock(filters,
                                   kernel_size,
                                   strides=1,
                                   padding=padding,
                                   activation=activation,
                                   relu_alpha=0.2,
                                   name=f"{name}_conv2d_{idx + 1}",
                                   is_upscale=True,
                                   **kwargs)
                       for idx in range(2)]
        logger.debug("Initialized %s", self.__class__.__name__)

    def __call__(self, inputs: KerasTensor) -> KerasTensor:
        """ Call the UpscaleDNY block

        Parameters
        ----------
        inputs: :class:`keras.KerasTensor`
            The input to the block

        Returns
        -------
        :class:`keras.KerasTensor`
            The output from the block
        """
        var_x = self._upsample(inputs)
        for conv in (self._convs):
            var_x = conv(var_x)
        return var_x


# << OTHER BLOCKS >>
class ResidualBlock():
    """ Residual block from dfaker.

    Parameters
    ----------
    filters: int
        The dimensionality of the output space (i.e. the number of output filters in the
        convolution)
    kernel_size: int, optional
        An integer or tuple/list of 2 integers, specifying the height and width of the 2D
        convolution window. Can be a single integer to specify the same value for all spatial
        dimensions. Default: 3
    padding: ["valid", "same"], optional
        The padding to use. Default: `"same"`
    kwargs: dict
        Any additional Keras standard layer keyword arguments to pass to the Convolutional 2D layer

    Returns
    -------
    tensor
        The output tensor from the Upscale layer
    """
    def __init__(self,
                 filters: int,
                 kernel_size: int | tuple[int, int] = 3,
                 padding: str = "same",
                 **kwargs) -> None:
        logger.debug(parse_class_init(locals()))

        self._name = _get_name(f"residual_{filters}")
        self._use_reflect_padding = cfg.reflect_padding()

        self._filters = filters
        self._kernel_size = (kernel_size,
                             kernel_size) if isinstance(kernel_size, int) else kernel_size
        self._padding = "valid" if self._use_reflect_padding else padding
        self._kwargs = kwargs

        self._layers = self._get_layers()
        self._add = layers.Add()
        self._activation = layers.LeakyReLU(negative_slope=0.2, name=f"{self._name}_leakyrelu_3")
        logger.debug("Initialized %s", self.__class__.__name__)

    def _get_layers(self) -> list[layers.Layer]:
        """ Obtain the layer chain for the block

        Returns
        -------
        list[:class:`keras.layers.Layer]
            The layers, in the correct order, to pass the tensor through
        """
        retval: list[layers.Layer] = []
        if self._use_reflect_padding:
            retval.append(ReflectionPadding2D(stride=1,
                                              kernel_size=self._kernel_size[0],
                                              name=f"{self._name}_reflectionpadding2d_0"))

        retval.append(Conv2D(self._filters,  # pyright:ignore[reportArgumentType]
                             kernel_size=self._kernel_size,
                             padding=self._padding,
                             name=f"{self._name}_conv2d_0",
                             **self._kwargs))
        retval.append(layers.LeakyReLU(negative_slope=0.2, name=f"{self._name}_leakyrelu_1"))

        if self._use_reflect_padding:
            retval.append(ReflectionPadding2D(stride=1,
                                              kernel_size=self._kernel_size[0],
                                              name=f"{self._name}_reflectionpadding2d_1"))

        kwargs = {key: val for key, val in self._kwargs.items() if key != "kernel_initializer"}
        if not cfg.conv_aware_init():
            kwargs["kernel_initializer"] = initializers.VarianceScaling(scale=0.2,
                                                                        mode="fan_in",
                                                                        distribution="uniform")
        retval.append(Conv2D(self._filters,  # pyright:ignore[reportArgumentType]
                             kernel_size=self._kernel_size,
                             padding=self._padding,
                             name=f"{self._name}_conv2d_1",
                             **kwargs))

        logger.debug("%s layers: %s", self.__class__.__name__, retval)
        return retval

    def __call__(self, inputs: KerasTensor) -> KerasTensor:
        """ Call the Faceswap Residual Block.

        Parameters
        ----------
        inputs: :class:`keras.KerasTensor`
            The input to the layer

        Returns
        -------
        :class:`keras.KerasTensor`
            The output tensor from the Upscale Layer
        """
        var_x = inputs
        for layer in self._layers:
            var_x = layer(var_x)

        var_x = self._add([var_x, inputs])
        return self._activation(var_x)


__all__ = get_module_objects(__name__)
