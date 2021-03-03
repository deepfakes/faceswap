#!/usr/bin/env python3
""" Neural Network Blocks for faceswap.py. """

import logging

from keras.layers import (Activation, Add, BatchNormalization, Concatenate, Conv2D as KConv2D,
                          Conv2DTranspose, DepthwiseConv2D as KDepthwiseConv2d, LeakyReLU, PReLU,
                          SeparableConv2D, UpSampling2D)
from keras.initializers import he_uniform, VarianceScaling

from .initializers import ICNR, ConvolutionAware
from .layers import PixelShuffler, ReflectionPadding2D, Swish, KResizeImages
from .normalization import InstanceNormalization

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


_CONFIG = dict()
_NAMES = dict()


def set_config(configuration):
    """ Set the global configuration parameters from the user's config file.

    These options are used when creating layers for new models.

    Parameters
    ----------
    configuration: dict
        The configuration options that exist in the training configuration files that pertain
        specifically to Custom Faceswap Layers. The keys should be: `icnr_init`, `conv_aware_init`
        and 'reflect_padding'
     """
    global _CONFIG  # pylint:disable=global-statement
    _CONFIG = configuration
    logger.debug("Set NNBlock configuration to: %s", _CONFIG)


def _get_name(name):
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
    global _NAMES  # pylint:disable=global-statement
    _NAMES[name] = _NAMES.setdefault(name, -1) + 1
    name = "{}_{}".format(name, _NAMES[name])
    logger.debug("Generating block name: %s", name)
    return name


#  << CONVOLUTIONS >>
def _get_default_initializer(initializer):
    """ Returns a default initializer of Convolutional Aware or he_uniform for convolutional
    layers.

    Parameters
    ----------
    initializer: :class:`keras.initializers.Initializer` or None
        The initializer that has been passed into the model. If this value is ``None`` then a
        default initializer will be set to 'he_uniform'. If Convolutional Aware initialization
        has been enabled, then any passed through initializer will be replaced with the
        Convolutional Aware initializer.

    Returns
    -------
    :class:`keras.initializers.Initializer`
        The kernel initializer to use for this convolutional layer. Either the original given
        initializer, he_uniform or convolutional aware (if selected in config options)
    """
    if _CONFIG["conv_aware_init"]:
        retval = ConvolutionAware()
    elif initializer is None:
        retval = he_uniform()
    else:
        retval = initializer
        logger.debug("Using model supplied initializer: %s", retval)
    logger.debug("Set default kernel_initializer: (original: %s current: %s)", initializer, retval)

    return retval


class Conv2D(KConv2D):  # pylint:disable=too-few-public-methods
    """ A standard Keras Convolution 2D layer with parameters updated to be more appropriate for
    Faceswap architecture.

    Parameters are the same, with the same defaults, as a standard :class:`keras.layers.Conv2D`
    except where listed below. The default initializer is updated to `he_uniform` or `convolutional
    aware` based on user configuration settings.

    Parameters
    ----------
    padding: str, optional
        One of `"valid"` or `"same"` (case-insensitive). Default: `"same"`. Note that `"same"` is
        slightly inconsistent across backends with `strides` != 1, as described
        `here <https://github.com/keras-team/keras/pull/9473#issuecomment-372166860/>`_.
    check_icnr_init: `bool`, optional
        ``True`` if the user configuration options should be checked to apply ICNR initialization
        to the layer. This should only be passed in from :class:`UpscaleBlock` layers.
        Default: ``False``
    """
    def __init__(self, *args, padding="same", check_icnr_init=False, **kwargs):
        if kwargs.get("name", None) is None:
            filters = kwargs["filters"] if "filters" in kwargs else args[0]
            kwargs["name"] = _get_name("conv2d_{}".format(filters))
        initializer = _get_default_initializer(kwargs.pop("kernel_initializer", None))
        if check_icnr_init and _CONFIG["icnr_init"]:
            initializer = ICNR(initializer=initializer)
            logger.debug("Using ICNR Initializer: %s", initializer)
        super().__init__(*args, padding=padding, kernel_initializer=initializer, **kwargs)


class DepthwiseConv2D(KDepthwiseConv2d):  # pylint:disable=too-few-public-methods
    """ A standard Keras Depthwise Convolution 2D layer with parameters updated to be more
    appropriate for Faceswap architecture.

    Parameters are the same, with the same defaults, as a standard
    :class:`keras.layers.DepthwiseConv2D` except where listed below. The default initializer is
    updated to `he_uniform` or `convolutional aware` based on user configuration settings.

    Parameters
    ----------
    padding: str, optional
        One of `"valid"` or `"same"` (case-insensitive). Default: `"same"`. Note that `"same"` is
        slightly inconsistent across backends with `strides` != 1, as described
        `here <https://github.com/keras-team/keras/pull/9473#issuecomment-372166860/>`_.
    check_icnr_init: `bool`, optional
        ``True`` if the user configuration options should be checked to apply ICNR initialization
        to the layer. This should only be passed in from :class:`UpscaleBlock` layers.
        Default: ``False``
    """
    def __init__(self, *args, padding="same", check_icnr_init=False, **kwargs):
        if kwargs.get("name", None) is None:
            kwargs["name"] = _get_name("dwconv2d")
        initializer = _get_default_initializer(kwargs.pop("depthwise_initializer", None))
        if check_icnr_init and _CONFIG["icnr_init"]:
            initializer = ICNR(initializer=initializer)
            logger.debug("Using ICNR Initializer: %s", initializer)
        super().__init__(*args, padding=padding, depthwise_initializer=initializer, **kwargs)


class Conv2DOutput():  # pylint:disable=too-few-public-methods
    """ A Convolution 2D layer that separates out the activation layer to explicitly set the data
    type on the activation to float 32 to fully support mixed precision training.

    The Convolution 2D layer uses default parameters to be more appropriate for Faceswap
    architecture.

    Parameters are the same, with the same defaults, as a standard :class:`keras.layers.Conv2D`
    except where listed below. The default initializer is updated to he_uniform or convolutional
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
    def __init__(self, filters, kernel_size, activation="sigmoid", padding="same", **kwargs):
        self._name = kwargs.pop("name") if "name" in kwargs else _get_name(
            "conv_output_{}".format(filters))
        self._filters = filters
        self._kernel_size = kernel_size
        self._activation = activation
        self._padding = padding
        self._kwargs = kwargs

    def __call__(self, inputs):
        """ Call the Faceswap Convolutional Output Layer.

        Parameters
        ----------
        inputs: Tensor
            The input to the layer

        Returns
        -------
        Tensor
            The output tensor from the Convolution 2D Layer
        """
        var_x = Conv2D(self._filters,
                       self._kernel_size,
                       padding=self._padding,
                       name="{}_conv2d".format(self._name),
                       **self._kwargs)(inputs)
        var_x = Activation(self._activation, dtype="float32", name=self._name)(var_x)
        return var_x


class Conv2DBlock():  # pylint:disable=too-few-public-methods
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
    kwargs: dict
        Any additional Keras standard layer keyword arguments to pass to the Convolutional 2D layer
    """
    def __init__(self,
                 filters,
                 kernel_size=5,
                 strides=2,
                 padding="same",
                 normalization=None,
                 activation="leakyrelu",
                 use_depthwise=False,
                 **kwargs):
        self._name = kwargs.pop("name") if "name" in kwargs else _get_name(
            "conv_{}".format(filters))

        logger.debug("name: %s, filters: %s, kernel_size: %s, strides: %s, padding: %s, "
                     "normalization: %s, activation: %s, use_depthwise: %s, kwargs: %s)",
                     self._name, filters, kernel_size, strides, padding, normalization,
                     activation, use_depthwise, kwargs)

        self._use_reflect_padding = _CONFIG["reflect_padding"]

        self._args = (kernel_size, ) if use_depthwise else (filters, kernel_size)
        self._strides = strides
        self._padding = "valid" if self._use_reflect_padding else padding
        self._kwargs = kwargs
        self._normalization = None if not normalization else normalization.lower()
        self._activation = None if not activation else activation.lower()
        self._use_depthwise = use_depthwise

        self._assert_arguments()

    def _assert_arguments(self):
        """ Validate the given arguments. """
        assert self._normalization in ("batch", "instance", None), (
            "normalization should be 'batch', 'instance' or None")
        assert self._activation in ("leakyrelu", "swish", "prelu", None), (
            "activation should be 'leakyrelu', 'prelu', 'swish' or None")

    def __call__(self, inputs):
        """ Call the Faceswap Convolutional Layer.

        Parameters
        ----------
        inputs: Tensor
            The input to the layer

        Returns
        -------
        Tensor
            The output tensor from the Convolution 2D Layer
        """
        if self._use_reflect_padding:
            inputs = ReflectionPadding2D(stride=self._strides,
                                         kernel_size=self._args[-1],
                                         name="{}_reflectionpadding2d".format(self._name))(inputs)
        conv = DepthwiseConv2D if self._use_depthwise else Conv2D
        var_x = conv(*self._args,
                     strides=self._strides,
                     padding=self._padding,
                     name="{}_{}conv2d".format(self._name, "dw" if self._use_depthwise else ""),
                     **self._kwargs)(inputs)
        # normalization
        if self._normalization == "instance":
            var_x = InstanceNormalization(name="{}_instancenorm".format(self._name))(var_x)
        if self._normalization == "batch":
            var_x = BatchNormalization(axis=3, name="{}_batchnorm".format(self._name))(var_x)

        # activation
        if self._activation == "leakyrelu":
            var_x = LeakyReLU(0.1, name="{}_leakyrelu".format(self._name))(var_x)
        if self._activation == "swish":
            var_x = Swish(name="{}_swish".format(self._name))(var_x)
        if self._activation == "prelu":
            var_x = PReLU(name="{}_prelu".format(self._name))(var_x)

        return var_x


class SeparableConv2DBlock():  # pylint:disable=too-few-public-methods
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
    def __init__(self, filters, kernel_size=5, strides=2, **kwargs):
        self._name = _get_name("separableconv2d_{}".format(filters))
        logger.debug("name: %s, filters: %s, kernel_size: %s, strides: %s, kwargs: %s)",
                     self._name, filters, kernel_size, strides, kwargs)

        self._filters = filters
        self._kernel_size = kernel_size
        self._strides = strides

        initializer = _get_default_initializer(kwargs.pop("kernel_initializer", None))
        kwargs["kernel_initializer"] = initializer
        self._kwargs = kwargs

    def __call__(self, inputs):
        """ Call the Faceswap Separable Convolutional 2D Block.

        Parameters
        ----------
        inputs: Tensor
            The input to the layer

        Returns
        -------
        Tensor
            The output tensor from the Upscale Layer
        """
        var_x = SeparableConv2D(self._filters,
                                kernel_size=self._kernel_size,
                                strides=self._strides,
                                padding="same",
                                name="{}_seperableconv2d".format(self._name),
                                **self._kwargs)(inputs)
        var_x = Activation("relu", name="{}_relu".format(self._name))(var_x)
        return var_x


#  << UPSCALING >>

class UpscaleBlock():  # pylint:disable=too-few-public-methods
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
                 filters,
                 kernel_size=3,
                 padding="same",
                 scale_factor=2,
                 normalization=None,
                 activation="leakyrelu",
                 **kwargs):
        self._name = _get_name("upscale_{}".format(filters))
        logger.debug("name: %s. filters: %s, kernel_size: %s, padding: %s, scale_factor: %s, "
                     "normalization: %s, activation: %s, kwargs: %s)",
                     self._name, filters, kernel_size, padding, scale_factor, normalization,
                     activation, kwargs)

        self._filters = filters
        self._kernel_size = kernel_size
        self._padding = padding
        self._scale_factor = scale_factor
        self._normalization = normalization
        self._activation = activation
        self._kwargs = kwargs

    def __call__(self, inputs):
        """ Call the Faceswap Convolutional Layer.

        Parameters
        ----------
        inputs: Tensor
            The input to the layer

        Returns
        -------
        Tensor
            The output tensor from the Upscale Layer
        """
        var_x = Conv2DBlock(self._filters * self._scale_factor * self._scale_factor,
                            self._kernel_size,
                            strides=(1, 1),
                            padding=self._padding,
                            normalization=self._normalization,
                            activation=self._activation,
                            name="{}_conv2d".format(self._name),
                            check_icnr_init=_CONFIG["icnr_init"],
                            **self._kwargs)(inputs)
        var_x = PixelShuffler(name="{}_pixelshuffler".format(self._name),
                              size=self._scale_factor)(var_x)
        return var_x


class Upscale2xBlock():  # pylint:disable=too-few-public-methods
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
    def __init__(self, filters, kernel_size=3, padding="same", activation="leakyrelu",
                 interpolation="bilinear", sr_ratio=0.5, scale_factor=2, fast=False, **kwargs):
        self._name = _get_name("upscale2x_{}_{}".format(filters, "fast" if fast else "hyb"))

        self._fast = fast
        self._filters = filters if self._fast else filters - int(filters * sr_ratio)
        self._kernel_size = kernel_size
        self._padding = padding
        self._interpolation = interpolation
        self._activation = activation
        self._scale_factor = scale_factor
        self._kwargs = kwargs

    def __call__(self, inputs):
        """ Call the Faceswap Upscale 2x Layer.

        Parameters
        ----------
        inputs: Tensor
            The input to the layer

        Returns
        -------
        Tensor
            The output tensor from the Upscale Layer
        """
        var_x = inputs
        if not self._fast:
            var_x_sr = UpscaleBlock(self._filters,
                                    kernel_size=self._kernel_size,
                                    padding=self._padding,
                                    scale_factor=self._scale_factor,
                                    activation=self._activation,
                                    **self._kwargs)(var_x)
        if self._fast or (not self._fast and self._filters > 0):
            var_x2 = Conv2D(self._filters, 3,
                            padding=self._padding,
                            name="{}_conv2d".format(self._name),
                            **self._kwargs)(var_x)
            var_x2 = UpSampling2D(size=(self._scale_factor, self._scale_factor),
                                  interpolation=self._interpolation,
                                  name="{}_upsampling2D".format(self._name))(var_x2)
            if self._fast:
                var_x1 = UpscaleBlock(self._filters,
                                      kernel_size=self._kernel_size,
                                      padding=self._padding,
                                      scale_factor=self._scale_factor,
                                      activation=self._activation,
                                      **self._kwargs)(var_x)
                var_x = Add()([var_x2, var_x1])
            else:
                var_x = Concatenate(name="{}_concatenate".format(self._name))([var_x_sr, var_x2])
        else:
            var_x = var_x_sr
        return var_x


class UpscaleResizeImagesBlock():  # pylint:disable=too-few-public-methods
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
    def __init__(self, filters, kernel_size=3, padding="same", activation="leakyrelu",
                 scale_factor=2, interpolation="bilinear"):
        self._name = _get_name("upscale_ri_{}".format(filters))
        self._interpolation = interpolation
        self._size = scale_factor
        self._filters = filters
        self._kernel_size = kernel_size
        self._padding = padding
        self._activation = activation

    def __call__(self, inputs):
        """ Call the Faceswap Resize Images Layer.

        Parameters
        ----------
        inputs: Tensor
            The input to the layer

        Returns
        -------
        Tensor
            The output tensor from the Upscale Layer
        """
        var_x = inputs

        var_x_sr = KResizeImages(size=self._size,
                                 interpolation=self._interpolation,
                                 name="{}_resize".format(self._name))(var_x)
        var_x_sr = Conv2D(self._filters, self._kernel_size,
                          strides=1,
                          padding=self._padding,
                          name="{}_conv".format(self._name))(var_x_sr)
        var_x_us = Conv2DTranspose(self._filters, 3,
                                   strides=2,
                                   padding=self._padding,
                                   name="{}_convtrans".format(self._name))(var_x)
        var_x = Add()([var_x_sr, var_x_us])

        if self._activation == "leakyrelu":
            var_x = LeakyReLU(0.2, name="{}_leakyrelu".format(self._name))(var_x)
        if self._activation == "swish":
            var_x = Swish(name="{}_swish".format(self._name))(var_x)
        if self._activation == "prelu":
            var_x = PReLU(name="{}_prelu".format(self._name))(var_x)
        return var_x


# << OTHER BLOCKS >>
class ResidualBlock():  # pylint:disable=too-few-public-methods
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
    def __init__(self, filters, kernel_size=3, padding="same", **kwargs):
        self._name = _get_name("residual_{}".format(filters))
        logger.debug("name: %s, filters: %s, kernel_size: %s, padding: %s, kwargs: %s)",
                     self._name, filters, kernel_size, padding, kwargs)
        self._use_reflect_padding = _CONFIG["reflect_padding"]

        self._filters = filters
        self._kernel_size = kernel_size
        self._padding = "valid" if self._use_reflect_padding else padding
        self._kwargs = kwargs

    def __call__(self, inputs):
        """ Call the Faceswap Residual Block.

        Parameters
        ----------
        inputs: Tensor
            The input to the layer

        Returns
        -------
        Tensor
            The output tensor from the Upscale Layer
        """
        var_x = inputs
        if self._use_reflect_padding:
            var_x = ReflectionPadding2D(stride=1,
                                        kernel_size=self._kernel_size,
                                        name="{}_reflectionpadding2d_0".format(self._name))(var_x)
        var_x = Conv2D(self._filters,
                       kernel_size=self._kernel_size,
                       padding=self._padding,
                       name="{}_conv2d_0".format(self._name),
                       **self._kwargs)(var_x)
        var_x = LeakyReLU(alpha=0.2, name="{}_leakyrelu_1".format(self._name))(var_x)
        if self._use_reflect_padding:
            var_x = ReflectionPadding2D(stride=1,
                                        kernel_size=self._kernel_size,
                                        name="{}_reflectionpadding2d_1".format(self._name))(var_x)

        kwargs = {key: val for key, val in self._kwargs.items() if key != "kernel_initializer"}
        if not _CONFIG["conv_aware_init"]:
            kwargs["kernel_initializer"] = VarianceScaling(scale=0.2,
                                                           mode="fan_in",
                                                           distribution="uniform")
        var_x = Conv2D(self._filters,
                       kernel_size=self._kernel_size,
                       padding=self._padding,
                       name="{}_conv2d_1".format(self._name),
                       **kwargs)(var_x)

        var_x = Add()([var_x, inputs])
        var_x = LeakyReLU(alpha=0.2, name="{}_leakyrelu_3".format(self._name))(var_x)
        return var_x
