#!/usr/bin/env python3
""" Neural Network Blocks for faceswap.py. """

import logging

from keras.layers import Add, Concatenate, SeparableConv2D, UpSampling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation
from keras.initializers import he_uniform, VarianceScaling
from .initializers import ICNR, ConvolutionAware
from .layers import PixelShuffler, ReflectionPadding2D
from .normalization import InstanceNormalization

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class NNBlocks():
    """ Blocks that are often used for multiple models are stored here for easy access.

    This class is always brought in as ``self.blocks`` in all model plugins so that all models
    have access to them.

    The parameters passed into this class should ultimately originate from the user's training
    configuration file, rather than being hard-coded at the plugin level.

    Parameters
    ----------
    use_icnr_init: bool, Optional
        ``True`` if ICNR initialization should be used rather than the default. Default: ``False``
    use_convaware_init: bool, Optional
        ``True`` if Convolutional Aware initialization should be used rather than the default.
        Default: ``False``
    use_reflect_padding: bool, Optional
        ``True`` if Reflect Padding initialization should be used rather than the padding.
        Default: ``False``
    first_run: bool, Optional
        ``True`` if a model is being created for the first time, ``False`` if a model is being
        resumed. Used to prevent Convolutional Aware weights from being calculated when a model
        is being reloaded. Default: ``True``
    """
    def __init__(self,
                 use_icnr_init=False,
                 use_convaware_init=False,
                 use_reflect_padding=False,
                 first_run=True):
        logger.debug("Initializing %s: (use_icnr_init: %s, use_convaware_init: %s, "
                     "use_reflect_padding: %s, first_run: %s)",
                     self.__class__.__name__, use_icnr_init, use_convaware_init,
                     use_reflect_padding, first_run)
        self.names = dict()
        self.first_run = first_run
        self.use_icnr_init = use_icnr_init
        self.use_convaware_init = use_convaware_init
        self.use_reflect_padding = use_reflect_padding
        if self.use_convaware_init and self.first_run:
            logger.info("Using Convolutional Aware Initialization. Model generation will take a "
                        "few minutes...")
        logger.debug("Initialized %s", self.__class__.__name__)

    def _get_name(self, name):
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
        self.names[name] = self.names.setdefault(name, -1) + 1
        name = "{}_{}".format(name, self.names[name])
        logger.debug("Generating block name: %s", name)
        return name

    def _set_default_initializer(self, kwargs):
        """ Sets the default initializer for convolution 2D and Seperable convolution 2D layers
            to Convolutional Aware or he_uniform.

            if a specific initializer has been passed in from the model plugin, then the specified
            initializer will be used rather than the default.

            Parameters
            ----------
            kwargs: dict
                The keyword arguments for the current layer

            Returns
            -------
            dict
                The keyword arguments for the current layer with the initializer updated to
                the select default value
            """
        if "kernel_initializer" in kwargs:
            logger.debug("Using model specified initializer: %s", kwargs["kernel_initializer"])
            return kwargs
        if self.use_convaware_init:
            default = ConvolutionAware()
            if self.first_run:
                # Indicate the Convolutional Aware should be calculated on first run
                default._init = True  # pylint:disable=protected-access
        else:
            default = he_uniform()
        if kwargs.get("kernel_initializer", None) != default:
            kwargs["kernel_initializer"] = default
            logger.debug("Set default kernel_initializer to: %s", kwargs["kernel_initializer"])
        return kwargs

    @staticmethod
    def _switch_kernel_initializer(kwargs, initializer):
        """ Switch the initializer in the given kwargs to the given initializer and return the
        previous initializer to caller.

        For residual blocks and up-scaling, user selected initializer methods should replace those
        set by the model. This method updates the initializer for the layer, and returns the
        original initializer so that it can be set back to the layer's key word arguments for
        subsequent layers where the initializer should not be switched.

        Parameters
        ----------
        kwargs: dict
            The keyword arguments for the current layer
        initializer: keras or faceswap initializer class
            The initializer that should replace the current initializer that exists in keyword
            arguments

        Returns
        -------
        keras or faceswap initializer class
            The original initializer that existed in the given keyword arguments
        """
        original = kwargs.get("kernel_initializer", None)
        kwargs["kernel_initializer"] = initializer
        logger.debug("Switched kernel_initializer from %s to %s", original, initializer)
        return original

    def conv2d(self, input_tensor, filters, kernel_size, strides=(1, 1), padding="same", **kwargs):
        """ A standard Convolution 2D layer with correct initialization.

        This layer creates a convolution kernel that is convolved with the layer input to produce
        a tensor of outputs.

        Parameters
        ----------
        input_tensor: tensor
            The input tensor to the layer
        filters: int
            The dimensionality of the output space (i.e. the number of output filters in the
            convolution)
        kernel_size: int
            An integer or tuple/list of 2 integers, specifying the height and width of the 2D
            convolution window. Can be a single integer to specify the same value for all spatial
            dimensions
        strides: tuple, optional
            An integer or tuple/list of 2 integers, specifying the strides of the convolution along
            the height and width. Can be a single integer to specify the same value for all spatial
            dimensions. Default: `(1, 1)`
        padding: ["valid", "same"], optional
            The padding to use. Default: `"same"`
        kwargs: dict
            Any additional Keras standard layer keyword arguments

        Returns
        -------
        tensor
            The output tensor from the Convolution 2D Layer
        """
        logger.debug("input_tensor: %s, filters: %s, kernel_size: %s, strides: %s, padding: %s, "
                     "kwargs: %s)", input_tensor, filters, kernel_size, strides, padding, kwargs)
        if kwargs.get("name", None) is None:
            kwargs["name"] = self._get_name("conv2d_{}".format(input_tensor.shape[1]))
        kwargs = self._set_default_initializer(kwargs)
        var_x = Conv2D(filters, kernel_size,
                       strides=strides,
                       padding=padding,
                       **kwargs)(input_tensor)
        return var_x

    # <<< Original Model Blocks >>> #
    def conv(self, input_tensor, filters, kernel_size=5, strides=2, padding="same",
             use_instance_norm=False, res_block_follows=False, **kwargs):
        """ A standard Convolution 2D layer which applies user specified configuration to the
        layer.

        Adds reflection padding if it has been selected by the user, and other post-processing
        if requested by the plugin.

        Parameters
        ----------
        input_tensor: tensor
            The input tensor to the layer
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
        padding: ["valid", "same"], optional
            The padding to use. Default: `"same"`
        use_instance_norm: bool, optional
            ``True`` if instance normalization should be applied after the convolutional layer.
            Default: ``False``
        res_block_follows: bool, optional
            If a residual block will follow this layer, then this should be set to `True` to add
            a leaky ReLu after the convolutional layer. Default: ``False``
        kwargs: dict
            Any additional Keras standard layer keyword arguments

        Returns
        -------
        tensor
            The output tensor from the Convolution 2D Layer
        """
        logger.debug("input_tensor: %s, filters: %s, kernel_size: %s, strides: %s, "
                     "use_instance_norm: %s, kwargs: %s)", input_tensor, filters, kernel_size,
                     strides, use_instance_norm, kwargs)
        name = self._get_name("conv_{}".format(input_tensor.shape[1]))
        if self.use_reflect_padding:
            input_tensor = ReflectionPadding2D(
                stride=strides,
                kernel_size=kernel_size,
                name="{}_reflectionpadding2d".format(name))(input_tensor)
            padding = "valid"
        var_x = self.conv2d(input_tensor, filters,
                            kernel_size=kernel_size,
                            strides=strides,
                            padding=padding,
                            name="{}_conv2d".format(name),
                            **kwargs)
        if use_instance_norm:
            var_x = InstanceNormalization(name="{}_instancenorm".format(name))(var_x)
        if not res_block_follows:
            var_x = LeakyReLU(0.1, name="{}_leakyrelu".format(name))(var_x)
        return var_x

    def upscale(self, input_tensor, filters, kernel_size=3, padding="same",
                use_instance_norm=False, res_block_follows=False, scale_factor=2, **kwargs):
        """ An upscale layer for sub-pixel up-scaling.

        Adds reflection padding if it has been selected by the user, and other post-processing
        if requested by the plugin.

        Parameters
        ----------
        input_tensor: tensor
            The input tensor to the layer
        filters: int
            The dimensionality of the output space (i.e. the number of output filters in the
            convolution)
        kernel_size: int, optional
            An integer or tuple/list of 2 integers, specifying the height and width of the 2D
            convolution window. Can be a single integer to specify the same value for all spatial
            dimensions. Default: 3
        padding: ["valid", "same"], optional
            The padding to use. Default: `"same"`
        use_instance_norm: bool, optional
            ``True`` if instance normalization should be applied after the convolutional layer.
            Default: ``False``
        res_block_follows: bool, optional
            If a residual block will follow this layer, then this should be set to `True` to add
            a leaky ReLu after the convolutional layer. Default: ``False``
        scale_factor: int, optional
            The amount to upscale the image. Default: `2`
        kwargs: dict
            Any additional Keras standard layer keyword arguments

        Returns
        -------
        tensor
            The output tensor from the Upscale layer
        """
        logger.debug("input_tensor: %s, filters: %s, kernel_size: %s, use_instance_norm: %s, "
                     "kwargs: %s)", input_tensor, filters, kernel_size, use_instance_norm, kwargs)
        name = self._get_name("upscale_{}".format(input_tensor.shape[1]))
        if self.use_reflect_padding:
            input_tensor = ReflectionPadding2D(
                stride=1,
                kernel_size=kernel_size,
                name="{}_reflectionpadding2d".format(name))(input_tensor)
            padding = "valid"
        kwargs = self._set_default_initializer(kwargs)
        if self.use_icnr_init:
            original_init = self._switch_kernel_initializer(
                kwargs,
                ICNR(initializer=kwargs["kernel_initializer"]))
        var_x = self.conv2d(input_tensor, filters * scale_factor * scale_factor,
                            kernel_size=kernel_size,
                            padding=padding,
                            name="{}_conv2d".format(name),
                            **kwargs)
        if self.use_icnr_init:
            self._switch_kernel_initializer(kwargs, original_init)
        if use_instance_norm:
            var_x = InstanceNormalization(name="{}_instancenorm".format(name))(var_x)
        if not res_block_follows:
            var_x = LeakyReLU(0.1, name="{}_leakyrelu".format(name))(var_x)
        var_x = PixelShuffler(name="{}_pixelshuffler".format(name), size=scale_factor)(var_x)
        return var_x

    # <<< DLight Model Blocks >>> #
    def upscale2x(self, input_tensor, filters,
                  kernel_size=3, padding="same", interpolation="bilinear", res_block_follows=False,
                  sr_ratio=0.5, scale_factor=2, fast=False, **kwargs):
        """ Custom hybrid upscale layer for sub-pixel up-scaling.

        Most of up-scaling is approximating lighting gradients which can be accurately achieved
        using linear fitting. This layer attempts to improve memory consumption by splitting
        with bilinear and convolutional layers so that the sub-pixel update will get details
        whilst the bilinear filter will get lighting.

        Adds reflection padding if it has been selected by the user, and other post-processing
        if requested by the plugin.

        Parameters
        ----------
        input_tensor: tensor
            The input tensor to the layer
        filters: int
            The dimensionality of the output space (i.e. the number of output filters in the
            convolution)
        kernel_size: int, optional
            An integer or tuple/list of 2 integers, specifying the height and width of the 2D
            convolution window. Can be a single integer to specify the same value for all spatial
            dimensions. Default: 3
        padding: ["valid", "same"], optional
            The padding to use. Default: `"same"`
        interpolation: ["nearest", "bilinear"], optional
            Interpolation to use for up-sampling. Default: `"bilinear"`
        res_block_follows: bool, optional
            If a residual block will follow this layer, then this should be set to `True` to add
            a leaky ReLu after the convolutional layer. Default: ``False``
        scale_factor: int, optional
            The amount to upscale the image. Default: `2`
        sr_ratio: float, optional
            The proportion of super resolution (pixel shuffler) filters to use. Non-fast mode only.
            Default: `0.5`
        kwargs: dict
            Any additional Keras standard layer keyword arguments
        fast: bool, optional
            Use a faster up-scaling method that may appear more rugged. Default: ``False``

        Returns
        -------
        tensor
            The output tensor from the Upscale layer
        """
        name = self._get_name("upscale2x_{}".format("fast" if fast else "hyb"))
        var_x = input_tensor
        if not fast:
            sr_filters = int(filters * sr_ratio)
            filters = filters - sr_filters
            var_x_sr = self.upscale(var_x, filters,
                                    kernel_size=kernel_size,
                                    padding=padding,
                                    scale_factor=scale_factor,
                                    res_block_follows=res_block_follows,
                                    **kwargs)

        if fast or (not fast and filters > 0):
            var_x2 = self.conv2d(var_x, filters,
                                 kernel_size=3,
                                 padding=padding,
                                 name="{}_conv2d".format(name),
                                 **kwargs)
            var_x2 = UpSampling2D(size=(scale_factor, scale_factor),
                                  interpolation=interpolation,
                                  name="{}_upsampling2D".format(name))(var_x2)
            if fast:
                var_x1 = self.upscale(var_x, filters,
                                      kernel_size=kernel_size,
                                      padding=padding,
                                      scale_factor=scale_factor,
                                      res_block_follows=res_block_follows, **kwargs)
                var_x = Add()([var_x2, var_x1])
            else:
                var_x = Concatenate(name="{}_concatenate".format(name))([var_x_sr, var_x2])
        else:
            var_x = var_x_sr
        return var_x

    # <<< DFaker Model Blocks >>> #
    def res_block(self, input_tensor, filters, kernel_size=3, padding="same", **kwargs):
        """ Residual block.

        Parameters
        ----------
        input_tensor: tensor
            The input tensor to the layer
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
            Any additional Keras standard layer keyword arguments

        Returns
        -------
        tensor
            The output tensor from the Upscale layer
        """
        logger.debug("input_tensor: %s, filters: %s, kernel_size: %s, kwargs: %s)",
                     input_tensor, filters, kernel_size, kwargs)
        name = self._get_name("residual_{}".format(input_tensor.shape[1]))
        var_x = LeakyReLU(alpha=0.2, name="{}_leakyrelu_0".format(name))(input_tensor)
        if self.use_reflect_padding:
            var_x = ReflectionPadding2D(stride=1,
                                        kernel_size=kernel_size,
                                        name="{}_reflectionpadding2d_0".format(name))(var_x)
            padding = "valid"
        var_x = self.conv2d(var_x, filters,
                            kernel_size=kernel_size,
                            padding=padding,
                            name="{}_conv2d_0".format(name),
                            **kwargs)
        var_x = LeakyReLU(alpha=0.2, name="{}_leakyrelu_1".format(name))(var_x)
        if self.use_reflect_padding:
            var_x = ReflectionPadding2D(stride=1,
                                        kernel_size=kernel_size,
                                        name="{}_reflectionpadding2d_1".format(name))(var_x)
            padding = "valid"
        if not self.use_convaware_init:
            original_init = self._switch_kernel_initializer(kwargs, VarianceScaling(
                scale=0.2,
                mode="fan_in",
                distribution="uniform"))
        var_x = self.conv2d(var_x, filters,
                            kernel_size=kernel_size,
                            padding=padding,
                            **kwargs)
        if not self.use_convaware_init:
            self._switch_kernel_initializer(kwargs, original_init)
        var_x = Add()([var_x, input_tensor])
        var_x = LeakyReLU(alpha=0.2, name="{}_leakyrelu_3".format(name))(var_x)
        return var_x

    # <<< Unbalanced Model Blocks >>> #
    def conv_sep(self, input_tensor, filters, kernel_size=5, strides=2, **kwargs):
        """ Seperable Convolution Layer.

        Parameters
        ----------
        input_tensor: tensor
            The input tensor to the layer
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
            Any additional Keras standard layer keyword arguments

        Returns
        -------
        tensor
            The output tensor from the Upscale layer
        """
        logger.debug("input_tensor: %s, filters: %s, kernel_size: %s, strides: %s, kwargs: %s)",
                     input_tensor, filters, kernel_size, strides, kwargs)
        name = self._get_name("separableconv2d_{}".format(input_tensor.shape[1]))
        kwargs = self._set_default_initializer(kwargs)
        var_x = SeparableConv2D(filters,
                                kernel_size=kernel_size,
                                strides=strides,
                                padding="same",
                                name="{}_seperableconv2d".format(name),
                                **kwargs)(input_tensor)
        var_x = Activation("relu", name="{}_relu".format(name))(var_x)
        return var_x
