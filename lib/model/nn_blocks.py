#!/usr/bin/env python3
""" Neural Network Blocks for faceswap.py. """

import logging

from keras.layers import Add, SeparableConv2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation
from keras.initializers import he_uniform, VarianceScaling
from .initializers import ICNR, ConvolutionAware
from .layers import PixelShuffler, SubPixelUpscaling, ReflectionPadding2D
from .normalization import InstanceNormalization

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class NNBlocks():
    """ Blocks to use for creating models """
    def __init__(self, use_subpixel=False, use_icnr_init=False, use_convaware_init=False,
                 use_reflect_padding=False, first_run=True):
        logger.debug("Initializing %s: (use_subpixel: %s, use_icnr_init: %s, use_convaware_init: "
                     "%s, use_reflect_padding: %s, first_run: %s)",
                     self.__class__.__name__, use_subpixel, use_icnr_init, use_convaware_init,
                     use_reflect_padding, first_run)
        self.names = dict()
        self.first_run = first_run
        self.use_subpixel = use_subpixel
        self.use_icnr_init = use_icnr_init
        self.use_convaware_init = use_convaware_init
        self.use_reflect_padding = use_reflect_padding
        if self.use_convaware_init and self.first_run:
            logger.info("Using Convolutional Aware Initialization. Model generation will take a "
                        "few minutes...")
        logger.debug("Initialized %s", self.__class__.__name__)

    def get_name(self, name):
        """ Return unique layer name for requested block """
        self.names[name] = self.names.setdefault(name, -1) + 1
        name = "{}_{}".format(name, self.names[name])
        logger.debug("Generating block name: %s", name)
        return name

    def set_default_initializer(self, kwargs):
        """ Sets the default initializer for conv2D and Seperable conv2D layers
            to conv_aware or he_uniform().
            if a specific initializer has been passed in then the specified initializer
            will be used rather than the default """
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
    def switch_kernel_initializer(kwargs, initializer):
        """ Switch the initializer in the given kwargs to the given initializer
            and return the previous initializer to caller """
        original = kwargs.get("kernel_initializer", None)
        kwargs["kernel_initializer"] = initializer
        logger.debug("Switched kernel_initializer from %s to %s", original, initializer)
        return original

    def conv2d(self, inp, filters, kernel_size, strides=(1, 1), padding="same", **kwargs):
        """ A standard conv2D layer with correct initialization """
        logger.debug("inp: %s, filters: %s, kernel_size: %s, strides: %s, padding: %s, "
                     "kwargs: %s)", inp, filters, kernel_size, strides, padding, kwargs)
        if kwargs.get("name", None) is None:
            kwargs["name"] = self.get_name("conv2d_{}".format(inp.shape[1]))
        kwargs = self.set_default_initializer(kwargs)
        var_x = Conv2D(filters, kernel_size,
                       strides=strides,
                       padding=padding,
                       **kwargs)(inp)
        return var_x

    # <<< Original Model Blocks >>> #
    def conv(self, inp, filters, kernel_size=5, strides=2, padding="same",
             use_instance_norm=False, res_block_follows=False, **kwargs):
        """ Convolution Layer"""
        logger.debug("inp: %s, filters: %s, kernel_size: %s, strides: %s, use_instance_norm: %s, "
                     "kwargs: %s)", inp, filters, kernel_size, strides, use_instance_norm, kwargs)
        name = self.get_name("conv_{}".format(inp.shape[1]))
        if self.use_reflect_padding:
            inp = ReflectionPadding2D(stride=strides,
                                      kernel_size=kernel_size,
                                      name="{}_reflectionpadding2d".format(name))(inp)
            padding = "valid"
        var_x = self.conv2d(inp, filters,
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

    def upscale(self, inp, filters, kernel_size=3, padding="same",
                use_instance_norm=False, res_block_follows=False, scale_factor=2, **kwargs):
        """ Upscale Layer """
        logger.debug("inp: %s, filters: %s, kernel_size: %s, use_instance_norm: %s, kwargs: %s)",
                     inp, filters, kernel_size, use_instance_norm, kwargs)
        name = self.get_name("upscale_{}".format(inp.shape[1]))
        if self.use_reflect_padding:
            inp = ReflectionPadding2D(stride=1,
                                      kernel_size=kernel_size,
                                      name="{}_reflectionpadding2d".format(name))(inp)
            padding = "valid"
        kwargs = self.set_default_initializer(kwargs)
        if self.use_icnr_init:
            original_init = self.switch_kernel_initializer(
                kwargs,
                ICNR(initializer=kwargs["kernel_initializer"]))
        var_x = self.conv2d(inp, filters * scale_factor * scale_factor,
                            kernel_size=kernel_size,
                            padding=padding,
                            name="{}_conv2d".format(name),
                            **kwargs)
        if self.use_icnr_init:
            self.switch_kernel_initializer(kwargs, original_init)
        if use_instance_norm:
            var_x = InstanceNormalization(name="{}_instancenorm".format(name))(var_x)
        if not res_block_follows:
            var_x = LeakyReLU(0.1, name="{}_leakyrelu".format(name))(var_x)
        if self.use_subpixel:
            var_x = SubPixelUpscaling(name="{}_subpixel".format(name),
                                      scale_factor=scale_factor)(var_x)
        else:
            var_x = PixelShuffler(name="{}_pixelshuffler".format(name), size=scale_factor)(var_x)
        return var_x

    # <<< DFaker Model Blocks >>> #
    def res_block(self, inp, filters, kernel_size=3, padding="same", **kwargs):
        """ Residual block """
        logger.debug("inp: %s, filters: %s, kernel_size: %s, kwargs: %s)",
                     inp, filters, kernel_size, kwargs)
        name = self.get_name("residual_{}".format(inp.shape[1]))
        var_x = LeakyReLU(alpha=0.2, name="{}_leakyrelu_0".format(name))(inp)
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
            original_init = self.switch_kernel_initializer(kwargs, VarianceScaling(
                scale=0.2,
                mode="fan_in",
                distribution="uniform"))
        var_x = self.conv2d(var_x, filters,
                            kernel_size=kernel_size,
                            padding=padding,
                            **kwargs)
        if not self.use_convaware_init:
            self.switch_kernel_initializer(kwargs, original_init)
        var_x = Add()([var_x, inp])
        var_x = LeakyReLU(alpha=0.2, name="{}_leakyrelu_3".format(name))(var_x)
        return var_x

    # <<< Unbalanced Model Blocks >>> #
    def conv_sep(self, inp, filters, kernel_size=5, strides=2, **kwargs):
        """ Seperable Convolution Layer """
        logger.debug("inp: %s, filters: %s, kernel_size: %s, strides: %s, kwargs: %s)",
                     inp, filters, kernel_size, strides, kwargs)
        name = self.get_name("separableconv2d_{}".format(inp.shape[1]))
        kwargs = self.set_default_initializer(kwargs)
        var_x = SeparableConv2D(filters,
                                kernel_size=kernel_size,
                                strides=strides,
                                padding="same",
                                name="{}_seperableconv2d".format(name),
                                **kwargs)(inp)
        var_x = Activation("relu", name="{}_relu".format(name))(var_x)
        return var_x
