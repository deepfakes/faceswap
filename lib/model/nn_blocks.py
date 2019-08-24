#!/usr/bin/env python3
""" Neural Network Blocks for faceswap.py
    Blocks from:
        the original https://www.reddit.com/r/deepfakes/ code sample + contribs
        dfaker: https://github.com/dfaker/df
        shoanlu GAN: https://github.com/shaoanlu/faceswap-GAN"""

import logging
import tensorflow as tf
import keras.backend as K

from keras.layers import (add, Add, BatchNormalization, concatenate, Lambda, regularizers,
                          Permute, Reshape, SeparableConv2D, Softmax, UpSampling2D)
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation
from keras.initializers import he_uniform, VarianceScaling
from .initializers import ICNR, ConvolutionAware
from .layers import PixelShuffler, SubPixelUpscaling, ReflectionPadding2D, Scale
from .normalization import GroupNormalization, InstanceNormalization

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
        name = self.get_name("conv")
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
                use_instance_norm=False, res_block_follows=False, **kwargs):
        """ Upscale Layer """
        logger.debug("inp: %s, filters: %s, kernel_size: %s, use_instance_norm: %s, kwargs: %s)",
                     inp, filters, kernel_size, use_instance_norm, kwargs)
        name = self.get_name("upscale")
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
        var_x = self.conv2d(inp, filters * 4,
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
            var_x = SubPixelUpscaling(name="{}_subpixel".format(name))(var_x)
        else:
            var_x = PixelShuffler(name="{}_pixelshuffler".format(name))(var_x)
        return var_x

    # <<< DFaker Model Blocks >>> #
    def res_block(self, inp, filters, kernel_size=3, padding="same", **kwargs):
        """ Residual block """
        logger.debug("inp: %s, filters: %s, kernel_size: %s, kwargs: %s)",
                     inp, filters, kernel_size, kwargs)
        name = self.get_name("residual")
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
        name = self.get_name("separableconv2d")
        kwargs = self.set_default_initializer(kwargs)
        var_x = SeparableConv2D(filters,
                                kernel_size=kernel_size,
                                strides=strides,
                                padding="same",
                                name="{}_seperableconv2d".format(name),
                                **kwargs)(inp)
        var_x = Activation("relu", name="{}_relu".format(name))(var_x)
        return var_x

# <<< GAN V2.2 Blocks >>> #
# TODO Merge these into NNBLock class when porting GAN2.2


# Gan Constansts:
GAN22_CONV_INIT = "he_normal"
GAN22_REGULARIZER = 1e-4


# Gan Blocks:
def normalization(inp, norm="none", group="16"):
    """ GAN Normalization """
    if norm == "layernorm":
        var_x = GroupNormalization(group=group)(inp)
    elif norm == "batchnorm":
        var_x = BatchNormalization()(inp)
    elif norm == "groupnorm":
        var_x = GroupNormalization(group=16)(inp)
    elif norm == "instancenorm":
        var_x = InstanceNormalization()(inp)
    elif norm == "hybrid":
        if group % 2 == 1:
            raise ValueError("Output channels must be an even number for hybrid norm, "
                             "received {}.".format(group))
        filt = group
        var_x_0 = Lambda(lambda var_x: var_x[..., :filt // 2])(var_x)
        var_x_1 = Lambda(lambda var_x: var_x[..., filt // 2:])(var_x)
        var_x_0 = Conv2D(filt // 2,
                         kernel_size=1,
                         kernel_regularizer=regularizers.l2(GAN22_REGULARIZER),
                         kernel_initializer=GAN22_CONV_INIT)(var_x_0)
        var_x_1 = InstanceNormalization()(var_x_1)
        var_x = concatenate([var_x_0, var_x_1], axis=-1)
    else:
        var_x = inp
    return var_x


def upscale_ps(inp, filters, initializer, use_norm=False, norm="none"):
    """ GAN Upscaler - Pixel Shuffler """
    var_x = Conv2D(filters * 4,
                   kernel_size=3,
                   kernel_regularizer=regularizers.l2(GAN22_REGULARIZER),
                   kernel_initializer=initializer,
                   padding="same")(inp)
    var_x = LeakyReLU(0.2)(var_x)
    var_x = normalization(var_x, norm, filters) if use_norm else var_x
    var_x = PixelShuffler()(var_x)
    return var_x


def upscale_nn(inp, filters, use_norm=False, norm="none"):
    """ GAN Neural Network """
    var_x = UpSampling2D()(inp)
    var_x = reflect_padding_2d(var_x, 1)
    var_x = Conv2D(filters,
                   kernel_size=3,
                   kernel_regularizer=regularizers.l2(GAN22_REGULARIZER),
                   kernel_initializer="he_normal")(var_x)
    var_x = normalization(var_x, norm, filters) if use_norm else var_x
    return var_x


def reflect_padding_2d(inp, pad=1):
    """ GAN Reflect Padding (2D) """
    var_x = Lambda(lambda var_x: tf.pad(var_x,
                                        [[0, 0], [pad, pad], [pad, pad], [0, 0]],
                                        mode="REFLECT"))(inp)
    return var_x


def conv_gan(inp, filters, use_norm=False, strides=2, norm="none"):
    """ GAN Conv Block """
    var_x = Conv2D(filters,
                   kernel_size=3,
                   strides=strides,
                   kernel_regularizer=regularizers.l2(GAN22_REGULARIZER),
                   kernel_initializer=GAN22_CONV_INIT,
                   use_bias=False,
                   padding="same")(inp)
    var_x = Activation("relu")(var_x)
    var_x = normalization(var_x, norm, filters) if use_norm else var_x
    return var_x


def conv_d_gan(inp, filters, use_norm=False, norm="none"):
    """ GAN Discriminator Conv Block """
    var_x = inp
    var_x = Conv2D(filters,
                   kernel_size=4,
                   strides=2,
                   kernel_regularizer=regularizers.l2(GAN22_REGULARIZER),
                   kernel_initializer=GAN22_CONV_INIT,
                   use_bias=False,
                   padding="same")(var_x)
    var_x = LeakyReLU(alpha=0.2)(var_x)
    var_x = normalization(var_x, norm, filters) if use_norm else var_x
    return var_x


def res_block_gan(inp, filters, use_norm=False, norm="none"):
    """ GAN Res Block """
    var_x = Conv2D(filters,
                   kernel_size=3,
                   kernel_regularizer=regularizers.l2(GAN22_REGULARIZER),
                   kernel_initializer=GAN22_CONV_INIT,
                   use_bias=False,
                   padding="same")(inp)
    var_x = LeakyReLU(alpha=0.2)(var_x)
    var_x = normalization(var_x, norm, filters) if use_norm else var_x
    var_x = Conv2D(filters,
                   kernel_size=3,
                   kernel_regularizer=regularizers.l2(GAN22_REGULARIZER),
                   kernel_initializer=GAN22_CONV_INIT,
                   use_bias=False,
                   padding="same")(var_x)
    var_x = add([var_x, inp])
    var_x = LeakyReLU(alpha=0.2)(var_x)
    var_x = normalization(var_x, norm, filters) if use_norm else var_x
    return var_x


def self_attn_block(inp, n_c, squeeze_factor=8):
    """ GAN Self Attention Block
    Code borrows from https://github.com/taki0112/Self-Attention-GAN-Tensorflow
    """
    msg = "Input channels must be >= {}, recieved nc={}".format(squeeze_factor, n_c)
    assert n_c // squeeze_factor > 0, msg
    var_x = inp
    shape_x = var_x.get_shape().as_list()

    var_f = Conv2D(n_c // squeeze_factor, 1,
                   kernel_regularizer=regularizers.l2(GAN22_REGULARIZER))(var_x)
    var_g = Conv2D(n_c // squeeze_factor, 1,
                   kernel_regularizer=regularizers.l2(GAN22_REGULARIZER))(var_x)
    var_h = Conv2D(n_c, 1, kernel_regularizer=regularizers.l2(GAN22_REGULARIZER))(var_x)

    shape_f = var_f.get_shape().as_list()
    shape_g = var_g.get_shape().as_list()
    shape_h = var_h.get_shape().as_list()
    flat_f = Reshape((-1, shape_f[-1]))(var_f)
    flat_g = Reshape((-1, shape_g[-1]))(var_g)
    flat_h = Reshape((-1, shape_h[-1]))(var_h)

    var_s = Lambda(lambda var_x: K.batch_dot(var_x[0],
                                             Permute((2, 1))(var_x[1])))([flat_g, flat_f])

    beta = Softmax(axis=-1)(var_s)
    var_o = Lambda(lambda var_x: K.batch_dot(var_x[0], var_x[1]))([beta, flat_h])
    var_o = Reshape(shape_x[1:])(var_o)
    var_o = Scale()(var_o)

    out = add([var_o, inp])
    return out
