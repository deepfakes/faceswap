#!/usr/bin/env python3
""" Parent class for mask adjustments for faceswap.py converter """

import logging

import cv2
import numpy as np

from lib.model import masks as model_masks
from plugins.convert._config import Config

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def get_config(plugin_name, configfile=None):
    """ Return the config for the requested model """
    return Config(plugin_name, configfile=configfile).config_dict


class Adjustment():
    """ Parent class for adjustments """
    def __init__(self, mask_type, output_size, predicted_available, configfile=None, config=None):
        logger.debug("Initializing %s: (arguments: '%s', output_size: %s, "
                     "predicted_available: %s, configfile: %s, config: %s)",
                     self.__class__.__name__, mask_type, output_size, predicted_available,
                     configfile, config)
        self.config = self.set_config(configfile, config)
        logger.debug("config: %s", self.config)
        self.mask_type = self.get_mask_type(mask_type, predicted_available)
        self.dummy = np.zeros((output_size, output_size, 3), dtype='float32')

        self.skip = self.config.get("type", None) is None
        logger.debug("Initialized %s", self.__class__.__name__)

    def set_config(self, configfile, config):
        """ Set the config to either global config or passed in config """
        section = ".".join(self.__module__.split(".")[-2:])
        if config is None:
            retval = get_config(section, configfile=configfile)
        else:
            config.section = section
            retval = config.config_dict
            config.section = None
        logger.debug("Config: %s", retval)
        return retval

    @staticmethod
    def get_mask_type(mask_type, predicted_available):
        """ Return the requested mask type. Return default mask if
            predicted requested but not available """
        logger.debug("Requested mask_type: %s", mask_type)
        if mask_type == "predicted" and not predicted_available:
            mask_type = model_masks.get_default_mask()
            logger.warning("Predicted selected, but the model was not trained with a mask. "
                           "Switching to '%s'", mask_type)
        logger.debug("Returning mask_type: %s", mask_type)
        return mask_type

    def process(self, *args, **kwargs):
        """ Override for specific color adjustment process """
        raise NotImplementedError

    def run(self, *args, **kwargs):
        """ Perform selected adjustment on face """
        logger.trace("Performing mask adjustment: (plugin: %s, args: %s, kwargs: %s",
                     self.__module__, args, kwargs)
        retval = self.process(*args, **kwargs)
        return retval


class BlurMask():
    """ Factory class to return the correct blur object for requested blur
        Works for square images only.
        Currently supports Gaussian and Normalized Box Filters
    """
    def __init__(self, blur_type, mask, kernel_ratio, passes=1):
        """ image_size = height or width of original image
            mask = the mask to apply the blurring to
            kernel_ratio = kernel ratio as percentage of mask size
            diameter = True calculates approx diameter of mask for kernel, False
            passes = the number of passes to perform the blur """
        logger.trace("Initializing %s: (blur_type: '%s', mask_shape: %s, kernel_ratio: %s, "
                     "passes: %s)", self.__class__.__name__, blur_type, mask.shape, kernel_ratio,
                     passes)
        self.blur_type = blur_type.lower()
        self.mask = mask
        self.passes = passes
        kernel_size = self.get_kernel_size(kernel_ratio)
        self.kernel_size = self.get_kernel_tuple(kernel_size)
        logger.trace("Initialized %s", self.__class__.__name__)

    @property
    def blurred(self):
        """ The final blurred mask """
        func = self.func_mapping[self.blur_type]
        kwargs = self.get_kwargs()
        blurred = self.mask
        for i in range(self.passes):
            ksize = int(kwargs["ksize"][0])
            logger.trace("Pass: %s, kernel_size: %s", i + 1, (ksize, ksize))
            blurred = func(blurred, **kwargs)
            ksize = int(ksize * self.multipass_factor)
            kwargs["ksize"] = self.get_kernel_tuple(ksize)
        logger.trace("Returning blurred mask. Shape: %s", blurred.shape)
        return blurred

    @property
    def multipass_factor(self):
        """ Multipass Factor
            For multiple passes the kernel must be scaled down. This value is
            different for box filter and gaussian """
        factor = dict(gaussian=0.8,
                      normalized=0.5)
        return factor[self.blur_type]

    @property
    def sigma(self):
        """ Sigma for Gaussian Blur
            Returns zero so it is calculated from kernel size """
        return 0

    @property
    def func_mapping(self):
        """ Return a dict of function name to cv2 function """
        return dict(gaussian=cv2.GaussianBlur,  # pylint: disable = no-member
                    normalized=cv2.blur)  # pylint: disable = no-member

    @property
    def kwarg_requirements(self):
        """ Return a dict of function name to a list of required kwargs """
        return dict(gaussian=["ksize", "sigmaX"],
                    normalized=["ksize"])

    @property
    def kwarg_mapping(self):
        """ Return a dict of kwarg names to config item names """
        return dict(ksize=self.kernel_size,
                    sigmaX=self.sigma)

    def get_kernel_size(self, radius_ratio):
        """ Set the kernel size to absolute """
        mask_diameter = np.sqrt(np.sum(self.mask))
        radius = max(1, round(mask_diameter * radius_ratio / 100))
        kernel_size = int((radius * 2) + 1)
        logger.trace("kernel_size: %s", kernel_size)
        return kernel_size

    @staticmethod
    def get_kernel_tuple(kernel_size):
        """ Make sure kernel_size is odd and return it as a tupe """
        kernel_size += 1 if kernel_size % 2 == 0 else 0
        retval = (kernel_size, kernel_size)
        logger.trace(retval)
        return retval

    def get_kwargs(self):
        """ return valid kwargs for the requested blur """
        retval = {kword: self.kwarg_mapping[kword]
                  for kword in self.kwarg_requirements[self.blur_type]}
        logger.trace("BlurMask kwargs: %s", retval)
        return retval
