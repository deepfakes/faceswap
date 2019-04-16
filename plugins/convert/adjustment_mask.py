#!/usr/bin/env python3
""" Adjustments for the swap box and mask for faceswap.py converter
    Based on: https://gist.github.com/anonymous/d3815aba83a8f79779451262599b0955
    found on https://www.reddit.com/r/deepfakes/ """

import cv2

from lib.model import masks as model_masks
from ._base import Adjustment, logger, np


class Box(Adjustment):
    """ Manipulations that occur on the swap box
        Actions performed here occur prior to warping the face back to the background frame

        For actions that occur identically for each frame (e.g. blend_box), constants can
        be placed into self.func_constants to be compiled at launch, then referenced for
        each face. """
    def __init__(self, arguments, output_size):
        self.facesize = output_size
        super().__init__(arguments)

    @property
    def func_names(self):
        return ["blend"]

    def add_blend_func(self, action):
        """ Create the blurred mask for the blend box function """
        config = self.config["blend"]
        do_add = config.get("type", None) is not None
        if not do_add:
            self.add_function(action, do_add)
            return

        # Compile mask into func_constants
        # As gaussian blur technically blurs both sides of the mask, reduce the mask ratio by
        # half to give a more expected box
        mask_ratio = config["distance"] / 200
        erode = slice(round(self.facesize * mask_ratio), -round(self.facesize * mask_ratio))
        mask = np.zeros((self.facesize, self.facesize, 1)).astype("float32")
        mask[erode, erode] = 1.0

        radius = max(1, round(self.facesize * config["radius"] / 100))
        kernel_size = int((radius * 2) + 1)
        mask = BlurMask(config["type"], mask, kernel_size, config["passes"]).blurred

        self.func_constants[action] = mask
        self.add_function(action, do_add)

    def blend(self, **kwargs):
        """ The blend box function. Adds the created mask to the alpha channel """
        logger.trace("Blending box")
        new_face = kwargs["new_face"]
        mask = np.expand_dims(self.func_constants["blend"], axis=-1)
        new_face = np.clip(np.concatenate((new_face, mask), axis=-1), 0.0, 1.0)
        logger.trace("Blended box")
        return new_face


class Mask(Adjustment):
    """ Return the requested mask """

    def __init__(self, arguments, output_size):
        """ Set requested mask """
        self.dummy = np.zeros((output_size, output_size, 3), dtype='float32')
        self.warning_shown = False
        super().__init__(arguments)

    @property
    def func_names(self):
        return ["erode", "blend"]

    # Add Functions
    def add_erode_func(self, action):
        """ Add the erode function to funcs if requested """
        do_add = self.args.erosion_size != 0
        self.add_function(action, do_add)

    def add_blend_func(self, action):
        """ Add the blur function to funcs if requested """
        do_add = self.config["blend"].get("type", None) is not None
        self.add_function(action, do_add)

    # MASK MANIPULATIONS
    def erode(self, mask):
        """ Erode/dilate mask if requested """
        kernel = self.get_erosion_kernel(mask)
        if self.args.erosion_size > 0:
            logger.trace("Eroding mask")
            mask = cv2.erode(mask, kernel, iterations=1)  # pylint: disable=no-member
        else:
            logger.trace("Dilating mask")
            mask = cv2.dilate(mask, kernel, iterations=1)  # pylint: disable=no-member
        return mask

    def get_erosion_kernel(self, mask):
        """ Get the erosion kernel """
        erosion_ratio = self.args.erosion_size / 100
        mask_radius = np.sqrt(np.sum(mask)) / 2
        kernel_size = max(1, int(abs(erosion_ratio * mask_radius)))
        erosion_kernel = cv2.getStructuringElement(  # pylint: disable=no-member
            cv2.MORPH_ELLIPSE,  # pylint: disable=no-member
            (kernel_size, kernel_size))
        logger.trace("erosion_kernel shape: %s", erosion_kernel.shape)
        return erosion_kernel

    def blend(self, mask):
        """ Blur mask if requested """
        logger.trace("Blending mask")
        config = self.config["blend"]
        kernel_size = self.get_blur_kernel_size(mask, config["radius"])
        mask = BlurMask(config["type"], mask, kernel_size, config["passes"]).blurred
        return mask

    @staticmethod
    def get_blur_kernel_size(mask, radius_ratio):
        """ Set the kernel size to absolute """
        mask_diameter = np.sqrt(np.sum(mask))
        radius = max(1, round(mask_diameter * radius_ratio / 100))
        kernel_size = int((radius * 2) + 1)
        logger.trace("kernel_size: %s", kernel_size)
        return kernel_size

    # MASKS
    def mask(self, detected_face, predicted_mask):
        """ Return the mask from lib/model/masks and intersect with box """
        mask_type = self.args.mask_type
        if mask_type == "predicted" and predicted_mask is None:
            mask_type = model_masks.get_default_mask()
            if not self.warning_shown:
                logger.warning("Predicted selected, but the model was not trained with a mask. "
                               "Switching to '%s'", mask_type)
                self.warning_shown = True

        if mask_type == "none":
            mask = np.ones_like(self.dummy[:, :, 1])
        elif mask_type == "predicted":
            mask = predicted_mask
        else:
            landmarks = detected_face.reference_landmarks
            mask = getattr(model_masks, mask_type)(landmarks, self.dummy, channels=1).mask
        return mask

    @staticmethod
    def finalize_mask(mask):
        """ Finalize the mask """
        logger.trace("Finalizing mask")
        np.nan_to_num(mask, copy=False)
        np.clip(mask, 0.0, 1.0, out=mask)
        return mask

    # RETURN THE MASK
    def do_actions(self, *args, **kwargs):
        """ Override do actions to return a face mask """
        mask = self.mask(kwargs["detected_face"], kwargs["predicted_mask"])
        mask = self.finalize_mask(mask)
        raw_mask = mask.copy()

        for func in self.funcs:
            mask = func(mask)
        raw_mask = np.expand_dims(raw_mask, axis=-1) if raw_mask.ndim != 3 else raw_mask
        mask = np.expand_dims(mask, axis=-1) if mask.ndim != 3 else mask
        logger.trace("mask shape: %s, raw_mask shape: %s", mask.shape, raw_mask.shape)
        return mask, raw_mask


class BlurMask():
    """ Factory class to return the correct blur object for requested blur
        Works for square images only.
        Currently supports Gaussian and Normalized Box Filters
    """
    def __init__(self, blur_type, mask, kernel_size, passes=1):
        """ image_size = height or width of original image
            mask = the mask to apply the blurring to
            kernel_size = Initial Kernel size
            diameter = True calculates approx diameter of mask for kernel, False
            passes = the number of passes to perform the blur """
        logger.trace("Initializing %s: (blur_type: '%s', mask_shape: %s, kernel_size: %s, "
                     "passes: %s)", self.__class__.__name__, blur_type, mask.shape, kernel_size,
                     passes)
        self.blur_type = blur_type.lower()
        self.mask = mask
        self.passes = passes
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
