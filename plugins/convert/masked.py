#!/usr/bin/env python3
""" Masked converter for faceswap.py
    Based on: https://gist.github.com/anonymous/d3815aba83a8f79779451262599b0955
    found on https://www.reddit.com/r/deepfakes/ """

import logging
import os
from pathlib import Path

import cv2
import numpy as np
from lib.model import masks as model_masks
from plugins.convert._config import Config

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class Convert():
    """ Swap a source face with a target """
    def __init__(self, output_dir, training_size, padding, crop, arguments):
        logger.debug("Initializing %s: (output_dir: '%s', training_size: %s, padding: %s, "
                     "crop: %s, arguments: %s)", self.__class__.__name__, output_dir,
                     training_size, padding, crop, arguments)
        self.config = Config(None)
        self.crop = crop
        self.output_dir = output_dir
        self.args = arguments
        self.box = Box(training_size, padding, self.config)
        self.mask = Mask(arguments.mask_type,
                         training_size,
                         padding,
                         crop,
                         arguments.erosion_size,
                         self.config)
        logger.debug("Initialized %s", self.__class__.__name__)

    def get_config(self, section):
        """ Return the config dict for the requested section """
        self.config.section = section
        logger.trace("returning config for section: '%s'", section)
        return self.config.config_dict

    def process(self, in_queue, out_queue):
        """ Process items from the queue """
        logger.debug("Starting convert process. (in_queue: %s, out_queue: %s)",
                     in_queue, out_queue)
        while True:
            item = in_queue.get()
            if item == "EOF":
                logger.debug("Patch queue finished")
                # Signal EOF to other processes in pool
                in_queue.put(item)
                break
            logger.trace("Patch queue got: '%s'", item["filename"])

            try:
                image = self.patch_image(item)
            except Exception as err:
                # Log error and output original frame
                logger.error("Failed to convert image: '%s'. Reason: %s",
                             item["filename"], str(err))
                image = item["image"]

            out_file = str(self.output_dir / Path(item["filename"]).name)
            if self.args.draw_transparent:
                out_file = "{}.png".format(os.path.splitext(out_file)[0])
                logger.trace("Set extension to png: `%s`", out_file)

            logger.trace("Out queue put: %s", out_file)
            out_queue.put((out_file, image))

        out_queue.put("EOF")
        logger.debug("Completed convert process")

    def patch_image(self, predicted):
        """ Patch the image """
        logger.trace("Patching image: '%s'", predicted["filename"])
        frame_size = (predicted["image"].shape[1], predicted["image"].shape[0])
        new_image = self.get_new_image(predicted, frame_size)
        image_mask = self.get_image_mask(predicted, frame_size)
        patched_face = self.apply_fixes(predicted, new_image, image_mask)
        logger.trace("Patched image: '%s'", predicted["filename"])
        return patched_face

    def get_new_image(self, predicted, frame_size):
        """ Get the new face from the predictor and apply box manipulations """
        logger.trace("Getting: (filename: '%s', faces: %s)",
                     predicted["filename"], len(predicted["swapped_faces"]))
        new_image = predicted["image"].copy()
        source_faces = [face.aligned_face[:, :, :3] for face in predicted["detected_faces"]]
        interpolators = [face.adjusted_interpolators for face in predicted["detected_faces"]]

        for idx, new_face in enumerate(predicted["swapped_faces"]):
            src_face = source_faces[idx]
            old_face = predicted["original_faces"][idx]
            interpolator = interpolators[idx][1]

            # Perform the box functions on unwarped face
            for func in self.box.funcs:
                logger.trace("Performing function: %s", func)
                new_face = func(old_face=old_face, new_face=new_face)

            src_face[self.crop, self.crop] = new_face

            new_image = cv2.warpAffine(  # pylint: disable=no-member
                src_face,
                predicted["detected_faces"][idx].adjusted_matrix,
                frame_size,
                new_image,
                flags=cv2.WARP_INVERSE_MAP | interpolator,  # pylint: disable=no-member
                borderMode=cv2.BORDER_TRANSPARENT)  # pylint: disable=no-member
        logger.trace("Got filename: '%s'")
        return new_image

    def get_image_mask(self, predicted, frame_size):
        """ Get the image mask """
        logger.trace("Getting: '%s'", predicted["filename"])
        masks = list()
        for idx, detected_face in enumerate(predicted["detected_faces"]):
            pred_masks = predicted["masks"]
            print(pred_masks.shape)
            predicted_mask = pred_masks[idx] if pred_masks.any() else None
            mask = self.mask.get_mask(detected_face, frame_size, predicted_mask)
            masks.append(np.clip(mask, 0.0, 1.0, out=mask))
        logger.trace("Got: '%s'", predicted["filename"])
        return masks

    def apply_fixes(self, predicted, new_image, masks):
        """ Apply fixes """
        frame = predicted["image"][:, :, :3]
        new_image = new_image[:, :, :3]
        for idx, detected_face in enumerate(predicted["detected_faces"]):
            image_mask = masks[idx][:, :, :3].copy()
            landmarks = detected_face.landmarks_as_xy

            # TODO - force default for args.sharpen_image to ensure it isn't None
            if self.args.sharpen_image is not None and self.args.sharpen_image.lower() != "none":
                new_image = self.sharpen(new_image, self.args.sharpen_image)

            if self.args.avg_color_adjust:
                new_image = self.color_adjust(new_image, frame, image_mask)

            if self.args.match_histogram:
                new_image = self.color_hist_match(new_image, frame, image_mask)

            if self.args.seamless_clone:
                frame = self.seamless_clone(new_image, frame, image_mask)

            else:
                foreground = new_image * image_mask
                background = frame * (1.0 - image_mask)
                frame = foreground + background

            np.clip(frame, 0.0, 255.0, out=frame)
            if self.args.draw_transparent:
                # Adding a 4th channel should happen after all other channel operations
                # Add mask as 4th channel for saving as alpha on supported output formats
                frame = masks.dfl_full(landmarks, frame, channels=4).mask
        return np.rint(frame).astype('uint8')

    @staticmethod
    def sharpen(new, method):
        """ Sharpen using the unsharp=mask technique , subtracting a blurried image """
        np.clip(new, 0.0, 255.0, out=new)
        if method == "box_filter":
            kernel = np.ones((3, 3)) * (-1)
            kernel[1, 1] = 9
            new = cv2.filter2D(new, -1, kernel)  # pylint: disable=no-member
        elif method == "gaussian_filter":
            blur = cv2.GaussianBlur(new, (0, 0), 3.0)   # pylint: disable=no-member
            new = cv2.addWeighted(new, 1.5, blur, -.5, 0, new)  # pylint: disable=no-member

        return new

    @staticmethod
    def color_adjust(new, frame, img_mask):
        """ Adjust the mean of the color channels to be the same for the swap and old frame """
        for _ in [0, 1]:
            np.clip(new, 0.0, 255.0, out=new)
            diff = frame - new
            avg_diff = np.sum(diff * img_mask, axis=(0, 1))
            adjustment = avg_diff / np.sum(img_mask, axis=(0, 1))
            new = new + adjustment

        return new

    def color_hist_match(self, new, frame, img_mask):
        """ Match the histogram of the color intensity of each channel """
        np.clip(new, 0.0, 255.0, out=new)
        new = [self.hist_match(new[:, :, c], frame[:, :, c], img_mask[:, :, c]) for c in [0, 1, 2]]
        new = np.stack(new, axis=2)

        return new

    @staticmethod
    def hist_match(new, frame, img_mask):
        """  Construct the histogram of the color intensity of a channel
             for the swap and the original. Match the histogram of the original
             by interpolation
        """

        mask_indices = img_mask.nonzero()
        if len(mask_indices[0]) == 0:
            return new

        m_new = new[mask_indices].ravel()
        m_frame = frame[mask_indices].ravel()
        _, bin_idx, s_counts = np.unique(m_new, return_inverse=True, return_counts=True)
        t_values, t_counts = np.unique(m_frame, return_counts=True)
        s_quants = np.cumsum(s_counts, dtype='float32')
        t_quants = np.cumsum(t_counts, dtype='float32')
        s_quants /= s_quants[-1]  # cdf
        t_quants /= t_quants[-1]  # cdf
        interp_s_values = np.interp(s_quants, t_quants, t_values)
        new.put(mask_indices, interp_s_values[bin_idx])

        return new

    @staticmethod
    def seamless_clone(new, frame, img_mask):
        """ Seamless clone the swapped image into the old frame with cv2 """
        np.clip(new, 0.0, 255.0, out=new)
        height, width, _ = frame.shape
        height = height // 2
        width = width // 2
        y_indices, x_indices, _ = np.nonzero(img_mask)
        y_crop = slice(np.min(y_indices), np.max(y_indices))
        x_crop = slice(np.min(x_indices), np.max(x_indices))
        y_center = int(np.rint((np.max(y_indices) + np.min(y_indices)) / 2 + height))
        x_center = int(np.rint((np.max(x_indices) + np.min(x_indices)) / 2 + width))

        insertion = np.rint(new[y_crop, x_crop]).astype('uint8')
        insertion_mask = img_mask[y_crop, x_crop]
        insertion_mask[insertion_mask != 0] = 255
        insertion_mask = insertion_mask.astype('uint8')
        prior = np.pad(frame, ((height, height), (width, width), (0, 0)), 'constant')
        prior = prior.astype('uint8')

        blended = cv2.seamlessClone(insertion,  # pylint: disable=no-member
                                    prior,
                                    insertion_mask,
                                    (x_center, y_center),
                                    cv2.NORMAL_CLONE)  # pylint: disable=no-member
        blended = blended[height:-height, width:-width]

        return blended


class Box():
    """ Manipulations that occur on the swap box
        Actions performed here occur prior to warping the face back to the background frame

        For actions that occur identically for each frame (e.g. blend_box), constants can
        be placed into self.func_constants to be compiled at launch, then referenced for
        each face. """
    def __init__(self, training_size, padding, config):
        logger.trace("Initializing %s: (training_size: '%s', crop: %s)", self.__class__.__name__,
                     training_size, padding)

        self.config = config
        self.facesize = training_size - (padding * 2)
        self.func_constants = dict()
        self.funcs = list()
        self.add_functions()
        logger.trace("Initialized %s: Number of functions: %s",
                     self.__class__.__name__, len(self.funcs))

    def get_config(self, section):
        """ Return the config dict for the requested section """
        self.config.section = section
        logger.trace("returning config for section: '%s'", section)
        return self.config.config_dict

    def add_functions(self):
        """ Add the functions to be performed on the swap box """
        for action in ("crop_box", "blend_box"):
            getattr(self, action)()

    def crop_box(self):
        """ Crop the edges of the swap box to remove artefacts """
        config = self.get_config("box.crop")
        if config.get("pixels", 0) == 0:
            logger.debug("Crop box not selected")
            return
        logger.debug("Config: %s", config)
        crop = slice(config["pixels"], self.facesize - config["pixels"])
        self.func_constants["crop_box"] = crop
        logger.debug("Crop box added to funcs")
        self.funcs.append(self.crop_box_func)

    def crop_box_func(self, **kwargs):
        """ The crop box function """
        logger.trace("Cropping box")
        new_face = kwargs["new_face"]
        cropped_face = kwargs["old_face"].copy()
        crop = self.func_constants["crop_box"]
        cropped_face[crop, crop] = new_face[crop, crop]
        logger.trace("Cropped Box")
        return cropped_face

    def blend_box(self):
        """ Create the blurred mask for the blend box function """
        config = self.get_config("box.blend")
        if not config.get("type", None):
            logger.debug("Blend box not selected")
            return
        logger.debug("Config: %s", config)

        mask_ratio = 1 - (config["blending_box"] / 100)
        erode = slice(int(self.facesize * mask_ratio), -int(self.facesize * mask_ratio))
        mask = np.zeros((self.facesize, self.facesize, 3))
        mask[erode, erode] = 1.0

        kernel_ratio = config["kernel_size"] / 200
        kernel_size = int((self.facesize - (erode.start * 2)) * kernel_ratio)

        mask = BlurMask(config["type"], mask, kernel_size, config["passes"]).blurred
        self.func_constants["blend_box"] = mask
        self.funcs.append(self.blend_box_func)
        logger.debug("Blend box added to funcs")

    def blend_box_func(self, **kwargs):
        """ The blend box function """
        logger.trace("Blending box")
        old_face = kwargs["old_face"]
        new_face = kwargs["new_face"]
        mask = self.func_constants["blend_box"]
        new_face = (mask * new_face + (1.0 - mask) * old_face)
        logger.trace("Blended box")
        return new_face


class Mask():
    """ Return the requested mask """

    def __init__(self, mask_type, training_size, padding, crop, erosion_percent, config):
        """ Set requested mask """
        logger.debug("Initializing %s: (mask_type: '%s', training_size: %s, padding: %s)",
                     self.__class__.__name__, mask_type, training_size, padding)

        self.training_size = training_size
        self.padding = padding
        self.mask_type = mask_type
        self.crop = crop
        self.erosion_percent = erosion_percent
        self.config = config
        self.funcs = list()
        self.func_constants = dict()
        self.add_functions()
        logger.debug("Initialized %s", self.__class__.__name__)

    def get_config(self, section):
        """ Return the config dict for the requested section """
        self.config.section = section
        logger.trace("returning config for section: '%s'", section)
        return self.config.config_dict

    # ADD REQUESTED MASK MANIPULATION FUNCTIONS
    def add_functions(self):
        """ Add mask manipulation functions to self.funcs """
        for action in ("erode", "blur"):
            getattr(self, "add_{}_func".format(action))()

    def add_erode_func(self):
        """ Add the erode function to funcs if requested """
        if self.erosion_percent != 0:
            self.funcs.append(self.erode_mask)

    def add_blur_func(self):
        """ Add the blur function to funcs if requested """
        config = self.get_config("mask.blend")
        if not config.get("type", None):
            logger.debug("Mask blending not selected")
            return
        logger.debug("blur mask config: %s", config)
        self.func_constants["blur"] = {"type": config["type"],
                                       "passes": config["passes"],
                                       "ratio": config["kernel_size"] / 100}
        self.funcs.append(self.blur_mask)

    # MASK MANIPULATIONS
    def erode_mask(self, mask):
        """ Erode/dilate mask if requested """
        kernel = self.get_erosion_kernel(mask)
        if self.erosion_percent > 0:
            logger.trace("Eroding mask")
            mask = cv2.erode(mask, kernel, iterations=1)  # pylint: disable=no-member
        else:
            logger.trace("Dilating mask")
            mask = cv2.dilate(mask, kernel, iterations=1)  # pylint: disable=no-member
        return mask

    def get_erosion_kernel(self, mask):
        """ Get the erosion kernel """
        erosion_ratio = self.erosion_percent / 100
        mask_radius = np.sqrt(np.sum(mask)) / 2
        kernel_size = max(1, int(abs(erosion_ratio * mask_radius)))
        erosion_kernel = cv2.getStructuringElement(  # pylint: disable=no-member
            cv2.MORPH_ELLIPSE,  # pylint: disable=no-member
            (kernel_size, kernel_size))
        logger.trace("erosion_kernel shape: %s", erosion_kernel.shape)
        return erosion_kernel

    def blur_mask(self, mask):
        """ Blur mask if requested """
        logger.trace("Blending mask")
        blur_type = self.func_constants["blur"]["type"]
        kernel_ratio = self.func_constants["blur"]["ratio"]
        passes = self.func_constants["blur"]["passes"]
        kernel_size = self.get_blur_kernel_size(mask, kernel_ratio)
        mask = BlurMask(blur_type, mask, kernel_size, passes).blurred
        return mask

    @staticmethod
    def get_blur_kernel_size(mask, kernel_ratio):
        """ Set the kernel size to absolute """
        mask_diameter = np.sqrt(np.sum(mask))
        kernel_size = int(max(1, kernel_ratio * mask_diameter))
        logger.trace("kernel_size: %s", kernel_size)
        return kernel_size

    # RETURN THE MASK
    def get_mask(self, detected_face, image_size, predicted_mask=None):
        """ Return a face mask """
        mask = self.mask(detected_face, image_size, predicted_mask)
        mask = self.finalize_mask(mask)

        # TODO REMOVE
        loc = "/home/matt/fake/test/testing"
        cv2.imwrite("{}/facemask_pre.png".format(loc), mask * 255.0)

        for func in self.funcs:
            mask = func(mask)

        cv2.imwrite("{}/facemask_post.png".format(loc), mask * 255.0)

        logger.trace("mask shape: %s", mask.shape)
        return mask

    # MASKS
    def mask(self, detected_face, image_size, predicted_mask):
        """ Return the mask from lib/model/masks and intersect with box """
        if self.mask_type == "predicted" and predicted_mask is None:
            self.mask_type = model_masks.get_default_mask()
            logger.warning("Predicted selected, but the model was not trained with a mask. "
                           "Switching to '%s'", self.mask_type)

        dummy = np.zeros((image_size[1], image_size[0], 3), dtype='float32')
        rect_kwargs = dict(interpolator=detected_face.adjusted_interpolators[1],
                           matrix=detected_face.adjusted_matrix,
                           image_size=image_size,
                           dummy=dummy)
        if self.mask_type in ("none", "rect"):
            mask = getattr(self, self.mask_type)(**rect_kwargs)
        elif self.mask_type == "predicted":
            mask = predicted_mask
            print(mask.shape)
            mask = self.intersect_rect(mask, **rect_kwargs)
        else:
            landmarks = detected_face.landmarks_as_xy
            mask = getattr(model_masks, self.mask_type)(landmarks, dummy, channels=3).mask
            mask = self.intersect_rect(mask, **rect_kwargs)
        return mask

    def rect(self, interpolator, matrix, image_size, dummy):
        """ Namespace for rect mask. This is the same as 'none' in the cli """
        return self.none(interpolator, matrix, image_size, dummy)

    def none(self, interpolator, matrix, image_size, dummy):
        """ Rect Mask """
        logger.trace("Getting mask")
        ones = np.zeros((self.training_size, self.training_size, 3), dtype='float32')
        ones[self.crop, self.crop] = 1.0
        cv2.warpAffine(ones,  # pylint: disable=no-member
                       matrix,
                       image_size,
                       dummy,
                       flags=cv2.WARP_INVERSE_MAP | interpolator,  # pylint: disable=no-member
                       borderMode=cv2.BORDER_CONSTANT,  # pylint: disable=no-member
                       borderValue=0.0)
        return dummy

    def intersect_rect(self, hull_mask, **kwargs):
        """ Intersect the given hull mask with the roi """
        logger.trace("Intersecting rect")
        mask = self.rect(**kwargs)
        mask *= hull_mask
        return mask

    @staticmethod
    def finalize_mask(mask):
        """ Finalize the mask """
        logger.trace("Finalizing mask")
        np.nan_to_num(mask, copy=False)
        np.clip(mask, 0.0, 1.0, out=mask)
        return mask


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
        for i in range(self.passes):
            ksize = int(kwargs["ksize"][0])
            logger.trace("Pass: %s, kernel_size: %s", i + 1, (ksize, ksize))
            blurred = func(self.mask, **kwargs)
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
