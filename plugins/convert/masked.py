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

# TODO Option to tighten/loosen mask in match histogram


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
        self.pre_adjustments = PrePlacementAdjustments(training_size, arguments, self.config)
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
                # TODO Remove this debugging code
                import sys
                import traceback
                exc_info = sys.exc_info()
                traceback.print_exception(*exc_info)

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
        new_images, masks = self.get_new_image(predicted, frame_size)
        patched_face = self.apply_fixes(predicted, new_images, masks)
        logger.trace("Patched image: '%s'", predicted["filename"])
        return patched_face

    def get_new_image(self, predicted, frame_size):
        """ Get the new face from the predictor and apply box manipulations """
        logger.trace("Getting: (filename: '%s', faces: %s)",
                     predicted["filename"], len(predicted["swapped_faces"]))

        source_faces = [face.aligned_face[:, :, :3].astype("float32") / 255.0
                        for face in predicted["detected_faces"]]
        interpolators = [face.adjusted_interpolators for face in predicted["detected_faces"]]
        placeholders = list()
        masks = list()

        for idx, new_face in enumerate(predicted["swapped_faces"]):
            # TODO Check whether old_face can be replaced with a crop of src_face
            # TODO Crop box - old face reinsertion
            warped = np.zeros((frame_size[1], frame_size[0], 4), dtype="float32")
            new_face = new_face[:, :, :3]
            predicted_mask = new_face[:, :, :-1] if new_face.ndim == 4 else None
            src_face = source_faces[idx]
            swapped_face = np.concatenate((src_face, np.zeros(src_face.shape[:2] + (1, ))),
                                          axis=-1).astype("float32")
            old_face = predicted["original_faces"][idx]
            interpolator = interpolators[idx][1]

            # Create the box mask in the alpha channel of unwarped face
            for func in self.box.funcs:
                logger.trace("Performing function: %s", func)
                new_face = func(new_face)

            swapped_face[self.crop, self.crop] = new_face
            # Place mask into the alpha channel
            swapped_face = self.get_image_mask(swapped_face,
                                               predicted["detected_faces"][idx],
                                               predicted_mask)
            # Warp face with the mask
            warped = cv2.warpAffine(  # pylint: disable=no-member
                swapped_face,
                predicted["detected_faces"][idx].adjusted_matrix,
                frame_size,
                warped,
                flags=cv2.WARP_INVERSE_MAP | interpolator,  # pylint: disable=no-member
                borderMode=cv2.BORDER_TRANSPARENT)  # pylint: disable=no-member

            warped = np.clip(warped, 0.0, 1.0)
            # Output face to placeholder and mask to mask list
            masks.append(np.expand_dims(warped[:, :, -1], axis=2))
            placeholders.append(np.clip(warped[:, :, :3], 0.0, 1.0))

        placeholders = np.stack(placeholders) if placeholders else np.array(list())
        masks = np.stack(masks) if masks else np.array(list())
        logger.trace("Got filename: '%s'. (placeholders: %s, masks: %s)",
                     predicted["filename"], placeholders.shape, masks.shape)

        return placeholders, masks

    def get_image_mask(self, new_face, detected_face, predicted_mask):
        """ Get the image mask """
        logger.trace("Getting mask. Image shape: %s", new_face.shape)
        mask = self.mask.get_mask(detected_face, predicted_mask)
        if new_face.shape[2] == 4:
            logger.trace("Combining mask with alpha channel box mask")
            new_face[:, :, -1] = np.minimum(new_face[:, :, -1], mask.squeeze())
        else:
            logger.trace("Adding mask to alpha channel")
            new_face = np.concatenate((new_face, mask), -1)
        new_face = np.clip(new_face, 0.0, 1.0)
        logger.trace("Got mask. Image shape: %s", new_face.shape)
        return new_face

    def apply_fixes(self, predicted, swapped_frames, masks):
        """ Apply fixes """
        background = predicted["image"][:, :, :3] / 255.0
        foreground = np.zeros_like(background)

        for swapped_frame, mask in zip(swapped_frames, masks):
            swapped_frame = self.pre_adjustments.adjust(background, swapped_frame, mask)
            foreground += swapped_frame * mask
            background *= (1.0 - mask)

        # TODO - force default for args.sharpen_image to ensure it isn't None
        if self.args.sharpen_image is not None and self.args.sharpen_image.lower() != "none":
            foreground = self.sharpen(foreground, self.args.sharpen_image)

        frame = foreground + background
        frame = self.draw_transparent(frame, predicted)

        np.clip(frame, 0.0, 1.0, out=frame)
        return np.rint(frame * 255.0).astype("uint8")

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

    def draw_transparent(self, frame, predicted):
        """ Adding a 4th channel should happen after all other channel operations
            Add the default mask as 4th channel for saving as png with alpha channel """
        if not self.args.draw_transparent:
            return frame
        logger.trace("Creating transparent image: '%s'", predicted["filename"])
        mask_type = getattr(model_masks, model_masks.get_default_mask())
        final_mask = np.zeros(frame.shape[:2] + (1, ), dtype="float32")

        for detected_face in predicted["detected_faces"]:
            landmarks = detected_face.landmarks_as_xy
            final_mask = cv2.bitwise_or(final_mask,  # pylint: disable=no-member
                                        mask_type(landmarks, frame, channels=1).mask)
        frame = np.concatenate((frame, np.expand_dims(final_mask, axis=-1)), axis=-1)
        logger.trace("Created transparent image: '%s'", predicted["filename"])
        return frame


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

    def crop_box_func(self, new_face):
        """ The crop box function """
        logger.trace("Cropping box")
        crop = self.func_constants["crop_box"]
        new_face = new_face[crop, crop]
        logger.trace("Cropped Box")
        return new_face

    def blend_box(self):
        """ Create the blurred mask for the blend box function """
        config = self.get_config("box.blend")
        if not config.get("type", None):
            logger.debug("Blend box not selected")
            return
        logger.debug("Config: %s", config)

        mask_ratio = 1 - (config["blending_box"] / 100)
        erode = slice(int(self.facesize * mask_ratio), -int(self.facesize * mask_ratio))
        mask = np.zeros((self.facesize, self.facesize, 1))
        mask[erode, erode] = 1.0

        kernel_ratio = config["kernel_size"] / 200
        kernel_size = int((self.facesize - (erode.start * 2)) * kernel_ratio)

        mask = BlurMask(config["type"], mask, kernel_size, config["passes"]).blurred
        self.func_constants["blend_box"] = mask
        self.funcs.append(self.blend_box_func)
        logger.debug("Blend box added to funcs")

    def blend_box_func(self, new_face):
        """ The blend box function. Adds the created mask to the alpha channel """
        logger.trace("Blending box")
        mask = np.expand_dims(self.func_constants["blend_box"], axis=-1)
        new_face = np.clip(np.concatenate((new_face, mask), axis=-1), 0.0, 1.0)
        logger.trace("Blended box")
        return new_face


class Mask():
    """ Return the requested mask """

    def __init__(self, mask_type, training_size, padding, crop, erosion_percent, config):
        """ Set requested mask """
        logger.debug("Initializing %s: (mask_type: '%s', training_size: %s, padding: %s)",
                     self.__class__.__name__, mask_type, training_size, padding)

        self.padding = padding
        self.mask_type = mask_type
        self.crop = crop
        self.erosion_percent = erosion_percent
        self.config = config

        self.dummy = np.zeros((training_size, training_size, 3), dtype='float32')
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
    def get_mask(self, detected_face, predicted_mask=None):
        """ Return a face mask """
        mask = self.mask(detected_face, predicted_mask)
        mask = self.finalize_mask(mask)

        for func in self.funcs:
            mask = func(mask)
        if mask.ndim != 3:
            mask = np.expand_dims(mask, axis=-1)

        logger.trace("mask shape: %s", mask.shape)
        return mask

    # MASKS
    def mask(self, detected_face, predicted_mask):
        """ Return the mask from lib/model/masks and intersect with box """
        if self.mask_type == "predicted" and predicted_mask is None:
            self.mask_type = model_masks.get_default_mask()
            logger.warning("Predicted selected, but the model was not trained with a mask. "
                           "Switching to '%s'", self.mask_type)

        if self.mask_type in ("none", "rect"):
            mask = getattr(self, self.mask_type)()
        elif self.mask_type == "predicted":
            mask = predicted_mask
            mask = self.intersect_rect(mask)
        else:
            landmarks = detected_face.aligned_landmarks
            mask = getattr(model_masks, self.mask_type)(landmarks, self.dummy, channels=1).mask
            mask = self.intersect_rect(mask)
        return mask

    def rect(self):
        """ Namespace for rect mask. This is the same as 'none' in the cli """
        return self.none()

    def none(self):
        """ Rect Mask """
        logger.trace("Getting mask")
        mask = self.dummy[:, :, :1]
        mask[self.crop, self.crop] = 1.0
        return mask

    def intersect_rect(self, hull_mask):
        """ Intersect the given hull mask with the roi """
        logger.trace("Intersecting rect")
        mask = self.rect()
        mask *= hull_mask
        return mask

    @staticmethod
    def finalize_mask(mask):
        """ Finalize the mask """
        logger.trace("Finalizing mask")
        np.nan_to_num(mask, copy=False)
        np.clip(mask, 0.0, 1.0, out=mask)
        return mask


class PrePlacementAdjustments():
    """ Adjustments for the face

        As the mask is embedded to Alpha Channel, adjustments that use the mask
        must be performed prior to placement in the new image (i.e. when the mask has been
        applied)

        Otherwise, in instances where there are multiple faces, masks can clash if they are all
        placed into the final frame
        """
    def __init__(self, training_size, arguments, config):
        """ Set requested mask """
        logger.debug("Initializing %s: (arguments: '%s')",
                     self.__class__.__name__, arguments)

        self.training_size = training_size
        self.args = arguments
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

    # ADD REQUESTED PRE PLACEMENT MANIPULATION FUNCTIONS
    def add_functions(self):
        """ Add face manipulation functions to self.funcs """
        for action in ("avg_color_adjust", "match_histogram", "seamless_clone"):
            if hasattr(self.args, action) and getattr(self.args, action):
                getattr(self, "add_{}_func".format(action))(action)

    def add_function(self, action):
        """ Add the specified function to self.funcs """
        logger.debug("Adding: '%s'", action)
        self.funcs.append(getattr(self, action))

    def add_avg_color_adjust_func(self, action):
        """ Add the average color adjust function to funcs if requested """
        self.add_function(action)

    def add_match_histogram_func(self, action):
        """ Add the match histogram function to funcs if requested """
        config = self.get_config("face.match_histogram")
        logger.debug("%s config: %s", action, config)
        self.func_constants[action] = {"threshold": config["threshold"] / 100}
        self.add_function(action)

    def add_seamless_clone_func(self, action):
        """ Add the seamless clone function to funcs if requested """
        self.add_function(action)

    # IMAGE MANIPULATIONS
    @staticmethod
    def avg_color_adjust(old_face, new_face, mask):
        """ Adjust the mean of the color channels to be the same for the swap and old frame """
        for _ in [0, 1]:
            diff = old_face - new_face
            avg_diff = np.sum(diff * mask, axis=(0, 1))
            adjustment = avg_diff / np.sum(mask, axis=(0, 1))
            new_face = new_face + adjustment
        return new_face

    def match_histogram(self, old_face, new_face, mask):
        """ Match the histogram of the color intensity of each channel """
        mask_indices = np.nonzero(mask.squeeze() > 0.99)
        threshold = self.func_constants["match_histogram"]["threshold"]
        new_face = [self.hist_match(old_face[:, :, c],
                                    new_face[:, :, c],
                                    mask_indices,
                                    threshold)
                    for c in range(3)]
        new_face = np.stack(new_face, axis=-1)
        return new_face

    @staticmethod
    def hist_match(old_channel, new_channel, mask_indices, threshold):
        """  Construct the histogram of the color intensity of a channel
             for the swap and the original. Match the histogram of the original
             by interpolation
        """
        if mask_indices[0].size == 0:
            return new_channel

        old_masked = old_channel[mask_indices]
        new_masked = new_channel[mask_indices]
        _, bin_idx, s_counts = np.unique(new_masked, return_inverse=True, return_counts=True)
        t_values, t_counts = np.unique(old_masked, return_counts=True)
        s_quants = np.cumsum(s_counts, dtype='float32')
        t_quants = np.cumsum(t_counts, dtype='float32')
        s_quants = threshold * s_quants / s_quants[-1]  # cdf
        t_quants /= t_quants[-1]  # cdf
        interp_s_values = np.interp(s_quants, t_quants, t_values)
        new_channel[mask_indices] = interp_s_values[bin_idx]
        return new_channel

    @staticmethod
    def seamless_clone(old_face, new_face, mask):
        """ Seamless clone the swapped face into the old face with cv2 """
        height, width, _ = old_face.shape
        height = height // 2
        width = width // 2

        y_indices, x_indices, _ = np.nonzero(mask)
        y_crop = slice(np.min(y_indices), np.max(y_indices))
        x_crop = slice(np.min(x_indices), np.max(x_indices))
        y_center = int(np.rint((np.max(y_indices) + np.min(y_indices)) / 2 + height))
        x_center = int(np.rint((np.max(x_indices) + np.min(x_indices)) / 2 + width))

        insertion = np.rint(new_face[y_crop, x_crop] * 255.0).astype("uint8")
        insertion_mask = np.rint(mask[y_crop, x_crop] * 255.0).astype("uint8")
        insertion_mask[insertion_mask != 0] = 255
        prior = np.rint(np.pad(old_face * 255.0,
                               ((height, height), (width, width), (0, 0)),
                               'constant')).astype("uint8")

        blended = cv2.seamlessClone(insertion,  # pylint: disable=no-member
                                    prior,
                                    insertion_mask,
                                    (x_center, y_center),
                                    cv2.NORMAL_CLONE)  # pylint: disable=no-member
        blended = blended[height:-height, width:-width]

        return blended.astype("float32") / 255.0

    def adjust(self, old_face, swapped_face, mask):
        """ Perform selected adjustments on face """
        for func in self.funcs:
            swapped_face = func(old_face, swapped_face, mask)
            swapped_face = np.clip(swapped_face, 0.0, 1.0)
        return swapped_face


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
