#!/usr/bin/env python3
""" Masked converter for faceswap.py
    Based on: https://gist.github.com/anonymous/d3815aba83a8f79779451262599b0955
    found on https://www.reddit.com/r/deepfakes/ """

import logging
import cv2
import numpy as np
from lib.model.masks import dfl_full

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class Convert():
    """ Swap a source face with a target """
    def __init__(self, encoder, model, arguments):
        logger.debug("Initializing %s: (encoder: '%s', model: %s, arguments: %s",
                     self.__class__.__name__, encoder, model, arguments)
        self.encoder = encoder
        self.args = arguments
        self.input_size = model.input_shape[0]
        self.training_size = model.state.training_size
        self.training_coverage_ratio = model.training_opts["coverage_ratio"]
        self.input_mask_shape = model.state.mask_shapes[0] if model.state.mask_shapes else None
        self.crop = None
        self.mask = None
        logger.debug("Initialized %s", self.__class__.__name__)

    def patch_image(self, image, detected_face):
        """ Patch the image """
        logger.trace("Patching image")
        image = image.astype('float32')
        image_size = (image.shape[1], image.shape[0])
        coverage = int(self.training_coverage_ratio * self.training_size)
        padding = (self.training_size - coverage) // 2
        logger.trace("coverage: %s, padding: %s", coverage, padding)

        self.crop = slice(padding, self.training_size - padding)
        if not self.mask:  # Init the mask on first image
            self.mask = Mask(self.args.mask_type, self.training_size, padding, self.crop)

        detected_face.load_aligned(image, size=self.training_size, align_eyes=False)
        new_image = self.get_new_image(image, detected_face, coverage, image_size)
        image_mask = self.get_image_mask(detected_face, image_size)
        patched_face = self.apply_fixes(image, new_image, image_mask,
                                        detected_face.landmarks_as_xy)

        logger.trace("Patched image")
        return patched_face

    def get_new_image(self, image, detected_face, coverage, image_size):
        """ Get the new face from the predictor """
        logger.trace("coverage: %s", coverage)
        src_face = detected_face.aligned_face[:, :, :3]
        coverage_face = src_face[self.crop, self.crop]
        old_face = coverage_face.copy()
        coverage_face = cv2.resize(coverage_face,  # pylint: disable=no-member
                                   (self.input_size, self.input_size),
                                   interpolation=cv2.INTER_AREA)  # pylint: disable=no-member
        coverage_face = np.expand_dims(coverage_face, 0)
        np.clip(coverage_face / 255.0, 0.0, 1.0, out=coverage_face)

        if self.input_mask_shape:
            mask = np.zeros(self.input_mask_shape, np.float32)
            mask = np.expand_dims(mask, 0)
            feed = [coverage_face, mask]
        else:
            feed = [coverage_face]
        logger.trace("Input shapes: %s", [item.shape for item in feed])
        new_face = self.encoder(feed)[0]
        new_face = new_face.squeeze()
        logger.trace("Output shape: %s", new_face.shape)

        new_face = cv2.resize(new_face,  # pylint: disable=no-member
                              (coverage, coverage),
                              interpolation=cv2.INTER_CUBIC)  # pylint: disable=no-member
        np.clip(new_face * 255.0, 0.0, 255.0, out=new_face)

        if self.args.smooth_box:
            self.smooth_box(old_face, new_face)

        src_face[self.crop, self.crop] = new_face
        background = image.copy()
        interpolator = detected_face.adjusted_interpolators[1]
        new_image = cv2.warpAffine(  # pylint: disable=no-member
            src_face,
            detected_face.adjusted_matrix,
            image_size,
            background,
            flags=cv2.WARP_INVERSE_MAP | interpolator,  # pylint: disable=no-member
            borderMode=cv2.BORDER_TRANSPARENT)  # pylint: disable=no-member
        return new_image

    @staticmethod
    def smooth_box(old_face, new_face):
        """ Perform gaussian blur on the edges of the output rect """
        height = new_face.shape[0]
        crop = slice(0, height)
        erode = slice(height // 15, -height // 15)
        sigma = height / 16 # 10 for the default 160 size
        window = int(np.ceil(sigma * 3.0))
        window = window + 1 if window % 2 == 0 else window
        mask = np.zeros_like(new_face)
        mask[erode, erode] = 1.0
        mask = cv2.GaussianBlur(mask,  # pylint: disable=no-member
                                (window, window),
                                sigma)
        new_face[crop, crop] = (mask * new_face + (1.0 - mask) * old_face)

    def get_image_mask(self, detected_face, image_size):
        """ Get the image mask """
        mask = self.mask.get_mask(detected_face, image_size)
        if self.args.erosion_size != 0:
            kwargs = {'src': mask,
                      'kernel': self.set_erosion_kernel(mask),
                      'iterations': 1}
            if self.args.erosion_size > 0:
                mask = cv2.erode(**kwargs)  # pylint: disable=no-member
            else:
                mask = cv2.dilate(**kwargs)  # pylint: disable=no-member

        if self.args.blur_size != 0:
            blur_size = self.set_blur_size(mask)
            for _ in [1,2,3,4]: # pylint: disable=no-member
                mask = cv2.blur(mask, (blur_size, blur_size)) 

        return np.clip(mask, 0.0, 1.0, out=mask)

    def set_erosion_kernel(self, mask):
        """ Set the erosion kernel """
        erosion_ratio = self.args.erosion_size / 100
        mask_radius = np.sqrt(np.sum(mask)) / 2
        percent_erode = max(1, int(abs(erosion_ratio * mask_radius)))
        erosion_kernel = cv2.getStructuringElement(  # pylint: disable=no-member
            cv2.MORPH_ELLIPSE,  # pylint: disable=no-member
            (percent_erode, percent_erode))
        logger.trace("erosion_kernel shape: %s", erosion_kernel.shape)
        return erosion_kernel

    def set_blur_size(self, mask):
        """ Set the blur size to absolute or percentage """
        blur_ratio = self.args.blur_size / 100 / 1.6
        mask_radius = np.sqrt(np.sum(mask)) / 2
        blur_size = int(max(1, blur_ratio * mask_radius))
        logger.trace("blur_size: %s", blur_size)
        return blur_size

    def apply_fixes(self, original, face, mask, landmarks):
        """ Apply fixes """
        #TODO copies aren't likey neccesary and will slow calc... used when isolating issues
        new_image = face[:, :, :3].copy()
        image_mask = mask[:, :, :3].copy()
        frame = original[:, :, :3].copy()

        #TODO - force default for args.sharpen_image to ensure it isn't None
        if self.args.sharpen_image is not None and self.args.sharpen_image.lower() != "none":
            new_image = self.sharpen(new_image, self.args.sharpen_image)

        if self.args.avg_color_adjust:
            new_image = self.color_adjust(new_image, frame, image_mask)

        if self.args.match_histogram:
            new_image = self.color_hist_match(new_image, frame, image_mask)

        if self.args.seamless_clone:
            blended = self.seamless_clone(new_image, frame, image_mask)
        else:
            foreground = new_image * image_mask
            background = frame * (1.0 - image_mask)
            blended = foreground + background

        np.clip(blended, 0.0, 255.0, out=blended)
        if self.args.draw_transparent:
            # Adding a 4th channel should happen after all other channel operations
            # Add mask as 4th channel for saving as alpha on supported output formats
            blended = dfl_full(landmarks, blended, channels=4)

        return np.rint(blended).astype('uint8')

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
            new = cv2.addWeighted(new, 1.5, blur, -.5, 0, new) # pylint: disable=no-member

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
        new = np.stack((self.hist_match(new[:, :, c], frame[:, :, c], img_mask[:, :, c]) for c in [0, 1, 2]), axis=-1)

        return new

    @staticmethod
    def hist_match(new, frame, img_mask):
        """  Construct the histogram of the color intensity of a channel
             for the swap and the original. Match the histogram of the original
             by interpolation
        """
        mask_indices = np.nonzero(img_mask)
        if len(mask_indices[0]) == 0:
            return new

        m_new = new[mask_indices]
        m_frame = frame[mask_indices]
        _, bin_idx, s_counts = np.unique(m_new, return_inverse=True, return_counts=True)
        t_values, t_counts = np.unique(m_frame, return_counts=True)
        s_quants = np.cumsum(s_counts, dtype='float32')
        t_quants = np.cumsum(t_counts, dtype='float32')
        s_quants /= s_quants[-1]  # cdf
        t_quants /= t_quants[-1]  # cdf
        interp_s_values = np.interp(s_quants, t_quants, t_values)
        new[mask_indices] = interp_s_values[bin_idx]

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

class Mask():
    """ Return the requested mask """

    def __init__(self, mask_type, training_size, padding, crop):
        """ Set requested mask """
        logger.debug("Initializing %s: (mask_type: '%s', training_size: %s, padding: %s)",
                     self.__class__.__name__, mask_type, training_size, padding)

        self.training_size = training_size
        self.padding = padding
        self.mask_type = mask_type
        self.crop = crop

        logger.debug("Initialized %s", self.__class__.__name__)

    def get_mask(self, detected_face, image_size):
        """ Return a face mask """
        kwargs = {"matrix": detected_face.adjusted_matrix,
                  "interpolators": detected_face.adjusted_interpolators,
                  "landmarks": detected_face.landmarks_as_xy,
                  "image_size": image_size}
        logger.trace("kwargs: %s", kwargs)
        mask = getattr(self, self.mask_type)(**kwargs)
        mask = self.finalize_mask(mask)
        logger.trace("mask shape: %s", mask.shape)
        return mask

    def cnn(self, **kwargs):
        """ CNN Mask """
        # Insert FCN-VGG16 segmentation mask model here
        logger.info("cnn not yet implemented, using facehull instead")
        return self.facehull(**kwargs)

    def rect(self, **kwargs):
        """ Namespace for rect mask. This is the same as 'none' in the cli """
        return self.none(**kwargs)

    def none(self, **kwargs):
        """ Rect Mask """
        logger.trace("Getting mask")
        interpolator = kwargs["interpolators"][1]
        ones = np.zeros((self.training_size, self.training_size, 3), dtype='float32')
        mask = np.zeros((kwargs["image_size"][1], kwargs["image_size"][0], 3), dtype='float32')
        # central_core = slice(self.padding, -self.padding)
        ones[self.crop, self.crop] = 1.0
        cv2.warpAffine(ones,  # pylint: disable=no-member
                       kwargs["matrix"],
                       kwargs["image_size"],
                       mask,
                       flags=cv2.WARP_INVERSE_MAP | interpolator,  # pylint: disable=no-member
                       borderMode=cv2.BORDER_CONSTANT,  # pylint: disable=no-member
                       borderValue=0.0)
        return mask

    def dfl(self, **kwargs):
        """ DFaker Mask """
        logger.trace("Getting mask")
        dummy = np.zeros((kwargs["image_size"][1], kwargs["image_size"][0], 3), dtype='float32')
        mask = dfl_full(kwargs["landmarks"], dummy, channels=3)
        mask = self.intersect_rect(mask, **kwargs)
        return mask

    def facehull(self, **kwargs):
        """ Facehull Mask """
        logger.trace("Getting mask")
        mask = np.zeros((kwargs["image_size"][1], kwargs["image_size"][0], 3), dtype='float32')
        hull = cv2.convexHull(  # pylint: disable=no-member
            np.array(kwargs["landmarks"]).reshape((-1, 2)))
        cv2.fillConvexPoly(mask,  # pylint: disable=no-member
                           hull,
                           (1.0, 1.0, 1.0),
                           lineType=cv2.LINE_AA)  # pylint: disable=no-member
        mask = self.intersect_rect(mask, **kwargs)
        return mask

    @staticmethod
    def ellipse(**kwargs):
        """ Ellipse Mask """
        logger.trace("Getting mask")
        mask = np.zeros((kwargs["image_size"][1], kwargs["image_size"][0], 3), dtype='float32')
        ell = cv2.fitEllipse(  # pylint: disable=no-member
            np.array(kwargs["landmarks"]).reshape((-1, 2)))
        cv2.ellipse(mask,  # pylint: disable=no-member
                    box=ell,
                    color=(1.0, 1.0, 1.0),
                    thickness=-1)
        return mask

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
