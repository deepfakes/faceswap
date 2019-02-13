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

        if self.args.draw_transparent:
            new_image = dfl_full(detected_face.landmarks_as_xy, new_image, channels=4 )#Add mask as 4th channel for saving as alpha on supported output formats

            #This make sure that all the arrays match in size for later actions despite not actually using alpha in any way.
            image_mask = cv2.cvtColor(image_mask, cv2.COLOR_RGB2RGBA)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)

        patched_face = self.apply_fixes(image,
                                        new_image,
                                        image_mask,
                                        image_size)

        logger.trace("Patched image")
        return patched_face

    def get_new_image(self, image, detected_face, coverage, image_size):
        """ Get the new face from the predictor """
        logger.trace("coverage: %s", coverage)
        src_face = detected_face.aligned_face
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
        new_face[crop, crop] = (mask * new_face + (1.0 - mask ) * old_face)

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
            mask = cv2.blur(mask, (blur_size, blur_size))  # pylint: disable=no-member

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
        blur_ratio = self.args.blur_size / 100
        mask_radius = np.sqrt(np.sum(mask)) / 2
        blur_size = int(max(1, blur_ratio * mask_radius))
        logger.trace("blur_size: %s", blur_size)
        return blur_size

    def apply_fixes(self, frame, new_image, image_mask, image_size):
        """ Apply fixes """

        if self.args.sharpen_image is not None and self.args.sharpen_image.lower() != "none":
            np.clip(new_image, 0.0, 255.0, out=new_image)
            if self.args.sharpen_image == "box_filter":
                kernel = np.ones((3, 3)) * (-1)
                kernel[1, 1] = 9
                new_image = cv2.filter2D(new_image, -1, kernel)  # pylint: disable=no-member
            elif self.args.sharpen_image == "gaussian_filter":
                blur = cv2.GaussianBlur(new_image, (0, 0), 3.0)  # pylint: disable=no-member
                new_image = cv2.addWeighted(new_image,  # pylint: disable=no-member
                                         1.5,
                                         blur,
                                         -0.5,
                                         0,
                                         new_image)

        if self.args.avg_color_adjust:
            for _ in [0, 1]:
                np.clip(new_image, 0.0, 255.0, out=new_image)
                diff = frame - new_image
                avg_diff = np.sum(diff * image_mask, axis=(0, 1))
                adjustment = avg_diff / np.sum(image_mask, axis=(0, 1))
                new_image = new_image + adjustment

        if self.args.match_histogram:
            np.clip(new_image, 0.0, 255.0, out=new_image)
            new_image = self.color_hist_match(new_image, frame, image_mask)

        if self.args.seamless_clone and not self.args.draw_transparent:
            h, w, _ = frame.shape
            h = h // 2
            w = w // 2

            y_indices, x_indices, _ = np.nonzero(image_mask)
            y_crop = slice(np.min(y_indices), np.max(y_indices))
            x_crop = slice(np.min(x_indices), np.max(x_indices))
            y_center = int(np.rint((np.max(y_indices) + np.min(y_indices)) / 2) + h)
            x_center = int(np.rint((np.max(x_indices) + np.min(x_indices)) / 2) + w)

            '''
            # test with average of centroid rather than the h /2 , w/2 center
            y_center = int(np.rint(np.average(y_indices) + h)
            x_center = int(np.rint(np.average(x_indices) + w)
            '''

            insertion = np.rint(new_image[y_crop, x_crop, :]).astype('uint8')
            insertion_mask = image_mask[y_crop, x_crop, :]
            insertion_mask[insertion_mask != 0] = 255
            insertion_mask = insertion_mask.astype('uint8')

            prior = np.pad(frame, ((h, h), (w, w), (0, 0)), 'constant').astype('uint8')

            blended = cv2.seamlessClone(insertion,  # pylint: disable=no-member
                                        prior,
                                        insertion_mask,
                                        (x_center, y_center),
                                        cv2.NORMAL_CLONE)  # pylint: disable=no-member
            blended = blended[h:-h, w:-w]

        else:
            foreground = new_image * image_mask
            background = frame * (1.0 - image_mask)
            blended = foreground + background

        np.clip(blended, 0.0, 255.0, out=blended)

        return np.rint(blended).astype('uint8')

    def color_hist_match(self, new, frame, image_mask):
        for channel in [0, 1, 2]:
            new[:, :, channel] = self.hist_match(new[:, :, channel],
                                                 frame[:, :, channel],
                                                 image_mask[:, :, channel])
        # source = np.stack([self.hist_match(source[:,:,c], target[:,:,c],image_mask[:,:,c])
        #                      for c in [0,1,2]],
        #                     axis=2)
        return new

    def hist_match(self, new, frame, image_mask):

        mask_indices = np.nonzero(image_mask)
        if len(mask_indices[0]) == 0:
            return new

        m_new = new[mask_indices].ravel()
        m_frame = frame[mask_indices].ravel()
        s_values, bin_idx, s_counts = np.unique(m_new, return_inverse=True, return_counts=True)
        t_values, t_counts = np.unique(m_frame, return_counts=True)
        s_quants = np.cumsum(s_counts, dtype='float32')
        t_quants = np.cumsum(t_counts, dtype='float32')
        s_quants /= s_quants[-1]  # cdf
        t_quants /= t_quants[-1]  # cdf
        interp_s_values = np.interp(s_quants, t_quants, t_values)
        new.put(mask_indices, interp_s_values[bin_idx])

        '''
        bins = np.arange(256)
        template_CDF, _ = np.histogram(m_frame, bins=bins, density=True)
        flat_new_image = np.interp(m_source.ravel(), bins[:-1], template_CDF) * 255.0
        return flat_new_image.reshape(m_source.shape) * 255.0
        '''

        return new


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

    def ellipse(self, **kwargs):
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
