#!/usr/bin/env python3
""" Adjust converter for faceswap.py

    Based on the original https://www.reddit.com/r/deepfakes/ code sample
    Adjust code made by https://github.com/yangchen8710 """

import cv2
import numpy as np

from lib.utils import add_alpha_channel


class Convert():
    """ Adjust Converter """
    def __init__(self, encoder, smooth_mask=True, avg_color_adjust=True,
                 draw_transparent=False, **kwargs):
        self.encoder = encoder

        self.use_smooth_mask = smooth_mask
        self.use_avg_color_adjust = avg_color_adjust
        self.draw_transparent = draw_transparent

    def patch_image(self, frame, detected_face, size):
        """ Patch swapped face onto original image """
        # pylint: disable=no-member
        # assert image.shape == (256, 256, 3)
        padding = 48
        face_size = 256
        detected_face.load_aligned(frame, face_size, padding,
                                   align_eyes=False)
        src_face = detected_face.aligned_face

        crop = slice(padding, face_size - padding)
        process_face = src_face[crop, crop]
        old_face = process_face.copy()

        process_face = cv2.resize(process_face,
                                  (size, size),
                                  interpolation=cv2.INTER_AREA)
        process_face = np.expand_dims(process_face, 0)

        new_face = self.encoder(process_face / 255.0)[0]
        new_face = np.clip(new_face * 255, 0, 255).astype(src_face.dtype)
        new_face = cv2.resize(
            new_face,
            (face_size - padding * 2, face_size - padding * 2),
            interpolation=cv2.INTER_CUBIC)

        if self.use_avg_color_adjust:
            self.adjust_avg_color(old_face, new_face)
        if self.use_smooth_mask:
            self.smooth_mask(old_face, new_face)

        new_face = self.superpose(src_face, new_face, crop)
        new_image = frame.copy()

        if self.draw_transparent:
            new_image, new_face = self.convert_transparent(new_image,
                                                           new_face)

        cv2.warpAffine(
            new_face,
            detected_face.adjusted_matrix,
            (detected_face.frame_dims[1], detected_face.frame_dims[0]),
            new_image,
            flags=cv2.WARP_INVERSE_MAP | cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_TRANSPARENT)
        return new_image

    @staticmethod
    def adjust_avg_color(old_face, new_face):
        """ Perform average color adjustment """
        for i in range(new_face.shape[-1]):
            old_avg = old_face[:, :, i].mean()
            new_avg = new_face[:, :, i].mean()
            diff_int = (int)(old_avg - new_avg)
            for int_h in range(new_face.shape[0]):
                for int_w in range(new_face.shape[1]):
                    temp = (new_face[int_h, int_w, i] + diff_int)
                    if temp < 0:
                        new_face[int_h, int_w, i] = 0
                    elif temp > 255:
                        new_face[int_h, int_w, i] = 255
                    else:
                        new_face[int_h, int_w, i] = temp

    @staticmethod
    def smooth_mask(old_face, new_face):
        """ Smooth the mask """
        width, height, _ = new_face.shape
        crop = slice(0, width)
        mask = np.zeros_like(new_face)
        mask[height // 15:-height // 15, width // 15:-width // 15, :] = 255
        mask = cv2.GaussianBlur(mask,  # pylint: disable=no-member
                                (15, 15),
                                10)
        new_face[crop, crop] = (mask / 255 * new_face +
                                (1 - mask / 255) * old_face)

    @staticmethod
    def superpose(src_face, new_face, crop):
        """ Crop Face """
        new_image = src_face.copy()
        new_image[crop, crop] = new_face
        return new_image

    @staticmethod
    def convert_transparent(image, new_face):
        """ Add alpha channels to images and change to
            transparent background """
        height, width = image.shape[:2]
        image = np.zeros((height, width, 4), dtype=np.uint8)
        new_face = add_alpha_channel(new_face, 100)
        return image, new_face
