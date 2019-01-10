#!/usr/bin/env python3
""" Adjust converter for faceswap.py

    Based on the original https://www.reddit.com/r/deepfakes/ code sample
    Adjust code made by https://github.com/yangchen8710 """

import cv2
import numpy
from lib.aligner import get_align_mat
from lib.utils import add_alpha_channel


class Convert():
    """ Adjust Converter """
    def __init__(self, encoder, smooth_mask=True, avg_color_adjust=True,
                 draw_transparent=False, **kwargs):
        self.encoder = encoder

        self.use_smooth_mask = smooth_mask
        self.use_avg_color_adjust = avg_color_adjust
        self.draw_transparent = draw_transparent

    def patch_image(self, frame, detected_face, encoder_size):
        """ Patch swapped face onto original image """
        # pylint: disable=no-member
        # assert image.shape == (256, 256, 3)
        coverage = 160
        training_face_size = 256
        image_size = frame.shape[1], frame.shape[0]
        padding = (training_face_size - coverage ) // 2
        crop = slice(padding, training_face_size - padding)
        
        #mat = umeyama(np.array(face.landmarks_as_xy[17:]),LANDMARKS_2D,True)[0:2]
        mat = get_align_mat(detected_face, training_face_size,False)
        matrix = mat * (training_face_size - 2 * padding)
        matrix[:, 2] += padding
        src_face = cv2.warpAffine(frame.astype('float32'), matrix, (training_face_size, training_face_size))
            
        process_face = src_face[crop, crop]
        process_face = cv2.resize(process_face,(encoder_size, encoder_size),interpolation=cv2.INTER_AREA)
        process_face = numpy.expand_dims(process_face, 0) / 255.0

        new_face = self.encoder(process_face)[0]
        numpy.clip(new_face * 255.0, 0.0, 255.0, out=new_face)
        new_face = cv2.resize(new_face,(training_face_size - padding * 2, training_face_size - padding * 2),interpolation=cv2.INTER_CUBIC)
        src_face[crop, crop] = new_face
        
        new_mask = frame.copy().astype('float32')
        new_image = frame.copy().astype('float32')
        cv2.warpAffine(src_face, matrix, image_size, new_image,
                       flags=cv2.WARP_INVERSE_MAP | cv2.INTER_CUBIC,
                       borderMode=cv2.BORDER_TRANSPARENT)
        
        mask = numpy.zeros((training_face_size, training_face_size,3), dtype='float32')
        area = padding + coverage // 15
        central_core = slice(area, -area)
        mask[central_core, central_core,:] = 1.0
        mask = cv2.GaussianBlur(mask, (21, 21), 10)
                
        cv2.warpAffine(mask, matrix, image_size, new_mask,
                                flags = cv2.WARP_INVERSE_MAP | cv2.INTER_CUBIC,
                                borderMode = cv2.BORDER_CONSTANT,
                                borderValue = 0.0)
                                                                            
        frame = new_mask * new_image + (1 - new_mask) * frame
        return new_mask * 255


