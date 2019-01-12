#!/usr/bin/env python3
""" Masked converter for faceswap.py
    Based on: https://gist.github.com/anonymous/d3815aba83a8f79779451262599b0955
    found on https://www.reddit.com/r/deepfakes/ """

import logging
import cv2
import numpy
numpy.set_printoptions(threshold=numpy.nan)

from lib.aligner import get_align_mat
from lib.utils import add_alpha_channel

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class Convert():
    def __init__(self, encoder, trainer,
                 blur_size=2, seamless_clone=False, mask_type="facehullandrect",
                 erosion_size=0, match_histogram=False, sharpen_image=None,
                 draw_transparent=False, avg_color_adjust=False, coverage = 160,
                 input_size=64,**kwargs):
        self.encoder = encoder
        self.trainer = trainer
        self.blur_size = blur_size
        self.input_size = input_size
        self.coverage = coverage
        self.sharpen_image = sharpen_image
        self.match_histogram = match_histogram
        self.mask_type = mask_type.lower()
        self.draw_transparent = draw_transparent
        self.avg_color_adjust = avg_color_adjust
        self.erosion_size = erosion_size
        self.seamless_clone = False if draw_transparent else seamless_clone
        if abs(self.erosion_size) >= 1:
            e_size = (int(abs(self.erosion_size)),int(abs(self.erosion_size)))
            self.erosion_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                            e_size)
        
    def patch_image(self, image, face_detected, encoder_size):
        image_size = image.shape[1], image.shape[0]  #width,height
        image = image.astype('float32')
        training_size = 256
        align_eyes = False
        coverage = 160
        
        mat = get_align_mat(face_detected, training_size, align_eyes)
        padding = (training_size - coverage ) // 2
        crop = slice(padding, training_size - padding)
        matrix = mat * (training_size - 2 * padding)
        matrix[:, 2] += padding
        
        interpolators = self.get_matrix_scaling(matrix)
        
        new_image = self.get_new_image(image, matrix, crop, padding, training_size,
                                     encoder_size, image_size, interpolators)
                                     
        image_mask = self.get_image_mask(matrix, image_size, padding,
                                         training_size, interpolators,
                                         face_detected.landmarks_as_xy)
                                         
        patched_face = self.apply_fixes(image, new_image, image_mask, image_size)
        
        return patched_face

    def get_matrix_scaling(self, mat):
        x_scale = numpy.sqrt(mat[0,0]*mat[0,0] + mat[0,1]*mat[0,1])
        y_scale = ( mat[0,0] * mat[1,1] - mat[0,1] * mat[1,0] ) / x_scale 
        avg_scale = ( x_scale + y_scale ) * 0.5
        interpolator = cv2.INTER_CUBIC if avg_scale > 1.0 else cv2.INTER_AREA
        inverse_interpolator = cv2.INTER_AREA if avg_scale > 1.0 else cv2.INTER_CUBIC
        
        return interpolator, inverse_interpolator
        
    def get_new_image(self, image, mat, crop, padding, training_size, encoder_size, image_size, interpolators):
        src_face = cv2.warpAffine(image, mat, (training_size, training_size),
                                  flags = interpolators[0])
        coverage_face = src_face[crop, crop]
        coverage_face = cv2.resize(coverage_face, (encoder_size, encoder_size),
                                   interpolation = interpolators[0])
        coverage_face = numpy.expand_dims(coverage_face, 0)
        numpy.clip(coverage_face / 255.0, 0.0, 1.0, out=coverage_face)
        
        if 'GAN' in self.trainer:
            # change code to align with new GAN code
            print('error')
        else:
            new_face = self.encoder(coverage_face)[0]
            
        new_face = cv2.resize(new_face,(training_size - padding * 2, training_size - padding * 2),interpolation=cv2.INTER_CUBIC)
        numpy.clip(new_face * 255.0, 0.0, 255.0, out=new_face)
        src_face[crop, crop] = new_face
        
        background = image.copy()
        new_image = cv2.warpAffine(src_face, mat, image_size, background,
                                   flags = cv2.WARP_INVERSE_MAP | 
                                           interpolators[1],
                                   borderMode=cv2.BORDER_TRANSPARENT)
        return new_image

    def get_image_mask(self, mat, image_size, padding, training_size, interpolators, landmarks):
        
        mask = numpy.ones((image_size[1],image_size[0], 3),dtype='float32')

        if 'cnn' == self.mask_type:
            # Insert FCN-VGG16 segmentation mask model here
            self.mask_type = 'cnn'
            
        if 'smoothed' == self.mask_type:
            ones = numpy.zeros((training_size, training_size, 3), dtype='float32')
            area = padding + ( training_size - 2 * padding ) // 15
            central_core = slice(area, -area)
            ones[central_core, central_core,:] = 1.0
            ones = cv2.GaussianBlur(ones, (25,25), 10)
            mask = numpy.zeros((image_size[1],image_size[0], 3),dtype='float32')
            cv2.warpAffine(ones, mat, image_size, mask,
                           flags = cv2.WARP_INVERSE_MAP | interpolators[1],
                           borderMode = cv2.BORDER_CONSTANT, borderValue = 0.0)
        
        if 'rect' in self.mask_type:
            ones = numpy.zeros((training_size, training_size, 3), dtype='float32')
            central_core = slice(padding,-padding)
            ones[central_core,central_core] = 1.0
            mask = numpy.zeros((image_size[1],image_size[0], 3),dtype='float32')
            cv2.warpAffine(ones, mat, image_size, mask,
                           flags = cv2.WARP_INVERSE_MAP | interpolators[1],
                           borderMode = cv2.BORDER_CONSTANT, borderValue = 0.0)
                                        
        if 'facehull' in self.mask_type:
            hull_mask = numpy.zeros((image_size[1],image_size[0], 3),dtype='float32')
            hull = cv2.convexHull(numpy.array(landmarks).reshape((-1, 2)))
            cv2.fillConvexPoly(hull_mask, hull, (1.0, 1.0, 1.0), lineType = cv2.LINE_AA)
            mask *= hull_mask
        
        if 'ellipse' in self.mask_type:
            mask = numpy.zeros((image_size[1],image_size[0], 3),dtype='float32')
            e = cv2.fitEllipse(numpy.array(landmarks).reshape((-1, 2)))
            cv2.ellipse(mask, box=e, color=(1.0,1.0,1.0), thickness=-1)
        
        numpy.nan_to_num(mask, copy=False)
        numpy.clip(mask, 0.0, 1.0, out=mask)
        
        if self.erosion_size != 0:
            if abs(self.erosion_size) < 1.0:
                mask_radius = numpy.sqrt(numpy.sum(mask)) / 2
                percent_erode = max(1,int(abs(self.erosion_size * mask_radius)))
                self.erosion_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                                (percent_erode,
                                                                percent_erode))
            args = {'src':mask, 'kernel':self.erosion_kernel, 'iterations':1}
            mask = cv2.erode(**args) if self.erosion_size > 0 else cv2.dilate(**args)
            
        if self.blur_size != 0:
            if self.blur_size < 1.0:
                mask_radius = numpy.sqrt(numpy.sum(mask)) / 2
                self.blur_size = max(1,int(self.blur_size * mask_radius))
            mask = cv2.blur(mask, (int(self.blur_size),int(self.blur_size)))
            
        return numpy.clip(mask, 0.0, 1.0, out=mask)
        
    def apply_fixes(self, image, new_image, image_mask, image_size):
    
        masked = new_image * image_mask
        
        if self.draw_transparent:
            alpha = numpy.full((image_size[1],image_size[0],1), 
                               255.0,
                               dtype='float32')
            new_image = numpy.concatenate(new_image, alpha, axis=2)
            image_mask = numpy.concatenate(image_mask, alpha, axis=2)
            image = numpy.concatenate(image, alpha, axis=2)
            
        if self.sharpen_image is not None:
            numpy.clip(masked, 0.0, 255.0, out=masked)
            if self.sharpen_image == "box_filter":
                kernel = numpy.ones((3, 3)) * (-1)
                kernel[1, 1] = 9
                masked = cv2.filter2D(masked, -1, kernel)
            elif self.sharpen_image == "gaussian_filter":
                blur = cv2.GaussianBlur(masked, (0, 0), 3.0)
                masked = cv2.addWeighted(masked, 1.5, blur, -0.5, 0, masked)
        
        if self.avg_color_adjust:
            for iterations in [0,1]:
                numpy.clip(masked, 0.0, 255.0, out=masked)
                old_avgs = numpy.average(image * image_mask, axis=(0,1))
                new_avgs = numpy.average(masked, axis=(0,1))
                diff = old_avgs - new_avgs
                masked = masked + diff  
                
        if self.match_histogram:
            numpy.clip(masked, 0.0, 255.0, out=masked)
            masked = self.color_hist_match(masked, image, image_mask)
                                              
        if self.seamless_clone:
            #indices = numpy.argwhere(image_mask != 0)
            indices = numpy.nonzero(image_mask)
            x_center = numpy.mean(indices[1], dtype='uint8')
            y_center = numpy.mean(indices[0], dtype='uint8')
            print('here')
            blended = cv2.seamlessClone(masked.astype('uint8'),
                                        image.astype('uint8'),
                                        image_mask.astype('uint8'),
                                        (x_center, y_center),
                                        cv2.MIXED_CLONE)
        else:
            foreground = masked * image_mask
            background = image * ( 1.0 - image_mask )
            blended = foreground + background
        
        numpy.clip(blended, 0.0, 255.0, out=blended)

        return numpy.rint(blended).astype('uint8')
        
    def color_hist_match(self, source, target, image_mask):
        for channel in [0,1,2]:
            source[:, :, channel] = self.hist_match(source[:, :, channel],
                                                    target[:, :, channel],
                                                    image_mask[:, :, channel])
        return source
        
    def hist_match(self, source, template, image_mask):
        '''
        outshape = source.shape
        source = source.ravel()
        template = template.ravel()
        s_values, bin_idx, s_counts = numpy.unique(source,
                                                   return_inverse=True,
                                                   return_counts=True)
        t_values, t_counts = numpy.unique(template, return_counts=True)
        s_quants = numpy.cumsum(s_counts, dtype='float32')
        t_quants = numpy.cumsum(t_counts, dtype='float32')
        s_quants /= s_quants[-1]  # cdf
        t_quants /= t_quants[-1]  # cdf
        interp_s_values = numpy.interp(s_quants, t_quants, t_values)
        source = interp_s_values[bin_idx].reshape(outshape)
        '''
        bins = numpy.arange(256.0) #+1
        print('source', numpy.max(source), numpy.min(source))
        print('template', numpy.max(template), numpy.min(template))
        #source_CDF, _ = numpy.histogram(source, bins=bins, range=[0,256], density=True, weights=image_mask)
        template_CDF, _ = numpy.histogram(template, bins=bins, density=True) # weights=image_mask) 
        #print(source_CDF, template_CDF)
        #new_pixels = numpy.interp(source_CDF, template_CDF, bins[:-1])
        #source = new_pixels[source.astype('uint8').ravel()].reshape(source.shape)
        flat_new_image = numpy.interp(source.ravel(), bins[:-1], template_CDF) * 255.0
        
        return flat_new_image.reshape(source.shape) * 255.0# source