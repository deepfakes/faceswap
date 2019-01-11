#!/usr/bin/env python3
""" Masked converter for faceswap.py
    Based on: https://gist.github.com/anonymous/d3815aba83a8f79779451262599b0955
    found on https://www.reddit.com/r/deepfakes/ """

import logging
import cv2
import numpy

from lib.aligner import get_align_mat
from lib.utils import add_alpha_channel

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class Convert():
    def __init__(self, encoder, trainer,
                 blur_size=2, seamless_clone=False, mask_type="facehullandrect",
                 erosion_kernel_size=0, match_histogram=False, sharpen_image=None,
                 draw_transparent=False, avg_color_adjust=False, enlargement_scale = 0.0,
                 input_size=64,**kwargs):
        self.encoder = encoder
        self.trainer = trainer
        self.blur_size = blur_size
        self.input_size = input_size
        self.enlargement_scale = enlargement_scale
        self.sharpen_image = sharpen_image
        self.match_histogram = match_histogram
        self.mask_type = mask_type.lower()
        self.draw_transparent = draw_transparent
        self.avg_color_adjust = avg_color_adjust
        self.erosion_kernel_size = erosion_kernel_size
        self.seamless_clone = False if draw_transparent else seamless_clone
        self.erosion_size = 0 if erosion_kernel_size is None else erosion_kernel_size
        if abs(self.erosion_size) >= 1:
            e_size = (int(abs(self.erosion_size)),int(abs(self.erosion_size)))
            self.erosion_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                            e_size)
        
    def patch_image(self, image, face_detected, size):
        image_size = image.shape[1], image.shape[0]
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
        new_face = self.get_new_face(image, matrix, training_size,
                                     encoder_size, interpolators)
                                     
        image_mask = self.get_image_mask(matrix, image_size,
                                         training_size,  new_face,
                                         interpolators,
                                         face_detected.landmarks_as_xy)
                                         
        patched_face = self.apply_new_face(image, matrix, training_size,
                                           image_size, new_face, image_mask)
        
        return patched_face

    def get_matrix_scaling(self, mat):
        x_scale = numpy.sqrt(mat[0,0]*mat[0,0] + mat[0,1]*mat[0,1])
        y_scale = ( mat[0,0] * mat[1,1] - mat[0,1] * mat[1,0] ) / x_scale 
        avg_scale = ( x_scale + y_scale ) * 0.5
        interpolator = cv2.INTER_CUBIC if avg_scale > 1.0 else cv2.INTER_AREA
        inverse_interpolator = cv2.INTER_AREA if avg_scale > 1.0 else cv2.INTER_CUBIC
        
        return interpolator, inverse_interpolator
        
    def get_new_face(self, image, mat, training_size, encoder_size, interpolators):
        src_face = cv2.warpAffine(image, mat,
                              (training_size, training_size),
                              flags = interpolators[0])
        coverage_face = src_face[crop, crop]
        coverage_face = cv2.resize(coverage_face,
                                   (encoder_size, encoder_size),
                                   interpolation=interpolators[0])
        coverage_face = numpy.expand_dims(coverage_face, 0)
        numpy.clip(coverage_face / 255.0, 0.0, 1.0, out=norm_face)
        
        if 'GAN' in self.trainer:
            # change code to align with new GAN code
            print('error')
        else:
            new_face = self.encoder(norm_face)[0]
            
        new_face = cv2.resize(new_face,(training_face_size - padding * 2, training_face_size - padding * 2),interpolation=cv2.INTER_CUBIC)
            
        numpy.clip(new_face * 255.0, 0.0, 255.0, out=new_face)
        return new_face

    def get_image_mask(self, mat, image_size, training_size, new_face, inv_interpolator, landmarks):
        
        mask = numpy.ones((image_size, image_size, 3),dtype='float32')

        if 'cnn' == self.mask_type:
            # Insert FCn-VGG16 segmentation mask model here
            
        if 'dfaker' == self.mask_type:
            mask = numpy.zeros((training_face_size, training_face_size,3), dtype='float32')
            area = padding + coverage // 15
            central_core = slice(area, -area)
            mask[central_core, central_core,:] = 1.0
            mask = cv2.GaussianBlur(mask, (21, 21), 10)
            cv2.warpAffine(ones, mat, image_size, mask,
                           flags = cv2.WARP_INVERSE_MAP | interpolators[1],
                           borderMode = cv2.BORDER_CONSTANT, borderValue = 0.0)
        
        if 'rect' in self.mask_type:
            ones = numpy.ones((training_size, training_size, 3), dtype='float32')
            mask = numpy.zeros((image_size, image_size, 3),dtype='float32')
            cv2.warpAffine(ones, mat, image_size, mask,
                           flags = cv2.WARP_INVERSE_MAP | interpolators[1],
                           borderMode = cv2.BORDER_CONSTANT, borderValue = 0.0)
                                        
        if 'facehull' in self.mask_type:
            hull_mask = numpy.zeros((image_size, image_size, 3),dtype='float32')
            hull = cv2.convexHull(numpy.array(landmarks).reshape((-1, 2)))
            cv2.fillConvexPoly(hull_mask, hull, (1.0, 1.0, 1.0), lineType = cv2.LINE_AA)
            mask *= hull_mask
        
        if 'ellipse' in self.mask_type:
            e = cv2.fitEllipse(mask)
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
        
    def apply_new_face(self, image, mat, size, image_size, new_face, image_mask):
        outimage = None
        if self.draw_transparent:
            alpha = numpy.full((image_size[1],image_size[0],1), 
                               255.0,
                               dtype='float32')
            new_face = numpy.concatenate(new_face, alpha, axis=2)
            image_mask = numpy.concatenate(image_mask, alpha, axis=2)
            image = numpy.zeros((image_size[1],image_size[0], 4), dtype='float32')
        
        # bug / issues with transparent border not working
        # test further , cleanup mask replace for now
        # paste onto previous image in place
        new_image = cv2.warpAffine(new_face, mat, image_size,
                                   flags = cv2.WARP_INVERSE_MAP | 
                                           interpolators[1],
                                   borderMode = cv2.BORDER_CONSTANT,
                                   borderValue = (-1.0,-1.0,-1.0))
                                   
        cleanup_mask = (new_image == (-1.0,-1.0,-1.0))
        new_image[cleanup_mask] = image[cleanup_mask]
                       
        if self.sharpen_image is not None:
            numpy.clip(new_image, 0.0, 255.0, out=new_image)
            if self.sharpen_image == "box_filter":
                kernel = numpy.ones((3, 3)) * (-1)
                kernel[1, 1] = 9
                new_image = cv2.filter2D(new_image, -1, kernel)
            elif self.sharpen_image == "gaussian_filter":
                blur = cv2.GaussianBlur(new_image, (0, 0), 3.0)
                new_image = cv2.addWeighted(new_image, 1.5, blur, -0.5, 0, new_image)
        
        if self.avg_color_adjust:
            for iterations in [0,1]:
                numpy.clip(new_image, 0.0, 255.0, out=new_image)
                old_avgs = numpy.average(image * image_mask, axis=(0,1))
                new_avgs = numpy.average(new_image * image_mask, axis=(0,1))
                diff = old_avgs - new_avgs
                new_image = new_image + diff  
                
        if self.match_histogram:
            numpy.clip(new_image, 0.0, 255.0, out=new_image)
            new_image = self.color_hist_match(new_image,
                                              image,
                                              image_mask)
                                              
        if self.seamless_clone:
            region = numpy.argwhere(image_mask != 0)
            unitMask = image_mask[region] = 1
            if region.size > 0:
                x_center, y_center = region.mean(axis=0)
                cv2.seamlessClone(numpy.rint(new_image).astype('uint8'),
                                  image,
                                  unitMask,
                                  (x_center, y_center),
                                  blended,
                                  cv2.MIXED_CLONE)
        else:
            foreground = image_mask * new_image #new_image
            background = ( 1.0 - image_mask ) * image
            blended = foreground + background
        
        numpy.clip(blended, 0.0, 255.0, out=blended)

        return numpy.rint(blended).astype('uint8')
        
    def color_hist_match(self, source, target, mask):
        for channel in [0,1,2]:
            source[:, :, channel] = self.hist_match(source[:, :, channel],
                                                    target[:, :, channel],
                                                    mask[:, :, channel])
        return source
        
    def hist_match(self, source, template, mask=None):
        if mask is not None:
            source = source * mask
            template = template * mask
        
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
        #source = interp_s_values[bin_idx].reshape(outshape)
        '''
        
        bins = numpy.arange(256)
        #source_CDF, _ = np.histogram(source, bins=bins, density=True)
        template_CDF, _ = np.histogram(template, bins=bins, density=True)
        #new_pixels = numpy.interp(source_CDF, template_CDF, bins)
        #source = new_pixels[source.ravel()].astype('float32').reshape(source.shape)
        
        flat_new_image = numpy.interp(source.ravel(), bins, template_CDF*255.0)
        
        return flat_new_image.reshape(source.shape) # source