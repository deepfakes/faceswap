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
        mat = get_align_mat(face_detected,self.input_size,False)
        image = image.astype('float32')
        
        # insert Field of View Logic here to modify alignment mat
        # test various enlargment factors to umeyama's standard face
        

        self.enlargement_scale = 3.0/64.0     # @ eyebrows .. coverage = 160 ~ could have accuracy gains here...!
        #enlargement_scale = 3.0/64.0  @ temples  .. coverage = 180 test more
        #enlargement_scale = 6.0/64.0  @ ears     .. coverage = 200 test more
        #enlargement_scale = 12.0/64.0  @ mugshot  .. coverage = 220
        padding = int(self.enlargement_scale*size)
        mat = mat * (size - 2 * padding)
        mat[:, 2] += padding
        
        self.interpolator , self.inv_interpolator = self.get_matrix_scaling(mat)
        new_face = self.get_new_face(image, mat, size)
        image_mask = self.get_image_mask(image, mat, image_size, new_face,
                                         face_detected.landmarks_as_xy)
        patched_face = self.apply_new_face(image, mat, self.input_size,
                                           image_size, new_face, image_mask)
        
        return patched_face

    def get_matrix_scaling(self, mat):
        x_scale = numpy.sqrt(mat[0,0]*mat[0,0] + mat[0,1]*mat[0,1])
        y_scale = ( mat[0,0] * mat[1,1] - mat[0,1] * mat[1,0] ) / x_scale 
        avg_scale = ( x_scale + y_scale ) * 0.5
        interpolator = cv2.INTER_CUBIC if avg_scale > 1.0 else cv2.INTER_AREA
        inverse_interpolator = cv2.INTER_AREA if avg_scale > 1.0 else cv2.INTER_CUBIC
        
        return interpolator, inverse_interpolator
        
    def get_new_face(self, image, mat, size):
        face = cv2.warpAffine(image,mat,(size, size),flags = self.interpolator)
        face = numpy.expand_dims(face, 0)
        numpy.clip(face, 0.0, 255.0, out=face)
        new_face = None
        mask = None
        
        if 'GAN' in self.trainer:
            # change code to align with new GAN code
            '''
            normalized_face = face / 255.0 * 2.0 - 1.0
            fake_output = self.encoder(normalized_face)
            if "128" in self.trainer:
                fake_output = fake_output[0]
            mask = fake_output[:, :, :, :1]
            new_face = fake_output[:, :, :, 1:]
            new_face = mask * new_face + (1.0 - mask) * normalized_face
            new_face = (new_face[0] + 1.0) / 2.0
            '''
        else:
            normalized_face = face / 255.0
            new_face = self.encoder(normalized_face)[0]
            
        numpy.clip(new_face * 255.0, 0.0, 255.0, out=new_face)
        return new_face

    def get_image_mask(self, image, mat, image_size, new_face, landmarks):
        if 'rect' in self.mask_type:
            face_src = numpy.ones_like(new_face)
            image_mask = cv2.warpAffine(face_src, mat, image_size,
                                        flags = cv2.WARP_INVERSE_MAP |
                                                self.inv_interpolator,
                                        borderMode = cv2.BORDER_CONSTANT,
                                        borderValue = 0.0)
                                        
        if 'hull' in self.mask_type:
            hull_mask = numpy.zeros_like(image)
            hull = cv2.convexHull(numpy.array(landmarks).reshape((-1, 2)))
            cv2.fillConvexPoly(hull_mask, hull, (1.0, 1.0, 1.0), lineType = cv2.LINE_AA)
            image_mask = hull_mask if self.mask_type == 'facehullandrect' else hull_mask    

        numpy.nan_to_num(image_mask, copy=False)
        numpy.clip(image_mask, 0.0, 1.0, out=image_mask)
        if self.erosion_size != 0:
            if abs(self.erosion_size) < 1.0:
                mask_radius = numpy.sqrt(numpy.sum(image_mask)) / 2
                percent_erode = max(1,int(abs(self.erosion_size * mask_radius)))
                self.erosion_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                                (percent_erode,
                                                                percent_erode))
            args = {'src':image_mask, 'kernel':self.erosion_kernel, 'iterations':1}
            image_mask = cv2.erode(**args) if self.erosion_size > 0 else cv2.dilate(**args)
            
        if self.blur_size != 0:
            if self.blur_size < 1.0:
                mask_radius = numpy.sqrt(numpy.sum(image_mask)) / 2
                self.blur_size = max(1,int(self.blur_size * mask_radius))
            image_mask = cv2.blur(image_mask, (int(self.blur_size),int(self.blur_size)))
            
        numpy.clip(image_mask, 0.0, 1.0, out=image_mask)
        return image_mask
        
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
                                           self.inv_interpolator,
                                   borderMode = cv2.BORDER_CONSTANT,
                                   borderValue = (-1.0,-1.0,-1.0))
                                   
        cleanup_mask = (new_image == (-1.0,-1.0,-1.0))
        new_image[cleanup_mask] = image[cleanup_mask]
                       
        if self.sharpen_image is not None:
            numpy.clip(new_image, 0.0, 255.0, out=new_image)
            if self.sharpen_image == "bsharpen":
                kernel = numpy.ones((3, 3)) * (-1)
                kernel[1, 1] = 9
                new_image = cv2.filter2D(new_image, -1, kernel)
            elif self.sharpen_image == "gsharpen":
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
            unitMask = numpy.rint(numpy.clip(image_mask * 365, 0.0, 255.0)).astype('uint8')
            logger.info(unitMask.shape)
            logger.info(image.shape)
            maxregion = numpy.argwhere(unitMask == 255)
            if maxregion.size > 0:
                miny, minx = maxregion.min(axis=0)[:2]
                maxy, maxx = maxregion.max(axis=0)[:2]
                lenx = maxx - minx
                leny = maxy - miny
                mask = int(minx + (lenx // 2)),  int(miny + (leny // 2))
                outimage = cv2.seamlessClone(numpy.rint(new_image).astype('uint8'),
                                             image,
                                             unitMask,
                                             mask,
                                             cv2.NORMAL_CLONE)
        else:
            foreground = image_mask * new_image #new_image
            background = ( 1.0 - image_mask ) * image
            outimage = foreground + background
        
        numpy.clip(outimage, 0.0, 255.0, out=outimage)

        return numpy.rint(outimage).astype('uint8')
        
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
        
        flat_new_image = numpy.interp(source.ravel(),bins,template_CDF*255.0)
        
        return new_image.reshape(source.shape) # source