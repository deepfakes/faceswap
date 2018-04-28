# Based on: https://gist.github.com/anonymous/d3815aba83a8f79779451262599b0955 found on https://www.reddit.com/r/deepfakes/

import cv2
import numpy

from lib.aligner import get_align_mat

class Convert():
    def __init__(self, encoder, trainer, blur_size=2, seamless_clone=False, mask_type="facehullandrect", erosion_kernel_size=None, match_histogram=False, sharpen_image=None, **kwargs):
        self.encoder = encoder
        self.trainer = trainer
        self.erosion_kernel = None
        self.erosion_kernel_size = erosion_kernel_size
        if erosion_kernel_size is not None:
            if erosion_kernel_size > 0:
                self.erosion_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(erosion_kernel_size,erosion_kernel_size))
            elif erosion_kernel_size < 0:
                self.erosion_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(abs(erosion_kernel_size),abs(erosion_kernel_size)))
        self.blur_size = blur_size
        self.seamless_clone = seamless_clone
        self.sharpen_image = sharpen_image
        self.match_histogram = match_histogram
        self.mask_type = mask_type.lower() # Choose in 'FaceHullAndRect','FaceHull','Rect'

    def patch_image( self, image, face_detected, size ):

        image_size = image.shape[1], image.shape[0]

        mat = numpy.array(get_align_mat(face_detected, size, should_align_eyes=False)).reshape(2,3)

        if "GAN" not in self.trainer:
            mat = mat * size
        else:
            padding = int(48/256*size)
            mat = mat * (size - 2 * padding)
            mat[:,2] += padding

        new_face = self.get_new_face(image,mat,size)

        image_mask = self.get_image_mask( image, new_face, face_detected.landmarksAsXY(), mat, image_size )

        return self.apply_new_face(image, new_face, image_mask, mat, image_size, size)

    def apply_new_face(self, image, new_face, image_mask, mat, image_size, size):
        base_image = numpy.copy( image )
        new_image = numpy.copy( image )

        cv2.warpAffine( new_face, mat, image_size, new_image, cv2.WARP_INVERSE_MAP | cv2.INTER_CUBIC, cv2.BORDER_TRANSPARENT )
        
        if self.sharpen_image == "bsharpen":
            # Sharpening using filter2D
            kernel = numpy.ones((3, 3)) * (-1)
            kernel[1, 1] = 9
            new_image = cv2.filter2D(new_image, -1, kernel)
        elif self.sharpen_image == "gsharpen":
            # Sharpening using Weighted Method
            gaussain_blur = cv2.GaussianBlur(new_image, (0, 0), 3.0)
            new_image = cv2.addWeighted(
                new_image, 1.5, gaussain_blur, -0.5, 0, new_image)

        outimage = None
        if self.seamless_clone:
            unitMask = numpy.clip( image_mask * 365, 0, 255 ).astype(numpy.uint8)

            maxregion = numpy.argwhere(unitMask==255)

            if maxregion.size > 0:
              miny,minx = maxregion.min(axis=0)[:2]
              maxy,maxx = maxregion.max(axis=0)[:2]
              lenx = maxx - minx;
              leny = maxy - miny;
              masky = int(minx+(lenx//2))
              maskx = int(miny+(leny//2))
              outimage = cv2.seamlessClone(new_image.astype(numpy.uint8),base_image.astype(numpy.uint8),unitMask,(masky,maskx) , cv2.NORMAL_CLONE )

              return outimage

        foreground = cv2.multiply(image_mask, new_image.astype(float))
        background = cv2.multiply(1.0 - image_mask, base_image.astype(float))
        outimage = cv2.add(foreground, background)

        return outimage

    def hist_match(self, source, template, mask=None):
        # Code borrowed from:
        # https://stackoverflow.com/questions/32655686/histogram-matching-of-two-images-in-python-2-x
        masked_source = source
        masked_template = template

        if mask is not None:
            masked_source = source * mask
            masked_template = template * mask

        oldshape = source.shape
        source = source.ravel()
        template = template.ravel()
        masked_source = masked_source.ravel()
        masked_template = masked_template.ravel()
        s_values, bin_idx, s_counts = numpy.unique(source, return_inverse=True,
                                                return_counts=True)
        t_values, t_counts = numpy.unique(template, return_counts=True)
        ms_values, mbin_idx, ms_counts = numpy.unique(source, return_inverse=True,
                                                return_counts=True)
        mt_values, mt_counts = numpy.unique(template, return_counts=True)

        s_quantiles = numpy.cumsum(s_counts).astype(numpy.float64)
        s_quantiles /= s_quantiles[-1]
        t_quantiles = numpy.cumsum(t_counts).astype(numpy.float64)
        t_quantiles /= t_quantiles[-1]
        interp_t_values = numpy.interp(s_quantiles, t_quantiles, t_values)

        return interp_t_values[bin_idx].reshape(oldshape)

    def color_hist_match(self, src_im, tar_im, mask):
        matched_R = self.hist_match(src_im[:,:,0], tar_im[:,:,0], mask)
        matched_G = self.hist_match(src_im[:,:,1], tar_im[:,:,1], mask)
        matched_B = self.hist_match(src_im[:,:,2], tar_im[:,:,2], mask)
        matched = numpy.stack((matched_R, matched_G, matched_B), axis=2).astype(src_im.dtype)
        return matched

    def get_new_face(self, image, mat, size):
        face = cv2.warpAffine( image, mat, (size,size) )
        face = numpy.expand_dims( face, 0 )
        face_clipped = numpy.clip(face[0], 0, 255).astype( image.dtype )
        new_face = None
        mask = None

        if "GAN" not in self.trainer:
            normalized_face = face / 255.0
            new_face = self.encoder(normalized_face)[0]
            new_face = numpy.clip( new_face * 255, 0, 255 ).astype( image.dtype )
        else:
            normalized_face = face / 255.0 * 2 - 1
            fake_output = self.encoder(normalized_face)
            if "128" in self.trainer: # TODO: Another hack to switch between 64 and 128
                fake_output = fake_output[0]
            mask = fake_output[:,:,:, :1]
            new_face = fake_output[:,:,:, 1:]
            new_face = mask * new_face + (1 - mask) * normalized_face
            new_face = numpy.clip((new_face[0] + 1) * 255 / 2, 0, 255).astype( image.dtype )

        if self.match_histogram:
            new_face = self.color_hist_match(new_face, face_clipped, mask)

        return new_face

    def get_image_mask(self, image, new_face, landmarks, mat, image_size):

        face_mask = numpy.zeros(image.shape,dtype=float)
        if 'rect' in self.mask_type:
            face_src = numpy.ones(new_face.shape,dtype=float)
            cv2.warpAffine( face_src, mat, image_size, face_mask, cv2.WARP_INVERSE_MAP | cv2.INTER_CUBIC, cv2.BORDER_TRANSPARENT )

        hull_mask = numpy.zeros(image.shape,dtype=float)
        if 'hull' in self.mask_type:
            hull = cv2.convexHull( numpy.array( landmarks ).reshape((-1,2)).astype(int) ).flatten().reshape( (-1,2) )
            cv2.fillConvexPoly( hull_mask,hull,(1,1,1) )

        if self.mask_type == 'rect':
            image_mask = face_mask
        elif self.mask_type == 'facehull':
            image_mask = hull_mask
        else:
            image_mask = ((face_mask*hull_mask))


        if self.erosion_kernel is not None:
            if self.erosion_kernel_size > 0:
                image_mask = cv2.erode(image_mask,self.erosion_kernel,iterations = 1)
            elif self.erosion_kernel_size < 0:
                dilation_kernel = abs(self.erosion_kernel)
                image_mask = cv2.dilate(image_mask,dilation_kernel,iterations = 1)

        if self.blur_size!=0:
            image_mask = cv2.blur(image_mask,(self.blur_size,self.blur_size))

        return image_mask
