# Based on: https://gist.github.com/anonymous/d3815aba83a8f79779451262599b0955 found on https://www.reddit.com/r/deepfakes/

import cv2
import numpy

from lib.aligner import get_align_mat

class Convert():
    def __init__(self, encoder, blur_size=2, seamless_clone=False, mask_type="facehullandrect", erosion_kernel_size=None, **kwargs):
        self.encoder = encoder

        self.erosion_kernel = None
        if erosion_kernel_size is not None:
            self.erosion_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(erosion_kernel_size,erosion_kernel_size))

        self.blur_size = blur_size
        self.seamless_clone = seamless_clone
        self.mask_type = mask_type.lower() # Choose in 'FaceHullAndRect','FaceHull','Rect'

    def patch_image( self, image, face_detected ):
        size = 64
        image_size = image.shape[1], image.shape[0]

        mat = numpy.array(get_align_mat(face_detected)).reshape(2,3) * size

        new_face = self.get_new_face(image,mat,size)

        image_mask = self.get_image_mask( image, new_face, face_detected, mat, image_size )

        return self.apply_new_face(image, new_face, image_mask, mat, image_size, size)

    def apply_new_face(self, image, new_face, image_mask, mat, image_size, size):
        base_image = numpy.copy( image )
        new_image = numpy.copy( image )

        cv2.warpAffine( new_face, mat, image_size, new_image, cv2.WARP_INVERSE_MAP | cv2.INTER_CUBIC, cv2.BORDER_TRANSPARENT )

        outImage = None
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

    def get_new_face(self, image, mat, size):
        face = cv2.warpAffine( image, mat, (size,size) )
        face = numpy.expand_dims( face, 0 )
        new_face = self.encoder( face / 255.0 )[0]

        return numpy.clip( new_face * 255, 0, 255 ).astype( image.dtype )

    def get_image_mask(self, image, new_face, face_detected, mat, image_size):

        face_mask = numpy.zeros(image.shape,dtype=float)
        if 'rect' in self.mask_type:
            face_src = numpy.ones(new_face.shape,dtype=float)
            cv2.warpAffine( face_src, mat, image_size, face_mask, cv2.WARP_INVERSE_MAP | cv2.INTER_CUBIC, cv2.BORDER_TRANSPARENT )

        hull_mask = numpy.zeros(image.shape,dtype=float)
        if 'hull' in self.mask_type:
            hull = cv2.convexHull( numpy.array( face_detected.landmarksAsXY() ).reshape((-1,2)).astype(int) ).flatten().reshape( (-1,2) )
            cv2.fillConvexPoly( hull_mask,hull,(1,1,1) )

        if self.mask_type == 'rect':
            image_mask = face_mask
        elif self.mask_type == 'faceHull':
            image_mask = hull_mask
        else:
            image_mask = ((face_mask*hull_mask))


        if self.erosion_kernel is not None:
            image_mask = cv2.erode(image_mask,self.erosion_kernel,iterations = 1)

        if self.blur_size!=0:
            image_mask = cv2.blur(image_mask,(self.blur_size,self.blur_size))

        return image_mask
