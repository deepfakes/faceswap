# Based on the original https://www.reddit.com/r/deepfakes/ code sample
# Adjust code made by https://github.com/yangchen8710

import cv2
import numpy
import os

class Convert(object):
    def __init__(self, encoder, smooth_mask=True, avg_color_adjust=True, **kwargs):
        self.encoder = encoder

        self.use_smooth_mask = smooth_mask
        self.use_avg_color_adjust = avg_color_adjust

    def patch_image( self, original, face_detected, size ):
        #assert image.shape == (256, 256, 3)
        image = cv2.resize(face_detected.image, (256, 256))
        crop = slice(48, 208)
        face = image[crop, crop]
        old_face = face.copy()

        face = cv2.resize(face, (size, size))
        face = numpy.expand_dims(face, 0)
        new_face = self.encoder(face / 255.0)[0]
        new_face = numpy.clip(new_face * 255, 0, 255).astype(image.dtype)
        new_face = cv2.resize(new_face, (160, 160))

        if self.use_avg_color_adjust:
            self.adjust_avg_color(old_face,new_face)
        if self.use_smooth_mask:
            self.smooth_mask(old_face,new_face)

        new_face = self.superpose(image, new_face, crop)
        original[slice(face_detected.y, face_detected.y + face_detected.h), slice(face_detected.x, face_detected.x + face_detected.w)] = cv2.resize(new_face, (face_detected.w, face_detected.h))
        return original

    def adjust_avg_color(self,img_old,img_new):
        w,h,c = img_new.shape
        for i in range(img_new.shape[-1]):
            old_avg = img_old[:, :, i].mean()
            new_avg = img_new[:, :, i].mean()
            diff_int = (int)(old_avg - new_avg)
            for m in range(img_new.shape[0]):
                for n in range(img_new.shape[1]):
                    temp = (img_new[m,n,i] + diff_int)
                    if temp < 0:
                        img_new[m,n,i] = 0
                    elif temp > 255:
                        img_new[m,n,i] = 255
                    else:
                        img_new[m,n,i] = temp

    def smooth_mask(self,img_old,img_new):
        w,h,c = img_new.shape
        crop = slice(0,w)
        mask = numpy.zeros_like(img_new)
        mask[h//15:-h//15,w//15:-w//15,:] = 255
        mask = cv2.GaussianBlur(mask,(15,15),10)
        img_new[crop,crop] = mask/255*img_new + (1-mask/255)*img_old

    def superpose(self,image, new_face, crop):
        new_image = image.copy()
        new_image[crop, crop] = new_face
        return new_image
