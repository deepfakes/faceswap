# Based on the https://github.com/shaoanlu/faceswap-GAN repo (master/FaceSwap_GAN_v2_train.ipynb)

import cv2
import numpy

class Convert(object):
    def __init__(self, encoder, **kwargs):
        self.encoder = encoder

    def patch_image( self, original, face_detected ):
        face = cv2.resize(face_detected.image, (64, 64))
        face = numpy.expand_dims(face, 0) / 255.0 * 2 - 1
        mask, new_face = self.encoder(face)
        new_face = mask * new_face + (1 - mask) * face
        new_face = numpy.clip((new_face[0] + 1) * 255 / 2, 0, 255).astype('uint8')

        original[face_detected.y: face_detected.y + face_detected.h, face_detected.x: face_detected.x + face_detected.w] = cv2.resize(new_face, (face_detected.w, face_detected.h))
        return original
