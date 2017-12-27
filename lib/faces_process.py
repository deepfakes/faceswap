import cv2
import numpy
import os

from .aligner import Aligner
from .model import autoencoder_A
from .model import autoencoder_B
from .model import encoder, decoder_A, decoder_B

def adjust_avg_color(img_old,img_new):
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

def smooth_mask(img_old,img_new):
    w,h,c = img_new.shape
    crop = slice(0,w)
    mask = numpy.zeros_like(img_new)
    mask[h//15:-h//15,w//15:-w//15,:] = 255
    mask = cv2.GaussianBlur(mask,(15,15),10)
    img_new[crop,crop] = mask/255*img_new + (1-mask/255)*img_old

def convert_one_image(image, 
                      model_dir="models",
                      swap_model=False,
                      use_aligner=False,
                      use_smooth_mask=True,
                      use_avg_color_adjust=True):
    face_A = '/decoder_A.h5' if not swap_model else '/decoder_B.h5'
    face_B = '/decoder_B.h5' if not swap_model else '/decoder_A.h5'

    encoder.load_weights(model_dir + "/encoder.h5")
    decoder_A.load_weights(model_dir + face_A)
    decoder_B.load_weights(model_dir + face_B)

    autoencoder = autoencoder_B

    shapePredictor = "contrib/shape_predictor_68_face_landmarks.dat"
    humanFaceDetector = "contrib/mmod_human_face_detector.dat"
    if not os.path.exists(shapePredictor):
        print("{} file not found.\n"
              "Landmark file can be found in http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
              "\nUnzip it in the contrib/ folder.".format(shapePredictor))
        return
    if use_aligner:
        aligner = Aligner(shapePredictor, humanFaceDetector)

    assert image.shape == (256, 256, 3)
    crop = slice(48, 208)
    face = image[crop, crop]
    old_face = face.copy()

    face = cv2.resize(face, (64, 64))
    face = numpy.expand_dims(face, 0)
    new_face = autoencoder.predict(face / 255.0)[0]
    new_face = numpy.clip(new_face * 255, 0, 255).astype(image.dtype)
    new_face = cv2.resize(new_face, (160, 160))

    if use_avg_color_adjust:
        adjust_avg_color(old_face,new_face)
    if use_smooth_mask:
        smooth_mask(old_face,new_face)

    # Aligner is not ready to use yet
    if use_aligner:
        return aligner.align(image.copy(), new_face)
    else:
        return superpose(image, new_face, crop)


def superpose(image, new_face, crop):
    new_image = image.copy()
    new_image[crop, crop] = new_face
    return new_image
