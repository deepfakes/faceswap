import cv2
import numpy

from model import autoencoder_A
from model import autoencoder_B
from model import encoder, decoder_A, decoder_B

encoder  .load_weights( "models/encoder.h5"   )
decoder_A.load_weights( "models/decoder_A.h5" )
decoder_B.load_weights( "models/decoder_B.h5" )

autoencoder = autoencoder_B

def convert_one_image( image ):
    assert image.shape == (256,256,3)
    crop = slice(48,208)
    face = image[crop,crop]
    face = cv2.resize( face, (64,64) )
    face = numpy.expand_dims( face, 0 )
    new_face = autoencoder.predict( face / 255.0 )[0]
    new_face = numpy.clip( new_face * 255, 0, 255 ).astype( image.dtype )
    new_face = cv2.resize( new_face, (160,160) )
    new_image = image.copy()
    new_image[crop,crop] = new_face
    return new_image
