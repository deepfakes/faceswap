import cv2
import numpy
from pathlib import Path

from utils import get_image_paths

from model import autoencoder_A
from model import autoencoder_B
from model import encoder, decoder_A, decoder_B

encoder  .load_weights( "models/encoder.h5"   )
decoder_A.load_weights( "models/decoder_A.h5" )
decoder_B.load_weights( "models/decoder_B.h5" )

images_A = get_image_paths( "data/trump" )
images_B = get_image_paths( "data/cage" )

def convert_one_image( autoencoder, image ):
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

output_dir = Path( 'output' )
output_dir.mkdir( parents=True, exist_ok=True )

for fn in images_A:
    image = cv2.imread(fn)
    new_image = convert_one_image( autoencoder_B, image )
    output_file = output_dir / Path(fn).name
    cv2.imwrite( str(output_file), new_image )