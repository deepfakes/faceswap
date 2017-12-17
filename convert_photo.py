import cv2
import numpy
from pathlib import Path

from utils import get_image_paths
from faces_detect import crop_faces
from faces_process import convert_one_image

images_SRC = get_image_paths( "original" )

output_dir = Path( 'modified' )
output_dir.mkdir( parents=True, exist_ok=True )

for fn in images_SRC:
    image = cv2.imread(fn)
    for ((x,w),(y,h),face) in crop_faces( image ):
        new_face = convert_one_image( cv2.resize(face, (256,256)) )
        image[slice(y,y+h),slice(x,x+w)] = cv2.resize(new_face, (w,h))
    
    output_file = output_dir / Path(fn).name
    cv2.imwrite( str(output_file) , image )
