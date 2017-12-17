import cv2
import numpy
from pathlib import Path

from utils import get_image_paths
from faces_detect import crop_faces

images_SRC = get_image_paths( "src" )

output_dir = Path( 'extract' )
output_dir.mkdir( parents=True, exist_ok=True )

for fn in images_SRC:
    image = cv2.imread(fn)
    for (idx, (p1,p2,img)) in enumerate(crop_faces( image )):
        final = cv2.resize(img, (256,256))
        output_file = output_dir / Path(fn).stem
        cv2.imwrite( str(output_file) + str(idx) + Path(fn).suffix, final )
