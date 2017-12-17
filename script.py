import cv2
import numpy
from pathlib import Path

from utils import get_image_paths

from faces_process import convert_one_image

images_A = get_image_paths( "data/trump" )
images_B = get_image_paths( "data/cage" )

output_dir = Path( 'output' )
output_dir.mkdir( parents=True, exist_ok=True )

for fn in images_A:
    image = cv2.imread(fn)
    new_image = convert_one_image( image )
    output_file = output_dir / Path(fn).name
    cv2.imwrite( str(output_file), new_image )