import cv2
import numpy
from pathlib import Path

from lib.utils import get_image_paths, get_folder
from lib.faces_process import convert_one_image

output_dir = get_folder( 'output' )

images_A = get_image_paths( 'data/trump' )
images_B = get_image_paths( 'data/cage' )

for fn in images_A:
    image = cv2.imread(fn)
    new_image = convert_one_image( image )
    output_file = output_dir / Path(fn).name
    cv2.imwrite( str(output_file), new_image )