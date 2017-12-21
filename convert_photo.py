import cv2
import numpy
from pathlib import Path

from lib.utils import get_image_paths, get_folder
from lib.faces_detect import crop_faces
from lib.faces_process import convert_one_image

output_dir = get_folder('modified')

images_SRC = get_image_paths('original')

for fn in images_SRC:
    image = cv2.imread(fn)
    for face in crop_faces(image):
        new_face = convert_one_image(cv2.resize(face.image, (256, 256)))
        image[slice(face.y, face.y + face.h), slice(face.x, face.x + face.w)] = cv2.resize(new_face, (face.w, face.h))

    output_file = output_dir / Path(fn).name
    cv2.imwrite(str(output_file), image)
