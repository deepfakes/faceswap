import cv2
from pathlib import Path

from lib.utils import get_image_paths, get_folder
from lib.faces_detect import crop_faces

output_dir = get_folder('extract')

images_SRC = get_image_paths('src')

for fn in images_SRC:
    image = cv2.imread(fn)
    for (idx, (p1, p2, img)) in enumerate(crop_faces(image)):
        final = cv2.resize(img, (256, 256))
        output_file = output_dir / Path(fn).stem
        cv2.imwrite(str(output_file) + str(idx) + Path(fn).suffix, final)
