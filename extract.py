import cv2
import numpy
from pathlib import Path

from utils import get_image_paths

images_SRC = get_image_paths( "src" )

# Give right path to the xml file or put it directly in current folder
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def crop_faces( image ):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        crop_img = image[y: y + h, x: x + w]
        final = cv2.resize(crop_img, (255,255))
        yield final

output_dir = Path( 'extract' )
#output_dir.mkdir( parents=True, exist_ok=True )

for fn in images_SRC:
    image = cv2.imread(fn)
#Add : cv.EqualizeHist(image, image) ?
    for (idx, img) in enumerate(crop_faces( image )):
        output_file = output_dir / Path(fn).stem
        cv2.imwrite( str(output_file) + str(idx) + Path(fn).suffix, img )
