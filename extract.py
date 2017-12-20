import argparse
import cv2
from pathlib import Path
import os

from lib.utils import get_image_paths, get_folder, FullPaths
from lib.faces_detect import crop_faces

parser = argparse.ArgumentParser(
    description='Extracts faces from a collection of pictures and saves them to a separate directory',
    epilog="Questions and feedback: https://github.com/deepfakes/faceswap-playground"
)

parser.add_argument('-i', '--input-dir',
                    action=FullPaths,
                    dest="input_dir",
                    default="src",
                    help="Input directory. A directory containing the files \
                    you wish to extract faces from. Defaults to 'src'")
parser.add_argument('-o', '--output-dir',
                    action=FullPaths,
                    dest="output_dir",
                    default="extract",
                    help="Output directory. This is where the cropped faces will \
                    be stored. Defaults to 'extract'")
parser.add_argument('-v', action="store_true", dest="verbose", default=False, help="Show verbose output")

arguments = parser.parse_args()

print("Input Directory: {}".format(arguments.input_dir))
print("Output Directory: {}".format(arguments.output_dir))
print('Starting, this may take a while...')

output_dir = get_folder(arguments.output_dir)
try:
    images_SRC = get_image_paths(arguments.input_dir)
except FileNotFoundError:
    print('Input directory not found. Please ensure it exists.')
    exit(1)

verify_output = False
images_found = len(images_SRC)
images_processed = 0
faces_detected = 0

for fn in images_SRC:
    if arguments.verbose:
        print('Processing image: {}'.format(os.path.basename(fn)))

    try:
        image = cv2.imread(fn)
        for (idx, (p1, p2, img)) in enumerate(crop_faces(image)):
            if idx > 0 and arguments.verbose:
                print('- Found more than one face!')
                verify_output = True

            # resize and save
            final = cv2.resize(img, (256, 256))
            output_file = output_dir / Path(fn).stem
            cv2.imwrite(str(output_file) + str(idx) + Path(fn).suffix, final)
            faces_detected = faces_detected + 1

        images_processed = images_processed + 1
    except Exception as e:
        print('Failed to extract from image: {}' . fn)

print('-------------------------')
print('Images found:        {}'.format(images_found))
print('Images processed:    {}'.format(images_processed))
print('Faces detected:      {}'.format(faces_detected))
print('-------------------------')

if verify_output:
    print('Note:')
    print('Multiple faces were detected in one or more pictures.')
    print('Please double check your results before you start training.')
    print('-------------------------')
print('Done!')
