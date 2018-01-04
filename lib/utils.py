import cv2
import numpy

from pathlib import Path
from scandir import scandir


def get_folder(path):
    output_dir = Path(path)
    # output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def get_image_paths(directory):
    return [x.path for x in scandir(directory) if x.name.endswith('.jpg') or x.name.endswith('.jpeg') or x.name.endswith('.png')]
