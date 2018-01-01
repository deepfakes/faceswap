import argparse

import cv2
import numpy
import sys

from pathlib import Path
from scandir import scandir


def get_folder(path):
    output_dir = Path(path)
    # output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def get_image_paths(directory):
    return [x.path for x in scandir(directory) if x.name.endswith('.jpg') or x.name.endswith('.png')]


def load_images(image_paths, convert=None):
    iter_all_images = (cv2.imread(fn) for fn in image_paths)
    if convert:
        iter_all_images = (convert(img) for img in iter_all_images)
    for i, image in enumerate(iter_all_images):
        if i == 0:
            all_images = numpy.empty((len(image_paths), ) + image.shape, dtype=image.dtype)
        all_images[i] = image
    return all_images


def get_transpose_axes(n):
    if n % 2 == 0:
        y_axes = list(range(1, n - 1, 2))
        x_axes = list(range(0, n - 1, 2))
    else:
        y_axes = list(range(0, n - 1, 2))
        x_axes = list(range(1, n - 1, 2))
    return y_axes, x_axes, [n - 1]


def stack_images(images):
    images_shape = numpy.array(images.shape)
    new_axes = get_transpose_axes(len(images_shape))
    new_shape = [numpy.prod(images_shape[x]) for x in new_axes]
    return numpy.transpose(
        images,
        axes=numpy.concatenate(new_axes)
        ).reshape(new_shape)


class FullHelpArgumentParser(argparse.ArgumentParser):
    """Identical to the built-in argument parser, but on error
    it prints full help message instead of just usage information
    """
    def error(self, message):
        self.print_help(sys.stderr)
        args = {'prog': self.prog, 'message': message}
        self.exit(2, '%(prog)s: error: %(message)s\n' % args)
