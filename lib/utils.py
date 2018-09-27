#!/usr/bin python3
""" Utilities available across all scripts """

import os
from os.path import basename, exists, join
import queue as Queue
import threading
import warnings

from pathlib import Path

import cv2
import numpy as np


# Global variables
_image_extensions = ['.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff']
_video_extensions = ['.avi', '.flv', '.mkv', '.mov', '.mp4', '.mpeg', '.webm']


def get_folder(path):
    """ Return a path to a folder, creating it if it doesn't exist """
    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def get_image_paths(directory, exclude=list(), debug=False):
    """ Return a list of images that reside in a folder """
    image_extensions = _image_extensions
    exclude_names = [basename(Path(x).stem[:Path(x).stem.rfind('_')] +
                              Path(x).suffix) for x in exclude]
    dir_contents = list()

    if not exists(directory):
        directory = get_folder(directory)

    dir_scanned = sorted(os.scandir(directory), key=lambda x: x.name)
    for chkfile in dir_scanned:
        if any([chkfile.name.lower().endswith(ext)
                for ext in image_extensions]):
            if chkfile.name in exclude_names:
                if debug:
                    print("Already processed %s" % chkfile.name)
                continue
            else:
                dir_contents.append(chkfile.path)

    return dir_contents


def backup_file(directory, filename):
    """ Backup a given file by appending .bk to the end """
    origfile = join(directory, filename)
    backupfile = origfile + '.bk'
    if exists(backupfile):
        os.remove(backupfile)
    if exists(origfile):
        os.rename(origfile, backupfile)


def set_system_verbosity(loglevel):
    """ Set the verbosity level of tensorflow and suppresses
        future and deprecation warnings from any modules
        From:
        https://stackoverflow.com/questions/35911252/disable-tensorflow-debugging-information
        Can be set to:
        0 - all logs shown
        1 - filter out INFO logs
        2 - filter out WARNING logs
        3 - filter out ERROR logs  """
    # TODO suppress tensorflow deprecation warnings """

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = loglevel
    if loglevel != '0':
        for warncat in (FutureWarning, DeprecationWarning):
            warnings.simplefilter(action='ignore', category=warncat)


def rotate_image_by_angle(image, angle,
                          rotated_width=None, rotated_height=None):
    """ Rotate an image by a given angle.
        From: https://stackoverflow.com/questions/22041699 """

    height, width = image.shape[:2]
    image_center = (width/2, height/2)
    rotation_matrix = cv2.getRotationMatrix2D(image_center, -1.*angle, 1.)
    if rotated_width is None or rotated_height is None:
        abs_cos = abs(rotation_matrix[0, 0])
        abs_sin = abs(rotation_matrix[0, 1])
        if rotated_width is None:
            rotated_width = int(height*abs_sin + width*abs_cos)
        if rotated_height is None:
            rotated_height = int(height*abs_cos + width*abs_sin)
    rotation_matrix[0, 2] += rotated_width/2 - image_center[0]
    rotation_matrix[1, 2] += rotated_height/2 - image_center[1]
    return (cv2.warpAffine(image,
                           rotation_matrix,
                           (rotated_width, rotated_height)),
            rotation_matrix)


def rotate_landmarks(face, rotation_matrix):
    """ Rotate the landmarks and bounding box for faces
        found in rotated images.
        Pass in a DetectedFace object"""
    rotation_matrix = cv2.invertAffineTransform(rotation_matrix)
    bounding_box = [[face.x, face.y],
                    [face.x + face.w, face.y],
                    [face.x + face.w, face.y + face.h],
                    [face.x, face.y + face.h]]
    landmarks = face.landmarksXY
    rotated = list()
    for item in (bounding_box, landmarks):
        points = np.array(item, np.int32)
        points = np.expand_dims(points, axis=0)
        transformed = cv2.transform(points,
                                    rotation_matrix).astype(np.int32)
        rotated.append(transformed.squeeze())

    # Bounding box should follow x, y planes, so get min/max
    # for non-90 degree rotations
    pnt_x = min([pnt[0] for pnt in rotated[0]])
    pnt_y = min([pnt[1] for pnt in rotated[0]])
    pnt_x1 = max([pnt[0] for pnt in rotated[0]])
    pnt_y1 = max([pnt[1] for pnt in rotated[0]])
    face.x = int(pnt_x)
    face.y = int(pnt_y)
    face.w = int(pnt_x1 - pnt_x)
    face.h = int(pnt_y1 - pnt_y)
    face.r = 0
    face.landmarksXY = [tuple(point) for point in rotated[1].tolist()]
    return face


class BackgroundGenerator(threading.Thread):
    """ Run a queue in the background. From:
        https://stackoverflow.com/questions/7323664/ """
    # See below why prefetch count is flawed
    def __init__(self, generator, prefetch=1):
        threading.Thread.__init__(self)
        self.queue = Queue.Queue(prefetch)
        self.generator = generator
        self.daemon = True
        self.start()

    def run(self):
        """ Put until queue size is reached.
            Note: put blocks only if put is called while queue has already
            reached max size => this makes 2 prefetched items! One in the
            queue, one waiting for insertion! """
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def iterator(self):
        """ Iterate items out of the queue """
        while True:
            next_item = self.queue.get()
            if next_item is None:
                break
            yield next_item

from lib.training_data import TrainingDataGenerator
from time import time

class Timelapse:
    @classmethod
    def CreateTimelapse(test, input_dir, output_dir, trainer):
        #self.input_dir = input
        #self.output_dir = output
        #self.trainer = trainer

        if input_dir is None and output_dir is None:
            return None

        if input_dir is None or output_dir is None:
            raise Exception("To enable the timelapse, you have to supply both parameters "
                            "(--timelapse-input and --timelapse-ouput).")

        return Timelapse(input_dir, output_dir, trainer)

    def __init__(self, input, output, trainer):
        self.input_dir = input
        self.output_dir = output
        self.trainer = trainer

        if not os.path.isdir(self.output_dir):
            print('Error: {} does not exist'.format(self.output_dir))
            exit(1)

        self.images_A = self.read_input_images(os.path.join(self.input_dir, "A"))
        self.images_B = self.read_input_images(os.path.join(self.input_dir, "B"))

    def read_input_images(self, input_dir):
        if not os.path.isdir(input_dir):
            print('Error: {} does not exist'.format(input_dir))
            exit(1)

        if not os.listdir(input_dir):
            print('Error: {} contains no images'.format(input_dir))
            exit(1)

        input_images = get_image_paths(input_dir)

        random_transform_args = {
            'rotation_range': 0,
            'zoom_range': 0,
            'shift_range': 0,
            'random_flip': 0
        }

        generator = TrainingDataGenerator(random_transform_args, 160)
        batch = generator.minibatchAB(input_images, 6)

        return next(batch)[2]

    def work(self):
        image = self.trainer.show_sample(self.images_A, self.images_B)
        cv2.imwrite(os.path.join(self.output_dir, str(time()) + ".png"), image)