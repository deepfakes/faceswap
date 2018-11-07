#!/usr/bin python3
""" Utilities available across all scripts """

import os
import warnings

from pathlib import Path
from re import finditer
from time import time

import cv2
import numpy as np

import dlib
from lib.faces_detect import DetectedFace
from lib.training_data import TrainingDataGenerator

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
    exclude_names = [os.path.basename(Path(x).stem[:Path(x).stem.rfind('_')] +
                                      Path(x).suffix) for x in exclude]
    dir_contents = list()

    if not os.path.exists(directory):
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
    origfile = os.path.join(directory, filename)
    backupfile = origfile + '.bk'
    if os.path.exists(backupfile):
        os.remove(backupfile)
    if os.path.exists(origfile):
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
        Pass in a DetectedFace object, Alignments dict or DLib rectangle"""
    if isinstance(face, DetectedFace):
        bounding_box = [[face.x, face.y],
                        [face.x + face.w, face.y],
                        [face.x + face.w, face.y + face.h],
                        [face.x, face.y + face.h]]
        landmarks = face.landmarksXY

    elif isinstance(face, dict):
        bounding_box = [[face.get("x", 0), face.get("y", 0)],
                        [face.get("x", 0) + face.get("w", 0),
                         face.get("y", 0)],
                        [face.get("x", 0) + face.get("w", 0),
                         face.get("y", 0) + face.get("h", 0)],
                        [face.get("x", 0),
                         face.get("y", 0) + face.get("h", 0)]]
        landmarks = face.get("landmarksXY", list())

    elif isinstance(face, dlib.rectangle):
        bounding_box = [[face.left(), face.top()],
                        [face.right(), face.top()],
                        [face.right(), face.bottom()],
                        [face.left(), face.bottom()]]
        landmarks = list()
    else:
        raise ValueError("Unsupported face type")

    rotation_matrix = cv2.invertAffineTransform(rotation_matrix)
    rotated = list()
    for item in (bounding_box, landmarks):
        if not item:
            continue
        points = np.array(item, np.int32)
        points = np.expand_dims(points, axis=0)
        transformed = cv2.transform(points,
                                    rotation_matrix).astype(np.int32)
        rotated.append(transformed.squeeze())

    # Bounding box should follow x, y planes, so get min/max
    # for non-90 degree rotations
    pt_x = min([pnt[0] for pnt in rotated[0]])
    pt_y = min([pnt[1] for pnt in rotated[0]])
    pt_x1 = max([pnt[0] for pnt in rotated[0]])
    pt_y1 = max([pnt[1] for pnt in rotated[0]])

    if isinstance(face, DetectedFace):
        face.x = int(pt_x)
        face.y = int(pt_y)
        face.w = int(pt_x1 - pt_x)
        face.h = int(pt_y1 - pt_y)
        face.r = 0
        if len(rotated) > 1:
            face.landmarksXY = [tuple(point) for point in rotated[1].tolist()]
    elif isinstance(face, dict):
        face["x"] = int(pt_x)
        face["y"] = int(pt_y)
        face["w"] = int(pt_x1 - pt_x)
        face["h"] = int(pt_y1 - pt_y)
        face["r"] = 0
        if len(rotated) > 1:
            face["landmarksXY"] = [tuple(point)
                                   for point in rotated[1].tolist()]
    else:
        face = dlib.rectangle(int(pt_x), int(pt_y), int(pt_x1), int(pt_y1))

    return face


def camel_case_split(identifier):
    """ Split a camel case name
        from: https://stackoverflow.com/questions/29916065 """
    matches = finditer(
        ".+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)",
        identifier)
    return [m.group(0) for m in matches]


class Timelapse:
    """ Time lapse function for training """
    @classmethod
    def create_timelapse(cls, input_dir_a, input_dir_b, output_dir, trainer):
        """ Create the time lapse """
        if input_dir_a is None and input_dir_b is None and output_dir is None:
            return None

        if input_dir_a is None or input_dir_b is None:
            raise ValueError("To enable the timelapse, you have to supply "
                             "all the parameters (--timelapse-input-A and "
                             "--timelapse-input-B).")

        if output_dir is None:
            output_dir = get_folder(os.path.join(trainer.model.model_dir,
                                                 "timelapse"))

        return Timelapse(input_dir_a, input_dir_b, output_dir, trainer)

    def __init__(self, input_dir_a, input_dir_b, output, trainer):
        self.output_dir = output
        self.trainer = trainer

        if not os.path.isdir(self.output_dir):
            print('Error: {} does not exist'.format(self.output_dir))
            exit(1)

        self.files_a = self.read_input_images(input_dir_a)
        self.files_b = self.read_input_images(input_dir_b)

        btchsz = min(len(self.files_a), len(self.files_b))

        self.images_a = self.get_image_data(self.files_a, btchsz)
        self.images_b = self.get_image_data(self.files_b, btchsz)

    @staticmethod
    def read_input_images(input_dir):
        """ Get the image paths """
        if not os.path.isdir(input_dir):
            print('Error: {} does not exist'.format(input_dir))
            exit(1)

        if not os.listdir(input_dir):
            print('Error: {} contains no images'.format(input_dir))
            exit(1)

        return get_image_paths(input_dir)

    def get_image_data(self, input_images, batch_size):
        """ Get training images """
        random_transform_args = {
            'rotation_range': 0,
            'zoom_range': 0,
            'shift_range': 0,
            'random_flip': 0
        }

        zoom = 1
        if hasattr(self.trainer.model, 'IMAGE_SHAPE'):
            zoom = self.trainer.model.IMAGE_SHAPE[0] // 64

        generator = TrainingDataGenerator(random_transform_args, 160, zoom)
        batch = generator.minibatchAB(input_images, batch_size,
                                      doShuffle=False)

        return next(batch)[2]

    def work(self):
        """ Write out timelapse image """
        image = self.trainer.show_sample(self.images_a, self.images_b)
        cv2.imwrite(os.path.join(self.output_dir,
                                 str(int(time())) + ".png"), image)
