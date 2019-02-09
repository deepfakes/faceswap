#!/usr/bin python3
""" Utilities available across all scripts """

import logging
import os
import warnings

from hashlib import sha1
from pathlib import Path
from re import finditer

import cv2
import numpy as np

import dlib

from lib.faces_detect import DetectedFace
from lib.logger import get_loglevel


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

# Global variables
_image_extensions = [  # pylint: disable=invalid-name
    ".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"]
_video_extensions = [  # pylint: disable=invalid-name
    ".avi", ".flv", ".mkv", ".mov", ".mp4", ".mpeg", ".webm"]


def get_folder(path):
    """ Return a path to a folder, creating it if it doesn't exist """
    logger.debug("Requested path: '%s'", path)
    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.debug("Returning: '%s'", output_dir)
    return output_dir


def get_image_paths(directory):
    """ Return a list of images that reside in a folder """
    image_extensions = _image_extensions
    dir_contents = list()

    if not os.path.exists(directory):
        logger.debug("Creating folder: '%s'", directory)
        directory = get_folder(directory)

    dir_scanned = sorted(os.scandir(directory), key=lambda x: x.name)
    logger.debug("Scanned Folder contains %s files", len(dir_scanned))
    logger.trace("Scanned Folder Contents: %s", dir_scanned)

    for chkfile in dir_scanned:
        if any([chkfile.name.lower().endswith(ext)
                for ext in image_extensions]):
            logger.trace("Adding '%s' to image list", chkfile.path)
            dir_contents.append(chkfile.path)

    logger.debug("Returning %s images", len(dir_contents))
    return dir_contents


def hash_image_file(filename):
    """ Return an image file's sha1 hash """
    img = cv2.imread(filename)  # pylint: disable=no-member
    img_hash = sha1(img).hexdigest()
    logger.trace("filename: '%s', hash: %s", filename, img_hash)
    return img_hash


def hash_encode_image(image, extension):
    """ Encode the image, get the hash and return the hash with
        encoded image """
    img = cv2.imencode(extension, image)[1]  # pylint: disable=no-member
    f_hash = sha1(
        cv2.imdecode(img, cv2.IMREAD_UNCHANGED)).hexdigest()  # pylint: disable=no-member
    return f_hash, img


def backup_file(directory, filename):
    """ Backup a given file by appending .bk to the end """
    logger.trace("Backing up: '%s'", filename)
    origfile = os.path.join(directory, filename)
    backupfile = origfile + '.bk'
    if os.path.exists(backupfile):
        logger.trace("Removing existing file: '%s'", backup_file)
        os.remove(backupfile)
    if os.path.exists(origfile):
        logger.trace("Renaming: '%s' to '%s'", origfile, backup_file)
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

    numeric_level = get_loglevel(loglevel)
    loglevel = "2" if numeric_level > 15 else "0"
    logger.debug("System Verbosity level: %s", loglevel)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = loglevel
    if loglevel != '0':
        for warncat in (FutureWarning, DeprecationWarning, UserWarning):
            warnings.simplefilter(action='ignore', category=warncat)


def rotate_landmarks(face, rotation_matrix):
    # pylint: disable=c-extension-no-member
    """ Rotate the landmarks and bounding box for faces
        found in rotated images.
        Pass in a DetectedFace object, Alignments dict or DLib rectangle"""
    logger.trace("Rotating landmarks: (rotation_matrix: %s, type(face): %s",
                 rotation_matrix, type(face))
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

    elif isinstance(face,
                    dlib.rectangle):  # pylint: disable=c-extension-no-member
        bounding_box = [[face.left(), face.top()],
                        [face.right(), face.top()],
                        [face.right(), face.bottom()],
                        [face.left(), face.bottom()]]
        landmarks = list()
    else:
        raise ValueError("Unsupported face type")

    logger.trace("Original landmarks: %s", landmarks)

    rotation_matrix = cv2.invertAffineTransform(  # pylint: disable=no-member
        rotation_matrix)
    rotated = list()
    for item in (bounding_box, landmarks):
        if not item:
            continue
        points = np.array(item, np.int32)
        points = np.expand_dims(points, axis=0)
        transformed = cv2.transform(points,  # pylint: disable=no-member
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
            rotated_landmarks = [tuple(point) for point in rotated[1].tolist()]
            face.landmarksXY = rotated_landmarks
    elif isinstance(face, dict):
        face["x"] = int(pt_x)
        face["y"] = int(pt_y)
        face["w"] = int(pt_x1 - pt_x)
        face["h"] = int(pt_y1 - pt_y)
        face["r"] = 0
        if len(rotated) > 1:
            rotated_landmarks = [tuple(point) for point in rotated[1].tolist()]
            face["landmarksXY"] = rotated_landmarks
    else:
        rotated_landmarks = dlib.rectangle(  # pylint: disable=c-extension-no-member
            int(pt_x), int(pt_y), int(pt_x1), int(pt_y1))
        face = rotated_landmarks

    logger.trace("Rotated landmarks: %s", rotated_landmarks)
    return face


def camel_case_split(identifier):
    """ Split a camel case name
        from: https://stackoverflow.com/questions/29916065 """
    matches = finditer(
        ".+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)",
        identifier)
    return [m.group(0) for m in matches]


def safe_shutdown():
    """ Close queues, threads and processes in event of crash """
    logger.debug("Safely shutting down")
    from lib.queue_manager import queue_manager
    from lib.multithreading import terminate_processes
    queue_manager.terminate_queues()
    terminate_processes()
    logger.debug("Cleanup complete. Shutting down queue manager and exiting")
    queue_manager._log_queue.put(None)  # pylint: disable=protected-access
    while not queue_manager._log_queue.empty():  # pylint: disable=protected-access
        continue
    queue_manager.manager.shutdown()
