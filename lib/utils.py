#!/usr/bin python3
""" Utilities available across all scripts """

import logging
import os
import sys
import urllib
import warnings
import zipfile
from socket import timeout as socket_timeout, error as socket_error

from hashlib import sha1
from pathlib import Path
from re import finditer
from tqdm import tqdm

import numpy as np
import cv2


from lib.faces_detect import DetectedFace


# Global variables
_image_extensions = [  # pylint:disable=invalid-name
    ".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"]
_video_extensions = [  # pylint:disable=invalid-name
    ".avi", ".flv", ".mkv", ".mov", ".mp4", ".mpeg", ".webm"]


def get_folder(path, make_folder=True):
    """ Return a path to a folder, creating it if it doesn't exist """
    logger = logging.getLogger(__name__)  # pylint:disable=invalid-name
    logger.debug("Requested path: '%s'", path)
    output_dir = Path(path)
    if not make_folder and not output_dir.exists():
        logger.debug("%s does not exist", path)
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.debug("Returning: '%s'", output_dir)
    return output_dir


def get_image_paths(directory):
    """ Return a list of images that reside in a folder """
    logger = logging.getLogger(__name__)  # pylint:disable=invalid-name
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


def full_path_split(path):
    """ Split a given path into all of it's separate components """
    logger = logging.getLogger(__name__)  # pylint:disable=invalid-name
    allparts = list()
    while True:
        parts = os.path.split(path)
        if parts[0] == path:   # sentinel for absolute paths
            allparts.insert(0, parts[0])
            break
        elif parts[1] == path:  # sentinel for relative paths
            allparts.insert(0, parts[1])
            break
        else:
            path = parts[0]
            allparts.insert(0, parts[1])
    logger.trace("path: %s, allparts: %s", path, allparts)
    return allparts


def cv2_read_img(filename, raise_error=False):
    """ Read an image with cv2 and check that an image was actually loaded.
        Logs an error if the image returned is None. or an error has occured.

        Pass raise_error=True if error should be raised """
    logger = logging.getLogger(__name__)  # pylint:disable=invalid-name
    logger.trace("Requested image: '%s'", filename)
    success = True
    image = None
    try:
        image = cv2.imread(filename)  # pylint:disable=no-member,c-extension-no-member
        if image is None:
            raise ValueError
    except TypeError:
        success = False
        msg = "Error while reading image (TypeError): '{}'".format(filename)
        logger.error(msg)
        if raise_error:
            raise Exception(msg)
    except ValueError:
        success = False
        msg = ("Error while reading image. This is most likely caused by special characters in "
               "the filename: '{}'".format(filename))
        logger.error(msg)
        if raise_error:
            raise Exception(msg)
    except Exception as err:  # pylint:disable=broad-except
        success = False
        msg = "Failed to load image '{}'. Original Error: {}".format(filename, str(err))
        logger.error(msg)
        if raise_error:
            raise Exception(msg)
    logger.trace("Loaded image: '%s'. Success: %s", filename, success)
    return image


def hash_image_file(filename):
    """ Return an image file's sha1 hash """
    logger = logging.getLogger(__name__)  # pylint:disable=invalid-name
    img = cv2_read_img(filename, raise_error=True)
    img_hash = sha1(img).hexdigest()
    logger.trace("filename: '%s', hash: %s", filename, img_hash)
    return img_hash


def hash_encode_image(image, extension):
    """ Encode the image, get the hash and return the hash with
        encoded image """
    img = cv2.imencode(extension, image)[1]  # pylint:disable=no-member,c-extension-no-member
    f_hash = sha1(
        cv2.imdecode(  # pylint:disable=no-member,c-extension-no-member
            img,
            cv2.IMREAD_UNCHANGED)).hexdigest()  # pylint:disable=no-member,c-extension-no-member
    return f_hash, img


def backup_file(directory, filename):
    """ Backup a given file by appending .bk to the end """
    logger = logging.getLogger(__name__)  # pylint:disable=invalid-name
    logger.trace("Backing up: '%s'", filename)
    origfile = os.path.join(directory, filename)
    backupfile = origfile + '.bk'
    if os.path.exists(backupfile):
        logger.trace("Removing existing file: '%s'", backup_file)
        os.remove(backupfile)
    if os.path.exists(origfile):
        logger.trace("Renaming: '%s' to '%s'", origfile, backup_file)
        os.rename(origfile, backupfile)


def keras_backend_quiet():
    """ Suppresses the "Using x backend" message when importing
        backend from keras """
    stderr = sys.stderr
    sys.stderr = open(os.devnull, 'w')
    from keras import backend as K
    sys.stderr = stderr
    return K


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

    logger = logging.getLogger(__name__)  # pylint:disable=invalid-name
    from lib.logger import get_loglevel
    numeric_level = get_loglevel(loglevel)
    loglevel = "2" if numeric_level > 15 else "0"
    logger.debug("System Verbosity level: %s", loglevel)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = loglevel
    if loglevel != '0':
        for warncat in (FutureWarning, DeprecationWarning, UserWarning):
            warnings.simplefilter(action='ignore', category=warncat)


def deprecation_warning(func_name, additional_info=None):
    """ Log at warning level that a function will be removed in future """
    logger = logging.getLogger(__name__)  # pylint:disable=invalid-name
    logger.debug("func_name: %s, additional_info: %s", func_name, additional_info)
    msg = "{}  has been deprecated and will be removed from a future update.".format(func_name)
    if additional_info is not None:
        msg += " {}".format(additional_info)
    logger.warning(msg)


def rotate_landmarks(face, rotation_matrix):
    # pylint:disable=c-extension-no-member
    """ Rotate the landmarks and bounding box for faces
        found in rotated images.
        Pass in a DetectedFace object, Alignments dict or bounding box dict
        (as defined in lib/plugins/extract/detect/_base.py) """
    logger = logging.getLogger(__name__)  # pylint:disable=invalid-name
    logger.trace("Rotating landmarks: (rotation_matrix: %s, type(face): %s",
                 rotation_matrix, type(face))
    # Detected Face Object
    if isinstance(face, DetectedFace):
        bounding_box = [[face.x, face.y],
                        [face.x + face.w, face.y],
                        [face.x + face.w, face.y + face.h],
                        [face.x, face.y + face.h]]
        landmarks = face.landmarksXY

    # Alignments Dict
    elif isinstance(face, dict) and "x" in face:
        bounding_box = [[face.get("x", 0), face.get("y", 0)],
                        [face.get("x", 0) + face.get("w", 0),
                         face.get("y", 0)],
                        [face.get("x", 0) + face.get("w", 0),
                         face.get("y", 0) + face.get("h", 0)],
                        [face.get("x", 0),
                         face.get("y", 0) + face.get("h", 0)]]
        landmarks = face.get("landmarksXY", list())

    # Bounding Box Dict
    elif isinstance(face, dict) and "left" in face:
        bounding_box = [[face["left"], face["top"]],
                        [face["right"], face["top"]],
                        [face["right"], face["bottom"]],
                        [face["left"], face["bottom"]]]
        landmarks = list()

    else:
        raise ValueError("Unsupported face type")

    logger.trace("Original landmarks: %s", landmarks)

    rotation_matrix = cv2.invertAffineTransform(  # pylint:disable=no-member
        rotation_matrix)
    rotated = list()
    for item in (bounding_box, landmarks):
        if not item:
            continue
        points = np.array(item, np.int32)
        points = np.expand_dims(points, axis=0)
        transformed = cv2.transform(points,  # pylint:disable=no-member
                                    rotation_matrix).astype(np.int32)
        rotated.append(transformed.squeeze())

    # Bounding box should follow x, y planes, so get min/max
    # for non-90 degree rotations
    pt_x = min([pnt[0] for pnt in rotated[0]])
    pt_y = min([pnt[1] for pnt in rotated[0]])
    pt_x1 = max([pnt[0] for pnt in rotated[0]])
    pt_y1 = max([pnt[1] for pnt in rotated[0]])
    width = pt_x1 - pt_x
    height = pt_y1 - pt_y

    if isinstance(face, DetectedFace):
        face.x = int(pt_x)
        face.y = int(pt_y)
        face.w = int(width)
        face.h = int(height)
        face.r = 0
        if len(rotated) > 1:
            rotated_landmarks = [tuple(point) for point in rotated[1].tolist()]
            face.landmarksXY = rotated_landmarks
    elif isinstance(face, dict) and "x" in face:
        face["x"] = int(pt_x)
        face["y"] = int(pt_y)
        face["w"] = int(width)
        face["h"] = int(height)
        face["r"] = 0
        if len(rotated) > 1:
            rotated_landmarks = [tuple(point) for point in rotated[1].tolist()]
            face["landmarksXY"] = rotated_landmarks
    else:
        face["left"] = int(pt_x)
        face["top"] = int(pt_y)
        face["right"] = int(pt_x1)
        face["bottom"] = int(pt_y1)

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
    logger = logging.getLogger(__name__)  # pylint:disable=invalid-name
    logger.debug("Safely shutting down")
    from lib.queue_manager import queue_manager
    from lib.multithreading import terminate_processes
    queue_manager.terminate_queues()
    terminate_processes()
    logger.debug("Cleanup complete. Shutting down queue manager and exiting")
    queue_manager._log_queue.put(None)  # pylint:disable=protected-access
    while not queue_manager._log_queue.empty():  # pylint:disable=protected-access
        continue
    queue_manager.manager.shutdown()


class FaceswapError(Exception):
    """ Faceswap Error for handling specific errors with useful information """
    pass  # pylint:disable=unnecessary-pass


class GetModel():
    """ Check for models in their cache path
        If available, return the path, if not available, get, unzip and install model

        model_filename: The name of the model to be loaded (see notes below)
        cache_dir:      The model cache folder of the current plugin calling this class
                        IE: The folder that holds the model to be loaded.
        git_model_id:   The second digit in the github tag that identifies this model.
                        See https://github.com/deepfakes-models/faceswap-models for more
                        information

        NB: Models must have a certain naming convention:
            IE: <model_name>_v<version_number>.<extension>
            EG: s3fd_v1.pb

            Multiple models can exist within the model_filename. They should be passed as a list
            and follow the same naming convention as above. Any differences in filename should
            occur AFTER the version number.
            IE: [<model_name>_v<version_number><differentiating_information>.<extension>]
            EG: [mtcnn_det_v1.1.py, mtcnn_det_v1.2.py, mtcnn_det_v1.3.py]
                [resnet_ssd_v1.caffemodel, resnet_ssd_v1.prototext]
        """

    def __init__(self, model_filename, cache_dir, git_model_id):
        self.logger = logging.getLogger(__name__)  # pylint:disable=invalid-name
        if not isinstance(model_filename, list):
            model_filename = [model_filename]
        self.model_filename = model_filename
        self.cache_dir = cache_dir
        self.git_model_id = git_model_id
        self.url_base = "https://github.com/deepfakes-models/faceswap-models/releases/download"
        self.chunk_size = 1024  # Chunk size for downloading and unzipping
        self.retries = 6
        self.get()
        self.model_path = self._model_path

    @property
    def _model_full_name(self):
        """ Return the model full name from the filename(s) """
        common_prefix = os.path.commonprefix(self.model_filename)
        retval = os.path.splitext(common_prefix)[0]
        self.logger.trace(retval)
        return retval

    @property
    def _model_name(self):
        """ Return the model name from the model full name """
        retval = self._model_full_name[:self._model_full_name.rfind("_")]
        self.logger.trace(retval)
        return retval

    @property
    def _model_version(self):
        """ Return the model version from the model full name """
        retval = int(self._model_full_name[self._model_full_name.rfind("_") + 2:])
        self.logger.trace(retval)
        return retval

    @property
    def _model_path(self):
        """ Return the model path(s) in the cache folder """
        retval = [os.path.join(self.cache_dir, fname) for fname in self.model_filename]
        retval = retval[0] if len(retval) == 1 else retval
        self.logger.trace(retval)
        return retval

    @property
    def _model_zip_path(self):
        """ Full path to downloaded zip file """
        retval = os.path.join(self.cache_dir, "{}.zip".format(self._model_full_name))
        self.logger.trace(retval)
        return retval

    @property
    def _model_exists(self):
        """ Check model(s) exist """
        if isinstance(self._model_path, list):
            retval = all(os.path.exists(pth) for pth in self._model_path)
        else:
            retval = os.path.exists(self._model_path)
        self.logger.trace(retval)
        return retval

    @property
    def _plugin_section(self):
        """ Get the plugin section from the config_dir """
        path = os.path.normpath(self.cache_dir)
        split = path.split(os.sep)
        retval = split[split.index("plugins") + 1]
        self.logger.trace(retval)
        return retval

    @property
    def _url_section(self):
        """ Return the section ID in github for this plugin type """
        sections = dict(extract=1, train=2, convert=3)
        retval = sections[self._plugin_section]
        self.logger.trace(retval)
        return retval

    @property
    def _url_download(self):
        """ Base URL for models """
        tag = "v{}.{}.{}".format(self._url_section, self.git_model_id, self._model_version)
        retval = "{}/{}/{}.zip".format(self.url_base, tag, self._model_full_name)
        self.logger.trace("Download url: %s", retval)
        return retval

    @property
    def _url_partial_size(self):
        """ Return how many bytes have already been downloaded """
        zip_file = self._model_zip_path
        retval = os.path.getsize(zip_file) if os.path.exists(zip_file) else 0
        self.logger.trace(retval)
        return retval

    def get(self):
        """ Check the model exists, if not, download and unzip into location """
        if self._model_exists:
            self.logger.debug("Model exists: %s", self._model_path)
            return
        self.download_model()
        self.unzip_model()
        os.remove(self._model_zip_path)

    def download_model(self):
        """ Download model zip to cache dir """
        self.logger.info("Downloading model: '%s' from: %s", self._model_name, self._url_download)
        for attempt in range(self.retries):
            try:
                downloaded_size = self._url_partial_size
                req = urllib.request.Request(self._url_download)
                if downloaded_size != 0:
                    req.add_header("Range", "bytes={}-".format(downloaded_size))
                response = urllib.request.urlopen(req, timeout=10)
                self.logger.debug("header info: {%s}", response.info())
                self.logger.debug("Return Code: %s", response.getcode())
                self.write_zipfile(response, downloaded_size)
                break
            except (socket_error, socket_timeout,
                    urllib.error.HTTPError, urllib.error.URLError) as err:
                if attempt + 1 < self.retries:
                    self.logger.warning("Error downloading model (%s). Retrying %s of %s...",
                                        str(err), attempt + 2, self.retries)
                else:
                    self.logger.error("Failed to download model. Exiting. (Error: '%s', URL: "
                                      "'%s')", str(err), self._url_download)
                    self.logger.info("You can try running again to resume the download.")
                    self.logger.info("Alternatively, you can manually download the model from: %s "
                                     "and unzip the contents to: %s",
                                     self._url_download, self.cache_dir)
                    exit(1)

    def write_zipfile(self, response, downloaded_size):
        """ Write the model zip file to disk """
        length = int(response.getheader("content-length")) + downloaded_size
        if length == downloaded_size:
            self.logger.info("Zip already exists. Skipping download")
            return
        write_type = "wb" if downloaded_size == 0 else "ab"
        with open(self._model_zip_path, write_type) as out_file:
            pbar = tqdm(desc="Downloading",
                        unit="B",
                        total=length,
                        unit_scale=True,
                        unit_divisor=1024)
            if downloaded_size != 0:
                pbar.update(downloaded_size)
            while True:
                buffer = response.read(self.chunk_size)
                if not buffer:
                    break
                pbar.update(len(buffer))
                out_file.write(buffer)

    def unzip_model(self):
        """ Unzip the model file to the cachedir """
        self.logger.info("Extracting: '%s'", self._model_name)
        try:
            zip_file = zipfile.ZipFile(self._model_zip_path, "r")
            self.write_model(zip_file)
        except Exception as err:  # pylint:disable=broad-except
            self.logger.error("Unable to extract model file: %s", str(err))
            exit(1)

    def write_model(self, zip_file):
        """ Extract files from zipfile and write, with progress bar """
        length = sum(f.file_size for f in zip_file.infolist())
        fnames = zip_file.namelist()
        self.logger.debug("Zipfile: Filenames: %s, Total Size: %s", fnames, length)
        pbar = tqdm(desc="Decompressing",
                    unit="B",
                    total=length,
                    unit_scale=True,
                    unit_divisor=1024)
        for fname in fnames:
            out_fname = os.path.join(self.cache_dir, fname)
            self.logger.debug("Extracting from: '%s' to '%s'", self._model_zip_path, out_fname)
            zipped = zip_file.open(fname)
            with open(out_fname, "wb") as out_file:
                while True:
                    buffer = zipped.read(self.chunk_size)
                    if not buffer:
                        break
                    pbar.update(len(buffer))
                    out_file.write(buffer)
        zip_file.close()
