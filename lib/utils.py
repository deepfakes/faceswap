#!/usr/bin python3
""" Utilities available across all scripts """

import importlib
import json
import logging
import os
import sys
import urllib
import warnings
import zipfile

from pathlib import Path
from re import finditer
from multiprocessing import current_process
from socket import timeout as socket_timeout, error as socket_error

from tqdm import tqdm

# Global variables
_image_extensions = [  # pylint:disable=invalid-name
    ".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"]
_video_extensions = [  # pylint:disable=invalid-name
    ".avi", ".flv", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg", ".webm", ".wmv",
    ".ts", ".vob"]


class _Backend():  # pylint:disable=too-few-public-methods
    """ Return the backend from config/.faceswap of from the `FACESWAP_BACKEND` Environment
    Variable.

    If file doesn't exist and a variable hasn't been set, create the config file. """
    def __init__(self):
        self._backends = {"1": "amd", "2": "cpu", "3": "nvidia"}
        self._config_file = self._get_config_file()
        self.backend = self._get_backend()

    @classmethod
    def _get_config_file(cls):
        """ Obtain the location of the main Faceswap configuration file.

        Returns
        -------
        str
            The path to the Faceswap configuration file
        """
        pypath = os.path.dirname(os.path.realpath(sys.argv[0]))
        config_file = os.path.join(pypath, "config", ".faceswap")
        return config_file

    def _get_backend(self):
        """ Return the backend from either the `FACESWAP_BACKEND` Environment Variable or from
        the :file:`config/.faceswap` configuration file. If neither of these exist, prompt the user
        to select a backend.

        Returns
        -------
        str
            The backend configuration in use by Faceswap
        """
        # Check if environment variable is set, if so use that
        if "FACESWAP_BACKEND" in os.environ:
            fs_backend = os.environ["FACESWAP_BACKEND"].lower()
            print("Setting Faceswap backend from environment variable to "
                  "{}".format(fs_backend.upper()))
            return fs_backend
        # Intercept for sphinx docs build
        if sys.argv[0].endswith("sphinx-build"):
            return "nvidia"
        if not os.path.isfile(self._config_file):
            self._configure_backend()
        while True:
            try:
                with open(self._config_file, "r") as cnf:
                    config = json.load(cnf)
                break
            except json.decoder.JSONDecodeError:
                self._configure_backend()
                continue
        fs_backend = config.get("backend", None)
        if fs_backend is None or fs_backend.lower() not in self._backends.values():
            fs_backend = self._configure_backend()
        if current_process().name == "MainProcess":
            print("Setting Faceswap backend to {}".format(fs_backend.upper()))
        return fs_backend.lower()

    def _configure_backend(self):
        """ Get user input to select the backend that Faceswap should use.

        Returns
        -------
        str
            The backend configuration in use by Faceswap
        """
        print("First time configuration. Please select the required backend")
        while True:
            selection = input("1: AMD, 2: CPU, 3: NVIDIA: ")
            if selection not in ("1", "2", "3"):
                print("'{}' is not a valid selection. Please try again".format(selection))
                continue
            break
        fs_backend = self._backends[selection].lower()
        config = {"backend": fs_backend}
        with open(self._config_file, "w") as cnf:
            json.dump(config, cnf)
        print("Faceswap config written to: {}".format(self._config_file))
        return fs_backend


_FS_BACKEND = _Backend().backend


def get_backend():
    """ Get the backend that Faceswap is currently configured to use.

    Returns
    -------
    str
        The backend configuration in use by Faceswap
    """
    return _FS_BACKEND


def set_backend(backend):
    """ Override the configured backend with the given backend.

    Parameters
    ----------
    backend: ["amd", "cpu", "nvidia"]
        The backend to set faceswap to
    """
    global _FS_BACKEND  # pylint:disable=global-statement
    _FS_BACKEND = backend.lower()


def get_folder(path, make_folder=True):
    """ Return a path to a folder, creating it if it doesn't exist

    Parameters
    ----------
    path: str
        The path to the folder to obtain
    make_folder: bool, optional
        ``True`` if the folder should be created if it does not already exist, ``False`` if the
        folder should not be created

    Returns
    -------
    :class:`pathlib.Path` or `None`
        The path to the requested folder. If `make_folder` is set to ``False`` and the requested
        path does not exist, then ``None`` is returned
    """
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
    """ Obtain a list of full paths that reside within a folder.

    Parameters
    ----------
    directory: str
        The folder that contains the images to be returned

    Returns
    -------
    list
        The list of full paths to the images contained within the given folder
    """
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


def convert_to_secs(*args):
    """ Convert a time to seconds.

    Parameters
    ----------
    args: tuple
        2 or 3 ints. If 2 ints are supplied, then (`minutes`, `seconds`) is implied. If 3 ints are
        supplied then (`hours`, `minutes`, `seconds`) is implied.

    Returns
    -------
    int
        The given time converted to seconds
    """
    logger = logging.getLogger(__name__)  # pylint:disable=invalid-name
    logger.debug("from time: %s", args)
    retval = 0.0
    if len(args) == 1:
        retval = float(args[0])
    elif len(args) == 2:
        retval = 60 * float(args[0]) + float(args[1])
    elif len(args) == 3:
        retval = 3600 * float(args[0]) + 60 * float(args[1]) + float(args[2])
    logger.debug("to secs: %s", retval)
    return retval


def full_path_split(path):
    """ Split a full path to a location into all of it's separate components.

    Parameters
    ----------
    path: str
        The full path to be split

    Returns
    -------
    list
        The full path split into a separate item for each part

    Example
    -------
    >>> path = "/foo/baz/bar"
    >>> full_path_split(path)
    >>> ["foo", "baz", "bar"]
    """
    logger = logging.getLogger(__name__)  # pylint:disable=invalid-name
    allparts = list()
    while True:
        parts = os.path.split(path)
        if parts[0] == path:   # sentinel for absolute paths
            allparts.insert(0, parts[0])
            break
        if parts[1] == path:  # sentinel for relative paths
            allparts.insert(0, parts[1])
            break
        path = parts[0]
        allparts.insert(0, parts[1])
    logger.trace("path: %s, allparts: %s", path, allparts)
    return allparts


def set_system_verbosity(log_level):
    """ Set the verbosity level of tensorflow and suppresses future and deprecation warnings from
    any modules

    Parameters
    ----------
    log_level: str
        The requested Faceswap log level

    References
    ----------
    https://stackoverflow.com/questions/35911252/disable-tensorflow-debugging-information
    Can be set to:
    0: all logs shown. 1: filter out INFO logs. 2: filter out WARNING logs. 3: filter out ERROR
    logs.
    """

    logger = logging.getLogger(__name__)  # pylint:disable=invalid-name
    from lib.logger import get_loglevel  # pylint:disable=import-outside-toplevel
    numeric_level = get_loglevel(log_level)
    log_level = "2" if numeric_level > 15 else "0"
    logger.debug("System Verbosity level: %s", log_level)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = log_level
    if log_level != '0':
        for warncat in (FutureWarning, DeprecationWarning, UserWarning):
            warnings.simplefilter(action='ignore', category=warncat)


def deprecation_warning(function, additional_info=None):
    """ Log at warning level that a function will be removed in a future update.

    Parameters
    ----------
    function: str
        The function that will be deprecated.
    additional_info: str, optional
        Any additional information to display with the deprecation message. Default: ``None``
    """
    logger = logging.getLogger(__name__)  # pylint:disable=invalid-name
    logger.debug("func_name: %s, additional_info: %s", function, additional_info)
    msg = "{}  has been deprecated and will be removed from a future update.".format(function)
    if additional_info is not None:
        msg += " {}".format(additional_info)
    logger.warning(msg)


def camel_case_split(identifier):
    """ Split a camel case name

    Parameters
    ----------
    identifier: str
        The camel case text to be split

    Returns
    -------
    list
        A list of the given identifier split into it's constituent parts


    References
    ----------
    https://stackoverflow.com/questions/29916065
    """
    matches = finditer(
        ".+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)",
        identifier)
    return [m.group(0) for m in matches]


def safe_shutdown(got_error=False):
    """ Close all tracked queues and threads in event of crash or on shut down.

    Parameters
    ----------
    got_error: bool, optional
        ``True`` if this function is being called as the result of raised error, otherwise
        ``False``. Default: ``False``
    """
    logger = logging.getLogger(__name__)  # pylint:disable=invalid-name
    logger.debug("Safely shutting down")
    from lib.queue_manager import queue_manager  # pylint:disable=import-outside-toplevel
    queue_manager.terminate_queues()
    logger.debug("Cleanup complete. Shutting down queue manager and exiting")
    sys.exit(1 if got_error else 0)


class FaceswapError(Exception):
    """ Faceswap Error for handling specific errors with useful information.

    Raises
    ------
    FaceswapError
        on a captured error
    """
    pass  # pylint:disable=unnecessary-pass


class GetModel():  # Pylint:disable=too-few-public-methods
    """ Check for models in their cache path.

    If available, return the path, if not available, get, unzip and install model

    Parameters
    ----------
    model_filename: str or list
        The name of the model to be loaded (see notes below)
    cache_dir: str
        The model cache folder of the current plugin calling this class. IE: The folder that holds
        the model to be loaded.
    git_model_id: int
        The second digit in the github tag that identifies this model. See
        https://github.com/deepfakes-models/faceswap-models for more information

    Notes
    ------
    Models must have a certain naming convention: `<model_name>_v<version_number>.<extension>`
    (eg: `s3fd_v1.pb`).

    Multiple models can exist within the model_filename. They should be passed as a list and follow
    the same naming convention as above. Any differences in filename should occur AFTER the version
    number: `<model_name>_v<version_number><differentiating_information>.<extension>` (eg:
    `["mtcnn_det_v1.1.py", "mtcnn_det_v1.2.py", "mtcnn_det_v1.3.py"]`, `["resnet_ssd_v1.caffemodel"
    ,"resnet_ssd_v1.prototext"]`
    """

    def __init__(self, model_filename, cache_dir, git_model_id):
        self.logger = logging.getLogger(__name__)
        if not isinstance(model_filename, list):
            model_filename = [model_filename]
        self._model_filename = model_filename
        self._cache_dir = cache_dir
        self._git_model_id = git_model_id
        self._url_base = "https://github.com/deepfakes-models/faceswap-models/releases/download"
        self._chunk_size = 1024  # Chunk size for downloading and unzipping
        self._retries = 6
        self._get()

    @property
    def _model_full_name(self):
        """ str: The full model name from the filename(s). """
        common_prefix = os.path.commonprefix(self._model_filename)
        retval = os.path.splitext(common_prefix)[0]
        self.logger.trace(retval)
        return retval

    @property
    def _model_name(self):
        """ str: The model name from the model's full name. """
        retval = self._model_full_name[:self._model_full_name.rfind("_")]
        self.logger.trace(retval)
        return retval

    @property
    def _model_version(self):
        """ int: The model's version number from the model full name. """
        retval = int(self._model_full_name[self._model_full_name.rfind("_") + 2:])
        self.logger.trace(retval)
        return retval

    @property
    def model_path(self):
        """ str: The model path(s) in the cache folder. """
        retval = [os.path.join(self._cache_dir, fname) for fname in self._model_filename]
        retval = retval[0] if len(retval) == 1 else retval
        self.logger.trace(retval)
        return retval

    @property
    def _model_zip_path(self):
        """ str: The full path to downloaded zip file. """
        retval = os.path.join(self._cache_dir, "{}.zip".format(self._model_full_name))
        self.logger.trace(retval)
        return retval

    @property
    def _model_exists(self):
        """ bool: ``True`` if the model exists in the cache folder otherwise ``False``. """
        if isinstance(self.model_path, list):
            retval = all(os.path.exists(pth) for pth in self.model_path)
        else:
            retval = os.path.exists(self.model_path)
        self.logger.trace(retval)
        return retval

    @property
    def _plugin_section(self):
        """ str: The plugin section from the config_dir """
        path = os.path.normpath(self._cache_dir)
        split = path.split(os.sep)
        retval = split[split.index("plugins") + 1]
        self.logger.trace(retval)
        return retval

    @property
    def _url_section(self):
        """ int: The section ID in github for this plugin type. """
        sections = dict(extract=1, train=2, convert=3)
        retval = sections[self._plugin_section]
        self.logger.trace(retval)
        return retval

    @property
    def _url_download(self):
        """ strL Base download URL for models. """
        tag = "v{}.{}.{}".format(self._url_section, self._git_model_id, self._model_version)
        retval = "{}/{}/{}.zip".format(self._url_base, tag, self._model_full_name)
        self.logger.trace("Download url: %s", retval)
        return retval

    @property
    def _url_partial_size(self):
        """ float: How many bytes have already been downloaded. """
        zip_file = self._model_zip_path
        retval = os.path.getsize(zip_file) if os.path.exists(zip_file) else 0
        self.logger.trace(retval)
        return retval

    def _get(self):
        """ Check the model exists, if not, download the model, unzip it and place it in the
        model's cache folder. """
        if self._model_exists:
            self.logger.debug("Model exists: %s", self.model_path)
            return
        self._download_model()
        self._unzip_model()
        os.remove(self._model_zip_path)

    def _download_model(self):
        """ Download the model zip from github to the cache folder. """
        self.logger.info("Downloading model: '%s' from: %s", self._model_name, self._url_download)
        for attempt in range(self._retries):
            try:
                downloaded_size = self._url_partial_size
                req = urllib.request.Request(self._url_download)
                if downloaded_size != 0:
                    req.add_header("Range", "bytes={}-".format(downloaded_size))
                response = urllib.request.urlopen(req, timeout=10)
                self.logger.debug("header info: {%s}", response.info())
                self.logger.debug("Return Code: %s", response.getcode())
                self._write_zipfile(response, downloaded_size)
                break
            except (socket_error, socket_timeout,
                    urllib.error.HTTPError, urllib.error.URLError) as err:
                if attempt + 1 < self._retries:
                    self.logger.warning("Error downloading model (%s). Retrying %s of %s...",
                                        str(err), attempt + 2, self._retries)
                else:
                    self.logger.error("Failed to download model. Exiting. (Error: '%s', URL: "
                                      "'%s')", str(err), self._url_download)
                    self.logger.info("You can try running again to resume the download.")
                    self.logger.info("Alternatively, you can manually download the model from: %s "
                                     "and unzip the contents to: %s",
                                     self._url_download, self._cache_dir)
                    sys.exit(1)

    def _write_zipfile(self, response, downloaded_size):
        """ Write the model zip file to disk.

        Parameters
        ----------
        response: :class:`urllib.request.urlopen`
            The response from the model download task
        downloaded_size: int
            The amount of bytes downloaded so far
        """
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
                buffer = response.read(self._chunk_size)
                if not buffer:
                    break
                pbar.update(len(buffer))
                out_file.write(buffer)
            pbar.close()

    def _unzip_model(self):
        """ Unzip the model file to the cache folder """
        self.logger.info("Extracting: '%s'", self._model_name)
        try:
            zip_file = zipfile.ZipFile(self._model_zip_path, "r")
            self._write_model(zip_file)
        except Exception as err:  # pylint:disable=broad-except
            self.logger.error("Unable to extract model file: %s", str(err))
            sys.exit(1)

    def _write_model(self, zip_file):
        """ Extract files from zip file and write, with progress bar.

        Parameters
        ----------
        zip_file: str
            The downloaded model zip file
        """
        length = sum(f.file_size for f in zip_file.infolist())
        fnames = zip_file.namelist()
        self.logger.debug("Zipfile: Filenames: %s, Total Size: %s", fnames, length)
        pbar = tqdm(desc="Decompressing",
                    unit="B",
                    total=length,
                    unit_scale=True,
                    unit_divisor=1024)
        for fname in fnames:
            out_fname = os.path.join(self._cache_dir, fname)
            self.logger.debug("Extracting from: '%s' to '%s'", self._model_zip_path, out_fname)
            zipped = zip_file.open(fname)
            with open(out_fname, "wb") as out_file:
                while True:
                    buffer = zipped.read(self._chunk_size)
                    if not buffer:
                        break
                    pbar.update(len(buffer))
                    out_file.write(buffer)
        zip_file.close()
        pbar.close()


class KerasFinder(importlib.abc.MetaPathFinder):
    """ Importlib Abstract Base Class for intercepting the import of Keras and returning either
    Keras (AMD backend) or tensorflow.keras (any other backend).

    The Importlib documentation is sparse at best, and real world examples are pretty much
    non-existent. Coupled with this, the import ``tensorflow.keras`` does not resolve so we need
    to split out to the actual location of Keras within ``tensorflow_core``. This method works, but
    it relies on hard coded paths, and is likely to not be the most robust.

    A custom loader is not used, as we can use the standard loader once we have returned the
    correct spec.
    """
    def __init__(self):
        self._logger = logging.getLogger(__name__)
        self._backend = get_backend()
        self._tf_keras_locations = [["tensorflow_core", "python", "keras", "api", "_v2"],
                                    ["tensorflow", "python", "keras", "api", "_v2"]]

    def find_spec(self, fullname, path, target=None):  # pylint:disable=unused-argument
        """ Obtain the spec for either keras or tensorflow.keras depending on the backend in use.

        If keras is not passed in as part of the :attr:`fullname` or the path is not ``None``
        (i.e this is a dependency import) then this returns ``None`` to use the standard import
        library.

        Parameters
        ----------
        fullname: str
            The absolute name of the module to be imported
        path: str
            The search path for the module
        target: module object, optional
            Inherited from parent but unused

        Returns
        -------
        :class:`importlib.ModuleSpec`
            The spec for the Keras module to be imported
        """
        prefix = fullname.split(".")[0]
        suffix = fullname.split(".")[-1]
        if prefix != "keras" or path is not None:
            return None
        self._logger.debug("Importing '%s' as keras for backend: '%s'",
                           "keras" if self._backend == "amd" else "tf.keras", self._backend)
        path = sys.path if path is None else path
        for entry in path:
            locations = ([os.path.join(entry, *location)
                          for location in self._tf_keras_locations]
                         if self._backend != "amd" else [entry])
            for location in locations:
                self._logger.debug("Scanning: '%s' for '%s'", location, suffix)
                if os.path.isdir(os.path.join(location, suffix)):
                    filename = os.path.join(location, suffix, "__init__.py")
                    submodule_locations = [os.path.join(location, suffix)]
                else:
                    filename = os.path.join(location, suffix + ".py")
                    submodule_locations = None
                if not os.path.exists(filename):
                    continue
                retval = importlib.util.spec_from_file_location(
                    fullname,
                    filename,
                    submodule_search_locations=submodule_locations)
                self._logger.debug("Found spec: %s", retval)
                return retval
        self._logger.debug("Spec not found for '%s'. Falling back to default import", fullname)
        return None
