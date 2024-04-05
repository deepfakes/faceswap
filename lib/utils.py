#!/usr/bin python3
""" Utilities available across all scripts """
from __future__ import annotations
import json
import logging
import os
import sys
import tkinter as tk
import typing as T
import warnings
import zipfile

from multiprocessing import current_process
from re import finditer
from socket import timeout as socket_timeout, error as socket_error
from threading import get_ident
from time import time
from urllib import request, error as urlliberror

import numpy as np
from tqdm import tqdm

if T.TYPE_CHECKING:
    from argparse import Namespace
    from http.client import HTTPResponse

# Global variables
IMAGE_EXTENSIONS = [".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"]
VIDEO_EXTENSIONS = [".avi", ".flv", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg", ".webm", ".wmv",
                    ".ts", ".vob"]
_TF_VERS: tuple[int, int] | None = None
ValidBackends = T.Literal["nvidia", "cpu", "apple_silicon", "directml", "rocm"]


class _Backend():  # pylint:disable=too-few-public-methods
    """ Return the backend from config/.faceswap of from the `FACESWAP_BACKEND` Environment
    Variable.

    If file doesn't exist and a variable hasn't been set, create the config file. """
    def __init__(self) -> None:
        self._backends: dict[str, ValidBackends] = {"1": "cpu",
                                                    "2": "directml",
                                                    "3": "nvidia",
                                                    "4": "apple_silicon",
                                                    "5": "rocm"}
        self._valid_backends = list(self._backends.values())
        self._config_file = self._get_config_file()
        self.backend = self._get_backend()

    @classmethod
    def _get_config_file(cls) -> str:
        """ Obtain the location of the main Faceswap configuration file.

        Returns
        -------
        str
            The path to the Faceswap configuration file
        """
        pypath = os.path.dirname(os.path.realpath(sys.argv[0]))
        config_file = os.path.join(pypath, "config", ".faceswap")
        return config_file

    def _get_backend(self) -> ValidBackends:
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
            fs_backend = T.cast(ValidBackends, os.environ["FACESWAP_BACKEND"].lower())
            assert fs_backend in T.get_args(ValidBackends), (
                f"Faceswap backend must be one of {T.get_args(ValidBackends)}")
            print(f"Setting Faceswap backend from environment variable to {fs_backend.upper()}")
            return fs_backend
        # Intercept for sphinx docs build
        if sys.argv[0].endswith("sphinx-build"):
            return "nvidia"
        if not os.path.isfile(self._config_file):
            self._configure_backend()
        while True:
            try:
                with open(self._config_file, "r", encoding="utf8") as cnf:
                    config = json.load(cnf)
                break
            except json.decoder.JSONDecodeError:
                self._configure_backend()
                continue
        fs_backend = config.get("backend", "").lower()
        if not fs_backend or fs_backend not in self._backends.values():
            fs_backend = self._configure_backend()
        if current_process().name == "MainProcess":
            print(f"Setting Faceswap backend to {fs_backend.upper()}")
        return fs_backend

    def _configure_backend(self) -> ValidBackends:
        """ Get user input to select the backend that Faceswap should use.

        Returns
        -------
        str
            The backend configuration in use by Faceswap
        """
        print("First time configuration. Please select the required backend")
        while True:
            txt = ", ".join([": ".join([key, val.upper().replace("_", " ")])
                             for key, val in self._backends.items()])
            selection = input(f"{txt}: ")
            if selection not in self._backends:
                print(f"'{selection}' is not a valid selection. Please try again")
                continue
            break
        fs_backend = self._backends[selection]
        config = {"backend": fs_backend}
        with open(self._config_file, "w", encoding="utf8") as cnf:
            json.dump(config, cnf)
        print(f"Faceswap config written to: {self._config_file}")
        return fs_backend


_FS_BACKEND: ValidBackends = _Backend().backend


def get_backend() -> ValidBackends:
    """ Get the backend that Faceswap is currently configured to use.

    Returns
    -------
    str
        The backend configuration in use by Faceswap. One of  ["cpu", "directml", "nvidia", "rocm",
        "apple_silicon"]

    Example
    -------
    >>> from lib.utils import get_backend
    >>> get_backend()
    'nvidia'
    """
    return _FS_BACKEND


def set_backend(backend: str) -> None:
    """ Override the configured backend with the given backend.

    Parameters
    ----------
    backend: ["cpu", "directml", "nvidia", "rocm", "apple_silicon"]
        The backend to set faceswap to

    Example
    -------
    >>> from lib.utils import set_backend
    >>> set_backend("nvidia")
    """
    global _FS_BACKEND  # pylint:disable=global-statement
    backend = T.cast(ValidBackends, backend.lower())
    _FS_BACKEND = backend


def get_tf_version() -> tuple[int, int]:
    """ Obtain the major. minor version of currently installed Tensorflow.

    Returns
    -------
    tuple[int, int]
        A tuple of the form (major, minor) representing the version of TensorFlow that is installed

    Example
    -------
    >>> from lib.utils import get_tf_version
    >>> get_tf_version()
    (2, 10)
    """
    global _TF_VERS  # pylint:disable=global-statement
    if _TF_VERS is None:
        import tensorflow as tf  # pylint:disable=import-outside-toplevel
        split = tf.__version__.split(".")[:2]
        _TF_VERS = (int(split[0]), int(split[1]))
    return _TF_VERS


def get_folder(path: str, make_folder: bool = True) -> str:
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
    str or `None`
        The path to the requested folder. If `make_folder` is set to ``False`` and the requested
        path does not exist, then ``None`` is returned

    Example
    -------
    >>> from lib.utils import get_folder
    >>> get_folder('/tmp/myfolder')
    '/tmp/myfolder'

    >>> get_folder('/tmp/myfolder', make_folder=False)
    ''
    """
    logger = logging.getLogger(__name__)
    logger.debug("Requested path: '%s'", path)
    if not make_folder and not os.path.isdir(path):
        logger.debug("%s does not exist", path)
        return ""
    os.makedirs(path, exist_ok=True)
    logger.debug("Returning: '%s'", path)
    return path


def get_image_paths(directory: str, extension: str | None = None) -> list[str]:
    """ Gets the image paths from a given directory.

    The function searches for files with the specified extension(s) in the given directory, and
    returns a list of their paths. If no extension is provided, the function will search for files
    with any of the following extensions: '.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff'

    Parameters
    ----------
    directory: str
        The directory to search in
    extension: str
        The file extension to search for. If not provided, all image file types will be searched
        for

    Returns
    -------
    list[str]
        The list of full paths to the images contained within the given folder

    Example
    -------
    >>> from lib.utils import get_image_paths
    >>> get_image_paths('/path/to/directory')
    ['/path/to/directory/image1.jpg', '/path/to/directory/image2.png']
    >>> get_image_paths('/path/to/directory', '.jpg')
    ['/path/to/directory/image1.jpg']
    """
    logger = logging.getLogger(__name__)
    image_extensions = IMAGE_EXTENSIONS if extension is None else [extension]
    dir_contents = []

    if not os.path.exists(directory):
        logger.debug("Creating folder: '%s'", directory)
        directory = get_folder(directory)

    dir_scanned = sorted(os.scandir(directory), key=lambda x: x.name)
    logger.debug("Scanned Folder contains %s files", len(dir_scanned))
    logger.trace("Scanned Folder Contents: %s", dir_scanned)  # type:ignore[attr-defined]

    for chkfile in dir_scanned:
        if any(chkfile.name.lower().endswith(ext) for ext in image_extensions):
            logger.trace("Adding '%s' to image list", chkfile.path)  # type:ignore[attr-defined]
            dir_contents.append(chkfile.path)

    logger.debug("Returning %s images", len(dir_contents))
    return dir_contents


def get_dpi() -> float | None:
    """ Gets the DPI (dots per inch) of the display screen.

    Returns
    -------
    float or ``None``
        The DPI of the display screen or ``None`` if the dpi couldn't be obtained (ie: if the
        function is called on a headless system)

    Example
    -------
    >>> from lib.utils import get_dpi
    >>> get_dpi()
    96.0
    """
    logger = logging.getLogger(__name__)
    try:
        root = tk.Tk()
        dpi = root.winfo_fpixels('1i')
    except tk.TclError:
        logger.warning("Display not detected. Could not obtain DPI")
        return None

    return float(dpi)


def convert_to_secs(*args: int) -> int:
    """  Convert time in hours, minutes, and seconds to seconds.

    Parameters
    ----------
    *args: int
        1, 2 or 3 ints. If 2 ints are supplied, then (`minutes`, `seconds`) is implied. If 3 ints
        are supplied then (`hours`, `minutes`, `seconds`) is implied.

    Returns
    -------
    int
        The given time converted to seconds

    Example
    -------
    >>> from lib.utils import convert_to_secs
    >>> convert_to_secs(1, 30, 0)
    5400
    >>> convert_to_secs(0, 15, 30)
    930
    >>> convert_to_secs(0, 0, 45)
    45
    """
    logger = logging.getLogger(__name__)
    logger.debug("from time: %s", args)
    retval = 0.0
    if len(args) == 1:
        retval = float(args[0])
    elif len(args) == 2:
        retval = 60 * float(args[0]) + float(args[1])
    elif len(args) == 3:
        retval = 3600 * float(args[0]) + 60 * float(args[1]) + float(args[2])
    retval = int(retval)
    logger.debug("to secs: %s", retval)
    return retval


def full_path_split(path: str) -> list[str]:
    """ Split a file path into all of its parts.

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
    >>> from lib.utils import full_path_split
    >>> full_path_split("/usr/local/bin/python")
    ['usr', 'local', 'bin', 'python']
    >>> full_path_split("relative/path/to/file.txt")
    ['relative', 'path', 'to', 'file.txt']]
    """
    logger = logging.getLogger(__name__)
    allparts: list[str] = []
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
    logger.trace("path: %s, allparts: %s", path, allparts)  # type:ignore[attr-defined]
    # Remove any empty strings which may have got inserted
    allparts = [part for part in allparts if part]
    return allparts


def set_system_verbosity(log_level: str):
    """ Set the verbosity level of tensorflow and suppresses future and deprecation warnings from
    any modules.

    This function sets the `TF_CPP_MIN_LOG_LEVEL` environment variable to control the verbosity of
    TensorFlow output, as well as filters certain warning types to be ignored. The log level is
    determined based on the input string `log_level`.

    Parameters
    ----------
    log_level: str
        The requested Faceswap log level.

    References
    ----------
    https://stackoverflow.com/questions/35911252/disable-tensorflow-debugging-information

    Example
    -------
    >>> from lib.utils import set_system_verbosity
    >>> set_system_verbosity('warning')
    """
    logger = logging.getLogger(__name__)
    from lib.logger import get_loglevel  # pylint:disable=import-outside-toplevel
    numeric_level = get_loglevel(log_level)
    log_level = "3" if numeric_level > 15 else "0"
    logger.debug("System Verbosity level: %s", log_level)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = log_level
    if log_level != '0':
        for warncat in (FutureWarning, DeprecationWarning, UserWarning):
            warnings.simplefilter(action='ignore', category=warncat)


def deprecation_warning(function: str, additional_info: str | None = None) -> None:
    """ Log a deprecation warning message.

    This function logs a warning message to indicate that the specified function has been
    deprecated and will be removed in future. An optional additional message can also be included.

    Parameters
    ----------
    function: str
        The name of the function that will be deprecated.
    additional_info: str, optional
        Any additional information to display with the deprecation message. Default: ``None``

    Example
    -------
    >>> from lib.utils import deprecation_warning
    >>> deprecation_warning('old_function', 'Use new_function instead.')
    """
    logger = logging.getLogger(__name__)
    logger.debug("func_name: %s, additional_info: %s", function, additional_info)
    msg = f"{function} has been deprecated and will be removed from a future update."
    if additional_info is not None:
        msg += f" {additional_info}"
    logger.warning(msg)


def handle_deprecated_cliopts(arguments: Namespace) -> Namespace:
    """ Handle deprecated command line arguments and update to correct argument.

    Deprecated cli opts will be provided in the following format:
    `"depr_<option_key>_<deprecated_opt>_<new_opt>"`

    Parameters
    ----------
    arguments: :class:`argpares.Namespace`
        The passed in faceswap cli arguments

    Returns
    -------
    :class:`argpares.Namespace`
        The cli arguments with deprecated values mapped to the correct entry
    """
    logger = logging.getLogger(__name__)

    for key, selected in vars(arguments).items():
        if not key.startswith("depr_") or key.startswith("depr_") and selected is None:
            continue  # Not a deprecated opt
        if isinstance(selected, bool) and not selected:
            continue  # store-true opt with default value

        opt, old, new = key.replace("depr_", "").rsplit("_", maxsplit=2)
        deprecation_warning(f"Command line option '-{old}'", f"Use '-{new}, --{opt}' instead")

        exist = getattr(arguments, opt)
        if exist == selected:
            logger.debug("Keeping existing '%s' value of '%s'", opt, exist)
        else:
            logger.debug("Updating arg '%s' from '%s' to '%s' from deprecated opt",
                         opt, exist, selected)

    return arguments


def camel_case_split(identifier: str) -> list[str]:
    """ Split a camelCase string into a list of its individual parts

    Parameters
    ----------
    identifier: str
        The camelCase text to be split

    Returns
    -------
    list[str]
        A list of the individual parts of the camelCase string.

    References
    ----------
    https://stackoverflow.com/questions/29916065

    Example
    -------
    >>> from lib.utils import camel_case_split
    >>> camel_case_split('camelCaseExample')
    ['camel', 'Case', 'Example']
    """
    matches = finditer(
        ".+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)",
        identifier)
    return [m.group(0) for m in matches]


def safe_shutdown(got_error: bool = False) -> None:
    """ Safely shut down the system.

    This function terminates the queue manager and exits the program in a clean and orderly manner.
    An optional boolean parameter can be used to indicate whether an error occurred during the
    program's execution.

    Parameters
    ----------
    got_error: bool, optional
        ``True`` if this function is being called as the result of raised error. Default: ``False``

    Example
    -------
    >>> from lib.utils import safe_shutdown
    >>> safe_shutdown()
    >>> safe_shutdown(True)
    """
    logger = logging.getLogger(__name__)
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

    Example
    -------
    >>> from lib.utils import FaceswapError
    >>> try:
    ...     # Some code that may raise an error
    ... except SomeError:
    ...     raise FaceswapError("There was an error while running the code")
    FaceswapError: There was an error while running the code
    """
    pass  # pylint:disable=unnecessary-pass


class GetModel():
    """ Check for models in the cache path.

    If available, return the path, if not available, get, unzip and install model

    Parameters
    ----------
    model_filename: str or list
        The name of the model to be loaded (see notes below)
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

    Example
    -------
    >>> from lib.utils import GetModel
    >>> model_downloader = GetModel("s3fd_keras_v2.h5", 11)
    """

    def __init__(self, model_filename: str | list[str], git_model_id: int) -> None:
        self.logger = logging.getLogger(__name__)
        if not isinstance(model_filename, list):
            model_filename = [model_filename]
        self._model_filename = model_filename
        self._cache_dir = os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])), ".fs_cache")
        self._git_model_id = git_model_id
        self._url_base = "https://github.com/deepfakes-models/faceswap-models/releases/download"
        self._chunk_size = 1024  # Chunk size for downloading and unzipping
        self._retries = 6
        self._get()

    @property
    def _model_full_name(self) -> str:
        """ str: The full model name from the filename(s). """
        common_prefix = os.path.commonprefix(self._model_filename)
        retval = os.path.splitext(common_prefix)[0]
        self.logger.trace(retval)  # type:ignore[attr-defined]
        return retval

    @property
    def _model_name(self) -> str:
        """ str: The model name from the model's full name. """
        retval = self._model_full_name[:self._model_full_name.rfind("_")]
        self.logger.trace(retval)  # type:ignore[attr-defined]
        return retval

    @property
    def _model_version(self) -> int:
        """ int: The model's version number from the model full name. """
        retval = int(self._model_full_name[self._model_full_name.rfind("_") + 2:])
        self.logger.trace(retval)  # type:ignore[attr-defined]
        return retval

    @property
    def model_path(self) -> str | list[str]:
        """ str or list[str]: The model path(s) in the cache folder.

        Example
        -------
        >>> from lib.utils import GetModel
        >>> model_downloader = GetModel("s3fd_keras_v2.h5", 11)
        >>> model_downloader.model_path
        '/path/to/s3fd_keras_v2.h5'
        """
        paths = [os.path.join(self._cache_dir, fname) for fname in self._model_filename]
        retval: str | list[str] = paths[0] if len(paths) == 1 else paths
        self.logger.trace(retval)  # type:ignore[attr-defined]
        return retval

    @property
    def _model_zip_path(self) -> str:
        """ str: The full path to downloaded zip file. """
        retval = os.path.join(self._cache_dir, f"{self._model_full_name}.zip")
        self.logger.trace(retval)  # type:ignore[attr-defined]
        return retval

    @property
    def _model_exists(self) -> bool:
        """ bool: ``True`` if the model exists in the cache folder otherwise ``False``. """
        if isinstance(self.model_path, list):
            retval = all(os.path.exists(pth) for pth in self.model_path)
        else:
            retval = os.path.exists(self.model_path)
        self.logger.trace(retval)  # type:ignore[attr-defined]
        return retval

    @property
    def _url_download(self) -> str:
        """ strL Base download URL for models. """
        tag = f"v{self._git_model_id}.{self._model_version}"
        retval = f"{self._url_base}/{tag}/{self._model_full_name}.zip"
        self.logger.trace("Download url: %s", retval)  # type:ignore[attr-defined]
        return retval

    @property
    def _url_partial_size(self) -> int:
        """ int: How many bytes have already been downloaded. """
        zip_file = self._model_zip_path
        retval = os.path.getsize(zip_file) if os.path.exists(zip_file) else 0
        self.logger.trace(retval)  # type:ignore[attr-defined]
        return retval

    def _get(self) -> None:
        """ Check the model exists, if not, download the model, unzip it and place it in the
        model's cache folder. """
        if self._model_exists:
            self.logger.debug("Model exists: %s", self.model_path)
            return
        self._download_model()
        self._unzip_model()
        os.remove(self._model_zip_path)

    def _download_model(self) -> None:
        """ Download the model zip from github to the cache folder. """
        self.logger.info("Downloading model: '%s' from: %s", self._model_name, self._url_download)
        for attempt in range(self._retries):
            try:
                downloaded_size = self._url_partial_size
                req = request.Request(self._url_download)
                if downloaded_size != 0:
                    req.add_header("Range", f"bytes={downloaded_size}-")
                with request.urlopen(req, timeout=10) as response:
                    self.logger.debug("header info: {%s}", response.info())
                    self.logger.debug("Return Code: %s", response.getcode())
                    self._write_zipfile(response, downloaded_size)
                break
            except (socket_error, socket_timeout,
                    urlliberror.HTTPError, urlliberror.URLError) as err:
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

    def _write_zipfile(self, response: HTTPResponse, downloaded_size: int) -> None:
        """ Write the model zip file to disk.

        Parameters
        ----------
        response: :class:`http.client.HTTPResponse`
            The response from the model download task
        downloaded_size: int
            The amount of bytes downloaded so far
        """
        content_length = response.getheader("content-length")
        content_length = "0" if content_length is None else content_length
        length = int(content_length) + downloaded_size
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

    def _unzip_model(self) -> None:
        """ Unzip the model file to the cache folder """
        self.logger.info("Extracting: '%s'", self._model_name)
        try:
            with zipfile.ZipFile(self._model_zip_path, "r") as zip_file:
                self._write_model(zip_file)
        except Exception as err:  # pylint:disable=broad-except
            self.logger.error("Unable to extract model file: %s", str(err))
            sys.exit(1)

    def _write_model(self, zip_file: zipfile.ZipFile) -> None:
        """ Extract files from zip file and write, with progress bar.

        Parameters
        ----------
        zip_file: :class:`zipfile.ZipFile`
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
        pbar.close()


class DebugTimes():
    """ A simple tool to help debug timings.

    Parameters
    ----------
    min: bool, Optional
        Display minimum time taken in summary stats. Default: ``True``
    mean: bool, Optional
        Display mean time taken in summary stats. Default: ``True``
    max: bool, Optional
        Display maximum time taken in summary stats. Default: ``True``

    Example
    -------
    >>> from lib.utils import DebugTimes
    >>> debug_times = DebugTimes()
    >>> debug_times.step_start("step 1")
    >>> # do something here
    >>> debug_times.step_end("step 1")
    >>> debug_times.summary()
    ----------------------------------
    Step             Count   Min
    ----------------------------------
    step 1           1       0.000000
    """
    def __init__(self,
                 show_min: bool = True, show_mean: bool = True, show_max: bool = True) -> None:
        self._times: dict[str, list[float]] = {}
        self._steps: dict[str, float] = {}
        self._interval = 1
        self._display = {"min": show_min, "mean": show_mean, "max": show_max}

    def step_start(self, name: str, record: bool = True) -> None:
        """ Start the timer for the given step name.

        Parameters
        ----------
        name: str
            The name of the step to start the timer for
        record: bool, optional
            ``True`` to record the step time, ``False`` to not record it.
            Used for when you have conditional code to time, but do not want to insert if/else
            statements in the code. Default: `True`

        Example
        -------
        >>> from lib.util import DebugTimes
        >>> debug_times = DebugTimes()
        >>> debug_times.step_start("Example Step")
        >>> # do something here
        >>> debug_times.step_end("Example Step")
        """
        if not record:
            return
        storename = name + str(get_ident())
        self._steps[storename] = time()

    def step_end(self, name: str, record: bool = True) -> None:
        """ Stop the timer and record elapsed time for the given step name.

        Parameters
        ----------
        name: str
            The name of the step to end the timer for
        record: bool, optional
            ``True`` to record the step time, ``False`` to not record it.
            Used for when you have conditional code to time, but do not want to insert if/else
            statements in the code. Default: `True`

        Example
        -------
        >>> from lib.util import DebugTimes
        >>> debug_times = DebugTimes()
        >>> debug_times.step_start("Example Step")
        >>> # do something here
        >>> debug_times.step_end("Example Step")
        """
        if not record:
            return
        storename = name + str(get_ident())
        self._times.setdefault(name, []).append(time() - self._steps.pop(storename))

    @classmethod
    def _format_column(cls, text: str, width: int) -> str:
        """ Pad the given text to be aligned to the given width.

        Parameters
        ----------
        text: str
            The text to be formatted
        width: int
            The size of the column to insert the text into

        Returns
        -------
        str
            The text with the correct amount of padding applied
        """
        return f"{text}{' ' * (width - len(text))}"

    def summary(self, decimal_places: int = 6, interval: int = 1) -> None:
        """ Print a summary of step times.

        Parameters
        ----------
        decimal_places: int, optional
            The number of decimal places to display the summary elapsed times to. Default: 6
        interval: int, optional
            How many times summary must be called before printing to console. Default: 1

        Example
        -------
        >>> from lib.utils import DebugTimes
        >>> debug = DebugTimes()
        >>> debug.step_start("test")
        >>> time.sleep(0.5)
        >>> debug.step_end("test")
        >>> debug.summary()
        ----------------------------------
        Step             Count   Min
        ----------------------------------
        test             1       0.500000
        """
        interval = max(1, interval)
        if interval != self._interval:
            self._interval += 1
            return

        name_col = max(len(key) for key in self._times) + 4
        items_col = 8
        time_col = (decimal_places + 4) * sum(1 for v in self._display.values() if v)
        separator = "-" * (name_col + items_col + time_col)
        print("")
        print(separator)
        header = (f"{self._format_column('Step', name_col)}"
                  f"{self._format_column('Count', items_col)}")
        header += f"{self._format_column('Min', time_col)}" if self._display["min"] else ""
        header += f"{self._format_column('Avg', time_col)}" if self._display["mean"] else ""
        header += f"{self._format_column('Max', time_col)}" if self._display["max"] else ""
        print(header)
        print(separator)
        for key, val in self._times.items():
            num = str(len(val))
            contents = f"{self._format_column(key, name_col)}{self._format_column(num, items_col)}"
            if self._display["min"]:
                _min = f"{np.min(val):.{decimal_places}f}"
                contents += f"{self._format_column(_min, time_col)}"
            if self._display["mean"]:
                avg = f"{np.mean(val):.{decimal_places}f}"
                contents += f"{self._format_column(avg, time_col)}"
            if self._display["max"]:
                _max = f"{np.max(val):.{decimal_places}f}"
                contents += f"{self._format_column(_max, time_col)}"
            print(contents)
        self._interval = 1
