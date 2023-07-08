#!/usr/bin python3
""" Pytest unit tests for :mod:`lib.utils` """
import os
import platform
import time
import typing as T
import warnings
import zipfile

from io import StringIO
from socket import timeout as socket_timeout, error as socket_error
from shutil import rmtree
from unittest.mock import MagicMock
from urllib import error as urlliberror

import pytest
import pytest_mock

from lib import utils
from lib.utils import (
    _Backend, camel_case_split, convert_to_secs, DebugTimes, deprecation_warning, FaceswapError,
    full_path_split, get_backend, get_dpi, get_folder, get_image_paths, get_tf_version, GetModel,
    safe_shutdown, set_backend, set_system_verbosity)

from lib.logger import log_setup
# Need to setup logging to avoid trace/verbose errors
log_setup("DEBUG", "pytest_utils.log", "PyTest, False")


# pylint:disable=protected-access


# Backend tests
def test_set_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """ Test the :func:`~lib.utils.set_backend` function

    Parameters
    ----------
    monkeypatch: :class:`pytest.MonkeyPatch`
        Monkey patching _FS_BACKEND
    """
    monkeypatch.setattr(utils, "_FS_BACKEND", "cpu")  # _FS_BACKEND already defined
    set_backend("directml")
    assert utils._FS_BACKEND == "directml"
    monkeypatch.delattr(utils, "_FS_BACKEND")  # _FS_BACKEND is not already defined
    set_backend("rocm")
    assert utils._FS_BACKEND == "rocm"


def test_get_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """ Test the :func:`~lib.utils.get_backend` function

    Parameters
    ----------
    monkeypatch: :class:`pytest.MonkeyPatch`
        Monkey patching _FS_BACKEND
    """
    monkeypatch.setattr(utils, "_FS_BACKEND", "apple-silicon")
    assert get_backend() == "apple-silicon"


def test__backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """ Test the :class:`~lib.utils._Backend` class

    Parameters
    ----------
    monkeypatch: :class:`pytest.MonkeyPatch`
        Monkey patching :func:`os.environ`, :func:`os.path.isfile`, :func:`builtins.open` and
        :func:`builtins.input`
    """
    monkeypatch.setattr("os.environ", {"FACESWAP_BACKEND": "nvidia"})  # Environment variable set
    backend = _Backend()
    assert backend.backend == "nvidia"

    monkeypatch.setattr("os.environ", {})  # Environment variable not set, dummy in config file
    monkeypatch.setattr("os.path.isfile", lambda x: True)
    monkeypatch.setattr("builtins.open", lambda *args, **kwargs: StringIO('{"backend": "cpu"}'))
    backend = _Backend()
    assert backend.backend == "cpu"

    monkeypatch.setattr("os.path.isfile", lambda x: False)  # no config file, dummy in user input
    monkeypatch.setattr("builtins.input", lambda x: "3")
    backend = _Backend()
    assert backend._configure_backend() == "nvidia"


# Folder and path utils
def test_get_folder(tmp_path: str) -> None:
    """ Unit test for :func:`~lib.utils.get_folder`

    Parameters
    ----------
    tmp_path: str
        pytest temporary path to generate folders
    """
    # New folder
    path = os.path.join(tmp_path, "test_new_folder")
    expected_output = path
    assert not os.path.isdir(path)
    assert get_folder(path) == expected_output
    assert os.path.isdir(path)

    # Test not creating a new folder when it already exists
    path = os.path.join(tmp_path, "test_new_folder")
    expected_output = path
    assert os.path.isdir(path)
    stats = os.stat(path)
    assert get_folder(path) == expected_output
    assert os.path.isdir(path)
    assert stats == os.stat(path)

    # Test not creating a new folder when make_folder is False
    path = os.path.join(tmp_path, "test_no_folder")
    expected_output = ""
    assert get_folder(path, make_folder=False) == expected_output
    assert not os.path.isdir(path)


def test_get_image_paths(tmp_path: str) -> None:
    """ Unit test for :func:`~lib.utils.test_get_image_paths`

    Parameters
    ----------
    tmp_path: str
        pytest temporary path to generate folders
    """
    # Test getting image paths from a folder with no images
    test_folder = os.path.join(tmp_path, "test_image_folder")
    os.makedirs(test_folder)
    assert not get_image_paths(test_folder)

    # Populate 2 different image files and 1 text file
    test_jpg_path = os.path.join(test_folder, "test_image.jpg")
    test_png_path = os.path.join(test_folder, "test_image.png")
    test_txt_path = os.path.join(test_folder, "test_file.txt")
    for fname in (test_jpg_path, test_png_path, test_txt_path):
        with open(fname, "a", encoding="utf-8"):
            pass

    # Test getting any image paths from a folder with images and random files
    exists = [os.path.join(test_folder, img)
              for img in os.listdir(test_folder) if os.path.splitext(img)[-1] != ".txt"]
    assert sorted(get_image_paths(test_folder)) == sorted(exists)

    # Test getting image paths from a folder with images with a specific extension
    exists = [os.path.join(test_folder, img)
              for img in os.listdir(test_folder) if os.path.splitext(img)[-1] == ".png"]
    assert sorted(get_image_paths(test_folder, extension=".png")) == sorted(exists)


_PARAMS = [("/path/to/file.txt", ["/", "path", "to", "file.txt"]),  # Absolute
           ("/path/to/directory/", ["/", "path", "to", "directory"]),
           ("/path/to/directory", ["/", "path", "to", "directory"]),
           ("path/to/file.txt", ["path", "to", "file.txt"]),  # Relative
           ("path/to/directory/", ["path", "to", "directory"]),
           ("path/to/directory", ["path", "to", "directory"]),
           ("", []),  # Edge cases
           ("/", ["/"]),
           (".", ["."]),
           ("..", [".."])]


@pytest.mark.parametrize("path,result", _PARAMS, ids=[f'"{p[0]}"' for p in _PARAMS])
def test_full_path_split(path: str, result: list[str]) -> None:
    """ Test the :func:`~lib.utils.full_path_split` function works correctly

    Parameters
    ----------
    path: str
        The path to test
    result: list
        The expected result from the path
    """
    split = full_path_split(path)
    assert isinstance(split, list)
    assert split == result


_PARAMS = [("camelCase", ["camel", "Case"]),
           ("camelCaseTest", ["camel", "Case", "Test"]),
           ("camelCaseTestCase", ["camel", "Case", "Test", "Case"]),
           ("CamelCase", ["Camel", "Case"]),
           ("CamelCaseTest", ["Camel", "Case", "Test"]),
           ("CamelCaseTestCase", ["Camel", "Case", "Test", "Case"]),
           ("CAmelCASETestCase", ["C", "Amel", "CASE", "Test", "Case"]),
           ("camelcasetestcase", ["camelcasetestcase"]),
           ("CAMELCASETESTCASE", ["CAMELCASETESTCASE"]),
           ("", [])]


@pytest.mark.parametrize("text, result", _PARAMS, ids=[f'"{p[0]}"' for p in _PARAMS])
def test_camel_case_split(text: str, result: list[str]) -> None:
    """ Test the :func:`~lib.utils.camel_case_spli` function works correctly

    Parameters
    ----------
    text: str
        The camel case text to test
    result: list
        The expected result from the path
    """
    split = camel_case_split(text)
    assert isinstance(split, list)
    assert split == result


# General utils
def test_get_tf_version() -> None:
    """ Test the :func:`~lib.utils.get_tf_version` function version returns correctly in range """
    tf_version = get_tf_version()
    assert (2, 10) <= tf_version < (2, 11)


def test_get_dpi() -> None:
    """ Test the :func:`~lib.utils.get_dpi` function version returns correctly in a sane
    range """
    dpi = get_dpi()
    assert isinstance(dpi, float) or dpi is None
    if dpi is None:  # No display detected
        return
    assert dpi > 0
    assert dpi < 600.0


_SECPARAMS = [((1, ), 1),  # 1 argument
              ((10, ), 10),
              ((0, 1), 1),
              ((0, 60), 60),  # 2 arguments
              ((1, 0), 60),
              ((1, 1), 61),
              ((0, 0, 1), 1),
              ((0, 0, 60), 60),  # 3 arguments
              ((0, 1, 0), 60),
              ((1, 0, 0), 3600),
              ((1, 1, 1), 3661)]


@pytest.mark.parametrize("args,result", _SECPARAMS, ids=[str(p[0]) for p in _SECPARAMS])
def test_convert_to_secs(args: tuple[int, ...], result: int) -> None:
    """ Test the :func:`~lib.utils.convert_to_secs` function works correctly

    Parameters
    ----------
    args: tuple
        Tuple of 1, 2 or 3 integers to pass to the function
    result: int
        The expected results for the args tuple
    """
    secs = convert_to_secs(*args)
    assert isinstance(secs, int)
    assert secs == result


@pytest.mark.parametrize("log_level", ["DEBUG", "INFO", "WARNING", "ERROR"])
def test_set_system_verbosity(log_level: str) -> None:
    """ Test the :func:`~lib.utils.set_system_verbosity` function works correctly

    Parameters
    ----------
    log_level: str
        The logging loglevel in upper text format
    """
    # Set TF Env Variable
    tf_set_level = "0" if log_level == "DEBUG" else "3"
    set_system_verbosity(log_level)
    tf_get_level = os.environ["TF_CPP_MIN_LOG_LEVEL"]
    assert tf_get_level == tf_set_level
    warn_filters = [filt for filt in warnings.filters
                    if filt[0] == "ignore"
                    and filt[2] in (FutureWarning, DeprecationWarning, UserWarning)]
    # Python Warnings
    # DeprecationWarning is already ignored by default, so there should be 1 warning for debug
    # warning. 3 for the rest
    num_warnings = 1 if log_level == "DEBUG" else 3
    warn_count = len(warn_filters)
    assert warn_count == num_warnings


@pytest.mark.parametrize("additional_info", [None, "additional information"])
def test_deprecation_warning(caplog: pytest.LogCaptureFixture, additional_info: str) -> None:
    """ Test the :func:`~lib.utils.deprecation_warning` function works correctly

    Parameters
    ----------
    caplog: :class:`pytest.LogCaptureFixture`
        Pytest's log capturing fixture
    additional_info: str
        Additional information to pass to the warning function
    """
    func_name = "function_name"
    test = f"{func_name} has been deprecated and will be removed from a future update."
    if additional_info:
        test = f"{test} {additional_info}"
    deprecation_warning(func_name, additional_info=additional_info)
    assert test in caplog.text


@pytest.mark.parametrize("got_error", [True, False])
def test_safe_shutdown(caplog: pytest.LogCaptureFixture, got_error: bool) -> None:
    """ Test the :func:`~lib.utils.safe_shutdown` function works correctly

    Parameters
    ----------
    caplog: :class:`pytest.LogCaptureFixture`
        Pytest's log capturing fixture
    got_error: bool
        The got_error parameter to pass to safe_shutdown
    """
    caplog.set_level("DEBUG")
    with pytest.raises(SystemExit) as wrapped_exit:
        safe_shutdown(got_error=got_error)

    exit_value = 1 if got_error else 0
    assert wrapped_exit.typename == "SystemExit"
    assert wrapped_exit.value.code == exit_value
    assert "Safely shutting down" in caplog.messages
    assert "Cleanup complete. Shutting down queue manager and exiting" in caplog.messages


def test_faceswap_error():
    """ Test the :class:`~lib.utils.FaceswapError` raises correctly """
    with pytest.raises(Exception):
        raise FaceswapError


# GetModel class
@pytest.fixture(name="get_model_instance")
def fixture_get_model_instance(monkeypatch: pytest.MonkeyPatch,
                               tmp_path: pytest.TempdirFactory,
                               request: pytest.FixtureRequest) -> GetModel:
    """ Create a fixture of the :class:`~lib.utils.GetModel` object, prevent _get() from running at
    __init__ and point the cache_dir at our local test folder """
    cache_dir = os.path.join(str(tmp_path), "get_model")
    os.mkdir(cache_dir)

    model_filename = "test_model_file_v1.h5"
    git_model_id = 123

    original_get = GetModel._get
    # Patch out _get() so it is not called from __init__()
    monkeypatch.setattr(utils.GetModel, "_get", lambda x: None)
    model_instance = GetModel(model_filename, git_model_id)
    # Reinsert _get() so we can test it
    monkeypatch.setattr(model_instance, "_get", original_get)
    model_instance._cache_dir = cache_dir

    def teardown():
        rmtree(cache_dir)

    request.addfinalizer(teardown)
    return model_instance


_INPUT = ("test_model_file_v3.h5",
          ["test_multi_model_file_v1.1.npy", "test_multi_model_file_v1.2.npy"])
_EXPECTED = ((["test_model_file_v3.h5"], "test_model_file_v3", "test_model_file", 3),
             (["test_multi_model_file_v1.1.npy", "test_multi_model_file_v1.2.npy"],
              "test_multi_model_file_v1", "test_multi_model_file", 1))


@pytest.mark.parametrize("filename,results", zip(_INPUT, _EXPECTED), ids=[str(i) for i in _INPUT])
def test_get_model_model_filename_input(
        get_model_instance: GetModel,  # pylint:disable=unused-argument
        filename: str | list[str],
        results: str | list[str]) -> None:
    """ Test :class:`~lib.utils.GetModel` filename parsing works

    Parameters
    ---------
    get_model_instance: `~lib.utils.GetModel`
        The patched instance of the class
    filename: list or str
        The test filenames
    results: tuple
        The expected results for :attr:`_model_filename`,  :attr:`_model_full_name`,
         :attr:`_model_name`,  :attr:`_model_version` respectively
    """
    model = GetModel(filename, 123)
    assert model._model_filename == results[0]
    assert model._model_full_name == results[1]
    assert model._model_name == results[2]
    assert model._model_version == results[3]


def test_get_model_attributes(get_model_instance: GetModel) -> None:
    """ Test :class:`~lib.utils.GetModel` private attributes set correctly

    Parameters
    ---------
    get_model_instance: `~lib.utils.GetModel`
        The patched instance of the class
    """
    model = get_model_instance
    assert model._git_model_id == 123
    assert model._url_base == ("https://github.com/deepfakes-models/faceswap-models"
                               "/releases/download")
    assert model._chunk_size == 1024
    assert model._retries == 6


def test_get_model_properties(get_model_instance: GetModel) -> None:
    """ Test :class:`~lib.utils.GetModel` calculated attributes return correctly

    Parameters
    ---------
    get_model_instance: `~lib.utils.GetModel`
        The patched instance of the class
    """
    model = get_model_instance
    assert model.model_path == os.path.join(model._cache_dir, "test_model_file_v1.h5")
    assert model._model_zip_path == os.path.join(model._cache_dir, "test_model_file_v1.zip")
    assert not model._model_exists
    assert model._url_download == ("https://github.com/deepfakes-models/faceswap-models/releases/"
                                   "download/v123.1/test_model_file_v1.zip")
    assert model._url_partial_size == 0


@pytest.mark.parametrize("model_exists", (True, False))
def test_get_model__get(mocker: pytest_mock.MockerFixture,
                        get_model_instance: GetModel,
                        model_exists: bool) -> None:
    """ Test :func:`~lib.utils.GetModel._get` executes logic correctly

    Parameters
    ---------
    mocker: :class:`pytest_mock.MockerFixture`
        Mocker for dummying in function calls
    get_model_instance: `~lib.utils.GetModel`
        The patched instance of the class
    model_exists: bool
        For testing the function when a model exists and when it does not
    """
    model = get_model_instance
    model._download_model = T.cast(MagicMock, mocker.MagicMock())  # type:ignore
    model._unzip_model = T.cast(MagicMock, mocker.MagicMock())  # type:ignore
    os_remove = mocker.patch("os.remove")

    if model_exists:  # Dummy in a model file
        assert isinstance(model.model_path, str)
        with open(model.model_path, "a", encoding="utf-8"):
            pass

    model._get(model)  # type:ignore

    assert (model_exists and not model._download_model.called) or (
        not model_exists and model._download_model.called)
    assert (model_exists and not model._unzip_model.called) or (
        not model_exists and model._unzip_model.called)
    assert model_exists or not (model_exists and os_remove.called)
    os_remove.reset_mock()


_DLPARAMS = [(None, None),
             (socket_error, ()),
             (socket_timeout, ()),
             (urlliberror.URLError, ("test_reason", )),
             (urlliberror.HTTPError, ("test_uri", 400, "", "", 0))]


@pytest.mark.parametrize("error_type,error_args", _DLPARAMS, ids=[str(p[0]) for p in _DLPARAMS])
def test_get_model__download_model(mocker: pytest_mock.MockerFixture,
                                   get_model_instance: GetModel,
                                   error_type: T.Any,
                                   error_args: tuple[str | int, ...]) -> None:
    """ Test :func:`~lib.utils.GetModel._download_model` executes its logic correctly

    Parameters
    ---------
    mocker: :class:`pytest_mock.MockerFixture`
        Mocker for dummying in function calls
    get_model_instance: `~lib.utils.GetModel`
        The patched instance of the class
    error_type: connection error type or ``None``
        Connection error type to mock, or ``None`` for succesful download
    error_args: tuple
        The arguments to be passed to the exception to be raised
    """
    mock_urlopen = mocker.patch("urllib.request.urlopen")
    if not error_type:  # Model download is successful
        get_model_instance._write_zipfile = T.cast(MagicMock, mocker.MagicMock())  # type:ignore
        get_model_instance._download_model()
        assert mock_urlopen.called
        assert get_model_instance._write_zipfile.called
    else:  # Test that the process exits on download errors
        mock_urlopen.side_effect = error_type(*error_args)
        with pytest.raises(SystemExit):
            get_model_instance._download_model()
    mock_urlopen.reset_mock()


@pytest.mark.parametrize("dl_type", ["complete", "new", "continue"])
def test_get_model__write_zipfile(mocker: pytest_mock.MockerFixture,
                                  get_model_instance: GetModel,
                                  dl_type: str) -> None:
    """ Test :func:`~lib.utils.GetModel._write_zipfile` executes its logic correctly

    Parameters
    ---------
    mocker: :class:`pytest_mock.MockerFixture`
        Mocker for dummying in function calls
    get_model_instance: `~lib.utils.GetModel`
        The patched instance of the class
    dl_type: str
        The type of read to attemp
    """
    response = mocker.MagicMock()
    assert not os.path.isfile(get_model_instance._model_zip_path)

    downloaded = 10 if dl_type == "complete" else 0
    response.getheader.return_value = 0

    if dl_type in ("new", "continue"):
        chunks = [32, 64, 128, 256, 512, 1024]
        data = [b"\x00" * size for size in chunks] + [b""]
        response.getheader.return_value = sum(chunks)
        response.read.side_effect = data

    if dl_type == "continue":  # Write a partial download of the correct size
        with open(get_model_instance._model_zip_path, "wb") as partial:
            partial.write(b"\x00" * sum(chunks))
        downloaded = os.path.getsize(get_model_instance._model_zip_path)

    get_model_instance._write_zipfile(response, downloaded)

    if dl_type == "complete":  # Already downloaded. No more tests
        assert not response.read.called
        return

    assert response.read.call_count == len(data)  # all data read
    assert os.path.isfile(get_model_instance._model_zip_path)
    downloaded_size = os.path.getsize(get_model_instance._model_zip_path)
    downloaded_size = downloaded_size if dl_type == "new" else downloaded_size // 2
    assert downloaded_size == sum(chunks)


def test_get_model__unzip_model(mocker: pytest_mock.MockerFixture,
                                get_model_instance: GetModel) -> None:
    """ Test :func:`~lib.utils.GetModel._unzip_model` executes its logic correctly

    Parameters
    ---------
    mocker: :class:`pytest_mock.MockerFixture`
        Mocker for dummying in function calls
    get_model_instance: `~lib.utils.GetModel`
        The patched instance of the class
    """
    mock_zipfile = mocker.patch("zipfile.ZipFile")
    # Successful
    get_model_instance._unzip_model()
    assert mock_zipfile.called
    mock_zipfile.reset_mock()
    # Error
    mock_zipfile.side_effect = zipfile.BadZipFile()
    with pytest.raises(SystemExit):
        get_model_instance._unzip_model()
    mock_zipfile.reset_mock()


def test_get_model__write_model(mocker: pytest_mock.MockerFixture,
                                get_model_instance: GetModel) -> None:
    """ Test :func:`~lib.utils.GetModel._write_model` executes its logic correctly

    Parameters
    ---------
    mocker: :class:`pytest_mock.MockerFixture`
        Mocker for dummying in function calls
    get_model_instance: `~lib.utils.GetModel`
        The patched instance of the class
    """
    out_file = os.path.join(get_model_instance._cache_dir, get_model_instance._model_filename[0])
    chunks = [8, 16, 32, 64, 128, 256, 512, 1024]
    data = [b"\x00" * size for size in chunks] + [b""]
    assert not os.path.isfile(out_file)
    mock_zipfile = mocker.patch("zipfile.ZipFile")
    mock_zipfile.namelist.return_value = get_model_instance._model_filename
    mock_zipfile.open.return_value = mock_zipfile
    mock_zipfile.read.side_effect = data
    get_model_instance._write_model(mock_zipfile)
    assert mock_zipfile.read.call_count == len(data)
    assert os.path.isfile(out_file)
    assert os.path.getsize(out_file) == sum(chunks)


# DebugTimes class
def test_debug_times():
    """ Test :class:`~lib.utils.DebugTimes` executes its logic correctly  """
    debug_times = DebugTimes()

    debug_times.step_start("Test1")
    time.sleep(0.1)
    debug_times.step_end("Test1")

    debug_times.step_start("Test2")
    time.sleep(0.2)
    debug_times.step_end("Test2")

    debug_times.step_start("Test1")
    time.sleep(0.1)
    debug_times.step_end("Test1")

    debug_times.summary()

    # Ensure that the summary method prints the min, mean, and max times for each step
    assert debug_times._display["min"] is True
    assert debug_times._display["mean"] is True
    assert debug_times._display["max"] is True

    # Ensure that the summary method includes the correct number of items for each step
    assert len(debug_times._times["Test1"]) == 2
    assert len(debug_times._times["Test2"]) == 1

    # Ensure that the summary method includes the correct min, mean, and max times for each step
    # Github workflow for macos-latest can swing out a fair way
    threshold = 2e-1 if platform.system() == "Darwin" else 1e-1
    assert min(debug_times._times["Test1"]) == pytest.approx(0.1, abs=threshold)
    assert min(debug_times._times["Test2"]) == pytest.approx(0.2, abs=threshold)
    assert max(debug_times._times["Test1"]) == pytest.approx(0.1, abs=threshold)
    assert max(debug_times._times["Test2"]) == pytest.approx(0.2, abs=threshold)
    assert (sum(debug_times._times["Test1"]) /
            len(debug_times._times["Test1"])) == pytest.approx(0.1, abs=threshold)
    assert (sum(debug_times._times["Test2"]) /
            len(debug_times._times["Test2"]) == pytest.approx(0.2, abs=threshold))
