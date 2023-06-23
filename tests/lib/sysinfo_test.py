#!/usr/bin python3
""" Pytest unit tests for :mod:`lib.sysinfo` """

import locale
import os
import platform
import sys
import typing as T

from collections import namedtuple
from io import StringIO
from unittest.mock import MagicMock

import pytest
import pytest_mock

from lib.gpu_stats import GPUInfo
from lib.sysinfo import _Configs, _State, _SysInfo, CudaCheck, get_sysinfo

# pylint:disable=protected-access


# _SysInfo
@pytest.fixture(name="sys_info_instance")
def sys_info_fixture() -> _SysInfo:
    """ Single :class:~`lib.utils._SysInfo` object for tests

    Returns
    -------
    :class:`~lib.utils.sysinfo._SysInfo`
        The class instance for testing
    """
    return _SysInfo()


def test_init(sys_info_instance: _SysInfo) -> None:
    """ Test :class:`~lib.utils.sysinfo._SysInfo` __init__ and attributes

    Parameters
    ----------
    sys_info_instance: :class:`~lib.utils.sysinfo._SysInfo`
        The class instance to test
    """
    assert isinstance(sys_info_instance, _SysInfo)

    assert hasattr(sys_info_instance, "_state_file")
    assert isinstance(sys_info_instance._state_file, str)

    assert hasattr(sys_info_instance, "_configs")
    assert isinstance(sys_info_instance._configs, str)

    assert hasattr(sys_info_instance, "_system")
    assert isinstance(sys_info_instance._system, dict)
    assert sys_info_instance._system == {"platform": platform.platform(),
                                         "system": platform.system().lower(),
                                         "machine": platform.machine(),
                                         "release": platform.release(),
                                         "processor": platform.processor(),
                                         "cpu_count": os.cpu_count()}

    assert hasattr(sys_info_instance, "_python")
    assert isinstance(sys_info_instance._python, dict)
    assert sys_info_instance._python == {"implementation": platform.python_implementation(),
                                         "version": platform.python_version()}

    assert hasattr(sys_info_instance, "_gpu")
    assert isinstance(sys_info_instance._gpu, GPUInfo)

    assert hasattr(sys_info_instance, "_cuda_check")
    assert isinstance(sys_info_instance._cuda_check, CudaCheck)


def test_properties(sys_info_instance: _SysInfo) -> None:
    """ Test :class:`~lib.utils.sysinfo._SysInfo` properties

    Parameters
    ----------
    sys_info_instance: :class:`~lib.utils.sysinfo._SysInfo`
        The class instance to test
    """
    assert hasattr(sys_info_instance, "_encoding")
    assert isinstance(sys_info_instance._encoding, str)
    assert sys_info_instance._encoding == locale.getpreferredencoding()

    assert hasattr(sys_info_instance, "_is_conda")
    assert isinstance(sys_info_instance._is_conda, bool)
    assert sys_info_instance._is_conda == ("conda" in sys.version.lower() or
                                           os.path.exists(os.path.join(sys.prefix, "conda-meta")))

    assert hasattr(sys_info_instance, "_is_linux")
    assert isinstance(sys_info_instance._is_linux, bool)
    if platform.system().lower() == "linux":
        assert sys_info_instance._is_linux and sys_info_instance._system["system"] == "linux"
        assert not sys_info_instance._is_macos
        assert not sys_info_instance._is_windows

    assert hasattr(sys_info_instance, "_is_macos")
    assert isinstance(sys_info_instance._is_macos, bool)
    if platform.system().lower() == "darwin":
        assert sys_info_instance._is_macos and sys_info_instance._system["system"] == "darwin"
        assert not sys_info_instance._is_linux
        assert not sys_info_instance._is_windows

    assert hasattr(sys_info_instance, "_is_windows")
    assert isinstance(sys_info_instance._is_windows, bool)
    if platform.system().lower() == "windows":
        assert sys_info_instance._is_windows and sys_info_instance._system["system"] == "windows"
        assert not sys_info_instance._is_linux
        assert not sys_info_instance._is_macos

    assert hasattr(sys_info_instance, "_is_virtual_env")
    assert isinstance(sys_info_instance._is_virtual_env, bool)

    assert hasattr(sys_info_instance, "_ram_free")
    assert isinstance(sys_info_instance._ram_free, int)

    assert hasattr(sys_info_instance, "_ram_total")
    assert isinstance(sys_info_instance._ram_total, int)

    assert hasattr(sys_info_instance, "_ram_available")
    assert isinstance(sys_info_instance._ram_available, int)

    assert hasattr(sys_info_instance, "_ram_used")
    assert isinstance(sys_info_instance._ram_used, int)

    assert hasattr(sys_info_instance, "_fs_command")
    assert isinstance(sys_info_instance._fs_command, str)

    assert hasattr(sys_info_instance, "_installed_pip")
    assert isinstance(sys_info_instance._installed_pip, str)

    assert hasattr(sys_info_instance, "_installed_conda")
    assert isinstance(sys_info_instance._installed_conda, str)

    assert hasattr(sys_info_instance, "_conda_version")
    assert isinstance(sys_info_instance._conda_version, str)


def test_full_info(sys_info_instance: _SysInfo) -> None:
    """ Test the sys_info method of :class:`~lib.utils.sysinfo._SysInfo` returns as expected

    Parameters
    ----------
    sys_info_instance: :class:`~lib.utils.sysinfo._SysInfo`
        The class instance to test
    """
    assert hasattr(sys_info_instance, "full_info")
    sys_info = sys_info_instance.full_info()
    assert isinstance(sys_info, str)
    assert "backend:" in sys_info
    assert "os_platform:" in sys_info
    assert "os_machine:" in sys_info
    assert "os_release:" in sys_info
    assert "py_conda_version:" in sys_info
    assert "py_implementation:" in sys_info
    assert "py_version:" in sys_info
    assert "py_command:" in sys_info
    assert "py_virtual_env:" in sys_info
    assert "sys_cores:" in sys_info
    assert "sys_processor:" in sys_info
    assert "sys_ram:" in sys_info
    assert "encoding:" in sys_info
    assert "git_branch:" in sys_info
    assert "git_commits:" in sys_info
    assert "gpu_cuda:" in sys_info
    assert "gpu_cudnn:" in sys_info
    assert "gpu_driver:" in sys_info
    assert "gpu_devices:" in sys_info
    assert "gpu_vram:" in sys_info
    assert "gpu_devices_active:" in sys_info


def test__format_ram(sys_info_instance: _SysInfo, monkeypatch: pytest.MonkeyPatch) -> None:
    """ Test the _format_ram method of :class:`~lib.utils.sysinfo._SysInfo` returns as expected

    Parameters
    ----------
    sys_info_instance: :class:`~lib.utils.sysinfo._SysInfo`
        The class instance to test
    monkeypatch: :class:`pytest.MonkeyPatch`
        Monkey patching psutil.virtual_memory to be consistent
    """
    assert hasattr(sys_info_instance, "_format_ram")
    svmem = namedtuple("svmem", ["available", "free", "total", "used"])
    data = svmem(12345678, 1234567, 123456789, 123456)
    monkeypatch.setattr("psutil.virtual_memory", lambda *args, **kwargs: data)
    ram_info = sys_info_instance._format_ram()

    assert isinstance(ram_info, str)
    assert ram_info == "Total: 117MB, Available: 11MB, Used: 0MB, Free: 1MB"


# get_sys_info
def test_get_sys_info(mocker: pytest_mock.MockerFixture) -> None:
    """ Thest that the :func:`~lib.utils.sysinfo.get_sysinfo` function executes correctly

    Parameters
    ----------
    mocker: :class:`pytest_mock.MockerFixture`
        Mocker for checking full_info called from _SysInfo
    """
    sys_info = get_sysinfo()
    assert isinstance(sys_info, str)
    full_info = mocker.patch("lib.sysinfo._SysInfo.full_info")
    get_sysinfo()
    assert full_info.called


# _Configs
@pytest.fixture(name="configs_instance")
def configs_fixture():
    """ Pytest fixture for :class:`~lib.utils.sysinfo._Configs`

    Returns
    -------
    :class:`~lib.utils.sysinfo._Configs`
        The class instance for testing
    """
    return _Configs()


def test__configs__init__(configs_instance: _Configs) -> None:
    """ Test __init__ and attributes for :class:`~lib.utils.sysinfo._Configs`

    Parameters
    ----------
    configs_instance: :class:`~lib.utils.sysinfo._Configs`
        The class instance to test
    """
    assert hasattr(configs_instance, "config_dir")
    assert isinstance(configs_instance.config_dir, str)
    assert hasattr(configs_instance, "configs")
    assert isinstance(configs_instance.configs, str)


def test__configs__get_configs(configs_instance: _Configs) -> None:
    """ Test __init__ and attributes for :class:`~lib.utils.sysinfo._Configs`

    Parameters
    ----------
    configs_instance: :class:`~lib.utils.sysinfo._Configs`
        The class instance to test
    """
    assert hasattr(configs_instance, "_get_configs")
    assert isinstance(configs_instance._get_configs(), str)


def test__configs__parse_configs(configs_instance: _Configs,
                                 mocker: pytest_mock.MockerFixture) -> None:
    """ Test _parse_configs function for :class:`~lib.utils.sysinfo._Configs`

    Parameters
    ----------
    configs_instance: :class:`~lib.utils.sysinfo._Configs`
        The class instance to test
    mocker: :class:`pytest_mock.MockerFixture`
        Mocker for dummying in function calls
    """
    assert hasattr(configs_instance, "_parse_configs")
    assert isinstance(configs_instance._parse_configs([]), str)
    configs_instance._parse_ini = T.cast(MagicMock, mocker.MagicMock())  # type:ignore
    configs_instance._parse_json = T.cast(MagicMock,  mocker.MagicMock())  # type:ignore
    configs_instance._parse_configs(config_files=["test.ini", ".faceswap"])
    assert configs_instance._parse_ini.called
    assert configs_instance._parse_json.called


def test__configs__parse_ini(configs_instance: _Configs,
                             monkeypatch: pytest.MonkeyPatch) -> None:
    """ Test _parse_ini function for :class:`~lib.utils.sysinfo._Configs`

    Parameters
    ----------
    configs_instance: :class:`~lib.utils.sysinfo._Configs`
        The class instance to test
    monkeypatch: :class:`pytest.MonkeyPatch`
        Monkey patching :func:`builtins.open` to dummy in ini file
    """
    assert hasattr(configs_instance, "_parse_ini")

    file = ("[test.ini_header]\n"
            "# Test Header\n\n"
            "param = value")
    monkeypatch.setattr("builtins.open", lambda *args, **kwargs: StringIO(file))

    converted = configs_instance._parse_ini("test.ini")
    assert isinstance(converted, str)
    assert converted == ("\n[test.ini_header]\n"
                         "param:                    value\n")


def test__configs__parse_json(configs_instance: _Configs,
                              monkeypatch: pytest.MonkeyPatch) -> None:
    """ Test _parse_json function for :class:`~lib.utils.sysinfo._Configs`

    Parameters
    ----------
    configs_instance: :class:`~lib.utils.sysinfo._Configs`
        The class instance to test
    monkeypatch: :class:`pytest.MonkeyPatch`
        Monkey patching :func:`builtins.open` to dummy in json file

    """
    assert hasattr(configs_instance, "_parse_json")
    file = '{"test": "param"}'
    monkeypatch.setattr("builtins.open", lambda *args, **kwargs: StringIO(file))

    converted = configs_instance._parse_json(".file")
    assert isinstance(converted, str)
    assert converted == ("test:                     param\n")


def test__configs__format_text(configs_instance: _Configs) -> None:
    """ Test _format_text function for :class:`~lib.utils.sysinfo._Configs`

    Parameters
    ----------
    configs_instance: :class:`~lib.utils.sysinfo._Configs`
        The class instance to test
    """
    assert hasattr(configs_instance, "_format_text")
    key, val = "  test_key ", "test_val "
    formatted = configs_instance._format_text(key, val)
    assert isinstance(formatted, str)
    assert formatted == "test_key:                 test_val\n"


# _State
@pytest.fixture(name="state_instance")
def state_fixture():
    """ Pytest fixture for :class:`~lib.utils.sysinfo._State`

    Returns
    -------
    :class:`~lib.utils.sysinfo._State`
        The class instance for testing
    """
    return _State()


def test__state__init__(state_instance: _State) -> None:
    """ Test __init__ and attributes for :class:`~lib.utils.sysinfo._State`

    Parameters
    ----------
    state_instance: :class:`~lib.utils.sysinfo._State`
        The class instance to test
    """
    assert hasattr(state_instance, '_model_dir')
    assert state_instance._model_dir is None
    assert hasattr(state_instance, '_trainer')
    assert state_instance._trainer is None
    assert hasattr(state_instance, 'state_file')
    assert isinstance(state_instance.state_file, str)


def test__state__is_training(state_instance: _State,
                             monkeypatch: pytest.MonkeyPatch) -> None:
    """ Test _is_training function for :class:`~lib.utils.sysinfo._State`

    Parameters
    ----------
    state_instance: :class:`~lib.utils.sysinfo._State`
        The class instance to test
    monkeypatch: :class:`pytest.MonkeyPatch`
        Monkey patching :func:`sys.argv` to dummy in commandline args

    """
    assert hasattr(state_instance, '_is_training')
    assert isinstance(state_instance._is_training, bool)
    assert not state_instance._is_training
    monkeypatch.setattr("sys.argv", ["faceswap.py", "train"])
    assert state_instance._is_training
    monkeypatch.setattr("sys.argv", ["faceswap.py", "extract"])
    assert not state_instance._is_training


def test__state__get_arg(state_instance: _State,
                         monkeypatch: pytest.MonkeyPatch) -> None:
    """ Test _get_arg function for :class:`~lib.utils.sysinfo._State`

    Parameters
    ----------
    state_instance: :class:`~lib.utils.sysinfo._State`
        The class instance to test
    monkeypatch: :class:`pytest.MonkeyPatch`
        Monkey patching :func:`sys.argv` to dummy in commandline args
        :func:`builtins.input`
    """
    assert hasattr(state_instance, '_get_arg')
    assert state_instance._get_arg("-t", "--test_arg") is None
    monkeypatch.setattr("sys.argv", ["test", "command", "-t", "test_option"])
    assert state_instance._get_arg("-t", "--test_arg") == "test_option"


def test__state__get_state_file(state_instance: _State,
                                mocker: pytest_mock.MockerFixture,
                                monkeypatch: pytest.MonkeyPatch) -> None:
    """ Test _get_state_file function for :class:`~lib.utils.sysinfo._State`

    Parameters
    ----------
    state_instance: :class:`~lib.utils.sysinfo._State`
        The class instance to test
    mocker: :class:`pytest_mock.MockerFixture`
        Mocker for dummying in function calls
    monkeypatch: :class:`pytest.MonkeyPatch`
        Monkey patching :func:`sys.argv` to dummy in commandline args
        :func:`builtins.input`
`   """
    assert hasattr(state_instance, '_get_state_file')
    assert isinstance(state_instance._get_state_file(), str)

    mock_is_training = mocker.patch("lib.sysinfo._State._is_training")

    # Not training or missing training arguments
    mock_is_training.return_value = False
    assert state_instance._get_state_file() == ""
    mock_is_training.return_value = False

    monkeypatch.setattr(state_instance, "_model_dir", None)
    assert state_instance._get_state_file() == ""
    monkeypatch.setattr(state_instance, "_model_dir", "test_dir")

    monkeypatch.setattr(state_instance, "_trainer", None)
    assert state_instance._get_state_file() == ""
    monkeypatch.setattr(state_instance, "_trainer", "test_trainer")

    # Training but file not found
    assert state_instance._get_state_file() == ""

    # State file is just a json dump
    file = ('{\n'
            '   "test": "json",\n'
            '}')
    monkeypatch.setattr("os.path.isfile", lambda *args, **kwargs: True)
    monkeypatch.setattr("builtins.open", lambda *args, **kwargs: StringIO(file))
    assert state_instance._get_state_file().endswith(file)
