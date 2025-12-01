#!/usr/bin python3
""" Pytest unit tests for :mod:`lib.system.sysinfo` """

import platform
import typing as T

from collections import namedtuple
from io import StringIO
from unittest.mock import MagicMock

import pytest
import pytest_mock

# pylint:disable=import-error
from lib.gpu_stats import GPUInfo
from lib.system.sysinfo import _Configs, _State, _SysInfo, get_sysinfo
from lib.system import Cuda, Packages, ROCm, System

# pylint:disable=protected-access


# _SysInfo
@pytest.fixture(name="sys_info_instance")
def sys_info_fixture() -> _SysInfo:
    """ Single :class:`~lib.system.sysinfo._SysInfo` object for tests """
    return _SysInfo()


def test_init(sys_info_instance: _SysInfo) -> None:
    """ Test :class:`lib.system.sysinfo._SysInfo` __init__ and attributes """
    assert isinstance(sys_info_instance, _SysInfo)

    attrs = ["_state_file", "_configs", "_system",
             "_python", "_packages", "_gpu", "_cuda", "_rocm"]
    assert all(a in sys_info_instance.__dict__ for a in attrs)
    assert all(a in attrs for a in sys_info_instance.__dict__)

    assert isinstance(sys_info_instance._state_file, str)
    assert isinstance(sys_info_instance._configs, str)
    assert isinstance(sys_info_instance._system, System)
    assert isinstance(sys_info_instance._python, dict)
    assert sys_info_instance._python == {"implementation": platform.python_implementation(),
                                         "version": platform.python_version()}
    assert isinstance(sys_info_instance._packages, Packages)
    assert isinstance(sys_info_instance._gpu, GPUInfo)
    assert isinstance(sys_info_instance._cuda, Cuda)
    assert isinstance(sys_info_instance._rocm, ROCm)


def test_properties(sys_info_instance: _SysInfo) -> None:
    """ Test :class:`lib.system.sysinfo._SysInfo` properties """
    ints = ["_ram_free",  "_ram_total", "_ram_available", "_ram_used"]
    strs = ["_fs_command", "_conda_version", "_git_commits", "_cuda_versions",
            "_cuda_version", "_cudnn_versions", "_rocm_version", "_rocm_versions"]

    for prop in ints:
        assert hasattr(sys_info_instance, prop), f"sysinfo missing property '{prop}'"
        assert isinstance(getattr(sys_info_instance, prop),
                          int), f"sysinfo property '{prop}' not int"

    for prop in strs:
        assert hasattr(sys_info_instance, prop), f"sysinfo missing property '{prop}'"
        assert isinstance(getattr(sys_info_instance, prop),
                          str), f"sysinfo property '{prop}' not str"


def test_get_gpu_info(sys_info_instance: _SysInfo) -> None:
    """ Test _get_gpu_info method of :class:`lib.system.sysinfo._SysInfo` returns as expected """
    assert hasattr(sys_info_instance, "_get_gpu_info")
    gpu_info = sys_info_instance._get_gpu_info()
    assert isinstance(gpu_info, GPUInfo)


def test__format_ram(sys_info_instance: _SysInfo, monkeypatch: pytest.MonkeyPatch) -> None:
    """ Test the _format_ram method of :class:`lib.system.sysinfo._SysInfo` """
    assert hasattr(sys_info_instance, "_format_ram")
    svmem = namedtuple("svmem", ["available", "free", "total", "used"])
    data = svmem(12345678, 1234567, 123456789, 123456)
    monkeypatch.setattr("psutil.virtual_memory", lambda *args, **kwargs: data)
    ram_info = sys_info_instance._format_ram()

    assert isinstance(ram_info, str)
    assert ram_info == "Total: 117MB, Available: 11MB, Used: 0MB, Free: 1MB"


def test_full_info(sys_info_instance: _SysInfo) -> None:
    """ Test the full_info method of :class:`lib.system.sysinfo._SysInfo` returns as expected """
    assert hasattr(sys_info_instance, "full_info")
    sys_info = sys_info_instance.full_info()
    assert isinstance(sys_info, str)

    sections = ["System Information", "Pip Packages", "Configs"]
    for section in sections:
        assert section in sys_info, f"Section {section} not in full_info"
    if sys_info_instance._system.is_conda:
        assert "Conda Packages" in sys_info
    else:
        assert "Conda Packages" not in sys_info

    keys = ["backend", "os_platform", "os_machine", "os_release", "py_conda_version",
            "py_implementation", "py_version", "py_command", "py_virtual_env", "sys_cores",
            "sys_processor", "sys_ram", "encoding", "git_branch", "git_commits",
            "gpu_cuda_versions", "gpu_cuda", "gpu_cudnn", "gpu_rocm_versions", "gpu_rocm_version",
            "gpu_driver", "gpu_devices", "gpu_vram", "gpu_devices_active"]
    for key in keys:
        assert f"{key}:" in sys_info, f"'{key}:' not in full_info"


# get_sys_info
def test_get_sys_info(mocker: pytest_mock.MockerFixture) -> None:
    """ Thest that the :func:`~lib.utils.sysinfo.get_sysinfo` function executes correctly """
    sys_info = get_sysinfo()
    assert isinstance(sys_info, str)
    full_info = mocker.patch("lib.system.sysinfo._SysInfo.full_info")
    get_sysinfo()
    assert full_info.called


# _Configs
@pytest.fixture(name="configs_instance")
def configs_fixture():
    """ Pytest fixture for :class:`~lib.utils.sysinfo._Configs` """
    return _Configs()


def test__configs__init__(configs_instance: _Configs) -> None:
    """ Test __init__ and attributes for :class:`~lib.utils.sysinfo._Configs` """
    assert hasattr(configs_instance, "config_dir")
    assert isinstance(configs_instance.config_dir, str)
    assert hasattr(configs_instance, "configs")
    assert isinstance(configs_instance.configs, str)


def test__configs__get_configs(configs_instance: _Configs) -> None:
    """ Test __init__ and attributes for :class:`~lib.utils.sysinfo._Configs` """
    assert hasattr(configs_instance, "_get_configs")
    assert isinstance(configs_instance._get_configs(), str)


def test__configs__parse_configs(configs_instance: _Configs,
                                 mocker: pytest_mock.MockerFixture) -> None:
    """ Test _parse_configs function for :class:`~lib.utils.sysinfo._Configs` """
    assert hasattr(configs_instance, "_parse_configs")
    assert isinstance(configs_instance._parse_configs([]), str)
    configs_instance._parse_ini = T.cast(MagicMock, mocker.MagicMock())  # type:ignore
    configs_instance._parse_json = T.cast(MagicMock,  mocker.MagicMock())  # type:ignore
    configs_instance._parse_configs(config_files=["test.ini", ".faceswap"])
    assert configs_instance._parse_ini.called
    assert configs_instance._parse_json.called


def test__configs__parse_ini(configs_instance: _Configs,
                             monkeypatch: pytest.MonkeyPatch) -> None:
    """ Test _parse_ini function for :class:`~lib.utils.sysinfo._Configs` """
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
    """ Test _parse_json function for :class:`~lib.utils.sysinfo._Configs` """
    assert hasattr(configs_instance, "_parse_json")
    file = '{"test": "param"}'
    monkeypatch.setattr("builtins.open", lambda *args, **kwargs: StringIO(file))

    converted = configs_instance._parse_json(".file")
    assert isinstance(converted, str)
    assert converted == ("test:                     param\n")


def test__configs__format_text(configs_instance: _Configs) -> None:
    """ Test _format_text function for :class:`~lib.utils.sysinfo._Configs` """
    assert hasattr(configs_instance, "_format_text")
    key, val = "  test_key ", "test_val "
    formatted = configs_instance._format_text(key, val)
    assert isinstance(formatted, str)
    assert formatted == "test_key:                 test_val\n"


# _State
@pytest.fixture(name="state_instance")
def state_fixture():
    """ Pytest fixture for :class:`~lib.utils.sysinfo._State` """
    return _State()


def test__state__init__(state_instance: _State) -> None:
    """ Test __init__ and attributes for :class:`~lib.utils.sysinfo._State` """
    assert hasattr(state_instance, "_model_dir")
    assert state_instance._model_dir is None
    assert hasattr(state_instance, "_trainer")
    assert state_instance._trainer is None
    assert hasattr(state_instance, "state_file")
    assert isinstance(state_instance.state_file, str)


def test__state__is_training(state_instance: _State,
                             monkeypatch: pytest.MonkeyPatch) -> None:
    """ Test _is_training function for :class:`~lib.utils.sysinfo._State` """
    assert hasattr(state_instance, "_is_training")
    assert isinstance(state_instance._is_training, bool)
    assert not state_instance._is_training
    monkeypatch.setattr("sys.argv", ["faceswap.py", "train"])
    assert state_instance._is_training
    monkeypatch.setattr("sys.argv", ["faceswap.py", "extract"])
    assert not state_instance._is_training


def test__state__get_arg(state_instance: _State,
                         monkeypatch: pytest.MonkeyPatch) -> None:
    """ Test _get_arg function for :class:`~lib.utils.sysinfo._State` """
    assert hasattr(state_instance, "_get_arg")
    assert state_instance._get_arg("-t", "--test_arg") is None
    monkeypatch.setattr("sys.argv", ["test", "command", "-t", "test_option"])
    assert state_instance._get_arg("-t", "--test_arg") == "test_option"


def test__state__get_state_file(state_instance: _State,
                                mocker: pytest_mock.MockerFixture,
                                monkeypatch: pytest.MonkeyPatch) -> None:
    """ Test _get_state_file function for :class:`~lib.utils.sysinfo._State` """
    assert hasattr(state_instance, "_get_state_file")
    assert isinstance(state_instance._get_state_file(), str)

    mock_is_training = mocker.patch("lib.system.sysinfo._State._is_training")

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
