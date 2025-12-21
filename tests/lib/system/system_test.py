#!/usr/bin python3
""" Pytest unit tests for :mod:`lib.system.system` """

import ctypes
import locale
import os
import platform
import sys

import pytest
import pytest_mock

# pylint:disable=import-error
import lib.system.system as system_mod
from lib.system.system import _lines_from_command, VALID_PYTHON, Packages, System
# pylint:disable=protected-access


def test_valid_python() -> None:
    """ Confirm python version has a min and max and that it is Python 3 """
    assert len(VALID_PYTHON) == 2
    assert all(len(v) == 2 for v in VALID_PYTHON)
    assert all(isinstance(x, int) for v in VALID_PYTHON for x in v)
    assert all(v[0] == 3 for v in VALID_PYTHON)
    assert VALID_PYTHON[0] <= VALID_PYTHON[1]


def test_lines_from_command(mocker: pytest_mock.MockerFixture) -> None:
    """ Confirm lines from command executes as expected """
    input_ = ["test", "input"]
    subproc_out = "   this  \nis\n  test\noutput  \n"
    mock_run = mocker.patch("lib.system.system.run")
    mock_run.return_value.stdout = subproc_out
    result = _lines_from_command(input_)
    assert mock_run.called
    assert result == subproc_out.splitlines()


# System
@pytest.fixture(name="system_instance")
def system_fixture() -> System:
    """ Single :class:`lib.system.System` object for tests """
    return System()


def test_system_init(system_instance: System) -> None:
    """ Test :class:`lib.system.System` __init__ and attributes """
    assert isinstance(system_instance, System)

    attrs = ["platform", "system", "machine", "release", "processor", "cpu_count",
             "python_implementation", "python_version", "python_architecture", "encoding",
             "is_conda", "is_admin", "is_virtual_env"]
    assert all(a in system_instance.__dict__ for a in attrs)
    assert all(a in attrs for a in system_instance.__dict__)

    assert system_instance.platform == platform.platform()
    assert system_instance.system == platform.system().lower()
    assert system_instance.machine == platform.machine()
    assert system_instance.release == platform.release()
    assert system_instance.processor == platform.processor()
    assert system_instance.cpu_count == os.cpu_count()
    assert system_instance.python_implementation == platform.python_implementation()
    assert system_instance.python_version == platform.python_version()
    assert system_instance.python_architecture == platform.architecture()[0]
    assert system_instance.encoding == locale.getpreferredencoding()
    assert system_instance.is_conda == ("conda" in sys.version.lower() or
                                        os.path.exists(os.path.join(sys.prefix, "conda-meta")))
    assert isinstance(system_instance.is_admin, bool)
    assert isinstance(system_instance.is_virtual_env, bool)


def test_system_properties(system_instance: System) -> None:
    """ Test :class:`lib.system.System` properties """
    assert hasattr(system_instance, "is_linux")
    assert isinstance(system_instance.is_linux, bool)
    if platform.system().lower() == "linux":
        assert system_instance.is_linux
        assert not system_instance.is_macos
        assert not system_instance.is_windows

    assert hasattr(system_instance, "is_macos")
    assert isinstance(system_instance.is_macos, bool)
    if platform.system().lower() == "darwin":
        assert system_instance.is_macos
        assert not system_instance.is_linux
        assert not system_instance.is_windows

    assert hasattr(system_instance, "is_windows")
    assert isinstance(system_instance.is_windows, bool)
    if platform.system().lower() == "windows":
        assert system_instance.is_windows
        assert not system_instance.is_linux
        assert not system_instance.is_macos


def test_system_get_permissions(system_instance: System) -> None:
    """ Test :class:`lib.system.System` _get_permissions method """
    assert hasattr(system_instance, "_get_permissions")
    is_admin = system_instance._get_permissions()
    if platform.system() == "Windows":
        assert is_admin == (ctypes.windll.shell32.IsUserAnAdmin() != 0)  # type:ignore
    else:
        assert is_admin == (os.getuid() == 0)  # type:ignore  # pylint:disable=no-member


def test_system_check_virtual_env(system_instance: System,
                                  monkeypatch: pytest.MonkeyPatch) -> None:
    """ Test :class:`lib.system.System` _check_virtual_env method """
    system_instance.is_conda = True
    monkeypatch.setattr(system_mod.sys, "prefix", "/home/user/miniconda3/envs/testenv")
    assert system_instance._check_virtual_env()
    monkeypatch.setattr(system_mod.sys, "prefix", "/home/user/miniconda3/bin/")
    assert not system_instance._check_virtual_env()

    system_instance.is_conda = False
    monkeypatch.setattr(system_mod.sys, "base_prefix", "/home/user/venv/")
    monkeypatch.setattr(system_mod.sys, "prefix", "/usr/bin/")
    assert system_instance._check_virtual_env()
    monkeypatch.setattr(system_mod.sys, "base_prefix", "/usr/bin/")
    assert not system_instance._check_virtual_env()


def test_system_validate_python(system_instance: System,
                                monkeypatch: pytest.MonkeyPatch,
                                mocker: pytest_mock.MockerFixture) -> None:
    """ Test :class:`lib.system.System` _validate_python method """
    monkeypatch.setattr(system_mod, "VALID_PYTHON", (((3, 11), (3, 13))))
    monkeypatch.setattr(system_mod.sys, "version_info", (3, 12, 0))
    monkeypatch.setattr("builtins.input", lambda _: "")
    system_instance.python_architecture = "64bit"

    assert system_instance.validate_python()
    assert system_instance.validate_python(max_version=(3, 12))

    sys_exit = mocker.patch("lib.system.system.sys.exit")
    system_instance.python_architecture = "32bit"
    system_instance.validate_python()
    assert sys_exit.called
    system_instance.python_architecture = "64bit"

    system_instance.validate_python(max_version=(3, 11))
    assert sys_exit.called

    for vers in ((3, 10, 0), (3, 14, 0)):
        monkeypatch.setattr(system_mod.sys, "version_info", vers)
        system_instance.validate_python()
        assert sys_exit.called


@pytest.mark.parametrize("system_name, machine, is_conda, should_exit", [
    ("other", "x86_64", False, True),  # Unsupported OS
    ("darwin", "arm64", True, False),  # Apple Silicon inside conda
    ("darwin", "arm64", False, True),  # Apple Silicon outside conda
    ("linux", "x86_64", True, False),  # Supported
    ("windows", "x86_64", True, False),  # Supported
    ])
def test_system_validate(system_instance: System,
                         mocker: pytest_mock.MockerFixture,
                         system_name,
                         machine,
                         is_conda,
                         should_exit) -> None:
    """ Test :class:`lib.system.System` _validate method """
    validate_python = mocker.patch("lib.system.System.validate_python")
    system_instance.system = system_name
    system_instance.machine = machine
    system_instance.is_conda = is_conda
    sys_exit = mocker.patch("lib.system.system.sys.exit")
    system_instance.validate()
    if should_exit:
        assert sys_exit.called
    else:
        assert not sys_exit.called
        assert validate_python.called


# Packages
@pytest.fixture(name="packages_instance")
def packages_fixture() -> Packages:
    """ Single :class:`lib.system.Packages` object for tests """
    return Packages()


def test_packages_init(packages_instance: Packages, mocker: pytest_mock.MockerFixture) -> None:
    """ Test :class:`lib.system.Packages` __init__ and attributes """
    assert isinstance(packages_instance, Packages)

    attrs = ["_conda_exe", "_installed_python", "_installed_conda"]
    assert all(a in packages_instance.__dict__ for a in attrs)
    assert all(a in attrs for a in packages_instance.__dict__)

    assert isinstance(packages_instance._conda_exe,
                      str) or packages_instance._conda_exe is None
    assert isinstance(packages_instance._installed_python, dict)
    assert isinstance(packages_instance._installed_conda,
                      list) or packages_instance._installed_conda is None

    which = mocker.patch("lib.system.system.which")
    Packages()
    which.assert_called_once_with("conda")


def test_packages_properties(packages_instance: Packages) -> None:
    """ Test :class:`lib.system.Packages` properties """
    for prop in ("installed_python", "installed_conda"):
        assert hasattr(packages_instance, prop)
        assert isinstance(getattr(packages_instance, prop), dict)
        pretty = f"{prop}_pretty"
        assert hasattr(packages_instance, pretty)
        assert isinstance(getattr(packages_instance, pretty), str)


def test_packages_get_installed_python(packages_instance: Packages,
                                       mocker: pytest_mock.MockerFixture,
                                       monkeypatch: pytest.MonkeyPatch) -> None:
    """ Test :class:`lib.system.Packages` get_installed_python method """
    lines_from_command = mocker.patch("lib.system.system._lines_from_command")
    monkeypatch.setattr(system_mod.sys, "executable", "python")
    out = packages_instance._get_installed_python()
    lines_from_command.assert_called_once_with(["python", "-m", "pip", "freeze", "--local"])
    assert isinstance(out, dict)

    monkeypatch.setattr(system_mod, "_lines_from_command", lambda _: ["pacKage1==1.0.0",
                                                                      "PACKAGE2==1.1.0",
                                                                      "# Ignored",
                                                                      "malformed=1.2.3",
                                                                      "package3==0.2.1"])
    out = packages_instance._get_installed_python()
    assert out == {"package1": "1.0.0", "package2": "1.1.0", "package3": "0.2.1"}


def test_packages_get_installed_conda(packages_instance: Packages,
                                      mocker: pytest_mock.MockerFixture,
                                      monkeypatch: pytest.MonkeyPatch) -> None:
    """ Test :class:`lib.system.Packages` get_installed_conda method """
    packages_instance._conda_exe = None
    packages_instance._installed_conda = None
    packages_instance._get_installed_conda()
    assert packages_instance._installed_conda is None

    packages_instance._conda_exe = "conda"
    lines_from_command = mocker.patch("lib.system.system._lines_from_command")
    packages_instance._get_installed_conda()
    lines_from_command.assert_called_once_with(["conda", "list", "--show-channel-urls"])

    monkeypatch.setattr(system_mod, "_lines_from_command", lambda _: [])
    packages_instance._get_installed_conda()
    assert packages_instance._installed_conda == ["Could not get Conda package list"]

    _pkgs = [
        "package1            4.15.0           pypi_0              pypi",
        "pkg2                2025b            h78e105d_0          conda-forge",
        "Packag3             3.1.3            pypi_0              defaults"]
    monkeypatch.setattr(system_mod, "_lines_from_command", lambda _: _pkgs)
    packages_instance._get_installed_conda()
    assert packages_instance._installed_conda == _pkgs
