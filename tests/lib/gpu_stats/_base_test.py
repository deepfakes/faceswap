#!/usr/bin python3
""" Pytest unit tests for :mod:`lib.gpu_stats._base` """
import typing as T

from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest
import pytest_mock

# pylint:disable=protected-access
from lib.gpu_stats import _base
from lib.gpu_stats._base import BiggestGPUInfo, GPUInfo, _GPUStats, set_exclude_devices
from lib.utils import get_backend


def test_set_exclude_devices(monkeypatch: pytest.MonkeyPatch) -> None:
    """ Test that :func:`~lib.gpu_stats._base.set_exclude_devices` adds devices

    Parameters
    ----------
    monkeypatch: :class:`pytest.MonkeyPatch`
        Monkey patching _EXCLUDE_DEVICES
    """
    monkeypatch.setattr(_base, "_EXCLUDE_DEVICES", [])
    assert not _base._EXCLUDE_DEVICES
    set_exclude_devices([0, 1])
    assert _base._EXCLUDE_DEVICES == [0, 1]


@dataclass
class _DummyData:
    """ Dummy data for initializing and testing :class:`~lib.gpu_stats._base._GPUStats` """
    device_count = 2
    active_devices = [0, 1]
    handles = [0, 1]
    driver = "test_driver"
    device_names = ['test_device_0', 'test_device_1']
    vram = [1024, 2048]
    free_vram = [512, 1024]


@pytest.fixture(name="gpu_stats_instance")
def fixture__gpu_stats_instance(mocker: pytest_mock.MockerFixture) -> _GPUStats:
    """ Create a fixture of the :class:`~lib.gpu_stats._base._GPUStats` object

    Parameters
    ----------
    mocker: :class:`pytest_mock.MockerFixture`
        Mocker for dummying in function calls
    """
    mocker.patch.object(_GPUStats, '_initialize')
    mocker.patch.object(_GPUStats, '_shutdown')
    mocker.patch.object(_GPUStats, '_get_device_count', return_value=_DummyData.device_count)
    mocker.patch.object(_GPUStats, '_get_active_devices', return_value=_DummyData.active_devices)
    mocker.patch.object(_GPUStats, '_get_handles', return_value=_DummyData.handles)
    mocker.patch.object(_GPUStats, '_get_driver', return_value=_DummyData.driver)
    mocker.patch.object(_GPUStats, '_get_device_names', return_value=_DummyData.device_names)
    mocker.patch.object(_GPUStats, '_get_vram', return_value=_DummyData.vram)
    mocker.patch.object(_GPUStats, '_get_free_vram', return_value=_DummyData.free_vram)
    gpu_stats = _GPUStats()
    return gpu_stats


def test__gpu_stats_init_(gpu_stats_instance: _GPUStats) -> None:
    """ Test that the base :class:`~lib.gpu_stats._base._GPUStats` class initializes correctly

    Parameters
    ----------
    gpu_stats_instance: :class:`_GPUStats`
        Fixture instance of the _GPUStats base class
    """
    # Ensure that the object is initialized and shutdown correctly
    assert gpu_stats_instance._is_initialized is False
    assert T.cast(MagicMock, gpu_stats_instance._initialize).call_count == 1
    assert T.cast(MagicMock, gpu_stats_instance._shutdown).call_count == 1

    # Ensure that the object correctly gets and stores the device count, active devices,
    # handles, driver, device names, and VRAM information
    assert gpu_stats_instance.device_count == _DummyData.device_count
    assert gpu_stats_instance._active_devices == _DummyData.active_devices
    assert gpu_stats_instance._handles == _DummyData.handles
    assert gpu_stats_instance._driver == _DummyData.driver
    assert gpu_stats_instance._device_names == _DummyData.device_names
    assert gpu_stats_instance._vram == _DummyData.vram


def test__gpu_stats_properties(gpu_stats_instance: _GPUStats) -> None:
    """ Test that the :class:`~lib.gpu_stats._base._GPUStats` properties are set and formatted
    correctly.

    Parameters
    ----------
    gpu_stats_instance: :class:`_GPUStats`
        Fixture instance of the _GPUStats base class
    """
    assert gpu_stats_instance.cli_devices == ['0: test_device_0', '1: test_device_1']
    assert gpu_stats_instance.sys_info == GPUInfo(vram=_DummyData.vram,
                                                  vram_free=_DummyData.free_vram,
                                                  driver=_DummyData.driver,
                                                  devices=_DummyData.device_names,
                                                  devices_active=_DummyData.active_devices)


def test__gpu_stats_get_card_most_free(mocker: pytest_mock.MockerFixture,
                                       gpu_stats_instance: _GPUStats) -> None:
    """ Confirm that :func:`ib.gpu_stats._base._GPUStats.get_card_most_free` functions
    correctly

    Parameters
    ----------
    mocker: :class:`pytest_mock.MockerFixture`
        Mocker for dummying in function calls
    gpu_stats_instance: :class:`_GPUStats`
        Fixture instance of the _GPUStats base class
    """
    assert gpu_stats_instance.get_card_most_free() == BiggestGPUInfo(card_id=1,
                                                                     device='test_device_1',
                                                                     free=1024,
                                                                     total=2048)
    mocker.patch.object(_GPUStats, '_get_active_devices', return_value=[])
    gpu_stats = _GPUStats()
    assert gpu_stats.get_card_most_free() == BiggestGPUInfo(card_id=-1,
                                                            device='No GPU devices found',
                                                            free=2048,
                                                            total=2048)


def test__gpu_stats_exclude_all_devices(gpu_stats_instance: _GPUStats) -> None:
    """ Ensure that the object correctly returns whether all devices are excluded

    Parameters
    ----------
    gpu_stats_instance: :class:`_GPUStats`
        Fixture instance of the _GPUStats base class
    """
    assert gpu_stats_instance.exclude_all_devices is False
    set_exclude_devices([0, 1])
    assert gpu_stats_instance.exclude_all_devices is True


def test__gpu_stats_no_active_devices(
        caplog: pytest.LogCaptureFixture,
        gpu_stats_instance: _GPUStats,  # pylint:disable=unused-argument
        mocker: pytest_mock.MockerFixture) -> None:
    """ Ensure that no active GPUs raises a warning when not in CPU mode

    Parameters
    ----------
    caplog: :class:`pytest.LogCaptureFixture`
        Pytest's log capturing fixture
    gpu_stats_instance: :class:`_GPUStats`
        Fixture instance of the _GPUStats base class
    mocker: :class:`pytest_mock.MockerFixture`
        Mocker for dummying in function calls
    """
    if get_backend() == "cpu":
        return
    caplog.set_level("WARNING")
    mocker.patch.object(_GPUStats, '_get_active_devices', return_value=[])
    _GPUStats()
    assert "No GPU detected" in caplog.messages
